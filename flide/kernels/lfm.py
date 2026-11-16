"""Fused pieces of LFM3's layers (backbones/lfm3/fused.py): the gated short convolution in one pass, and the sigmoid
gate on attention's output in place. Both sum in float32 and round once."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _short_conv_kernel(bcx, w, prev, y, T, C, s_bcx, s_prev, s_y, K: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < C
    row = bcx + t * s_bcx
    acc = (tl.load(row + offs, mask=m, other=0.0).to(tl.float32)
           * tl.load(row + 2 * C + offs, mask=m, other=0.0).to(tl.float32)
           * tl.load(w + offs * K + (K - 1), mask=m, other=0.0).to(tl.float32))
    for j in tl.static_range(1, K):
        p = tl.load(prev + t * s_prev + (j - 1))
        ok = m & (p < T)
        pr = bcx + p * s_bcx
        bx = (tl.load(pr + offs, mask=ok, other=0.0).to(tl.float32)
              * tl.load(pr + 2 * C + offs, mask=ok, other=0.0).to(tl.float32))
        acc += bx * tl.load(w + offs * K + (K - 1 - j), mask=m, other=0.0).to(tl.float32)
    gate = tl.load(row + C + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(y + t * s_y + offs, (gate * acc).to(y.dtype.element_ty), mask=m)


def short_conv(bcx: torch.Tensor, weight: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
    """LFM's gated short convolution: bcx [T, 3C] is in_proj's output (B, C, x), weight [C, K] (the last tap reads the
    token itself), prev [T, K - 1] each token's earlier stream rows (T where there is none). Returns
    C * conv(B * x) [T, C] in bcx's dtype, the products and sums in float32."""
    T, C3 = bcx.shape
    C, K = weight.shape
    if C3 != 3 * C or bcx.stride(1) != 1:
        raise ValueError(f"short_conv: bcx {tuple(bcx.shape)} with last-dim stride {bcx.stride(1)} for {C} channels")
    w = weight.contiguous()
    y = torch.empty(T, C, device=bcx.device, dtype=bcx.dtype)
    if T:
        BLOCK = 1024
        _short_conv_kernel[(T, triton.cdiv(C, BLOCK))](bcx, w, prev, y, T, C, bcx.stride(0), prev.stride(0), y.stride(0),
                                                       K=K, BLOCK=BLOCK, num_warps=4)
    return y


@triton.jit
def _gate_kernel(o, g, s_ot, s_oh, s_gt, H, D, BLOCK_D: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    m = offs < D
    z = tl.load(g + t * s_gt + h).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-z))
    ptr = o + t * s_ot + h * s_oh + offs
    x = tl.load(ptr, mask=m, other=0.0).to(tl.float32)
    tl.store(ptr, (x * s).to(o.dtype.element_ty), mask=m)


def sigmoid_gate_(o: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """o [T, H, D] *= sigmoid(g [T, H]) per head, in place (float32 math, one rounding); o's last dim contiguous."""
    T, H, D = o.shape
    if o.stride(2) != 1 or g.stride(1) != 1:
        raise ValueError("sigmoid_gate_: o and g need a contiguous last dimension")
    if T:
        _gate_kernel[(T, H)](o, g, o.stride(0), o.stride(1), g.stride(0), H, D,
                             BLOCK_D=triton.next_power_of_2(D), num_warps=1)
    return o

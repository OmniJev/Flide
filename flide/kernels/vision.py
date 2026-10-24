"""Fused steps of the Qwen3.5 vision tower: LayerNorm with the residual add before it, and the rotary embedding of
the queries and keys, each one kernel (transformers runs them as a dozen elementwise kernels per block, which at
batch one is most of the tower's time). Arithmetic in float32 like transformers', rounded once on output; the
residual sum is rounded to the stream's dtype before it is normalized, as transformers rounds it.

Also the image processor's resize of uint8 images, on the GPU and bit for bit (`resize_u8`).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import triton
import triton.language as tl


def _p2(n: int) -> int:
    return 1 << (n - 1).bit_length()


@triton.jit
def _ln_kernel(X, D, W, B, H_OUT, Y, N, sx, sd, sh, sy, eps, ADD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    x = tl.load(X + row * sx + cols, mask=m, other=0.0)
    if ADD:
        x = (x.to(tl.float32) + tl.load(D + row * sd + cols, mask=m, other=0.0).to(tl.float32)).to(x.dtype)
        tl.store(H_OUT + row * sh + cols, x, mask=m)
    xf = x.to(tl.float32)
    mean = tl.sum(xf, 0) / N
    c = tl.where(m, xf - mean, 0.0)
    var = tl.sum(c * c, 0) / N
    y = c * tl.math.rsqrt(var + eps) * tl.load(W + cols, mask=m, other=0.0).to(tl.float32) \
        + tl.load(B + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row * sy + cols, y.to(Y.dtype.element_ty), mask=m)


def layer_norm(x, w, b, eps: float, residual=None):
    """LayerNorm(x [+ residual]) with weight and bias. With a residual, returns (x + residual, normalized)."""
    T, N = x.shape
    y = torch.empty_like(x)
    h = torch.empty_like(x) if residual is not None else x
    if T:
        d = residual if residual is not None else x
        _ln_kernel[(T,)](x, d, w, b, h, y, N, x.stride(0), d.stride(0), h.stride(0), y.stride(0), eps,
                         ADD=residual is not None, BLOCK=_p2(N), num_warps=4 if N <= 2048 else 8)
    return (h, y) if residual is not None else y


@triton.jit
def _rope_kernel(QKV, COS, SIN, Q, K, n, H, s_t, s_w, s_h, sc_t, so_t, so_h, HALF: tl.constexpr, HP: tl.constexpr,
                 BT: tl.constexpr):
    """A block of BT tokens, every head of the queries and keys (the token's cos / sin read once)."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    i = tl.arange(0, HP)
    mi = (t < n)[:, None] & (i < HALF)[None, :]
    tt = t.to(tl.int64)[:, None]
    ii = i[None, :]
    c1 = tl.load(COS + tt * sc_t + ii, mask=mi).to(tl.float32)
    c2 = tl.load(COS + tt * sc_t + HALF + ii, mask=mi).to(tl.float32)
    s1 = tl.load(SIN + tt * sc_t + ii, mask=mi).to(tl.float32)
    s2 = tl.load(SIN + tt * sc_t + HALF + ii, mask=mi).to(tl.float32)
    for which in tl.static_range(2):
        for h in range(H):
            base = QKV + tt * s_t + which * s_w + h * s_h
            x1 = tl.load(base + ii, mask=mi).to(tl.float32)
            x2 = tl.load(base + HALF + ii, mask=mi).to(tl.float32)
            a = x1 * c1
            b = -x2 * s1
            y1 = a + b
            a = x2 * c2
            b = x1 * s2
            y2 = a + b
            if which == 0:
                out = Q + tt * so_t + h * so_h
            else:
                out = K + tt * so_t + h * so_h
            tl.store(out + ii, y1.to(Q.dtype.element_ty), mask=mi)
            tl.store(out + HALF + ii, y2.to(Q.dtype.element_ty), mask=mi)


def rope(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """qkv [n, 3, H, D] (the qkv projection viewed per head), cos / sin [n, D] -> rotated q, k [n, H, D] contiguous:
    x * cos + rotate_half(x) * sin in float32, as transformers' apply_rotary_pos_emb_vision computes it."""
    n, _, H, D = qkv.shape
    q = torch.empty(n, H, D, device=qkv.device, dtype=qkv.dtype)
    k = torch.empty_like(q)
    if n:
        BT = 16
        _rope_kernel[(triton.cdiv(n, BT),)](qkv, cos, sin, q, k, n, H, qkv.stride(0), qkv.stride(1), qkv.stride(2),
                                            cos.stride(0), q.stride(0), q.stride(1), HALF=D // 2, HP=_p2(D // 2), BT=BT,
                                            num_warps=4, enable_fp_fusion=False)
    return q, k


@triton.jit
def _resample_kernel(SRC, DST, START, WT, total, n_out, B, s_a, s_j, K, prec, half, BLOCK: tl.constexpr):
    """One pass of the separable resize over a [A, n_in, B] view into [A, n_out, B]: output (a, i, b) is the int16
    weights of output i times the inputs (a, start_i + k, b), summed in int32 from half a unit, shifted right by the
    weights' precision and clamped to 0..255."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    b = offs % B
    i = (offs // B) % n_out
    a = offs // (B * n_out)
    start = tl.load(START + i, mask=m, other=0)
    acc = tl.zeros([BLOCK], tl.int32) + half
    for k in range(K):
        w = tl.load(WT + i * K + k, mask=m, other=0)
        x = tl.load(SRC + a * s_a + (start + k) * s_j + b, mask=m & (w != 0), other=0).to(tl.int32)
        acc += w * x
    out = tl.minimum(tl.maximum(acc >> prec, 0), 255)
    tl.store(DST + offs, out.to(tl.uint8), mask=m)


_WEIGHTS: dict = {}


def _resize_weights(n_in: int, n_out: int, device):
    """Per output pixel the first input pixel and the int16 weights (as int32 [n_out, K]), and their precision in bits:
    torch's antialiased bicubic weights for uint8 images (Keys' cubic, a = -0.5, widened by the scale when shrinking,
    normalized to sum 1 in float64, then scaled by the largest power of two that keeps them in int16 and rounded half
    away from zero), computed in the same order of float64 operations."""
    key = (n_in, n_out, str(device))
    hit = _WEIGHTS.get(key)
    if hit is not None:
        return hit
    scale = n_in / n_out
    support = 2.0 * scale if scale >= 1.0 else 2.0
    inv = 1.0 / scale if scale >= 1.0 else 1.0
    K = int(math.ceil(support)) * 2 + 1
    center = scale * (np.arange(n_out) + 0.5)
    lo = np.maximum(np.trunc(center - support + 0.5), 0).astype(np.int64)
    size = np.minimum(np.trunc(center + support + 0.5), n_in).astype(np.int64) - lo
    j = np.arange(K)[None]
    x = np.abs((j + lo[:, None] - center[:, None] + 0.5) * inv)
    a = -0.5
    w = np.where(x < 1.0, ((a + 2.0) * x - (a + 3.0)) * x * x + 1,
                 np.where(x < 2.0, (((x - 5) * x + 8) * x - 4) * a, 0.0))
    w = np.where(j < size[:, None], w, 0.0)
    total = np.zeros(n_out)
    for k in range(K):
        total += w[:, k]
    w = np.where(total[:, None] != 0, w / np.where(total == 0, 1.0, total)[:, None], w)
    wmax = float(w.max())
    prec = 0
    while prec < 22 and int(0.5 + wmax * (1 << (prec + 1))) < (1 << 15):
        prec += 1
    v = w * (1 << prec)
    wi = np.where(v < 0, np.trunc(-0.5 + v), np.trunc(0.5 + v)).astype(np.int32)
    hit = _WEIGHTS[key] = (torch.from_numpy(lo.astype(np.int32)).to(device), torch.from_numpy(wi).to(device), prec)
    return hit


def resize_u8(img: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """[C, h, w] uint8 on the GPU -> [C, height, width] uint8: the antialiased bicubic resize torch runs for uint8
    images on the CPU (torchvision's resize of uint8 tensors, what the Qwen image processors call), bit for bit. Its
    integer arithmetic: a horizontal pass into uint8, then a vertical one, each skipped when that side keeps its size."""
    C, h, w = img.shape
    x = img.contiguous()
    BLOCK = 1024
    if width != w:
        start, wt, prec = _resize_weights(w, width, x.device)
        out = torch.empty(C, h, width, dtype=torch.uint8, device=x.device)
        total = out.numel()
        _resample_kernel[(triton.cdiv(total, BLOCK),)](x, out, start, wt, total, width, 1, w, 1, wt.shape[1], prec,
                                                       1 << (prec - 1), BLOCK=BLOCK)
        x = out
    if height != h:
        start, wt, prec = _resize_weights(h, height, x.device)
        out = torch.empty(C, height, width, dtype=torch.uint8, device=x.device)
        total = out.numel()
        _resample_kernel[(triton.cdiv(total, BLOCK),)](x, out, start, wt, total, height, width, h * width, width,
                                                       wt.shape[1], prec, 1 << (prec - 1), BLOCK=BLOCK)
        x = out
    return x

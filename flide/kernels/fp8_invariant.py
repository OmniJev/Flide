"""The float8 product of kernels/fp8.py for the batch-invariant mode: the same operands (float8 weights with one scale
per output row, inputs scaled per token on the fly) and the same result formula (the float8 products summed in float32,
times the token's scale and the row's), with every output element summed in one order whatever the number of rows.
One k step (BK) for every tile shape, the whole reduction in one program, no split along it; each step's float8 dot
starts from zero and is added to the float32 sum, as the float8 experts of kernels/moe.py do. The scaled matmul of
cuBLAS picks its split and kernel by the shape, so its bits move with the batch; this one does not.

On compute capability 9 and later the tiles stream through the Hopper tensor-memory accelerator; elsewhere (Ada) a
portable kernel with the same k step.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .invariant import _alloc, _local, budget

BK = 128   # float8 products: k steps of 32 per instruction


@triton.jit
def _mm_kernel(X, W, SX, SW, Y, M, N, K, sxm, swn, sym, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               GROUP: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    width = GROUP * num_n
    group = pid // width
    first = group * GROUP
    size = min(num_m - first, GROUP)
    pm = first + (pid % width) % size
    pn = (pid % width) // size
    rm = pm.to(tl.int64) * BM + tl.arange(0, BM)
    rn = pn.to(tl.int64) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        a = tl.load(X + rm[:, None] * sxm + kk[None, :], mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(W + rn[None, :] * swn + kk[:, None], mask=(rn[None, :] < N) & (kk[:, None] < K), other=0.0)
        acc += tl.dot(a, b)
    sx = tl.load(SX + rm, mask=rm < M, other=0.0)
    sw = tl.load(SW + rn, mask=rn < N, other=0.0)
    y = acc * sx[:, None] * sw[None, :]
    tl.store(Y + rm[:, None] * sym + rn[None, :], y.to(Y.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _mm_tma_kernel(X, W, SX, SW, Y, M, N, K, sxm, swn, sym, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   GROUP: tl.constexpr, NUM_SMS: tl.constexpr):
    x_desc = tl.make_tensor_descriptor(X, shape=[M, K], strides=[sxm, 1], block_shape=[BM, BK])
    w_desc = tl.make_tensor_descriptor(W, shape=[N, K], strides=[swn, 1], block_shape=[BN, BK])
    y_desc = tl.make_tensor_descriptor(Y, shape=[M, N], strides=[sym, 1], block_shape=[BM, BN])
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    k_tiles = tl.cdiv(K, BK)
    num_tiles = num_m * num_n
    width = GROUP * num_n
    for tile in tl.range(tl.program_id(0), num_tiles, NUM_SMS, flatten=True):
        group = tile // width
        first = group * GROUP
        size = min(num_m - first, GROUP)
        pm = first + (tile % width) % size
        pn = (tile % width) // size
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for ki in range(k_tiles):
            a = x_desc.load([pm * BM, ki * BK])
            b = w_desc.load([pn * BN, ki * BK])
            acc += tl.dot(a, b.T)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        sx = tl.load(SX + rm, mask=rm < M, other=0.0)
        sw = tl.load(SW + rn, mask=rn < N, other=0.0)
        y_desc.store([pm * BM, pn * BN], (acc * sx[:, None] * sw[None, :]).to(Y.dtype.element_ty))


from . import gpu   # below the kernels: Triton keys a compiled kernel by the line it starts on


# TMA tile shapes (BM, BN, BK, warps, stages), widest first; any of them sums an element in the same order
WIDE, NARROW, SMALL = (128, 128, BK, 8, 4), (64, 128, BK, 4, 4), (64, 64, BK, 4, 3)
SMALL_M = 512   # rows at or below: the narrow shapes


def _needs(cfg: tuple, out_bytes: int) -> int:
    """Shared memory a TMA shape asks for: the float8 stages of both operands, the output tile and a margin."""
    BM, BN, bk, _, stages = cfg
    return stages * (BM + BN) * bk + BM * BN * out_bytes + 2048


def pick(M: int, budget_bytes: int, out_bytes: int = 2) -> tuple | None:
    order = (NARROW, SMALL) if M <= SMALL_M else (WIDE, NARROW, SMALL)
    return next((c for c in order if _needs(c, out_bytes) <= budget_bytes), None)


_sms = None


def mm_scaled(xq: torch.Tensor, sx: torch.Tensor, q: torch.Tensor, sw: torch.Tensor, out_dtype: torch.dtype,
              cfg: tuple | None = None) -> torch.Tensor:
    """(xq [M, K] * sx[M]) @ (q [N, K] * sw[N]).T in `out_dtype`, float8 operands with unit stride along K, summed in
    float32 in one fixed order; `cfg` forces a tile shape (tests)."""
    global _sms
    M, K = xq.shape
    N = q.shape[0]
    sx, sw = sx.reshape(-1), sw.reshape(-1)
    y = torch.empty((M, N), dtype=out_dtype, device=xq.device)
    if M == 0:
        return y
    tma = (gpu.tma(xq.device) and K % 16 == 0 and xq.stride(0) % 16 == 0
           and q.stride(0) % 16 == 0 and (y.stride(0) * y.element_size()) % 16 == 0)
    shape = cfg or (pick(M, budget(xq.device), y.element_size()) if tma else None)
    if tma and shape is not None:
        BM, BN, bk, warps, stages = shape
        if _sms is None:
            _sms = gpu.sms(xq.device)
        if not getattr(_local, "allocator", False):
            triton.set_allocator(_alloc)
            _local.allocator = True
        tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
        _mm_tma_kernel[(min(_sms, tiles),)](xq, q, sx, sw, y, M, N, K, xq.stride(0), q.stride(0), y.stride(0), BM=BM,
                                           BN=BN, BK=bk, GROUP=8, NUM_SMS=_sms, num_warps=warps, num_stages=stages)
        return y
    BM, BN, bk, warps, stages = cfg or (64, 128, BK, 4, 3)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _mm_kernel[grid](xq, q, sx, sw, y, M, N, K, xq.stride(0), q.stride(0), y.stride(0), BM=BM, BN=BN, BK=bk, GROUP=8,
                     num_warps=warps, num_stages=stages)
    return y

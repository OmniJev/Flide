"""Matrix products whose every output element is summed in the same order whatever the number of rows and wherever a
row sits: one fixed tile shape, the whole reduction in one program, no split along it. With them a request's numbers
do not depend on the other requests of its round (`settings.batch_invariant`).

On compute capability 9 and later bf16 products run a persistent kernel that streams its tiles through the Hopper
tensor-memory accelerator; elsewhere, and for the float32 token head, a portable kernel."""
from __future__ import annotations

import threading

import torch
import triton
import triton.language as tl

CFG = (128, 256, 64, 8, 3)
SMALL_M, SMALL_CFG = 256, (64, 128, 64, 4, 4)
_sms = None
_local = threading.local()


@triton.jit
def _mm_kernel(X, W, Y, M, N, K, sxm, swn, sym, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               GROUP: tl.constexpr, IEEE: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    width = GROUP * num_n
    group = pid // width
    first = group * GROUP
    size = min(num_m - first, GROUP)
    pm = first + (pid % width) % size
    pn = (pid % width) // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        a = tl.load(X + rm[:, None] * sxm + kk[None, :], mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(W + rn[None, :] * swn + kk[:, None], mask=(rn[None, :] < N) & (kk[:, None] < K), other=0.0)
        if IEEE:
            acc += tl.dot(a, b, input_precision="ieee")
        else:
            acc += tl.dot(a, b)
    tl.store(Y + rm[:, None] * sym + rn[None, :], acc.to(Y.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _mm_tma_kernel(X, W, Y, M, N, K, sxm, swn, sym, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
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
            acc = tl.dot(a, b.T, acc)
        y_desc.store([pm * BM, pn * BN], acc.to(Y.dtype.element_ty))


def _alloc(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def mm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None = None, cfg: tuple | None = None) -> torch.Tensor:
    """x [M, K] @ w[N, K].T, rows of x and w contiguous in K; `cfg` forces a tile shape (tests)."""
    global _sms
    if x.stride(-1) != 1:
        x = x.contiguous()
    if w.stride(-1) != 1:
        w = w.contiguous()
    M, K = x.shape
    N = w.shape[0]
    y = out if out is not None else torch.empty((M, N), dtype=x.dtype, device=x.device)
    if M == 0:
        return y
    ieee = x.dtype == torch.float32
    tma = (not ieee and torch.cuda.get_device_capability(x.device)[0] >= 9 and K % 8 == 0
           and x.stride(0) % 8 == 0 and w.stride(0) % 8 == 0 and y.stride(0) % 8 == 0 and y.stride(-1) == 1)
    if tma:
        BM, BN, BK, warps, stages = cfg or (SMALL_CFG if SMALL_M and M <= SMALL_M else CFG)
        if _sms is None:
            _sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        if not getattr(_local, "allocator", False):
            triton.set_allocator(_alloc)
            _local.allocator = True
        tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
        _mm_tma_kernel[(min(_sms, tiles),)](x, w, y, M, N, K, x.stride(0), w.stride(0), y.stride(0), BM=BM, BN=BN,
                                           BK=BK, GROUP=8, NUM_SMS=_sms, num_warps=warps, num_stages=stages)
        return y
    BM, BN, BK, warps, stages = (64, 64, 32, 4, 2) if ieee else (128, 128, 64, 8, 4)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _mm_kernel[grid](x, w, y, M, N, K, x.stride(0), w.stride(0), y.stride(0), BM=BM, BN=BN, BK=BK, GROUP=8,
                     IEEE=ieee, num_warps=warps, num_stages=stages)
    return y

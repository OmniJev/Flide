"""8-bit integer weights for the decoder's matrix products: the weight in int8 with one scale per output row
(amax / 127), the input scaled per token on the fly (one Triton kernel), the product summed in int32 and scaled back
to the model's dtype in the same kernel. An int32 sum is exact, so it does not depend on the order of its terms: every
output element has the same bits whatever the tile shape, the split of the reduction or the rows beside it, and the
products are batch invariant as they are. An all-zero token (padding) keeps a tiny scale and stays zero. Needs compute
capability 8.0 or newer (Ampere, Ada, Hopper, Blackwell).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .fp8 import mm as mm16
from .invariant import _alloc, _local, budget

INT8_MAX = 127.0
_sms = None


def supported(device) -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(device) >= (8, 0)


def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, K) weight -> (N, K) int8 and (N, 1) float32 scale with w ~ q * scale."""
    scale = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) * (1.0 / INT8_MAX)
    return (w.float() / scale).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8), scale


@triton.jit
def _quantize_tokens(X, INV, Q, S, K, sx, sq, BLOCK: tl.constexpr, SMOOTH: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    amax = tl.zeros([BLOCK], tl.float32)
    for k0 in range(0, K, BLOCK):
        cols = k0 + tl.arange(0, BLOCK)
        x = tl.load(X + row * sx + cols, mask=cols < K, other=0.0).to(tl.float32)
        if SMOOTH:
            x = x * tl.load(INV + cols, mask=cols < K, other=0.0)
        amax = tl.maximum(amax, tl.abs(x))
    s = tl.maximum(tl.max(amax, 0), 1e-12) * (1.0 / 127.0)
    tl.store(S + row, s)
    for k0 in range(0, K, BLOCK):
        cols = k0 + tl.arange(0, BLOCK)
        x = tl.load(X + row * sx + cols, mask=cols < K, other=0.0).to(tl.float32)
        if SMOOTH:
            x = x * tl.load(INV + cols, mask=cols < K, other=0.0)
        q = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(x, s)), -127.0), 127.0)
        tl.store(Q + row * sq + cols, q.to(tl.int8), mask=cols < K)


def quantize_tokens(x: torch.Tensor, inv: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(M, K) activations (unit stride along K), each channel times `inv` [K] when given -> int8 [M, K] and a [M]
    float32 scale per token."""
    M, K = x.shape
    q = torch.empty(M, K, device=x.device, dtype=torch.int8)
    s = torch.empty(M, device=x.device, dtype=torch.float32)
    if M:
        _quantize_tokens[(M,)](x, x if inv is None else inv, q, s, K, x.stride(0), q.stride(0), BLOCK=1024,
                               SMOOTH=inv is not None, num_warps=4)
    return q, s


@triton.jit
def _mm_kernel(X, W, SX, SW, Y, WS, M, N, K, sxm, swn, sym, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               GROUP: tl.constexpr, SPLIT: tl.constexpr):
    pid = tl.program_id(0) // SPLIT
    part = tl.program_id(0) % SPLIT
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
    per = tl.cdiv(tl.cdiv(K, BK), SPLIT) * BK
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k0 in range(part * per, min(K, part * per + per), BK):
        kk = k0 + rk
        a = tl.load(X + rm[:, None].to(tl.int64) * sxm + kk[None, :], mask=(rm[:, None] < M) & (kk[None, :] < K),
                    other=0)
        # read row by row and transposed: Triton 3.8 converts a block read column by column through shared memory, 1.4x slower
        b = tl.load(W + rn[:, None].to(tl.int64) * swn + kk[None, :], mask=(rn[:, None] < N) & (kk[None, :] < K),
                    other=0).T
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
    inside = (rm[:, None] < M) & (rn[None, :] < N)
    if SPLIT == 1:
        sx = tl.load(SX + rm, mask=rm < M, other=0.0)
        sw = tl.load(SW + rn, mask=rn < N, other=0.0)
        y = acc.to(tl.float32) * sx[:, None] * sw[None, :]
        tl.store(Y + rm[:, None].to(tl.int64) * sym + rn[None, :], y.to(Y.dtype.element_ty), mask=inside)
    else:   # int32 sums are exact, so the parts add up to the same bits in any order
        tl.atomic_add(WS + rm[:, None].to(tl.int64) * N + rn[None, :], acc, mask=inside, sem="relaxed")


@triton.jit
def _mm_tma_kernel(X, W, SX, SW, Y, WS, M, N, K, sxm, swn, sym, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr, GROUP: tl.constexpr, NUM_SMS: tl.constexpr, SPLIT: tl.constexpr):
    x_desc = tl.make_tensor_descriptor(X, shape=[M, K], strides=[sxm, 1], block_shape=[BM, BK])
    w_desc = tl.make_tensor_descriptor(W, shape=[N, K], strides=[swn, 1], block_shape=[BN, BK])
    y_desc = tl.make_tensor_descriptor(Y, shape=[M, N], strides=[sym, 1], block_shape=[BM, BN])
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    k_tiles = tl.cdiv(K, BK)
    per = tl.cdiv(k_tiles, SPLIT)
    num_tiles = num_m * num_n
    width = GROUP * num_n
    for work in tl.range(tl.program_id(0), num_tiles * SPLIT, NUM_SMS, flatten=True):
        tile = work // SPLIT
        part = work % SPLIT
        group = tile // width
        first = group * GROUP
        size = min(num_m - first, GROUP)
        pm = first + (tile % width) % size
        pn = (tile % width) // size
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        for ki in range(part * per, min(k_tiles, part * per + per)):
            a = x_desc.load([pm * BM, ki * BK])
            b = w_desc.load([pn * BN, ki * BK])
            acc = tl.dot(a, b.T, acc, out_dtype=tl.int32)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        if SPLIT == 1:
            sx = tl.load(SX + rm, mask=rm < M, other=0.0)
            sw = tl.load(SW + rn, mask=rn < N, other=0.0)
            y = acc.to(tl.float32) * sx[:, None] * sw[None, :]
            y_desc.store([pm * BM, pn * BN], y.to(Y.dtype.element_ty))
        else:
            tl.atomic_add(WS + rm[:, None].to(tl.int64) * N + rn[None, :], acc,
                          mask=(rm[:, None] < M) & (rn[None, :] < N), sem="relaxed")


@triton.jit
def _scale_kernel(WS, SX, SW, Y, M, N, sym, BLOCK: tl.constexpr):
    """The split products' int32 sums, scaled as the kernels above scale theirs."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.load(WS + row * N + cols, mask=cols < N, other=0)
    sx = tl.load(SX + row)
    sw = tl.load(SW + cols, mask=cols < N, other=0.0)
    y = acc.to(tl.float32) * sx * sw
    tl.store(Y + row * sym + cols, y.to(Y.dtype.element_ty), mask=cols < N)


def _cfg(M: int, N: int, sms: int) -> tuple:
    """Tile and launch shape (BM, BN, BK, warps, stages) of the portable kernel, from bench/int8_tiles.py on an A40 (the
    products of Qwen3.5 0.8B to 27B, 1 to 8192 rows: 3% from the best on average); any choice gives the same bits. Large
    tiles only where they fill half of the GPU: splitting a reduction to fill it is slower than smaller tiles."""
    if M > 128:
        if 2 * triton.cdiv(M, 128) * triton.cdiv(N, 128) >= sms:
            return (128, 128, 64, 8, 4) if M >= 512 else (128, 128, 128, 8, 3)
        if 2 * triton.cdiv(M, 64) * triton.cdiv(N, 128) >= sms:
            return 64, 128, 128, 4, 4
    return 64, 64, 128, 4, 4


def _tma_cfg(M: int, N: int) -> tuple:
    if M <= 256 and N <= 4096:
        return 64, 64, 256, 4, 4
    if M <= 64:
        return 64, 128, 128, 4, 4
    return 128, 128, 256, 8, 3


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0 and t.stride(-1) == 1 and (t.stride(0) * t.element_size()) % 16 == 0


def _split(tiles: int, k_tiles: int, sms: int) -> int:
    """Parts of the reduction for a product with too few tiles to fill the GPU, each at least 8 steps long."""
    split = 1
    while split < 16 and tiles * split * 2 <= sms and k_tiles >= split * 2 * 8:
        split *= 2
    return split


def mm_scaled(xq: torch.Tensor, sx: torch.Tensor, q: torch.Tensor, sw: torch.Tensor, out_dtype: torch.dtype,
              out: torch.Tensor | None = None, cfg: tuple | None = None, split: int | None = None) -> torch.Tensor:
    """(xq [M, K] * sx[M]) @ (q [N, K] * sw[N]).T in `out_dtype`, int8 operands summed in int32; `out` may be a column
    slice of a wider buffer; `cfg` and `split` force a tile shape and a split of the reduction (tests)."""
    global _sms
    M, K = xq.shape
    N = q.shape[0]
    y = out if out is not None else torch.empty((M, N), dtype=out_dtype, device=xq.device)
    if M == 0:
        return y
    if _sms is None:
        _sms = torch.cuda.get_device_properties(xq.device).multi_processor_count
    # the TMA tiles need the shared memory of H100 and H200 (an sm_120 or sm_121 card holds 99 KB: the portable kernel)
    tma = (torch.cuda.get_device_capability(xq.device)[0] >= 9 and budget(xq.device) >= 163856
           and _aligned(xq) and _aligned(q) and _aligned(y))
    BM, BN, BK, warps, stages = cfg or (_tma_cfg(M, N) if tma else _cfg(M, N, _sms))
    tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    if split is None:
        split = _split(tiles, triton.cdiv(K, BK), _sms) if M <= 256 else 1
    ws = torch.zeros((M, N), dtype=torch.int32, device=xq.device) if split > 1 else y
    if tma:
        if not getattr(_local, "allocator", False):
            triton.set_allocator(_alloc)
            _local.allocator = True
        _mm_tma_kernel[(min(_sms, tiles * split),)](xq, q, sx, sw, y, ws, M, N, K, xq.stride(0), q.stride(0),
                                                    y.stride(0), BM=BM, BN=BN, BK=BK, GROUP=8, NUM_SMS=_sms,
                                                    SPLIT=split, num_warps=warps, num_stages=stages)
    else:
        _mm_kernel[(tiles * split,)](xq, q, sx, sw, y, ws, M, N, K, xq.stride(0), q.stride(0), y.stride(0), BM=BM,
                                     BN=BN, BK=BK, GROUP=8, SPLIT=split, num_warps=warps, num_stages=stages)
    if split > 1:
        _scale_kernel[(M, triton.cdiv(N, 1024))](ws, sx, sw, y, M, N, y.stride(0), BLOCK=1024, num_warps=4)
    return y


class Int8Weight:
    """A weight in int8 with per-row scales, called as the matrix product `x @ w.T`. `tail` rows (the small gate
    projections of a merged Gated DeltaNet input projection) stay in the model's dtype and come last. With `inv` (one
    factor per input channel, SmoothQuant), the input is multiplied by it before quantization and the weight was
    divided by it, which leaves the product unchanged and moves an input channel's outliers into the weight."""

    def __init__(self, q: torch.Tensor, scale: torch.Tensor, tail: torch.Tensor | None, out_dtype: torch.dtype,
                 inv: torch.Tensor | None = None):
        self.q = q
        self.scale = scale.reshape(-1).float().contiguous()
        self.tail = tail
        self.out_dtype = out_dtype
        self.inv = inv
        self.shape = (q.shape[0] + (0 if tail is None else tail.shape[0]), q.shape[1])

    @classmethod
    def from_weight(cls, w: torch.Tensor, tail_rows: int = 0, smooth: torch.Tensor | None = None) -> "Int8Weight":
        """`smooth` [K]: the weight's columns are multiplied by it and the input divided by it."""
        n = w.shape[0] - tail_rows
        if smooth is None:
            q, s = quantize_rows(w[:n])
            return cls(q, s, w[n:].clone() if tail_rows else None, w.dtype)
        smooth = smooth.float()
        q, s = quantize_rows(w[:n].float() * smooth)
        return cls(q, s, w[n:].clone() if tail_rows else None, w.dtype, (1.0 / smooth).contiguous())

    def split(self, n: int) -> tuple["Int8Weight", "Int8Weight"]:
        """The first n output rows and the rest, sharing this weight's storage (no tail rows)."""
        if self.tail is not None:
            raise ValueError("a weight with tail rows does not split")
        return (Int8Weight(self.q[:n], self.scale[:n], None, self.out_dtype, self.inv),
                Int8Weight(self.q[n:], self.scale[n:], None, self.out_dtype, self.inv))

    def into(self, x: torch.Tensor, out: torch.Tensor) -> None:
        if x.stride(-1) != 1:
            x = x.contiguous()
        xq, sx = quantize_tokens(x, self.inv)
        n = self.q.shape[0]
        mm_scaled(xq, sx, self.q, self.scale, self.out_dtype, out=out if self.tail is None else out[:, :n])
        if self.tail is not None:
            out[:, n:].copy_(mm16(x, self.tail))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty(x.shape[0], self.shape[0], dtype=self.out_dtype, device=x.device)
        self.into(x, out)
        return out


class InputMax:
    """A 16-bit weight that keeps the largest absolute value each input channel took (the calibration of SmoothQuant)
    and otherwise multiplies as the weight would. With `fetch`, `w` is an int8 weight and every product is taken with the
    16-bit weight `fetch()` reads again, gone after it, so the inputs are those of the 16-bit model."""

    def __init__(self, w: torch.Tensor, fetch=None):
        self.w = w
        self.fetch = fetch
        self.shape = w.shape
        base = getattr(w, "base", w)   # a weight with its LoRA unmerged; an int8 weight before it is smoothed
        device = base.device if isinstance(base, torch.Tensor) else base.q.device
        self.amax = torch.zeros(w.shape[1], device=device, dtype=torch.float32)

    def split(self, n: int) -> tuple["InputMax", "InputMax"]:
        """The first n output rows and the rest, keeping their maxima in this weight's."""
        parts = (self.w[:n], self.w[n:]) if isinstance(self.w, torch.Tensor) else self.w.split(n)
        fetch = self.fetch
        a, b = (InputMax(p, None if fetch is None else (lambda at: lambda: fetch()[at])(at))
                for p, at in zip(parts, (slice(None, n), slice(n, None))))
        a.amax = b.amax = self.amax
        return a, b

    def into(self, x: torch.Tensor, out: torch.Tensor) -> None:
        out.copy_(self(x))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0]:
            torch.maximum(self.amax, x.detach().abs().amax(0).float(), out=self.amax)
        return mm16(x, self.w if self.fetch is None else self.fetch())


def smoothing(amax_x: torch.Tensor, w: torch.Tensor, alpha: float = 0.75) -> torch.Tensor:
    """SmoothQuant's per-channel factor max|x_j|^alpha / max|w_j|^(1 - alpha), rounded to a power of two so that
    scaling the input and the weight is exact (1 for a channel never seen)."""
    aw = w.float().abs().amax(0).clamp(min=1e-5)
    s = amax_x.clamp(min=1e-5).pow(alpha) / aw.pow(1 - alpha)
    s = torch.exp2(torch.log2(s.clamp(1e-4, 1e4)).round())
    return torch.where(amax_x > 0, s, torch.ones_like(s))

"""8-bit floating point weights for the decoder's matrix products: the weight in float8 (e4m3) with one
scale per output row (amax / 448), the input scaled per token on the fly (one Triton kernel), the product from
torch._scaled_mm (cuBLAS FP8, row-wise scales) in the model's dtype. An all-zero token (padding) keeps a tiny scale
and stays zero. Needs compute capability 8.9 or newer (Ada, Hopper, Blackwell).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .. import settings

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def supported(device) -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(device) >= (8, 9)


def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, K) weight -> (N, K) float8 and (N, 1) float32 scale with w ~ q * scale (the layout of row-scaled FP8
    checkpoints, which load to the same tensors)."""
    scale = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) * (1.0 / FP8_MAX)
    return (w.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8), scale


@triton.jit
def _quantize_tokens(X, Q, S, K, sx, sq, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    amax = tl.zeros([BLOCK], tl.float32)
    for k0 in range(0, K, BLOCK):
        cols = k0 + tl.arange(0, BLOCK)
        x = tl.load(X + row * sx + cols, mask=cols < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(x))
    s = tl.maximum(tl.max(amax, 0), 1e-12) * (1.0 / 448.0)
    tl.store(S + row, s)
    for k0 in range(0, K, BLOCK):
        cols = k0 + tl.arange(0, BLOCK)
        x = tl.load(X + row * sx + cols, mask=cols < K, other=0.0).to(tl.float32)
        q = tl.minimum(tl.maximum(tl.math.div_rn(x, s), -448.0), 448.0)
        tl.store(Q + row * sq + cols, q.to(Q.dtype.element_ty), mask=cols < K)


def quantize_tokens(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(M, K) activations (unit stride along K) -> float8 [M, K] and a [M, 1] float32 scale per token."""
    M, K = x.shape
    q = torch.empty(M, K, device=x.device, dtype=FP8)
    s = torch.empty(M, 1, device=x.device, dtype=torch.float32)
    if M:
        _quantize_tokens[(M,)](x, q, s, K, x.stride(0), q.stride(0), BLOCK=1024, num_warps=4)
    return q, s


class Fp8Weight:
    """A weight in float8 with per-row scales, called as the matrix product `x @ w.T`. `tail` rows (the small gate
    projections of a merged Gated DeltaNet input projection) stay in the model's dtype and come last."""

    def __init__(self, q: torch.Tensor, scale: torch.Tensor, tail: torch.Tensor | None, out_dtype: torch.dtype):
        self.q = q
        self.scale = scale.reshape(1, -1).contiguous()
        self.tail = tail
        self.out_dtype = out_dtype
        self.shape = (q.shape[0] + (0 if tail is None else tail.shape[0]), q.shape[1])

    @classmethod
    def from_weight(cls, w: torch.Tensor, tail_rows: int = 0) -> "Fp8Weight":
        n = w.shape[0] - tail_rows
        q, s = quantize_rows(w[:n])
        return cls(q, s, w[n:].clone() if tail_rows else None, w.dtype)

    def split(self, n: int) -> tuple["Fp8Weight", "Fp8Weight"]:
        """The first n output rows and the rest, sharing this weight's storage (no tail rows)."""
        if self.tail is not None:
            raise ValueError("a weight with tail rows does not split")
        return (Fp8Weight(self.q[:n], self.scale[0, :n], None, self.out_dtype),
                Fp8Weight(self.q[n:], self.scale[0, n:], None, self.out_dtype))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.stride(-1) != 1:
            x = x.contiguous()
        xq, sx = quantize_tokens(x)
        y = torch._scaled_mm(xq, self.q.t(), scale_a=sx, scale_b=self.scale, out_dtype=self.out_dtype)
        if self.tail is not None:
            y = torch.cat([y, x @ self.tail.T], 1)
        return y


def mm(x: torch.Tensor, w) -> torch.Tensor:
    """x @ w.T for a plain weight, the FP8 product for an Fp8Weight."""
    if settings.batch_invariant and isinstance(w, torch.Tensor):
        from .invariant import mm as imm

        return imm(x, w)
    return x @ w.T if isinstance(w, torch.Tensor) else w(x)


def mm_into(x: torch.Tensor, w, out: torch.Tensor) -> None:
    """`mm` written into `out`."""
    if settings.batch_invariant and isinstance(w, torch.Tensor) and out.is_contiguous():
        from .invariant import mm as imm

        imm(x, w, out)
    elif isinstance(w, torch.Tensor):
        torch.mm(x, w.T, out=out)
    elif hasattr(w, "into"):
        w.into(x, out)
    else:
        out.copy_(w(x))

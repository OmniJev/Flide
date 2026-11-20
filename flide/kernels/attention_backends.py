"""Optional attention backends. Selection is process-wide and fixed before graph capture.

The generic Triton kernel remains responsible for all rows not explicitly covered by
an optimized backend. Metadata describes semantics, never a model/checkpoint name.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .. import settings
from .attention import _tile_source

log = logging.getLogger(__name__)
BACKENDS = ("auto", "triton", "cute")
CUTE_HEAD_DIMS = (64, 128, 256)
_warned = set()


def configure(name: str | None = None) -> None:
    name = settings.attention_backend if name is None else name
    if name not in BACKENDS:
        raise ValueError(f"attention_backend must be one of {BACKENDS}, got {name!r}")
    settings.attention_backend = name
    if name == "cute":
        _cute()   # Explicit requests fail early on missing/incompatible optional dependencies.
    log.info("attention backend: %s (unsupported inputs retain Triton)", name)


def _cute():
    from . import cute_attention
    try:
        return cute_attention.load()
    except (ImportError, AttributeError) as exc:
        if settings.attention_backend == "cute":
            raise RuntimeError("CuTe attention dependencies are unavailable; install the documented optional "
                               "CuTe runtime with pip install '.[cute]' or select --attention-backend triton") from exc
        if "unavailable" not in _warned:
            log.info("CuTe attention unavailable; using Triton: %s", exc)
            _warned.add("unavailable")
        return None


@dataclass(frozen=True)
class ExactAttention:
    """Device metadata shared across layers and rebuilt on every graph replay."""
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    used: torch.Tensor
    k_lengths: torch.Tensor
    ranges: torch.Tensor
    tiles: torch.Tensor
    covered: torch.Tensor
    kv_capacity: int


@triton.jit
def _prepare_exact(R, ROWS, TABLE, CUQ, CUK, USED, KLENS, COVERED,
                   T: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
                   NT: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr):
    tile = tl.program_id(0)
    lo, hi = tl.load(ROWS + tile * 10), tl.load(ROWS + tile * 10 + 1)
    q = lo + tl.arange(0, 64)
    m = (q >= 0) & (q < hi) & (q < T)
    a = tl.load(R + q * SR, m, 0); b = tl.load(R + q * SR + SC, m, 0)
    c = tl.load(R + q * SR + 2 * SC, m, 0); d = tl.load(R + q * SR + 3 * SC, m, 0)
    valid = (lo >= 0) & (hi <= T) & (hi > lo) & (hi - lo <= 64)
    valid &= tl.sum((m & (((a < b) & ((a < 0) | (b > K1))) |
                           ((c < d) & ((c < 0) | (d > K2))))).to(tl.int32), 0) == 0
    # The same union/intersection semantics as Triton, with the CuTe KV tile size.
    a0, a1, a2, a3 = _tile_source(a, b, m & valid, 64, True)
    b0, b1, b2, b3 = _tile_source(c, d, m & valid, 64, True)
    tl.store(CUQ + tile, tl.maximum(lo, 0)); tl.store(CUK + tile, a0)
    if tile == NT - 1:
        tl.store(CUQ + NT, T); tl.store(CUK + NT, 0)
    tl.store(USED + tile, tl.where(valid, hi - lo, 0))
    tl.store(KLENS + tile, a3 - a0 + b3 - b0)
    tl.store(TABLE + tile * 10, lo); tl.store(TABLE + tile * 10 + 1, hi)
    tl.store(TABLE + tile * 10 + 2, a0); tl.store(TABLE + tile * 10 + 3, a1)
    tl.store(TABLE + tile * 10 + 4, a2); tl.store(TABLE + tile * 10 + 5, a3)
    tl.store(TABLE + tile * 10 + 6, b0); tl.store(TABLE + tile * 10 + 7, b1)
    tl.store(TABLE + tile * 10 + 8, b2); tl.store(TABLE + tile * 10 + 9, b3)
    tl.store(COVERED + q, valid, mask=m)


def prepare_attention(ranges, tiles, *, dtype, head_dim, kv_capacity=None):
    """Reuse existing token tiles and exact ranges; no model names or CPU readback.

    Exact ranges already encode causal, bidirectional and clipped-window visibility.
    Shapes or semantics without this representation retain the generic Triton path.
    """
    if (settings.attention_backend == "triton" or settings.batch_invariant or not ranges.is_cuda
            or dtype != torch.bfloat16 or head_dim not in CUTE_HEAD_DIMS or ranges.ndim != 2 or ranges.shape[0] != 4
            or ranges.shape[1] == 0 or torch.cuda.get_device_capability(ranges.device) != (12, 0)
            or _cute() is None):
        return None
    if tiles is None:
        return None
    if (tiles.ndim != 2 or tiles.shape[1] != 10 or tiles.dtype != torch.int32
            or tiles.device != ranges.device or not tiles.is_contiguous()):
        return None
    t, n = ranges.shape[1], tiles.shape[0]
    if not n:
        return None
    capacity = t if kv_capacity is None else kv_capacity
    r = ranges.to(torch.int32).contiguous()
    args = dict(device=r.device, dtype=torch.int32)
    cuq, cuk = torch.empty(n + 1, **args), torch.empty(n + 1, **args)
    used, lengths = torch.empty(n, **args), torch.empty(n, **args)
    table = torch.empty(n, 10, **args)
    covered = torch.zeros(t, dtype=torch.uint8, device=r.device)
    _prepare_exact[(n,)](r, tiles, table, cuq, cuk, used, lengths, covered,
                         t, capacity, t, n, r.stride(1), r.stride(0))
    return ExactAttention(cuq, cuk, used, lengths, r, table, covered, capacity)


def optimized_attention(q, k1, v1, k2, v2, scale, window, metadata):
    if (metadata is None or settings.attention_backend == "triton" or settings.batch_invariant or window
            or q.ndim != 3 or q.shape[0] != metadata.covered.numel()
            or metadata.kv_capacity > k1.shape[0] or k2.shape[0] != q.shape[0]
            or k1.shape != v1.shape or k2.shape != v2.shape or k1.shape[1] != k2.shape[1]
            or q.shape[1] % k1.shape[1]
            or q.shape[-1] not in CUTE_HEAD_DIMS
            or any(x.dtype != torch.bfloat16 or x.shape[-1] != q.shape[-1] or x.device != q.device
                   or x.stride(-1) != 1 or x.data_ptr() % 16 or any(s % 8 for s in x.stride()[:-1])
                   for x in (q, k1, v1, k2, v2))):
        return None
    backend = _cute()
    if backend is None:
        return None
    # CUDA execution failures must propagate; a failed launch cannot safely fall back.
    return backend(q, k1, v1, k2, v2, metadata, scale)

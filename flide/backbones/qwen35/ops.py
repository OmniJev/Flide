"""Pointwise pieces of the Qwen3.5 decoder (the Triton kernels of kernels/pointwise.py, compiled once per process
with no code generation at run time) and the last layer at the read rows. Arithmetic as in transformers' Qwen3.5
modules."""
from __future__ import annotations

import torch

from ... import settings
from ...kernels import pointwise as pw


def merge(linears) -> torch.Tensor:
    """One weight matrix for several nn.Linear that read the same input; each module's weight
    becomes a row slice of it, so the HF modules keep working and no memory is added."""
    for m in linears:
        if m.bias is not None:
            raise ValueError("fused projections assume bias-free linear layers")
    w = torch.cat([m.weight.data for m in linears], 0)
    at = 0
    for m in linears:
        n = m.weight.shape[0]
        m.weight.data = w[at: at + n]
        at += n
    return w


def rms(x, w, eps: float):
    return pw.rms_norm(x, w, eps)


def final_norm(norm, h):
    return pw.rms_norm(h, norm.weight, norm.eps) if settings.batch_invariant else norm(h)


def add_rms(h, d, w, eps: float):
    return pw.rms_norm(h, w, eps, residual=d)


def swiglu(gu, n: int):
    return pw.swiglu(gu, n)


def gated_rms(o, z, w, eps: float):
    return pw.gated_rms(o, z, w, eps)


def qk_norm_rope(q, k, cos, sin, wq, wk, eps: float):
    return pw.qk_norm_rope(q, k, cos, sin, wq, wk, eps)


def norm_rope(x, cos, sin, w, eps: float):
    return pw.norm_rope(x, cos, sin, w, eps)


def gate(o, g):
    return pw.gate(o, g)


def split_rows(w, n: int):
    """A weight (`x @ w.T`) as its first n output rows and the rest, sharing its storage."""
    return (w[:n], w[n:]) if isinstance(w, torch.Tensor) else w.split(n)


def last_qkv(m, w, x, rows, cos, sin):
    """Merged attention projection of the last layer: q and gate [n, heads, hd] of `rows` (q normed and rotated),
    k (normed, rotated) and v [T, kv heads, hd] of every row."""
    from ...kernels.fp8 import mm

    hd = m.head_dim
    nh, nkv = w["heads"]
    split = w.get("split")
    if split is None or split[0] is not w["qkv"]:
        split = w["split"] = (w["qkv"], *split_rows(w["qkv"], nh * 2 * hd))
    _, wq, wkv = split
    T = x.shape[0]
    kv = mm(x, wkv)
    k = norm_rope(kv[:, : nkv * hd].view(T, nkv, hd), cos, sin, m.k_norm.weight, m.q_norm.eps)
    v = kv[:, nkv * hd:].view(T, nkv, hd)
    qg = mm(x.index_select(0, rows), wq).view(-1, nh, 2 * hd)
    q = norm_rope(qg[..., :hd], cos.index_select(0, rows), sin.index_select(0, rows), m.q_norm.weight, m.q_norm.eps)
    return q, qg[..., hd:], k, v


def rows_ranges(r, rows, cfg: dict, exact: bool = True):
    """Tile table of the queries at `rows` from every query's key ranges r [4, T]; with `exact` also their ranges
    [4, n] for the kernel's masking (else None: it masks by metadata)."""
    from ...kernels import attention as attn_kernel

    rr = r.index_select(1, rows)
    R = attn_kernel.block_ranges(rr[0], rr[1], rr[2], rr[3], cfg["BLOCK_M"], cfg["BLOCK_N"], exact=exact,
                                 groups=rows.shape[0])
    return R, (rr.int() if exact else None)

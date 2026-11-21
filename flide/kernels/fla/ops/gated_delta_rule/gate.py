"""Copied from flash-linear-attention 0.6.0 (fla/ops/gated_delta_rule/gate.py), forward path only. Copyright (c) 2023-2026
Songlin Yang, Yu Zhang, Zhiyuan Li; MIT license (THIRD_PARTY_NOTICES)."""

import torch
import triton
import triton.language as tl
from flide.kernels.fla.ops.utils.index import prepare_chunk_indices
from flide.kernels.fla.ops.utils.op import exp
from flide.kernels.fla.ops.utils.softplus import softplus
from flide.kernels.fla.utils import input_guard


@triton.heuristics({
    'HAS_BIAS': lambda args: args['dt_bias'] is not None,
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['H', 'BT', 'IS_VARLEN', 'REVERSE'],
)
@triton.jit(do_not_specialize=['T'])
def gdn_gate_chunk_cumsum_scalar_kernel(
    g,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_g = g + bos * H + i_h + o_t * H
    p_o = o + bos * H + i_h + o_t * H

    b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
    if HAS_BIAS:
        b_g = b_g + tl.load(dt_bias + i_h).to(tl.float32)
    b_A = tl.load(A_log + i_h).to(tl.float32)
    b_gate = -exp(b_A) * softplus(b_g)

    b_o = tl.cumsum(b_gate, axis=0)
    if REVERSE:
        b_z = tl.sum(b_gate, axis=0)
        b_o = -b_o + b_z[None] + b_gate
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t)


@input_guard
def gdn_gate_chunk_cumsum(
    g: torch.Tensor,
    A_log: torch.Tensor,
    chunk_size: int,
    scale: float = None,
    dt_bias: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    B, T, H = g.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = torch.empty_like(g, dtype=output_dtype or g.dtype)
    gdn_gate_chunk_cumsum_scalar_kernel[(NT, B * H)](
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        o=o,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        REVERSE=False,
    )
    return o

"""Copied from flash-linear-attention 0.6.0 (fla/ops/gated_delta_rule/chunk.py), forward path only. Copyright (c) 2023-2026
Songlin Yang, Yu Zhang, Zhiyuan Li; MIT license (THIRD_PARTY_NOTICES)."""

import torch

from flide.kernels.fla.modules.norm.l2norm import l2norm_fwd
from flide.kernels.fla.ops.common.gate import fused_beta_sigmoid_fwd
from flide.kernels.fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from flide.kernels.fla.ops.common.chunk_o import chunk_fwd_o
from flide.kernels.fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from flide.kernels.fla.ops.gated_delta_rule.gate import gdn_gate_chunk_cumsum
from flide.kernels.fla.ops.utils.constant import RCP_LN2
from flide.kernels.fla.ops.utils.index import prepare_chunk_indices
from flide.kernels.fla.utils import input_guard


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    chunk_size: int = 64,
    initial_state_indices: torch.Tensor | None = None,
):
    g_input = g if use_gate_in_kernel else None
    if use_gate_in_kernel:
        g = gdn_gate_chunk_cumsum(
            g=g,
            A_log=A_log,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
    else:
        raise ValueError("only the gate computed in the kernel is copied (use_gate_in_kernel=True)")
    # obtain WY representation. u is actually the new v.
    # fused kkt + solve_tril + recompute_w_u
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        chunk_size=chunk_size,
    )

    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        chunk_size=chunk_size,
    )
    return g, o, A, final_state, initial_state, g_input


@input_guard
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    chunk_size: int = 64,
    initial_state_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """fla's chunk_gated_delta_rule with ChunkGatedDeltaRuleFunction.forward, for inference: the same checks, the same
    calls in the same order (its input_guard makes every tensor argument contiguous first), no autograd.

    For packed input, initial_state_indices maps each sequence to a valid row in initial_state.
    Multiple sequences may read the same row without modifying it.
    """
    if q.shape[2] != k.shape[2]:
        raise ValueError(f"q and k must have the same number of heads, but got q.shape[2]={q.shape[2]} and "
                         f"k.shape[2]={k.shape[2]}")
    H, HV = q.shape[2], v.shape[2]
    if HV % H != 0:
        raise ValueError(f"num_v_heads (HV={HV}) must be evenly divisible by num_heads (H={H})")
    if chunk_size not in (16, 32, 64):
        raise ValueError(f"`chunk_size` must be 16, 32, or 64 for Gated Delta Rule, got {chunk_size}.")
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.")
    if initial_state_indices is not None:
        if initial_state is None or cu_seqlens is None or initial_state_indices.shape != (len(cu_seqlens) - 1,):
            raise ValueError("indexed initial state requires a state pool and one index per packed sequence")
        if initial_state_indices.device != initial_state.device or initial_state_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("initial_state_indices must contain integers on the same device as initial_state")
    elif cu_seqlens is not None and initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
        raise ValueError(f"The number of initial states is expected to be equal to the number of input sequences, "
                         f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.")
    if use_gate_in_kernel and A_log is None:
        raise ValueError("A_log must be provided when use_gate_in_kernel=True.")
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`.")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
    if use_beta_sigmoid_in_kernel:
        beta = fused_beta_sigmoid_fwd(beta, scale=2.0 if allow_neg_eigval else 1.0)
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu)
    _, o, _, final_state, _, _ = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        chunk_size=chunk_size,
    )
    return o.to(q.dtype), final_state

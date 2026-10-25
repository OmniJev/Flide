"""Gated DeltaNet over a flat token stream with a fixed launch grid, for CUDA graphs that take any mix of requests.

The stream holds sequences in slots: prefix slots first, then branch slots (each branch starts from the final state
of its request's prefix). Lengths, chunk lists and state indices are device tensors filled before each replay; slots
past the real ones have length zero. The per-chunk kernels run over a fixed-length chunk list whose entries past the
real chunks point at an empty slot; a graph sizes the list for its worst round (many short sequences), so a long
sequence fills under half of it and those entries must cost nothing. fla's fused kkt + solve kernel returns on them.
The recurrence over chunks and the outputs run here in one kernel, once per level of prefix slots and once for the
branch slots (reading their initial state through an index); it skips empty slots and takes the initial state by
index. It is derived from flash-linear-attention's chunk_gated_delta_rule_fwd_kernel_h_blockdim64 and
chunk_fwd_kernel_o, MIT licence, Songlin Yang, Yu Zhang, Zhiyuan Li.

The short causal convolution reads each token's three predecessors through an index (the last prefix tokens for the
first tokens of a branch, nothing before a sequence's start), so no convolution state is kept either.

Prefixes can run in levels (sessions: up to a save point, then the rest): slots of level 1 start from zero or from a
saved state (by index), slots of each later level and the branches from a final state of an earlier level (by index);
every level's final states stay available for saving.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

RCP_LN2 = 1.4426950408889634


def _blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def solve(k, g, beta, cu, chunks, HV: int, chunk_size: int = 64):
    """(I + A)^-1 of every chunk in the list, A the strictly lower, gated and beta-weighted part of K K^T, as
    [1, T, HV, 64] in k's dtype (fla's fused kkt + solve kernel, which returns on padding entries)."""
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel

    if chunk_size != 64:
        raise ValueError(f"chunk size {chunk_size}: the fused solve takes 64")
    _, T, H, K = k.shape
    A = torch.zeros(1, T, HV, chunk_size, device=k.device, dtype=k.dtype)
    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(chunks.shape[0], HV)](
        k=k, g=g, beta=beta, A=A, cu_seqlens=cu, chunk_indices=chunks, T=T, H=H, HV=HV, K=K, BT=chunk_size, BC=16)
    return A


def _config_o(HV: int) -> dict:
    """The fused recurrence's launch configuration, fixed (an autotuner sees one round, usually a single sequence at
    capture, while the choice matters most for rounds of several: up to 2x on H100). Swept over single long sequences
    and rounds of 4 long or 16 short requests: 64-wide value blocks, 4 warps, 2 stages are
    within 4% of the best everywhere at 32 and 48 value heads; at 16 the 32-wide block costs single sequences less.
    Blackwell runs 2 warps (`_configs`), where a 64-wide block spills."""
    from fla.utils import check_shared_mem

    if _blackwell():
        return dict(BV=32, num_warps=2, num_stages=2)
    wide = HV >= 32 and check_shared_mem("ada")
    return dict(BV=64 if wide else 32, num_warps=4, num_stages=2 if check_shared_mem("ampere") else 1)


@triton.jit
def _slot_x(q, k, v, A, beta, g, o, h0, h0_idx, ht, cu, i_n, i_h, i_v, scale, H: tl.constexpr, HV: tl.constexpr,
            K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, USE_H0: tl.constexpr,
            STORE_HT: tl.constexpr, H0_SK: tl.constexpr, H0_SV: tl.constexpr):
    """The recurrence over one slot's chunks and its outputs, with the corrected values made from the chunk's solved
    matrix: v_new = (I + A)^-1 diag(beta) v - (I + A)^-1 diag(beta) ((exp(g) k) S), so no w, u kernel and none of its
    two [T, HV, 128] outputs. The two products run apart, v as stored and (exp(g) k) S rounded to 16 bits: rounding
    their difference instead loses the digits the solved matrix multiplies up where it is ill-conditioned. H0_SK,
    H0_SV: the strides of a key and a value in a row of h0."""
    bos, eos = tl.load(cu + i_n).to(tl.int32), tl.load(cu + i_n + 1).to(tl.int32)
    T = eos - bos
    if T > 0:
        NT = tl.cdiv(T, BT)
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        pv = v + (bos * HV + i_h).to(tl.int64) * V
        pk = k + (bos * H + i_h // (HV // H)).to(tl.int64) * K
        pq = q + (bos * H + i_h // (HV // H)).to(tl.int64) * K
        pa = A + (bos * HV + i_h).to(tl.int64) * BT
        pb = beta + (bos * HV + i_h).to(tl.int64)
        po = o + (bos * HV + i_h).to(tl.int64) * V

        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        o_k1 = tl.arange(0, 64)
        m_k1 = o_k1 < K
        o_k2 = 64 + o_k1
        m_k2 = o_k2 < K
        o_c = tl.arange(0, BT)
        if USE_H0:
            src = tl.load(h0_idx + i_n).to(tl.int64)
            p0 = h0 + (src * HV + i_h) * K * V
            b_h1 += tl.load(p0 + o_k1[:, None] * H0_SK + o_v[None, :] * H0_SV, mask=m_k1[:, None] & m_v[None, :],
                            other=0.0).to(tl.float32)
            if K > 64:
                b_h2 += tl.load(p0 + o_k2[:, None] * H0_SK + o_v[None, :] * H0_SV, mask=m_k2[:, None] & m_v[None, :],
                                other=0.0).to(tl.float32)

        for i_t in range(NT):
            o_t = i_t * BT + o_c
            m_t = o_t < T
            last_idx = min((i_t + 1) * BT, T) - 1
            b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
            b_g = tl.load(g + (bos * HV + i_h).to(tl.int64) + o_t * HV, mask=m_t, other=0.0).to(tl.float32)
            e_g = tl.math.exp2(b_g)

            s_h1 = b_h1.to(pq.dtype.element_ty)
            b_k1 = tl.load(pk + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_t[:, None] & m_k1[None, :], other=0.0)
            b_s = tl.dot((b_k1 * e_g[:, None]).to(b_k1.dtype), s_h1)
            b_q = tl.load(pq + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_t[:, None] & m_k1[None, :], other=0.0)
            b_o = tl.dot(b_q, s_h1)
            b_A = tl.dot(b_q, tl.trans(b_k1))
            if K > 64:
                s_h2 = b_h2.to(pq.dtype.element_ty)
                b_k2 = tl.load(pk + o_t[:, None] * (H * K) + o_k2[None, :], mask=m_t[:, None] & m_k2[None, :],
                               other=0.0)
                b_s += tl.dot((b_k2 * e_g[:, None]).to(b_k2.dtype), s_h2)
                b_q = tl.load(pq + o_t[:, None] * (H * K) + o_k2[None, :], mask=m_t[:, None] & m_k2[None, :],
                              other=0.0)
                b_o += tl.dot(b_q, s_h2)
                b_A += tl.dot(b_q, tl.trans(b_k2))
            b_x = tl.load(pv + o_t[:, None] * (HV * V) + o_v[None, :], mask=m_t[:, None] & m_v[None, :], other=0.0)
            b_b = tl.load(pb + o_t * HV, mask=m_t, other=0.0).to(tl.float32)
            b_T = tl.load(pa + o_t[:, None] * (HV * BT) + o_c[None, :], mask=m_t[:, None], other=0.0).to(tl.float32)
            b_T = (b_T * b_b[None, :]).to(pq.dtype.element_ty)
            b_v = tl.dot(b_T, b_x) - tl.dot(b_T, b_s.to(pq.dtype.element_ty))

            b_vs = b_v.to(pq.dtype.element_ty)
            b_o = b_o * e_g[:, None]
            b_A = b_A * tl.math.exp2(b_g[:, None] - b_g[None, :])
            b_A = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t), b_A, 0)
            b_o = b_o * scale + tl.dot(b_A.to(b_vs.dtype), b_vs) * scale
            tl.store(po + o_t[:, None] * (HV * V) + o_v[None, :], b_o.to(po.dtype.element_ty),
                     mask=m_t[:, None] & m_v[None, :])

            b_v = b_v * tl.where(m_t, tl.math.exp2(b_g_last - b_g), 0)[:, None]
            b_g_last = tl.math.exp2(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            b_v = b_v.to(pk.dtype.element_ty)
            b_h1 += tl.dot(tl.trans(b_k1), b_v)
            if K > 64:
                b_h2 += tl.dot(tl.trans(b_k2), b_v)

        if STORE_HT:
            pt = ht + ((i_n * HV + i_h) * K * V).to(tl.int64)
            tl.store(pt + o_k1[:, None] * V + o_v[None, :], b_h1.to(ht.dtype.element_ty),
                     mask=m_k1[:, None] & m_v[None, :])
            if K > 64:
                tl.store(pt + o_k2[:, None] * V + o_v[None, :], b_h2.to(ht.dtype.element_ty),
                         mask=m_k2[:, None] & m_v[None, :])


@triton.jit(do_not_specialize=["T"])
def _fwd_xo(q, k, v, A, beta, g, o, h0, h0_idx, ht, cu, scale, T, H: tl.constexpr, HV: tl.constexpr,
            K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, USE_H0: tl.constexpr,
            STORE_HT: tl.constexpr, H0_SK: tl.constexpr, H0_SV: tl.constexpr):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    _slot_x(q, k, v, A, beta, g, o, h0, h0_idx, ht, cu, i_nh // HV, i_nh % HV, i_v, scale, H, HV, K, V, BT, BV,
            USE_H0, STORE_HT, H0_SK, H0_SV)


def fwd_xo(q, k, v, A, beta, g, o, cu, scale: float, h0=None, h0_idx=None, ht=None, chunk_size: int = 64) -> None:
    """Outputs o and final states ht of the slots cu[:-1] from the solved matrices of `solve` and the values
    themselves (`_slot_x`); a slot starts from h0[h0_idx[slot]] when h0 is given, else from zero."""
    _, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    if K > 128:
        raise ValueError(f"key head dim {K} > 128")
    N = cu.shape[0] - 1
    cfg = _config_o(HV)
    _fwd_xo[(triton.cdiv(V, cfg["BV"]), N * HV)](
        q, k, v, A, beta, g, o, h0, h0_idx, ht, cu, scale, T, H=H, HV=HV, K=K, V=V, BT=chunk_size,
        USE_H0=h0 is not None, STORE_HT=ht is not None, H0_SK=V, H0_SV=1, **cfg)


@triton.jit
def _conv(X, X2, W, SRC, Q, Kt, Vt, T, sx, eps, KD: tl.constexpr, VD: tl.constexpr, BLOCK_T: tl.constexpr,
          BLOCK_C: tl.constexpr, SAVED: tl.constexpr, NORM: tl.constexpr):
    pid_t, pid_c = tl.program_id(0), tl.program_id(1)
    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    c0 = pid_c * BLOCK_C
    c = c0 + tl.arange(0, BLOCK_C)
    mt = t < T
    C: tl.constexpr = 2 * KD + VD
    mc = c < C
    m = mt[:, None] & mc[None, :]
    acc = tl.load(X + t[:, None] * sx + c[None, :], mask=m, other=0.0).to(tl.float32) * \
        tl.load(W + c * 4 + 3, mask=mc, other=0.0).to(tl.float32)[None, :]
    for j in tl.static_range(3):
        s = tl.load(SRC + j * T + t, mask=mt, other=-1)
        ok = (s >= 0) & (s < T) & mt
        xs = tl.load(X + s[:, None].to(tl.int64) * sx + c[None, :], mask=ok[:, None] & mc[None, :], other=0.0)
        if SAVED:
            far = (s >= T) & mt
            xs += tl.load(X2 + (s - T)[:, None].to(tl.int64) * C + c[None, :], mask=far[:, None] & mc[None, :],
                          other=0.0)
        acc += xs.to(tl.float32) * tl.load(W + c * 4 + (2 - j), mask=mc, other=0.0).to(tl.float32)[None, :]
    y = acc * tl.sigmoid(acc)
    if NORM:
        if c0 < 2 * KD:
            yb = y.to(Q.dtype.element_ty).to(tl.float32)
            y = yb * (1 / tl.sqrt(tl.sum(yb * yb, 1) + eps))[:, None]
    if c0 < KD:
        tl.store(Q + t[:, None] * KD + c[None, :], y.to(Q.dtype.element_ty), mask=m)
    elif c0 < 2 * KD:
        tl.store(Kt + t[:, None] * KD + (c[None, :] - KD), y.to(Kt.dtype.element_ty), mask=m)
    else:
        tl.store(Vt + t[:, None] * VD + (c[None, :] - 2 * KD), y.to(Vt.dtype.element_ty), mask=m)


def conv(x, weight, src, kd: int, vd: int, saved=None, l2norm_heads: int = 0):
    """Causal depthwise convolution (width 4) and SiLU over a flat stream. x [T, >= 2*kd + vd] (row stride free),
    weight [2*kd + vd, 4], src [3, T] int32: stream index of each token's predecessor 1, 2 and 3 back (-1: none;
    T + r: row r of `saved` [rows, 2*kd + vd], contiguous). Returns q [T, kd], k [T, kd], v [T, vd], each contiguous;
    with `l2norm_heads` (the key head size, which must be the 128-channel block), q and k L2-normalized per head."""
    T = x.shape[0]
    q = torch.empty(T, kd, device=x.device, dtype=x.dtype)
    k = torch.empty(T, kd, device=x.device, dtype=x.dtype)
    v = torch.empty(T, vd, device=x.device, dtype=x.dtype)
    BLOCK_C = 128
    if kd % BLOCK_C or vd % BLOCK_C:
        raise ValueError(f"key / value widths {kd} / {vd} must be multiples of {BLOCK_C}")
    BLOCK_T = 32
    grid = (triton.cdiv(T, BLOCK_T), (2 * kd + vd) // BLOCK_C)
    if l2norm_heads and l2norm_heads != BLOCK_C:
        raise ValueError(f"fused L2 norm needs key heads of {BLOCK_C} channels, not {l2norm_heads}")
    _conv[grid](x, x if saved is None else saved, weight, src, q, k, v, T, x.stride(0), 1e-6, KD=kd, VD=vd,
                BLOCK_T=BLOCK_T, BLOCK_C=BLOCK_C, SAVED=saved is not None, NORM=bool(l2norm_heads), num_warps=4)
    return q, k, v


@triton.jit
def _gates(P, BETA, A, T, sp, NV: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_T: tl.constexpr):
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    h = tl.arange(0, BLOCK_H)
    m = (t < T)[:, None] & (h < NV)[None, :]
    b = tl.load(P + t[:, None].to(tl.int64) * sp + h[None, :], mask=m, other=0.0).to(tl.float32)
    tl.store(BETA + t[:, None] * NV + h[None, :], 1.0 * tl.sigmoid(b), mask=m)
    a = tl.load(P + t[:, None].to(tl.int64) * sp + NV + h[None, :], mask=m, other=0.0)
    tl.store(A + t[:, None] * NV + h[None, :], a, mask=m)


def gates(p, nv: int):
    """The beta logits and gate inputs of a Gated DeltaNet projection (the last 2 * nv columns of p) -> sigmoid(beta)
    float32 [1, T, nv] (as fla's fused_beta_sigmoid) and the gate input [1, T, nv] contiguous."""
    T = p.shape[0]
    tail = p[:, p.shape[1] - 2 * nv:]
    beta = torch.empty(1, T, nv, device=p.device, dtype=torch.float32)
    a = torch.empty(1, T, nv, device=p.device, dtype=p.dtype)
    if T:
        _gates[(triton.cdiv(T, 64),)](tail, beta, a, T, p.stride(0), NV=nv, BLOCK_H=triton.next_power_of_2(nv),
                                      BLOCK_T=64, num_warps=4)
    return a, beta


def flat(q, k, v, a, beta, A_log, dt_bias, cu, chunks, levels, h0_idx, resume=None, resume_idx=None,
         chunk_size: int = 64, prepared: bool = False):
    """Gated DeltaNet over a flat stream: (output [1, T, HV, V], final states [sum(levels), HV, K, V] float32 of the
    prefix slots, level by level). q, k [1, T, H, K]; v [1, T, HV, V]; a, beta [1, T, HV] raw (gate input and beta
    logits); cu [S + 1] over all slots (the prefix levels' slots in order, then the branch slots); chunks [NT, 2] the
    stream's chunk list (slot, chunk), padded with chunks of an empty slot; h0_idx [S - levels[0]]: for every slot
    after level 1, the final state (row of the returned states) it starts from; level 1 starts from
    resume[resume_idx[i]] when given, else from zero. `prepared`: q and k already L2-normalized, beta already through
    its sigmoid (`conv(..., l2norm_heads=...)`, `gates`)."""
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.common.gate import fused_beta_sigmoid
    from fla.ops.gated_delta_rule.gate import gdn_gate_chunk_cumsum

    if not prepared:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
        beta = fused_beta_sigmoid(beta, scale=1.0)
    _, T, HV, V = v.shape
    K = k.shape[-1]
    g = gdn_gate_chunk_cumsum(a, A_log, chunk_size, scale=RCP_LN2, dt_bias=dt_bias, cu_seqlens=cu, chunk_indices=chunks)
    A = solve(k, g, beta, cu, chunks, HV, chunk_size)
    o = torch.empty_like(v)
    ht = torch.empty(sum(levels), HV, K, V, device=k.device, dtype=torch.float32)
    P0 = levels[0]
    fwd_xo(q, k, v, A, beta, g, o, cu[: P0 + 1], K ** -0.5, h0=resume, h0_idx=resume_idx, ht=ht[:P0],
           chunk_size=chunk_size)
    at = P0
    for n in levels[1:]:
        fwd_xo(q, k, v, A, beta, g, o, cu[at: at + n + 1], K ** -0.5, h0=ht, h0_idx=h0_idx[at - P0: at - P0 + n],
               ht=ht[at: at + n], chunk_size=chunk_size)
        at += n
    fwd_xo(q, k, v, A, beta, g, o, cu[at:], K ** -0.5, h0=ht, h0_idx=h0_idx[at - P0:], chunk_size=chunk_size)
    return o, ht

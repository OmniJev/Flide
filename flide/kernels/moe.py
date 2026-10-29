"""Mixture-of-experts feed-forward (Qwen3.5 MoE, Gemma 4 26B-A4B): the router's top k experts for every token, then
each expert's (token, slot) pairs in blocks of one expert through two grouped products, as fused MoE kernels run them.

Every shape is fixed by the token count (the pairs are sorted and each expert's group padded to a block on the
device), so the products replay inside CUDA graphs. A token's output comes from its own rows alone, in a fixed order
over its k experts, so forking and the batch-invariant mode keep their guarantees. The router and shared expert gate
logits (in float32), the routing (softmax, top k, renormalized), the sort of the pairs by expert and the sum over a
token's experts with the shared expert are one kernel each.

In 16 bits the expert rows go to the sum over a token's experts in float32, so the layer's output is rounded once. With
FP8 (quantize_experts) the routed experts are float8 (e4m3) with one scale per expert and output row, the rows going in
are scaled per row on the fly (kernels/fp8.py), and the grouped products multiply float8 by float8 into float32, as
kernels/fp8.py does for the dense products; the router and the shared expert keep their 16-bit weights.
"""
from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 128, 64
BK8 = 128


@functools.lru_cache(maxsize=None)
def _big(device) -> bool:
    return getattr(torch.cuda.get_device_properties(device), "shared_memory_per_block_optin", 0) >= 160 * 1024


def tile(pairs: int, fp8: bool, device) -> tuple[int, int, int, int, int]:
    """(block rows, block columns, warps, stages, row blocks per group) for a layer of `pairs` (token, expert) pairs,
    from bench/moe_sweep.py on H200 (Qwen3.5-35B-A3B's and Gemma 4 26B-A4B's experts). The larger tiles need the 227 KB
    of shared memory per block of H100 / H200 (120 to 144 KB; about 99 KB on consumer and workstation cards). A row's
    sums do not depend on the tile or the order the blocks run in (the k step is fixed; float8 keeps 64 block rows,
    whose products sum like the default's), so the bits do not change with the round."""
    big = _big(device)
    if big and fp8:
        return (64, 128, 4, 4, 8) if pairs >= 15000 else (64, 128, 8, 4, 4)
    if big and pairs >= 28000:
        return 128, 256, 8, 3, 8
    return BM, BN, 4, 3, 4


@triton.jit
def _grouped_kernel(A, B, C, WT, ROWS, BLK_E, USED, n_pairs, N, KD, topk, sa, sbe, sbn, sc, NBM,
                    BY_TOKEN: tl.constexpr, MUL_W: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                    BK: tl.constexpr, GM: tl.constexpr):
    if GM:
        pid = tl.program_id(0)
        width = GM * tl.cdiv(N, BN)
        first = pid // width * GM
        size = tl.minimum(NBM - first, GM)
        pm = first + pid % width % size
        pn = pid % width // size
    else:
        pm = tl.program_id(0)
        pn = tl.program_id(1)
    if pm * BM >= tl.load(USED):
        return
    e = tl.load(BLK_E + pm).to(tl.int64)
    pairs = tl.load(ROWS + pm * BM + tl.arange(0, BM))
    valid = pairs < n_pairs
    arow = pairs // topk if BY_TOKEN else pairs
    cols = pn * BN + tl.arange(0, BN)
    mc = cols < N
    ks = tl.arange(0, BK)
    a_ptr = A + arow[:, None].to(tl.int64) * sa + ks[None, :]
    b_ptr = B + e * sbe + cols[None, :].to(tl.int64) * sbn + ks[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, KD, BK):
        mk = ks < KD - k0
        a = tl.load(a_ptr, mask=valid[:, None] & mk[None, :], other=0.0)
        b = tl.load(b_ptr, mask=mk[:, None] & mc[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptr += BK
        b_ptr += BK
    if MUL_W:
        acc = acc * tl.load(WT + pairs, mask=valid, other=0.0)[:, None]
    c_ptr = C + pairs[:, None].to(tl.int64) * sc + cols[None, :]
    tl.store(c_ptr, acc.to(C.dtype.element_ty), mask=valid[:, None] & mc[None, :])


@triton.jit
def _grouped_fp8_kernel(A, SA, B, SB, C, WT, ROWS, BLK_E, USED, n_pairs, N, KD, topk, sa, sbe, sbn, sse, sc, NBM,
                        BY_TOKEN: tl.constexpr, MUL_W: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                        BK: tl.constexpr, GM: tl.constexpr):
    if GM:
        pid = tl.program_id(0)
        width = GM * tl.cdiv(N, BN)
        first = pid // width * GM
        size = tl.minimum(NBM - first, GM)
        pm = first + pid % width % size
        pn = pid % width // size
    else:
        pm = tl.program_id(0)
        pn = tl.program_id(1)
    if pm * BM >= tl.load(USED):
        return
    e = tl.load(BLK_E + pm).to(tl.int64)
    pairs = tl.load(ROWS + pm * BM + tl.arange(0, BM))
    valid = pairs < n_pairs
    arow = pairs // topk if BY_TOKEN else pairs
    cols = pn * BN + tl.arange(0, BN)
    mc = cols < N
    ks = tl.arange(0, BK)
    a_ptr = A + arow[:, None].to(tl.int64) * sa + ks[None, :]
    b_ptr = B + e * sbe + cols[None, :].to(tl.int64) * sbn + ks[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, KD, BK):
        mk = ks < KD - k0
        a = tl.load(a_ptr, mask=valid[:, None] & mk[None, :], other=0.0)
        b = tl.load(b_ptr, mask=mk[:, None] & mc[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptr += BK
        b_ptr += BK
    sb = tl.load(SB + e * sse + cols, mask=mc, other=0.0)
    acc = acc * tl.load(SA + arow, mask=valid, other=0.0)[:, None] * sb[None, :]
    if MUL_W:
        acc = acc * tl.load(WT + pairs, mask=valid, other=0.0)[:, None]
    c_ptr = C + pairs[:, None].to(tl.int64) * sc + cols[None, :]
    tl.store(c_ptr, acc.to(C.dtype.element_ty), mask=valid[:, None] & mc[None, :])


@triton.jit
def _router_kernel(X, WR, WG, L, G, T, E, H, sx, swr, sl, HAS_G: tl.constexpr, BT: tl.constexpr, BE: tl.constexpr,
                   BK: tl.constexpr):
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    cols = tl.program_id(1) * BE + tl.arange(0, BE)
    mr = rows < T
    mc = cols < E
    ks = tl.arange(0, BK)
    acc = tl.zeros((BT, BE), dtype=tl.float32)
    g = tl.zeros((BT,), dtype=tl.float32)
    for k0 in range(0, H, BK):
        mk = k0 + ks < H
        a = tl.load(X + rows[:, None].to(tl.int64) * sx + (k0 + ks)[None, :], mask=mr[:, None] & mk[None, :], other=0.0)
        b = tl.load(WR + cols[None, :].to(tl.int64) * swr + (k0 + ks)[:, None], mask=mk[:, None] & mc[None, :],
                    other=0.0)
        acc += tl.dot(a, b)
        if HAS_G:
            wg = tl.load(WG + k0 + ks, mask=mk, other=0.0).to(tl.float32)
            g += tl.sum(a.to(tl.float32) * wg[None, :], 1)
    tl.store(L + rows[:, None].to(tl.int64) * sl + cols[None, :], acc, mask=mr[:, None] & mc[None, :])
    if HAS_G:
        tl.store(G + rows, g, mask=mr & (tl.program_id(1) == 0))


def router(x: torch.Tensor, wr: torch.Tensor, wg: torch.Tensor | None = None):
    """(router logits [T, E], shared expert gate logits [T, 1] or None without `wg`) for rows x [T, H], both in
    float32: summed in float32 and never rounded to 16 bits, where experts whose 16-bit logits tie would be picked by
    the order of a sort."""
    if x.stride(-1) != 1:
        x = x.contiguous()
    T, H = x.shape
    E = wr.shape[0]
    logits = torch.empty(T, E, device=x.device, dtype=torch.float32)
    gate = torch.empty(T, 1, device=x.device, dtype=torch.float32) if wg is not None else None
    if T:
        BE = min(256, triton.next_power_of_2(E))
        _router_kernel[(triton.cdiv(T, 16), triton.cdiv(E, BE))](x, wr, wg if wg is not None else wr, logits,
                                                                  gate if gate is not None else logits, T, E, H,
                                                                  x.stride(0), wr.stride(0), logits.stride(0),
                                                                  HAS_G=wg is not None, BT=16, BE=BE, BK=64,
                                                                  num_warps=4, num_stages=3)
    return logits, gate


@triton.jit
def _route_kernel(L, W, IDS, PS, E, sl, K: tl.constexpr, BE: tl.constexpr, SCALED: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    e = tl.arange(0, BE)
    x = tl.load(L + t * sl + e, mask=e < E, other=float("-inf")).to(tl.float32)
    p = tl.exp(x - tl.max(x, 0))
    p = tl.where(e < E, p / tl.sum(p, 0), -1.0)
    slot = tl.arange(0, K)
    top = tl.zeros([K], tl.float32)
    idx = tl.zeros([K], tl.int32)
    for j in tl.static_range(K):
        m = tl.max(p, 0)
        i = tl.min(tl.where(p == m, e, BE), 0)
        top = tl.where(slot == j, m, top)
        idx = tl.where(slot == j, i, idx)
        p = tl.where(e == i, -1.0, p)
    if SCALED:
        tl.store(W + t * K + slot, top / tl.sum(top, 0) * tl.load(PS + idx).to(tl.float32))
    else:
        tl.store(W + t * K + slot, top / tl.sum(top, 0))
    tl.store(IDS + t * K + slot, idx)


def route(logits: torch.Tensor, k: int, scale: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(weights [T, k] float32, experts [T, k] int32) from router logits [T, E]: the softmax's top k in descending
    order, renormalized to sum to one, then times each expert's `scale` [E] when given (Gemma 4)."""
    T, E = logits.shape
    w = torch.empty(T, k, device=logits.device, dtype=torch.float32)
    ids = torch.empty(T, k, device=logits.device, dtype=torch.int32)
    if T:
        _route_kernel[(T,)](logits, w, ids, scale if scale is not None else w, E, logits.stride(0), K=k,
                            BE=triton.next_power_of_2(E), SCALED=scale is not None, num_warps=4)
    return w, ids


@triton.jit
def _count_kernel(IDS, COUNTS, n, BLOCK: tl.constexpr):
    e = tl.program_id(0)
    c = tl.zeros([BLOCK], tl.int32)
    for i0 in range(0, n, BLOCK):
        i = i0 + tl.arange(0, BLOCK)
        c += (tl.load(IDS + i, mask=i < n, other=-1) == e).to(tl.int32)
    tl.store(COUNTS + e, tl.sum(c, 0))


@triton.jit
def _place_kernel(IDS, ENDS, PADDED, ROWS, BLK_E, n, BM: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.program_id(0)
    end = tl.load(ENDS + e)
    start = end - tl.load(PADDED + e)
    at = start
    for i0 in range(0, n, BLOCK):
        i = i0 + tl.arange(0, BLOCK)
        hit = tl.load(IDS + i, mask=i < n, other=-1) == e
        h = hit.to(tl.int32)
        tl.store(ROWS + at + tl.cumsum(h, 0) - 1, i, mask=hit)
        at += tl.sum(h, 0)
    for r0 in range(at, end, BLOCK):
        r = r0 + tl.arange(0, BLOCK)
        tl.store(ROWS + r, n, mask=r < end)
    for b0 in range(start // BM, end // BM, BLOCK):
        b = b0 + tl.arange(0, BLOCK)
        tl.store(BLK_E + b, e, mask=b < end // BM)


def align(ids: torch.Tensor, experts: int, bm: int = BM):
    """(rows, expert of each block, rows in use) for top-k ids [T, k]: the pair indices (token * k + slot) sorted by
    expert, in pair order within an expert, each expert's group padded to a multiple of `bm` with the index T * k,
    which no product reads. Blocks past the rows in use are left unset (the products stop before them)."""
    T, K = ids.shape
    n = T * K
    dev = ids.device
    flat = ids.reshape(-1)
    counts = torch.empty(experts, dtype=torch.int32, device=dev)
    _count_kernel[(experts,)](flat, counts, n, BLOCK=1024, num_warps=4)
    padded = (counts + bm - 1) // bm * bm
    ends = torch.cumsum(padded, 0, dtype=torch.int32)
    total = -(-(n + experts * (bm - 1)) // bm) * bm
    rows = torch.empty(total, dtype=torch.int32, device=dev)
    blocks = torch.empty(total // bm, dtype=torch.int32, device=dev)
    _place_kernel[(experts,)](flat, ends, padded, rows, blocks, n, BM=bm, BLOCK=1024, num_warps=4)
    return rows, blocks, ends[-1:]


@triton.jit
def _combine_kernel(Y, S, G, OUT, H, K: tl.constexpr, BH: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BH + tl.arange(0, BH)
    m = c < H
    acc = tl.zeros([BH], tl.float32)
    for j in tl.static_range(K):
        acc += tl.load(Y + (t * K + j) * H + c, mask=m, other=0.0).to(tl.float32)
    g = tl.load(G + t).to(tl.float32)
    sh = tl.load(S + t * H + c, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + t * H + c, (acc + sh / (1.0 + tl.exp(-g))).to(OUT.dtype.element_ty), mask=m)


def combine(y: torch.Tensor, shared: torch.Tensor, gate: torch.Tensor, k: int) -> torch.Tensor:
    """sum over each token's k expert rows of y [T * k, H] plus the shared expert's output [T, H] scaled by the
    sigmoid of its gate logit [T, 1], summed in float32 and rounded once."""
    T, H = shared.shape
    out = torch.empty_like(shared)
    if T:
        _combine_kernel[(T, triton.cdiv(H, 1024))](y, shared.contiguous(), gate.contiguous(), out, H, K=k, BH=1024,
                                                    num_warps=4)
    return out


def _tile(t: tuple | None) -> tuple[int, int, int, int, int]:
    """(block rows, block columns, warps, stages, row blocks per group), the group 0 when a tile names none."""
    t = tuple(t or (BM, BN, 4, 3))
    return t + (0,) * (5 - len(t))


def grouped(a: torch.Tensor, w: torch.Tensor, out: torch.Tensor, rows, blocks, used, topk: int, by_token: bool,
            weights: torch.Tensor | None = None, tile: tuple | None = None) -> torch.Tensor:
    """out[pair] = a[token of pair or pair] @ w[expert of pair].T, times the pair's weight when given. `tile` (block
    rows as `align` padded them, block columns, warps, stages, and optionally the row blocks run as a group through
    every column tile); the k step is fixed, so a row's sums are too."""
    bm, bn, warps, stages, gm = _tile(tile)
    n_pairs = out.shape[0]
    N, KD = w.shape[1], w.shape[2]
    nbm = blocks.shape[0]
    grid = (nbm * triton.cdiv(N, bn),) if gm else (nbm, triton.cdiv(N, bn))
    _grouped_kernel[grid](a, w, out, weights if weights is not None else out, rows, blocks, used, n_pairs, N, KD, topk,
                          a.stride(0), w.stride(0), w.stride(1), out.stride(0), nbm, BY_TOKEN=by_token,
                          MUL_W=weights is not None, BM=bm, BN=bn, BK=BK, GM=gm, num_warps=warps, num_stages=stages)
    return out


def grouped_fp8(a: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor, out: torch.Tensor, rows, blocks, used,
                topk: int, by_token: bool, weights: torch.Tensor | None = None,
                tile: tuple | None = None) -> torch.Tensor:
    """`grouped` with float8 experts w [E, N, K] (scales [E, N]) and the rows going in scaled to float8 per row."""
    from .fp8 import quantize_tokens

    bm, bn, warps, stages, gm = _tile(tile)
    aq, sa = quantize_tokens(a if a.stride(-1) == 1 else a.contiguous())
    n_pairs = out.shape[0]
    N, KD = w.shape[1], w.shape[2]
    nbm = blocks.shape[0]
    grid = (nbm * triton.cdiv(N, bn),) if gm else (nbm, triton.cdiv(N, bn))
    _grouped_fp8_kernel[grid](aq, sa, w, w_scale, out, weights if weights is not None else out, rows, blocks, used,
                              n_pairs, N, KD, topk, aq.stride(0), w.stride(0), w.stride(1), w_scale.stride(0),
                              out.stride(0), nbm, BY_TOKEN=by_token, MUL_W=weights is not None, BM=bm, BN=bn, BK=BK8,
                              GM=gm, num_warps=warps, num_stages=stages)
    return out


@torch.no_grad()
def quantize_experts(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Experts [E, N, K] -> float8 [E, N, K] and float32 scales [E, N] with w ~ q * scale (amax / 448 per row), one
    expert at a time."""
    from .fp8 import quantize_rows

    q = torch.empty(w.shape, device=w.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(w.shape[:2], device=w.device, dtype=torch.float32)
    for e in range(w.shape[0]):
        qe, se = quantize_rows(w[e])
        q[e].copy_(qe)
        s[e].copy_(se[:, 0])
    return q, s


def experts(x: torch.Tensor, w: dict, top: torch.Tensor, ids: torch.Tensor, act) -> torch.Tensor:
    """The routed experts' rows for x [T, hidden] on the routing (top, ids) [T, k]: [T * k, hidden], each the expert's
    output times its weight, in float32 for 16-bit experts and in the activation dtype for float8 ones (kernels/fp8.py
    rounds far more). `act(gu, inter)` is the activation of the gate and up halves."""
    T, H = x.shape
    K, I = top.shape[1], w["inter"]
    fp8 = "experts_scale" in w
    t = tile(T * K, fp8, x.device)
    rows, blocks, used = align(ids, w["experts"].shape[0], t[0])
    if fp8:
        gu = grouped_fp8(x, w["experts"], w["experts_scale"], x.new_empty(T * K, 2 * I), rows, blocks, used, K, True,
                         tile=t)
        return grouped_fp8(act(gu, I), w["experts_down"], w["experts_down_scale"], x.new_empty(T * K, H), rows, blocks,
                           used, K, False, top.reshape(-1), tile=t)
    gu = grouped(x, w["experts"], x.new_empty(T * K, 2 * I), rows, blocks, used, K, True, tile=t)
    return grouped(act(gu, I), w["experts_down"], x.new_empty(T * K, H, dtype=torch.float32), rows, blocks, used, K,
                   False, top.reshape(-1), tile=t)


def moe(x: torch.Tensor, w: dict, mm, swiglu) -> torch.Tensor:
    """The feed-forward of a Qwen3.5 MoE layer for rows x [T, hidden]: the routed experts' weighted sum (the top k of
    the router's softmax, renormalized) plus the shared expert scaled by its sigmoid gate. `mm` and `swiglu` are the
    backbone's own products and activation."""
    K = w["topk"]
    wr, wg = w["router"], w["shared_gate"]
    if isinstance(wr, torch.Tensor) and isinstance(wg, torch.Tensor):
        logits, gate = router(x, wr, wg)
    else:
        logits, gate = mm(x, wr), mm(x, wg)
    top, ids = route(logits, K)
    y = experts(x, w, top, ids, swiglu)
    shared = mm(swiglu(mm(x, w["shared_gate_up"]), w["shared_inter"]), w["shared_down"])
    return combine(y, shared, gate, K)


@triton.jit
def _combine_norm_kernel(Y, D, W1, W2, OUT, H, sd, so, eps, K: tl.constexpr, BH: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BH)
    m = c < H
    acc = tl.zeros([BH], tl.float32)
    for j in tl.static_range(K):
        acc += tl.load(Y + (t * K + j) * H + c, mask=m, other=0.0).to(tl.float32)
    d = tl.load(D + t * sd + c, mask=m, other=0.0).to(tl.float32)
    a = d * tl.math.rsqrt(tl.sum(d * d, 0) / H + eps) * tl.load(W1 + c, mask=m, other=0.0).to(tl.float32)
    b = acc * tl.math.rsqrt(tl.sum(acc * acc, 0) / H + eps) * tl.load(W2 + c, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + t * so + c, (a + b).to(OUT.dtype.element_ty), mask=m)


def combine_norm(y: torch.Tensor, dense: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, k: int,
                 eps: float) -> torch.Tensor:
    """Gemma 4's two feed-forward branches joined: RMSNorm(dense) * w1 + RMSNorm(sum over each token's k expert rows
    of y [T * k, H]) * w2, in float32 and rounded once to dense's dtype."""
    T, H = dense.shape
    out = torch.empty_like(dense)
    if T:
        _combine_norm_kernel[(T,)](y, dense, w1, w2, out, H, dense.stride(0), out.stride(0), eps, K=k,
                                   BH=triton.next_power_of_2(H), num_warps=8 if H > 2048 else 4)
    return out

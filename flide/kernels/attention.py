"""Attention for packed decision streams, in Triton.

Every query token carries (job, branch, position). A key is visible to a query when they belong to the
same job, the key is in the job's prefix (branch -1) or in the query's own branch, and the key's
position is not after the query's (with a sliding window, also less than `window` positions before it).
Each token of padding is its own job. A fourth column (block), when given, also shows a key after the query when
both carry the same block number >= 0: the tokens of one image see each other (Gemma 4's sliding layers).

Keys come from two sources, so one kernel serves both layouts:
  packed stream: both sources are the stream itself; source 1 covers the job's prefix, source 2 the
      query's own branch, so the prefixes of other jobs in between are never visited;
  branch graph: source 1 is a prefix key/value cache, source 2 the branch rows.
Each block of queries visits the key range [lo, hi) of each source, the union over its queries; the
mask above decides the rest element by element. When both sources are the same tensor, the two unions must not
overlap: a key inside both is read twice by a query that sees it (the prefix range of a packed stream comes first,
branch and padding ranges after it). All lengths and ranges are read from device tensors,
so the kernel runs inside CUDA graphs with a fixed grid.

A query that sees no key (padding outside every range) gets zeros.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .. import settings

_NEG = tl.constexpr(-(2 ** 30))


@triton.jit
def _attend_full(acc, m_i, l_i, q, K, V, lo, hi, hk, sk_t, sk_h, sv_t, sv_h, scale_log2, offs_d, offs_v,
                 D: tl.constexpr, BLOCK_N: tl.constexpr, PAD_D: tl.constexpr):
    """Key blocks every query of the tile sees whole: no metadata, no mask (the same arithmetic as the masked loop
    on visible keys)."""
    for start in range(lo, hi, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        if PAD_D:
            k = tl.load(K + offs_n[:, None] * sk_t + hk * sk_h + offs_d[None, :], mask=offs_d[None, :] < D, other=0.0)
        else:
            k = tl.load(K + offs_n[:, None] * sk_t + hk * sk_h + offs_d[None, :])
        s = tl.dot(q, tl.trans(k)) * scale_log2
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        if PAD_D:
            v = tl.load(V + offs_n[:, None] * sv_t + hk * sv_h + offs_v[None, :], mask=offs_v[None, :] < D, other=0.0)
        else:
            v = tl.load(V + offs_n[:, None] * sv_t + hk * sv_h + offs_v[None, :])
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    return acc, m_i, l_i


@triton.jit
def _attend_source(acc, m_i, l_i, q, qj, qb, qp, qx, qlo, qhi, K, V, KJ, KB, KP, KX, lo, hi, hk, sk_t, sk_h, sv_t,
                   sv_h, scale_log2, window, offs_d, offs_v, D: tl.constexpr, BLOCK_N: tl.constexpr,
                   WINDOW: tl.constexpr, BLOCKS: tl.constexpr, PAD_D: tl.constexpr, EXACT: tl.constexpr):
    """Key blocks seen in part. EXACT: each query's own range [qlo, qhi) holds exactly the keys it sees (block_ranges
    with exact=True), so the range is the mask and no key metadata is read."""
    for start in range(lo, hi, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mk = offs_n < hi
        if PAD_D:
            mkd = mk[:, None] & (offs_d[None, :] < D)
        else:
            mkd = mk[:, None]
        k = tl.load(K + offs_n[:, None] * sk_t + hk * sk_h + offs_d[None, :], mask=mkd, other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale_log2
        if EXACT:
            vis = (offs_n[None, :] >= qlo[:, None]) & (offs_n[None, :] < qhi[:, None])
        else:
            kj = tl.load(KJ + offs_n, mask=mk, other=_NEG)
            kb = tl.load(KB + offs_n, mask=mk, other=-3)
            kp = tl.load(KP + offs_n, mask=mk, other=0)
            vis = (kj[None, :] == qj[:, None]) & ((kb[None, :] == -1) | (kb[None, :] == qb[:, None]))
            seen = kp[None, :] <= qp[:, None]
            if BLOCKS:
                kx = tl.load(KX + offs_n, mask=mk, other=-1)
                seen = seen | ((kx[None, :] == qx[:, None]) & (qx[:, None] >= 0))
            vis = vis & seen & mk[None, :]
            if WINDOW:
                vis = vis & ((qp[:, None] - kp[None, :]) < window)
        m_new = tl.maximum(m_i, tl.max(tl.where(vis, s, float("-inf")), 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.where(vis, tl.exp2(s - m_safe[:, None]), 0.0)
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        if PAD_D:
            mkv = mk[:, None] & (offs_v[None, :] < D)
        else:
            mkv = mk[:, None]
        v = tl.load(V + offs_n[:, None] * sv_t + hk * sv_h + offs_v[None, :], mask=mkv, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    return acc, m_i, l_i


@triton.jit
def _attention_fwd(Q, K1, V1, K2, V2, O, QJ, QB, QP, QX, K1J, K1B, K1P, K1X, K2J, K2B, K2P, K2X, R, QR, sqr_t, sqr_c,
                   scale_log2, Tq, window,
                   sq_t, sq_h, sk1_t, sk1_h, sv1_t, sv1_h, sk2_t, sk2_h, sv2_t, sv2_h, so_t, so_h,
                   G: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   WINDOW: tl.constexpr, BLOCKS: tl.constexpr, EXACT: tl.constexpr, SPLIT: tl.constexpr):
    h = tl.program_id(0) // SPLIT
    part = tl.program_id(0) % SPLIT
    pid = tl.num_programs(1) - 1 - tl.program_id(1)
    hk = h // G
    row_lo = tl.load(R + pid * 10)
    row_hi = tl.load(R + pid * 10 + 1)
    if row_hi <= row_lo:
        return
    offs_m = row_lo + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, DP)
    offs_v = part * (DP // SPLIT) + tl.arange(0, DP // SPLIT)
    mq = offs_m < row_hi
    if DP != D:
        mqd = mq[:, None] & (offs_d[None, :] < D)
    else:
        mqd = mq[:, None]
    q = tl.load(Q + offs_m[:, None] * sq_t + h * sq_h + offs_d[None, :], mask=mqd, other=0.0)
    qj = tl.load(QJ + offs_m, mask=mq, other=_NEG + 1)
    qb = tl.load(QB + offs_m, mask=mq, other=-4)
    qp = tl.load(QP + offs_m, mask=mq, other=0)
    if BLOCKS:
        qx = tl.load(QX + offs_m, mask=mq, other=-1)
    else:
        qx = qp
    if EXACT:
        qlo1 = tl.load(QR + offs_m * sqr_t, mask=mq, other=0)
        qhi1 = tl.load(QR + offs_m * sqr_t + sqr_c, mask=mq, other=0)
        qlo2 = tl.load(QR + offs_m * sqr_t + 2 * sqr_c, mask=mq, other=0)
        qhi2 = tl.load(QR + offs_m * sqr_t + 3 * sqr_c, mask=mq, other=0)
    else:
        qlo1, qhi1, qlo2, qhi2 = qp, qp, qp, qp
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, DP // SPLIT], tl.float32)
    lo1 = tl.load(R + pid * 10 + 2)
    f01 = tl.load(R + pid * 10 + 3)
    f11 = tl.load(R + pid * 10 + 4)
    hi1 = tl.load(R + pid * 10 + 5)
    lo2 = tl.load(R + pid * 10 + 6)
    f02 = tl.load(R + pid * 10 + 7)
    f12 = tl.load(R + pid * 10 + 8)
    hi2 = tl.load(R + pid * 10 + 9)
    acc, m_i, l_i = _attend_source(acc, m_i, l_i, q, qj, qb, qp, qx, qlo1, qhi1, K1, V1, K1J, K1B, K1P, K1X, lo1, f01,
                                   hk, sk1_t, sk1_h, sv1_t, sv1_h, scale_log2, window, offs_d, offs_v, D, BLOCK_N,
                                   WINDOW, BLOCKS, DP != D, EXACT)
    acc, m_i, l_i = _attend_full(acc, m_i, l_i, q, K1, V1, f01, f11, hk, sk1_t, sk1_h, sv1_t, sv1_h, scale_log2, offs_d,
                                 offs_v, D, BLOCK_N, DP != D)
    acc, m_i, l_i = _attend_source(acc, m_i, l_i, q, qj, qb, qp, qx, qlo1, qhi1, K1, V1, K1J, K1B, K1P, K1X, f11, hi1,
                                   hk, sk1_t, sk1_h, sv1_t, sv1_h, scale_log2, window, offs_d, offs_v, D, BLOCK_N,
                                   WINDOW, BLOCKS, DP != D, EXACT)
    acc, m_i, l_i = _attend_source(acc, m_i, l_i, q, qj, qb, qp, qx, qlo2, qhi2, K2, V2, K2J, K2B, K2P, K2X, lo2, f02,
                                   hk, sk2_t, sk2_h, sv2_t, sv2_h, scale_log2, window, offs_d, offs_v, D, BLOCK_N,
                                   WINDOW, BLOCKS, DP != D, EXACT)
    acc, m_i, l_i = _attend_full(acc, m_i, l_i, q, K2, V2, f02, f12, hk, sk2_t, sk2_h, sv2_t, sv2_h, scale_log2, offs_d,
                                 offs_v, D, BLOCK_N, DP != D)
    acc, m_i, l_i = _attend_source(acc, m_i, l_i, q, qj, qb, qp, qx, qlo2, qhi2, K2, V2, K2J, K2B, K2P, K2X, f12, hi2,
                                   hk, sk2_t, sk2_h, sv2_t, sv2_h, scale_log2, window, offs_d, offs_v, D, BLOCK_N,
                                   WINDOW, BLOCKS, DP != D, EXACT)
    out = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
    if DP != D:
        mqv = mq[:, None] & (offs_v[None, :] < D)
    else:
        mqv = mq[:, None]
    tl.store(O + offs_m[:, None] * so_t + h * so_h + offs_v[None, :], out.to(O.dtype.element_ty), mask=mqv)


def config(device, dtype, head_dim: int) -> dict:
    """Tile sizes that fit the part's shared memory: about 99 KB per block on consumer and workstation cards,
    227 KB on H100 / H200. On the large parts, 128-query tiles (H200, bfloat16: 349 TFLOP/s causal at 256-dim heads
    against 182 with 64-query tiles; 182 against 59 at 64-dim heads, where 4 warps beat 8)."""
    smem = getattr(torch.cuda.get_device_properties(device), "shared_memory_per_block_optin", 0)
    big = smem >= 160 * 1024
    dp = triton.next_power_of_2(head_dim)
    if dp > 256:
        if big and dtype != torch.float32:
            return {"BLOCK_M": 128, "BLOCK_N": 32, "num_warps": 8, "num_stages": 1, "SPLIT": 2}
        return {"BLOCK_M": 16, "BLOCK_N": 32, "num_warps": 8, "num_stages": 1}
    if dtype == torch.float32:
        return {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 4, "num_stages": 1}
    if not big:
        if dp > 128:
            return {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 8, "num_stages": 1}
        return {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 8, "num_stages": 2}
    if dp > 64:
        return {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 8, "num_stages": 2}
    return {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2}


@triton.jit
def _max2(a, b):
    return tl.maximum(a, b)


@triton.jit
def _group(lo1, hi1, lo2, hi2, BRANCHES: tl.constexpr):
    if BRANCHES:
        return tl.where(lo1 < hi1, lo1.to(tl.int64) * 4294967296 + tl.where(lo2 < hi2, lo2.to(tl.int64) + 1, 0), -1)
    return tl.where(lo1 < hi1, 2 * lo1.to(tl.int64) + (lo2 < hi2).to(tl.int64), -1)


@triton.jit
def _tile_rows(LO1, HI1, LO2, HI2, sr, ROW_LO, ROW_HI, T, n_tiles, BM: tl.constexpr, GROUPS: tl.constexpr,
               BRANCHES: tl.constexpr, BLOCK: tl.constexpr):
    """One program scanning the T rows: tiles of at most BM rows, new at every BM rows (GROUPS: counted from the start
    of the row's run of one group, and new at every run). Writes each tile's [row_lo, row_hi); unused tiles 0, 0."""
    count = 0
    first = 0
    for b in range(0, tl.cdiv(T, BLOCK)):
        rows = b * BLOCK + tl.arange(0, BLOCK)
        m = rows < T
        if GROUPS:
            lo1 = tl.load(LO1 + rows * sr, mask=m, other=0)
            hi1 = tl.load(HI1 + rows * sr, mask=m, other=0)
            lo2 = tl.load(LO2 + rows * sr, mask=m, other=0)
            hi2 = tl.load(HI2 + rows * sr, mask=m, other=0)
            g = _group(lo1, hi1, lo2, hi2, BRANCHES)
            prev = tl.load(LO1 + (rows - 1) * sr, mask=m & (rows > 0), other=0)
            prev_h1 = tl.load(HI1 + (rows - 1) * sr, mask=m & (rows > 0), other=0)
            prev_l2 = tl.load(LO2 + (rows - 1) * sr, mask=m & (rows > 0), other=0)
            prev_h2 = tl.load(HI2 + (rows - 1) * sr, mask=m & (rows > 0), other=0)
            gp = _group(prev, prev_h1, prev_l2, prev_h2, BRANCHES)
            new = (g != gp) | (rows == 0)
            start = tl.maximum(tl.associative_scan(tl.where(new & m, rows, -1), 0, _max2), first)
            first = tl.max(tl.where(m, start, 0), 0)
            begin = (new | ((rows - start) % BM == 0)) & m
        else:
            begin = (rows % BM == 0) & m
        idx = count + tl.cumsum(begin.to(tl.int32), 0) - 1
        tl.store(ROW_LO + idx, rows, mask=begin & (idx < n_tiles))
        tl.store(ROW_HI + idx - 1, rows, mask=begin & (idx > 0) & (idx <= n_tiles))
        count += tl.sum(begin.to(tl.int32), 0)
    tl.store(ROW_HI + count - 1, T, mask=(count > 0) & (count <= n_tiles))
    for b in range(0, tl.cdiv(n_tiles, BLOCK)):
        t = b * BLOCK + tl.arange(0, BLOCK)
        rest = (t >= count) & (t < n_tiles)
        tl.store(ROW_LO + t, 0, mask=rest)
        tl.store(ROW_HI + t, 0, mask=rest)


@triton.jit
def _tile_source(lo, hi, m, BLOCK_N: tl.constexpr, EXACT: tl.constexpr):
    big = 2147483647
    empty = (lo >= hi) | ~m
    ulo = tl.min(tl.where(empty, big, lo), 0)
    uhi = tl.max(tl.where(empty, 0, hi), 0)
    has = ulo < uhi
    ulo = tl.where(has, ulo, 0)
    uhi = tl.where(has, uhi, 0)
    f0 = uhi
    f1 = uhi
    if EXACT:
        any_empty = tl.max(((lo >= hi) & m).to(tl.int32), 0) > 0
        ilo = tl.max(tl.where(m, lo, -big), 0)
        ihi = tl.min(tl.where(m, hi, big), 0)
        a0 = ulo + tl.maximum(ilo - ulo + BLOCK_N - 1, 0) // BLOCK_N * BLOCK_N
        a1 = a0 + tl.maximum(ihi - a0, 0) // BLOCK_N * BLOCK_N
        ok = has & ~any_empty & (a1 > a0)
        f0 = tl.where(ok, a0, uhi)
        f1 = tl.where(ok, a1, uhi)
    return ulo, f0, f1, uhi


@triton.jit
def _tile_ranges(LO1, HI1, LO2, HI2, sr, ROW_LO, ROW_HI, R, BM: tl.constexpr, BLOCK_N: tl.constexpr,
                 EXACT: tl.constexpr):
    """One program per tile: its rows, then per source the union of its rows' key ranges and the run of whole key
    blocks inside every row's range (see block_ranges)."""
    t = tl.program_id(0)
    r_lo = tl.load(ROW_LO + t)
    r_hi = tl.load(ROW_HI + t)
    rows = r_lo + tl.arange(0, BM)
    m = rows < r_hi
    lo1 = tl.load(LO1 + rows * sr, mask=m, other=0)
    hi1 = tl.load(HI1 + rows * sr, mask=m, other=0)
    lo2 = tl.load(LO2 + rows * sr, mask=m, other=0)
    hi2 = tl.load(HI2 + rows * sr, mask=m, other=0)
    a, b, c, d = _tile_source(lo1, hi1, m, BLOCK_N, EXACT)
    e, f, g, h = _tile_source(lo2, hi2, m, BLOCK_N, EXACT)
    out = R + t * 10
    tl.store(out, r_lo)
    tl.store(out + 1, r_hi)
    tl.store(out + 2, a)
    tl.store(out + 3, b)
    tl.store(out + 4, c)
    tl.store(out + 5, d)
    tl.store(out + 6, e)
    tl.store(out + 7, f)
    tl.store(out + 8, g)
    tl.store(out + 9, h)


def block_ranges(lo1, hi1, lo2, hi2, block_m: int, block_n: int = 0, exact: bool = False, groups: int = 0,
                 tiles_from=None) -> torch.Tensor:
    """[n_tiles, 10] int32: per tile of at most `block_m` consecutive queries, its rows [row_lo, row_hi), then per source
    lo, f0, f1, hi: [lo, hi) the union of the per-query key ranges, [f0, f1) the run of whole `block_n`-key blocks
    (counted from lo) inside every query's own range, which the kernel reads without masking. Queries with an empty
    range must carry lo >= hi (any values); they do not widen the union and leave no unmasked run. `exact`: the caller's
    ranges hold only keys their query sees (job, branch, position and window agree everywhere inside them); otherwise
    every block is masked.

    Tiles are every `block_m` rows, or with `groups` > 0 runs of at most `block_m` rows of one group (queries with the
    same source-1 range start and the same kind: one job's prefix, one job's branches, padding), so that no tile spans
    two requests and scans the keys of both; with the batch-invariant products one group per branch, so a query's key
    blocks start where its own branch starts. `groups` bounds the number of runs of consecutive rows of one group; there
    are cdiv(T, block_m) + `groups` tiles, those past the real ones empty. Two kernels (a scan for the tiles' rows, a
    program per tile for its ranges), so the table costs two launches in a graph; `block_ranges_ref` is the same in
    torch operations. The four inputs are int32 vectors sharing one stride (rows of a [4, T] or columns of a [T, 4]).
    `tiles_from`: four such vectors whose groups decide the tiles in place of the ranges' own (a sliding window
    moves each query's range start, so its tiles follow the unwindowed ranges of the same queries)."""
    T = lo1.shape[0]
    n = triton.cdiv(T, block_m) + (groups if groups else 0)
    sr = lo1.stride(0)
    g = tiles_from or (lo1, hi1, lo2, hi2)
    sg = g[0].stride(0)
    if (any(x.stride(0) != sr or x.dtype != torch.int32 for x in (lo1, hi1, lo2, hi2))
            or any(x.stride(0) != sg or x.dtype != torch.int32 for x in g)):
        return block_ranges_ref(lo1, hi1, lo2, hi2, block_m, block_n, exact, groups, tiles_from)
    rows = torch.empty(2, n, dtype=torch.int32, device=lo1.device)
    R = torch.empty(n, 10, dtype=torch.int32, device=lo1.device)
    _tile_rows[(1,)](*g, sg, rows[0], rows[1], T, n, BM=block_m, GROUPS=bool(groups), BRANCHES=settings.batch_invariant,
                     BLOCK=1024, num_warps=8)
    _tile_ranges[(n,)](lo1, hi1, lo2, hi2, sr, rows[0], rows[1], R, BM=block_m, BLOCK_N=max(block_n, 1),
                       EXACT=bool(exact and block_n), num_warps=4)
    return R


def block_ranges_ref(lo1, hi1, lo2, hi2, block_m: int, block_n: int = 0, exact: bool = False,
                     groups: int = 0, tiles_from=None) -> torch.Tensor:
    """`block_ranges` in torch operations (the reference its kernels are tested against)."""
    T = lo1.shape[0]
    dev = lo1.device
    big = torch.iinfo(torch.int32).max
    ar = torch.arange(T, device=dev)
    if groups:
        g = _groups_ref(*(tiles_from or (lo1, hi1, lo2, hi2)))
        new = torch.ones(T, dtype=torch.bool, device=dev)
        new[1:] = g[1:] != g[:-1]
        first = torch.cummax(torch.where(new, ar, 0), 0).values
        tid = torch.cumsum((new | ((ar - first) % block_m == 0)).long(), 0) - 1
        n = triton.cdiv(T, block_m) + groups
    else:
        tid = ar // block_m
        n = triton.cdiv(T, block_m)

    def red(x, how, fill):
        return torch.full((n,), fill, dtype=x.dtype, device=dev).scatter_reduce(0, tid, x, how, include_self=True)

    row_hi = red(ar + 1, "amax", 0)
    row_lo = torch.where(row_hi > 0, red(ar, "amin", T), 0)

    def tile(lo, hi):
        lo, hi = lo.int(), hi.int()
        empty = lo >= hi
        ulo = red(torch.where(empty, big, lo), "amin", big)
        uhi = red(torch.where(empty, 0, hi), "amax", 0)
        has = ulo < uhi
        ulo, uhi = torch.where(has, ulo, 0), torch.where(has, uhi, 0)
        if not exact or not block_n:
            return ulo, uhi, uhi, uhi
        any_empty = red(empty.int(), "amax", 0) > 0
        ilo = red(lo, "amax", -big)
        ihi = red(hi, "amin", big)
        f0 = ulo + (ilo - ulo + block_n - 1).clamp(min=0) // block_n * block_n
        f1 = f0 + (ihi - f0).clamp(min=0) // block_n * block_n
        ok = has & ~any_empty & (f1 > f0)
        return ulo, torch.where(ok, f0, uhi), torch.where(ok, f1, uhi), uhi

    return torch.stack([row_lo.int(), row_hi.int(), *tile(lo1, hi1), *tile(lo2, hi2)], 1).to(torch.int32).contiguous()


def _groups_ref(lo1, hi1, lo2, hi2):
    if settings.batch_invariant:
        return torch.where(lo1 < hi1, lo1.long() * 2 ** 32 + torch.where(lo2 < hi2, lo2.long() + 1, 0), -1)
    return torch.where(lo1 < hi1, 2 * lo1.long() + (lo2 < hi2).long(), -1)


def runs(ranges, by_row: bool = False) -> int:
    """Runs of consecutive rows of one group (see `block_ranges`) in host key ranges: [4, T], or [T, 4] with `by_row`."""
    import numpy as np

    r = np.asarray(ranges)
    if by_row:
        r = r.T
    if settings.batch_invariant:
        g = np.where(r[0] < r[1], r[0].astype(np.int64) * 2 ** 32 + np.where(r[2] < r[3], r[2].astype(np.int64) + 1, 0), -1)
    else:
        g = np.where(r[0] < r[1], 2 * r[0].astype(np.int64) + (r[2] < r[3]), -1)
    return 1 + int(np.count_nonzero(g[1:] != g[:-1])) if g.size else 0


def attention(q, k1, v1, k2, v2, q_meta, k1_meta, k2_meta, ranges, scale: float, window: int = 0,
              cfg: dict | None = None, out=None, qranges=None):
    """q [Tq, H, D]; k*, v* [T*, Hkv, D] (last dim contiguous); *_meta = (job, branch, position[, block]) int32 [T*]
    (blocks on all three or none); ranges from `block_ranges` with cfg["BLOCK_M"] (one program per tile and head).
    `qranges`: the per-query ranges `ranges` came from with exact=True, int32 [4, Tq] (rows lo1, hi1, lo2, hi2, any
    strides); they then mask the partly seen key blocks, and the key metadata is not read.
    Returns [Tq, H, D] in q's dtype."""
    Tq, H, D = q.shape
    G = H // k2.shape[1]
    cfg = cfg or config(q.device, q.dtype, D)
    if out is None:
        out = torch.empty(Tq, H, D, device=q.device, dtype=q.dtype)
    split = cfg.get("SPLIT", 1)
    grid = (H * split, ranges.shape[0])
    blocks = len(q_meta) == 4
    if blocks != (len(k1_meta) == 4) or blocks != (len(k2_meta) == 4):
        raise ValueError("block numbers on every metadata or on none")
    x = lambda m: m[3] if blocks else m[0]
    if qranges is not None:
        if tuple(qranges.shape) != (4, Tq):
            raise ValueError(f"qranges {tuple(qranges.shape)}, expected (4, {Tq})")
        sqr_t, sqr_c = qranges.stride(1), qranges.stride(0)
    else:
        sqr_t = sqr_c = 0
    _attention_fwd[grid](
        q, k1, v1, k2, v2, out, *q_meta[:3], x(q_meta), *k1_meta[:3], x(k1_meta), *k2_meta[:3], x(k2_meta), ranges,
        qranges if qranges is not None else ranges, sqr_t, sqr_c, scale * 1.4426950408889634, Tq, window,
        q.stride(0), q.stride(1), k1.stride(0), k1.stride(1), v1.stride(0), v1.stride(1),
        k2.stride(0), k2.stride(1), v2.stride(0), v2.stride(1), out.stride(0), out.stride(1),
        G=G, D=D, DP=triton.next_power_of_2(D), BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], WINDOW=window > 0,
        BLOCKS=blocks, EXACT=qranges is not None, SPLIT=split,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return out

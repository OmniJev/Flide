"""The packed attention kernel against a dense masked reference in float64.

Covers the packed-stream layout (both sources are the stream: prefix range plus own-branch range), the
branch-graph layout (source 1 is a prefix cache with left padding, source 2 the right-padded branch
rows), grouped heads, head dims 128 / 256, a sliding window, image blocks whose tokens see each other, and
float32 / bfloat16.

    TRITON_F32_DEFAULT=ieee python tests/test_attention.py
"""
from __future__ import annotations

import os

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.kernels.attention import attention, block_ranges, block_ranges_ref, config, runs


def dense(q, k, v, qm, km, scale, window=0):
    """q [Tq,H,D], k/v [Tk,Hkv,D]; masks from (job, branch, pos[, block])."""
    H, Hk = q.shape[1], k.shape[1]
    k = k.repeat_interleave(H // Hk, 1)
    v = v.repeat_interleave(H // Hk, 1)
    qj, qb, qp = (x[:, None].long() for x in qm[:3])
    kj, kb, kp = (x[None, :].long() for x in km[:3])
    seen = kp <= qp
    if len(qm) == 4:
        qx, kx = qm[3][:, None].long(), km[3][None, :].long()
        seen = seen | ((qx == kx) & (qx >= 0))
    vis = (qj == kj) & ((kb == -1) | (kb == qb)) & seen
    if window:
        vis &= (qp - kp) < window
    s = torch.einsum("qhd,khd->hqk", q.double(), k.double()) * scale
    s = s.masked_fill(~vis[None], float("-inf"))
    p = torch.softmax(s, -1)
    p = torch.nan_to_num(p, nan=0.0)
    return torch.einsum("hqk,khd->qhd", p, v.double())


def packed_case(rng, n_jobs, device, blocks=False):
    job, branch, pos, lo1, hi1, lo2, hi2, blk = [], [], [], [], [], [], [], []
    starts = []
    prefixes = [rng.randrange(1, 300) for _ in range(n_jobs)]
    branches = [[rng.randrange(1, 60) for _ in range(rng.randrange(1, 6))] for _ in range(n_jobs)]
    for j, P in enumerate(prefixes):
        s0 = len(job)
        starts.append(s0)
        runs = {}
        if blocks and P > 10:
            a = rng.randrange(0, P - 5)
            e = min(P, a + rng.randrange(2, 40))
            runs = {i: (j, s0 + e) for i in range(a, e)}
        for i in range(P):
            job.append(j); branch.append(-1); pos.append(i)
            b_id, end = runs.get(i, (-1, 0))
            blk.append(b_id)
            lo1.append(s0); hi1.append(max(s0 + i + 1, end)); lo2.append(0); hi2.append(0)
    b = 0
    for j, (P, bl) in enumerate(zip(prefixes, branches)):
        for L in bl:
            s0 = len(job)
            for i in range(L):
                job.append(j); branch.append(b); pos.append(P + i); blk.append(-1)
                lo1.append(starts[j]); hi1.append(starts[j] + P); lo2.append(s0); hi2.append(s0 + i + 1)
            b += 1
    t = lambda x: torch.tensor(x, dtype=torch.int32, device=device)
    meta = (t(job), t(branch), t(pos)) + ((t(blk),) if blocks else ())
    return meta, (t(lo1), t(hi1), t(lo2), t(hi2))


def graph_case(rng, Kmax, B, Sb, device):
    P = rng.randrange(1, Kmax)
    start = Kmax - P
    idx = torch.arange(Kmax, device=device)
    k1 = (torch.where(idx >= start, 0, -10 - idx).int(), torch.full_like(idx, -1).int(), (idx - start).int())
    lens = [rng.randrange(1, Sb + 1) for _ in range(B)]
    job, branch, pos, lo1, hi1, lo2, hi2 = [], [], [], [], [], [], []
    for b, L in enumerate(lens):
        for i in range(Sb):
            r = b * Sb + i
            real = i < L
            job.append(0 if real else -100000 - r); branch.append(b if real else -2); pos.append(P + i if real else 0)
            lo1.append(start if real else 0); hi1.append(Kmax if real else 0)
            lo2.append(b * Sb if real else r); hi2.append(r + 1)
    t = lambda x: torch.tensor(x, dtype=torch.int32, device=device)
    q = (t(job), t(branch), t(pos))
    return k1, q, (t(lo1), t(hi1), t(lo2), t(hi2)), lens


def tight(rq, qpos, window):
    """The ranges as callers give them for sliding windows: each starts no more than `window` positions back."""
    lo1, hi1, lo2, hi2 = rq
    if not window:
        return rq
    idx = torch.arange(lo1.shape[0], device=lo1.device, dtype=lo1.dtype)
    lo1 = torch.maximum(lo1, torch.minimum(lo1 + qpos - (window - 1), hi1))
    lo2 = torch.maximum(lo2, torch.minimum(idx - (window - 1), hi2))
    return lo1, hi1, lo2, hi2


def both(q, k1, v1, k2, v2, qm, k1m, k2m, rq, scale, window, cfg):
    """The kernel with every block masked, with the unmasked runs (exact ranges), and with the partly seen blocks
    masked by each query's range instead of the key metadata; the three must agree bit for bit. Also with tiles that
    follow the query groups (no tile spans two jobs), which visits keys in other blocks."""
    rq = tight(rq, qm[2], window)
    plain = attention(q, k1, v1, k2, v2, qm, k1m, k2m, block_ranges(*rq, cfg["BLOCK_M"]), scale, window, cfg)
    fast = attention(q, k1, v1, k2, v2, qm, k1m, k2m, block_ranges(*rq, cfg["BLOCK_M"], cfg["BLOCK_N"], exact=True),
                     scale, window, cfg)
    qr = torch.stack(rq).int()
    by_range = attention(q, k1, v1, k2, v2, qm, k1m, k2m, block_ranges(*rq, cfg["BLOCK_M"], cfg["BLOCK_N"], exact=True),
                         scale, window, cfg, qranges=qr)
    n = runs(qr.cpu().numpy())
    split = attention(q, k1, v1, k2, v2, qm, k1m, k2m,
                      block_ranges(*rq, cfg["BLOCK_M"], cfg["BLOCK_N"], exact=True, groups=n), scale, window, cfg,
                      qranges=qr)
    return fast, bool(torch.equal(plain, fast)) and bool(torch.equal(fast, by_range)), split


def run(dtype, D, H, Hk, window, device, seed=0) -> bool:
    rng = random.Random(seed)
    tol = 2e-5 if dtype == torch.float32 else 2e-2
    ok = True
    cfg = config(device, dtype, D)
    g = torch.Generator(device=device).manual_seed(seed)
    for name, blocks in (("packed", False), ("blocks", True)):
        meta, ranges_q = packed_case(rng, 7, device, blocks=blocks)
        T = meta[0].shape[0]
        q, k, v = (torch.randn(T, h, D, device=device, generator=g).to(dtype) for h in (H, Hk, Hk))
        out, same, split = both(q, k, v, k, v, meta, meta, meta, ranges_q, D ** -0.5, window, cfg)
        ref = dense(q, k, v, meta, meta, D ** -0.5, window)
        err = float((out.double() - ref).abs().max())
        err_s = float((split.double() - ref).abs().max())
        good = err <= tol and same and err_s <= tol
        ok &= good
        print(f"  {name:7} {str(dtype):15} D={D} H={H}/{Hk} window={window:<4} T={T:<5} max err {err:.2e} "
              f"unmasked runs {'same bits' if same else 'DIFFERENT'}, job tiles {err_s:.2e} {'ok' if good else 'FAIL'}")

    Kmax, B, Sb = 512, 5, 48
    k1m, qm, ranges_g, lens = graph_case(rng, Kmax, B, Sb, device)
    q = torch.randn(B * Sb, H, D, device=device, generator=g).to(dtype)
    kc, vc = (torch.randn(Kmax, Hk, D, device=device, generator=g).to(dtype) for _ in range(2))
    kb, vb = (torch.randn(B * Sb, Hk, D, device=device, generator=g).to(dtype) for _ in range(2))
    out, same, split = both(q, kc, vc, kb, vb, qm, k1m, qm, ranges_g, D ** -0.5, window, cfg)
    ref = dense(q, torch.cat([kc, kb]), torch.cat([vc, vb]), qm,
                tuple(torch.cat([a, b]) for a, b in zip(k1m, qm)), D ** -0.5, window)
    real = torch.cat([torch.arange(b * Sb, b * Sb + L) for b, L in enumerate(lens)]).to(device)
    err = float((out.double()[real] - ref[real]).abs().max())
    err_s = float((split.double()[real] - ref[real]).abs().max())
    good = err <= tol and same and err_s <= tol
    ok &= good
    print(f"  graph   {str(dtype):15} D={D} H={H}/{Hk} window={window:<4} T={B * Sb:<5} max err {err:.2e} "
          f"unmasked runs {'same bits' if same else 'DIFFERENT'}, job tiles {err_s:.2e} {'ok' if good else 'FAIL'}")
    return ok


def tables(device) -> bool:
    """block_ranges (two Triton kernels) against block_ranges_ref (torch) on packed and graph layouts, fixed tiles and
    tiles that follow the query groups: equal tables."""
    rng = random.Random(1)
    ok = True
    for case in range(12):
        if case % 3 == 2:
            _, _, rq, _ = graph_case(rng, 512, rng.randrange(1, 9), rng.choice((16, 48, 200)), device)
        else:
            _, rq = packed_case(rng, rng.randrange(1, 20), device, blocks=case % 3 == 1)
        rq = tuple(x.contiguous() for x in rq)
        n = runs(torch.stack(rq).cpu().numpy())
        for bm, bn, exact, groups in ((128, 64, True, n), (64, 32, True, n), (128, 64, True, 0), (64, 64, False, 0),
                                      (128, 64, False, n), (16, 32, True, n + 5)):
            a = block_ranges(*rq, bm, bn, exact=exact, groups=groups)
            b = block_ranges_ref(*rq, bm, bn, exact=exact, groups=groups)
            same = a.shape == b.shape and bool(torch.equal(a, b))
            ok &= same
            if not same:
                print(f"  table case {case} bm {bm} bn {bn} exact {exact} groups {groups}: DIFFERENT")
        cut = (torch.minimum(torch.maximum(rq[0], rq[1] - 37), rq[1]).contiguous(), rq[1], rq[2], rq[3])
        for bm, bn in ((128, 64), (16, 32)):
            a = block_ranges(*cut, bm, bn, exact=True, groups=n, tiles_from=rq)
            b = block_ranges_ref(*cut, bm, bn, exact=True, groups=n, tiles_from=rq)
            same = a.shape == b.shape and bool(torch.equal(a, b))
            ok &= same
            if not same:
                print(f"  table case {case} bm {bm} bn {bn} windowed, tiles from the full ranges: DIFFERENT")
    stacked = torch.stack(rq)
    by_col = stacked.T.contiguous()
    same = bool(torch.equal(block_ranges(*stacked, 128, 64, exact=True, groups=n),
                            block_ranges(*(by_col[:, i] for i in range(4)), 128, 64, exact=True, groups=n)))
    ok &= same
    print(f"  tile tables: kernels {'equal' if ok else 'DIFFERENT'} to the torch reference")
    return ok


def main():
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    ok = tables(device)
    for dtype in (torch.float32, torch.bfloat16):
        for D, H, Hk in ((256, 8, 2), (128, 4, 4), (72, 16, 16), (512, 8, 1)):
            for window in (0, 100):
                ok &= run(dtype, D, H, Hk, window, device)
    print("->", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

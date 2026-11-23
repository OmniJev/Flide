"""The session store and the gemma4 session rounds, on the CPU: rows handed out and given back in pages, evictions
that spare what a forward reads, and the layout of a session round (backbones/gemma4/graphs.session_layout): every
query sees exactly the keys a request alone would see, every key it reads is in the context (from the stream or from
the store), saves write the positions their marks keep, and the reads come in request and branch order.

    python tests/test_gemma4_sessions.py
"""
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flide.backbones.gemma4.graphs import session_layout  # noqa: E402
from flide.runtime.session import PAGE, ArenaRows, Entry, KVArena, SessionCache  # noqa: E402

W = 48   # the sliding window of the made-up model
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def rows_for(m):
    return {"full": m, "sliding": m - max(0, m - W)}


def arena(pages_full=40, pages_sliding=20):
    layers = {0: ("sliding", (2, 4), (2, 4)), 1: ("full", (1, 8), (1, 8)), 2: ("sliding", (2, 4), (2, 4))}
    rb = {"sliding": 2 * (8 + 8) * 2, "full": (8 + 8) * 2}
    n = pages_full * PAGE * rb["full"] + pages_sliding * PAGE * rb["sliding"]
    return KVArena(layers, n, "cpu", torch.bfloat16,
                   {"full": pages_full * PAGE * rb["full"] / n, "sliding": pages_sliding * PAGE * rb["sliding"] / n})


def test_store():
    a = arena(20, 10)
    check("pages per pool", a.pages == {"full": 20, "sliding": 10}, str(a.pages))
    c = SessionCache(0, min_tokens=8, stride=64, margin=4)
    c.attach(a)
    rng = np.random.RandomState(0)
    kept = []
    for k in range(6):
        r = c.reserve({"full": 300, "sliding": 100})
        K = torch.randn(300, 1, 8).bfloat16()
        r.write(1, K, K + 1)
        e = Entry(list(rng.randint(100, 1000, 400)), {300: {}}, r)
        c.insert(e)
        kept.append((e, K))
    e, K = kept[-1]
    check("rows read back what was written", torch.equal(e.kv[1][0], K) and torch.equal(e.kv[1][1], K + 1))
    check("full pool evicts least recent", c.evictions == 2 and len(c.entries) == 4, f"{c.evictions} {len(c.entries)}")
    c.pinned = {id(kept[-1][0])}
    for k in range(3):
        c.insert(Entry(list(rng.randint(100, 1000, 700)), {600: {}}, c.reserve({"full": 600, "sliding": 300})))
    check("pinned entry stays", any(x is kept[-1][0] for x in c.entries.values()))
    c.pinned = set()
    check("too large for the store", c.reserve({"full": 10 ** 6, "sliding": 1}) is None)
    c.clear()
    check("clear gives every page back", {k: len(v) for k, v in a.free_pages.items()} == a.pages)


class _E:   # an entry as session_layout reads it
    def __init__(self, rows):
        self.kv = type("kv", (), {"rows": rows})()


def visible(lay, row, window=0):
    """(job, position) pairs the query at stream row `row` attends: the first range over the context, the second over
    the stream; with `window` only positions fewer than that many back."""
    r = lay["ranges"][row]
    out = set()
    q = lay["pos"][row]
    for c in range(r[0], r[1]):
        j, p = lay["cmeta"][0, c], lay["cmeta"][2, c]
        if not window or q - p < window:
            out.add((int(j), int(p)))
    for s in range(r[2], r[3]):
        p = lay["pos"][s]
        if not window or q - p < window:
            out.add((int(lay["job"][s]), int(p)))
    return out


def test_layout(trials=200):
    rng = random.Random(1)
    a = arena(400, 200)
    for t in range(trials):
        reqs, saves, want_reads = [], [], []
        for j in range(rng.randint(1, 5)):
            P = rng.randint(20, 300)
            pre = [rng.randint(5, 999) for _ in range(P)]
            nb = rng.choice([1, 1, 2, 3])
            sufs = [[rng.randint(5, 999) for _ in range(rng.randint(1, 9))] for _ in range(nb)]
            start = rng.choice([0, rng.randint(1, P - 1)])
            entry = None
            if start:
                rr = a.alloc(rows_for(start))
                entry = _E(rr)
            m = rng.choice([None, rng.randint(start + 1, P - 1) if start + 1 < P - 1 else None])
            saved = ArenaRows(a, a.alloc(rows_for(m))) if m else None
            reqs.append((pre, sufs, entry, start, m))
            saves.append((m, saved) if saved else (None, None))
        tokens = sum(len(p) + (len(s[0]) if len(s) == 1 else 0) - st + (0 if len(s) == 1 else sum(map(len, s)))
                     for p, s, _, st, _ in reqs)
        ctx = sum(len(p) + (len(s[0]) if len(s) == 1 else 0) for p, s, *_ in reqs)
        T, C = tokens + rng.randint(0, 40), ctx + 1 + rng.randint(0, 40)
        lay = session_layout(reqs, saves, T, C)
        ok = True
        filled = {}   # context row -> source, per kind
        cs = 0
        reads = {}
        at_stream = {}
        for j, (pre, sufs, entry, start, m) in enumerate(reqs):
            one = len(sufs) == 1
            n = len(pre) + (len(sufs[0]) if one else 0)
            # rows of this request's sequence in the stream, by position
            rows = [s for s in range(T) if lay["job"][s] == j and lay["branch"][s] == -1]
            at_stream[j] = rows
            if [lay["pos"][s] for s in rows] != list(range(start, n)):
                ok = False
            for s in rows:   # a prefix query sees its sequence up to itself
                q = int(lay["pos"][s])
                if visible(lay, s) != {(j, p) for p in range(q + 1)}:
                    ok = False
            if one:
                reads[j] = [rows[-1]]
            for kind in ("full", "sliding"):
                got = set()
                for c in np.concatenate(lay["idx"][("r_ctx_", kind)] or [np.zeros(0, np.int64)]).tolist():
                    got.add(c)
                filled[(j, kind)] = got
            cs += n
        b = 0
        for j, (pre, sufs, entry, start, m) in enumerate(reqs):
            if len(sufs) == 1:
                continue
            P = len(pre)
            reads[j] = []
            for k, sfx in enumerate(sufs):
                rows = [s for s in range(T) if lay["branch"][s] == b]
                b += 1
                for t_, s in enumerate(rows):
                    want = {(j, p) for p in range(P)} | {(j, P + u) for u in range(t_ + 1)}
                    if visible(lay, s) != want:
                        ok = False
                reads[j].append(rows[-1])
        reads = [x for j in range(len(reqs)) for x in reads[j]]
        check(f"round {t}: every query sees what it sees alone", ok)
        check(f"round {t}: reads in request and branch order", reads == lay["reads"], f"{reads} {lay['reads']}")
        # the context holds every position a query reads: from the stream at or after the resume point, from the
        # store before it (all of a full layer's, a sliding layer's last window)
        cover = True
        cs = 0
        for j, (pre, sufs, entry, start, m) in enumerate(reqs):
            n = len(pre) + (len(sufs[0]) if len(sufs) == 1 else 0)
            from_stream = {int(lay["s_ctx"][s]) for s in at_stream[j]}
            if from_stream != set(range(cs + start, cs + n)):
                cover = False
            for kind in ("full", "sliding"):
                lo = 0 if kind == "full" else max(0, start - W)
                want = set(range(cs + lo, cs + start))
                got = set()
                for c, st in zip(lay["idx"][("r_ctx_", kind)], lay["idx"][("r_store_", kind)]):
                    got |= {int(x) for x in c if cs <= x < cs + n}
                if got != want:
                    cover = False
            cs += n
        check(f"round {t}: context filled where it is read", cover)
        sv = True
        cs = 0
        k = {"full": 0, "sliding": 0}
        for j, ((pre, sufs, entry, start, m), (mm, saved)) in enumerate(zip(reqs, saves)):
            n = len(pre) + (len(sufs[0]) if len(sufs) == 1 else 0)
            if saved is not None:
                for kind in ("full", "sliding"):
                    ctx_rows = lay["idx"][("w_ctx_", kind)][k[kind]]
                    lo = 0 if kind == "full" else max(0, m - W)
                    if list(ctx_rows) != list(range(cs + lo, cs + m)) or not np.array_equal(
                            lay["idx"][("w_store_", kind)][k[kind]], saved.rows[kind]):
                        sv = False
                    k[kind] += 1
            cs += n
        check(f"round {t}: saves write their marks' positions", sv)
        pad = all(lay["job"][s] == -1 - s and lay["branch"][s] == -2 for s in range(tokens, T))
        check(f"round {t}: padding sees only itself", pad and all(visible(lay, s) == {(-1 - s, 0)}
                                                                  for s in range(tokens, T)))
        for _, saved in saves:
            if saved is not None:
                saved.free()
        for _, _, entry, _, _ in reqs:
            if entry is not None:
                a.free(entry.kv.rows)


if __name__ == "__main__":
    test_store()
    test_layout()
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")
    sys.exit(1 if FAILS else 0)

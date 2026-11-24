"""Cached-request timing on real text, one scenario at a time, for comparing two checkouts.

    FLIDE_ROOT=/path/to/main .venv/bin/python bench/session_ab.py --scenario siblings --reps 3

Each measurement is a request that hits the cache, and between measurements the same request is run with the cache
cleared, the way the reviewer's harness does it, so a hit is never served twice from the same entry.
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
root = os.environ.get("FLIDE_ROOT")
if root:
    sys.path.insert(0, root)

import torch  # noqa: E402

from flide.api.schema import SystemOneRequest  # noqa: E402
from flide.runtime.engine import Engine  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def real_text(chars):
    """Prose from the repository's own documentation, cut or repeated to `chars` characters."""
    out = []
    for name in ("README.md", "CONTRIBUTING.md", "docs/quickstart.md"):
        path = os.path.join(HERE, name)
        if os.path.exists(path):
            out.append(open(path, encoding="utf-8").read())
    text = "\n".join(out) or "the quick brown fox " * 200
    while len(text) < chars:
        text += "\n" + text
    return text[:chars]


def engine():
    kw = dict(capture=4096, budget=8192, session_min_tokens=256, backbone_options={"cuda_graphs": True})
    gb = os.environ.get("CACHE_GB", "4.0")
    if gb == "auto":                        # the engine's default: budget from free memory, attached after warmup
        kw["session_cache_gb"] = "auto"
    elif gb:
        kw["session_cache_gb"] = float(gb)
    eng = Engine("models/OneJev1.6-0.8B", device="cuda:0", **kw)
    if getattr(eng.sessions, "pstore", None) is None and hasattr(eng.sessions, "attach_prefix_store"):
        eng.sessions.attach_prefix_store(eng.backbone.device)
    return eng


def once(eng, state, question):
    plan = eng.format.plan(SystemOneRequest.model_validate(
        {"state": state, "questions": {"q": {"type": "noul", "instructions": question}}}))
    torch.cuda.synchronize(); t0 = time.perf_counter()
    eng.forward([plan], mode="graphs")
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["siblings", "agent", "repeat"], required=True)
    ap.add_argument("--chars", type=int, default=6000)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    text = real_text(a.chars)
    times = []
    for _ in range(a.reps):
        eng = engine()
        if a.scenario == "siblings":
            once(eng, text, "warm?")
            for k in range(1, 5):
                times.append(once(eng, text + f"\nnote {k}: " + "detail " * 40, f"question {k}?"))
                eng.sessions.clear(); once(eng, text + f"\nnote {k}: " + "detail " * 40, f"question {k}?")
        elif a.scenario == "agent":
            state = text
            once(eng, state, "step 0?")
            for step in range(1, 8):
                state = state + f"\nstep {step}: " + "detail " * 20
                times.append(once(eng, state, f"step {step}?"))
                eng.sessions.clear(); once(eng, state, f"step {step}?")
        else:
            once(eng, text, "warm?")
            for k in range(5):
                times.append(once(eng, text, "same?"))
                eng.sessions.clear(); once(eng, text, "same?")
        del eng
    print("%s cached median %.2f ms (n=%d)" % (a.scenario, statistics.median(times), len(times)))


main()

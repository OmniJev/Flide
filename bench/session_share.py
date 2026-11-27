"""The three session-cache scenarios, timed per request: five siblings sharing a prefix, an agent extending its
prefix a step at a time, and one prefix repeated. `FLIDE_ROOT` selects the checkout, so two revisions can be timed
against the same script.

    FLIDE_ROOT=/path/to/checkout python bench/session_share.py [--chars N] [--runs N]
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
root = os.environ.get("FLIDE_ROOT")
if root:
    sys.path.insert(0, root)     # takes precedence over this checkout

import torch  # noqa: E402

from flide.api.schema import SystemOneRequest  # noqa: E402
from flide.runtime.engine import Engine  # noqa: E402

TEXT = ("system policy: the agent must check every invoice against the ledger before answering. " * 4000)


def row(state, question):
    return {"state": state, "questions": {"q": {"type": "noul", "instructions": question}}}


def timed(eng, state, question):
    plan = eng.format.plan(SystemOneRequest.model_validate(row(state, question)))
    torch.cuda.synchronize()
    t0 = __import__("time").perf_counter()
    eng.forward([plan], mode="graphs")
    torch.cuda.synchronize()
    return (__import__("time").perf_counter() - t0) * 1000


def engine(**kw):
    eng = Engine("models/OneJev1.6-0.8B", device="cuda:0", capture=4096, budget=8192, session_cache_gb=4.0,
                 session_min_tokens=512, backbone_options={"cuda_graphs": True}, **kw)
    if getattr(eng.sessions, "pstore", None) is None:      # the shared prefix store is attached on the serve path
        eng.sessions.attach_prefix_store(eng.backbone.device)
    return eng


def siblings(chars, runs):
    shared = TEXT[:chars]
    out = []
    for _ in range(runs):
        eng = engine()
        timed(eng, shared, "warm?")
        ts = [timed(eng, shared + " " + " ".join(f"agent{a} event{j} happened." for j in range(20)), f"q{a}?")
              for a in range(1, 5)]
        out.append((statistics.median(ts), eng.sessions.stats()))
        del eng
    return out


def agent(chars, runs):
    out = []
    for _ in range(runs):
        eng = engine()
        state = TEXT[:chars]
        timed(eng, state, "step 0?")
        ts = []
        for step in range(1, 8):
            state = state + " " + " ".join(f"step{step} event{j} happened." for j in range(20))
            ts.append(timed(eng, state, f"step {step}?"))
        out.append((statistics.median(ts), eng.sessions.stats()))
        del eng
    return out


def repeats(chars, runs):
    out = []
    for _ in range(runs):
        eng = engine()
        state = TEXT[:chars]
        timed(eng, state, "warm?")
        ts = [timed(eng, state, "same?") for _ in range(5)]
        out.append((statistics.median(ts), eng.sessions.stats()))
        del eng
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=12000)
    ap.add_argument("--runs", type=int, default=2)
    a = ap.parse_args()
    print("checkout:", root or "the script's repository", "| prefix chars:", a.chars)
    for name, fn in (("siblings", siblings), ("agent", agent), ("repeats", repeats)):
        got = fn(a.chars, a.runs)
        for ms, st in got:
            print("%-9s %6.1f ms | entries %d | bytes %.0f MB | hits %d"
                  % (name, ms, st["entries"], st["bytes"] / 2 ** 20, st["hits"]), flush=True)


main()

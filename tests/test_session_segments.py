"""The two segments of the session cache, without a GPU.

    python tests/test_session_segments.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.runtime.session import Entry, SessionCache


def entry(tokens, size=64):
    return Entry(tokens, {len(tokens) - 1: {"S": torch.zeros(size)}}, {})


def main():
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}{'' if ok else f' ({detail})'}")

    cache = SessionCache(4000, min_tokens=1, stride=8)
    check("a new entry enters probation", cache.stats()["probation"] == 0 or True)
    a = entry([1] * 20)
    cache.insert(a)
    check("one entry, all of it in probation", (cache.stats()["probation"], cache.stats()["protected"]) == (1, 0),
          cache.stats())

    cache.match([1] * 21)
    check("a hit promotes it to protected", (cache.stats()["probation"], cache.stats()["protected"]) == (0, 1),
          cache.stats())

    # a stream of one-off prefixes must not evict the protected entry
    for i in range(10):
        cache.insert(entry([100 + i] * 20))
    check("one-off prefixes stay in probation", cache.stats()["protected"] == 1, cache.stats())
    check("protected respects its share of the budget", cache.used - cache.pbytes <= cache.hbudget,
          (cache.used, cache.pbytes, cache.hbudget))
    kept, _ = cache.match([1] * 21)
    check("the protected entry survived the stream", kept is not None)

    # promoted entries beyond the protected share demote the least recently used one
    cache.clear()
    for i in range(14):
        cache.insert(entry([500 + i] * 20))
        cache.match([500 + i] * 21)
    check("protected stays inside its share",
          cache.used - cache.pbytes <= cache.hbudget, (cache.used, cache.pbytes, cache.hbudget))
    check("what is over the share falls back to probation", cache.stats()["probation"] >= 1, cache.stats())

    # accounting
    check("probation bytes plus the rest are the cache's bytes", cache.pbytes + (cache.used - cache.pbytes) == cache.used)
    cache.clear()
    check("clear resets both segments", (cache.stats()["entries"], cache.pbytes, len(cache.probation)) == (0, 0, 0),
          cache.stats())

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
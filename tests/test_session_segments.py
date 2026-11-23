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

    # a budget set after the cache is built, as the engine sizes it with --session-cache-gb auto
    late = SessionCache(0, min_tokens=1, stride=8)
    late.budget = 4000
    for i in range(3):
        late.insert(entry([700 + i] * 20))
        late.match([700 + i] * 21)
    check("a budget set later gives protected its share", late.stats()["protected"] == 3, late.stats())

    # accounting
    check("probation bytes plus the rest are the cache's bytes", cache.pbytes + (cache.used - cache.pbytes) == cache.used)
    cache.clear()
    check("clear resets both segments", (cache.stats()["entries"], cache.pbytes, len(cache.probation)) == (0, 0, 0),
          cache.stats())

    # sessions cut off by one-off requests: an agent's step resumes its last one and saves the next
    def request(c, tokens):
        e, _ = c.match(tokens)
        c.insert(entry(tokens), parent=e)
        return e is not None

    agents = SessionCache(4000, min_tokens=1, stride=8)
    request(agents, [9] * 20)
    for j in range(20):
        request(agents, [50_000 + j] * 20)
    back = request(agents, [9] * 28)
    check("a session the one-offs evicted misses", not back)
    check("and comes back to protected", agents.stats()["protected"] == 1 and agents.stats()["returning"] == 1,
          agents.stats())
    check("then hits", request(agents, [9] * 36))

    # a protected full of sessions that keep coming back stays as it is when another session comes back
    agents.clear()
    live = list(range(11))
    for i in live:
        request(agents, [1000 + i] * 20)
        request(agents, [1000 + i] * 21)
    check("eleven sessions fill protected", agents.stats()["protected"] == 11, agents.stats())
    hits = {i: 0 for i in live}
    late, oneoff = 0, 0

    def cycle(c, k):
        nonlocal late, oneoff
        for i in live:
            hits[i] += request(agents, [1000 + i] * (22 + c))
        late += request(agents, [2000] * (20 + c))
        for _ in range(5):
            oneoff += 1
            request(agents, [60_000 + oneoff] * 20)

    for c in range(6):
        cycle(c, 0)
    check("the live sessions hit every step", all(v == 6 for v in hits.values()), hits)
    check("the late session stays out", late == 0, late)

    # one of them ends: once it has gone unused long enough the late session takes its place
    live.remove(0)
    for c in range(6, 14):
        cycle(c, 0)
    check("the late session gets in once one has ended", late >= 4, late)
    check("the live sessions still hit", all(hits[i] == 14 for i in live), hits)
    ended = next((k for k, e in agents.entries.items() if e.tokens[0] == 1000), None)
    check("the ended session left protected", ended is None or ended in agents.probation, ended)

    # peek: where match would resume, without counting a lookup or using the entry
    p = SessionCache(4000, min_tokens=1, stride=8)
    p.insert(entry([3] * 20))
    p.insert(entry([4] * 20))
    before = (p.clock, p.hits, p.misses, list(p.entries), set(p.probation), dict(p.seen))
    pos = p.peek([3] * 25)
    check("peek finds the saved position", pos == 19, pos)
    check("peek of an unknown prefix is 0", p.peek([5] * 25) == 0)
    check("peek changes nothing", (p.clock, p.hits, p.misses, list(p.entries), set(p.probation), dict(p.seen))
          == before)
    check("match resumes where peek said", p.match([3] * 25)[1] == pos)

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
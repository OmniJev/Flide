"""The session cache's bookkeeping without a GPU: an agent's next step replaces the entry it resumed from, the least
recently used entries are evicted under the budget, and both are counted.

    python tests/test_session_cache.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.runtime.session import Entry, SessionCache, auto_bytes, cache_size

GIB = 2 ** 30


class Blob:   # a tensor's size, as far as the cache counts it
    def __init__(self, n):
        self.n = n

    def numel(self):
        return self.n

    def element_size(self):
        return 1


def entry(tokens, mark, kv_per_token=10):
    return Entry(tokens[:mark], {mark: {"S": Blob(100)}}, {0: (Blob(kv_per_token * mark), Blob(0))})


def test_agent_keeps_one_entry():
    c = SessionCache(budget_bytes=10 ** 9, min_tokens=256, stride=256, margin=32)
    toks = list(range(100, 100 + 4096))
    prev = None
    for n in (1024, 2048, 3072, 4096):   # four steps, each the previous text and more
        hit, start = c.match(toks[:n])
        assert hit is prev, n
        mark = c.mark(n, start)
        e = entry(toks, mark)
        c.insert(e, parent=hit)
        prev = e
    s = c.stats()
    assert s["entries"] == 1 and s["replaced"] == 3 and s["evictions"] == 0 and s["hits"] == 3, s
    assert c.used == prev.bytes


def test_other_prefix_is_not_replaced():
    c = SessionCache(budget_bytes=10 ** 9, min_tokens=256, stride=256, margin=32)
    a = list(range(1000, 3000))
    b = list(range(5000, 7000))
    ea = entry(a, 1792)
    c.insert(ea)
    c.insert(entry(b, 1792), parent=ea)   # not over ea's tokens: both stay
    assert c.stats()["entries"] == 2 and c.stats()["replaced"] == 0


def test_lru_evictions_counted():
    one = entry(list(range(2048)), 1792).bytes
    c = SessionCache(budget_bytes=2 * one, min_tokens=256, stride=256, margin=32)
    for k in range(5):
        c.insert(entry(list(range(k * 10000, k * 10000 + 2048)), 1792))
    s = c.stats()
    assert s["entries"] == 2 and s["evictions"] == 3 and c.used <= c.budget, s
    hit, start = c.match(list(range(0, 2048)))   # the first agent's state is gone
    assert hit is None


def test_cache_size_flag():
    assert cache_size("auto") == "auto" and cache_size("AUTO") == "auto"
    assert cache_size("4") == 4.0 and cache_size("0") == 0.0
    for bad in ("four", "-1", ""):
        try:
            cache_size(bad)
        except Exception as e:
            assert "takes 'auto'" in str(e)
        else:
            raise AssertionError(bad)


def test_auto_size():
    # an H200 with a 9B model up: 100 GiB free, 6 GiB unused in the allocator, the warm-up peaked 6 GiB above now
    assert auto_bytes(100 * GIB, 140 * GIB, 30 * GIB, 24 * GIB, 30 * GIB) == 86 * GIB
    # a 24 GB card with little left: nothing
    assert auto_bytes(1 * GIB, 24 * GIB, 21 * GIB, 20 * GIB, 22 * GIB) == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

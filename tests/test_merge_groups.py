"""How the engine groups the requests of a batch round that start alike, without a GPU: who carries the shared start,
where it is cut, and when nothing is merged.

    python tests/test_merge_groups.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide import settings
from flide.runtime import engine as E
from flide.runtime.session import BLOCK


class Graphs:
    def run_sflat(self, items):   # only its presence matters to the grouping
        raise AssertionError("not run here")


class Stub:
    def __init__(self, sessions=True):
        self.sessions = object() if sessions else None
        self.backbone = SimpleNamespace(request_graphs=lambda: Graphs())

    def _sflat_reads_ok(self, plan):
        return True


def plan(tokens):
    return SimpleNamespace(prefix=list(tokens))


def groups(plans, sessions=True, skip=()):
    return E.Engine._merge_groups(Stub(sessions), plans, skip)


def main():
    fails = 0

    def check(name, got, want):
        nonlocal fails
        ok = got == want
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: {got!r}")

    shared = list(range(1000))
    own = lambda k, n=300: [100000 + 1000 * k + j for j in range(n)]   # noqa: E731
    three = [plan(shared + own(k)) for k in range(3)]
    cut = 1000 // BLOCK * BLOCK
    check("three requests on one 1000-token start: cut on the last whole block", groups(three),
          {0: (0, cut), 1: (0, cut), 2: (0, cut)})
    settings.batch_invariant = True
    try:
        check("the same under the batch-invariant products", groups(three), {0: (0, cut), 1: (0, cut), 2: (0, cut)})
    finally:
        settings.batch_invariant = False
    check("no session cache, no merge", groups(three, sessions=False), {})
    check("a skipped request stays out", groups(three, skip={1}), {0: (0, cut), 2: (0, cut)})
    short = [plan(list(range(400)) + own(k)) for k in range(3)]
    check("a start under MERGE_MIN_TOKENS is not merged", groups(short), {})
    mixed = three + [plan(shared[:600] + own(9))]
    g = groups(mixed)
    check("a request sharing less does not shorten the group's cut", (g.get(0), g.get(1), g.get(2)),
          ((0, cut), (0, cut), (0, cut)))
    check("and it waits for a group of its own", 3 in g, False)
    whole = [plan(shared), plan(shared + own(1))]
    check("a request whose whole prefix is the shared start is not cut at its end",
          all(n < len(whole[i].prefix) for i, (_, n) in groups(whole).items()), True)
    check("every cut is a whole number of blocks", all(n % BLOCK == 0 for _, n in groups(mixed).values()), True)

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

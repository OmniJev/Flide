"""The prefix store without a GPU: sharing, refcounts, leaf-first eviction and a stable key."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.runtime.store import Store, key_of


def main():
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}{'' if ok else f' ({detail})'}")

    s = Store(budget_bytes=1000)
    a = s.insert(list(range(100)), {64: {"s": 1}}, kv_bytes=400, session="A")
    b = s.insert(list(range(150)), {128: {"s": 2}}, kv_bytes=400, session="B")
    check("a child extends its parent instead of repeating it", b.parent is a, (b.parent is a, b.span))
    check("the child adds only its own block", b.tokens == list(range(100, 150)), b.tokens[:3])
    check("two nodes, both charged once", (len(s.nodes), s.used) == (2, 800), (len(s.nodes), s.used))

    node, pos = s.match(list(range(150)))
    check("the deepest saved position matches", pos == 128, pos)

    s.release(a)
    check("a parent stays while a child holds it", a.key in s.nodes, sorted(len(n.tokens) for n in s.nodes.values()))

    c = s.insert(list(range(150, 220)), {200: {"s": 3}}, kv_bytes=400)
    check("going over budget evicts a leaf", s.used <= s.budget and c in s.nodes.values(), (s.used, s.budget))

    check("keys are stable across calls", key_of(b"", list(range(10))) == key_of(b"", list(range(10))))
    check("keys differ when the block differs", key_of(b"", list(range(10))) != key_of(b"", list(range(11))))

    s.clear()
    check("clear empties the store", (s.used, len(s.nodes), len(s.by_position)) == (0, 0, 0),
          (s.used, len(s.nodes)))

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

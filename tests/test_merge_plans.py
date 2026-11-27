"""Grouping and splitting same-prefix plans, without a GPU."""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from flide.runtime.merge import group, split  # noqa: E402


@dataclass
class FakePlan:
    prefix: list
    suffixes: list
    reads: object
    attention: str = "causal"
    positions: object = None
    offset: int = 0
    vision: object = None


def plan(prefix, branches, tag="r"):
    return FakePlan(prefix=list(prefix), suffixes=[list(b) for b in branches], reads=(tag,))


def main():
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}{'' if ok else f' ({detail})'}")

    a = plan([1, 2, 3, 4], [[10], [11]])
    b = plan([1, 2, 3, 4], [[20]])
    c = plan([9, 9], [[30]])
    groups = group([a, b, c])
    check("two same-prefix plans form one group", len(groups) == 2, len(groups))
    merged, parts = groups[0]
    check("branches are concatenated", merged.suffixes == [[10], [11], [20]], merged.suffixes)
    check("reads follow the branches", merged.reads == [(("r",), ("r",), ("r",))][0] or len(merged.reads) == 3,
          merged.reads)
    check("the odd plan keeps its own place", groups[1][1] == [(2, 1)], groups[1][1])

    outputs = [["A0", "A1", "A2"], ["C0"]]
    back = split(outputs, groups, 3)
    check("outputs are split back per plan", back[0] == ["A0", "A1"] and back[1] == ["A2"] and back[2] == ["C0"], back)

    d = plan([], [[40]])
    e = plan([1], [[50]])
    check("an empty prefix never merges", len(group([d, d], )) == 2, [len(g[1]) for g in group([d, d])])
    check("different prefixes stay apart", len(group([e, c])) == 2, len(group([e, c])))

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

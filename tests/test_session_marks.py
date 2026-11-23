"""Session cache marks without a GPU: where a prefix is saved, and what a shorter shared prefix matches.

    python tests/test_session_marks.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.runtime.engine import session_saves
from flide.runtime.session import SessionCache, Entry


def main():
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}{'' if ok else f' ({detail})'}")

    cache = SessionCache(1 << 30, min_tokens=1024, stride=256)

    check("a prefix below the floor saves nothing", cache.marks(500) == (), cache.marks(500))
    check("a prefix saves the last stride mark and the one before it", cache.marks(1024) == (512, 768),
          cache.marks(1024))
    check("a prefix of 2918 saves 2560 and 2816", cache.marks(2918) == (2560, 2816), cache.marks(2918))
    check("a resume past the earlier mark keeps only the later one", cache.marks(2918, start=2800) == (2816,),
          cache.marks(2918, start=2800))
    check("a resume past the later mark saves nothing", cache.marks(2918, start=2900) == (),
          cache.marks(2918, start=2900))

    # a request that shares the preamble but diverges before the last stride mark still finds the earlier one
    tokens = list(range(2918))
    marked = cache.marks(len(tokens))
    cache.insert(Entry(tokens, {p: {"S": torch.zeros(4)} for p in marked}, {}))
    sibling = list(range(2734)) + list(range(50000, 50300))
    _, pos = cache.match(sibling)
    check("a sibling that diverges before the last mark matches the earlier one", pos == 2560, pos)
    longer = list(range(3000))
    _, pos = cache.match(longer)
    check("an extension matches the last mark", pos == 2816, pos)
    _, missed = cache.match(list(range(2000)) + list(range(50000, 51000)))
    check("a prefix that shares no mark misses", missed == 0, missed)

    # what the continuation graphs hand back, in each of their forms, as the list the engine saves from
    snap = ({0: "S"}, {})
    check("a list of saves stays as it is", session_saves([(512, snap), (768, snap)], 768) == [(512, snap), (768, snap)])
    check("no save without a mark", session_saves(None, None) == [])
    check("an empty list saves nothing", session_saves([], None) == [])
    check("one snapshot of the stepwise path saves at the mark", session_saves(snap, 768) == [(768, snap)])

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
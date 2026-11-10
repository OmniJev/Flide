"""Tile shapes without a GPU: which TMA shape the batch-invariant product picks for a shared-memory budget.

    python tests/test_invariant.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.kernels.invariant import CFG, NEEDS, SMALL_CFG, SMALL_M, pick

HOPPER = 232448
CONSUMER_BLACKWELL = 101376


def main():
    fails = 0

    def check(name, got, want):
        nonlocal fails
        ok = got == want
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name} ({got} vs {want})")

    check("the wide shape does not fit a 99 KB budget", NEEDS[CFG] > CONSUMER_BLACKWELL, True)
    check("the narrow shape fits a 99 KB budget", NEEDS[SMALL_CFG] <= CONSUMER_BLACKWELL, True)
    check("the wide shape fits Hopper's 227 KB", NEEDS[CFG] <= HOPPER, True)

    check("a long product on Hopper keeps the wide shape", pick(4096, HOPPER), CFG)
    check(f"{SMALL_M} rows or fewer keep the narrow shape on Hopper", pick(SMALL_M, HOPPER), SMALL_CFG)
    check("a long product on 99 KB takes the narrow shape", pick(4096, CONSUMER_BLACKWELL), SMALL_CFG)
    check("32 KB fits neither, so the portable kernel serves", pick(4096, 32768), None)
    check("a forced shape is returned unchanged", pick(4096, CONSUMER_BLACKWELL, cfg=CFG), CFG)

    for budget_bytes in (16384, 32768, CONSUMER_BLACKWELL, HOPPER, 1 << 20):
        for M in (1, SMALL_M, SMALL_M + 1, 4096):
            shape = pick(M, budget_bytes)
            if shape is not None and NEEDS[shape] > budget_bytes:
                print(f"FAIL picked {shape} for {budget_bytes} bytes at M={M}")
                fails += 1

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
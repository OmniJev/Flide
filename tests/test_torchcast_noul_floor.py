"""The torchcast family's yes/no floor, without a GPU: noul_floor as the release's model.py `_answer` reads it.

    python tests/test_torchcast_noul_floor.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins"))

import flide_torchcast as family

FLOOR = 0.801


def release(p_yes, floor):
    """model.py `_answer`, written out: what the family has to reproduce."""
    p = float(p_yes)
    if floor and max(p, 1.0 - p) < floor:
        return floor if p >= 0.5 else 1.0 - floor
    return p


def main():
    fails = 0

    def check(name, got, want):
        nonlocal fails
        ok = abs(got - want) < 1e-12
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: {got:.6f} want {want:.6f}")

    def refuses(name, call):
        nonlocal fails
        try:
            call()
        except ValueError:
            print(f"PASS {name}: refused")
            return
        fails += 1
        print(f"FAIL {name}: accepted")

    # a probability already above the floor is untouched
    check("0.9466 stays", family.noul_with_floor(0.9466, FLOOR), 0.9466)
    check("0.8138 stays", family.noul_with_floor(0.8138, FLOOR), 0.8138)
    # under the floor it moves onto its own side's floor, never across
    check("0.78 moves up to the floor", family.noul_with_floor(0.78, FLOOR), FLOOR)
    check("0.60 moves up to the floor", family.noul_with_floor(0.60, FLOOR), FLOOR)
    check("0.49 moves the no side up", family.noul_with_floor(0.49, FLOOR), 1.0 - FLOOR)
    check("0.30 moves the no side up", family.noul_with_floor(0.30, FLOOR), 1.0 - FLOOR)
    # the floor reads max(p, 1 - p): a lopsided probability is already committed and stays
    check("0.05 stays, its far side is 0.95", family.noul_with_floor(0.05, FLOOR), 0.05)
    check("0.95 stays", family.noul_with_floor(0.95, FLOOR), 0.95)
    # a tie goes to yes, as the release does
    check("0.5 tie goes to yes", family.noul_with_floor(0.5, FLOOR), FLOOR)
    # exactly on the floor is not under it
    check("0.801 on the floor is untouched", family.noul_with_floor(FLOOR, FLOOR), FLOOR)
    # floor 0 is off
    check("floor 0 leaves 0.51 alone", family.noul_with_floor(0.51, 0.0), 0.51)
    # and never moves the answer: the side is the same before and after
    for p in (0.999, 0.95, 0.85, 0.81, 0.79, 0.5, 0.3, 0.1, 0.001):
        check(f"answer side kept at p={p}", float(family.noul_with_floor(p, FLOOR) >= 0.5),
              1.0 if p >= 0.5 else 0.0)

    # the floor the checkpoint declares
    check("reads noul_floor", family.noul_floor_of({"noul_floor": FLOOR}), FLOOR)
    check("absent noul_floor is off", family.noul_floor_of({}), 0.0)
    check("null noul_floor is off", family.noul_floor_of({"noul_floor": None}), 0.0)
    refuses("noul_floor 0.4 is refused", lambda: family.noul_floor_of({"noul_floor": 0.4}))
    refuses("noul_floor 1.0 is refused", lambda: family.noul_floor_of({"noul_floor": 1.0}))

    # every value the release would return, over a grid
    bad = [p for p in (0.01 * i for i in range(101)) if abs(family.noul_with_floor(p, FLOOR) - release(p, FLOOR)) > 1e-12]
    print(f"{'PASS' if not bad else 'FAIL'} grid 0.00..1.00 matches the release: {len(bad)} differ")
    fails += bool(bad)

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

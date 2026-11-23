"""A default budget that does not fit at the session warmup halves until it does, and one the caller set does not
move, without a GPU.

    python tests/test_warm_budget.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.runtime.engine import Engine  # noqa: E402


def engine(budget, given, fits):
    e = Engine.__new__(Engine)
    e.budget, e._budget_given = budget, given
    e.ran = []

    def run_plans(plans, budget=None):
        tokens = sum(len(p.prefix) + sum(len(s) for s in p.suffixes) for p in plans)
        if tokens > fits:
            raise torch.OutOfMemoryError("CUDA out of memory")
        e.ran.append(tokens)

    e.run_plans = run_plans
    return e


def main():
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}{'' if ok else f' ({detail})'}")

    tok = lambda n: [1] * n
    e = engine(65536, False, 70000)
    e._warm_budget(tok)
    check("a budget that fits stays", e.budget == 65536, e.budget)

    e = engine(65536, False, 40000)
    e._warm_budget(tok)
    check("a default budget that does not fit halves once", e.budget == 32768, e.budget)
    check("and the warmup runs at it", len(e.ran) == 1 and e.ran[0] <= 40000, e.ran)

    e = engine(65536, False, 10000)
    e._warm_budget(tok)
    check("halves until it fits", e.budget == 8192, e.budget)

    e = engine(65536, True, 40000)
    try:
        e._warm_budget(tok)
        raised = False
    except torch.OutOfMemoryError:
        raised = True
    check("a budget the caller set is not moved: the error stands", raised and e.budget == 65536, e.budget)

    e = engine(65536, False, 100)
    try:
        e._warm_budget(tok)
        raised = False
    except torch.OutOfMemoryError:
        raised = True
    check("it stops at 4096", raised and e.budget == 4096, e.budget)

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

"""Which token budget a checkpoint asks for, without a GPU.

    python tests/test_budget.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.runtime.engine import _budget_hint


def main():
    fails = 0

    def check(name, got, want):
        nonlocal fails
        ok = got == want
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name} ({got!r} vs {want!r})")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        cfg = root / "decision_config.json"
        check("a checkpoint without decision_config.json", _budget_hint(d), None)
        cfg.write_text(json.dumps({"format_id": "x", "package": "flide"}))
        check("a decision_config.json without a budget", _budget_hint(d), None)
        cfg.write_text(json.dumps({"format_id": "x", "budget": 16384}))
        check("the budget a checkpoint names", _budget_hint(d), 16384)
        cfg.write_text(json.dumps({"format_id": "x", "budget": None}))
        check("a budget of None reads as no budget", _budget_hint(d), None)
        for bad in (0, -1, 1.5, "16384", True):
            cfg.write_text(json.dumps({"format_id": "x", "budget": bad}))
            try:
                got = _budget_hint(d)
            except ValueError:
                got = "ValueError"
            check(f"a budget of {bad!r} is refused", got, "ValueError")

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
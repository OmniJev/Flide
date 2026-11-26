"""A quantization without a batch-invariant form is refused with --batch-invariant (fp8 and int8 have one), verify takes --quantize, and serve
takes --budget.

    python tests/test_engine_flags.py
"""
import argparse
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.runtime.engine import refuse_invariant_quantized
from flide.api import serve
from flide.verify import add_parser


def main():
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}{'' if ok else f' ({detail})'}")

    def refused(batch_invariant, options):
        try:
            refuse_invariant_quantized(batch_invariant, options)
        except ValueError as exc:
            return str(exc)
        return ""

    check("fp8 with --batch-invariant is allowed (one summation order)", refused(True, {"quantize": "fp8"}) == "")
    message = refused(True, {"quantize": "nf4"})
    check("a quantization without an invariant form is refused with --batch-invariant", bool(message), message)
    check("the message names both flags", "--quantize nf4" in message and "--batch-invariant" in message, message)
    check("the message says why", "scaled matmul" in message and "split depends on the batch" in message, message)
    check("fp8 alone is allowed", refused(False, {"quantize": "fp8"}) == "")
    check("--batch-invariant alone is allowed", refused(True, {}) == "")
    check("a missing quantize value is not a conflict", refused(True, {"quantize": None}) == "")
    check("no options at all is allowed", refused(True, None) == "")
    check("int8 with --batch-invariant is allowed (exact int32 sums)", refused(True, {"quantize": "int8"}) == "")

    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    args = parser.parse_args(["verify", "--model", "m", "--requests", "r", "--quantize", "fp8"])
    check("verify takes --quantize fp8", args.quantize == "fp8", args.quantize)
    args = parser.parse_args(["verify", "--model", "m", "--requests", "r", "--quantize", "int8"])
    check("verify takes --quantize int8", args.quantize == "int8", args.quantize)
    args = parser.parse_args(["verify", "--model", "m", "--requests", "r"])
    check("verify leaves it unset by default", args.quantize is None, args.quantize)
    try:
        parser.parse_args(["verify", "--model", "m", "--requests", "r", "--quantize", "int4"])
        bad = False
    except SystemExit as exc:
        bad = exc.code == 2
    check("verify refuses a quantization it does not know", bad)

    parser = argparse.ArgumentParser()
    serve.add_parser(parser.add_subparsers())
    args = parser.parse_args(["serve", "--model", "m", "--budget", "8192"])
    check("serve takes --budget", args.budget == 8192, args.budget)
    args = parser.parse_args(["serve", "--model", "m"])
    check("serve leaves the budget to the card by default", args.budget is None, args.budget)
    check("serve passes the budget to the engine", "budget=args.budget" in inspect.getsource(serve))

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

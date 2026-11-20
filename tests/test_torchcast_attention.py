"""The torchcast family's attention follows the checkpoint, without a GPU or a checkpoint.

    python tests/test_torchcast_attention.py

The release reads decision_config.json's attention_mode; a checkpoint that declares "noncausal_full_attention" reads
every prompt whole both ways in the full-attention layers, so each question is a sequence of its own under attention
"blocks". This pins the mapping and refuses a mode the family does not read.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins"))

import flide_torchcast as family


def main():
    fails = 0

    def check(name, got, want):
        nonlocal fails
        ok = got == want
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: {got!r}")

    def refuses(name, call):
        nonlocal fails
        try:
            call()
        except ValueError:
            print(f"PASS {name}: refused")
            return
        fails += 1
        print(f"FAIL {name}: accepted")

    # the release's noncausal mask is the runtime's "blocks"
    check("noncausal_full_attention reads as blocks",
          family.attention_of({"attention_mode": "noncausal_full_attention"}), "blocks")
    # a checkpoint that declares nothing, or declares causal, stays causal
    check("no attention_mode stays causal", family.attention_of({}), "causal")
    check("attention_mode null stays causal", family.attention_of({"attention_mode": None}), "causal")
    check("attention_mode causal stays causal", family.attention_of({"attention_mode": "causal"}), "causal")
    # a mode nobody implements is refused, not silently read one way
    refuses("unknown attention_mode", lambda: family.attention_of({"attention_mode": "full"}))

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

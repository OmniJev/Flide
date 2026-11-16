"""The torchcast family's attention follows the checkpoint, without a GPU or a checkpoint.

    python tests/test_torchcast_attention.py

The release reads decision_config.json's attention_mode; this family read the Plan default "causal" whatever the
checkpoint declared, so every full-attention layer saw the prompt one way. This pins the mapping and the override.
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
    check("attention_mode causal stays causal", family.attention_of({"attention_mode": "causal"}), "causal")
    # the override wins over the checkpoint, so the two can be measured against each other
    check("override beats the checkpoint",
          family.attention_of({"attention_mode": "noncausal_full_attention"}, "causal"), "causal")
    check("override prefix on a silent checkpoint", family.attention_of({}, "prefix"), "prefix")
    # a mode nobody implements is refused, not silently read one way
    refuses("unknown attention_mode", lambda: family.attention_of({"attention_mode": "full"}))
    refuses("unknown override", lambda: family.attention_of({}, "bidir"))

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

"""The dense gemma4 FP8 conversion without a GPU: which weights move, and row slicing of a quantized projection.

    python tests/test_gemma4_fp8.py

The end-to-end check needs a gemma4 checkpoint; this covers the parts that do not.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.backbones.gemma4.fused import DENSE_KEYS, quantize_dense
from flide.kernels.fp8 import Fp8Weight


def weight(rows, cols=32):
    torch.manual_seed(rows * 1000 + cols)
    return torch.randn(rows, cols, dtype=torch.bfloat16)


def main():
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}{'' if ok else f' ({detail})'}")

    layers = []
    for _ in range(2):
        layers.append({key: weight(8) for key in DENSE_KEYS} | {"ple_gate": weight(8), "ple_proj": weight(8),
                                                                "inter": 8, "hd": 4, "nh": 2, "nkv": 1})
    n = quantize_dense(layers)
    check("every dense projection of every layer is converted", n == 2 * len(DENSE_KEYS), n)
    check("the converted weights are FP8 weights", all(isinstance(l[key], Fp8Weight) for l in layers for key in DENSE_KEYS))
    check("the per-layer input gate stays in the model dtype",
          all(isinstance(l["ple_gate"], torch.Tensor) for l in layers))
    check("the bookkeeping keys are untouched", all(l["inter"] == 8 and l["hd"] == 4 for l in layers))

    merged = Fp8Weight.from_weight(weight(12))
    q_rows, kv_rows = merged[:8], merged[8:]
    check("a merged projection splits into q and kv rows",
          q_rows.shape[0] == 8 and kv_rows.shape[0] == 4, (q_rows.shape, kv_rows.shape))
    check("the slices carry their own scales", q_rows.scale.shape == (1, 8) and kv_rows.scale.shape == (1, 4),
          (q_rows.scale.shape, kv_rows.scale.shape))
    check("a slice keeps the model dtype", q_rows.out_dtype == torch.bfloat16, q_rows.out_dtype)

    tailed = Fp8Weight.from_weight(weight(8), tail_rows=2)
    try:
        tailed[:4]
        refused = False
    except ValueError:
        refused = True
    check("a weight with tail rows refuses to slice", refused)

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
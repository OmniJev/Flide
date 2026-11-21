"""A float8 checkpoint whose weight scale sits in another shard than the weight (apertixai/cerno-27b-v1.1 has two):
the direct loader finds it through the index, on the CPU.

    python tests/test_load_split_scale.py
"""
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safetensors.torch import save_file  # noqa: E402

from flide.backbones.load import _load_into, tensor_files  # noqa: E402


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(8, 4, bias=False)
        self.b = torch.nn.Linear(8, 4, bias=False)


def main():
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}{'' if ok else f' ({detail})'}")

    torch.manual_seed(0)
    qa, qb = (torch.randn(4, 8).to(torch.float8_e4m3fn) for _ in range(2))
    sa, sb = torch.rand(4, 1) + 0.5, torch.rand(4, 1) + 0.5
    with tempfile.TemporaryDirectory() as d:
        save_file({"a.weight": qa, "a.weight_scale": sa, "b.weight_scale": sb}, f"{d}/model-00001-of-00002.safetensors")
        save_file({"b.weight": qb}, f"{d}/model-00002-of-00002.safetensors")
        index = {"a.weight": "model-00001-of-00002.safetensors", "a.weight_scale": "model-00001-of-00002.safetensors",
                 "b.weight_scale": "model-00001-of-00002.safetensors", "b.weight": "model-00002-of-00002.safetensors"}
        Path(d, "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": index}))
        check("the index gives every tensor's shard", set(tensor_files(d)) == set(index))
        fp8 = {}
        try:
            _load_into(Toy(), d, "cpu", None, (), lambda name: True, fp8, None)
        except Exception as exc:   # the bug: b's scale looked up in b's own shard
            check("loads", False, f"{type(exc).__name__}: {exc}")
        else:
            check("loads", True)
            check("a: float8 weight and scale from one shard",
                  torch.equal(fp8["a.weight"][0].view(torch.uint8), qa.view(torch.uint8))
                  and torch.equal(fp8["a.weight"][1], sa.float().reshape(-1, 1)))
            check("b: its scale from the other shard",
                  torch.equal(fp8["b.weight"][0].view(torch.uint8), qb.view(torch.uint8))
                  and torch.equal(fp8["b.weight"][1], sb.float().reshape(-1, 1)))

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

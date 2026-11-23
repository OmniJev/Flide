"""The session cache's default floor falls with the size of the model's layers, without a GPU.

    python tests/test_session_floor.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.runtime.engine import session_floor  # noqa: E402


class Layers(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty(int(params), device="meta"))


class Qwen:   # the layers on the backbone
    def __init__(self, params):
        self.layers = Layers(params)


class Gemma:   # the layers on its language model
    def __init__(self, params):
        self.lm = type("LM", (), {"layers": Layers(params)})()


def main():
    fails = 0

    def check(name, got, want):
        nonlocal fails
        ok = got == want
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}{'' if ok else f' (got {got}, want {want})'}")

    check("a 0.8B model keeps 2048", session_floor(Qwen(0.5e9)), 2048)
    check("2B of layers keeps 2048", session_floor(Qwen(2e9)), 2048)
    check("4B of layers halves it", session_floor(Qwen(4e9)), 1024)
    check("8B of layers: 512", session_floor(Qwen(8e9)), 512)
    check("in strides of 256", session_floor(Qwen(5e9)), 768)
    check("a 31B model: one stride", session_floor(Gemma(29e9)), 256)
    check("never below one stride", session_floor(Gemma(200e9)), 256)
    check("no layers at hand: 2048", session_floor(object()), 2048)

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

"""The default token budget per micro-batch, without a GPU.

    python tests/test_budget_default.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.runtime.engine import _total_memory, default_budget

GIB = 2 ** 30


def test_small_cards():
    for total in (23.52 * GIB, 24 * GIB, 15.6 * GIB):   # RTX 4090 / 3090 / laptop 5090, a 16 GB card
        assert default_budget(int(total)) == 16384, total


def test_large_cards():
    for total in (31.36 * GIB, 44.4 * GIB, 79.2 * GIB, 139.8 * GIB):   # RTX 5090, A40, H100, H200
        assert default_budget(int(total)) == 65536, total


def test_no_device():
    assert default_budget(0) == 65536
    assert _total_memory("cpu") == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

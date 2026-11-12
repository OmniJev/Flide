"""Causal plans without a prefix get their sequences' shared start as one, without a GPU.

    python tests/test_rooted.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.formats.base import Plan, Read
from flide.runtime.engine import _rooted

S = list(range(100, 160))


def test_one_stream_read_at_several_positions():
    p = _rooted(Plan([], [S], (Read("h", positions=(-30, -1)),)))
    assert p.prefix == S[:30] and p.suffixes == [S[30:]] and p.reads == (Read("h", positions=(-30, -1)),)


def test_shared_start_of_several_sequences():
    p = _rooted(Plan([], [S + [1, 2], S + [3, 4, 5]], (Read("h"),)))
    assert p.prefix == S and p.suffixes == [[1, 2], [3, 4, 5]]


def test_left_alone():
    for plan in (Plan([], [[7]], (Read("h"),)),                                  # one token
                 Plan([], [S, S], (Read("h"),), attention="blocks"),               # block attention
                 Plan([], [S], (Read("h", positions=(3,)),)),                      # counted from the start
                 Plan([], [[1, 2, 3], [4, 5]], (Read("h"),)),                      # nothing shared
                 Plan(S[:5], [S[5:]], (Read("h"),))):                              # a prefix already
        assert _rooted(plan) is plan


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

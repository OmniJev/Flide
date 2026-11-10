"""Session states that outlive the process, without a GPU.

    python tests/test_session_persist.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.runtime.session import Entry, SessionCache

GIB = 2 ** 30
TOKENS = list(range(3000))
POS = 2560


def entry(pos=POS):
    return Entry(TOKENS, {pos: {"S": torch.zeros(4, 8), "conv": torch.zeros(2, 3)}}, {0: torch.zeros(2, 4, pos, 8)})


def main():
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}{'' if ok else f' ({detail})'}")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "cache"
        first = SessionCache(GIB, min_tokens=2048, root=str(root), key="k1", device="cpu")
        first.insert(entry())
        check("nothing is written while the process runs", not list(root.glob("*.pt")), list(root.glob("*")))
        first.close()
        check("close writes the entry", len(list(root.glob("*.pt"))) == 1, list(root.glob("*.pt")))
        check("and records what it is valid for", json.loads((root / "meta.json").read_text())["key"] == "k1")

        second = SessionCache(GIB, min_tokens=2048, root=str(root), key="k1", device="cpu")
        check("a later process restores it", second.stats()["restored"] == 1, second.stats())
        hit, pos = second.match(TOKENS + [7, 8, 9])
        check("and a longer prefix resumes from it", hit is not None and pos == POS, (hit, pos))
        check("a prefix too short to be saved is not matched", second.match(TOKENS[:1000])[0] is None)

        third = SessionCache(GIB, min_tokens=2048, root=str(root), key="k2", device="cpu")
        check("another checkpoint's directory is not reused", third.stats()["restored"] == 0)
        check("and is emptied", not list(root.glob("*.pt")))

        cleared = SessionCache(GIB, min_tokens=2048, root=str(root), key="k2", device="cpu")
        cleared.insert(entry())
        cleared.clear()
        cleared.close()
        check("clear leaves nothing on disk", not list(root.glob("*.pt")))

        kept = SessionCache(GIB, min_tokens=2048, root=str(root), key="k3", keep=2, device="cpu")
        for p in (2560, 2816, 2944):
            kept.insert(entry(pos=p))
        kept.close()
        check("keep bounds what is written", len(list(root.glob("*.pt"))) == 2, list(root.glob("*.pt")))

        with tempfile.TemporaryDirectory() as small_td:
            small_root = Path(small_td) / "cache"
            small = SessionCache(1024, min_tokens=2048, root=str(small_root), key="k", device="cpu")
            small.insert(entry())
            small.close()
            check("an entry larger than the budget is not saved", not list(small_root.glob("*.pt")))

    # one directory at a time: another process must be refused, this one may reopen it
    with tempfile.TemporaryDirectory() as lock_td:
        lock_root = Path(lock_td) / "cache"
        mine = SessionCache(GIB, min_tokens=2048, root=str(lock_root), key="k", device="cpu")
        mine.insert(entry())
        again = SessionCache(GIB, min_tokens=2048, root=str(lock_root), key="k", device="cpu")
        check("this process may reopen its own directory", again.stats()["on_disk"] == 0, again.stats())
        probe = (
            "import sys; sys.path.insert(0, %r)\n"
            "from flide.runtime.session import SessionCache\n"
            "try:\n"
            "    SessionCache(1 << 30, root=%r, key='k', device='cpu')\n"
            "except ValueError as exc:\n"
            "    print('refused'); sys.exit(0)\n"
            "print('accepted')\n" % (str(Path(__file__).resolve().parents[1]), str(lock_root))
        )
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
        check("another process is refused", "refused" in out.stdout + out.stderr, (out.stdout, out.stderr[-200:]))
        mine.close()
        later = SessionCache(GIB, min_tokens=2048, root=str(lock_root), key="k", device="cpu")
        check("and the directory is readable once it is released", later.stats()["restored"] == 1, later.stats())
        later.close()

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
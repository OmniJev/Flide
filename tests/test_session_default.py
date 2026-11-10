"""The session cache a device gets by default, without a GPU.

    python tests/test_session_default.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.runtime.engine import session_default_gib
from flide.runtime.offline import cache_gb, resolve_cache_gb  # noqa: F401  (resolve is exercised below)

GIB = 2 ** 30


def main():
    fails = 0

    def check(name, got, want):
        nonlocal fails
        ok = got == want
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name} ({got!r} vs {want!r})")

    check("a 24 GiB card gets 3.6 GiB", session_default_gib(24 * GIB), 3.6)
    check("a 16 GiB card gets 2.4 GiB", session_default_gib(16 * GIB), 2.4)
    check("an 80 GiB card is capped at 4 GiB", session_default_gib(80 * GIB), 4.0)
    check("a 12 GiB card gets 1.8 GiB", session_default_gib(12 * GIB), 1.8)
    check("a 6 GiB card gets nothing (0.9 GiB is under the floor)", session_default_gib(6 * GIB), 0.0)
    check("no device gets nothing", session_default_gib(0), 0.0)
    check("a negative size is treated as none", session_default_gib(-1), 0.0)
    check("never more than 4 GiB", session_default_gib(500 * GIB), 4.0)

    check("--session-cache-gb auto is marked for the device", cache_gb("auto"), -1.0)
    check("auto is case-insensitive", cache_gb(" AUTO "), -1.0)
    check("--session-cache-gb 4 is four GiB", cache_gb("4"), 4.0)
    check("--session-cache-gb 0 turns it off", cache_gb("0"), 0.0)
    for bad in ("four", "-1", ""):
        try:
            got = cache_gb(bad)
        except ValueError:
            got = "ValueError"
        check(f"{bad!r} is refused", got, "ValueError")

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
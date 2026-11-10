"""`/stats` without a GPU: the endpoint answers, merges a batcher's numbers, and stays read-only.

    python tests/test_stats.py
"""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.api.server import create_app


class _Family:
    name = "fake"
    aliases = ()
    meta_key = "fake"


class _Cache:
    def __init__(self):
        self.cleared = 0

    def stats(self):
        return {"entries": 1, "bytes": 1024, "hits": 5, "misses": 2, "saved_tokens": 2048}

    def clear(self):
        self.cleared += 1


class _Batcher:
    def __init__(self):
        self.asked = 0

    def stats(self):
        self.asked += 1
        return {"rounds": 2, "requests": 7, "planning": 0, "queue_s": 0.1, "forward_s": 0.2,
                "max_round": 4, "avg_round": 3.5}


def endpoint(app):
    return next(r for r in app.routes if getattr(r, "path", None) == "/stats").endpoint


def payload(app):
    response = asyncio.run(endpoint(app)())
    return json.loads(response.body)


def main():
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}{'' if ok else f' ({detail})'}")

    app = create_app(SimpleNamespace(format=_Family(), sessions=None), model_name="m", batcher=None)
    body = payload(app)
    check("answers without a batcher", body.get("model") == "m", body)
    check("sessions is null without a cache", body.get("sessions") is None, body)
    check("carries format, engine and version", body.get("format") == "fake" and body.get("engine") == "flide"
          and body.get("version"), body)
    check("reports an uptime", isinstance(body.get("uptime_s"), (int, float)), body)

    batcher = _Batcher()
    cache = _Cache()
    app2 = create_app(SimpleNamespace(format=_Family(), sessions=cache), model_name="m", batcher=batcher)
    body2 = payload(app2)
    check("merges the batcher's numbers", body2.get("rounds") == 2 and body2.get("requests") == 7, body2)
    check("asks the batcher once", batcher.asked == 1, batcher.asked)
    check("reports the cache it holds", body2.get("sessions", {}).get("hits") == 5, body2)
    check("does not clear the cache", cache.cleared == 0, cache.cleared)

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
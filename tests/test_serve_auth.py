"""`serve --api-key` and `--no-meta` without a GPU: the app is called as ASGI directly, with an engine that answers
every question yes.

    python tests/test_serve_auth.py

With a key, every route but /health needs `Authorization: Bearer KEY` (401 otherwise, with WWW-Authenticate and the
request id header); without one nothing changes. The key also comes from FLIDE_API_KEY, as `serve` hands it to
its front-end processes. `with_meta=False` (FLIDE_NO_META) leaves the family's metadata out of every answer.
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.api.answers import make_noul  # noqa: E402
from flide.api.schema import SystemOneResponse, Usage  # noqa: E402
from flide.api.server import create_app  # noqa: E402

KEY = "s3cret-key"
BODY = {"state": "The parcel arrived broken.", "questions": {"refund": {"type": "noul", "instructions": "Refund?"}}}


class _Family:
    name = "fake"
    aliases = ()
    meta_key = "fake"
    max_choice_options = 255
    max_score_levels = 255


class _Engine:
    format = _Family()
    sessions = None

    def decide(self, req, **_):
        answers = {qid: make_noul(0.9) for qid in req.questions}
        return SystemOneResponse(model="", answers=answers, usage=Usage(input_tokens=3, output_tokens=1)), {"path": "x"}


def call(app, method: str, path: str, body=None, headers=()):
    """One HTTP request through the ASGI app: (status, headers, decoded JSON or None)."""
    raw = b"" if body is None else json.dumps(body).encode()
    hs = [(b"content-type", b"application/json"), *((k.encode(), v.encode()) for k, v in headers)]
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "", "headers": hs,
             "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 8000)}
    sent, got = [], {"done": False}

    async def receive():
        if got["done"]:
            return {"type": "http.disconnect"}
        got["done"] = True
        return {"type": "http.request", "body": raw, "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    data = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], {k.decode(): v.decode() for k, v in start.get("headers", ())}, \
        (json.loads(data) if data.startswith((b"{", b"[")) else None)


def main():
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}{'' if ok else f' ({detail})'}")

    os.environ.pop("FLIDE_API_KEY", None)
    os.environ.pop("FLIDE_NO_META", None)
    open_app = create_app(_Engine(), model_name="m")
    status, _, body = call(open_app, "POST", "/v1/systemone", BODY)
    check("no key: answers", status == 200 and body["answers"]["refund"]["noul"] == 0.9, (status, body))
    check("no key: the family's metadata is in the answer", body.get("fake") == {"path": "x"}, body)

    keyed = create_app(_Engine(), model_name="m", api_key=KEY)
    status, _, _ = call(keyed, "GET", "/health")
    check("key: /health needs none", status == 200, status)
    for path, method, body_ in (("/v1/models", "GET", None), ("/v1/systemone", "POST", BODY), ("/stats", "GET", None)):
        status, headers, body = call(keyed, method, path, body_)
        check(f"key: {path} without it is 401", status == 401 and headers.get("www-authenticate") == "Bearer"
              and "x-typesafe-request-id" in headers, (status, headers))
        status, _, _ = call(keyed, method, path, body_, [("authorization", "Bearer wrong")])
        check(f"key: {path} with a wrong one is 401", status == 401, status)
        status, _, _ = call(keyed, method, path, body_, [("authorization", f"Bearer {KEY}")])
        check(f"key: {path} with it answers", status == 200, status)
    status, _, _ = call(keyed, "POST", "/v1/systemone", BODY, [("authorization", KEY)])
    check("key: a bare key without the Bearer scheme is 401", status == 401, status)

    os.environ["FLIDE_API_KEY"] = KEY
    from_env = create_app(_Engine(), model_name="m")
    status, _, _ = call(from_env, "GET", "/v1/models")
    check("FLIDE_API_KEY sets the key", status == 401, status)
    status, _, _ = call(from_env, "GET", "/v1/models", headers=[("authorization", f"Bearer {KEY}")])
    check("and the key opens it", status == 200, status)
    os.environ.pop("FLIDE_API_KEY")

    bare = create_app(_Engine(), model_name="m", with_meta=False)
    status, _, body = call(bare, "POST", "/v1/systemone", BODY)
    check("no meta: answers without the family's metadata", status == 200 and "fake" not in body, body)
    status, _, body = call(bare, "POST", "/v1/systemone", {"requests": [BODY, BODY]})
    check("no meta: nor in a batch", status == 200 and all("fake" not in r for r in body["results"]), body)
    os.environ["FLIDE_NO_META"] = "1"
    status, _, body = call(create_app(_Engine(), model_name="m"), "POST", "/v1/systemone", BODY)
    check("FLIDE_NO_META=1 does the same", status == 200 and "fake" not in body, body)
    os.environ.pop("FLIDE_NO_META")

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

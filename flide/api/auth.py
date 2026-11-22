"""A bearer key in front of the API (`flide serve --api-key KEY`, or FLIDE_API_KEY in the environment): every
HTTP request but /health carries `Authorization: Bearer KEY` or gets 401. A plain ASGI middleware, as the server's
_Stamp, so a request costs one header lookup and a constant-time comparison."""
from __future__ import annotations

import hmac
import json

OPEN = ("/health",)   # a load balancer's probe needs no key


class BearerKey:
    def __init__(self, app, key: str, open_paths=OPEN):
        if not key:
            raise ValueError("an API key is a non-empty string")
        self.app, self.key, self.open = app, key.encode(), frozenset(open_paths)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") in self.open:
            return await self.app(scope, receive, send)
        auth = next((v for k, v in scope.get("headers", ()) if k == b"authorization"), b"")
        scheme, _, token = auth.partition(b" ")
        if scheme.lower() == b"bearer" and hmac.compare_digest(token.strip(), self.key):
            return await self.app(scope, receive, send)
        body = json.dumps({"detail": "missing or wrong API key"}).encode()
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                                (b"www-authenticate", b"Bearer")]})
        await send({"type": "http.response.body", "body": body})

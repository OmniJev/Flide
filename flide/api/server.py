"""HTTP server for the TypeSafe-compatible System One API.

    POST /v1/systemone   answer typed questions about a state
    GET  /v1/models      served model and its aliases
    GET  /health
    POST /flush_cache    empty the session cache
    GET  /               playground

Requests that arrive together share a GPU round (`flide.runtime.online.Batcher`); planning and
finishing run on CPU threads, so /health stays responsive while a long request is on the GPU.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError

from .. import __version__
from .schema import ModelMetadata, ModelMetadataList, SystemOneRequest

log = logging.getLogger("flide.server")


def _validation_error(exc: ValidationError) -> JSONResponse:
    detail = [{"loc": ["body", *[str(x) for x in e["loc"]]], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


class _Stamp:
    """Request id and server time headers, as a plain ASGI middleware (Starlette's `@app.middleware` runs every
    request through extra tasks and streams, which costs CPU the GPU thread and the planners need under load)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        rid = next((v for k, v in scope.get("headers", ()) if k == b"x-typesafe-request-id"), None)
        rid = rid or f"flide-{uuid.uuid4().hex[:16]}".encode()
        t0 = time.perf_counter()

        async def stamped(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", ()))
                headers.append((b"x-typesafe-request-id", rid))
                headers.append((b"x-flide-server-time-ms", f"{(time.perf_counter() - t0) * 1000:.1f}".encode()))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, stamped)


def create_app(engine, model_name: str, description: str = "", release_date: str = "", batcher=None,
               ready=None) -> FastAPI:
    """`ready`, when given, is asked by /health (503 "loading" until it says yes): front ends start before the model
    has loaded."""
    family = engine.format
    aliases = set(family.aliases) | {"jev-latest"}
    app = FastAPI(title="flide", version=__version__)
    app.state.engine = engine
    app.state.debias = 1
    lock = threading.Lock()
    playground = Path(__file__).parent / "playground" / "index.html"
    released = release_date

    def decide(req, **options):
        with lock:
            return engine.decide(req, **options)

    async def answer(req, **options):
        if batcher is None:
            return await asyncio.to_thread(decide, req, **options)
        t0 = time.perf_counter()
        plan = await asyncio.wrap_future(batcher.plan(req, **options))
        t1 = time.perf_counter()
        branches, info = await asyncio.wrap_future(batcher.forward(plan))
        t2 = time.perf_counter()
        response, meta = await asyncio.wrap_future(batcher.finish(req, plan, branches, **options))
        meta["timing"] = {"plan_s": t1 - t0, "queue_s": info["queue_s"], "forward_s": info["forward_s"],
                          "gpu_wait_s": t2 - t1, "total_s": time.perf_counter() - t0}
        meta["path"] = info["path"]
        meta["round"] = info["round"]
        if "replica" in info:
            meta["replica"] = info["replica"]
        return response, meta

    app.add_middleware(_Stamp)

    @app.get("/health")
    async def health():
        if ready is not None and not ready():
            return JSONResponse({"status": "loading", "model": model_name}, status_code=503)
        return {"status": "ok", "model": model_name, "format": family.name, "engine": "flide",
                "version": __version__}

    @app.get("/v1/models")
    async def models() -> ModelMetadataList:
        items = [ModelMetadata(name=model_name, description=description or f"{family.name} decision model",
                               release_date=released)]
        items += [ModelMetadata(name=a, description=f"alias for {model_name}", release_date=released)
                  for a in sorted(aliases)]
        return ModelMetadataList(models=items)

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=422,
                                content={"detail": [{"loc": ["body"], "msg": "invalid JSON", "type": "json_invalid"}]})
        try:
            req = SystemOneRequest.model_validate(body)
        except ValidationError as exc:
            return _validation_error(exc)
        requested = req.model or "jev-latest"
        if requested not in aliases and requested != model_name:
            return JSONResponse(status_code=404, content={
                "detail": f"model {requested!r} is not served; available: {model_name} plus aliases {sorted(aliases)}"})
        options = {"debias": int(request.query_params.get("debias", app.state.debias)),
                   "debug": bool(request.query_params.get("debug"))}
        try:
            response, meta = await answer(req, **options)
        except ValueError as exc:
            return JSONResponse(status_code=422,
                                content={"detail": [{"loc": ["body"], "msg": str(exc), "type": "value_error"}]})
        response.model = model_name
        payload = response.model_dump()
        payload[family.meta_key] = meta
        return JSONResponse(content=payload)

    @app.api_route("/flush_cache", methods=["GET", "POST"])
    async def flush_cache():
        """Empty the session cache (nothing to do without one), as SGLang's /flush_cache; for when no request is in
        flight, e.g. between benchmark runs. A front end asks the GPU process to."""
        remote = getattr(batcher, "flush_sessions", None)
        if remote is not None:
            return {"status": "ok", "sessions": await asyncio.wrap_future(remote())}
        sessions = getattr(engine, "sessions", None)
        if sessions is not None:
            sessions.clear()
        return {"status": "ok", "sessions": sessions.stats() if sessions is not None else None}

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        if playground.exists():
            return playground.read_text()
        return "<h1>flide</h1><p>POST /v1/systemone</p>"

    return app

"""`flide serve`: load a checkpoint and serve the System One API.

Options a family does not use are left out of its constructor, so one
command line serves every family.
"""
from __future__ import annotations

import inspect
import logging
from pathlib import Path

log = logging.getLogger("flide.serve")


def _family_options(cls, options: dict) -> dict:
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return options
    dropped = sorted(k for k in options if k not in params)
    if dropped:
        log.info("format %s does not take %s", cls.name, ", ".join(dropped))
    return {k: v for k, v in options.items() if k in params}


def default_name(model: str) -> str:
    """Directory name for a local path, the model's own for a snapshot in the Hub cache; for a Hub id, the parts after
    the owner (Qwen/Qwen3.5-4B -> Qwen3.5-4B)."""
    if Path(model).exists():
        p = Path(model.rstrip("/"))
        if p.parent.name == "snapshots" and p.parent.parent.name.startswith("models--"):
            return "--".join(p.parent.parent.name.split("--")[2:])
        return p.name
    parts = model.split("/")
    return "-".join(parts[1:]) if len(parts) > 1 else model


def freeze_heap() -> None:
    """Collect once, then keep every object alive after loading (module wrappers, tokenizer tables, captured graphs)
    out of the collector's view. A full collection then visits only what requests make: a few milliseconds where it
    took close to 300 under load, all of it with the GPU thread waiting for the interpreter and the GPU idle."""
    import gc

    gc.collect()
    gc.freeze()


def run(args) -> None:
    import uvicorn

    from .. import formats
    from ..runtime.engine import Engine, model_config, resolve_model
    from ..runtime.offline import parse_kv
    from .server import create_app

    model_dir = resolve_model(args.model)
    family = formats.detect(model_dir, model_config(model_dir), args.format)
    options = {"calibration": args.calibration, "max_branch_tokens": args.max_branch_tokens,
               "max_request_tokens": args.max_request_tokens, "media_root": args.media_root,
               **parse_kv(args.format_arg)}
    options = {k: v for k, v in options.items() if v is not None}
    name = args.name or default_name(args.model)
    fronts = None
    if args.frontends > 0:
        from . import frontend

        fronts = frontend.start(args.frontends, model_dir, family.name, _family_options(family, options), name,
                                args.host, args.port, args.planners, args.access_log)
    engine = Engine(model_dir, format=family.name, device=args.device, dtype=args.dtype,
                    backbone_options={"cuda_graphs": not args.no_cuda_graphs, "quantize": args.quantize},
                    session_cache_gb=args.session_cache_gb, session_min_tokens=args.session_min_tokens,
                    batch_invariant=args.batch_invariant, **_family_options(family, options))
    from ..runtime.online import Batcher

    batcher = Batcher(engine, planners=args.planners, window_ms=args.batch_window_ms, gather_ms=args.gather_ms)
    freeze_heap()
    if fronts is not None:
        return frontend.serve(batcher, fronts)
    app = create_app(engine, model_name=name, batcher=batcher)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=args.access_log)


def add_parser(sub) -> None:
    s = sub.add_parser("serve", help="start the System One API server", description=__doc__)
    s.add_argument("--model", required=True, help="local directory or Hub repository id")
    s.add_argument("--format", default=None, help="model family (default: detected from the checkpoint)")
    s.add_argument("--format-arg", action="append", metavar="KEY=VALUE", help="family option")
    s.add_argument("--name", default=None, help="model name reported by the API (default: from --model)")
    s.add_argument("--calibration", default=None, help="calibration.json (default: the one next to the weights)")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--dtype", default="bfloat16")
    s.add_argument("--media-root", default=None,
                   help="directory that local media paths in requests resolve against (without it, only URLs and base64)")
    s.add_argument("--no-cuda-graphs", action="store_true", help="run every request eagerly")
    s.add_argument("--batch-invariant", action="store_true",
                   help="batch-invariant products: every request gets the same bits at any load (slower)")
    s.add_argument("--quantize", default=None, choices=["fp8"],
                   help="decoder weights in float8 with per-row scales, inputs scaled per token (compute capability 8.9+); "
                        "changes the numbers a little")
    s.add_argument("--max-branch-tokens", type=int, default=None)
    s.add_argument("--max-request-tokens", type=int, default=None)
    s.add_argument("--session-cache-gb", type=float, default=0.0,
                   help="GPU memory for saved prefix states; a request that extends an earlier prefix resumes from it")
    s.add_argument("--session-min-tokens", type=int, default=2048, help="shortest prefix the session cache saves")
    s.add_argument("--planners", type=int, default=4, help="CPU threads that render and tokenize requests")
    s.add_argument("--frontends", type=int, default=0,
                   help="HTTP front-end processes (each plans and finishes requests with its own planners); 0 serves "
                        "from the GPU process itself")
    s.add_argument("--batch-window-ms", type=float, default=0.0,
                   help="wait this long after the first waiting request for more to join its GPU round")
    s.add_argument("--gather-ms", type=float, default=None,
                   help="wait up to this long for requests still being planned to join a round (default 2)")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--access-log", action="store_true",
                   help="log every request (off by default: at hundreds of requests a second the lines cost CPU)")
    s.set_defaults(func=run)

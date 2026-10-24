"""Several HTTP front ends for one GPU.

Under load one Python process runs out of interpreter: parsing and validating requests, rendering and tokenizing
them, finishing answers and the GPU thread's own work all take the same lock. With `--frontends N` the process that
loads the model keeps only the GPU side (the batcher and a thread that feeds it), and N front-end processes, each with
the family alone (tokenizer, prompt rendering, answer rules) and no model, serve the API on the same port (the kernel
spreads connections over them, SO_REUSEPORT). A front end plans a request, sends the plan to the GPU process, gets the
head outputs back and finishes the answer. Requests still being planned are counted in the GPU process too, so a
round can wait for them as it does in one process. The front ends start before the model loads (they need the
tokenizer only) and answer /health with "loading" until the GPU process is ready.

With several replicas one GPU process per GPU holds a copy of the model, and the front ends pick a replica for every
request (`route.Router`): an agent's next step goes where its saved state is, unless that replica is much busier.
"""
from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import itertools
import logging
import socket
import threading
import types

from ..runtime.online import start_planners

log = logging.getLogger("flide.frontend")


class RemoteBatcher:
    """The batcher's interface (plan, forward, finish) inside a front end: plan and finish here, forward in a GPU
    process (`requests` holds one queue per replica, `router` picks among them)."""

    def __init__(self, family, fid: int, requests, replies, planners: int, router=None):
        self.family = family
        self.fid = fid
        self.requests = requests
        self.replies = replies
        self.router = router
        self._replica: dict[int, int] = {}
        self._sent: dict[int, int] = {}
        self._local = threading.local()
        self._planners = cf.ThreadPoolExecutor(max_workers=planners, thread_name_prefix="flide-plan",
                                               initializer=self._init_planner)
        start_planners(self._planners, planners)
        self._ids = itertools.count()
        self._waiting: dict[int, cf.Future] = {}
        self._lock = threading.Lock()
        threading.Thread(target=self._read, name="flide-replies", daemon=True).start()

    def _init_planner(self):
        self._local.format = self.family.for_thread()

    def _format(self):
        return getattr(self._local, "format", None) or self.family

    def plan(self, request, **options) -> cf.Future:
        r = self.router.pick(request.state) if self.router is not None else 0
        self.requests[r].put(("begin",))

        def run():
            plan = self._format().plan(request, **options)
            with self._lock:
                self._replica[id(plan)] = r
            return plan

        fut = self._planners.submit(run)
        fut.add_done_callback(lambda f: not f.cancelled() and f.exception() is not None and self._failed(r))
        return fut

    def _failed(self, r: int) -> None:
        self.requests[r].put(("fail",))
        if self.router is not None:
            self.router.done(r)

    def finish(self, request, plan, branches, **options) -> cf.Future:
        return self._planners.submit(lambda: self._format().finish(request, plan, branches, **options))

    def forward(self, plan) -> cf.Future:
        rid = next(self._ids)
        fut: cf.Future = cf.Future()
        with self._lock:
            self._waiting[rid] = fut
            r = self._replica.pop(id(plan))
            self._sent[rid] = r
        self.requests[r].put(("plan", self.fid, rid, dataclasses.replace(plan, data=None)))
        return fut

    def flush_sessions(self) -> cf.Future:
        """Empty the session cache of every replica; its stats after (a list with several replicas), or None without
        one."""
        futs = []
        for q in self.requests:
            rid = next(self._ids)
            f: cf.Future = cf.Future()
            with self._lock:
                self._waiting[rid] = f
            q.put(("flush", self.fid, rid))
            futs.append(f)
        if len(futs) == 1:
            return futs[0]
        out: cf.Future = cf.Future()
        cf.wait(futs)
        out.set_result([f.result() for f in futs])
        return out

    def _read(self):
        while True:
            rid, result, error = self.replies.get()
            with self._lock:
                fut = self._waiting.pop(rid)
                r = self._sent.pop(rid, None)
            if r is not None and self.router is not None:
                self.router.done(r)
                if error is None:
                    result[1]["replica"] = r
            if error is None:
                fut.set_result(result)
            else:
                fut.set_exception(RuntimeError(error))


def run_frontend(fid: int, model_dir: str, family_name: str, family_options: dict, model_name: str, requests, replies,
                 host: str, port: int, planners: int, access_log: bool, ready=(), route=None) -> None:
    import uvicorn

    from .. import formats
    from .route import Router
    from .server import create_app

    family = formats.get(family_name)(model_dir, **family_options)
    router = Router(*route[:2], fid, len(requests), *route[2:]) if route is not None else None
    bridge = RemoteBatcher(family, fid, requests, replies, planners, router)
    app = create_app(types.SimpleNamespace(format=family), model_name=model_name, batcher=bridge,
                     ready=lambda: all(e.is_set() for e in ready))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((host, port))
    config = uvicorn.Config(app, log_level="warning", access_log=access_log, backlog=2048)
    uvicorn.Server(config).run(sockets=[sock])


def dispatch(batcher, requests, replies) -> None:
    """GPU process: plans from the front ends into the batcher, head outputs back to the front end that asked."""

    def answer(f, fid, rid):
        exc = f.exception()
        replies[fid].put((rid, None, f"{type(exc).__name__}: {exc}") if exc is not None else (rid, f.result(), None))

    while True:
        msg = requests.get()
        if msg[0] == "begin":
            batcher._planned(1)
        elif msg[0] == "fail":
            batcher._planned(-1)
        elif msg[0] == "flush":
            _, fid, rid = msg
            sessions = batcher.engine.sessions
            if sessions is not None:
                sessions.clear()
            replies[fid].put((rid, sessions.stats() if sessions is not None else None, None))
        else:
            _, fid, rid, plan = msg
            batcher.forward(plan).add_done_callback(lambda f, fid=fid, rid=rid: answer(f, fid, rid))


def start(n: int, model_dir: str, family_name: str, family_options: dict, model_name: str, host: str, port: int,
          planners: int, access_log: bool, replicas: int = 1, slack: int = 8):
    """Start n front ends for `replicas` GPU processes (they serve /health as "loading" until every replica is up);
    returns what `serve` or `serve_replicas` needs."""
    import multiprocessing as mp

    from .route import shared

    ctx = mp.get_context("spawn")
    requests = [ctx.Queue() for _ in range(replicas)]
    replies = [ctx.Queue() for _ in range(n)]
    ready = [ctx.Event() for _ in range(replicas)]
    route = (*shared(ctx, replicas, n), n, slack) if replicas > 1 else None
    procs = [ctx.Process(target=run_frontend, name=f"flide-front-{i}", daemon=True,
                         args=(i, model_dir, family_name, family_options, model_name, requests, replies[i], host, port,
                               planners, access_log, ready, route)) for i in range(n)]
    for p in procs:
        p.start()
    log.info("%d front ends for %d replica%s on %s:%d", n, replicas, "s" if replicas > 1 else "", host, port)
    return procs, requests, replies, ready


def serve(batcher, fronts) -> None:
    """Feed the batcher from the front ends `start` made until stopped."""
    procs, requests, replies, ready = fronts
    ready[0].set()
    try:
        dispatch(batcher, requests[0], replies)
    finally:
        for p in procs:
            p.terminate()


def run_replica(i: int, build, requests, replies, ready) -> None:
    """One replica's GPU process: build its batcher (`build()`, which loads the model), then feed it."""
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s replica {i} %(name)s %(levelname)s %(message)s")
    batcher = build()
    ready.set()
    log.info("replica %d ready", i)
    dispatch(batcher, requests, replies)


def serve_replicas(build, gpus: list[str], fronts) -> None:
    """One GPU process per entry of `gpus` (the CUDA_VISIBLE_DEVICES it sees), each running `build` and serving the
    front ends `start` made, until one of them stops."""
    import multiprocessing as mp
    import os
    from multiprocessing.connection import wait

    procs, requests, replies, ready = fronts
    ctx = mp.get_context("spawn")
    workers = []
    before = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        for i, devices in enumerate(gpus):
            os.environ["CUDA_VISIBLE_DEVICES"] = devices
            w = ctx.Process(target=run_replica, name=f"flide-replica-{i}", args=(i, build, requests[i], replies, ready[i]))
            w.start()
            workers.append(w)
    finally:
        if before is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = before
    log.info("%d replicas on GPUs %s", len(gpus), " | ".join(gpus))
    try:
        wait([w.sentinel for w in workers])
        for i, w in enumerate(workers):
            if w.exitcode is not None:
                log.error("replica %d stopped (exit code %s)", i, w.exitcode)
    finally:
        for p in workers + procs:
            p.terminate()

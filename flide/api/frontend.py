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
import queue
import socket
import threading
import types
from multiprocessing.connection import wait

from fastapi import HTTPException

from ..runtime.online import PlannerPool

log = logging.getLogger("flide.frontend")


class _PipeOutbox:
    """Send on a private pipe without blocking HTTP or GPU threads on pipe I/O."""

    def __init__(self, writer):
        self._writer = writer
        self._messages = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._closed = False
        threading.Thread(target=self._send, name="flide-ipc-send", daemon=True).start()

    def put(self, message) -> None:
        with self._lock:
            if not self._closed:
                self._messages.put(message)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                try:
                    while True:
                        self._messages.get_nowait()
                except queue.Empty:
                    self._messages.put(None)

    def _send(self) -> None:
        try:
            with self._writer:
                while (message := self._messages.get()) is not None:
                    self._writer.send(message)
                    del message
        except OSError:
            pass
        finally:
            self.close()


class _ReplyInbox:
    def __init__(self, readers):
        self._readers = list(readers)

    def get(self):
        try:
            return wait(self._readers)[0].recv()
        except (EOFError, OSError) as exc:
            for reader in self._readers:
                reader.close()
            raise EOFError from exc


class RemoteBatcher:
    """The batcher's interface (plan, forward, finish) inside a front end: plan and finish here, forward in a GPU
    process (`requests` holds one queue per replica, `router` picks among them)."""

    def __init__(self, family, fid: int, requests, replies, planners: int, router=None, on_disconnect=None):
        self.family = family
        self.fid = fid
        self.requests = requests
        self.replies = replies
        self.router = router
        self._replica: dict[int, int] = {}   # id(plan) -> its replica, from plan to forward
        self._sent: dict[cf.Future, tuple[int, int]] = {}
        self._reserved: dict[cf.Future, int] = {}
        self._local = threading.local()
        self._planners = PlannerPool(planners, self._init_planner)
        self._ids = itertools.count()
        self._waiting: dict[int, cf.Future] = {}
        self._lock = threading.Lock()
        self._disconnected = threading.Event()
        self._on_disconnect = on_disconnect
        threading.Thread(target=self._read, name="flide-replies", daemon=True).start()

    def connected(self) -> bool:
        return not self._disconnected.is_set()

    def _check_connected(self) -> None:
        if not self.connected():
            raise HTTPException(status_code=503, detail="GPU process is unavailable")

    def _disconnect(self) -> None:
        with self._lock:
            self._disconnected.set()
            waiting = list(self._waiting.values())
            self._waiting.clear()
            self._sent.clear()
            self._reserved.clear()
        if self._on_disconnect is not None:
            self._on_disconnect()
        for future in waiting:
            if future.set_running_or_notify_cancel():
                future.set_exception(HTTPException(status_code=503, detail="GPU process is unavailable"))
        for outbox in self.requests:
            if isinstance(outbox, _PipeOutbox):
                outbox.close()

    def _init_planner(self):
        self._local.format = self.family.for_thread()

    def _format(self):
        return getattr(self._local, "format", None) or self.family

    def plan(self, request, **options) -> cf.Future:
        self._check_connected()
        r = self.router.pick(request.state) if self.router is not None else 0
        self.requests[r].put(("begin", self.fid))

        def run():
            plan = self._format().plan(request, **options)
            with self._lock:
                self._replica[id(plan)] = r
            return plan

        fut = self._planners.submit(run)
        fut.add_done_callback(lambda f: (f.cancelled() or f.exception() is not None) and self._failed(r))
        return fut

    def _failed(self, r: int) -> None:
        self.requests[r].put(("fail", self.fid))
        if self.router is not None:
            self.router.done(r)

    def follow_up(self, request, plan, branches, **options) -> cf.Future:
        """The family's next plan for a request, routed as a new plan is, or None."""
        def run():
            nxt = self._format().follow_up(request, plan, branches, **options)
            if nxt is not None:
                r = self.router.pick(request.state) if self.router is not None else 0
                self.requests[r].put(("begin", self.fid))
                with self._lock:
                    self._replica[id(nxt)] = r
            return nxt

        return self._planners.submit(run)

    def finish(self, request, plan, branches, **options) -> cf.Future:
        return self._planners.submit(lambda: self._format().finish(request, plan, branches, **options))

    def discard_plan(self, plan) -> None:
        with self._lock:
            r = self._replica.pop(id(plan))
        self._failed(r)

    def forward(self, plan) -> cf.Future:
        rid = next(self._ids)
        fut: cf.Future = cf.Future()
        with self._lock:
            r = self._replica.pop(id(plan))
            self._check_connected()
            self._waiting[rid] = fut
            self._sent[fut] = (rid, r)
        self.requests[r].put(("plan", self.fid, rid, dataclasses.replace(plan, data=None)))
        return fut

    def cancel_forward(self, future: cf.Future) -> None:
        with self._lock:
            sent = self._sent.get(future)
        if sent is not None:
            rid, r = sent
            self.requests[r].put(("cancel", self.fid, rid))

    def reserve(self, count: int) -> list[cf.Future]:
        tickets = [cf.Future() for _ in range(count)]
        ids = [next(self._ids) for _ in tickets]
        with self._lock:
            self._check_connected()
            self._waiting.update(zip(ids, tickets))
            self._reserved.update(zip(tickets, ids))
        self.requests[0].put(("admit", self.fid, ids))
        return tickets

    def release(self, ticket: cf.Future) -> None:
        with self._lock:
            rid = self._reserved.pop(ticket, None)
            if rid is None:
                return
            self._waiting.pop(rid, None)
        ticket.cancel()
        self.requests[0].put(("release", self.fid, rid))

    def flush_sessions(self) -> cf.Future:
        """Empty the session cache of every replica; its stats after (a list with several replicas), or None without
        one."""
        return self._control("flush")

    def stats(self, *, memory_details: bool = False) -> cf.Future:
        """The GPU process's numbers, one dict per replica it runs (a list when there are several)."""
        return self._control("stats", True) if memory_details else self._control("stats")

    def _control(self, command: str, *options) -> cf.Future:
        calls = {next(self._ids): cf.Future() for _ in self.requests}
        with self._lock:
            self._check_connected()
            self._waiting.update(calls)
        for outbox, rid in zip(self.requests, calls):
            outbox.put((command, self.fid, rid, *options))
        futs = list(calls.values())
        if len(futs) == 1:
            return futs[0]
        out: cf.Future = cf.Future()
        pending = set(futs)
        lock = threading.Lock()

        def done(future) -> None:
            with lock:
                pending.remove(future)
                if pending:
                    return
            if out.set_running_or_notify_cancel():
                try:
                    results = [f.result() for f in futs]
                except Exception as exc:
                    out.set_exception(exc)
                else:
                    out.set_result(results)

        for f in futs:
            f.add_done_callback(done)
        return out

    def _read(self):
        while True:
            try:
                rid, result, error = self.replies.get()
            except EOFError:
                self._disconnect()
                return
            with self._lock:
                fut = self._waiting.pop(rid, None)
                sent = self._sent.pop(fut, None)
            if fut is None or not fut.set_running_or_notify_cancel():
                continue
            if sent is not None and self.router is not None:
                _, r = sent
                self.router.done(r)
                if error is None:
                    result[1]["replica"] = r
            if error is None:
                fut.set_result(result)
            else:
                fut.set_exception(RuntimeError(error))


def run_frontend(fid: int, model_dir: str, family_name: str, family_options: dict, model_name: str, requests, replies,
                 host: str, port: int, planners: int, access_log: bool, ready=(), route=None, admission=False,
                 api_key: str | None = None, include_meta: bool = True) -> None:
    import uvicorn

    from .. import formats
    from .route import Router
    from .server import create_app

    family = formats.get(family_name)(model_dir, **family_options)
    router = Router(*route[:2], fid, len(requests), *route[2:]) if route is not None else None
    server = None

    def stop_accepting():
        if server is not None:
            server.should_exit = True

    bridge = RemoteBatcher(family, fid, [_PipeOutbox(writer) for writer in requests], _ReplyInbox(replies), planners,
                           router, on_disconnect=stop_accepting)
    app = create_app(types.SimpleNamespace(format=family), model_name=model_name, batcher=bridge,
                     ready=lambda: bridge.connected() and all(e.is_set() for e in ready), admission=bridge if admission else None,
                     api_key=api_key, include_meta=include_meta)
    config = uvicorn.Config(app, log_level="warning", access_log=access_log, backlog=2048)
    server = uvicorn.Server(config)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind((host, port))
        if bridge.connected():
            server.run(sockets=[sock])


def dispatch(batcher, requests, replies, admission=None) -> None:
    """GPU process: plans from the front ends into the batcher, head outputs back to the front end that asked."""

    pending: list[dict[int, cf.Future]] = [{} for _ in replies]
    pending_lock = threading.Lock()
    reserved: list[dict[int, cf.Future]] = [{} for _ in replies]
    planning = [0 for _ in replies]
    dead: set[int] = set()

    def send(fid, message):
        if fid not in dead:
            replies[fid].put(message)

    def reclaim_slots(fid):
        """After a frontend dies, its remaining forwards each need one slot until collected."""
        with pending_lock:
            outstanding = len(pending[fid])
        tickets = reserved[fid]
        for _ in range(len(tickets) - outstanding):
            _, ticket = tickets.popitem()
            admission.release(ticket)

    def answer(f, fid, rid):
        with pending_lock:
            pending[fid].pop(rid)
            if fid in dead:
                requests.put(("reclaim", fid))
                return
        if f.cancelled():
            send(fid, (rid, None, "request cancelled"))
            return
        exc = f.exception()
        send(fid, (rid, None, f"{type(exc).__name__}: {exc}") if exc is not None else (rid, f.result(), None))

    while True:
        msg = requests.get()
        command, fid = msg[:2]
        if command == "reclaim":
            reclaim_slots(fid)
        elif fid in dead:
            continue
        elif command == "frontend_exit":
            with pending_lock:
                dead.add(fid)
                outstanding = list(pending[fid].values())
            batcher._planned(-planning[fid])
            planning[fid] = 0
            for rid, ticket in list(reserved[fid].items()):
                if not ticket.done():
                    del reserved[fid][rid]
                    admission.release(ticket)
            for future in outstanding:
                batcher.cancel_forward(future)
            reclaim_slots(fid)
            if len(dead) == len(replies):
                return
        elif command == "begin":
            planning[fid] += 1
            batcher._planned(1)
        elif command == "fail":
            planning[fid] -= 1
            batcher._planned(-1)
        elif command == "admit":
            _, fid, ids = msg
            for rid, ticket in zip(ids, admission.reserve(len(ids))):
                if not ticket.done() or ticket.result():
                    reserved[fid][rid] = ticket
                ticket.add_done_callback(lambda f, fid=fid, rid=rid: send(fid, (rid, f.result(), None)))
        elif command == "release":
            _, fid, rid = msg
            ticket = reserved[fid].pop(rid, None)
            if ticket is not None:
                admission.release(ticket)
        elif command == "cancel":
            _, fid, rid = msg
            with pending_lock:
                fut = pending[fid].get(rid)
            if fut is not None:
                batcher.cancel_forward(fut)
        elif command == "stats":
            _, fid, rid, *options = msg
            sessions = batcher.engine.sessions
            payload = batcher.stats(memory_details=True) if options and options[0] else batcher.stats()
            if admission is not None:
                payload.update(admission.stats())
            payload["sessions"] = sessions.stats() if sessions is not None else None
            replies[fid].put((rid, payload, None))
        elif command == "flush":
            _, fid, rid = msg
            sessions = batcher.engine.sessions
            if sessions is not None:
                sessions.clear()
            replies[fid].put((rid, sessions.stats() if sessions is not None else None, None))
        elif command == "plan":
            _, fid, rid, plan = msg
            planning[fid] -= 1
            fut = batcher.forward(plan)
            with pending_lock:
                pending[fid][rid] = fut
            fut.add_done_callback(lambda f, fid=fid, rid=rid: answer(f, fid, rid))


def _dispatch_channels(batcher, channels, admission=None) -> None:
    requests = queue.SimpleQueue()
    replies = [_PipeOutbox(writer) for _, writer in channels]

    def receive(fid, reader):
        try:
            with reader:
                while True:
                    requests.put(reader.recv())
        except (EOFError, OSError):
            pass
        finally:
            replies[fid].close()
            requests.put(("frontend_exit", fid))

    for fid, (reader, _) in enumerate(channels):
        threading.Thread(target=receive, args=(fid, reader), name="flide-ipc-recv", daemon=True).start()
    dispatch(batcher, requests, replies, admission)


def start(n: int, model_dir: str, family_name: str, family_options: dict, model_name: str, host: str, port: int,
          planners: int, access_log: bool, replicas: int = 1, slack: int = 8, admission: bool = False,
          api_key: str | None = None, include_meta: bool = True):
    """Start n front ends for `replicas` GPU processes (they serve /health as "loading" until every replica is up);
    returns what `serve` or `serve_replicas` needs."""
    import multiprocessing as mp

    from .route import shared

    ctx = mp.get_context("spawn")
    channels = [[] for _ in range(replicas)]
    ready = [ctx.Event() for _ in range(replicas)]
    route = (*shared(ctx, replicas, n), n, slack) if replicas > 1 else None
    procs = []
    for fid in range(n):
        requests, replies = [], []
        for replica_channels in channels:
            request_reader, request_writer = ctx.Pipe(duplex=False)
            reply_reader, reply_writer = ctx.Pipe(duplex=False)
            requests.append(request_writer)
            replies.append(reply_reader)
            replica_channels.append((request_reader, reply_writer))
        proc = ctx.Process(target=run_frontend, name=f"flide-front-{fid}", daemon=True,
                           args=(fid, model_dir, family_name, family_options, model_name, requests, replies, host, port,
                                 planners, access_log, ready, route, admission, api_key, include_meta))
        proc.start()
        procs.append(proc)
        for connection in requests + replies:
            connection.close()
    log.info("%d front ends for %d replica%s on %s:%d", n, replicas, "s" if replicas > 1 else "", host, port)
    return procs, channels, ready


def serve(batcher, fronts, admission=None) -> None:
    """Feed the batcher from the front ends `start` made until stopped."""
    procs, channels, ready = fronts
    ready[0].set()
    try:
        _dispatch_channels(batcher, channels[0], admission)
    finally:
        for p in procs:
            p.terminate()
    raise SystemExit("all HTTP front ends exited")


def run_replica(i: int, build, channels, ready) -> None:
    """One replica's GPU process: build its batcher (`build()`, which loads the model), then feed it."""
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s replica {i} %(name)s %(levelname)s %(message)s")
    batcher = build()
    ready.set()
    log.info("replica %d ready", i)
    _dispatch_channels(batcher, channels)
    raise SystemExit("all HTTP front ends exited")


def serve_replicas(build, gpus: list[str], fronts) -> None:
    """One GPU process per entry of `gpus` (the CUDA_VISIBLE_DEVICES it sees), each running `build` and serving the
    front ends `start` made, until one of them stops."""
    import multiprocessing as mp
    import os
    procs, channels, ready = fronts
    ctx = mp.get_context("spawn")
    workers = []
    before = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        for i, devices in enumerate(gpus):
            os.environ["CUDA_VISIBLE_DEVICES"] = devices   # a spawned process takes the environment of its start
            w = ctx.Process(target=run_replica, name=f"flide-replica-{i}", args=(i, build, channels[i], ready[i]))
            w.start()
            workers.append(w)
            for channel in channels[i]:
                for connection in channel:
                    connection.close()
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

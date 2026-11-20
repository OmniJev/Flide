"""Exercise online admission and cancellation with the real scheduler and a CPU stub model."""
import asyncio
import concurrent.futures as cf
import http.client
import json
import multiprocessing as mp
import os
import queue
import socket
import struct
import threading
import time
import weakref
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from flide.api.admission import Admission
from flide.api.frontend import RemoteBatcher, _dispatch_channels, dispatch, run_frontend, serve
from flide.api.schema import SystemOneResponse, Usage
from flide.api.server import _stage, create_app
from flide.runtime.online import Batcher


@dataclass
class _Plan:
    state: str
    tokens: int
    data: object = None


class _Engine:
    budget = 100
    backbone = SimpleNamespace(device="cpu")
    sessions = None
    name = "fake"
    aliases = ()
    meta_key = "meta"

    def __init__(self):
        self.format = self
        self.planned = []
        self.launched = []
        self.unblock = threading.Event()

    def for_thread(self):
        return self

    def plan(self, request, **options):
        self.planned.append(request.state)
        return _Plan(request.state, 1)

    def follow_up(self, request, plan, branches, **options):
        return None

    def finish(self, request, plan, branches, **options):
        return SystemOneResponse(model="fake", answers={}, usage=Usage(input_tokens=1, output_tokens=1)), {}

    def forward_launch(self, plans, *, mode):
        self.launched.append([plan.state for plan in plans])
        if any(plan.state == "hold" for plan in plans):
            assert self.unblock.wait(timeout=5)
        return plans

    def forward_collect(self, plans):
        return [None] * len(plans), ["packed"] * len(plans)


class _TracingBatcher(Batcher):
    def __init__(self, engine, **options):
        self.forwarded = {}
        super().__init__(engine, **options)

    def forward(self, plan):
        future = super().forward(plan)
        self.forwarded[plan.state] = future
        return future


def _request(app, body=None, *, path="/v1/systemone", method="POST", headers=()):
    messages = asyncio.Queue()
    messages.put_nowait({"type": "http.request", "body": json.dumps(body).encode()})
    status = None

    async def send(message):
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]

    async def run():
        await app({"type": "http", "method": method, "path": path, "headers": headers, "query_string": b""},
                  messages.get, send)
        return status

    return asyncio.create_task(run()), lambda: messages.put_nowait({"type": "http.disconnect"})


def _body(state):
    return {"state": state, "questions": {"flag": {"type": "noul", "instructions": "A question"}}}


async def _wait_for(predicate):
    while not predicate():
        await asyncio.sleep(0.001)


@pytest.mark.parametrize("remote", [False, True])
def test_online_limits_and_disconnect(monkeypatch, remote):
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    engine = _Engine()
    batcher = _TracingBatcher(engine, planners=1, gather_ms=0)
    admission = Admission(max_running=2, max_queued=1)
    if remote:
        queue_to_gpu = queue.Queue()
        replies = [queue.Queue(), queue.Queue()]
        threading.Thread(target=dispatch, args=(batcher, queue_to_gpu, replies, admission), daemon=True).start()
        bridges = [RemoteBatcher(engine, index, [queue_to_gpu], reply, planners=1)
                   for index, reply in enumerate(replies)]
        app, other = [create_app(engine, "fake", batcher=bridge, admission=bridge) for bridge in bridges]
    else:
        app = other = create_app(engine, "fake", batcher=batcher, admission=admission)

    async def scenario():
        first, disconnect_first = _request(app, _body("hold"))
        await _wait_for(lambda: engine.launched == [["hold"]])
        second, disconnect_second = _request(other, _body("gpu queue"))
        await _wait_for(lambda: "gpu queue" in batcher.forwarded)
        rejected, _ = _request(other, {"requests": [_body("too"), _body("many")]})
        assert await rejected == 429
        queued, disconnect_queued = _request(app, _body("admission queue"))
        await _wait_for(lambda: admission.stats() == {"running_requests": 2, "queued_requests": 1})
        overflow, _ = _request(other, _body("overflow"))
        assert await overflow == 429
        disconnect_queued()
        assert await queued == 499
        disconnect_second()
        assert await second == 499
        assert batcher.forwarded["gpu queue"].cancelled()
        await _wait_for(lambda: admission.stats() == {"running_requests": 1, "queued_requests": 0})
        replacement, _ = _request(other, _body("replacement"))
        await _wait_for(lambda: "replacement" in batcher.forwarded)
        disconnect_first()
        await asyncio.sleep(0.02)
        assert admission.stats()["running_requests"] == 2
        assert not first.done()
        engine.unblock.set()
        assert await first == 499
        assert await replacement == 200
        assert engine.planned == ["hold", "gpu queue", "replacement"]
        assert engine.launched == [["hold"], ["replacement"]]
        await _wait_for(lambda: admission.stats() == {"running_requests": 0, "queued_requests": 0})
        assert batcher.stats()["planning"] == 0

    try:
        asyncio.run(asyncio.wait_for(scenario(), timeout=3))
    finally:
        engine.unblock.set()


def test_batch_token_budget_preserves_oversized_requests(monkeypatch):
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    engine = _Engine()
    batcher = Batcher(engine, planners=1, max_round_tokens=10, gather_ms=0)
    batcher._planned(6)
    futures = [batcher.forward(_Plan("hold", 10))]
    try:
        futures += [batcher.forward(_Plan(str(tokens), tokens)) for tokens in (4, 5, 6, 14, 2)]
        engine.unblock.set()
        for future in futures:
            future.result(timeout=2)
        assert engine.launched == [["hold"], ["4", "5"], ["6"], ["14"], ["2"]]
    finally:
        engine.unblock.set()


def test_limited_frontend_rejects_requests_before_model_ready():
    engine = _Engine()
    admission = Admission(max_running=1, max_queued=0)
    app = create_app(engine, "fake", admission=admission, ready=lambda: False)

    async def check():
        response, _ = _request(app, _body("loading"))
        assert await response == 503
        assert admission.stats() == {"running_requests": 0, "queued_requests": 0}

    asyncio.run(asyncio.wait_for(check(), timeout=2))


def test_api_key_optional_and_checked_before_admission():
    engine = _Engine()
    admission = Admission(max_running=1, max_queued=0)
    admission.reserve(1)
    open_app = create_app(engine, "fake")
    protected_app = create_app(engine, "fake", admission=admission, api_key="test-key")

    async def check():
        async def status(app, path, headers=()):
            method = "POST" if path == "/v1/systemone" else "GET"
            response, _ = _request(app, _body("example"), path=path, method=method, headers=headers)
            return await response

        correct = [(b"authorization", b"Bearer test-key")]
        wrong = [(b"authorization", b"Bearer bad-key")]
        assert await status(open_app, "/health") == 200
        for path in ("/health", "/stats", "/v1/models", "/v1/systemone", "/flush_cache"):
            assert await status(protected_app, path) == 401
            assert await status(protected_app, path, wrong) == 401
        assert await status(protected_app, "/health", correct + wrong) == 401
        assert await status(protected_app, "/health", correct) == 200
        assert await status(protected_app, "/v1/systemone", correct) == 429

    asyncio.run(asyncio.wait_for(check(), timeout=2))


def _crashing_frontend(writer, reader, sent):
    writer.send(("admit", 0, [0, 1, 2, 3]))
    for rid in range(3):
        assert reader.recv() == (rid, True, None)
    for rid, state in ((4, "hold"), (5, "abandoned")):
        writer.send(("begin", 0))
        writer.send(("plan", 0, rid, _Plan(state, 1)))
    writer.send(("begin", 0))
    os.write(writer.fileno(), struct.pack("!i", 100) + b"incomplete frame")
    sent.set()
    threading.Event().wait(15)


def test_frontend_death_reclaims_slots_and_isolates_partial_messages(monkeypatch):
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    context = mp.get_context("spawn")
    dead_input, dead_writer = context.Pipe(duplex=False)
    dead_reader, dead_output = context.Pipe(duplex=False)
    live_input, live_writer = context.Pipe(duplex=False)
    live_reader, live_output = context.Pipe(duplex=False)
    sent = context.Event()
    process = context.Process(target=_crashing_frontend, args=(dead_writer, dead_reader, sent))
    process.start()
    dead_writer.close()
    dead_reader.close()
    engine = _Engine()
    batcher = _TracingBatcher(engine, planners=1, gather_ms=0, max_round_tokens=1)
    admission = Admission(max_running=3, max_queued=2)
    channels = [(dead_input, dead_output), (live_input, live_output)]
    threading.Thread(target=_dispatch_channels, args=(batcher, channels, admission), daemon=True).start()

    def reply():
        assert live_reader.poll(3), "the surviving front end did not receive a reply"
        return live_reader.recv()

    def stats_until(predicate):
        for revision in range(100):
            rid = 100 + revision
            live_writer.send(("stats", 1, rid))
            reply_id, stats, error = reply()
            assert reply_id == rid and error is None
            if predicate(stats):
                return stats
            time.sleep(0.01)
        pytest.fail(f"unexpected stats: {stats}")

    try:
        assert sent.wait(5)
        live_writer.send(("admit", 1, [40]))
        stats = stats_until(lambda stats: stats["queued_requests"] == 2 and stats["planning"] == 1 and
                            engine.launched == [["hold"]])
        assert stats["running_requests"] == 3
        process.kill()
        process.join(timeout=3)
        assert process.exitcode is not None and process.exitcode < 0

        assert reply() == (40, True, None)
        stats = stats_until(lambda stats: stats["queued_requests"] == 0 and stats["planning"] == 0)
        assert stats["running_requests"] == 2
        assert not batcher.forwarded["hold"].done()
        assert batcher.forwarded["abandoned"].cancelled()
        live_writer.send(("begin", 1))
        live_writer.send(("plan", 1, 41, _Plan("replacement", 1)))
        engine.unblock.set()
        reply_id, _, error = reply()
        assert reply_id == 41 and error is None
        live_writer.send(("release", 1, 40))
        stats = stats_until(lambda stats: stats["running_requests"] == stats["queued_requests"] == 0)
        assert stats["planning"] == 0
        assert engine.launched == [["hold"], ["replacement"]]
    finally:
        engine.unblock.set()
        if process.is_alive():
            process.kill()
        process.join(timeout=3)
        live_writer.close()
        live_reader.close()


@pytest.mark.parametrize("remote", [False, True])
def test_cancelled_planning_releases_input_before_worker_is_free(monkeypatch, remote):
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    engine = _Engine()
    started = threading.Event()

    def plan(request, **options):
        if request.state == "hold":
            started.set()
            assert engine.unblock.wait(timeout=5)
        return _Plan(request.state, 1)

    engine.plan = plan
    batcher = (RemoteBatcher(engine, 0, [queue.Queue()], queue.Queue(), planners=1) if remote else
               Batcher(engine, planners=1))
    held = batcher.plan(SimpleNamespace(state="hold"))

    class Input:
        state = "cancelled"

    try:
        assert started.wait(2)
        for _ in range(12):
            request = Input()
            request.data = bytearray(1024 * 1024)
            reference = weakref.ref(request)
            future = batcher.plan(request)
            del request
            assert reference() is not None
            assert future.cancel()
            assert reference() is None
        assert not held.done()
        next_request = batcher.plan(SimpleNamespace(state="replacement"))
        engine.unblock.set()
        batcher.discard_plan(held.result(timeout=2))
        batcher.discard_plan(next_request.result(timeout=2))
        if not remote:
            assert batcher.stats()["planning"] == 0
    finally:
        engine.unblock.set()


def _stub_http_frontend(requests, replies, ready, port):
    from flide import formats

    formats.get = lambda name: lambda model_dir: _Engine()
    run_frontend(0, "unused", "fake", {}, "fake", [requests], [replies], "127.0.0.1", port, 1, False, [ready])


def _http_status(port, path, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        if body is None:
            connection.request("GET", path)
        else:
            connection.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        response.read()
        return response.status
    finally:
        connection.close()


def test_gpu_disconnect_fails_waiting_http_calls_and_stops_frontend():
    context = mp.get_context("spawn")
    request_reader, request_writer = context.Pipe(duplex=False)
    reply_reader, reply_writer = context.Pipe(duplex=False)
    ready = context.Event()
    ready.set()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    proc = context.Process(target=_stub_http_frontend, args=(request_writer, reply_reader, ready, port))
    proc.start()
    request_writer.close()
    reply_reader.close()
    try:
        deadline = time.monotonic() + 5
        while True:
            try:
                assert _http_status(port, "/health") == 200
                break
            except ConnectionRefusedError:
                assert proc.is_alive() and time.monotonic() < deadline
                time.sleep(0.01)
        with cf.ThreadPoolExecutor(max_workers=2) as client:
            inference = client.submit(_http_status, port, "/v1/systemone", _body("pending"))
            for command in ("begin", "plan"):
                assert request_reader.poll(2)
                assert request_reader.recv()[0] == command
            waiting = client.submit(_http_status, port, "/stats")
            assert request_reader.poll(2)
            assert request_reader.recv()[0] == "stats"
            reply_writer.close()
            assert waiting.result(timeout=3) == 503
            assert inference.result(timeout=3) == 503
        proc.join(timeout=3)
        assert proc.exitcode == 0
        with pytest.raises(ConnectionRefusedError):
            _http_status(port, "/health")
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=3)
        reply_writer.close()
        request_reader.close()


def _exit_after_admitted(writer, reader):
    writer.send(("admit", 0, [0]))
    assert reader.recv() == (0, True, None)


def test_main_process_exits_when_last_frontend_closes(monkeypatch):
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    context = mp.get_context("spawn")
    request_reader, request_writer = context.Pipe(duplex=False)
    reply_reader, reply_writer = context.Pipe(duplex=False)
    proc = context.Process(target=_exit_after_admitted, args=(request_writer, reply_reader))
    proc.start()
    request_writer.close()
    reply_reader.close()
    admission = Admission(max_running=1, max_queued=0)
    outcome = cf.Future()

    def run_main():
        try:
            serve(Batcher(_Engine(), planners=1), ([proc], [[(request_reader, reply_writer)]], [threading.Event()]),
                  admission)
        except BaseException as exc:
            outcome.set_exception(exc)
        else:
            outcome.set_result(None)

    threading.Thread(target=run_main, daemon=True).start()
    try:
        with pytest.raises(SystemExit, match="all HTTP front ends exited"):
            outcome.result(timeout=5)
        assert admission.stats() == {"running_requests": 0, "queued_requests": 0}
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=3)


def test_repeated_cancellation_keeps_slot_until_gpu_finishes(monkeypatch):
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    engine = _Engine()
    batcher = _TracingBatcher(engine, planners=1)
    cancelling = threading.Event()
    original_cancel = batcher.cancel_forward

    def cancel_forward(future):
        original_cancel(future)
        cancelling.set()

    batcher.cancel_forward = cancel_forward
    admission = Admission(max_running=1, max_queued=0)
    app = create_app(engine, "fake", batcher=batcher, admission=admission)

    async def check():
        running, disconnect = _request(app, _body("hold"))
        await _wait_for(lambda: engine.launched == [["hold"]])
        disconnect()
        await _wait_for(cancelling.is_set)
        assert running.cancel()
        overflow, _ = _request(app, _body("overflow"))
        assert await overflow == 429
        assert not running.done()
        engine.unblock.set()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert admission.stats()["running_requests"] == 0
        replacement, _ = _request(app, _body("replacement"))
        assert await replacement == 200

    try:
        asyncio.run(asyncio.wait_for(check(), timeout=3))
    finally:
        engine.unblock.set()


def test_repeated_cancellation_discards_completed_plan():
    async def check():
        plan = cf.Future()
        assert plan.set_running_or_notify_cancel()
        discarded = []
        request = asyncio.create_task(_stage(plan, discard=discarded.append))
        await asyncio.sleep(0)
        for _ in range(2):
            request.cancel()
            await asyncio.sleep(0)
            assert not request.done()
        plan.set_result("planned")
        with pytest.raises(asyncio.CancelledError):
            await request
        assert discarded == ["planned"]

    asyncio.run(asyncio.wait_for(check(), timeout=2))

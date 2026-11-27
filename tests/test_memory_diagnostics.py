"""Allocator diagnostics stay read-only and report the serving device and graph pool."""
import asyncio
import json
import queue
import threading
from types import SimpleNamespace

import pytest
import torch

from flide.api.server import create_app
from flide.api.frontend import RemoteBatcher, dispatch
from flide.runtime.diagnostics import diagnostics


def engine(device="cuda:1"):
    graphs = SimpleNamespace(graphs={("flat", 128): object(), ("flat", 256): object(),
                                     ("sflat", 128): object()}, pool=(9, 1), failed={("seq", 64)})
    return SimpleNamespace(backbone=SimpleNamespace(device=device), _diagnostic_graphs=graphs,
                           sessions=None, format=SimpleNamespace(name="fake", aliases=(), meta_key="meta"))


@pytest.fixture
def cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostics must not synchronize, free memory, or reset counters")

    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    values = {"allocated_bytes.all.current": 100, "reserved_bytes.all.current": 400,
              "active_bytes.all.current": 120, "inactive_split_bytes.all.current": 30,
              "allocated_bytes.all.peak": 200, "reserved_bytes.all.peak": 500,
              "num_alloc_retries": 2, "num_ooms": 1}

    def stats(device):
        assert device == 1
        return values

    monkeypatch.setattr(torch.cuda, "memory_stats", stats)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (600, 1024) if device == 1 else forbidden())
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda device: .5)
    monkeypatch.setattr(torch.cuda, "memory_snapshot", forbidden)
    return monkeypatch


def test_lightweight_stats_do_not_snapshot_or_mutate(cuda):
    result = diagnostics(engine())
    assert result["memory"]["allocated_bytes"] == 100
    assert result["memory"]["reserved_unallocated_bytes"] == 300
    assert result["memory"]["device_used_bytes"] == 424
    assert result["memory"]["oom_count"] == 1
    assert result["memory"]["allocator_limit_bytes"] == 512
    assert result["cuda_graphs"] == {"count": 3, "by_kind": {"flat": 2, "sflat": 1},
                                     "failed_count": 1, "pool_id": [9, 1]}


@pytest.mark.parametrize("legacy", [False, True])
def test_pool_snapshot_filters_device_and_pool(cuda, legacy):
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        if legacy and kwargs:
            raise TypeError("older snapshot signature")
        return [dict(device=d, segment_pool_id=p, total_size=n, allocated_size=a, active_size=b)
                for d, p, n, a, b in [(1, (9, 1), 80, 20, 30), (1, (9, 1), 40, 10, 20),
                                      (0, (9, 1), 999, 999, 999), (1, (0, 0), 500, 100, 100)]]

    cuda.setattr(torch.cuda, "memory_snapshot", snapshot)
    result = diagnostics(engine(), memory_details=True)
    assert calls[0] == {"mempool_id": (9, 1), "include_traces": False}
    assert len(calls) == (2 if legacy else 1)
    assert result["cuda_graphs"]["pool_memory"] == {
        "source": "allocator_snapshot", "segments": 2, "reserved_bytes": 120,
        "allocated_bytes": 30, "active_bytes": 50, "inactive_bytes": 70}
    assert len(result["cuda_graphs"]["shapes"]) == 3


def test_cpu_does_not_touch_cuda_allocator(cuda):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU diagnostics must not query CUDA")

    cuda.setattr(torch.cuda, "memory_stats", forbidden)
    cuda.setattr(torch.cuda, "mem_get_info", forbidden)
    assert diagnostics(engine("cpu"))["memory"]["available"] is False


def test_stats_endpoint_passes_details_to_serving_batcher(cuda):
    model = engine()

    class Batcher:
        def stats(self, *, memory_details=False):
            return {"requests": 7, **diagnostics(model, memory_details=memory_details)}

    cuda.setattr(torch.cuda, "memory_snapshot", lambda **kwargs: [])
    app = create_app(model, model_name="m", batcher=Batcher())
    endpoint = next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/stats")
    result = json.loads(asyncio.run(endpoint(memory_details=True)).body)
    assert result["requests"] == 7
    assert result["cuda_graphs"]["pool_memory"]["reserved_bytes"] == 0
    assert result["memory"]["device"] == "cuda:1"


def test_details_survive_frontend_control_channel(cuda):
    model = engine()
    cuda.setattr(torch.cuda, "memory_snapshot", lambda **kwargs: [])

    class Batcher:
        engine = model

        def _planned(self, n):
            assert n == 0

        def stats(self, *, memory_details=False):
            return diagnostics(model, memory_details=memory_details)

    class Inbox(queue.Queue):
        def get(self):
            value = super().get()
            if value is None:
                raise EOFError
            return value

    family = SimpleNamespace(for_thread=lambda: family)
    requests, replies = queue.Queue(), Inbox()
    worker = threading.Thread(target=dispatch, args=(Batcher(), requests, [replies]), daemon=True)
    worker.start()
    remote = RemoteBatcher(family, 0, [requests], replies, planners=1)
    try:
        assert "pool_memory" not in remote.stats().result(timeout=2)["cuda_graphs"]
        result = remote.stats(memory_details=True).result(timeout=2)
        assert result["cuda_graphs"]["count"] == 3
        assert result["cuda_graphs"]["pool_memory"]["reserved_bytes"] == 0
    finally:
        requests.put(("frontend_exit", 0))
        worker.join(timeout=2)
        replies.put(None)
        remote._planners._pool.shutdown(wait=True)

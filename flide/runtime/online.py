"""Online serving: many clients, one GPU.

Requests are planned on a pool of CPU threads (each with its own copy of the family, since tokenizers keep
mutable state), then queued for the GPU thread. The GPU thread takes everything waiting, optionally after
a short window, and runs it as one round:

A round runs as one replay of a flat graph: every prefix and branch of the round in one stream with no padding
between them, graphs captured at start; a request alone runs this way too. Rounds that do not fit the flat graphs
run as a packed eager forward.

Head outputs go back to the request's thread, which finishes the answer. Planning and finishing of other
requests overlap the GPU round.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("flide.online")


def start_planners(pool: cf.ThreadPoolExecutor, n: int) -> None:
    """Start all n threads of a planner pool now. Each copies the family (its tokenizer) when it starts, a few hundred
    milliseconds that would otherwise stall the requests that first need it once load grows."""
    barrier = threading.Barrier(n + 1)
    for _ in range(n):
        pool.submit(barrier.wait)
    barrier.wait()


@dataclass
class _Item:
    plan: Any
    future: cf.Future
    t_in: float = field(default_factory=time.perf_counter)


class Batcher:
    def __init__(self, engine, planners: int = 4, window_ms: float = 0.0, max_round_tokens: int | None = None,
                 gather_ms: float | None = None):
        self.engine = engine
        self.window = window_ms / 1000
        # while requests are still being planned, a round waits up to `gather_ms` for them (under load the answers
        # of a round go out together and the next requests come back one by one; taking only what is queued would
        # split them into rounds too small to batch)
        self.gather = (2.0 if gather_ms is None else gather_ms) / 1000
        self._planning = 0
        self._count = threading.Lock()
        self.max_round_tokens = max_round_tokens or engine.budget
        graphs = getattr(engine.backbone, "request_graphs", lambda: None)()
        if graphs is not None and hasattr(graphs, "capture_flat"):
            graphs.capture_flat()
            # a round is at most one flat replay: under load the next round is queued while this one runs, so small
            # rounds keep requests in arrival order instead of waiting behind several replays
            if max_round_tokens is None:
                self.max_round_tokens = graphs.flat_max
        self._carry: _Item | None = None   # the request that did not fit the last round: first in the next
        self.q: queue.Queue[_Item] = queue.Queue()
        self._local = threading.local()
        self._planners = cf.ThreadPoolExecutor(max_workers=planners, thread_name_prefix="flide-plan",
                                               initializer=self._init_planner)
        start_planners(self._planners, planners)
        self.rounds = 0
        self.requests = 0
        self.queue_s = 0.0
        self.forward_s = 0.0
        self.max_round = 0
        self._thread = threading.Thread(target=self._loop, name="flide-gpu", daemon=True)
        self._thread.start()

    def _init_planner(self):
        self._local.format = self.engine.format.for_thread()

    def _format(self):
        f = getattr(self._local, "format", None)
        return f if f is not None else self.engine.format

    # ---- caller side ----------------------------------------------------------------------------

    def _planned(self, n: int) -> None:
        with self._count:
            self._planning += n

    def plan(self, request, **options) -> cf.Future:
        """Plan a request; the caller passes the plan to `forward` (a plan that fails is no longer counted)."""
        self._planned(1)
        fut = self._planners.submit(lambda: self._format().plan(request, **options))
        fut.add_done_callback(lambda f: f.exception() is not None and self._planned(-1))
        return fut

    def follow_up(self, request, plan, branches, **options) -> cf.Future:
        """The family's next plan for a request after `plan`, or None; a next plan is counted until its `forward`."""
        def run():
            nxt = self._format().follow_up(request, plan, branches, **options)
            if nxt is not None:
                self._planned(1)
            return nxt

        return self._planners.submit(run)

    def finish(self, request, plan, branches, **options) -> cf.Future:
        return self._planners.submit(lambda: self._format().finish(request, plan, branches, **options))

    def forward(self, plan) -> cf.Future:
        """Head outputs of one plan, as `Format.finish` takes them."""
        item = _Item(plan, cf.Future())
        self.q.put(item)
        self._planned(-1)
        return item.future

    def stats(self) -> dict:
        """What this batcher has served since start, and where the GPU thread's time went."""
        return {"rounds": self.rounds, "requests": self.requests, "planning": self._planning,
                "queue_s": round(self.queue_s, 3), "forward_s": round(self.forward_s, 3),
                "max_round": self.max_round, "avg_round": round(self.requests / self.rounds, 2) if self.rounds else 0.0}

    # ---- GPU thread ------------------------------------------------------------------------------

    def _take(self) -> list[_Item]:
        if self._carry is not None:
            items, self._carry = [self._carry], None
        else:
            items = [self.q.get()]
        tokens = items[0].plan.tokens
        start = time.perf_counter()
        while tokens < self.max_round_tokens:
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                waited = time.perf_counter() - start
                limit = max(self.window, self.gather if self._planning > 0 else 0.0)
                if waited >= limit:
                    break
                try:
                    item = self.q.get(timeout=min(limit - waited, 0.0005))
                except queue.Empty:
                    continue
            if tokens + item.plan.tokens > self.max_round_tokens:
                self._carry = item
                break
            items.append(item)
            tokens += item.plan.tokens
        return items

    def _loop(self):
        import torch

        torch.cuda.set_device(self.engine.backbone.device)
        inflight = None   # (items, launched round, start): queued on the GPU, outputs not yet collected
        while True:
            if inflight is not None and self._carry is None and self.q.empty():
                self._deliver(*inflight)
                inflight = None
                continue
            items = self._take()
            t0 = time.perf_counter()
            try:
                launched = self.engine.forward_launch([it.plan for it in items], mode="batch")
            except Exception as exc:  # one bad round must not take the server down
                log.exception("round of %d requests failed", len(items))
                for it in items:
                    it.future.set_exception(exc)
                launched = None
            # the next round is on the GPU behind this one: now wait for the previous round's outputs
            if inflight is not None:
                self._deliver(*inflight)
            inflight = (items, launched, t0) if launched is not None else None

    def _deliver(self, items, launched, t0):
        try:
            outs, paths = self.engine.forward_collect(launched)
        except Exception as exc:
            log.exception("round of %d requests failed", len(items))
            for it in items:
                it.future.set_exception(exc)
            return
        dt = time.perf_counter() - t0
        self.rounds += 1
        self.requests += len(items)
        self.queue_s += sum(t0 - it.t_in for it in items)
        self.forward_s += dt
        self.max_round = max(self.max_round, len(items))
        for it, out, path in zip(items, outs, paths):
            it.future.set_result((out, {"path": path, "round": len(items), "queue_s": t0 - it.t_in,
                                        "forward_s": dt}))

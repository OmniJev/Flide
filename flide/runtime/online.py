"""Online serving: many clients, one GPU.

Requests are planned on a pool of CPU threads (each with its own copy of the family, since tokenizers keep
mutable state), then queued for the GPU thread. The GPU thread takes everything waiting, optionally after
a short window, and runs it as one round:

A round runs as one replay of a flat graph: every prefix and branch of the round in one stream with no padding
between them, graphs captured at start; a request alone runs this way too. Rounds that do not fit the flat graphs
run as a packed eager forward.

With the session cache, a request that resumes a saved state (an agent's next step) computes a small part of its
tokens. The GPU thread then takes the waiting requests in order of when each would be done if it ran alone from its
arrival (arrival plus its tokens to compute at the rate the GPU has computed so far), so such a request goes ahead of
long ones that came shortly before it, and a long one waits at most about its own run time longer than in arrival
order (`_take_ordered`).

Head outputs go back to the request's thread, which finishes the answer. Planning and finishing of other
requests overlap the GPU round.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("flide.online")


class PlannerPool:
    """CPU work that can be removed from the waiting queue as soon as its future is cancelled."""

    def __init__(self, workers: int, initializer: Callable):
        self._workers = workers
        self._draining = 0
        self._lock = threading.Lock()
        self._pending: OrderedDict[cf.Future, Callable] = OrderedDict()
        self._pool = cf.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="flide-plan", initializer=initializer)
        barrier = threading.Barrier(workers + 1)
        for _ in range(workers):
            self._pool.submit(barrier.wait)
        barrier.wait()

    def submit(self, work: Callable) -> cf.Future:
        future: cf.Future = cf.Future()
        future.add_done_callback(self._discard)
        with self._lock:
            self._pending[future] = work
            if self._draining < self._workers:
                try:
                    self._pool.submit(self._drain)
                except RuntimeError:
                    del self._pending[future]
                    raise
                self._draining += 1
        return future

    def _discard(self, future: cf.Future) -> None:
        if future.cancelled():
            with self._lock:
                self._pending.pop(future, None)

    def _drain(self) -> None:
        while True:
            with self._lock:
                if not self._pending:
                    self._draining -= 1
                    return
                future, work = self._pending.popitem(last=False)
            if future.set_running_or_notify_cancel():
                try:
                    result = work()
                except BaseException as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(result)
                    del result
            del future, work


@dataclass
class _Item:
    plan: Any
    future: cf.Future
    t_in: float = field(default_factory=time.perf_counter)
    cost: int = 0   # tokens to compute (`Engine.cost`), set when the GPU thread first sees it in an ordered batcher


class _Pending:
    def __init__(self):
        self._condition = threading.Condition()
        self._items: OrderedDict[cf.Future, _Item] = OrderedDict()

    def put(self, item: _Item) -> None:
        with self._condition:
            self._items[item.future] = item
            self._condition.notify()

    def discard(self, future: cf.Future) -> None:
        with self._condition:
            self._items.pop(future, None)

    def get(self, timeout: float | None = None) -> _Item:
        with self._condition:
            if not self._condition.wait_for(lambda: self._items, timeout):
                raise queue.Empty
            return self._items.popitem(last=False)[1]


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
        self.ordered = getattr(engine, "sessions", None) is not None
        self._waiting: list[_Item] = []   # requests the ordered GPU thread has taken off the queue, with their cost
        self._computed = 0                # tokens the rounds so far computed, and the GPU time they took
        self._busy = 0.0
        self._last = 0.0
        self.q = _Pending()
        self._local = threading.local()
        self._planners = PlannerPool(planners, self._init_planner)
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
        fut.add_done_callback(lambda f: (f.cancelled() or f.exception() is not None) and self._planned(-1))
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

    def discard_plan(self, plan) -> None:
        self._planned(-1)

    def forward(self, plan) -> cf.Future:
        """Head outputs of one plan, as `Format.finish` takes them."""
        item = _Item(plan, cf.Future())
        item.future.add_done_callback(lambda f: f.cancelled() and self.q.discard(f))
        self.q.put(item)
        self._planned(-1)
        return item.future

    def cancel_forward(self, future: cf.Future) -> None:
        future.cancel()

    def stats(self) -> dict:
        """What this batcher has served since start, and where the GPU thread's time went."""
        return {"rounds": self.rounds, "requests": self.requests, "planning": self._planning,
                "queue_s": round(self.queue_s, 3), "forward_s": round(self.forward_s, 3),
                "max_round": self.max_round, "avg_round": round(self.requests / self.rounds, 2) if self.rounds else 0.0}

    # ---- GPU thread ------------------------------------------------------------------------------

    def _take(self, wait: bool) -> list[_Item]:
        if self._carry is not None:
            items, self._carry = [self._carry], None
        else:
            try:
                items = [self.q.get(timeout=None if wait else 0)]
            except queue.Empty:
                return []
        tokens = items[0].plan.tokens
        start = time.perf_counter()
        while tokens < self.max_round_tokens:
            try:
                item = self.q.get(timeout=0)
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

    def _pull(self, timeout: float | None = None) -> None:
        """Move the queued requests to the waiting list with their cost: waits for one up to `timeout` (None: as long
        as it takes, 0: not at all)."""
        try:
            item = self.q.get(timeout=timeout)
        except queue.Empty:
            return
        while True:
            item.cost = self.engine.cost(item.plan)
            self._waiting.append(item)
            try:
                item = self.q.get(timeout=0)
            except queue.Empty:
                return

    def _take_ordered(self, wait: bool) -> list[_Item]:
        """A round from the waiting requests, the soonest done first (arrival plus cost over the rate computed so far;
        arrival order until a round has run). A request costing more than twice what the round holds so far waits
        for the next round, so a long one does not hold back the short ones ahead of it. Requests cancelled while
        they waited leave the list; without `wait`, nothing waiting gives an empty round."""
        if not self._waiting:
            self._pull(None if wait else 0)
        start = time.perf_counter()
        while sum(it.plan.tokens for it in self._waiting) < self.max_round_tokens:
            self._pull(0)
            waited = time.perf_counter() - start
            limit = max(self.window, self.gather if self._planning > 0 else 0.0)
            if waited >= limit:
                break
            self._pull(min(limit - waited, 0.0005))
        self._waiting = [it for it in self._waiting if not it.future.cancelled()]
        rate = self._computed / self._busy if self._busy > 0 else 0.0
        self._waiting.sort(key=(lambda it: it.t_in + it.cost / rate) if rate > 0 else (lambda it: it.t_in))
        n = tokens = cost = 0
        for it in self._waiting:
            if n and (tokens + it.plan.tokens > self.max_round_tokens or it.cost > 2 * cost):
                break
            n += 1
            tokens += it.plan.tokens
            cost += it.cost
        items, self._waiting = self._waiting[:n], self._waiting[n:]
        return items

    def _loop(self):
        import torch

        torch.cuda.set_device(self.engine.backbone.device)
        inflight = None   # (items, launched round, start): queued on the GPU, outputs not yet collected
        while True:
            wait = inflight is None
            items = self._take_ordered(wait) if self.ordered else self._take(wait)
            items = [item for item in items if item.future.set_running_or_notify_cancel()]
            if not items:
                if inflight is not None:
                    self._deliver(*inflight)
                    inflight = None
                continue
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
        now = time.perf_counter()
        dt = now - t0
        if self.ordered:   # GPU time of this round: from its launch or the previous round's end, whichever is later
            self._busy += now - max(t0, self._last)
            self._computed += sum(it.cost for it in items)
            self._last = now
        self.rounds += 1
        self.requests += len(items)
        self.queue_s += sum(t0 - it.t_in for it in items)
        self.forward_s += dt
        self.max_round = max(self.max_round, len(items))
        for it, out, path in zip(items, outs, paths):
            it.future.set_result((out, {"path": path, "round": len(items), "queue_s": t0 - it.t_in,
                                        "forward_s": dt}))

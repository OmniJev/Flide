"""Admission slots for logical requests, shared by all HTTP front ends of a server."""
from __future__ import annotations

import concurrent.futures as cf
import threading
from collections import OrderedDict


class Admission:
    def __init__(self, max_running: int, max_queued: int | None):
        self.max_running = max_running
        self.max_queued = max_queued
        self._lock = threading.Lock()
        self._running: set[cf.Future] = set()
        self._queued: OrderedDict[cf.Future, None] = OrderedDict()

    def reserve(self, count: int) -> list[cf.Future]:
        """Reserve all items atomically. A ticket resolves when it starts or is rejected."""
        tickets = [cf.Future() for _ in range(count)]
        with self._lock:
            available = self.max_running - len(self._running)
            if self.max_queued is not None and count > available + self.max_queued - len(self._queued):
                for ticket in tickets:
                    ticket.set_result(False)
                return tickets
            for ticket in tickets[:available]:
                self._running.add(ticket)
                ticket.set_result(True)
            self._queued.update(dict.fromkeys(tickets[available:]))
        return tickets

    def release(self, ticket: cf.Future) -> None:
        with self._lock:
            if ticket not in self._running:
                if ticket in self._queued:
                    del self._queued[ticket]
                    ticket.set_result(False)
                return
            self._running.remove(ticket)
            if self._queued:
                following, _ = self._queued.popitem(last=False)
                self._running.add(following)
                following.set_result(True)

    def stats(self) -> dict:
        with self._lock:
            return {"running_requests": len(self._running), "queued_requests": len(self._queued)}

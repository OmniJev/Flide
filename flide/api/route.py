"""Which replica answers a request, when one copy of the model runs on each of several GPUs.

A replica keeps the saved states of the requests it served (its session cache), so a request whose state extends one
that a replica served resumes there, as an agent's next step does. Every front end hashes the start of a request's
state at each whole stride of characters and looks up the deepest of these prefixes in a table that holds, for each,
the replica that last served it and whether a request ever went on past it with other text.

A deepest prefix that no request went past is where an earlier request ended, so this request continues it, and it
goes to that replica unless the replica holds more than `slack` requests in flight beyond the least busy one. A
deepest prefix that some request went past is an opening that several requests share, such as an agent's system
prompt, and the request goes to the least busy replica (ties taken in turn), so agents that share an opening spread
over the replicas, each of which then saves the opening once. The table and the counts of requests in flight live in shared
memory, each front end counting in a row of its own, so every front end routes alike with no message on the way.
"""
from __future__ import annotations

import hashlib
import itertools
import json

STRIDE = 1024
SLOTS = 1 << 20
_TAG = 20


def _text(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def marks(state, stride: int = STRIDE) -> list[int]:
    """Keys of the state's prefixes at every whole stride, the deepest last."""
    data = _text(state).encode()
    h = hashlib.blake2b(digest_size=8)
    out = []
    for i in range(0, len(data) - stride + 1, stride):
        h.update(data[i:i + stride])
        out.append(int.from_bytes(h.digest(), "little") >> 1)
    return out


def shared(ctx, replicas: int, fronts: int):
    """The shared table and in-flight counts for `Router`s in `fronts` processes."""
    return ctx.Array("q", SLOTS, lock=False), ctx.Array("i", replicas * fronts, lock=False)


class Router:
    def __init__(self, table, loads, fid: int, replicas: int, fronts: int, slack: int = 8, stride: int = STRIDE):
        self.table, self.loads = table, loads
        self.fid, self.n, self.fronts = fid, replicas, fronts
        self.slack, self.stride = slack, stride
        self._turn = itertools.count(fid)

    def _get(self, key: int) -> tuple[int, bool] | None:
        """(replica, whether a request went on past this prefix), or None when no replica served it."""
        v = self.table[key & (SLOTS - 1)]
        return ((v & 0xFF) - 1, bool(v >> 8 & 1)) if v and v >> 9 == key >> _TAG else None

    def _put(self, key: int, r: int, past: bool) -> None:
        self.table[key & (SLOTS - 1)] = (key >> _TAG) << 9 | int(past) << 8 | (r + 1)

    def load(self) -> list[int]:
        return [sum(self.loads[f * self.n + r] for f in range(self.fronts)) for r in range(self.n)]

    def pick(self, state) -> int:
        """The replica for a request with this state, counted in flight until `done`."""
        keys = marks(state, self.stride)
        load = self.load()
        t = next(self._turn)
        r = min(range(self.n), key=lambda i: (load[i], (i - t) % self.n))
        for k in reversed(keys):
            hit = self._get(k)
            if hit is not None and hit[0] < self.n:
                owner, shared = hit
                if not shared and load[owner] <= load[r] + self.slack:
                    r = owner
                break
        for i, k in enumerate(keys):
            past = i < len(keys) - 1
            if not past:
                old = self._get(k)
                past = old is not None and old[1]
            self._put(k, r, past)
        self.loads[self.fid * self.n + r] += 1
        return r

    def done(self, r: int) -> None:
        self.loads[self.fid * self.n + r] -= 1

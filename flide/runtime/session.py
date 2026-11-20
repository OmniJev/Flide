"""Session cache: saved backbone states at token positions of earlier prefixes.

An agent asks about a growing state step after step, so most of each prefix is the previous step's prefix.
After a forward, a job's state is saved at a few positions of its prefix (marks): the last stride multiple a
margin before its end, and its next-to-last token. A later request whose prefix starts with the same tokens
resumes from the furthest saved position it matches and computes only the rest.

For qwen3_5 a saved state is exact: the Gated DeltaNet recurrent state in float32 and the convolution window
at the position, plus the full-attention keys and values before it. Nothing is approximated, so a resumed
request computes the same function as a fresh one (rounding differs as it does between any two packings).

Lookup hashes the prefix in blocks of 64 tokens, so a probe costs one pass over the request's tokens; a hit is
confirmed token by token. An entry saved by a request that resumed from another, over the same tokens and further,
replaces that one: an agent's step holds every key and value its previous step held, so one agent keeps one entry.

The cache has two segments. A new entry goes to *probation*, a share of the budget; a hit promotes it to
*protected*, and while protected is over its share the least recently used of it falls back to probation. Eviction
takes probation first. A request seen once therefore sits in probation and cannot push out a prefix that keeps
coming back, while a prefix seen twice hits while still in probation -- unlike admitting only a second sighting,
which would compute it twice.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict


BLOCK = 64
PROTECTED = 0.75   # of the budget that entries a request has hit may hold; the rest is probation


def _bytes(x) -> int:
    if isinstance(x, dict):
        return sum(_bytes(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(_bytes(v) for v in x)
    return x.numel() * x.element_size() if hasattr(x, "numel") else 0


def reserve(nbytes: int, device) -> None:
    """Have the CUDA caching allocator hold `nbytes` as one block, so cache entries are carved from it: the cache
    grows by an entry per request, and a fresh device allocation can stall for 100s of ms."""
    import torch

    if nbytes > 0 and torch.device(device).type == "cuda":
        torch.empty(nbytes, dtype=torch.uint8, device=device)


KEEP = 0.10   # of the device's memory that an "auto" cache leaves free, for requests larger than the warm-up's


def cache_size(text: str) -> float | str:
    """A --session-cache-gb value: a size in GiB, or "auto"."""
    if str(text).strip().lower() == "auto":
        return "auto"
    try:
        gib = float(text)
    except ValueError:
        gib = -1.0
    if not gib >= 0:
        raise argparse.ArgumentTypeError(f"--session-cache-gb takes 'auto' or a size in GiB, not {text!r}")
    return gib


def auto_bytes(free: int, total: int, reserved: int, allocated: int, peak: int, keep: float = KEEP) -> int:
    """Bytes an "auto" session cache takes: what the device has left once the weights, the graphs and the warm-up's
    peak work space are in (free on the device plus what the caching allocator holds unused, less that work space),
    less `keep` of the device's memory; 0 when that is under 512 MiB."""
    room = free + (reserved - allocated) - (peak - allocated) - int(keep * total)
    return room if room >= 2 ** 29 else 0


class Entry:
    """A prefix and the backbone state at some of its positions: states[pos] holds what the backbone needs to
    continue from pos (for qwen3_5: "S" and "conv" per linear-attention layer); kv[layer] the full-attention keys
    and values of the whole prefix."""

    def __init__(self, tokens, states: dict, kv: dict):
        self.tokens = list(tokens)
        self.states = states
        self.kv = kv
        self.bytes = _bytes(states) + _bytes(kv)


class _Hashes:
    """Chained block hashes of a token list: key(pos) identifies tokens[:pos]."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.blocks = [0]
        h = 0
        for k in range(len(tokens) // BLOCK):
            h = hash((h, tuple(tokens[k * BLOCK: (k + 1) * BLOCK])))
            self.blocks.append(h)

    def key(self, pos: int):
        k = pos // BLOCK
        return pos, hash((self.blocks[k], tuple(self.tokens[k * BLOCK: pos])))


class SessionCache:
    def __init__(self, budget_bytes: int, min_tokens: int = 2048, stride: int = 256, margin: int = 32):
        self.budget = budget_bytes
        self.min_tokens = min_tokens
        self.stride = stride
        self.margin = margin
        self.entries: OrderedDict[int, Entry] = OrderedDict()
        self.index: dict[tuple, int] = {}
        self.positions: dict[int, int] = {}
        self.used = 0
        self._next = 0
        self.hits = self.misses = self.saved_tokens = 0
        self.evictions = self.replaced = 0
        self.probation: set[int] = set()   # entries no request has hit yet
        self.pbytes = 0                    # bytes the probation entries hold

    @property
    def hbudget(self) -> int:
        """Bytes the protected entries may hold, read from the budget when used: an engine with an automatic size
        sets the budget after it builds the cache."""
        return int(self.budget * PROTECTED)

    def mark(self, n_prefix: int, start: int = 0) -> int | None:
        """Where to save the state of a prefix of n tokens that starts computing at `start`: the last stride multiple
        a margin before its end, so an agent's next step (the same text with more after it) and a repeat of the same
        prefix with other questions both find it."""
        if n_prefix < self.min_tokens:
            return None
        m = (n_prefix - self.margin) // self.stride * self.stride
        return m if start < m < n_prefix else None

    def marks(self, n_prefix: int, start: int = 0) -> tuple[int, ...]:
        """The last stride mark and the one before it. The last is where a request that only extends this prefix
        resumes from; the earlier one is what a request that shares the prefix but diverges before the last mark
        still matches -- a shared preamble with a different tail, which otherwise hits nothing and recomputes the
        whole trunk. Both are stride multiples, so the graph path's ladder could carry them too."""
        if n_prefix < self.min_tokens:
            return ()
        m = self.mark(n_prefix, start)
        if m is None:
            return ()
        out = [m]
        previous = m - self.stride
        if previous > start:
            out.insert(0, previous)
        return tuple(out)

    def match(self, tokens) -> tuple[Entry | None, int]:
        """The entry and position to resume `tokens` (a prefix) from: the furthest saved position, at most one
        token before the end, whose tokens all match."""
        if len(tokens) < self.min_tokens or not self.entries:
            self.misses += 1
            return None, 0
        hashes = _Hashes(tokens)
        for pos in sorted((p for p in self.positions if p < len(tokens)), reverse=True):
            eid = self.index.get(hashes.key(pos))
            if eid is None:
                continue
            entry = self.entries[eid]
            if entry.tokens[:pos] == tokens[:pos]:
                self.entries.move_to_end(eid)
                self._promote(eid)
                self.hits += 1
                self.saved_tokens += pos
                return entry, pos
        self.misses += 1
        return None, 0

    def insert(self, entry: Entry, parent: Entry | None = None) -> None:
        """Keep `entry`, evicting the least recently used ones it needs room from; `parent`, the entry the request
        that saved it resumed from, goes first when `entry` covers its tokens.

        A new entry enters probation unless the request resumed from a cached entry: that prefix has already been
        reused once, so it goes to protected and one-off requests cannot push it out. Protected is capped at
        `PROTECTED` of the budget, and the least recently used of it falls back to probation when it is over.
        Eviction takes probation first."""
        if entry.bytes > self.budget or not entry.states:
            return
        if parent is not None and len(parent.tokens) <= len(entry.tokens) \
                and entry.tokens[:len(parent.tokens)] == parent.tokens:
            eid = next((k for k, e in self.entries.items() if e is parent), None)
            if eid is not None:
                self._remove(eid)
                self.replaced += 1
        while self.used + entry.bytes > self.budget and self.entries:
            self._remove(self._victim())
            self.evictions += 1
        eid = self._next
        self._next += 1
        self.entries[eid] = entry
        self.used += entry.bytes
        if parent is None:
            self.probation.add(eid)
            self.pbytes += entry.bytes
        else:
            self._demote_over_share(eid)
        hashes = _Hashes(entry.tokens)
        for pos in entry.states:
            self.index[hashes.key(pos)] = eid
            self.positions[pos] = self.positions.get(pos, 0) + 1

    def _demote_over_share(self, keep: int | None = None) -> None:
        """While protected holds more than its share of the budget, its least recently used entry falls back to
        probation (`keep` is the entry just placed there)."""
        while self.used - self.pbytes > self.hbudget:
            oldest = next((k for k in self.entries if k not in self.probation and k != keep), None)
            if oldest is None:
                break
            self.probation.add(oldest)
            self.pbytes += self.entries[oldest].bytes

    def _victim(self, probation_only: bool = False) -> int:
        """The entry to evict: the least recently used one in probation, else the least recently used overall."""
        for eid in self.entries:
            if eid in self.probation:
                return eid
        if probation_only:
            return None
        return next(iter(self.entries))

    def _promote(self, eid: int) -> None:
        """A hit moves an entry from probation to protected, and protected gives back what it holds over its share."""
        if eid not in self.probation:
            return
        entry = self.entries[eid]
        self.probation.discard(eid)
        self.pbytes -= entry.bytes
        self._demote_over_share(eid)

    def _remove(self, eid: int) -> None:
        entry = self.entries.pop(eid)
        self.used -= entry.bytes
        if eid in self.probation:
            self.probation.discard(eid)
            self.pbytes -= entry.bytes
        hashes = _Hashes(entry.tokens)
        for pos in entry.states:
            key = hashes.key(pos)
            if self.index.get(key) == eid:
                del self.index[key]
            n = self.positions.get(pos, 0) - 1
            if n > 0:
                self.positions[pos] = n
            else:
                self.positions.pop(pos, None)

    def clear(self) -> None:
        """Drop every entry (requests that already matched one keep their reference)."""
        self.entries.clear()
        self.index.clear()
        self.positions.clear()
        self.probation.clear()
        self.used = 0
        self.pbytes = 0

    def stats(self) -> dict:
        return {"entries": len(self.entries), "bytes": self.used, "probation": len(self.probation),
                "protected": len(self.entries) - len(self.probation), "hits": self.hits, "misses": self.misses,
                "saved_tokens": self.saved_tokens, "evictions": self.evictions, "replaced": self.replaced}

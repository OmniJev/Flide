"""Session cache: saved backbone states at token positions of earlier prefixes.

An agent asks about a growing state step after step, so most of each prefix is the previous step's prefix.
After a forward, a job's state is saved at a few positions of its prefix (marks): the last stride multiple a
margin before its end, and its next-to-last token. A later request whose prefix starts with the same tokens
resumes from the furthest saved position it matches and computes only the rest.

For qwen3_5 a saved state is exact: the Gated DeltaNet recurrent state in float32 and the convolution window
at the position, plus the full-attention keys and values before it. Nothing is approximated, so a resumed
request computes the same function as a fresh one (rounding differs as it does between any two packings).

Lookup hashes the prefix in blocks of 64 tokens, so a probe costs one pass over the request's tokens; a hit is
confirmed token by token. Entries live on the GPU and are evicted least recently used under a byte budget.
"""
from __future__ import annotations

from collections import OrderedDict


BLOCK = 64


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

    def mark(self, n_prefix: int, start: int = 0) -> int | None:
        """Where to save the state of a prefix of n tokens that starts computing at `start`: the last stride multiple
        a margin before its end, so an agent's next step (the same text with more after it) and a repeat of the same
        prefix with other questions both find it."""
        if n_prefix < self.min_tokens:
            return None
        m = (n_prefix - self.margin) // self.stride * self.stride
        return m if start < m < n_prefix else None

    def marks(self, n_prefix: int, start: int = 0) -> tuple[int, ...]:
        m = self.mark(n_prefix, start)
        return () if m is None else (m,)

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
                self.hits += 1
                self.saved_tokens += pos
                return entry, pos
        self.misses += 1
        return None, 0

    def insert(self, entry: Entry) -> None:
        if entry.bytes > self.budget or not entry.states:
            return
        while self.used + entry.bytes > self.budget and self.entries:
            self._evict()
        eid = self._next
        self._next += 1
        self.entries[eid] = entry
        self.used += entry.bytes
        hashes = _Hashes(entry.tokens)
        for pos in entry.states:
            self.index[hashes.key(pos)] = eid
            self.positions[pos] = self.positions.get(pos, 0) + 1

    def _evict(self) -> None:
        eid, entry = self.entries.popitem(last=False)
        self.used -= entry.bytes
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
        self.used = 0

    def stats(self) -> dict:
        return {"entries": len(self.entries), "bytes": self.used, "hits": self.hits, "misses": self.misses,
                "saved_tokens": self.saved_tokens}

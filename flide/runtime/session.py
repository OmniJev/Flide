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

An evicted entry leaves its save points behind for a while (ghosts). A request that misses but extends one of them
is a session the full cache cut off, an agent whose next step came after one-off requests had filled probation, so
its new entry goes to protected while protected has room, or in place of a protected entry left unused for over
STALE times as long as the session took to come back (a session that ended). Protected entries that keep coming
back are never displaced that way, so a cache too small for every session keeps most of them instead of cycling
through all of them.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict

from .store import Node, Store

BLOCK = 64
PROTECTED = (
    0.75  # of the budget that entries a request has hit may hold; the rest is probation
)
GHOSTS = 4096  # save points of evicted entries remembered
STALE = 2  # a protected entry gives way to a returning session left unused this many times its return time


def _bytes(x) -> int:
    if hasattr(x, "arena_bytes"):
        return x.arena_bytes
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


KEEP = 0.10  # of the device's memory that an "auto" cache leaves free, for requests larger than the warm-up's


def cache_size(text: str) -> float | str:
    """A --session-cache-gb value: a size in GiB, or "auto"."""
    if str(text).strip().lower() == "auto":
        return "auto"
    try:
        gib = float(text)
    except ValueError:
        gib = -1.0
    if not gib >= 0:
        raise argparse.ArgumentTypeError(
            f"--session-cache-gb takes 'auto' or a size in GiB, not {text!r}"
        )
    return gib


def auto_bytes(
    free: int, total: int, reserved: int, allocated: int, peak: int, keep: float = KEEP
) -> int:
    """Bytes an "auto" session cache takes: what the device has left once the weights, the graphs and the warm-up's
    peak work space are in (free on the device plus what the caching allocator holds unused, less that work space),
    less `keep` of the device's memory; 0 when that is under 512 MiB."""
    room = free + (reserved - allocated) - (peak - allocated) - int(keep * total)
    return room if room >= 2**29 else 0


PAGE = 64  # rows a KVArena hands out at a time


class KVArena:
    """Saved keys and values of a backbone whose state is its attention keys alone (gemma4), in one block of device
    memory the session cache holds whole. Per layer kind ("full", "sliding") a pool of rows, one row holding one
    position's keys and values in every layer of that kind, handed out in pages of PAGE rows. An entry keeps row
    indices, so the CUDA graphs read and write saved positions in place, and nothing is allocated per request. One more
    page per pool takes the writes of padding."""

    def __init__(self, layers: dict, nbytes: int, device, dtype, shares: dict):
        """`layers`: layer index -> (kind, key row shape, value row shape); `shares`: kind -> fraction of `nbytes`."""
        import math

        import torch

        self.device = torch.device(device)
        self.kind = {i: k for i, (k, _, _) in layers.items()}
        elem = torch.empty(0, dtype=dtype).element_size()
        self.row_bytes: dict[str, int] = {}
        for k, ks, vs in layers.values():
            self.row_bytes[k] = (
                self.row_bytes.get(k, 0) + (math.prod(ks) + math.prod(vs)) * elem
            )
        self.pages = {
            k: max(0, int(nbytes * shares.get(k, 0.0)) // (b * PAGE))
            for k, b in self.row_bytes.items()
        }
        self.K, self.V = {}, {}
        for i, (k, ks, vs) in layers.items():
            n = (self.pages[k] + 1) * PAGE
            self.K[i] = torch.zeros((n,) + tuple(ks), device=self.device, dtype=dtype)
            self.V[i] = torch.zeros((n,) + tuple(vs), device=self.device, dtype=dtype)
        self.free_pages = {k: list(range(n - 1, -1, -1)) for k, n in self.pages.items()}
        self.capacity = sum(n * PAGE * self.row_bytes[k] for k, n in self.pages.items())

    def dump(self, kind: str) -> int:
        """The first row of the pool's padding page."""
        return self.pages[kind] * PAGE

    def alloc(self, need: dict):
        """Rows for `need` (kind -> count) in position order, or None when a pool has too few free pages."""
        import numpy as np

        want = {k: -(-n // PAGE) for k, n in need.items()}
        if any(len(self.free_pages[k]) < p for k, p in want.items()):
            return None
        out = {}
        for k, n in need.items():
            # one consecutive run of pages when the pool has it, so the rows are contiguous and a resume copies them
            # with one copy rather than one per saved block (292 us against 32 us per layer at 2.9k tokens)
            pages = self._run_of_pages(k, want[k])
            if pages is None:
                pages = sorted(self.free_pages[k].pop() for _ in range(want[k]))
            out[k] = (np.array(pages, np.int64)[:, None] * PAGE + np.arange(PAGE)).reshape(-1)[:n]
        return out

    def _run_of_pages(self, kind: str, count: int):
        """Take `count` consecutive free pages and return their numbers, or None when the pool has no such run."""
        free = sorted(self.free_pages[kind])
        for i in range(len(free) - count + 1):
            if free[i + count - 1] - free[i] == count - 1:
                take = free[i: i + count]
                for page in take:
                    self.free_pages[kind].remove(page)
                return take
        return None

    def free(self, rows: dict) -> None:
        for k, r in rows.items():
            self.free_pages[k].extend(sorted(set((r // PAGE).tolist()), reverse=True))


class ArenaRows:
    """The rows of a KVArena an entry holds: per kind, row indices in the order of the positions they keep (a full
    layer's from position 0 to the save point, a sliding layer's the last `window` before it). Indexing by layer gives
    that layer's (K, V), as an entry's `kv` dict does."""

    def __init__(self, arena: KVArena, rows: dict):
        import torch

        self.arena = arena
        self.rows = rows
        self.dev = {k: torch.from_numpy(r).to(arena.device) for k, r in rows.items()}
        self.arena_bytes = sum(len(r) * arena.row_bytes[k] for k, r in rows.items())

    def __getitem__(self, layer: int):
        a = self.arena
        idx = self.dev[a.kind[layer]]
        return a.K[layer].index_select(0, idx), a.V[layer].index_select(0, idx)

    def write(self, layer: int, K, V) -> None:
        a = self.arena
        idx = self.dev[a.kind[layer]]
        a.K[layer].index_copy_(0, idx, K)
        a.V[layer].index_copy_(0, idx, V)

    def free(self) -> None:
        if self.rows:
            self.arena.free(self.rows)
            self.rows, self.dev = {}, {}


class KV:
    """Keys and values of a prefix as segments `[(tensor, start, end)]`, oldest first.

    A plain tensor is one segment; an entry that shares its trunk with a sibling holds several, the shared blocks and
    its own. Readers copy the segments where they belong (`runs()` merges adjacent ones into a page), so nothing needs
    the whole prefix contiguous and a resume does not do one copy per saved block.
    """

    __slots__ = ("parts",)

    def __init__(self, parts):
        self.parts = list(parts)

    @classmethod
    def wrap(cls, x) -> "KV":
        """A plain tensor as one segment; an object that only reports its size (a test's stand-in) has no rows."""
        if isinstance(x, cls):
            return x
        return cls([(x, 0, x.shape[0] if getattr(x, "shape", None) else 0)])

    def rows(self):
        return self.parts

    def runs(self):
        """The parts with adjacent ones merged where both the source rows and the saved positions are contiguous.

        The arena hands a shared prefix's blocks out in pages, so this turns one copy per 64-token block into one per
        page. Merged runs are slices of a view over the whole storage, so a run is exactly as long as its positions.
        """
        import torch

        parts = self.parts
        if len(parts) < 2:
            return parts
        first = parts[0][0]
        row = first.stride(0) * first.element_size()
        n = first.untyped_storage().size() // max(1, row)
        base = torch.as_strided(first, (n,) + tuple(first.shape[1:]), first.stride(), 0)
        out = []
        for t, lo, hi in parts:
            off = (t.data_ptr() - base.data_ptr()) // max(1, row)
            if out:
                plo, phi, poff = out[-1]
                if lo == phi and poff + (phi - plo) == off:
                    out[-1] = (plo, phi + (hi - lo), poff)
                    continue
            out.append((lo, hi, off))
        return [(base[off: off + (hi - lo)], lo, hi) for lo, hi, off in out]

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t, _, _ in self.parts)

    def __getitem__(self, item):
        """`[:p]` for callers that only need a prefix: a segment cut at `p`."""
        lo, hi = (0, len(self)) if isinstance(item, slice) else (item, item + 1)
        hi = len(self) if hi is None else min(hi, len(self))
        out = []
        for t, a, b in self.parts:
            if b <= lo:
                continue
            out.append((t, max(a, lo) - a, min(b, hi) - a))
            if b >= hi:
                break
        return KV(out)

    def __len__(self) -> int:
        return self.parts[-1][2] if self.parts else 0


class Entry:
    """A prefix and the backbone state at some of its positions: states[pos] holds what the backbone needs to
    continue from pos (for qwen3_5: "S" and "conv" per linear-attention layer); kv[layer] the full-attention keys
    and values of the whole prefix."""

    def __init__(self, tokens, states: dict, kv: dict):
        self.tokens = list(tokens)
        self.states = states
        if hasattr(kv, "items"):
            self.kv = {layer: (KV.wrap(k), KV.wrap(v)) for layer, (k, v) in kv.items()}
            kv_bytes = sum(pair[0].nbytes + pair[1].nbytes for pair in self.kv.values())
        else:  # Gemma 4's KVArena rows are allocated and released by the graph cache.
            self.kv = kv
            kv_bytes = _bytes(kv)
        self.bytes = _bytes(states) + kv_bytes


class _Hashes:
    """Chained block hashes of a token list: key(pos) identifies tokens[:pos]."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.blocks = [0]
        h = 0
        for k in range(len(tokens) // BLOCK):
            h = hash((h, tuple(tokens[k * BLOCK : (k + 1) * BLOCK])))
            self.blocks.append(h)

    def key(self, pos: int):
        k = pos // BLOCK
        return pos, hash((self.blocks[k], tuple(self.tokens[k * BLOCK : pos])))


def _free(entry) -> None:
    """Give an entry's rows back to its store (entries that hold their own tensors have nothing to give)."""
    free = getattr(entry.kv, "free", None)
    if free is not None:
        free()


class SessionCache:
    def __init__(
        self,
        budget_bytes: int,
        min_tokens: int = 2048,
        stride: int = 256,
        margin: int = 32,
    ):
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
        self.probation: set[int] = set()  # entries no request has hit yet
        self.pbytes = 0  # bytes the probation entries hold
        self.store = None  # Gemma 4's KVArena; prefix sharing uses pstore
        self.pstore: Store | None = None
        self.node_of: dict[int, tuple[Node, tuple[bytes, ...]]] = {}
        self.pinned: set[int] = (
            set()
        )  # ids of entries a forward being set up reads: `reserve` leaves them
        self.clock = 0  # lookups so far
        self.seen: dict[int, int] = {}  # entry id -> the lookup that last used it
        self.ghosts: OrderedDict[tuple, int] = (
            OrderedDict()
        )  # save point of an evicted entry -> when it was used
        self.gpos: dict[int, int] = {}  # positions of the ghosts, with counts
        self.returning = 0  # misses that extended a ghost

    def attach(self, store) -> None:
        """Keep the saved keys and values in `store` (a KVArena): the budget becomes its capacity, and a save
        reserves its rows before the forward that writes them."""
        self.store = store
        self.budget = store.capacity

    def attach_prefix_store(self, device) -> int:
        """Reserve one shared prefix/KV arena for the cache's non-Gemma session path."""
        self.pstore = Store(self.budget)
        self.pstore.configure(device)
        return self.pstore.budget

    def reserve_prefix(self, need: dict, dtype=None):
        """Arena rows for a save point before the forward, from the prefix store (None when it has no room)."""
        return self.pstore.reserve_rows(need, dtype) if getattr(self, "pstore", None) is not None else None

    def reserve(self, need: dict, keep=()):
        """Rows of the store for a state about to be saved (kind -> count), as ArenaRows, evicting what it needs room
        from (probation first, least recently used first) but no entry in `keep` or `pinned`; None when it does not
        fit even then."""
        skip = self.pinned | {id(e) for e in keep if e is not None}
        while True:
            rows = self.store.alloc(need)
            if rows is not None:
                return ArenaRows(self.store, rows)
            eid = self._victim(skip=skip)
            if eid is None:
                return None
            self._evict(eid)

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
        self.clock += 1
        eid, pos = self._find(tokens)
        if eid is None:
            self.misses += 1
            return None, 0
        self.entries.move_to_end(eid)
        self.seen[eid] = self.clock
        self._promote(eid)
        self.hits += 1
        self.saved_tokens += pos
        return self.entries[eid], pos

    def peek(self, tokens) -> int:
        """The position `match` would resume `tokens` from (0 for none), without counting a lookup or using an entry."""
        return self._find(tokens)[1]

    def _find(self, tokens) -> tuple[int | None, int]:
        if len(tokens) < self.min_tokens or not self.entries:
            return None, 0
        hashes = _Hashes(tokens)
        for pos in sorted((p for p in self.positions if p < len(tokens)), reverse=True):
            eid = self.index.get(hashes.key(pos))
            if eid is not None and self.entries[eid].tokens[:pos] == tokens[:pos]:
                return eid, pos
        return None, 0

    def insert(self, entry: Entry, parent: Entry | None = None) -> None:
        """Keep `entry`, evicting the least recently used ones it needs room from; `parent`, the entry the request
        that saved it resumed from, goes first when `entry` covers its tokens.

        A new entry enters probation unless the request resumed from a cached entry: that prefix has already been
        reused once, so it goes to protected and one-off requests cannot push it out. A new entry that extends a
        ghost goes to protected too when `_admit` says so. Protected is capped at `PROTECTED` of the budget, and the
        least recently used of it falls back to probation when it is over. Eviction takes probation first."""
        if (self.pstore is None and entry.bytes > self.budget) or not entry.states:
            _free(entry)
            return
        node = None
        state_keys = ()
        if self.pstore is not None:
            states = dict(entry.states)
            if parent is not None and states and len(states) < 2:
                earliest = min(states)
                inherited = [
                    pos
                    for pos in parent.states
                    if pos < earliest
                    and pos <= len(entry.tokens)
                    and entry.tokens[:pos] == parent.tokens[:pos]
                ]
                if inherited:
                    pos = max(inherited)
                    states[pos] = parent.states[pos]
            stored = None
            hint = None
            if parent is not None:
                for k, e in self.entries.items():
                    if e is parent:
                        hint = self.node_of.get(k, (None,))[0]
                        break
            while stored is None:
                stored = self.pstore.insert(entry.tokens, states, entry.kv, parent_hint=hint)
                if stored is not None:
                    break
                victim = self._victim()
                if victim is None:
                    _free(entry)
                    return
                self._evict(victim)
            node, stored_states, state_keys, cost = stored
            entry.states = stored_states
            layers = self.pstore.entry_data(node)
            entry.kv = {
                layer: (KV(keys), KV(values))
                for layer, (keys, values) in layers.items()
            }
            entry.bytes = cost
            self.used = self.pstore.used
        if (
            parent is not None
            and len(parent.tokens) <= len(entry.tokens)
            and entry.tokens[: len(parent.tokens)] == parent.tokens
        ):
            eid = next((k for k, e in self.entries.items() if e is parent), None)
            if eid is not None:
                self._remove(eid)
                self.replaced += 1
        back = self._came_back(entry.tokens) if parent is None else None
        if self.pstore is None:
            while self.used + entry.bytes > self.budget and self.entries:
                victim = self._victim()
                if victim is None:
                    break
                self._evict(victim)
        eid = self._next
        self._next += 1
        if node is not None:
            self.node_of[eid] = (node, state_keys)
        self.entries[eid] = entry
        if node is None:
            self.used += entry.bytes
        else:
            self.used = self.pstore.used
        self.seen[eid] = self.clock
        if back is not None:
            self.returning += 1
        if parent is None and (back is None or not self._admit(eid, self.clock - back)):
            self.probation.add(eid)
            self.pbytes += entry.bytes
        else:
            self._demote_over_share(eid)
        hashes = _Hashes(entry.tokens)
        for pos in entry.states:
            self.index[hashes.key(pos)] = eid
            self.positions[pos] = self.positions.get(pos, 0) + 1

    def _admit(self, eid: int, interval: int) -> bool:
        """Whether entry `eid`, of a session that came back after `interval` lookups, goes to protected: when it fits
        protected's share, or when the least recently used protected entry has gone unused for over STALE times
        that long."""
        if self.used - self.pbytes <= self.hbudget:
            return True
        oldest = next(
            (k for k in self.entries if k not in self.probation and k != eid), None
        )
        return oldest is not None and self.clock - self.seen.get(
            oldest, 0
        ) > STALE * max(1, interval)

    def _evict(self, eid: int) -> None:
        """Remove an entry the budget needs room from, its save points kept as ghosts."""
        entry = self.entries[eid]
        hashes = _Hashes(entry.tokens)
        when = self.seen.get(eid, self.clock)
        for pos in entry.states:
            key = hashes.key(pos)
            if key not in self.ghosts:
                self.gpos[pos] = self.gpos.get(pos, 0) + 1
            self.ghosts[key] = when
            self.ghosts.move_to_end(key)
        while len(self.ghosts) > GHOSTS:
            key, _ = self.ghosts.popitem(last=False)
            self._forget(key[0])
        self._remove(eid)
        self.evictions += 1

    def _came_back(self, tokens) -> int | None:
        """When an evicted entry whose save point `tokens` extends was last used (the latest of several), its ghosts
        dropped; None when there is none."""
        if not self.ghosts:
            return None
        hashes = _Hashes(tokens)
        back = None
        for pos in [p for p in self.gpos if p <= len(tokens)]:
            when = self.ghosts.pop(hashes.key(pos), None)
            if when is not None:
                self._forget(pos)
                back = when if back is None else max(back, when)
        return back

    def _forget(self, pos: int) -> None:
        n = self.gpos.get(pos, 0) - 1
        if n > 0:
            self.gpos[pos] = n
        else:
            self.gpos.pop(pos, None)

    def _demote_over_share(self, keep: int | None = None) -> None:
        """While protected holds more than its share of the budget, its least recently used entry falls back to
        probation (`keep` is the entry just placed there)."""
        while self.used - self.pbytes > self.hbudget:
            oldest = next(
                (k for k in self.entries if k not in self.probation and k != keep), None
            )
            if oldest is None:
                break
            self.probation.add(oldest)
            self.pbytes += self.entries[oldest].bytes

    def _victim(self, probation_only: bool = False, skip=frozenset()) -> int:
        """Evict a leaf first, preferring probation and then the least recently used entry."""
        pools = [
            [
                k
                for k, e in self.entries.items()
                if k in self.probation and id(e) not in skip
            ]
        ]
        if not probation_only:
            pools.append(
                [
                    k
                    for k, e in self.entries.items()
                    if k not in self.probation and id(e) not in skip
                ]
            )
        for candidates in pools:
            if not candidates:
                continue
            if self.pstore is not None:
                leaves = [
                    k
                    for k in candidates
                    if (ref := self.node_of.get(k)) is not None
                    and ref[0].refs == 1
                    and not ref[0].children
                ]
                if leaves:
                    return leaves[0]
            return candidates[0]
        return None

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
        self.seen.pop(eid, None)
        ref = self.node_of.pop(eid, None)
        if ref is not None:
            self.pstore.release(*ref)
            self.used = self.pstore.used
        _free(entry)
        if ref is None:
            self.used -= entry.bytes
        if eid in self.probation:
            self.probation.discard(eid)
            self.pbytes -= entry.bytes
        self.pbytes = min(self.pbytes, self.used)
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
        """Drop every entry (requests that already matched one keep their reference; with a store its rows go back to
        it, so nothing may be reading them)."""
        for eid, e in list(self.entries.items()):
            ref = self.node_of.pop(eid, None)
            if ref is not None:
                self.pstore.release(*ref)
            else:
                _free(e)
        if self.pstore is not None:
            self.pstore.clear()
        self.entries.clear()
        self.index.clear()
        self.positions.clear()
        self.probation.clear()
        self.seen.clear()
        self.ghosts.clear()
        self.gpos.clear()
        self.used = 0
        self.pbytes = 0
        self.node_of.clear()

    def stats(self) -> dict:
        return {
            "entries": len(self.entries),
            "bytes": self.used,
            "probation": len(self.probation),
            "protected": len(self.entries) - len(self.probation),
            "hits": self.hits,
            "misses": self.misses,
            "saved_tokens": self.saved_tokens,
            "evictions": self.evictions,
            "replaced": self.replaced,
            "returning": self.returning,
        }

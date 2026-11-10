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

import hashlib
import json
import os
import logging
from collections import OrderedDict
from pathlib import Path


BLOCK = 64
_FORMAT = 1
log = logging.getLogger("flide.session")
_HELD: dict[str, object] = {}


def _plain(x):
    """What torch.save can take: a detached CPU copy of every tensor, because tensors of different dtypes can share one
    storage and torch refuses to save that."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x.detach().to("cpu", copy=True) if hasattr(x, "detach") else x


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
    def __init__(self, budget_bytes: int, min_tokens: int = 2048, stride: int = 256, margin: int = 32,
                 root: str | None = None, key: str | None = None, keep: int = 8, device: str = "cuda:0"):
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
        self.root = Path(root) if root else None
        self.keep = keep
        self.device = device
        self._loaded = 0
        self._closed = False
        if self.root is not None:
            self._attach(key)

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
        """Drop every entry (requests that already matched one keep their reference), on disk too."""
        self.entries.clear()
        self.index.clear()
        self.positions.clear()
        self.used = 0
        if self.root is not None:
            self._wipe()

    def _wipe(self) -> None:
        for path in self.root.glob("*.pt"):
            path.unlink(missing_ok=True)

    def _attach(self, key: str | None) -> None:
        """Take over what a previous process saved in `root`, oldest first so the budget keeps the newest, when it
        was saved from the same checkpoint on the same kind of device; anything else there is thrown away."""
        import torch

        self.root.mkdir(parents=True, exist_ok=True)
        self._hold()
        meta = self.root / "meta.json"
        want = {"format": _FORMAT, "key": key}
        try:
            have = json.loads(meta.read_text())
        except (OSError, ValueError):
            have = None
        if have != want:
            self._wipe()
            meta.write_text(json.dumps(want))
            return
        for path in sorted(self.root.glob("*.pt")):
            try:
                saved = torch.load(path, map_location=self.device, weights_only=True)
                entry = Entry(saved["tokens"], saved["states"], saved["kv"])
            except Exception:
                path.unlink(missing_ok=True)
                continue
            self.insert(entry)
            self._loaded += 1

    def _hold(self) -> None:
        """One process at a time: close() rewrites the directory, so a second server must not point at the same one."""
        try:
            import fcntl
        except ImportError:
            return
        root = str(self.root)
        if root in _HELD:                 # this process already owns the directory
            return
        fd = (self.root / ".lock").open("w")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fd.close()
            raise ValueError(f"{self.root} is in use by another flide process") from None
        _HELD[root] = fd

    def close(self) -> None:
        """Write the newest `keep` entries to `root` (a process that stops cleanly leaves the next one its context)."""
        if self.root is None or self._closed:
            return
        self._closed = True
        import torch

        self._wipe()
        for i, entry in enumerate(list(self.entries.values())[-self.keep:]):
            tmp = self.root / f".{i:08d}.tmp"
            try:
                torch.save({"tokens": entry.tokens, "states": _plain(entry.states), "kv": _plain(entry.kv)}, tmp)
                os.replace(tmp, self.root / f"{i:08d}.pt")
            except Exception:
                tmp.unlink(missing_ok=True)
                log.warning("session cache: could not write %s", self.root / f"{i:08d}.pt", exc_info=True)
        held = _HELD.pop(str(self.root), None)
        if held is not None:
            held.close()

    def stats(self) -> dict:
        return {"entries": len(self.entries), "bytes": self.used, "hits": self.hits, "misses": self.misses,
                "saved_tokens": self.saved_tokens, "restored": self._loaded,
                "on_disk": len(list(self.root.glob("*.pt"))) if self.root is not None else 0}


def fingerprint(model_dir: str, dtype: str, device: str) -> str:
    """What a saved state is valid for: the checkpoint's own config, the dtype, the device's capability, and the
    software that computed it -- a state written by other kernels is not the same state, even if the tensors load."""
    import torch

    from .. import __version__

    h = hashlib.sha256()
    for name in ("config.json", "decision_config.json"):
        path = Path(model_dir) / name
        h.update(path.read_bytes() if path.is_file() else b"\0")
    h.update(str(dtype).encode())
    h.update(__version__.encode())
    h.update(torch.__version__.encode())
    try:
        import triton

        h.update(triton.__version__.encode())
    except ImportError:
        pass
    if str(device).startswith("cuda") and torch.cuda.is_available():
        h.update(str(torch.cuda.get_device_capability(device)).encode())
    return h.hexdigest()[:32]

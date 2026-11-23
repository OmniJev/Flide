"""A prefix tree whose K/V and recurrent-state tensors share one reserved buffer."""

from __future__ import annotations

import hashlib
from collections import OrderedDict

BLOCK = 64


def key_of(parent: bytes, tokens) -> bytes:
    """A stable key for a node, independent of Python's per-process hash seed."""
    h = hashlib.blake2b(parent, digest_size=16)
    for i in range(0, len(tokens), BLOCK):
        block = tokens[i : i + BLOCK]
        h.update(b"".join(int(t).to_bytes(8, "little", signed=True) for t in block))
    return h.digest()


class Node:
    """One prefix extension, with only its newly-owned rows and snapshots."""

    __slots__ = (
        "allocations",
        "bytes",
        "children",
        "key",
        "kv",
        "parent",
        "refs",
        "sessions",
        "span",
        "states",
        "tokens",
    )

    def __init__(self, tokens, key: bytes, parent: Node | None = None):
        self.tokens = list(tokens)
        self.key = key
        self.parent = parent
        self.children: dict[bytes, Node] = {}
        self.kv = {}
        self.states = {}
        self.refs = 1
        self.sessions: set = set()
        self.span = parent.end if parent is not None else 0
        self.allocations: list[tuple[int, int]] = []
        self.bytes = 0

    @property
    def end(self) -> int:
        return self.span + len(self.tokens)


class Store:
    """A trie over prefix extensions with leaf-first release and a fixed byte arena.

    Tensor views are carved from one ``uint8`` allocation. The small free-list
    allocator is byte-addressed so K/V and recurrent state tensors of different
    dtypes can share it. Their original dtype and shape are preserved.
    """

    def __init__(self, budget_bytes: int):
        self.budget = max(0, int(budget_bytes))
        self.used = 0
        self.nodes: OrderedDict[bytes, Node] = OrderedDict()
        self.by_position: dict[int, set[bytes]] = {}
        self.evictions = 0
        self.buffer = None
        self.device = None
        self.free: list[tuple[int, int]] = []

    def configure(self, device) -> None:
        """Reserve the store's backing block before requests start."""
        import torch

        if self.nodes:
            raise RuntimeError(
                "cannot configure a prefix store after inserting entries"
            )
        self.buffer = torch.empty(self.budget, dtype=torch.uint8, device=device)
        self.device = torch.device(device)
        self.free = [(0, self.budget)] if self.budget else []

    def _alloc(self, nbytes: int, alignment: int) -> int | None:
        if nbytes == 0:
            return 0
        for i, (start, size) in enumerate(self.free):
            at = (start + alignment - 1) // alignment * alignment
            padding = at - start
            if size - padding < nbytes:
                continue
            end = at + nbytes
            tail = start + size - end
            replacement = []
            if padding:
                replacement.append((start, padding))
            if tail:
                replacement.append((end, tail))
            self.free[i : i + 1] = replacement
            return at
        return None

    def _free(self, offset: int, nbytes: int) -> None:
        if not nbytes:
            return
        self.free.append((offset, nbytes))
        self.free.sort()
        merged: list[tuple[int, int]] = []
        for start, size in self.free:
            if merged and merged[-1][0] + merged[-1][1] == start:
                prev, length = merged[-1]
                merged[-1] = (prev, length + size)
            else:
                merged.append((start, size))
        self.free = merged

    def _allocate_tensor(self, shape, dtype, device, allocations):
        import math

        import torch

        element_size = torch.empty((), dtype=dtype).element_size()
        nbytes = math.prod(shape) * element_size
        if self.buffer is None:
            self.configure(device)
        if torch.device(device) != self.device:
            raise ValueError(
                f"prefix-store tensor is on {device}, expected {self.device}"
            )
        offset = self._alloc(nbytes, element_size)
        if offset is None:
            return None
        allocations.append((offset, nbytes))
        if not nbytes:
            return torch.empty(shape, dtype=dtype, device=device)
        view = self.buffer.narrow(0, offset, nbytes).view(dtype).view(shape)
        return view

    def _copy_tensor(self, source, allocations: list[tuple[int, int]]):
        view = self._allocate_tensor(
            source.shape, source.dtype, source.device, allocations
        )
        if view is not None:
            view.copy_(source)
        return view

    def _copy_tree(self, value, allocations):
        import torch

        if isinstance(value, torch.Tensor):
            return self._copy_tensor(value, allocations)
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                out[key] = self._copy_tree(item, allocations)
                if out[key] is None:
                    return None
            return out
        if isinstance(value, list):
            out = [self._copy_tree(item, allocations) for item in value]
            return None if any(item is None for item in out) else out
        if isinstance(value, tuple):
            out = tuple(self._copy_tree(item, allocations) for item in value)
            return None if any(item is None for item in out) else out
        return value

    @staticmethod
    def _rows(kv):
        return kv.rows() if hasattr(kv, "rows") else [(kv, 0, len(kv))]

    def _copy_rows(self, source, lo: int, hi: int, allocations):
        pieces = []
        for tensor, start, end in self._rows(source):
            a, b = max(lo, start), min(hi, end)
            if a < b:
                pieces.append((tensor, start, a, b))
        if not pieces:
            return None
        first = pieces[0][0]
        out = self._allocate_tensor(
            (hi - lo, *first.shape[1:]), first.dtype, first.device, allocations
        )
        if out is None:
            return None
        at = lo
        for tensor, source_start, start, end in pieces:
            if tensor.dtype != first.dtype or tensor.shape[1:] != first.shape[1:]:
                return None
            if start != at:
                return None
            out[start - lo : end - lo].copy_(
                tensor[start - source_start : end - source_start]
            )
            at = end
        if at != hi:
            return None
        return out

    def _path(self, node: Node) -> list[Node]:
        out = []
        while node is not None:
            out.append(node)
            node = node.parent
        return list(reversed(out))

    def _has_state(self, parent: Node | None, pos: int) -> bool:
        node = parent
        while node is not None:
            if pos in node.states:
                return True
            node = node.parent
        return False

    def _parent(self, tokens) -> tuple[Node | None, int]:
        parent, best = None, 0
        for node in self.nodes.values():
            end = node.end
            if end <= best or end > len(tokens):
                continue
            if self._prefix_equals(node, end, tokens):
                parent, best = node, end
        return parent, best

    def insert(self, tokens, states: dict, kv: dict, session=None) -> Node | None:
        """Add one prefix entry, copying only K/V rows and states not already held by ancestors.

        Returns ``None`` without changing the store if the arena cannot fit the
        new data. The caller can then evict cache entries and retry.
        """
        parent, shared = self._parent(tokens)
        node = parent if parent is not None and shared == len(tokens) else None
        if node is None:
            key = key_of(parent.key if parent is not None else b"", tokens[shared:])
            node = self.nodes.get(key)
            if node is not None and not self._prefix_equals(node, len(tokens), tokens):
                # Do not let even a digest collision replace an unrelated live path.
                base = key
                nonce = 1
                while node is not None and not self._prefix_equals(
                    node, len(tokens), tokens
                ):
                    key = hashlib.blake2b(
                        base + nonce.to_bytes(8, "little"), digest_size=16
                    ).digest()
                    node = self.nodes.get(key)
                    nonce += 1

        if node is not None:
            allocations: list[tuple[int, int]] = []
            copied_states = {}
            try:
                for pos, state in states.items():
                    if pos > len(tokens) or self._has_state(node, pos):
                        continue
                    copied = self._copy_tree(state, allocations)
                    if copied is None:
                        self._rollback(allocations)
                        return None
                    copied_states[pos] = copied
            except Exception:
                self._rollback(allocations)
                raise
            node.states.update(copied_states)
            for pos in copied_states:
                self.by_position.setdefault(pos, set()).add(node.key)
            node.allocations.extend(allocations)
            added = sum(size for _, size in allocations)
            node.bytes += added
            self.used += added
            node.refs += 1
            if session is not None:
                node.sessions.add(session)
            return node

        node = Node(tokens[shared:], key, parent)
        missing_states = {
            pos: state
            for pos, state in states.items()
            if pos <= len(tokens) and not self._has_state(parent, pos)
        }
        allocations: list[tuple[int, int]] = []
        copied_states = {}
        try:
            for pos, state in missing_states.items():
                copied_states[pos] = self._copy_tree(state, allocations)
                if copied_states[pos] is None:
                    self._rollback(allocations)
                    return None

            copied_kv = {}
            for layer, (keys, values) in kv.items():
                k = self._copy_rows(keys, shared, len(tokens), allocations)
                v = self._copy_rows(values, shared, len(tokens), allocations)
                if k is None or v is None:
                    self._rollback(allocations)
                    return None
                copied_kv[layer] = (k, v)
        except Exception:
            if allocations:
                self._rollback(allocations)
            raise

        node.states = copied_states
        node.kv = copied_kv
        node.allocations = allocations
        node.bytes = sum(size for _, size in allocations)
        if parent is not None:
            parent.children[key] = node
        if session is not None:
            node.sessions.add(session)
        self.nodes[key] = node
        self.used += node.bytes
        for pos in node.states:
            self.by_position.setdefault(pos, set()).add(key)
        return node

    def _rollback(self, allocations) -> None:
        for offset, size in allocations:
            self._free(offset, size)

    def _prefix_equals(self, node: Node, pos: int, tokens) -> bool:
        """Whether the path to ``node`` spells ``tokens[:pos]``."""
        at = 0
        for part in self._path(node):
            if at >= pos:
                break
            n = min(len(part.tokens), pos - at)
            if tokens[at : at + n] != part.tokens[:n]:
                return False
            at += n
        return at >= pos

    def match(self, tokens) -> tuple[Node | None, int]:
        """Return the deepest saved position whose tokens match."""
        best, pos = None, 0
        for node in self.nodes.values():
            for mark in node.states:
                if pos < mark <= len(tokens) and self._prefix_equals(
                    node, mark, tokens
                ):
                    best, pos = node, mark
        return best, pos

    def entry_data(self, node: Node) -> tuple[dict, dict]:
        """Build a view of the states and K/V segments held along a node's path."""
        states, layers = {}, {}
        for part in self._path(node):
            states.update(part.states)
            for layer, (k, v) in part.kv.items():
                layers.setdefault(layer, ([], []))
                layers[layer][0].append((k, part.span, part.end))
                layers[layer][1].append((v, part.span, part.end))
        return states, layers

    def _drop(self, node: Node) -> int:
        self.nodes.pop(node.key, None)
        if node.parent is not None:
            node.parent.children.pop(node.key, None)
        for pos in node.states:
            holders = self.by_position.get(pos)
            if holders is not None:
                holders.discard(node.key)
                if not holders:
                    self.by_position.pop(pos, None)
        for offset, size in node.allocations:
            self._free(offset, size)
        self.used -= node.bytes
        return node.bytes

    def release(self, node: Node) -> int:
        """Release one entry reference, collecting newly-unreferenced ancestors."""
        node.refs -= 1
        freed = 0
        while node.refs <= 0 and not node.children and not node.sessions:
            parent = node.parent
            freed += self._drop(node)
            if parent is None:
                break
            node = parent
        return freed

    def clear(self) -> None:
        """Forget every entry and make the arena available again."""
        self.nodes.clear()
        self.by_position.clear()
        self.used = 0
        self.free = (
            [(0, self.budget)] if self.buffer is not None and self.budget else []
        )

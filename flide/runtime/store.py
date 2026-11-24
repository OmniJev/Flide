"""A prefix tree whose K/V and recurrent-state tensors share one reserved buffer."""

from __future__ import annotations

import hashlib
from collections import OrderedDict

BLOCK = 64
NODE = 64      # tokens per stored node; the tests pin this granularity, and a zero-copy save makes it costless


def key_of(parent: bytes, tokens) -> bytes:
    """A stable key for a node, independent of Python's per-process hash seed."""
    h = hashlib.blake2b(parent, digest_size=16)
    for i in range(0, len(tokens), BLOCK):
        block = tokens[i : i + BLOCK]
        h.update(b"".join(int(t).to_bytes(8, "little", signed=True) for t in block))
    return h.digest()


class Node:
    """One token block, with only the K/V rows it owns."""

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
        "tokens",
    )

    def __init__(self, tokens, key: bytes, parent: Node | None = None):
        self.tokens = list(tokens)
        self.key = key
        self.parent = parent
        self.children: dict[bytes, Node] = {}
        self.kv = {}
        self.refs = 0
        self.sessions: set = set()
        self.span = parent.end if parent is not None else 0
        self.allocations: list[tuple[int, int]] = []
        self.bytes = 0

    @property
    def end(self) -> int:
        return self.span + len(self.tokens)


class StateSnapshot:
    """One saved recurrent state, shared by entries with the same prefix and position."""

    __slots__ = ("position", "tokens", "value", "refs", "allocations", "bytes")

    def __init__(self, position, tokens, value, allocations):
        self.position = position
        self.tokens = tuple(tokens)
        self.value = value
        self.refs = 0
        self.allocations = allocations
        self.bytes = sum(size for _, size in allocations)


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
        self.state_records: dict[bytes, StateSnapshot] = {}
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

    def _gather_rows(self, source, lo: int, hi: int):
        """Rows `[lo, hi)` of `source` as one tensor: a view when it is one tensor, a concatenation otherwise."""
        one = self._view_rows(source, lo, hi)
        if one is not None:
            return one
        import torch

        parts = []
        for tensor, start, end in self._rows(source):
            a, b = max(lo, start), min(hi, end)
            if a < b:
                parts.append(tensor[a - start: b - start])
        return torch.cat(parts)

    def _view_rows(self, source, lo: int, hi: int):
        """A view of `source[lo:hi]` when it is one tensor: the saved keys and values are already a private clone."""
        if not hasattr(source, "rows"):
            return source[lo:hi] if getattr(source, "shape", None) else None
        parts = []
        for tensor, start, end in source.rows():
            a, b = max(lo, start), min(hi, end)
            if a < b:
                parts.append(tensor[a - start: b - start])
        return parts[0] if len(parts) == 1 else None

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

    def _parent(self, tokens) -> tuple[Node | None, int]:
        parent, best = None, 0
        for node in self.nodes.values():
            end = node.end
            if end <= best or end > len(tokens):
                continue
            if self._prefix_equals(node, end, tokens):
                parent, best = node, end
        return parent, best

    def _state_key(
        self, position: int, prefix, staged: dict[bytes, StateSnapshot]
    ) -> bytes:
        seed = b"state" + position.to_bytes(8, "little", signed=True)
        base = key_of(seed, prefix)
        key, nonce = base, 0
        target = tuple(prefix)
        while True:
            record = staged.get(key) or self.state_records.get(key)
            if record is None or (
                record.position == position and record.tokens == target
            ):
                return key
            nonce += 1
            key = hashlib.blake2b(
                base + nonce.to_bytes(8, "little"), digest_size=16
            ).digest()

    def _node_key(
        self, parent: Node | None, tokens, staged: dict[bytes, Node]
    ) -> bytes:
        base = key_of(parent.key if parent is not None else b"", tokens)
        key, nonce = base, 0
        while True:
            node = staged.get(key) or self.nodes.get(key)
            if node is None or (node.parent is parent and node.tokens == list(tokens)):
                return key
            nonce += 1
            key = hashlib.blake2b(
                base + nonce.to_bytes(8, "little"), digest_size=16
            ).digest()

    def insert(self, tokens, states: dict, kv: dict, session=None):
        """Insert token blocks, K/V tails, and independently referenced save points.

        Nodes are split at ``BLOCK`` boundaries, so siblings share their common
        token blocks even when neither complete prefix contains the other.
        Returns ``(leaf, stored_states, state_keys, added_bytes)`` or ``None``
        without changing the store if its arena cannot fit the insertion.
        """
        parent, shared = self._parent(tokens)
        staged_nodes: list[Node] = []
        staged_by_key: dict[bytes, Node] = {}
        kv_allocations: list[tuple[int, int]] = []
        state_allocations: list[tuple[int, int]] = []
        new_states: dict[bytes, StateSnapshot] = {}
        state_keys: list[bytes] = []
        stored_states = {}

        try:
            # K/V storage is chunked independently of the input's original
            # allocation boundaries.
            cursor, leaf = shared, parent
            tail = len(tokens) - cursor
            blocks: dict = {}
            block_allocations: list[tuple[int, int]] = []
            if tail > 0:
                # one arena block a layer for the whole tail, copied once, then sliced per node: the keys and values
                # stay in the arena (their rows are contiguous, which also makes a resume one copy a layer) while the
                # save costs one copy a layer instead of one per node
                for layer, (keys, values) in kv.items():
                    k0 = self._rows(keys)[0][0]
                    v0 = self._rows(values)[0][0]
                    kt = self._allocate_tensor((tail, *k0.shape[1:]), k0.dtype, k0.device, block_allocations)
                    vt = self._allocate_tensor((tail, *v0.shape[1:]), v0.dtype, v0.device, block_allocations)
                    if kt is None or vt is None:
                        raise MemoryError("prefix-store arena is full")
                    kt.copy_(self._gather_rows(keys, cursor, len(tokens)))
                    vt.copy_(self._gather_rows(values, cursor, len(tokens)))
                    blocks[layer] = (kt, vt)
            if cursor < len(tokens):
                while cursor < len(tokens):
                    end = min(cursor + NODE, len(tokens))
                    block = tokens[cursor:end]
                    key = self._node_key(leaf, block, staged_by_key)
                    existing = staged_by_key.get(key) or self.nodes.get(key)
                    if (
                        existing is not None
                        and existing.parent is leaf
                        and existing.tokens == list(block)
                    ):
                        leaf = existing
                        cursor = end
                        continue
                    node = Node(block, key, leaf)
                    for layer in kv:
                        kt, vt = blocks[layer]
                        node.kv[layer] = (kt[cursor - shared: end - shared], vt[cursor - shared: end - shared])
                    node.allocations = list(block_allocations) if not staged_nodes else []
                    node.bytes = sum(t.numel() * t.element_size() for pair in node.kv.values() for t in pair)
                    staged_nodes.append(node)
                    staged_by_key[key] = node
                    leaf = node
                    cursor = end

            leaf = leaf or parent
            if leaf is None:
                raise ValueError("a session prefix store needs at least one token")

            # State snapshots have their own lifetime: retaining the last few
            # marks does not keep every obsolete mark on the K/V ancestor path.
            for position, state in states.items():
                if not 0 <= position <= len(tokens):
                    continue
                prefix = tokens[:position]
                key = self._state_key(position, prefix, new_states)
                record = new_states.get(key) or self.state_records.get(key)
                if record is None:
                    allocation_start = len(state_allocations)
                    value = self._copy_tree(state, state_allocations)
                    if value is None:
                        raise MemoryError("prefix-store arena is full")
                    allocations = state_allocations[allocation_start:]
                    record = StateSnapshot(position, prefix, value, allocations)
                    new_states[key] = record
                stored_states[position] = record.value
                state_keys.append(key)
        except MemoryError:
            self._rollback(kv_allocations)
            self._rollback(state_allocations)
            return None
        except Exception:
            self._rollback(kv_allocations)
            self._rollback(state_allocations)
            raise

        # Publish only after all data has been copied successfully.
        for node in staged_nodes:
            if node.parent is not None:
                node.parent.children[node.key] = node
            self.nodes[node.key] = node
        leaf.refs += 1
        if session is not None:
            leaf.sessions.add(session)
        for key, record in new_states.items():
            self.state_records[key] = record
            self.by_position.setdefault(record.position, set()).add(key)
        for key in state_keys:
            record = new_states.get(key) or self.state_records[key]
            record.refs += 1

        added_bytes = sum(node.bytes for node in staged_nodes) + sum(
            record.bytes for record in new_states.values()
        )
        self.used += added_bytes
        return leaf, stored_states, tuple(state_keys), added_bytes

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

    def match(self, tokens) -> tuple[StateSnapshot | None, int]:
        """Return the deepest saved state whose prefix tokens match."""
        best, pos = None, 0
        for record in self.state_records.values():
            mark = record.position
            if pos < mark <= len(tokens) and tuple(tokens[:mark]) == record.tokens:
                best, pos = record, mark
        return best, pos

    def entry_data(self, node: Node) -> dict:
        """Build an entry's K/V segment view from its prefix path."""
        layers = {}
        for part in self._path(node):
            for layer, (k, v) in part.kv.items():
                layers.setdefault(layer, ([], []))
                layers[layer][0].append((k, part.span, part.end))
                layers[layer][1].append((v, part.span, part.end))
        return layers

    def _drop(self, node: Node) -> int:
        self.nodes.pop(node.key, None)
        if node.parent is not None:
            node.parent.children.pop(node.key, None)
        for offset, size in node.allocations:
            self._free(offset, size)
        self.used -= node.bytes
        return node.bytes

    def release(self, node: Node, state_keys=()) -> int:
        """Release an entry's state marks and node reference, collecting unused storage."""
        freed = 0
        for key in state_keys:
            record = self.state_records.get(key)
            if record is None:
                continue
            record.refs -= 1
            if record.refs > 0:
                continue
            self.state_records.pop(key, None)
            holders = self.by_position.get(record.position)
            if holders is not None:
                holders.discard(key)
                if not holders:
                    self.by_position.pop(record.position, None)
            for offset, size in record.allocations:
                self._free(offset, size)
            self.used -= record.bytes
            freed += record.bytes
        node.refs -= 1
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
        self.state_records.clear()
        self.by_position.clear()
        self.used = 0
        self.free = (
            [(0, self.budget)] if self.buffer is not None and self.budget else []
        )

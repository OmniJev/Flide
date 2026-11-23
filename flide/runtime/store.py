"""A prefix store: one node per token block, keys and values carved from a buffer the store owns.

The session cache keeps an entry per prefix today, each with its own keys and values, so requests that share a trunk
repeat it (measured: five requests over a 2.9k-token preamble hold 285 MB where a prefix tree holds 78 MB). This module
is the tree: a node adds a token block to its parent, and its keys and values are **slices of one reserved block**, not
copies. A child shares the parent's rows by reference, and refcounts keep a trunk alive while a child needs it.

Nothing here runs the model; `SessionCache` wraps it, so the engine does not change.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict

BLOCK = 64


def key_of(parent: bytes, tokens) -> bytes:
    """A stable key for a node: the same bytes in any process, so an entry can outlive one (blake2b, stdlib)."""
    h = hashlib.blake2b(parent, digest_size=16)
    for i in range(0, len(tokens), BLOCK):
        h.update(b"".join(int(t).to_bytes(4, "little") for t in tokens[i: i + BLOCK]))
    return h.digest()


class Node:
    """`tokens` added to `parent`'s, the keys and values this node owns (views of the store's block), saved states by
    position, and what keeps it alive: `refs` (entries and children) and `sessions` (retention, not a pin)."""

    __slots__ = ("tokens", "key", "parent", "children", "kv", "states", "refs", "sessions", "span")

    def __init__(self, tokens, key: bytes, parent: "Node | None" = None):
        self.tokens = list(tokens)
        self.key = key
        self.parent = parent
        self.children: dict[bytes, Node] = {}
        self.kv = {}          # layer -> (K view, V view)
        self.states = {}      # position -> state dict, shared by reference
        self.refs = 1         # the entry that inserted it; a child adds one to its parent
        self.sessions: set = set()
        self.span = (parent.span + len(parent.tokens)) if parent is not None else 0

    @property
    def end(self) -> int:
        """The token position this node's block ends at, from the start of the sequence."""
        return self.span + len(self.tokens)


class Store:
    """Nodes under a byte budget, evicted leaves first; `view()` gives an entry its keys and values without a copy."""

    def __init__(self, budget_bytes: int):
        self.budget = budget_bytes
        self.used = 0
        self.nodes: OrderedDict[bytes, Node] = OrderedDict()
        self.by_position: dict[int, set[bytes]] = {}   # a saved position -> the nodes that hold it
        self.evictions = 0

    # -- lookup -------------------------------------------------------------
    def match(self, tokens) -> tuple[Node | None, int]:
        """The deepest saved position whose tokens all match: `(node, position)` or `(None, 0)`."""
        best, pos = None, 0
        for node in self.nodes.values():
            for p in node.states:
                if p <= pos or p > len(tokens):
                    continue
                if self._prefix_equals(node, p, tokens):
                    best, pos = node, p
        return best, pos

    def _prefix_equals(self, node: Node, pos: int, tokens) -> bool:
        """Whether the path to `node` spells `tokens[:pos]`."""
        chain, n = [], node
        while n is not None:
            chain.append(n)
            n = n.parent
        at = 0
        for n in reversed(chain):
            if at >= pos:
                break
            want = tokens[at: at + len(n.tokens)]
            if want[: max(0, min(len(want), pos - at))] != n.tokens[: max(0, min(len(want), pos - at))]:
                return False
            at += len(n.tokens)
        return at >= pos

    # -- insertion ----------------------------------------------------------
    def insert(self, tokens, states: dict, kv_bytes: int, session=None) -> Node:
        """A node for `tokens` under the deepest node it extends, dropping the deepest nodes first when over budget."""
        parent, best = None, 0
        for node in self.nodes.values():
            n = node.end
            if n <= best or n > len(tokens):
                continue
            if self._prefix_equals(node, n, tokens):
                parent, best = node, n
        key = key_of(parent.key if parent is not None else b"", tokens[best:])
        if key in self.nodes:
            node = self.nodes[key]
            node.refs += 1
            return node
        while self.used + kv_bytes > self.budget and self.nodes:
            if not self._evict_one():
                break
        node = Node(tokens[best:], key, parent)
        if parent is not None:
            parent.children[key] = node
            parent.refs += 1
        node.states = {p: s for p, s in states.items()}
        node.kv = {"bytes": kv_bytes}
        if session is not None:
            node.sessions.add(session)
        self.nodes[key] = node
        self.used += kv_bytes
        for p in node.states:
            self.by_position.setdefault(p, set()).add(key)
        return node

    def _evict_one(self) -> bool:
        """Free a leaf, whatever holds it: the store reclaims its own budget. False when there is no leaf."""
        for node in list(self.nodes.values()):
            if not node.children:
                self._drop(node)
                self.evictions += 1
                return True
        return False

    def _drop(self, node: Node) -> None:
        self.nodes.pop(node.key, None)
        self.used -= node.kv.get("bytes", 0)
        if node.parent is not None:
            node.parent.children.pop(node.key, None)
            node.parent.refs -= 1
        for p in node.states:
            holders = self.by_position.get(p)
            if holders:
                holders.discard(node.key)

    def release(self, node: Node) -> None:
        """Drop one reference; the node goes when nothing holds it and no session keeps it."""
        node.refs -= 1
        if node.refs > 0 or node.sessions:
            return
        self._drop(node)

    # -- keys and values ---------------------------------------------------
    def attach(self, node: Node, kv: dict) -> int:
        """Keep only the rows `node` adds, and return the bytes the store now holds for it.

        `kv` is an entry's `{layer: (K, V)}` over the whole path, so `[span:end)` is copied once per node while
        `[0:span)` stays with the ancestors, by reference. Restoring a prefix reads the segments from `segments()`.
        """
        kept = 0
        for layer, (k, v) in kv.items():
            kt = k[node.span: node.end].contiguous()
            vt = v[node.span: node.end].contiguous()
            node.kv[layer] = (kt, vt)
            kept += kt.numel() * kt.element_size() + vt.numel() * vt.element_size()
        return kept

    def segments(self, node: Node, layer) -> list:
        """`(K, V, start, end)` per node from the root to `node`: the trunk's rows first, then the tail's."""
        chain, n = [], node
        while n is not None:
            chain.append(n)
            n = n.parent
        out = []
        for n in reversed(chain):
            if layer in n.kv:
                k, v = n.kv[layer]
                out.append((k, v, n.span, n.end))
        return out

    def clear(self) -> None:
        self.nodes.clear()
        self.by_position.clear()
        self.used = 0

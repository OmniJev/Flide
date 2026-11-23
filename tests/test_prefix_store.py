"""The shared prefix store: one arena, shared K/V tails, and refcounted release."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.runtime.session import Entry, SessionCache
from flide.runtime.store import Store, key_of


def test_store_shares_prefix_rows_in_one_arena():
    store = Store(1 << 20)
    store.configure("cpu")
    k = torch.arange(100, dtype=torch.float32).reshape(100, 1)
    v = k + 1000
    root, root_states, root_state_keys, _ = store.insert(
        list(range(100)), {64: {"s": torch.tensor([64.0])}}, {0: (k, v)}
    )
    child_k, child_v = (
        torch.arange(150, dtype=torch.float32).reshape(150, 1),
        torch.arange(150, dtype=torch.float32).reshape(150, 1) + 1000,
    )
    child, child_states, _, _ = store.insert(
        [*range(150)], {128: {"s": torch.tensor([128.0])}}, {0: (child_k, child_v)}
    )

    assert root is not None and child is not None
    assert child.parent is root
    assert child.tokens == list(range(100, 150))
    assert store.used == 1208  # 100 rows of K/V + 50 rows of K/V + two 4-byte states
    parts = store.entry_data(child)[0][0]
    assert [(lo, hi) for _, lo, hi in parts] == [(0, 64), (64, 100), (100, 150)]
    assert parts[0][0].data_ptr() == root.parent.kv[0][0].data_ptr()
    assert parts[1][0].data_ptr() == root.kv[0][0].data_ptr()
    assert parts[2][0].data_ptr() == child.kv[0][0].data_ptr()
    assert torch.equal(torch.cat([part[0] for part in parts[:2]]), k)
    assert torch.equal(parts[2][0], child_k[100:])
    assert torch.equal(child_states[128]["s"], torch.tensor([128.0]))


def test_siblings_share_full_blocks_before_their_divergence():
    store = Store(1 << 20)
    store.configure("cpu")
    common = list(range(100))

    def kv(tokens):
        keys = torch.arange(len(tokens), dtype=torch.float32).reshape(-1, 1)
        return {0: (keys, keys + 1)}

    first_tokens = common + list(range(1000, 1100))
    second_tokens = common + list(range(2000, 2100))
    first, _, _, _ = store.insert(first_tokens, {}, kv(first_tokens))
    assert store._parent(second_tokens)[1] == 64
    second, _, _, _ = store.insert(second_tokens, {}, kv(second_tokens))

    assert first is not None and second is not None
    assert store.used == (len(first_tokens) + len(second_tokens) - 64) * 8
    first_root, second_root = first, second
    while first_root.parent is not None:
        first_root = first_root.parent
    while second_root.parent is not None:
        second_root = second_root.parent
    assert first_root is second_root
    assert first_root.end == 64
    segments = store.entry_data(second)[0][0]
    assert [(lo, hi) for _, lo, hi in segments] == [
        (0, 64),
        (64, 128),
        (128, 192),
        (192, 200),
    ]


def test_session_cache_reuses_trunk_across_siblings():
    cache = SessionCache(1 << 20, min_tokens=1, stride=64, margin=8)
    cache.attach_prefix_store("cpu")

    def make_entry(tokens, mark):
        k = torch.arange(len(tokens), dtype=torch.float32).reshape(-1, 1)
        v = k + 1
        return Entry(tokens, {mark: {"s": torch.tensor([float(mark)])}}, {0: (k, v)})

    common = list(range(100))
    root = make_entry(common, 64)
    cache.insert(root)
    first = make_entry(common + list(range(100, 150)), 128)
    cache.insert(first, parent=root)
    sibling = make_entry(common + list(range(1000, 1050)), 128)
    cache.insert(sibling, parent=first)

    assert cache.used == 1612
    assert cache.pstore is not None
    assert cache.pstore.used == cache.used
    assert len(cache.pstore.nodes) == 4  # two shared blocks plus distinct tails
    assert sum(node.span == 0 for node in cache.pstore.nodes.values()) == 1
    assert sum(node.span == 64 for node in cache.pstore.nodes.values()) == 1
    assert sum(node.span == 100 for node in cache.pstore.nodes.values()) == 2

    cache.clear()
    assert cache.used == 0
    assert cache.pstore.used == 0
    assert not cache.pstore.nodes


def test_agent_releases_obsolete_state_marks():
    cache = SessionCache(1 << 20, min_tokens=1, stride=64, margin=8)
    cache.attach_prefix_store("cpu")
    parent = None
    for mark in range(64, 64 * 9, 64):
        tokens = list(range(mark + 8))
        keys = torch.arange(len(tokens), dtype=torch.float32).reshape(-1, 1)
        entry = Entry(
            tokens,
            {mark: {"s": torch.full((256,), float(mark))}},
            {0: (keys, keys + 1)},
        )
        cache.insert(entry, parent=parent)
        parent = entry

    assert cache.stats()["entries"] == 1
    assert sorted(
        record.position for record in cache.pstore.state_records.values()
    ) == [
        64 * 7,
        64 * 8,
    ]
    assert (
        sum(record.bytes for record in cache.pstore.state_records.values())
        == 2 * 256 * 4
    )


def test_store_reclaims_parent_after_last_child_reference():
    store = Store(1 << 16)
    store.configure("cpu")
    root_kv = {0: (torch.zeros(64, 1), torch.zeros(64, 1))}
    root, _, root_state_keys, _ = store.insert(
        list(range(64)), {32: {"s": torch.zeros(2)}}, root_kv
    )
    child, _, child_state_keys, _ = store.insert(
        list(range(96)),
        {80: {"s": torch.zeros(2)}},
        {0: (torch.zeros(96, 1), torch.zeros(96, 1))},
    )
    assert root is not None and child is not None
    before = store.used
    assert (
        store.release(root, root_state_keys) == 8
    )  # child still depends on the K/V trunk
    freed = store.release(child, child_state_keys)
    assert freed + 8 == before
    assert store.used == 0
    assert not store.nodes


def test_store_reuses_one_arena_for_mixed_tensor_dtypes():
    store = Store(4096)
    store.configure("cpu")
    keys = torch.arange(16, dtype=torch.float16).reshape(16, 1)
    values = torch.arange(16, dtype=torch.bfloat16).reshape(16, 1)
    state = {
        "s": torch.arange(4, dtype=torch.float32),
        "conv": [torch.ones(2, dtype=torch.bfloat16)],
    }
    node, states, _, _ = store.insert(list(range(16)), {8: state}, {0: (keys, values)})

    assert node is not None
    layers = store.entry_data(node)
    assert store.used == 16 * 2 + 16 * 2 + 4 * 4 + 2 * 2
    assert torch.equal(layers[0][0][0][0], keys)
    assert torch.equal(layers[0][1][0][0], values)
    assert torch.equal(states[8]["s"], state["s"])
    assert torch.equal(states[8]["conv"][0], state["conv"][0])


def test_session_cache_evicts_a_leaf_when_shared_arena_is_full():
    cache = SessionCache(1900, min_tokens=1, stride=64, margin=8)
    cache.attach_prefix_store("cpu")

    def make_entry(tokens, mark):
        k = torch.arange(len(tokens), dtype=torch.float32).reshape(-1, 1)
        return Entry(
            tokens, {mark: {"s": torch.tensor([float(mark)])}}, {0: (k, k + 1)}
        )

    root_tokens = list(range(100))
    root = make_entry(root_tokens, 64)
    cache.insert(root)
    first = make_entry(root_tokens + list(range(100, 150)), 128)
    cache.insert(first, parent=root)
    second = make_entry(root_tokens + list(range(1000, 1050)), 128)
    cache.insert(second, parent=first)
    assert cache.used == 1612

    third = make_entry(root_tokens + list(range(2000, 2050)), 128)
    cache.insert(third, parent=second)
    assert cache.evictions == 1
    assert cache.used <= cache.budget
    assert cache.pstore.used == cache.used
    assert any(node.span == 0 for node in cache.pstore.nodes.values())


def test_store_rejects_oversized_insert_without_leaking_arena_ranges():
    store = Store(128)
    store.configure("cpu")
    oversized = store.insert(
        list(range(64)), {}, {0: (torch.zeros(64, 4), torch.zeros(64, 4))}
    )
    assert oversized is None
    assert store.used == 0
    assert store.free == [(0, 128)]


def test_keys_are_stable_and_content_sensitive():
    assert key_of(b"", list(range(10))) == key_of(b"", list(range(10)))
    assert key_of(b"", list(range(10))) != key_of(b"", list(range(11)))
    assert key_of(b"a", [10, 20]) != key_of(b"b", [10, 20])

"""Run the requests of a round that share a prefix as one multi-branch plan.

Eight requests over one 6,045-token prefix: 1,224 ms as separate plans, 174 ms as one plan with eight branches, and the
outputs are bit-for-bit equal **when the invariant products are on** (without them the merged shape rounds differently,
6.2e-02, so merging is an approximation there). See #62.

A group is only formed when the plans are the same computation around different branches: same prefix, same attention
mode, same positions, no media. Everything else passes through untouched, and the outputs are split back so each plan's
`data` and `finish` are exactly as before.
"""
from __future__ import annotations

from dataclasses import replace


def _reads(plan) -> list:
    """One read tuple per branch, whether the plan carries a shared tuple or a list."""
    return list(plan.reads) if isinstance(plan.reads, list) else [plan.reads] * len(plan.suffixes)


def _key(plan):
    if plan.vision is not None or not plan.prefix or not plan.suffixes:
        return None
    positions = None if plan.positions is None else tuple(int(v) for v in plan.positions.reshape(-1).tolist())
    return (tuple(plan.prefix), plan.attention, plan.offset, positions)


def group(plans):
    """`[(plan to run, [(original index, branches it contributes), ...])]`, in the order the plans were given.

    Plans that cannot share a prefix appear alone, with their own branches, so the caller runs the same list it would
    have run without this module.
    """
    groups: dict = {}
    order = []
    for i, plan in enumerate(plans):
        if plan is None:
            order.append((i, None))
            continue
        key = _key(plan)
        if key is None:
            order.append((i, None))
            continue
        if key not in groups:
            groups[key] = []
            order.append((i, key))
        groups[key].append(i)

    out = []
    for i, key in order:
        if key is None:
            out.append((plans[i], [(i, len(plans[i].suffixes))] if plans[i] is not None else []))
            continue
        idx = groups[key]
        suffixes, reads = [], []
        for j in idx:
            suffixes.extend(plans[j].suffixes)
            reads.extend(_reads(plans[j]))
        merged = replace(plans[i], suffixes=suffixes, reads=reads)
        out.append((merged, [(j, len(plans[j].suffixes)) for j in idx]))
    return out


def split(outputs, groups, n=None):
    """`out[original index]` = that plan's own branch outputs, taken back out of the merged ones."""
    out = [None] * (n if n is not None else sum(len(parts) for _, parts in groups))
    for slot, (_, parts) in enumerate(groups):
        merged = outputs[slot]
        at = 0
        for i, n in parts:
            out[i] = merged[at: at + n]
            at += n
    return out

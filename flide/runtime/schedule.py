"""Cut plans into micro-batches under a token budget.

Plans are sorted by token count so a micro-batch holds plans of similar size. A plan larger than the
budget is split by its suffixes into pieces that each carry the full prefix; a piece that still exceeds
the budget (one long branch) gets a micro-batch of its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..formats.base import Plan, Read
from .pack import Job


@dataclass
class Piece:
    request: int
    branches: list[int]
    job: Job


@dataclass
class MicroBatch:
    pieces: list[Piece] = field(default_factory=list)
    tokens: int = 0


def _piece(i: int, plan: Plan, branches: list[int], session: dict | None = None, first: bool = True) -> Piece:
    """`session` (start, resume, marks) resumes the prefix from a saved state; only the first piece of a plan
    saves states at the marks."""
    extra = {}
    if session:
        extra = {"start": session.get("start", 0), "resume": session.get("resume"),
                 "marks": session.get("marks", ()) if first else ()}
    job = Job(plan.prefix, [plan.suffixes[b] for b in branches], [plan.branch_reads(b) for b in branches],
              plan.vision, plan.positions, plan.offset, blocks=plan.attention == "blocks", **extra)
    return Piece(i, branches, job)


def split_plan(i: int, plan: Plan, budget: int, session: dict | None = None) -> list[Piece]:
    if not plan.prefix:
        if plan.vision is not None:
            raise ValueError("a plan without a prefix takes no media")
        blocks = plan.attention == "blocks"
        out = []
        for b, s in enumerate(plan.suffixes):
            if len(s) < 2:
                raise ValueError("a sequence of its own needs at least two tokens")
            if blocks:
                reads = tuple(Read(r.head, r.layer, tuple(p - 1 if p < 0 else p for p in r.positions))
                              for r in plan.branch_reads(b))
                out.append(Piece(i, [b], Job(s, [s[-1:]], [reads], blocks=True)))
            else:
                out.append(Piece(i, [b], Job(s[:-1], [s[-1:]], [plan.branch_reads(b)])))
        return out
    if plan.tokens - (session or {}).get("start", 0) <= budget:
        return [_piece(i, plan, list(range(len(plan.suffixes))), session)]
    P = len(plan.prefix) - (session or {}).get("start", 0)
    pieces, cur, size = [], [], P
    for b, s in enumerate(plan.suffixes):
        if cur and size + len(s) > budget:
            pieces.append(_piece(i, plan, cur, session, first=not pieces))
            cur, size = [], P
        cur.append(b)
        size += len(s)
    pieces.append(_piece(i, plan, cur, session, first=not pieces))
    return pieces


def schedule(plans: Sequence[Plan | None], budget: int, sessions: dict | None = None) -> list[MicroBatch]:
    pieces = []
    for i, p in enumerate(plans):
        if p is not None:
            pieces.extend(split_plan(i, p, budget, (sessions or {}).get(i)))
    pieces.sort(key=lambda x: x.job.tokens)
    batches: list[MicroBatch] = []
    cur = MicroBatch()
    for piece in pieces:
        t = piece.job.tokens
        if cur.pieces and cur.tokens + t > budget:
            batches.append(cur)
            cur = MicroBatch()
        cur.pieces.append(piece)
        cur.tokens += t
    if cur.pieces:
        batches.append(cur)
    return batches

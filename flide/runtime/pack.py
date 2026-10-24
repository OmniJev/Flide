"""A micro-batch as one flat token stream, with no batch dimension and no padding.

A job is one shared prefix plus n suffixes (the branches of one plan, or part of them). The stream holds
every prefix of the micro-batch first and every suffix after them:

    [prefix_0 | prefix_1 | ... | suffix_0_0 | suffix_0_1 | ... | suffix_1_0 | ...]

Per-token layers do not care about the order. Linear-attention layers run in two calls, one over the
prefix part and one over the suffix part, so both parts are contiguous. Full attention reads the job,
branch and position of every token from this metadata.

Sessions. A job can resume from a saved state (`start` > 0, `resume`): its prefix tokens before `start` are not
in the stream, the backbone loads their recurrent state and keys instead. A job can also ask for its state at
`marks` inside the prefix. Then the prefix part is cut into levels at the marks,

    [level 0 of every job | level 1 | ... | suffixes]

and linear-attention layers run one call per level, each starting from the state the level before left. Full
attention then reads prefix keys from a context buffer that holds every job's whole prefix in order (saved
keys first), so a job's prefix needs no contiguous span of the stream.

Reads become gather indices into the stream, one index tensor per layer key, and a list of groups that
say which rows of which layer go to which head and back to which branch and read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from ..formats.base import Read

if TYPE_CHECKING:
    import torch


@dataclass
class Job:
    prefix: list[int]
    suffixes: list[list[int]]
    reads: list[tuple[Read, ...]]
    vision: Any = None
    positions: Any = None
    offset: int = 0
    start: int = 0
    resume: Any = None
    marks: tuple[int, ...] = ()

    @property
    def tokens(self) -> int:
        """Tokens this job puts in the stream."""
        return len(self.prefix) - self.start + sum(len(s) for s in self.suffixes)


@dataclass
class ReadGroup:
    layer: int | None
    head: str | None
    start: int
    end: int
    owners: list[tuple[int, int, int]] = field(default_factory=list)


@dataclass
class PackedBatch:
    input_ids: "torch.Tensor"
    positions: "torch.Tensor"
    job: "torch.Tensor"
    branch: "torch.Tensor"
    cu_prefix: "torch.Tensor"
    cu_suffix: "torch.Tensor"
    cu_prefix_cpu: "torch.Tensor"
    cu_suffix_cpu: "torch.Tensor"
    branch_job: "torch.Tensor"
    reads: dict
    groups: list[ReadGroup]
    n_prefix_tokens: int
    n_jobs: int
    n_branches: int
    key_ranges: "torch.Tensor | None" = None
    embeds: "torch.Tensor | None" = None
    pos3: "torch.Tensor | None" = None
    vision: list = field(default_factory=list)
    cache: dict = field(default_factory=dict)
    levels: list = field(default_factory=list)
    ctx: dict | None = None
    resumes: list = field(default_factory=list)
    marks: dict = field(default_factory=dict)
    saved: dict = field(default_factory=dict)
    jobs: list = field(default_factory=list)
    attn_runs: int = 0

    @property
    def n_tokens(self) -> int:
        return int(self.input_ids.shape[0])


def pack(jobs: Sequence[Job], device) -> PackedBatch:
    import numpy as np
    import torch

    from ..kernels.attention import runs

    bounds = []
    for j, job in enumerate(jobs):
        if not job.prefix or not job.suffixes or any(len(s) == 0 for s in job.suffixes):
            raise ValueError(f"job {j} needs a non-empty prefix and non-empty suffixes")
        P = len(job.prefix)
        if not 0 <= job.start < P or (job.start > 0) != (job.resume is not None):
            raise ValueError(f"job {j}: resume start {job.start} needs a saved state and must be inside its {P} prefix tokens")
        if (job.start or job.marks) and (job.positions is not None or job.offset or job.vision is not None):
            raise ValueError(f"job {j}: sessions take text-only prefixes")
        bounds.append([job.start] + sorted({m for m in job.marks if job.start < m < P}) + [P])
    split = any(len(b) > 2 or b[0] > 0 for b in bounds)
    ctx_start, at = [], 0
    for job in jobs:
        ctx_start.append(at)
        at += len(job.prefix)
    T = sum(job.tokens for job in jobs)
    ids = np.empty(T, np.int64)
    pos = np.empty(T, np.int64)
    job_of = np.empty(T, np.int64)
    branch_of = np.empty(T, np.int64)
    ranges = np.empty((T, 4), np.int64)
    n = 0
    segs: list[list[tuple[int, int, int]]] = [[] for _ in jobs]
    levels = []
    for k in range(max(len(b) - 1 for b in bounds)):
        lo = n
        lens, owners, ends = [], [], []
        for j, job in enumerate(jobs):
            if k + 1 >= len(bounds[j]):
                continue
            a, e = bounds[j][k], bounds[j][k + 1]
            m = e - a
            base = ctx_start[j] if split else n - a
            segs[j].append((a, e, n))
            r = ranges[n: n + m]
            r[:, 0] = base
            r[:, 1] = np.arange(base + a + 1, base + e + 1)
            r[:, 2:] = 0
            ids[n: n + m] = job.prefix[a:e]
            pos[n: n + m] = np.arange(a, e)
            job_of[n: n + m] = j
            branch_of[n: n + m] = -1
            n += m
            lens.append(m)
            owners.append(j)
            if k + 2 < len(bounds[j]):
                ends.append((len(owners) - 1, j, e))
        cu = [0]
        for m in lens:
            cu.append(cu[-1] + m)
        levels.append({"lo": lo, "hi": n, "cu_list": cu, "jobs_list": owners, "ends": ends})
    n_prefix = n

    def stream_index(j: int, a: int) -> int:
        for lo, hi, st in segs[j]:
            if lo <= a < hi:
                return st + a - lo
        raise ValueError(f"read at prefix position {a} of job {j}, which resumed from a saved state at {jobs[j].start}")

    cu_s = [0]
    branch_job: list[int] = []
    gather: dict = {}
    groups: dict[tuple, ReadGroup] = {}
    pending: dict[tuple, list] = {}
    b = 0
    for j, job in enumerate(jobs):
        P = len(job.prefix)
        p_lo = ctx_start[j] if split else segs[j][0][2]
        for s, reads in zip(job.suffixes, job.reads):
            start, m = n, len(s)
            r = ranges[n: n + m]
            r[:, 0] = p_lo
            r[:, 1] = p_lo + P
            r[:, 2] = start
            r[:, 3] = np.arange(start + 1, start + m + 1)
            ids[n: n + m] = s
            pos[n: n + m] = np.arange(P, P + m)
            job_of[n: n + m] = j
            branch_of[n: n + m] = b
            n += m
            cu_s.append(cu_s[-1] + m)
            branch_job.append(j)
            L = P + m
            for ri, read in enumerate(reads):
                rows = []
                for p in read.positions:
                    a = L + p if p < 0 else p
                    if not 0 <= a < L:
                        raise ValueError(f"read position {p} outside a branch of {L} tokens")
                    rows.append(stream_index(j, a) if a < P else start + a - P)
                pending.setdefault((read.layer, read.head), []).append((b, ri, rows))
            b += 1
    for (layer, head), items in pending.items():
        idx = gather.setdefault(layer, [])
        g = groups[(layer, head)] = ReadGroup(layer, head, len(idx), len(idx))
        for br, ri, rows in items:
            idx.extend(rows)
            g.owners.append((br, ri, len(rows)))
        g.end = len(idx)

    def t(x):
        return torch.tensor(x, dtype=torch.long, device=device)

    def arr(x, dtype=np.int64):
        return torch.from_numpy(np.ascontiguousarray(x, dtype=dtype)).to(device)

    pos3 = None
    if any(j.positions is not None or j.offset for j in jobs):
        cols = []
        for job in jobs:
            P = len(job.prefix)
            if job.positions is not None:
                cols.append(torch.as_tensor(job.positions, dtype=torch.long).reshape(3, P).cpu())
            else:
                cols.append(torch.arange(P).expand(3, P))
        for job in jobs:
            P = len(job.prefix)
            for s in job.suffixes:
                cols.append(torch.arange(P + job.offset, P + job.offset + len(s)).expand(3, len(s)))
        pos3 = torch.cat(cols, 1).to(device)

    for lv in levels:
        lv["cu_cpu"] = torch.tensor(lv["cu_list"], dtype=torch.long)
        lv["cu"] = lv["cu_cpu"].to(device)
        lv["jobs"] = t(lv["jobs_list"])
    ctx = None
    if split:
        plen = np.array([len(job.prefix) for job in jobs], np.int64)
        starts = np.array(ctx_start, np.int64)
        index = starts[job_of[:n_prefix]] + pos[:n_prefix]
        cjob = np.repeat(np.arange(len(jobs)), plen)
        cpos = np.arange(at) - np.repeat(starts, plen)
        ctx = {"len": at, "start": ctx_start, "P": plen.tolist(), "index": arr(index),
               "meta": (arr(cjob, np.int32), torch.full((at,), -1, dtype=torch.int32, device=device), arr(cpos, np.int32))}
    cu_p = levels[0]["cu_list"] if len(levels) == 1 else [0]
    return PackedBatch(input_ids=arr(ids), positions=arr(pos), job=arr(job_of), branch=arr(branch_of),
                       cu_prefix=t(cu_p), cu_suffix=t(cu_s),
                       cu_prefix_cpu=torch.tensor(cu_p, dtype=torch.long), cu_suffix_cpu=torch.tensor(cu_s, dtype=torch.long),
                       branch_job=t(branch_job), reads={k: t(v) for k, v in gather.items()},
                       groups=list(groups.values()), n_prefix_tokens=n_prefix, n_jobs=len(jobs), n_branches=b,
                       key_ranges=arr(ranges, np.int32), pos3=pos3,
                       vision=[j.vision for j in jobs if j.vision is not None], levels=levels, ctx=ctx,
                       resumes=[(j, job.resume, job.start) for j, job in enumerate(jobs) if job.start],
                       marks={j: tuple(bounds[j][1:-1]) for j in range(len(jobs)) if len(bounds[j]) > 2},
                       jobs=list(jobs), attn_runs=runs(ranges, by_row=True))

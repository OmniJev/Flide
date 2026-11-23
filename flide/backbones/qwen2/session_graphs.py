"""Causal Qwen2 CUDA graphs with KV rows saved in Flide's session arena."""
from __future__ import annotations

import bisect

import numpy as np
import torch

from ... import settings
from ...formats.base import FINAL, Read
from ...runtime.pack import Job, pack
from .ops import ArenaContext

CONTEXT_LADDER = (512, 1024, 2048, 4096, 8192, 16384)
MAX_STREAM = 4096


class SessionGraphs:
    def __init__(self, graphs, store, sessions):
        self.graphs, self.store, self.cache = graphs, store, sessions
        self.host = torch.empty((2, CONTEXT_LADDER[-1]), dtype=torch.int32, pin_memory=True)
        self.indices = torch.empty_like(self.host, device=graphs.device)
        self.host_array = self.host.numpy()

    def shape(self, items):
        if not items:
            return None
        context = sum(len(pre) for pre, *_ in items)
        active = sum(len(pre) - start + sum(map(len, suffixes)) for pre, suffixes, _, start, _ in items)
        reads = sum(len(item[1]) for item in items)
        size = self.graphs.flat_rung(len(items), active, reads, limit_padding=False)
        bucket = bisect.bisect_left(CONTEXT_LADDER, context)
        if size is None or size > MAX_STREAM or bucket == len(CONTEXT_LADDER):
            return None
        return size, CONTEXT_LADDER[bucket]

    def _stage_dummy(self, size, context):
        self.graphs._stage([([0], [[0]], None)], size)
        self.indices[:, :context].fill_(-1)

    def _stage(self, items, saves, size, context):
        g = self.graphs
        g.staged.synchronize()
        reads = (Read(None),)
        jobs = [Job(pre, suffixes, [reads] * len(suffixes), start=start, resume=entry, marks=(mark,) if mark else ())
                for pre, suffixes, entry, start, mark in items]
        batch = pack(jobs, "cpu")
        ctx = batch.ctx
        # A warm session graph always has a resume or a requested save point. On a cold miss that has neither,
        # treat the ordinary leading prefix rows as the context without changing the stream's ranges.
        if ctx is None:
            starts = np.cumsum([0] + [len(item[0]) for item in items[:-1]])
            prefix_rows = np.arange(batch.n_prefix_tokens, dtype=np.int32)
        else:
            starts, prefix_rows = ctx["start"], ctx["index"].numpy()
        source, destination = self.host_array[:, :context]
        source.fill(-1)
        destination.fill(-1)
        source[prefix_rows] = g.offsets[:batch.n_prefix_tokens]
        for i, ((_, _, entry, start, _), (mark, save)) in enumerate(zip(items, saves)):
            at = starts[i]
            if start:
                source[at:at + start] = -2 - entry.kv.rows["full"][:start]
            if save is not None:
                destination[at:at + mark] = save.rows["full"]

        data = g.host_array[:8 * size].reshape(8, size)
        n = batch.n_tokens
        data.fill(0)
        data[0, :n] = batch.input_ids.numpy()
        data[1, :n] = batch.positions.numpy()
        data[2, :n] = batch.job.numpy()
        data[3, :n] = batch.branch.numpy()
        data[4:, :n] = batch.key_ranges.numpy().T
        reads = g.host_array[8 * size:8 * size + g._read_limit(size)]
        reads.fill(size - 1)
        actual = batch.reads[FINAL].numpy()
        reads[:len(actual)] = actual
        used = 8 * size + len(reads)
        g.inputs[:used].copy_(g.host[:used], non_blocking=True)
        self.indices[:, :context].copy_(self.host[:, :context], non_blocking=True)
        g.staged.record()
        return len(actual)

    def _body(self, size, context):
        g = self.graphs
        ids, pos, job, branch = g.inputs[:4 * size].view(4, size)
        ranges = g.inputs[4 * size:8 * size].view(4, size)
        reads = g.inputs[8 * size:8 * size + g._read_limit(size)].long()
        hidden = g.backbone.lm.embed_tokens(ids)
        ctx = ArenaContext(self.store, *self.indices[:, :context])
        groups = 2 * g._request_limit(size) + g._read_limit(size) + 1
        return g.backbone.decode(hidden, pos, (job, branch, pos), ranges, groups, {FINAL: reads}, ctx)[FINAL]

    @torch.inference_mode()
    def run(self, items):
        shape = self.shape(items)
        if shape is None:
            return None
        g = self.graphs
        with g.lock, torch.cuda.device(g.device):
            torch.cuda.current_stream().wait_event(g.done)
            saves = []
            submitted = False
            try:
                key = (*shape, settings.batch_invariant)
                captured = g._record(key, lambda: self._stage_dummy(*shape), lambda: self._body(*shape))
                if captured is None:
                    return None
                keep = [item[2] for item in items]
                for pre, _, _, start, mark in items:
                    rows = self.cache.reserve({"full": mark}, keep=keep) if mark and start < mark < len(pre) else None
                    saves.append((mark, rows))
                n = self._stage(items, saves, *shape)
                graph, output = captured
                graph.replay()
                result = output[:n].clone()
                snaps = [[(mark, ({mark: {}}, rows))] if rows is not None else [] for mark, rows in saves]
                submitted = True
                return result, snaps
            finally:
                if not submitted:
                    for _, rows in saves:
                        if rows is not None:
                            rows.free()
                g.done.record()

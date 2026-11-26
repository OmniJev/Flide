"""Bucketed CUDA graphs for full-attention Qwen text decisions."""
from __future__ import annotations

import bisect
import logging
import threading
import time

import numpy as np
import torch

from ... import settings
from ...formats.base import FINAL
from ...runtime.tp import capture_mode
from ..base import capture_progress

log = logging.getLogger("flide.graphs")

LADDER = (32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 4096, 5120, 6144, 8192)
MAX_READS = 256
FULL = "full"   # graphs that read every row of the stream (a head over all positions), captured when first needed


class Qwen2Graphs:
    media = False
    block_attention = True
    prefix_attention = True
    read_spots = True
    flat_max = LADDER[-1]

    def __init__(self, backbone):
        self.backbone = backbone
        self.device = backbone.device
        self.graphs = {}
        self.failed = set()
        self.sessions = None
        self.lock = threading.Lock()
        with torch.cuda.device(self.device):
            self.pool = torch.cuda.graph_pool_handle()
            self.staged = torch.cuda.Event()
            self.done = torch.cuda.Event()
            self.host = torch.zeros(9 * LADDER[-1], dtype=torch.int32, pin_memory=True)
            self.inputs = torch.empty_like(self.host, device=self.device)
        self.host_array = self.host.numpy()
        self.offsets = np.arange(LADDER[-1], dtype=np.int32)

    @staticmethod
    def _read_limit(tokens):
        return min(MAX_READS, max(8, tokens // 32))

    @staticmethod
    def _request_limit(tokens):
        return min(64, max(1, tokens // 16))

    @staticmethod
    def flat_cost(item):
        prefix, suffixes = item[:2]
        if len(item) < 5 or item[4] is None:
            reads = len(suffixes)
        else:
            reads = len(item[4]) if len(suffixes) == 1 else sum(map(len, item[4]))
        return len(prefix) + sum(map(len, suffixes)), reads

    def _rung(self, requests, tokens, reads, *, limit_padding=True, branches=0):
        """(size, full) of the graph that runs that many requests, tokens and reads (`branches` of them), or None. A graph that
        reads every row (`full`) is taken where the reads a smaller graph has room for would mean a rung over twice as
        large: a head over all positions asks for as many reads as tokens."""
        if requests == 0 or reads <= 0:
            return None
        first = bisect.bisect_left(LADDER, tokens)
        small = None
        for size in LADDER[first:]:
            if limit_padding and tokens >= 1024 and size - tokens > tokens // 8:
                break   # the packed path avoids running the extra rows through every decoder layer
            if requests > self._request_limit(size):
                continue
            if small is None and reads <= size and branches <= MAX_READS:
                small = size
            if reads <= self._read_limit(size):
                return (size, False) if small is None or size <= 2 * small else (small, True)
        return (small, True) if small is not None else None

    def flat_rung(self, requests, tokens, reads, *, limit_padding=True):
        """The rung of the graphs that read at most `_read_limit` rows, or None: the size of a round the engine composes (and
        of the session graphs)."""
        if requests == 0 or not 0 < reads <= MAX_READS:
            return None
        first = bisect.bisect_left(LADDER, tokens)
        for size in LADDER[first:]:
            if limit_padding and tokens >= 1024 and size - tokens > tokens // 8:
                return None   # the packed path avoids running the extra rows through every decoder layer
            if requests <= self._request_limit(size) and reads <= self._read_limit(size):
                return size
        return None

    def whole(self, item) -> bool:
        """A request whose reads take a graph of every row (`FULL`): it runs alone."""
        tokens, reads = self.flat_cost(item)
        rung = self._rung(1, tokens, reads, branches=len(item[1]))
        return rung is not None and rung[1]

    @staticmethod
    def fits(prefix_len, suffix_lens):
        return False   # all graph replays go through run_flat

    def _stage(self, items, size, full=False):
        self.staged.synchronize()
        data = self.host_array[:8 * size].reshape(8, size)
        ids, pos, job, branch, lo1, hi1, lo2, hi2 = data
        reads = self.host_array[8 * size:8 * size + (size if full else self._read_limit(size))]
        data.fill(0)
        reads.fill(size - 1)
        at = 0
        starts = []
        for j, (prefix, _, _, *tail) in enumerate(items):
            n = len(prefix)
            sl = slice(at, at + n)
            ids[sl], pos[sl], job[sl], branch[sl], lo1[sl] = prefix, self.offsets[:n], j, -1, at
            hi1[sl] = at + n if tail and tail[0] else at + self.offsets[1:n + 1]
            starts.append(at)
            at += n
        read_at = 0
        branch_id = 0
        for j, (prefix, suffixes, _, *tail) in enumerate(items):
            length = len(prefix)
            mode = tail[0] if tail else False
            spots = tail[1] if len(tail) > 1 else None
            for b, suffix in enumerate(suffixes):
                end = at + len(suffix)
                sl = slice(at, end)
                ids[sl], pos[sl], job[sl], branch[sl] = suffix, length + self.offsets[:len(suffix)], j, branch_id
                lo1[sl], hi1[sl], lo2[sl] = starts[j], starts[j] + length, at
                hi2[sl] = end if mode is True else at + self.offsets[1:len(suffix) + 1]
                if spots is None:
                    reads[read_at] = end - 1
                    read_at += 1
                else:
                    offsets = np.asarray(spots if len(suffixes) == 1 else spots[b], dtype=np.int32)
                    if len(suffixes) == 1:
                        rows = np.where(offsets < length, starts[j] + offsets, at + offsets - length)
                    else:
                        rows = at + offsets
                    reads[read_at:read_at + len(offsets)] = rows
                    read_at += len(offsets)
                at = end
                branch_id += 1
        used = 8 * size + len(reads)
        self.inputs[:used].copy_(self.host[:used], non_blocking=True)
        self.staged.record()
        return read_at

    def _body(self, size, groups, full=False):
        ids, pos, job, branch = self.inputs[:4 * size].view(4, size)
        reads = self.inputs[8 * size:8 * size + (size if full else self._read_limit(size))].long()
        hidden = self.backbone.lm.embed_tokens(ids)
        ranges = self.inputs[4 * size:8 * size].view(4, size)
        return self.backbone.decode(hidden, pos, (job, branch, pos), ranges, groups, {FINAL: reads})[FINAL]

    def _capture(self, size, full=False):
        # requests and branches bound the runs of rows sharing a range: every branch is read once or more
        groups = self._request_limit(size) + (MAX_READS if full else self._read_limit(size)) + 1
        key = (size, settings.batch_invariant, FULL) if full else (size, settings.batch_invariant)
        return self._record(key, lambda: self._stage([([0] * (size - 2), [[0]], None)], size, full),
                            lambda: self._body(size, groups, full))

    def _record(self, key, stage, body):
        if key in self.graphs:
            return self.graphs[key]
        if key in self.failed:
            return None
        try:
            stage()
            body()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool, capture_error_mode=capture_mode()):
                output = body()
        except RuntimeError as exc:
            log.warning("CUDA graph (%s, shape %s) not captured: %s", self.backbone.config.model_type, key,
                        str(exc)[:240])
            self.failed.add(key)
            return None
        result = graph, output
        self.graphs[key] = result
        return result

    @torch.inference_mode()
    def capture_flat(self, log_fn=None, max_tokens=None):
        t0 = time.perf_counter()
        report = log_fn or log.info
        with self.lock, torch.cuda.device(self.device):
            sizes = [s for s in reversed(LADDER) if (max_tokens is None or s <= max_tokens)
                     and (s, settings.batch_invariant) not in self.graphs
                     and (s, settings.batch_invariant) not in self.failed]
            if not sizes:
                return
            torch.cuda.current_stream().wait_event(self.done)
            for index, size in enumerate(sizes, 1):
                self._capture(size)
                capture_progress(index, len(sizes), t0, f"CUDA graphs ({self.backbone.config.model_type})", report)
            self.done.record()
        report(f"captured {len(self.graphs)} CUDA graphs ({self.backbone.config.model_type}) in "
               f"{time.perf_counter() - t0:.1f}s")

    def capture_all(self, log_fn=None, max_tokens=None):
        self.capture_flat(log_fn, max_tokens)

    @torch.inference_mode()
    def capture_sflat(self):
        """Warm the largest session work space before the engine allocates its KV arena."""
        from .session_graphs import CONTEXT_LADDER, MAX_STREAM, SessionGraphs

        shape = MAX_STREAM, CONTEXT_LADDER[-1]
        key = ("warmup", *shape, settings.batch_invariant)
        with self.lock, torch.cuda.device(self.device):
            torch.cuda.current_stream().wait_event(self.done)
            scratch = SessionGraphs(self, self.backbone.kv_arena(0), None)
            self._record(key, lambda: scratch._stage_dummy(*shape), lambda: scratch._body(*shape))
            self.graphs.pop(key, None)
            self.failed.discard(key)
            self.done.record()

    @torch.inference_mode()
    def run_flat(self, items):
        if not items or any(not item[0] or not item[1] or not all(item[1]) or item[2] is not None for item in items):
            return None
        tokens, reads = map(sum, zip(*(self.flat_cost(item) for item in items)))
        rung = self._rung(len(items), tokens, reads, branches=sum(len(item[1]) for item in items))
        if rung is None:
            return None
        size, full = rung
        with self.lock, torch.cuda.device(self.device):
            torch.cuda.current_stream().wait_event(self.done)
            try:
                captured = self._capture(size, full)
                if captured is None:
                    return None
                graph, output = captured
                n = self._stage(items, size, full)
                graph.replay()
                return output[:n].clone()
            finally:
                self.done.record()

    def set_store(self, store, sessions, capture=True):
        from .session_graphs import SessionGraphs

        with self.lock, torch.cuda.device(self.device):
            self.done.synchronize()
            self.graphs = {shape: graph for shape, graph in self.graphs.items() if len(shape) == 2}
            self.failed = {shape for shape in self.failed if len(shape) == 2}
            self.sessions = SessionGraphs(self, store, sessions)

    def sflat_shape(self, items):
        return self.sessions.shape(items) if self.sessions is not None else None

    def run_sflat(self, items):
        return self.sessions.run(items) if self.sessions is not None else None

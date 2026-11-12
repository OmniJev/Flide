"""Requests from CUDA graphs (lfm2), one or several per replay.

LFM2 keeps no recurrent state either, so a round is one packed stream as on gemma4 (graphs.py there): every prefix,
then every branch, then padding, with the attention metadata (job, branch, position, key ranges) and the
convolution's gather indices (each token's earlier inputs in its own sequence: a branch's first tokens read the end
of its prefix) staged next to the tokens. One graph per length rung serves every round; a one-branch request runs as
one sequence. Padding tokens are jobs of their own that see themselves only and read zeros before them, so they never
touch a real token. The graph gathers the final-norm hidden state at every branch's last token. Image features are
computed before the replay and written into the round's embeddings.
"""
from __future__ import annotations

import bisect
import logging
import time

import numpy as np
import torch

from ..base import capture_progress
from . import _Meta

log = logging.getLogger("flide.graphs")

LADDER = (tuple(range(64, 512, 32)) + tuple(range(512, 4096, 64)) + tuple(range(4096, 8192, 128))
          + tuple(range(8192, 16385, 256)))
BMAX = 1024
REQUESTS = 64


class Lfm2Graphs:
    block_attention = False
    prefix_attention = False

    def __init__(self, bb, ladder=LADDER, bmax: int = BMAX):
        self.bb = bb
        self.media = bb.vlm is not None
        self.ladder = tuple(ladder)
        self.bmax = bmax
        self.keep = bb.decoder().keep
        dev = bb.device
        n = self.ladder[-1]
        self.ids = torch.zeros(n, dtype=torch.long, device=dev)
        self.pos = torch.zeros(n, dtype=torch.long, device=dev)
        self.meta = torch.zeros(3, n, dtype=torch.int32, device=dev)   # job, branch, position
        self.ranges = torch.zeros(n, 4, dtype=torch.int32, device=dev)
        self.prev = torch.zeros(n, max(1, self.keep), dtype=torch.long, device=dev)
        self.reads = torch.zeros(bmax, dtype=torch.long, device=dev)
        self.emb = torch.zeros(n, bb.config.hidden_size, device=dev, dtype=bb.dtype)
        self.host = torch.zeros(9 + max(1, self.keep), n, dtype=torch.long, pin_memory=True)
        self.host_reads = torch.zeros(bmax, dtype=torch.long, pin_memory=True)
        self.out = torch.zeros(bmax, bb.config.hidden_size, device=dev, dtype=bb.dtype)
        self.staged = torch.cuda.Event()
        self.staged.record()
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.failed: set[int] = set()
        self.flat_max = self.ladder[-1]

    def nread(self, L: int) -> int:
        return min(self.bmax, max(8, L // 8))

    @staticmethod
    def nreq(L: int) -> int:
        return min(REQUESTS, max(1, L // 32))

    def _rung(self, tokens: int, reads: int, requests: int = 1) -> int | None:
        i = bisect.bisect_left(self.ladder, tokens)
        while i < len(self.ladder) and (self.nread(self.ladder[i]) < reads or self.nreq(self.ladder[i]) < requests):
            i += 1
        return self.ladder[i] if i < len(self.ladder) else None

    def fits(self, prefix_len: int, suffix_lens: list[int]) -> bool:
        return self._rung(prefix_len + sum(suffix_lens), max(1, len(suffix_lens))) is not None

    def shapes(self):
        return list(self.ladder)

    def flat_shapes(self):
        return [("flat", L) for L in self.ladder]

    @staticmethod
    def flat_cost(req) -> tuple[int, int, int]:
        pre, sufs = req[0], req[1]
        return len(pre) + sum(map(len, sufs)), (len(sufs) - 1 if len(sufs) > 1 else 0), 0

    def flat_rung(self, requests: int, tokens: int, branches: int, chunks: int = 0) -> int | None:
        return self._rung(tokens, requests + branches, requests) if requests else None

    def flat_shape(self, reqs) -> int | None:
        cost = [self.flat_cost(r) for r in reqs]
        return self.flat_rung(len(reqs), sum(c[0] for c in cost), sum(c[1] for c in cost))

    def _body(self, L: int):
        dec = self.bb.decoder()
        meta = _Meta((self.meta[0, :L], self.meta[1, :L], self.meta[2, :L]), self.ranges[:L],
                     groups=2 * self.nreq(L) + 1)
        h, _ = dec(self.emb[:L], self.pos[:L], meta, self.prev[:L])
        n = self.nread(L)
        self.out[:n].copy_(dec.final(h.index_select(0, self.reads[:n])))

    @torch.inference_mode()
    def capture(self, L: int) -> bool:
        if L in self.graphs:
            return True
        if L in self.failed:
            return False
        try:
            self._stage_dummy(L)
            self._body(L)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._body(L)
            torch.cuda.synchronize()
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            log.warning("CUDA graph (lfm2, %d tokens) not captured (%s: %s); that length runs eagerly", L,
                        type(exc).__name__, str(exc)[:200])
            self.failed.add(L)
            torch.cuda.empty_cache()
            return False
        self.graphs[L] = g
        return True

    def capture_flat(self, log_fn=None, max_tokens: int | None = None) -> None:
        self.capture_all(log_fn, max_tokens)

    def capture_all(self, log_fn=None, max_tokens: int | None = None) -> None:
        t0 = time.perf_counter()
        lengths = sorted((x for x in self.ladder if max_tokens is None or x <= max_tokens), reverse=True)
        for i, L in enumerate(lengths, 1):
            self.capture(L)
            capture_progress(i, len(lengths), t0, "CUDA graphs (lfm2)", log_fn or log.info)
        (log_fn or log.info)(f"captured {len(self.graphs)} CUDA graphs (lfm2) in {time.perf_counter() - t0:.1f}s")

    def _stage_dummy(self, L: int) -> None:
        ar = torch.arange(L, device=self.ids.device)
        self.ids[:L].fill_(0)
        self.pos[:L].copy_(ar)
        self.meta[0, :L].copy_(-1 - ar)
        self.meta[1, :L].fill_(-2)
        self.meta[2, :L].zero_()
        self.emb[:L].zero_()
        self.ranges[:L, :2].zero_()
        self.ranges[:L, 2].copy_(ar)
        self.ranges[:L, 3].copy_(ar + 1)
        self.prev[:L].fill_(L)
        self.reads.zero_()

    @torch.inference_mode()
    def run(self, prefix: list[int], suffixes: list[list[int]], lazy: bool = True):
        if not lazy:
            L = self.flat_shape([(prefix, suffixes)])
            if L is None or L not in self.graphs:
                return None
        return self.run_flat([(prefix, suffixes)])

    @torch.inference_mode()
    def run_flat(self, reqs):
        """Final-norm hidden states at the last token of every branch of a round of (prefix, branches) requests, in
        request and branch order, one replay; None when the round does not fit."""
        if any(len(r) > 3 and r[3] for r in reqs) or any(len(r) > 4 and r[4] is not None for r in reqs):
            return None   # causal requests read at their branches' last tokens only
        L = self.flat_shape(reqs)
        if L is None or not self.capture(L):
            return None
        self._stage(reqs, L)
        self.graphs[L].replay()
        return self.out[: sum(max(1, len(r[1])) for r in reqs)].clone()

    def _stage(self, reqs, L: int) -> None:
        self.staged.synchronize()
        K = self.keep
        h = self.host.numpy()[:, :L]
        ids, pos, job, br, mpos = h[0], h[1], h[2], h[3], h[4]
        r = h[5:9]
        prev = h[9: 9 + max(1, K)]
        reads = self.host_reads.numpy()
        items = []
        for req in reqs:
            media = req[2] if len(req) > 2 else None
            items.append((req[0] + req[1][0], [], media) if len(req[1]) == 1 else (req[0], req[1], media))
        ks = np.arange(1, max(1, K) + 1)[:, None]
        at, starts = 0, []
        for j, (seq, _, _) in enumerate(items):
            n = len(seq)
            s = slice(at, at + n)
            ar = np.arange(n)
            ids[s] = seq
            pos[s] = ar
            job[s] = j
            br[s] = -1
            r[0, s] = at
            r[1, s] = ar + at + 1
            r[2:, s] = 0
            back = ar[None] - ks
            prev[:, s] = np.where(back >= 0, at + back, L)
            starts.append(at)
            at += n
        nread = b = 0
        for j, (seq, sufs, _) in enumerate(items):
            n, p0 = len(seq), starts[j]
            if not sufs:
                reads[nread] = p0 + n - 1
                nread += 1
            for sfx in sufs:
                m = len(sfx)
                s = slice(at, at + m)
                ar = np.arange(m)
                ids[s] = sfx
                pos[s] = ar + n
                job[s] = j
                br[s] = b
                r[0, s] = p0
                r[1, s] = p0 + n
                r[2, s] = at
                r[3, s] = ar + at + 1
                back = ar[None] - ks
                pre = n + back
                prev[:, s] = np.where(back >= 0, at + back, np.where(pre >= 0, p0 + pre, L))
                reads[nread] = at + m - 1
                nread += 1
                at += m
                b += 1
        s = slice(at, L)
        ar = np.arange(at, L)
        ids[s] = 0
        pos[s] = 0
        job[s] = -1 - ar
        br[s] = -2
        r[0:2, s] = 0
        r[2, s] = ar
        r[3, s] = ar + 1
        prev[:, s] = L
        mpos[:] = pos
        reads[nread:] = 0
        t = self.host[:, :L]
        self.ids[:L].copy_(t[0], non_blocking=True)
        self.pos[:L].copy_(t[1], non_blocking=True)
        self.meta[:, :L].copy_(t[2:5], non_blocking=True)
        self.ranges[:L].copy_(t[5:9].T, non_blocking=True)
        self.prev[:L].copy_(t[9: 9 + max(1, K)].T, non_blocking=True)
        self.reads.copy_(self.host_reads, non_blocking=True)
        self.staged.record()
        self.emb[:L] = self.bb.lm.embed_tokens(self.ids[:L])
        for j, (seq, _, media) in enumerate(items):
            if media is not None:
                s = slice(starts[j], starts[j] + len(seq))
                self.emb[s] = self.bb.encode_vision(self.ids[s], self.emb[s], [media[0]])

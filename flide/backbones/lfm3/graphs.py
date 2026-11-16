"""Requests from CUDA graphs (lfm3), as lfm2's (lfm2/graphs.py: one packed stream a round, one graph per length rung)
with what LFM3 adds: the sliding layers read the causal ranges cut to their window, and the full-attention layers a
second set of ranges, staged with the tokens, so the same graphs run causal rounds (those ranges are the causal ones)
and attention "set" (`run_set`: a plan's sequences, each a state and its questions, read at every answer slot).
"""
from __future__ import annotations

import numpy as np
import torch

from ..gemma4 import _Meta, window_ranges
from ..lfm2.graphs import Lfm2Graphs


class Lfm3Graphs(Lfm2Graphs):
    set_attention = True

    def __init__(self, bb, **kw):
        super().__init__(bb, **kw)
        n = self.ladder[-1]
        self.window = int(bb.config.sliding_window or 0)
        self.sranges = torch.zeros(n, 4, dtype=torch.int32, device=bb.device)
        self.host_set = torch.zeros(4, n, dtype=torch.int32, pin_memory=True)

    def _body(self, L: int):
        dec = self.bb.decoder()
        tok = (self.meta[0, :L], self.meta[1, :L], self.meta[2, :L])
        groups = 2 * self.nreq(L) + 1
        r = self.ranges[:L]
        meta = _Meta(tok, r, groups=groups)
        sl = _Meta(tok, window_ranges(r, tok[2], self.window), groups=groups, tiles=r)
        full = _Meta(tok, self.sranges[:L], groups=self.nread(L) + groups)
        h, _ = dec(self.emb[:L], self.pos[:L], meta, self.prev[:L], set_meta=full, sl_meta=sl)
        n = self.nread(L)
        self.out[:n].copy_(dec.final(h.index_select(0, self.reads[:n])))

    def _stage_dummy(self, L: int) -> None:
        super()._stage_dummy(L)
        self.sranges[:L].copy_(self.ranges[:L])

    def _stage(self, reqs, L: int) -> None:
        super()._stage(reqs, L)
        self.sranges[:L].copy_(self.ranges[:L])   # causal rounds: the full-attention layers read causally too

    @torch.inference_mode()
    def run_set(self, seqs, segments, positions, lazy: bool = True):
        """Final-norm hidden states at every slot of a "set" plan's sequences (`positions` per sequence, from its start
        or negative from its end), in sequence and slot order, one replay; None when they do not fit the graphs."""
        T = sum(len(s) for s in seqs)
        nreads = sum(len(p) for p in positions)
        L = self._rung(T, nreads, len(seqs))
        if L is None or (not lazy and L not in self.graphs) or not self.capture(L):
            return None
        self._stage_set(seqs, segments, positions, L)
        self.graphs[L].replay()
        return self.out[:nreads].clone()

    def _stage_set(self, seqs, segments, positions, L: int) -> None:
        self.staged.synchronize()
        K = self.keep
        h = self.host.numpy()[:, :L]
        ids, pos, job, br, mpos = h[0], h[1], h[2], h[3], h[4]
        r = h[5:9]
        prev = h[9: 9 + max(1, K)]
        sr = self.host_set.numpy()[:, :L]
        reads = self.host_reads.numpy()
        ks = np.arange(1, max(1, K) + 1)[:, None]
        at = nread = 0
        for j, (seq, seg, ps) in enumerate(zip(seqs, segments, positions)):
            n = len(seq)
            s = slice(at, at + n)
            ar = np.arange(n)
            ids[s] = seq
            pos[s] = ar
            job[s] = j
            br[s] = -1
            r[0, s] = at
            r[1, s] = at + ar + 1
            r[2:, s] = 0
            back = ar[None] - ks
            prev[:, s] = np.where(back >= 0, at + back, L)
            S = seg[0]   # a state token sees the whole sequence, a question's token the state and its own question
            sr[:, at: at + S] = np.array([[at], [at + n], [0], [0]])
            for a, e in zip(seg, seg[1:]):
                sr[:, at + a: at + e] = np.array([[at], [at + S], [at + a], [at + e]])
            for p in ps:
                reads[nread] = at + (p if p >= 0 else n + p)
                nread += 1
            at += n
        s = slice(at, L)
        ar = np.arange(at, L)
        ids[s] = 0
        pos[s] = 0
        job[s] = -1 - ar
        br[s] = -2
        r[0:2, s] = 0
        r[2, s] = ar
        r[3, s] = ar + 1
        sr[:, s] = r[:, s]
        prev[:, s] = L
        mpos[:] = pos
        reads[nread:] = 0
        t = self.host[:, :L]
        self.ids[:L].copy_(t[0], non_blocking=True)
        self.pos[:L].copy_(t[1], non_blocking=True)
        self.meta[:, :L].copy_(t[2:5], non_blocking=True)
        self.ranges[:L].copy_(t[5:9].T, non_blocking=True)
        self.prev[:L].copy_(t[9: 9 + max(1, K)].T, non_blocking=True)
        self.sranges[:L].copy_(self.host_set[:, :L].T, non_blocking=True)
        self.reads.copy_(self.host_reads, non_blocking=True)
        self.staged.record()
        self.emb[:L] = self.bb.lm.embed_tokens(self.ids[:L])

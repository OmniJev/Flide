"""Requests from CUDA graphs (gemma4), one or several per replay.

Gemma 4 keeps no recurrent state, so a round of requests (prefixes and branches) is one packed stream, and one graph
per length rung serves every round: every prefix, then every branch, then padding, and the attention metadata (job,
branch, position per token, key ranges) is staged next to the tokens. A one-branch request runs as one sequence (a
prefix and a branch with the batch-invariant products). The interface is the flat graphs' of qwen3_5 (flat_cost,
flat_rung, run_flat), so the engine batches both the same way. The kernel's two key ranges per query tile must not
overlap: the first covers the prefix, the second the branches and the padding. Branching lives in that metadata alone.
Padding tokens are jobs of their own that attend to themselves only, so they never touch a real token. The graph runs
the checkpoint's own transformers decoder with flide's attention and gathers the final-norm hidden state at every
branch's last token. Rounds of requests with block attention (formats.base.Plan) replay graphs of their own, captured
when first needed: the global layers read the staged block ranges with a block number per token, the sliding layers the
same queries causally.
"""
from __future__ import annotations

import bisect
import logging
import time

import torch

from ... import settings
from ..base import capture_progress
from . import _Meta, window_ranges

log = logging.getLogger("flide.graphs")

LADDER = (tuple(range(64, 512, 32)) + tuple(range(512, 4096, 64)) + tuple(range(4096, 8192, 128))
          + tuple(range(8192, 16385, 256)))
BMAX = 1024
REQUESTS = 64


class GemmaGraphs:
    block_attention = True

    def __init__(self, bb, ladder=LADDER, bmax: int = BMAX):
        self.bb = bb
        self.media = bb.vlm is not None
        self.ladder = tuple(ladder)
        self.bmax = bmax
        dev = bb.device
        n = self.ladder[-1]
        self.ids = torch.zeros(n, dtype=torch.long, device=dev)
        self.raw = torch.zeros(n, dtype=torch.long, device=dev)
        self.pos = torch.zeros(n, dtype=torch.long, device=dev)
        self.meta = torch.zeros(4, n, dtype=torch.int32, device=dev)
        self.ranges = torch.zeros(n, 4, dtype=torch.int32, device=dev)
        self.gblk = torch.zeros(n, dtype=torch.int32, device=dev)
        self.reads = torch.zeros(bmax, dtype=torch.long, device=dev)
        self.emb = torch.zeros(n, bb.config.hidden_size, device=dev, dtype=bb.dtype)
        self.host = torch.zeros(12, n, dtype=torch.long, pin_memory=True)
        self.host_reads = torch.zeros(bmax, dtype=torch.long, pin_memory=True)
        self.out = torch.zeros(bmax, bb.config.hidden_size, device=dev, dtype=bb.dtype)
        self.staged = torch.cuda.Event()
        self.staged.record()
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.failed: set[int] = set()
        self.bgraphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.bfailed: set[int] = set()

        self.flat_max = self.ladder[-1]

    def nread(self, L: int) -> int:
        """Read rows a replay of length L holds (its last layer runs at those rows only)."""
        return min(self.bmax, max(8, L // 8))

    @staticmethod
    def nreq(L: int) -> int:
        """Requests a replay of length L holds: each is a run of prefix rows and one of branch rows in the attention
        tiles, so the tile table has room for 2 nreq + 1 runs (the padding one)."""
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
        """(tokens, branch reads beyond one, 0) a (prefix, branches) request takes in a stream."""
        pre, sufs = req[0], req[1]
        return len(pre) + sum(map(len, sufs)), (len(sufs) - 1 if len(sufs) > 1 else 0), 0

    def flat_rung(self, requests: int, tokens: int, branches: int, chunks: int = 0) -> int | None:
        return self._rung(tokens, requests + branches, requests) if requests else None

    def flat_shape(self, reqs) -> int | None:
        cost = [self.flat_cost(r) for r in reqs]
        return self.flat_rung(len(reqs), sum(c[0] for c in cost), sum(c[1] for c in cost))

    def _body(self, L: int, blocks: bool = False):
        bb = self.bb
        tok = (self.meta[0, :L], self.meta[1, :L], self.meta[2, :L])
        r = self.ranges[:L]
        groups = self.nreq(L) + self.nread(L) + 1 if settings.batch_invariant else 2 * self.nreq(L) + 1
        if blocks:
            br, qpos = self.meta[1, :L], self.meta[2, :L]
            ar = torch.arange(L, device=r.device, dtype=r.dtype)
            c = r.clone()
            c[:, 1] = torch.where(br < 0, torch.minimum(r[:, 1], r[:, 0] + qpos + 1), r[:, 1])
            c[:, 3] = torch.where(br >= 0, torch.minimum(r[:, 3], ar + 1), r[:, 3])
            full = _Meta(tok + (self.gblk[:L],), r, groups=groups)
            sliding = _Meta(tok, window_ranges(c, qpos, bb.config.sliding_window or 0), groups=groups, tiles=c)
            self._decode(L, full, sliding)
            return
        if bb.bidir:
            r = r.clone()
            r[:, 1] = torch.minimum(r[:, 1], r[:, 0] + self.meta[2, :L] + 1)
        full = _Meta(tok, r, groups=groups)
        sliding = _Meta(tok + (self.meta[3, :L],) if bb.bidir else tok,
                        window_ranges(self.ranges[:L], self.meta[2, :L], bb.config.sliding_window or 0), groups=groups,
                        tiles=r)
        self._decode(L, full, sliding)

    def _decode(self, L: int, full, sliding):
        bb, lm = self.bb, self.bb.lm
        dec = bb.decoder()
        if dec is not None:
            n = self.nread(L)
            h, _ = dec(self.emb[:L], self.ids[:L], self.pos[:L], {"full_attention": full, "sliding_attention": sliding},
                       rows=self.reads[:n])
            self.out[:n].copy_(dec.final(h))
            return
        pli = lm.get_per_layer_inputs(self.ids[None, :L], None) if bb.config.hidden_size_per_layer_input else None
        out = lm(inputs_embeds=self.emb[None, :L], per_layer_inputs=pli, position_ids=self.pos[None, :L],
                 attention_mask={"full_attention": full, "sliding_attention": sliding}, use_cache=False,
                 return_dict=True)
        self.out.copy_(out.last_hidden_state[0].index_select(0, self.reads))

    @torch.inference_mode()
    def capture(self, L: int, blocks: bool = False) -> bool:
        graphs, failed = (self.bgraphs, self.bfailed) if blocks else (self.graphs, self.failed)
        if L in graphs:
            return True
        if L in failed:
            return False
        try:
            self._stage_dummy(L)
            self._body(L, blocks)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._body(L, blocks)
            torch.cuda.synchronize()
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            log.warning("CUDA graph (gemma4, %d tokens) not captured (%s: %s); that length runs eagerly", L,
                        type(exc).__name__, str(exc)[:200])
            failed.add(L)
            torch.cuda.empty_cache()
            return False
        graphs[L] = g
        return True

    def capture_flat(self, log_fn=None, max_tokens: int | None = None) -> None:
        self.capture_all(log_fn, max_tokens)

    def capture_all(self, log_fn=None, max_tokens: int | None = None) -> None:
        t0 = time.perf_counter()
        lengths = sorted((x for x in self.ladder if max_tokens is None or x <= max_tokens), reverse=True)
        for i, L in enumerate(lengths, 1):
            self.capture(L)
            capture_progress(i, len(lengths), t0, "CUDA graphs (gemma4)", log_fn or log.info)
        (log_fn or log.info)(f"captured {len(self.graphs)} CUDA graphs (gemma4) in {time.perf_counter() - t0:.1f}s")

    def _stage_dummy(self, L: int) -> None:
        ar = torch.arange(L, device=self.ids.device)
        self.ids[:L].fill_(0)
        self.pos[:L].copy_(ar)
        self.meta[0, :L].copy_(-1 - ar)
        self.meta[1, :L].fill_(-2)
        self.meta[2, :L].zero_()
        self.meta[3, :L].fill_(-1)
        self.gblk[:L].fill_(-1)
        self.emb[:L].zero_()
        self.ranges[:L, :2].zero_()
        self.ranges[:L, 2].copy_(ar)
        self.ranges[:L, 3].copy_(ar + 1)
        self.reads.zero_()

    @torch.inference_mode()
    def run(self, prefix: list[int], suffixes: list[list[int]], lazy: bool = True):
        """Final-norm hidden states [n_branches, hidden] at the last token of every branch, or None (with
        `lazy=False` also when its graph was not captured already)."""
        if not lazy:
            L = self.flat_shape([(prefix, suffixes)])
            if L is None or L not in self.graphs:
                return None
        return self.run_flat([(prefix, suffixes)])

    @torch.inference_mode()
    def run_flat(self, reqs):
        """Final-norm hidden states at the last token of every branch of a round of (prefix, branches) requests, in
        request and branch order, one replay; None when the round does not fit."""
        L = self.flat_shape(reqs)
        blocks = any(len(r) > 3 and r[3] for r in reqs)
        if blocks and any(len(r) > 2 and r[2] is not None for r in reqs):
            return None
        if L is None or not self.capture(L, blocks):
            return None
        self._stage(reqs, L)
        (self.bgraphs if blocks else self.graphs)[L].replay()
        return self.out[: sum(max(1, len(r[1])) for r in reqs)].clone()

    def _stage(self, reqs, L: int) -> None:
        import numpy as np

        self.staged.synchronize()
        from . import vision_runs

        bb = self.bb
        h = self.host.numpy()[:, :L]
        ids, pos, job, br, mpos, blk = h[0], h[1], h[2], h[3], h[4], h[5]
        r = h[6:10]
        raw, gblk = h[10], h[11]
        reads = self.host_reads.numpy()
        items = []
        for req in reqs:
            media = req[2] if len(req) > 2 else None
            one = len(req[1]) == 1 and not (settings.batch_invariant and req[0])
            items.append((req[0] + req[1][0], [], media) if one else (req[0], req[1], media))
        blocks = [len(req) > 3 and bool(req[3]) for req in reqs]
        blk[:] = -1
        if any(blocks):
            gblk[:] = -1
        at, starts, nblk = 0, [], 0
        for j, (seq, _, media) in enumerate(items):
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
            if blocks[j]:
                p = len(reqs[j][0])
                r[1, s] = np.where(ar < p, at + p, at + n)
                gblk[s] = (ar >= p).astype(np.int64)
            if media is not None and bb.bidir:
                for a, e in vision_runs(ids[s], bb.placeholders):
                    blk[at + a: at + e] = nblk
                    r[1, at + a: at + e] = at + e
                    nblk += 1
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
                r[3, s] = at + m if blocks[j] else ar + at + 1
                if blocks[j]:
                    gblk[s] = 1
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
        mpos[:] = pos
        reads[nread:] = 0
        raw[:] = ids
        if any(it[2] is not None for it in items):
            ids[np.isin(ids, list(bb.placeholders))] = bb.pad_id
        t = self.host[:, :L]
        self.ids[:L].copy_(t[0], non_blocking=True)
        self.pos[:L].copy_(t[1], non_blocking=True)
        self.meta[:, :L].copy_(t[2:6], non_blocking=True)
        self.ranges[:L].copy_(t[6:10].T, non_blocking=True)
        self.raw[:L].copy_(t[10], non_blocking=True)
        if any(blocks):
            self.gblk[:L].copy_(t[11], non_blocking=True)
        self.reads.copy_(self.host_reads, non_blocking=True)
        self.staged.record()
        self.emb[:L] = bb.lm.embed_tokens(self.ids[:L])
        for j, (seq, _, media) in enumerate(items):
            if media is not None:
                s = slice(starts[j], starts[j] + len(seq))
                self.emb[s] = bb.encode_vision(self.raw[s], self.emb[s], [media[0]], graphs=bb.vision_graphs())

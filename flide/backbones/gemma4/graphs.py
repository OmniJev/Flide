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

Sessions (the session cache with its store, runtime.session.KVArena). A round of requests that resume nothing but save
a state replays a flat graph that also copies each save point's keys and values from the stream into the store's rows
("fs" shapes). A round where some request resumes replays a session graph ("s" shapes): the stream holds each
request's tokens from its resume point and its branches, and every layer reads its keys from a context buffer laid out
as the packed path's (pack.py): every request's whole sequence in order, the saved positions gathered from the store,
the new ones from the stream, the save points written back to the store. Every index is a buffer staged with the round
(padding entries point at dump rows), so one graph serves any round of its shape.
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

# stream lengths: the padding at the end is under 32 tokens below 512, under 64 below 4096 and at most 1/32 of the
# round from there on (every token of padding costs as much as a real one in the matrix products)
LADDER = (tuple(range(64, 512, 32)) + tuple(range(512, 4096, 64)) + tuple(range(4096, 8192, 128))
          + tuple(range(8192, 16385, 256)))


def fine_ladder(step: int, up_to: int = 1024) -> tuple[int, ...]:
    """LADDER with a length every `step` tokens below `up_to` (at least 8 tokens apart)."""
    step = max(8, int(step))
    return tuple(range(LADDER[0], up_to, step)) + tuple(x for x in LADDER if x >= up_to)


BMAX = 1024   # reads per replay (the branches of a round), at most an eighth of its length
REQUESTS = 64   # requests per replay, at most one per 32 tokens of its length
SESSION_REQUESTS = 16   # requests per session replay
# stream lengths of session replays (a request's tokens from its resume point and its branches) and their context
# lengths; with the flat lengths up to FS_MAX (rounds that only save), every session shape is captured with the store,
# so a round never waits for a capture or for the work space one needs, and a larger round is split into replays
# (steps of 128 tokens to 2048, then 256: above 1024 a round pads under 1/8 of its length)
SESSION_LADDER = tuple(range(128, 2049, 128)) + tuple(range(2304, 4097, 256))
CTX_LADDER = (1024, 2048, 4096, 8192, 16384)
FS_MAX = 4096
SLOTS = (1, 4)   # sliding-layer rows a replay reads or writes, in windows (one per request that resumes or saves)


def session_layout(reqs, saves, T: int, C: int):
    """The host side of a session round ("s" shapes), as numpy arrays: the stream (each request's tokens from its resume
    point, then every branch, then padding) with its job, branch, position and key ranges against the context buffer,
    the context's metadata (job, -1, position), the stream rows copied into the context, the reads in request and
    branch order, and per (kind) the context rows filled from the store and the store rows they come from (resumes)
    and the context rows written back to the store and where (saves). `saves`: per request (mark, ArenaRows) or
    (None, None)."""
    import numpy as np

    ids, pos = np.zeros(T, np.int64), np.zeros(T, np.int64)
    job, br, r = np.empty(T, np.int64), np.empty(T, np.int64), np.zeros((T, 4), np.int64)
    s_row, s_ctx = np.zeros(T, np.int64), np.full(T, C - 1, np.int64)
    cmeta = np.zeros((3, C), np.int64)
    cmeta[0], cmeta[1] = -(2 ** 30), -1
    idx = {(a, k): [] for a in ("r_ctx_", "r_store_", "w_ctx_", "w_store_") for k in ("full", "sliding")}
    parts, cstart, reads = [], [], [[] for _ in reqs]
    at = cs = 0
    for j, ((pre, sufs, entry, start, _), (m, saved)) in enumerate(zip(reqs, saves)):
        one = len(sufs) == 1
        seq = list(pre) + list(sufs[0]) if one else list(pre)
        n, k = len(seq), len(seq) - start
        parts.append((seq, [] if one else sufs, len(pre)))
        cstart.append(cs)
        cmeta[0, cs: cs + n], cmeta[1, cs: cs + n], cmeta[2, cs: cs + n] = j, -1, np.arange(n)
        s = slice(at, at + k)
        q = np.arange(start, n)
        ids[s], pos[s], job[s], br[s] = seq[start:], q, j, -1
        r[s, 0], r[s, 1] = cs, cs + q + 1
        s_row[s], s_ctx[s] = np.arange(at, at + k), cs + q
        if one:
            reads[j].append(at + k - 1)
        if entry is not None and start > 0:
            for kind, rr in entry.kv.rows.items():
                idx[("r_ctx_", kind)].append(cs + np.arange(start - len(rr), start))
                idx[("r_store_", kind)].append(rr)
        if saved is not None:
            for kind, rr in saved.rows.items():
                idx[("w_ctx_", kind)].append(cs + np.arange(m - len(rr), m))
                idx[("w_store_", kind)].append(rr)
        at += k
        cs += n
    b = 0
    for j, (seq, sufs, P) in enumerate(parts):
        for sfx in sufs:
            m = len(sfx)
            s = slice(at, at + m)
            ids[s], pos[s], job[s], br[s] = sfx, P + np.arange(m), j, b
            r[s, 0], r[s, 1], r[s, 2], r[s, 3] = cstart[j], cstart[j] + P, at, np.arange(at + 1, at + m + 1)
            reads[j].append(at + m - 1)
            at += m
            b += 1
    ar = np.arange(at, T)   # padding: jobs of their own that see themselves through the second range
    ids[at:], pos[at:], job[at:], br[at:] = 0, 0, -1 - ar, -2
    r[at:, 0], r[at:, 1], r[at:, 2], r[at:, 3] = 0, 0, ar, ar + 1
    return {"ids": ids, "pos": pos, "job": job, "branch": br, "ranges": r, "s_row": s_row, "s_ctx": s_ctx,
            "cmeta": cmeta, "reads": [x for rs in reads for x in rs], "idx": idx}


def _put_rows(dst, idx, src) -> None:
    """dst.index_copy_(0, idx, src), with whole rows moved as 8-byte words when they allow: torch's index_copy takes
    one element per thread, so a bfloat16 row copies four times fewer elements that way. The same bits land."""
    row = dst[0].numel() * dst.element_size() if dst.shape[0] else 0
    if row % 8 == 0 and dst.dtype == src.dtype and dst.is_contiguous() and src.is_contiguous():
        dst.view(dst.shape[0], -1).view(torch.int64).index_copy_(0, idx, src.view(src.shape[0], -1).view(torch.int64))
    else:
        dst.index_copy_(0, idx, src)


class _SaveCtx:
    """Save points of a flat replay ("fs" shapes): each saving request's keys and values at its save rows, from the
    stream into the session store, before the attention reads the stream as it always does."""

    def __init__(self, g, L: int, S: int):
        self.g, self.L, self.S = g, L, S

    def keys(self, module, k, v):
        if module.is_kv_shared_layer:
            return k, v
        g, i = self.g, module.layer_idx
        kind = g.store.kind[i]
        n = self.L if kind == "full" else self.S
        rows, dst = g.sb["w_row_" + kind][:n], g.sb["w_store_" + kind][:n]
        _put_rows(g.store.K[i], dst, k.index_select(0, rows))
        _put_rows(g.store.V[i], dst, v.index_select(0, rows))
        return k, v


class _SessionCtx:
    """The context buffer of a session replay ("s" shapes), as the packed path's _Ctx: per layer every request's whole
    sequence in order, its saved positions gathered from the session store, its new ones copied from the stream, then
    each save point's rows written back to the store. Zeros elsewhere, as a query tile can span rows no request has."""

    def __init__(self, g, T: int, C: int, S: int):
        self.g, self.T, self.C, self.S = g, T, C, S
        self.shared: dict[str, tuple] = {}

    def keys(self, module, k, v):
        if module.is_kv_shared_layer:
            return self.shared[module.layer_type]
        g, i, C, sb = self.g, module.layer_idx, self.C, self.g.sb
        Kc, Vc = k.new_zeros((C,) + k.shape[1:]), v.new_zeros((C,) + v.shape[1:])
        rows, at = sb["s_row"][: self.T], sb["s_ctx"][: self.T]
        _put_rows(Kc, at, k.index_select(0, rows))
        _put_rows(Vc, at, v.index_select(0, rows))
        kind = g.store.kind[i]
        n = C if kind == "full" else self.S
        ka, va = g.store.K[i], g.store.V[i]
        at, src = sb["r_ctx_" + kind][:n], sb["r_store_" + kind][:n]
        _put_rows(Kc, at, ka.index_select(0, src))
        _put_rows(Vc, at, va.index_select(0, src))
        at, dst = sb["w_ctx_" + kind][:n], sb["w_store_" + kind][:n]
        _put_rows(ka, dst, Kc.index_select(0, at))
        _put_rows(va, dst, Vc.index_select(0, at))
        if module.store_full_length_kv:
            self.shared[module.layer_type] = (Kc, Vc)
        return Kc, Vc


class GemmaGraphs:
    block_attention = True   # rounds of block-attention requests replay the graphs in `bgraphs`
    prefix_attention = True   # and prefix-attention ones (fourth field "prefix": the branches one way)
    alone_reads = True   # a sequence read whole under block attention, as its own request read at its last token

    def __init__(self, bb, ladder=LADDER, bmax: int = BMAX, pick: int = 0):
        """`pick`: capture_all times each graph below that many tokens, and a round then replays the fastest length
        within 64 tokens above its own (the matrix products' kernels change with the length, and at some lengths a
        longer one runs faster: on an H200 Gemma 4 31B replays 416 tokens 5 ms slower than 424)."""
        self.bb = bb
        self.media = bb.vlm is not None
        self.ladder = tuple(ladder)
        self.bmax = bmax
        self.pick = int(pick or 0)
        self.cost: dict[int, float] = {}
        self.best: list[int] | None = None
        dev = bb.device
        n = self.ladder[-1]
        self.ids = torch.zeros(n, dtype=torch.long, device=dev)       # placeholders as padding (per-layer inputs)
        self.raw = torch.zeros(n, dtype=torch.long, device=dev)       # as the request has them (where features go)
        self.pos = torch.zeros(n, dtype=torch.long, device=dev)
        self.meta = torch.zeros(4, n, dtype=torch.int32, device=dev)  # job, branch, position, image block
        self.ranges = torch.zeros(n, 4, dtype=torch.int32, device=dev)
        self.gblk = torch.zeros(n, dtype=torch.int32, device=dev)     # the global layers' block number (block attention)
        self.reads = torch.zeros(bmax, dtype=torch.long, device=dev)
        # embeddings of the round, vision features in place, filled before each replay
        self.emb = torch.zeros(n, bb.config.hidden_size, device=dev, dtype=bb.dtype)
        self.host = torch.zeros(12, n, dtype=torch.long, pin_memory=True)  # ids, pos, job, branch, pos, block,
        # 4 ranges, raw ids, global block
        self.host_reads = torch.zeros(bmax, dtype=torch.long, pin_memory=True)
        self.out = torch.zeros(bmax, bb.config.hidden_size, device=dev, dtype=bb.dtype)
        self.staged = torch.cuda.Event()
        self.staged.record()
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.failed: set[int] = set()
        self.bgraphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.bfailed: set[int] = set()
        # sessions: the store and the cache it belongs to (set_store), the session graphs by shape, their index buffers
        self.store = self.sessions = None
        self.window = int(bb.config.sliding_window or 0)
        self.sgraphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self.sfailed: set[tuple] = set()
        self.sb: dict | None = None

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
        if i < len(self.ladder) and self.best is not None:   # a longer length holds the round too, and replays faster
            i = self.best[i]
        return self.ladder[i] if i < len(self.ladder) else None

    def _time(self, L: int, g) -> None:
        """The replay time of the graph of length L, its dummy round staged (capture_all with `pick`)."""
        g.replay()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(3):
            g.replay()
        b.record()
        b.synchronize()
        self.cost[L] = a.elapsed_time(b) / 3

    def _pick(self) -> None:
        """For each length, the length within 64 tokens above it to replay instead: the first that runs at least 5%
        faster than the best so far, among the lengths timed."""
        best = []
        for i, L in enumerate(self.ladder):
            b = i
            if L in self.cost:
                for j in range(i + 1, len(self.ladder)):
                    M = self.ladder[j]
                    if M - L > 64 or M not in self.cost:
                        break
                    if self.cost[M] < 0.95 * self.cost[self.ladder[b]]:
                        b = j
            best.append(b)
        self.best = best

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

    def _body(self, L: int, blocks: bool = False, save: int = 0):
        """`save`: the session store's sliding rows the replay writes ("fs" shapes); 0 for none."""
        bb = self.bb
        tok = (self.meta[0, :L], self.meta[1, :L], self.meta[2, :L])
        r = self.ranges[:L]
        ctx = _SaveCtx(self, L, save) if save else None
        # with the batch-invariant products every branch is a group of its own
        groups = self.nreq(L) + self.nread(L) + 1 if settings.batch_invariant else 2 * self.nreq(L) + 1
        if blocks:   # each block whole on the global layers; the sliding layers causal (prefix rows to their position,
            # branch rows to themselves), then cut to the window
            br, qpos = self.meta[1, :L], self.meta[2, :L]
            ar = torch.arange(L, device=r.device, dtype=r.dtype)
            c = r.clone()
            c[:, 1] = torch.where(br < 0, torch.minimum(r[:, 1], r[:, 0] + qpos + 1), r[:, 1])
            c[:, 3] = torch.where(br >= 0, torch.minimum(r[:, 3], ar + 1), r[:, 3])
            full = _Meta(tok + (self.gblk[:L],), r, groups=groups)
            sliding = full if bb.no_window else \
                _Meta(tok, window_ranges(c, qpos, bb.config.sliding_window or 0), groups=groups, tiles=c)
            self._decode(L, full, sliding)
            return
        if bb.bidir:   # the image-widened ranges are the sliding layers'; the global layers stay causal
            r = r.clone()
            r[:, 1] = torch.minimum(r[:, 1], r[:, 0] + self.meta[2, :L] + 1)
        full = _Meta(tok, r, ctx=ctx, groups=groups)
        sliding = _Meta(tok + (self.meta[3, :L],) if bb.bidir else tok,
                        window_ranges(self.ranges[:L], self.meta[2, :L], bb.config.sliding_window or 0), ctx=ctx,
                        groups=groups, tiles=r)
        self._decode(L, full, sliding)

    def _sbody(self, T: int, C: int, S: int):
        """A session replay: the stream of T rows, a context of C rows, S sliding rows read and written."""
        tok = (self.meta[0, :T], self.meta[1, :T], self.meta[2, :T])
        r = self.ranges[:T]
        k1 = (self.sb["cmeta"][0, :C], self.sb["cmeta"][1, :C], self.sb["cmeta"][2, :C])
        ctx = _SessionCtx(self, T, C, S)
        groups = 2 * SESSION_REQUESTS + 1   # each request's prefix rows, its branch rows, and the padding
        full = _Meta(tok, r, k1, ctx, groups=groups)
        sliding = _Meta(tok, window_ranges(r, self.meta[2, :T], self.window), k1, ctx, groups=groups, tiles=r)
        self._decode(T, full, sliding)

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
            if self.capture(L) and L < self.pick and L not in self.cost:
                self._time(L, self.graphs[L])
            capture_progress(i, len(lengths), t0, "CUDA graphs (gemma4)", log_fn or log.info)
        if self.cost:
            self._pick()
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
        if blocks and not self.bb.no_window and any(len(r) > 2 and r[2] is not None for r in reqs):
            return None   # block attention with a window takes text only
        if any(len(r) > 4 and r[4] is not None and (len(r[1]) != 1 or len(r[4]) != 1) for r in reqs):
            return None   # positions to read: one, in a one-branch request (`alone_reads`)
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
            # one branch runs with its prefix as one sequence; with the batch-invariant products a branch of its own,
            # as when it has siblings, so a question gets the bits it gets forked from its state
            one = len(req[1]) == 1 and not (settings.batch_invariant and req[0])
            items.append((req[0] + req[1][0], [], media) if one else (req[0], req[1], media))
        blocks = [len(req) > 3 and bool(req[3]) for req in reqs]
        two = [len(req) > 3 and req[3] == "prefix" for req in reqs]   # prefix attention: only the prefix both ways
        blk[:] = -1
        if any(blocks):   # only rounds with block attention read the global layers' block numbers
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
            if blocks[j]:   # the prefix whole; a one-branch request's branch, run as part of its sequence, whole too
                p = len(reqs[j][0])
                if two[j]:   # prefix attention: that branch one way, outside every block
                    r[1, s] = np.where(ar < p, at + p, ar + at + 1)
                    gblk[s] = np.where(ar < p, 0, -1)
                else:
                    r[1, s] = np.where(ar < p, at + p, at + n)
                    gblk[s] = (ar >= p).astype(np.int64)
            if media is not None and bb.bidir and not any(blocks):   # the tokens of one image see each other (sliding
                # layers); under block attention (no_window) every layer reads the image's block whole, as _block_meta
                for a, e in vision_runs(ids[s], bb.placeholders):
                    blk[at + a: at + e] = nblk
                    r[1, at + a: at + e] = at + e
                    nblk += 1
            starts.append(at)
            at += n
        nread = b = 0
        for j, (seq, sufs, _) in enumerate(items):
            n, p0 = len(seq), starts[j]
            if not sufs:   # a one-branch request read at its last token, or at the position its fifth field gives
                spot = reqs[j][4] if len(reqs[j]) > 4 else None
                reads[nread] = p0 + (spot[0] if spot else n - 1)
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
                r[3, s] = at + m if blocks[j] and not two[j] else ar + at + 1
                if blocks[j]:
                    gblk[s] = -1 if two[j] else 1
                reads[nread] = at + m - 1
                nread += 1
                at += m
                b += 1
        # padding: jobs of their own seeing themselves through the second key range (a self-range in the first one,
        # the prefixes', would stretch a query tile's prefix range over branch tokens, which its branch queries would
        # then read through both ranges)
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
        return starts

    # ---- sessions ------------------------------------------------------------------------------------------------

    def _sbufs(self) -> dict:
        """The session replays' index buffers, sized for the largest shapes (built on first use)."""
        if self.sb is None:
            dev, T, C, S = self.bb.device, self.ladder[-1], CTX_LADDER[-1], max(1, self.window) * SLOTS[-1]
            z = lambda n: torch.zeros(n, dtype=torch.long, device=dev)
            self.sb = {"s_row": z(T), "s_ctx": z(T), "cmeta": torch.zeros(3, C, dtype=torch.int32, device=dev),
                       "w_row_full": z(T), "w_row_sliding": z(S)}
            for kind, n in (("full", C), ("sliding", S)):
                for name in ("r_ctx_", "r_store_", "w_ctx_", "w_store_"):
                    self.sb[name + kind] = z(n)
        return self.sb

    def set_store(self, store, sessions, capture: bool = True, log_fn=None) -> None:
        """The session cache's store and the cache (whose `reserve` hands out save rows); the session graphs of the
        shapes a request alone takes are captured now when `capture`."""
        self.store, self.sessions = store, sessions
        self.sgraphs.clear()
        self.sfailed.clear()
        if not capture or store is None:
            return
        t0 = time.perf_counter()
        W = max(1, self.window)
        shapes = [("fs", L, W * n) for L in self.ladder if L <= FS_MAX for n in SLOTS]
        shapes += [("s", T, C, W * n) for T in SESSION_LADDER for C in CTX_LADDER for n in SLOTS]
        for i, s in enumerate(sorted(shapes, key=lambda s: -s[1]), 1):
            self._capture_s(s)
            capture_progress(i, len(shapes), t0, "session graphs (gemma4)", log_fn or log.info)
        (log_fn or log.info)(f"captured {len(self.sgraphs)} session graphs (gemma4) in {time.perf_counter() - t0:.1f}s")

    def capture_sflat(self, log_fn=None) -> None:
        """Before the session cache is sized: the largest session shapes, captured on a store of one page and dropped,
        so the graph pool holds their work space when the cache takes what is left."""
        W = max(1, self.window)
        self.store = self.bb.kv_arena(0)
        for s in (("s", SESSION_LADDER[-1], CTX_LADDER[-1], W * SLOTS[-1]), ("fs", FS_MAX, W * SLOTS[-1])):
            self._capture_s(s)
        self.sgraphs.clear()
        self.sfailed.clear()
        self.store = None
        torch.cuda.synchronize()

    def sflat_shape(self, reqs):
        """The session shape a round of (prefix, branches, entry or None, start, mark or None) requests fits, or None:
        ("fs", L, S) when none resumes, ("s", T, C, S) otherwise."""
        if (not reqs or len(reqs) > SESSION_REQUESTS or self.store is None or settings.batch_invariant
                or any(r[2] is not None and not hasattr(r[2].kv, "rows") for r in reqs)):
            return None
        tokens = reads = ctx = nres = nsave = 0
        for pre, sufs, entry, start, mark in reqs:
            one = len(sufs) == 1
            n = len(pre) + (len(sufs[0]) if one else 0)
            tokens += n - start + (0 if one else sum(map(len, sufs)))
            reads += 1 if one else len(sufs)
            ctx += n
            nres += entry is not None and start > 0
            nsave += bool(mark and start < mark < len(pre))
        slots = next((s for s in SLOTS if s >= max(nres, nsave, 1)), None)
        if slots is None:
            return None
        S = max(1, self.window) * slots
        if not nres:
            L = self._rung(tokens, reads, len(reqs))
            return None if L is None or L > FS_MAX else ("fs", L, S)
        T = next((x for x in SESSION_LADDER if x >= tokens and self.nread(x) >= reads), None)
        C = next((x for x in CTX_LADDER if x > ctx), None)
        return None if T is None or C is None else ("s", T, C, S)

    @torch.inference_mode()
    def run_sflat(self, reqs):
        """Final-norm hidden states at every branch's last token of a round of session requests (request and branch
        order) and per request its save points, [(mark, (states, rows))] or []: one replay; None when the round does not
        fit. The rows come from the cache's `reserve` (a save that finds no room is skipped)."""
        shape = self.sflat_shape(reqs)
        if shape is None:
            return None
        keep = [r[2] for r in reqs]
        saves = []
        for pre, sufs, entry, start, mark in reqs:
            m = mark if mark and start < mark < len(pre) else None
            rows = self.sessions.reserve(self.bb.session_rows(m), keep=keep) if m else None
            saves.append((m, rows) if rows is not None else (None, None))
        if not self._capture_s(shape):
            for _, rows in saves:
                if rows is not None:
                    rows.free()
            return None
        if shape[0] == "fs":
            self._stage_fs(reqs, saves, shape[1], shape[2])
        else:
            self._stage_s(reqs, saves, *shape[1:])
        self.sgraphs[shape].replay()
        n = sum(1 if len(r[1]) == 1 else len(r[1]) for r in reqs)
        snaps = [[(m, ({m: {}}, rows))] if rows is not None else [] for m, rows in saves]
        return self.out[:n].clone(), snaps

    @torch.inference_mode()
    def _capture_s(self, shape) -> bool:
        if shape in self.sgraphs:
            return True
        if shape in self.sfailed or self.store is None:
            return False
        self._sbufs()
        body = ((lambda: self._body(shape[1], save=shape[2])) if shape[0] == "fs"
                else (lambda: self._sbody(*shape[1:])))
        try:
            self._stage_dummy(shape[1])
            self._stage_sdummy(shape)
            body()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                body()
            torch.cuda.synchronize()
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            log.warning("session graph (gemma4, %s) not captured (%s: %s); those rounds run packed", shape,
                        type(exc).__name__, str(exc)[:200])
            self.sfailed.add(shape)
            torch.cuda.empty_cache()
            return False
        self.sgraphs[shape] = g
        return True

    def _stage_sdummy(self, shape) -> None:
        """Every index of a session shape at its dump rows (a round of padding only)."""
        sb, st = self.sb, self.store
        C = shape[2] if shape[0] == "s" else CTX_LADDER[0]
        sb["s_row"].zero_()
        sb["s_ctx"].fill_(C - 1)
        sb["cmeta"][0].fill_(-(2 ** 30))
        sb["cmeta"][1].fill_(-1)
        sb["cmeta"][2].zero_()
        for kind in ("full", "sliding"):
            sb["w_row_" + kind].zero_()
            sb["r_ctx_" + kind].fill_(C - 1)
            sb["w_ctx_" + kind].fill_(C - 1)
            sb["r_store_" + kind].fill_(st.dump(kind))
            sb["w_store_" + kind].fill_(st.dump(kind))

    def _put(self, name: str, values, n: int, pad: int) -> None:
        """Index buffer `name`: `values` (concatenated arrays), then `pad` up to `n` entries."""
        import numpy as np

        v = np.concatenate(values) if values else np.zeros(0, np.int64)
        a = np.full(n, pad, np.int64)
        a[: len(v)] = v
        self.sb[name][:n].copy_(torch.from_numpy(a))

    def _stage_fs(self, reqs, saves, L: int, S: int) -> None:
        """A flat round that saves: the flat staging, then each save point's stream rows and store rows."""
        import numpy as np

        starts = self._stage([(r[0], r[1]) for r in reqs], L)
        rows = {"full": [], "sliding": []}
        dst = {"full": [], "sliding": []}
        for j, (m, saved) in enumerate(saves):
            if saved is None:
                continue
            for kind, r in saved.rows.items():
                rows[kind].append(starts[j] + np.arange(m - len(r), m))
                dst[kind].append(r)
        for kind, n in (("full", L), ("sliding", S)):
            self._put("w_row_" + kind, rows[kind], n, 0)
            self._put("w_store_" + kind, dst[kind], n, self.store.dump(kind))

    def _stage_s(self, reqs, saves, T: int, C: int, S: int) -> None:
        """A session round: `session_layout` staged into the graph's buffers, the embeddings of its stream."""
        import numpy as np

        st = self.store
        lay = session_layout(reqs, saves, T, C)
        ids, pos, job, br, r, idx = lay["ids"], lay["pos"], lay["job"], lay["branch"], lay["ranges"], lay["idx"]
        s_row, s_ctx, cmeta = lay["s_row"], lay["s_ctx"], lay["cmeta"]
        rd = np.zeros(self.bmax, np.int64)
        rd[: len(lay["reads"])] = lay["reads"]
        self.staged.synchronize()
        self.ids[:T].copy_(torch.from_numpy(ids))
        self.pos[:T].copy_(torch.from_numpy(pos))
        self.meta[:3, :T].copy_(torch.from_numpy(np.stack([job, br, pos]).astype(np.int32)))
        self.meta[3, :T].fill_(-1)
        self.ranges[:T].copy_(torch.from_numpy(r.astype(np.int32)))
        self.reads.copy_(torch.from_numpy(rd))
        self.sb["s_row"][:T].copy_(torch.from_numpy(s_row))
        self.sb["s_ctx"][:T].copy_(torch.from_numpy(s_ctx))
        self.sb["cmeta"][:, :C].copy_(torch.from_numpy(cmeta.astype(np.int32)))
        for kind, n in (("full", C), ("sliding", S)):
            self._put("r_ctx_" + kind, idx[("r_ctx_", kind)], n, C - 1)
            self._put("w_ctx_" + kind, idx[("w_ctx_", kind)], n, C - 1)
            self._put("r_store_" + kind, idx[("r_store_", kind)], n, st.dump(kind))
            self._put("w_store_" + kind, idx[("w_store_", kind)], n, st.dump(kind))
        self.emb[:T] = self.bb.lm.embed_tokens(self.ids[:T])

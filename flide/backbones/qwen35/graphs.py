"""One request at a time from CUDA graphs (qwen3_5).

A request is a prefix and n branches, read at the last token of each branch.

- n = 1: prefix and branch are one causal sequence. It runs through a *sequence graph*: the tokens sit
  right-aligned in a buffer of the next length on a ladder, and the slots before them are padding.
- n > 1: the prefix runs through the sequence graph of its length, which also leaves every layer's state
  in fixed buffers (Gated DeltaNet recurrent state and convolution window, full-attention keys and values
  in a cache of `kmax` slots, the prefix's keys at slots 0..P-1). Then the branches run, up to 16 at a time,
  through a *branch graph* chosen by (branch count, branch width) on two ladders: right-padded rows that
  start from the prefix state.

Several requests at once go through a *flat graph* (one replay per round): every prefix, then every branch, then
padding to a length on a ladder, in one stream with no padding between them, the layout of the packed eager forward.
Gated DeltaNet layers run over sequence slots whose lengths, chunk lists and initial-state indices are filled before
the replay (kernels/gdn.py), full attention reads per-token key ranges, so a graph of length T takes any round of up
to T tokens, 64 requests and T / 32 + 16 sequences and chunks.

Sessions. A *continuation graph* (one right-padded row of a width on a ladder) takes the tokens after the
cached P, starting from the state in the buffers, and leaves the state after its last real token: padding
tokens get a decay gate of exactly 0 and an update gate of exactly 0, so they leave the recurrent state as it
is; the convolution window is gathered from the last real inputs; keys are appended at slots P..; P moves on.
A *two-segment continuation graph* runs segment A (up to a save point) and segment B (what follows it) in one
pass: Gated DeltaNet layers run A and B as two calls, so the exact state at the save point is copied out between
them, and every other layer runs once over both. A request that resumes from a saved state (or starts from
nothing) restores it into the buffers and runs A to its save point together with B: the rest of its prefix and,
with one branch, the branch itself (its last token is the readout); with several branches, the branch graph
follows from the end of the prefix.

Padding is exact. Left padding enters as zero embeddings; a zero vector stays zero through every layer
(RMSNorm, projections without bias, the convolution, the delta rule and attention all map zero to zero,
and a padding token attends only to itself), so the recurrent state reaches the first real token as it
would with no padding, and real tokens never attend to padding. Right padding comes after the readout.

Everything that depends on the request (lengths, positions, masks, key ranges) is computed inside the
graphs from small device buffers, so one graph serves every request of its shape. A shape that fails to
capture or does not fit falls back to the packed eager forward.
"""
from __future__ import annotations

import bisect
import logging
import time

import torch

from ...kernels import attention as attn_kernel
from ...kernels.fp8 import mm
from ...runtime.tp import capture_mode
from ... import settings
from .ops import add_rms, final_norm, gated_rms, last_qkv, qk_norm_rope, rms, rows_ranges, swiglu
from .ops import gate as gate_out

log = logging.getLogger("flide.graphs")

SEQUENCE_LADDER = (64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048,
                   2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 10240, 12288, 14336, 16384)
WIDTH_LADDER = (16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)
BATCH_LADDER = (2, 3, 4, 6, 8, 12, 16)
CONT_LADDER = (64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072,
               3584, 4096)
TAIL_LADDER = (256, 512, 1024)
FLAT_LADDER = (tuple(range(128, 512, 64)) + tuple(range(512, 2048, 128)) + tuple(range(2048, 4096, 128))
               + tuple(range(4096, 8192, 256)) + tuple(range(8192, 16385, 512)))
FLAT_REQUESTS = 64
CHUNK = 64
SESSION_REQUESTS = 16
SESSION_CTX = 65536
KMAX = 32768
BRANCH_TOKENS = 16384
_OFF = -1.0e4


def _rung(ladder, n: int) -> int | None:
    i = bisect.bisect_left(ladder, n)
    return ladder[i] if i < len(ladder) else None


def _grouped_copy(pairs) -> None:
    """dst.copy_(src) for every (dst, src): the contiguous pairs grouped by dtype into one foreach copy each (a few
    multi-tensor kernels instead of a copy per tensor), the rest one by one."""
    groups, rest = {}, []
    for d, s_ in pairs:
        if d.is_contiguous() and s_.is_contiguous():
            groups.setdefault(d.dtype, ([], []))
            groups[d.dtype][0].append(d)
            groups[d.dtype][1].append(s_)
        else:
            rest.append((d, s_))
    for ds, ss in groups.values():
        torch._foreach_copy_(ds, ss)
    for d, s_ in rest:
        d.copy_(s_)


def flat_dims(T: int) -> tuple[int, int, int]:
    """Prefix slots, branch slots and chunk-list length of the flat graph of length T."""
    chunks = T // 32 + 16
    return min(FLAT_REQUESTS, chunks), chunks, chunks


def sflat_dims(T: int) -> tuple[int, int, int]:
    """Session flat graph of length T: prefix slots (two levels of SESSION_REQUESTS: up to the save point, then the
    rest), branch slots, chunk-list length."""
    chunks = T // 32 + 16
    return 2 * SESSION_REQUESTS, chunks, chunks + SESSION_REQUESTS


def _flat_layout(T: int, sess: bool = False) -> dict:
    """Offsets of the per-round int32 inputs of the flat graph of length T in one staging buffer."""
    P, B, C = (sflat_dims if sess else flat_dims)(T)
    S = P + B
    fields = [("ids", T), ("pos", T), ("pos3", 3 * T), ("job", T), ("branch", T), ("ranges", 4 * T), ("src", 3 * T),
              ("cu", S + 1), ("offsets", S + 1), ("h0", S - P // (2 if sess else 1)), ("chunks", 2 * C), ("reads", S)]
    if sess:
        R = SESSION_REQUESTS
        fields += [("ctx", T), ("resume", R), ("snap", R), ("snapconv", 4 * R)]
    out, at = {}, 0
    for name, n in fields:
        out[name] = (at, at + n)
        at += n
    out["size"] = at
    return out


class RequestGraphs:
    def __init__(self, bb, sequence_ladder=SEQUENCE_LADDER, width_ladder=WIDTH_LADDER, batch_ladder=BATCH_LADDER,
                 kmax: int = KMAX, branch_tokens: int = BRANCH_TOKENS, cont_ladder=CONT_LADDER,
                 tail_ladder=TAIL_LADDER):
        self.bb = bb
        self.lm = bb.lm
        self.seq_ladder = tuple(sequence_ladder)
        self.width_ladder = tuple(width_ladder)
        self.batch_ladder = tuple(batch_ladder)
        self.kmax = kmax
        self.branch_tokens = branch_tokens
        self.cont_ladder = tuple(cont_ladder)
        self.tail_ladder = tuple(tail_ladder)
        self.mark_ladder = tuple(w for w in self.cont_ladder if w % 256 == 0)
        self.bmax = self.batch_ladder[-1]
        dev, dt = bb.device, bb.dtype
        self.device, self.dtype = dev, dt
        cfg = bb.config
        smax = max(self.seq_ladder[-1], self.branch_tokens, kmax + 1)
        self.ar = torch.arange(smax, device=dev)
        self.ids_seq = torch.zeros(self.seq_ladder[-1], dtype=torch.long, device=dev)
        self.ids_branch = torch.zeros(self.branch_tokens, dtype=torch.long, device=dev)
        self.lens = torch.ones(self.bmax, dtype=torch.long, device=dev)
        self.P = torch.zeros((), dtype=torch.long, device=dev)
        self.pos_start = torch.zeros((), dtype=torch.long, device=dev)
        self.media = bb.vlm is not None
        if self.media:
            self.emb_seq = torch.zeros(self.seq_ladder[-1], cfg.hidden_size, device=dev, dtype=dt)
            self.pos3_seq = torch.zeros(3, self.seq_ladder[-1], dtype=torch.long, device=dev)
        ncont = self.cont_ladder[-1] + self.tail_ladder[-1]
        self.ids_cont = torch.zeros(ncont, dtype=torch.long, device=dev)
        self.clen = torch.ones((), dtype=torch.long, device=dev)
        self.clen2 = torch.ones((), dtype=torch.long, device=dev)
        self.host_cont = torch.zeros(ncont, dtype=torch.long, pin_memory=True)
        self.host_seq = torch.zeros(self.seq_ladder[-1], dtype=torch.long, pin_memory=True)
        self.host_branch = torch.zeros(self.branch_tokens, dtype=torch.long, pin_memory=True)
        self.host_lens = torch.ones(self.bmax, dtype=torch.long, pin_memory=True)
        self.staged = torch.cuda.Event()
        self.staged.record()
        self.out = torch.zeros(self.bmax, cfg.hidden_size, device=dev, dtype=dt)
        self.k1_branch = torch.full((kmax,), -1, dtype=torch.int32, device=dev)
        hd = cfg.head_dim
        self.attn_cfg = attn_kernel.config(dev, dt, hd)
        self.K, self.V, self.conv, self.S = {}, {}, {}, {}
        self.conv_mark, self.S_mark = {}, {}
        for i, layer in enumerate(bb.layers):
            if bb.layer_types[i] == "linear_attention":
                m = layer.linear_attn
                kd, vd = m.key_dim, m.value_dim
                W = m.conv1d.weight.shape[-1]
                self.conv[i] = [torch.zeros(1, n, W, device=dev, dtype=dt) for n in (kd, kd, vd)]
                self.S[i] = torch.zeros(1, m.num_v_heads, m.head_k_dim, m.head_v_dim, device=dev, dtype=torch.float32)
                self.conv_mark[i] = [torch.zeros_like(c) for c in self.conv[i]]
                self.S_mark[i] = torch.zeros_like(self.S[i])
            else:
                nkv = bb._w[i]["heads"][1]
                self.K[i] = torch.zeros(kmax + 1, nkv, hd, device=dev, dtype=dt)
                self.V[i] = torch.zeros(kmax + 1, nkv, hd, device=dev, dtype=dt)
        nflat = max(_flat_layout(FLAT_LADDER[-1])["size"], _flat_layout(FLAT_LADDER[-1], True)["size"])
        self.flat_in = torch.zeros(nflat, dtype=torch.int32, device=dev)
        self.host_flat = torch.zeros(nflat, dtype=torch.int32, pin_memory=True)
        P, B, _ = flat_dims(FLAT_LADDER[-1])
        self.out_flat = torch.zeros(P + B, cfg.hidden_size, device=dev, dtype=dt)
        self.flat_max = FLAT_LADDER[-1]
        self.emb_flat = torch.zeros(FLAT_LADDER[-1], cfg.hidden_size, device=dev, dtype=dt)
        self.sess = None
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self.failed: set[tuple] = set()


    def fits(self, prefix_len: int, suffix_lens: list[int]) -> bool:
        if len(suffix_lens) == 1:
            return _rung(self.seq_ladder, prefix_len + suffix_lens[0]) is not None
        return prefix_len <= self.kmax and all(self._branch_shape(1, s) is not None for s in suffix_lens)

    def _branch_shape(self, n: int, width: int):
        w = _rung(self.width_ladder, width)
        b = _rung(self.batch_ladder, max(n, 2))
        if w is None or b is None or b * w > self.branch_tokens:
            return None
        return b, w

    def shapes(self) -> list[tuple]:
        out = [("seq", L) for L in self.seq_ladder]
        if self.media:
            out += [("mseq", L) for L in self.seq_ladder]
        out += [("branch", b, w) for w in self.width_ladder for b in self.batch_ladder if b * w <= self.branch_tokens]
        out += [("cont", w) for w in self.cont_ladder]
        return out

    def flat_shapes(self) -> list[tuple]:
        return [("flat", T) for T in FLAT_LADDER]


    def _sequence(self, L: int, media: bool = False):
        """Sequence graph body: tokens right-aligned in the last L slots of `ids_seq` (or, with media, their
        embeddings in `emb_seq` and M-RoPE positions in `pos3_seq`), `P` real tokens. Leaves the final-norm
        hidden state of the last token in `out[0]`; at L <= kmax also every layer's state for branches."""
        bb, lm = self.bb, self.lm
        N = self.ids_seq.shape[0]
        ar = self.ar[:L]
        pad = L - self.P
        valid = ar >= pad
        pos = torch.where(valid, ar - pad, 0)
        if media:
            h = self.emb_seq[N - L:] * valid[:, None].to(self.dtype)
            rope = torch.where(valid[None], self.pos3_seq[:, N - L:], 0).view(3, 1, L)
        else:
            h = lm.embed_tokens(self.ids_seq[N - L:]) * valid[:, None].to(self.dtype)
            rope = pos.view(1, 1, L).expand(3, 1, L)
        cos, sin = lm.rotary_emb(h[None], rope)
        cos, sin = cos[0], sin[0]
        tok = (torch.where(valid, 0, -1 - ar).int(), torch.where(valid, -1, -2).int(), pos.int())
        zero = torch.zeros_like(ar)
        R = attn_kernel.block_ranges(zero, zero, torch.where(valid, pad, ar), ar + 1, self.attn_cfg["BLOCK_M"],
                                     self.attn_cfg["BLOCK_N"], exact=True)
        keep = L <= self.kmax
        self._valid = valid
        last = self._last_layer()
        n0 = bb.layers[0].input_layernorm
        x = rms(h, n0.weight, n0.eps)
        for i, layer in enumerate(bb.layers):
            w = bb._w[i]
            if bb.layer_types[i] == "linear_attention":
                d = self._gdn(layer.linear_attn, w, x, i, None, keep)
            elif i == last:
                rows = ar[L - 1:]
                qr = torch.stack([zero, zero, torch.where(valid, pad, ar), ar + 1])
                d = self._attn_seq(layer.self_attn, w, x, cos, sin, tok, R, i, L, keep, rows, qr)
                h = h.index_select(0, rows)
            else:
                d = self._attn_seq(layer.self_attn, w, x, cos, sin, tok, R, i, L, keep)
            h, x = self._mlp(i, layer, w, h, d)
        self.out[:1].copy_(final_norm(lm.norm, h[-1:]))

    def _branches(self, B: int, Wd: int):
        """Branch graph body: B rows of width Wd in `ids_branch`, real lengths in `lens`, after a prefix of
        `P` tokens whose state the last sequence graph left in the buffers."""
        bb, lm = self.bb, self.lm
        T = B * Wd
        ids = self.ids_branch[:T]
        ar = self.ar[:T]
        rows = ar // Wd
        offs = ar - rows * Wd
        real = offs < self.lens[:B][rows]
        pos = self.P + offs
        h = lm.embed_tokens(ids)
        rope = self.pos_start + offs
        cos, sin = lm.rotary_emb(h[None], rope.view(1, 1, T).expand(3, 1, T))
        cos, sin = cos[0], sin[0]
        tok = (torch.where(real, 0, -(2 ** 20) - ar).int(), torch.where(real, rows, -2).int(),
               torch.where(real, pos, 0).int())
        ak = self.ar[: self.kmax]
        k1 = (torch.where(ak < self.P, 0, -1 - ak).int(), self.k1_branch, ak.int())
        R = attn_kernel.block_ranges(torch.zeros_like(ar), torch.where(real, self.P, 0),
                                     torch.where(real, rows * Wd, ar), ar + 1, self.attn_cfg["BLOCK_M"],
                                     self.attn_cfg["BLOCK_N"], exact=True)
        last = self.ar[:B] * Wd + self.lens[:B] - 1
        top = self._last_layer()
        n0 = bb.layers[0].input_layernorm
        x = rms(h, n0.weight, n0.eps)
        for i, layer in enumerate(bb.layers):
            w = bb._w[i]
            if bb.layer_types[i] == "linear_attention":
                d = self._gdn(layer.linear_attn, w, x, i, B, False)
            elif i == top:
                qr = torch.stack([torch.zeros_like(ar), torch.where(real, self.P, 0), torch.where(real, rows * Wd, ar),
                                  ar + 1])
                d = self._attn_branch(layer.self_attn, w, x, cos, sin, tok, k1, R, i, last, qr)
                h = h.index_select(0, last)
            else:
                d = self._attn_branch(layer.self_attn, w, x, cos, sin, tok, k1, R, i)
            h, x = self._mlp(i, layer, w, h, d)
        self.out[:B].copy_(final_norm(lm.norm, h if top >= 0 else h[last]))

    def _flat_inputs(self, T: int, sess: bool = False) -> dict:
        lay = _flat_layout(T, sess)
        return {k: self.flat_in[v[0]: v[1]] for k, v in lay.items() if k != "size"}

    def _session_buffers(self) -> dict:
        """Per layer, what session flat graphs read and leave besides the stream: saved states of the resumed requests
        (a last row of zeros for the others), their saved convolution windows (rows 1, 2, 3 back from the resume point;
        a last row of zeros), the states and convolution inputs at the save points, and a context buffer holding every
        request's prefix keys and values (saved ones first)."""
        if self.sess is None:
            bb, dev, dt, R = self.bb, self.device, self.dtype, SESSION_REQUESTS
            sb = {"S": {}, "conv": {}, "snapS": {}, "snapconv": {}, "snapconvT": {}, "convT": {}, "K": {}, "V": {}}
            for i, layer in enumerate(bb.layers):
                if bb.layer_types[i] == "linear_attention":
                    m = layer.linear_attn
                    cd = 2 * m.key_dim + m.value_dim
                    sb["S"][i] = torch.zeros(R + 1, m.num_v_heads, m.head_k_dim, m.head_v_dim, device=dev,
                                             dtype=torch.float32)
                    sb["conv"][i] = torch.zeros(3 * R + 1, cd, device=dev, dtype=dt)
                    sb["snapS"][i] = torch.zeros(R, *sb["S"][i].shape[1:], device=dev, dtype=torch.float32)
                    sb["snapconv"][i] = torch.zeros(4 * R, cd, device=dev, dtype=dt)
                    sb["snapconvT"][i] = torch.zeros(R, cd, 4, device=dev, dtype=dt)
                    sb["convT"][i] = torch.zeros(R, cd, 4, device=dev, dtype=dt)
                else:
                    shape = (SESSION_CTX, bb._w[i]["heads"][1], bb.config.head_dim)
                    sb["K"][i] = torch.zeros(shape, device=dev, dtype=dt)
                    sb["V"][i] = torch.zeros(shape, device=dev, dtype=dt)
            sb["meta"] = torch.zeros(3, SESSION_CTX, dtype=torch.int32, device=dev)
            sb["host_meta"] = torch.zeros(3, SESSION_CTX, dtype=torch.int32, pin_memory=True)
            self.sess = sb
        return self.sess

    def _flat(self, T: int, sess: bool = False):
        """Flat graph body: the round staged by `_stage_flat` (prefixes, branches, padding in one stream of T tokens).
        Leaves the final-norm hidden state at every read in `out_flat`. Session flat graphs (`sess`, staged by
        `_stage_sflat`) run prefixes in two levels from saved states, read prefix keys from the context buffer and
        leave the states at the save points."""
        bb, lm = self.bb, self.lm
        P, B, C = (sflat_dims if sess else flat_dims)(T)
        f = self._flat_inputs(T, sess)
        h = self.emb_flat[:T]
        cos, sin = lm.rotary_emb(h[None], f["pos3"].view(3, 1, T))
        cos, sin = cos[0], sin[0]
        tok = (f["job"], f["branch"], f["pos"])
        r = f["ranges"].view(4, T)
        R = attn_kernel.block_ranges(r[0], r[1], r[2], r[3], self.attn_cfg["BLOCK_M"], self.attn_cfg["BLOCK_N"],
                                     exact=True, groups=P + B + 1 if settings.batch_invariant else 2 * P + 1)
        src = f["src"].view(3, T)
        slots = (f["cu"], f["chunks"].view(C, 2), (P // 2, P // 2) if sess else (P,), f["h0"])
        if sess:
            sb = self._session_buffers()
            ctx_meta = (sb["meta"][0], sb["meta"][1], sb["meta"][2])
            ctx_idx = f["ctx"].long()
            for i in sb["convT"]:
                sb["conv"][i][: 3 * SESSION_REQUESTS].view(SESSION_REQUESTS, 3, -1).copy_(
                    sb["convT"][i][:, :, 1:].flip(2).transpose(1, 2))
        reads = f["reads"]
        top = self._last_layer()
        n0 = bb.layers[0].input_layernorm
        x = rms(h, n0.weight, n0.eps)
        for i, layer in enumerate(bb.layers):
            w = bb._w[i]
            if bb.layer_types[i] == "linear_attention":
                d = self._gdn_flat(layer.linear_attn, w, x, src, slots, i, f if sess else None)
            else:
                m = layer.self_attn
                if i == top:
                    q, gate, k, v = last_qkv(m, w, x, reads, cos, sin)
                    Rq, rq = rows_ranges(r, reads, self.attn_cfg)
                    tq = tuple(t.index_select(0, reads) for t in tok)
                else:
                    q, gate, k, v = self._qkv(m, w, x)
                    q, k = qk_norm_rope(q, k, cos, sin, m.q_norm.weight, m.k_norm.weight, m.q_norm.eps)
                    Rq, rq, tq = R, r, tok
                if sess:
                    Kc, Vc = sb["K"][i], sb["V"][i]
                    Kc.index_copy_(0, ctx_idx, k)
                    Vc.index_copy_(0, ctx_idx, v)
                    o = attn_kernel.attention(q, Kc, Vc, k, v, tq, ctx_meta, tok, Rq, m.scaling, 0, self.attn_cfg,
                                              qranges=rq)
                else:
                    o = attn_kernel.attention(q, k, v, k, v, tq, tok, tok, Rq, m.scaling, 0, self.attn_cfg, qranges=rq)
                d = mm(gate_out(o, gate), w["o"])
                if i == top:
                    h = h.index_select(0, reads)
            h, x = self._mlp(i, layer, w, h, d)
        self.out_flat[: reads.shape[0]].copy_(final_norm(lm.norm, h if top >= 0 else h[reads]))

    def _gdn_flat(self, m, w, x, src, slots, i: int = -1, sess=None):
        from ...kernels import gdn

        T = x.shape[0]
        kd, vd, nv = m.key_dim, m.value_dim, m.num_v_heads
        cd = 2 * kd + vd
        p = mm(x, w["in"])
        z = p[:, cd: cd + vd]
        a, beta = gdn.gates(p, nv)
        sb = self.sess if sess is not None else None
        q, k, v = gdn.conv(p, w["conv"], src, kd, vd, saved=None if sb is None else sb["conv"][i],
                           l2norm_heads=m.head_k_dim)
        cu, chunks, levels, h0 = slots
        o, ht = gdn.flat(q.view(1, T, m.num_k_heads, m.head_k_dim), k.view(1, T, m.num_k_heads, m.head_k_dim),
                         v.view(1, T, nv, m.head_v_dim), a, beta, m.A_log, m.dt_bias, cu, chunks, levels, h0,
                         resume=None if sb is None else sb["S"][i],
                         resume_idx=None if sb is None else sess["resume"], prepared=True)
        if sb is not None:
            sb["snapS"][i].copy_(ht.index_select(0, sess["snap"]))
            idx = sess["snapconv"]
            rows = torch.where((idx < T)[:, None], p[:, :cd].index_select(0, idx.clamp(max=T - 1)),
                               sb["conv"][i].index_select(0, (idx - T).clamp(min=0)))
            sb["snapconv"][i].copy_(rows)
            sb["snapconvT"][i].copy_(rows.view(SESSION_REQUESTS, 4, cd).flip(1).transpose(1, 2))
        y = gated_rms(o.reshape(T, nv, m.head_v_dim), z, m.norm.weight, m.norm.variance_epsilon)
        return mm(y, w["out"])

    def _continue(self, W: int, Wb: int = 0):
        """Continuation graph body. Segment A: `clen` real tokens at the start of `ids_cont[:W]`, after the `P` cached
        ones. With Wb, segment B: `clen2` real tokens at the start of `ids_cont[W:W + Wb]`, next in the sequence after
        A's real tokens; the recurrent state and convolution window at the end of A go to `S_mark` / `conv_mark`.
        Updates every layer's state, appends keys and values at slots P.., advances P, and leaves the final-norm
        hidden state of the last real token in `out[0]`."""
        bb, lm = self.bb, self.lm
        T = W + Wb
        ids = self.ids_cont[:T]
        ar = self.ar[:T]
        if Wb:
            in_b = ar >= W
            real = torch.where(in_b, ar - W < self.clen2, ar < self.clen)
            pos = self.P + torch.where(in_b, self.clen + ar - W, ar)
            last = W + self.clen2 - 1
        else:
            real = ar < self.clen
            pos = self.P + ar
            last = self.clen - 1
        h = lm.embed_tokens(ids)
        cos, sin = lm.rotary_emb(h[None], pos.view(1, 1, T).expand(3, 1, T))
        cos, sin = cos[0], sin[0]
        tok = (torch.where(real, 0, -(2 ** 20) - ar).int(), torch.where(real, 0, -2).int(), torch.where(real, pos, 0).int())
        ak = self.ar[: self.kmax]
        k1 = (torch.where(ak < self.P, 0, -1 - ak).int(), self.k1_branch, ak.int())
        zero = torch.zeros_like(ar)
        R = attn_kernel.block_ranges(zero, torch.where(real, self.P, 0), zero, torch.where(real, ar + 1, 0),
                                     self.attn_cfg["BLOCK_M"], self.attn_cfg["BLOCK_N"], exact=not Wb)
        dst = torch.where(real, pos, self.kmax)
        top = self._last_layer()
        n0 = bb.layers[0].input_layernorm
        x = rms(h, n0.weight, n0.eps)
        for i, layer in enumerate(bb.layers):
            w = bb._w[i]
            if bb.layer_types[i] == "linear_attention":
                d = self._gdn_continue(layer.linear_attn, w, x, i, real, W, Wb)
            else:
                m = layer.self_attn
                if i == top:
                    rows = last.view(1)
                    q, gate, k, v = last_qkv(m, w, x, rows, cos, sin)
                    qr = torch.stack([zero, torch.where(real, self.P, 0), zero, torch.where(real, ar + 1, 0)])
                    Rq, rq = rows_ranges(qr, rows, self.attn_cfg, exact=not Wb)
                    tq = tuple(t.index_select(0, rows) for t in tok)
                else:
                    q, gate, k, v = self._qkv(m, w, x)
                    q, k = qk_norm_rope(q, k, cos, sin, m.q_norm.weight, m.k_norm.weight, m.q_norm.eps)
                    Rq, rq, tq = R, None, tok
                o = attn_kernel.attention(q, self.K[i][: self.kmax], self.V[i][: self.kmax], k, v, tq, k1, tok, Rq,
                                          m.scaling, 0, self.attn_cfg, qranges=rq)
                self.K[i].index_copy_(0, dst, k)
                self.V[i].index_copy_(0, dst, v)
                d = mm(gate_out(o, gate), w["o"])
                if i == top:
                    h = h.index_select(0, last.view(1))
            h, x = self._mlp(i, layer, w, h, d)
        self.out[:1].copy_(final_norm(lm.norm, h if top >= 0 else h.index_select(0, last.view(1))))
        self.P.add_(self.clen + self.clen2 if Wb else self.clen)
        self.pos_start.copy_(self.P)

    def _gdn_continue(self, m, w, x, i, real, W: int, Wb: int = 0):
        """Gated DeltaNet over one or two right-padded segments from the state in the buffers. Padding rows get gate
        inputs whose decay and update are exactly 0, so the recurrent state after a segment is the state after its
        last real token; the convolution window is the last inputs before the end of the real tokens. With two
        segments the state between them is copied to the save-point buffers and B continues from it."""
        from fla.modules.convolution import causal_conv1d
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        T = x.shape[0]
        kd, vd, nv = m.key_dim, m.value_dim, m.num_v_heads
        cd = 2 * kd + vd
        p = mm(x, w["in"])
        z = p[:, cd: cd + vd]
        keep = real[:, None]
        beta = torch.where(keep, p[:, cd + vd: cd + vd + nv], _OFF).view(1, T, nv)
        a = torch.where(keep, p[:, cd + vd + nv:], _OFF).view(1, T, nv)
        segments = [(0, W, self.clen)] + ([(W, T, self.clen2)] if Wb else [])
        S, conv, outs = self.S[i], self.conv[i], []
        for g, (lo, hi, n) in enumerate(segments):
            parts, window = [], []
            for j, (clo, chi, heads, dim) in enumerate(((0, kd, m.num_k_heads, m.head_k_dim),
                                                        (kd, 2 * kd, m.num_k_heads, m.head_k_dim),
                                                        (2 * kd, cd, nv, m.head_v_dim))):
                xj = p[None, lo:hi, clo:chi]
                init = conv[j]
                y, _ = causal_conv1d(xj, w["conv"][clo:chi], None, activation=m.activation, initial_state=init,
                                     output_final_state=False)
                Wk = init.shape[-1]
                src = n + self.ar[:Wk]
                from_x = xj[0].index_select(0, (src - Wk).clamp(min=0)).T
                from_init = init[0].index_select(1, src.clamp(max=Wk - 1))
                window.append(torch.where(src >= Wk, from_x, from_init)[None])
                parts.append(y.view(1, hi - lo, heads, dim))
            q, k, v = parts
            o, S = chunk_gated_delta_rule(q, k, v, g=a[:, lo:hi], beta=beta[:, lo:hi], initial_state=S,
                                          output_final_state=True, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                                          A_log=m.A_log, dt_bias=m.dt_bias, use_beta_sigmoid_in_kernel=True)
            conv = window
            if g == 0 and Wb:
                self.S_mark[i].copy_(S)
                for c, t in zip(self.conv_mark[i], window):
                    c.copy_(t)
            outs.append(o)
        self.S[i].copy_(S)
        for c, t in zip(self.conv[i], conv):
            c.copy_(t)
        o = outs[0] if len(outs) == 1 else torch.cat(outs, 1)
        y = gated_rms(o.reshape(T, nv, m.head_v_dim), z, m.norm.weight, m.norm.variance_epsilon)
        return mm(y, w["out"])

    def _mlp(self, i, layer, w, h, d):
        bb = self.bb
        n2 = layer.post_attention_layernorm
        h, x = add_rms(h, d, n2.weight, n2.eps)
        d = mm(swiglu(mm(x, w["gate_up"]), w["inter"]), w["down"])
        if i + 1 < len(bb.layers):
            n1 = bb.layers[i + 1].input_layernorm
            return add_rms(h, d, n1.weight, n1.eps)
        return h + d, None

    def _gdn(self, m, w, x, i, B, keep):
        """Gated DeltaNet over one sequence (B None: from zero state, final state kept when `keep`) or over
        B right-padded rows starting from the kept state."""
        from fla.modules.convolution import causal_conv1d
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        T = x.shape[0]
        rows = 1 if B is None else B
        width = T // rows
        kd, vd, nv = m.key_dim, m.value_dim, m.num_v_heads
        cd = 2 * kd + vd
        p = mm(x, w["in"])
        z = p[:, cd: cd + vd]
        beta = p[:, cd + vd: cd + vd + nv].reshape(rows, width, nv)
        a = p[:, cd + vd + nv:].reshape(rows, width, nv)
        pr = p.view(rows, width, -1)
        parts = []
        for j, (lo, hi, heads, dim) in enumerate(((0, kd, m.num_k_heads, m.head_k_dim),
                                                   (kd, 2 * kd, m.num_k_heads, m.head_k_dim),
                                                   (2 * kd, cd, nv, m.head_v_dim))):
            init = None if B is None else self.conv[i][j].expand(B, -1, -1).contiguous()
            y, st = causal_conv1d(pr[:, :, lo:hi], w["conv"][lo:hi], None, activation=m.activation, initial_state=init,
                                  output_final_state=keep)
            if keep:
                self.conv[i][j].copy_(st)
            parts.append(y.view(rows, width, heads, dim))
        q, k, v = parts
        init = None if B is None else self.S[i].expand(B, -1, -1, -1).contiguous()
        o, S = chunk_gated_delta_rule(q, k, v, g=a, beta=beta, initial_state=init, output_final_state=keep,
                                      use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, A_log=m.A_log,
                                      dt_bias=m.dt_bias, use_beta_sigmoid_in_kernel=True)
        if keep:
            self.S[i].copy_(S)
        y = gated_rms(o.reshape(T, nv, m.head_v_dim), z, m.norm.weight, m.norm.variance_epsilon)
        return mm(y, w["out"])

    def _qkv(self, m, w, x):
        T = x.shape[0]
        hd = m.head_dim
        nh, nkv = w["heads"]
        p = mm(x, w["qkv"])
        qg = p[:, : nh * 2 * hd].view(T, nh, 2 * hd)
        k = p[:, nh * 2 * hd: nh * 2 * hd + nkv * hd].view(T, nkv, hd)
        v = p[:, nh * 2 * hd + nkv * hd:].view(T, nkv, hd)
        return qg[..., :hd], qg[..., hd:], k, v

    def _last_layer(self) -> int:
        """Index of the last layer when it runs at the read rows only, else -1."""
        bb = self.bb
        i = len(bb.layers) - 1
        return i if bb.layer_types[i] != "linear_attention" else -1

    def _rows_qkv(self, m, w, x, cos, sin, tok, R, rows, qr):
        """q, gate, k, v and the query side of the kernel call (metadata, tiles, exact ranges): every row, or
        queries at `rows` only from their key ranges qr [4, T]."""
        if rows is None:
            q, gate, k, v = self._qkv(m, w, x)
            q, k = qk_norm_rope(q, k, cos, sin, m.q_norm.weight, m.k_norm.weight, m.q_norm.eps)
            return q, gate, k, v, tok, R, None
        q, gate, k, v = last_qkv(m, w, x, rows, cos, sin)
        Rq, rq = rows_ranges(qr, rows, self.attn_cfg)
        return q, gate, k, v, tuple(t.index_select(0, rows) for t in tok), Rq, rq

    def _attn_seq(self, m, w, x, cos, sin, tok, R, i, L, keep, rows=None, qr=None):
        q, gate, k, v, tq, Rq, rq = self._rows_qkv(m, w, x, cos, sin, tok, R, rows, qr)
        if keep:
            dst = torch.where(self._valid, self.ar[:L] - (L - self.P), self.kmax)
            self.K[i].index_copy_(0, dst, k)
            self.V[i].index_copy_(0, dst, v)
        o = attn_kernel.attention(q, k, v, k, v, tq, tok, tok, Rq, m.scaling, 0, self.attn_cfg, qranges=rq)
        return mm(gate_out(o, gate), w["o"])

    def _attn_branch(self, m, w, x, cos, sin, tok, k1, R, i, rows=None, qr=None):
        q, gate, k, v, tq, Rq, rq = self._rows_qkv(m, w, x, cos, sin, tok, R, rows, qr)
        o = attn_kernel.attention(q, self.K[i][: self.kmax], self.V[i][: self.kmax], k, v, tq, k1, tok, Rq, m.scaling,
                                  0, self.attn_cfg, qranges=rq)
        return mm(gate_out(o, gate), w["o"])


    def _body(self, shape):
        if shape[0] == "branch":
            return lambda: self._branches(*shape[1:])
        if shape[0] == "cont":
            return lambda: self._continue(shape[1])
        if shape[0] == "cont2":
            return lambda: self._continue(shape[1], shape[2])
        if shape[0] == "flat":
            return lambda: self._flat(shape[1])
        if shape[0] == "sflat":
            return lambda: self._flat(shape[1], sess=True)
        return lambda: self._sequence(shape[1], media=shape[0] == "mseq")

    def _have(self, shape, lazy: bool) -> bool:
        return self.capture(shape) if lazy else shape in self.graphs

    @torch.inference_mode()
    def capture(self, shape) -> bool:
        if shape in self.graphs:
            return True
        if shape in self.failed:
            return False
        body = self._body(shape)
        try:
            self.P.fill_(64 if shape[0] in ("branch", "cont", "cont2") else shape[1])
            self.pos_start.fill_(64)
            self.lens.fill_(1)
            self.clen.fill_(1)
            self.clen2.fill_(1)
            if shape[0] == "flat":
                self._stage_flat([(list(range(shape[1])), [])], shape[1])
                torch.cuda.synchronize()
            if shape[0] == "sflat":
                self._stage_sflat([(list(range(shape[1])), [], None, 0, None)], shape[1])
                torch.cuda.synchronize()
            body()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool, capture_error_mode=capture_mode()):
                body()
            torch.cuda.synchronize()
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            log.warning("CUDA graph %s not captured (%s: %s); that shape runs eagerly", shape, type(exc).__name__,
                        str(exc)[:200])
            self.failed.add(shape)
            torch.cuda.empty_cache()
            return False
        self.graphs[shape] = g
        return True

    def capture_all(self, log_fn=None, max_tokens: int | None = None) -> None:
        """Capture every shape up to `max_tokens` (all when None), largest first so the shared pool is sized
        once. Shapes left out are captured on first use."""
        t0 = time.perf_counter()
        size = lambda s: s[1] * s[2] if s[0] == "branch" else s[1]
        shapes = sorted((s for s in self.shapes() if max_tokens is None or size(s) <= max_tokens), key=lambda s: -size(s))
        for s in shapes:
            self.capture(s)
        msg = f"captured {len(self.graphs)} CUDA graphs in {time.perf_counter() - t0:.1f}s"
        (log_fn or log.info)(msg)

    def capture_flat(self, log_fn=None, max_tokens: int | None = None) -> None:
        """Capture the flat graphs up to `max_tokens` (all when None)."""
        t0 = time.perf_counter()
        shapes = [s for s in self.flat_shapes() if max_tokens is None or s[1] <= max_tokens]
        for s in sorted(shapes, key=lambda s: -s[1]):
            self.capture(s)
        (log_fn or log.info)(f"captured {len(shapes)} flat graphs in {time.perf_counter() - t0:.1f}s")

    def capture_sessions(self, log_fn=None) -> None:
        """Capture the two-segment continuation graphs that sessions use (only when the session cache is on)."""
        t0 = time.perf_counter()
        shapes = [("cont2", a, b) for a in self.mark_ladder for b in self.tail_ladder]
        for s in sorted(shapes, key=lambda s: -(s[1] + s[2])):
            self.capture(s)
        (log_fn or log.info)(f"captured {len(shapes)} session graphs in {time.perf_counter() - t0:.1f}s")

    @torch.inference_mode()
    def run(self, prefix: list[int], suffixes: list[list[int]], lazy: bool = True):
        """Final-norm hidden states [n_branches, hidden] at the last token of every branch, or None when a
        shape is not available (the caller runs the request eagerly). `lazy=False` takes only graphs captured
        already: a capture in the middle of serving compiles and runs the shape first and stalls every request
        behind it.

        Every graph the request needs is captured before any of its data is staged: capturing runs the
        graph body on the shared buffers and would overwrite the request's lengths and the prefix state."""
        n = len(suffixes)
        if n == 1:
            seq = prefix + suffixes[0]
            L = _rung(self.seq_ladder, len(seq))
            if L is None or not self._have(("seq", L), lazy):
                return None
            self._stage_sequence(seq, L)
            self.graphs[("seq", L)].replay()
            return self.out[:1].clone()
        L = _rung(self.seq_ladder, len(prefix))
        if L is None or L > self.kmax:
            return None
        chunks = [suffixes[i: i + self.bmax] for i in range(0, n, self.bmax)]
        shapes = [self._branch_shape(len(c), max(len(s) for s in c)) for c in chunks]
        if any(sh is None for sh in shapes):
            return None
        if not all(self._have(s, lazy) for s in [("seq", L)] + [("branch", B, Wd) for B, Wd in shapes]):
            return None
        self._stage_sequence(prefix, L)
        self.graphs[("seq", L)].replay()
        return self._run_branches(chunks, shapes)

    def _stage_sequence(self, seq: list[int], L: int) -> None:
        self.staged.synchronize()
        N = self.host_seq.shape[0]
        self.host_seq[N - len(seq):] = torch.tensor(seq, dtype=torch.long)
        self.ids_seq[N - L:].copy_(self.host_seq[N - L:], non_blocking=True)
        self.staged.record()
        self.P.fill_(len(seq))
        self.pos_start.fill_(len(seq))

    def _run_branches(self, chunks, shapes) -> torch.Tensor:
        out = []
        for chunk, (B, Wd) in zip(chunks, shapes):
            self.staged.synchronize()
            host = self.host_branch[: B * Wd].view(B, Wd)
            for r, s in enumerate(chunk):
                host[r, : len(s)] = torch.tensor(s, dtype=torch.long)
                self.host_lens[r] = len(s)
            self.host_lens[len(chunk): B] = 1
            self.ids_branch[: B * Wd].copy_(self.host_branch[: B * Wd], non_blocking=True)
            self.lens[:B].copy_(self.host_lens[:B], non_blocking=True)
            self.staged.record()
            self.graphs[("branch", B, Wd)].replay()
            out.append(self.out[: len(chunk)].clone())
        return torch.cat(out)


    @staticmethod
    def flat_cost(req) -> tuple[int, int, int]:
        """(tokens, branch slots, chunks) a (prefix, branches[, media]) request takes in a flat stream; one branch runs
        with its prefix as one sequence (with the batch-invariant products a branch of its own, as when it has
        siblings)."""
        pre, sufs = req[0], req[1]
        if not sufs or len(sufs) == 1 and not settings.batch_invariant:
            n = len(pre) + sum(map(len, sufs))
            return n, 0, -(-n // CHUNK)
        return (len(pre) + sum(map(len, sufs)), len(sufs),
                -(-len(pre) // CHUNK) + sum(-(-len(x) // CHUNK) for x in sufs))

    @staticmethod
    def flat_rung(requests: int, tokens: int, branches: int, chunks: int) -> int | None:
        """Length of the flat graph a round of that size fits, or None."""
        if not requests or requests > FLAT_REQUESTS:
            return None
        for T in FLAT_LADDER:
            if T < tokens:
                continue
            if T > 2 * tokens + 256:
                return None
            P, B, C = flat_dims(T)
            if requests <= P and branches + 2 <= B and chunks + -(-(T - tokens) // CHUNK) <= C:
                return T
        return None

    def flat_shape(self, reqs) -> int | None:
        """Length of the flat graph a round of (prefix, branches[, media]) requests fits, or None."""
        cost = [self.flat_cost(r) for r in reqs]
        return self.flat_rung(len(reqs), *(sum(c[k] for c in cost) for k in range(3)))

    def _stage_flat(self, reqs, T: int) -> None:
        """Write a round into the flat graph's inputs: per-token ids, positions (sequence order for attention, M-RoPE for
        the rotary embedding), attention metadata and key ranges, convolution predecessors; per-slot lengths, chunk
        offsets and initial-state indices; the chunk list; reads. Then the token embeddings, with the vision features
        of requests that carry media (vision inputs, prefix M-RoPE positions, offset of the branches)."""
        import numpy as np

        P, B, C = flat_dims(T)
        S = P + B
        lay = _flat_layout(T)
        self.staged.synchronize()
        buf = self.host_flat.numpy()
        col = lambda k: buf[lay[k][0]: lay[k][1]]
        ids, pos, job, br = col("ids"), col("pos"), col("job"), col("branch")
        pos3 = col("pos3").reshape(3, T)
        r = col("ranges").reshape(4, T)
        src = col("src").reshape(3, T)
        cu, offsets, h0, reads = col("cu"), col("offsets"), col("h0"), col("reads")
        chunks = col("chunks").reshape(C, 2)
        lens = np.zeros(S, np.int64)
        items = []
        for req in reqs:
            pre, sufs = req[0], req[1]
            media = req[2] if len(req) > 2 else None
            items.append((pre + sufs[0], [], media, len(pre)) if len(sufs) == 1 and not settings.batch_invariant
                         else (pre, sufs, media, len(pre)))
        at = 0
        starts = []
        for j, (seq, _, media, p) in enumerate(items):
            n = len(seq)
            s = slice(at, at + n)
            ar = np.arange(n)
            ids[s] = seq
            pos[s] = ar
            if media is None:
                pos3[:, s] = ar
            else:
                _, rope, offset = media
                pos3[:, at: at + p] = np.asarray(rope).reshape(3, p)
                pos3[:, at + p: at + n] = np.arange(p + offset, n + offset)
            job[s] = j
            br[s] = -1
            r[0, s] = at
            r[1, s] = ar + at + 1
            r[2:, s] = 0
            for d in (1, 2, 3):
                src[d - 1, s] = np.where(ar >= d, ar + at - d, -1)
            lens[j] = n
            starts.append(at)
            at += n
        nread = 0
        b = 0
        for j, (seq, sufs, media, _) in enumerate(items):
            n, p0 = len(seq), starts[j]
            if not sufs:
                reads[nread] = p0 + n - 1
                nread += 1
            off = media[2] if media is not None else 0
            for sfx in sufs:
                m = len(sfx)
                s = slice(at, at + m)
                ar = np.arange(m)
                ids[s] = sfx
                pos[s] = ar + n
                pos3[:, s] = ar + n + off
                job[s] = j
                br[s] = b
                r[0, s] = p0
                r[1, s] = p0 + n
                r[2, s] = at
                r[3, s] = ar + at + 1
                for d in (1, 2, 3):
                    back = ar - d
                    src[d - 1, s] = np.where(back >= 0, at + back, np.where(n + back >= 0, p0 + n + back, -1))
                lens[P + b] = m
                h0[b] = j
                reads[nread] = at + m - 1
                nread += 1
                at += m
                b += 1
        if at < T:
            s = slice(at, T)
            ar = np.arange(at, T)
            ids[s] = 0
            pos[s] = 0
            pos3[:, s] = 0
            job[s] = -1 - ar
            br[s] = -2
            r[0:2, s] = 0
            r[2, s] = ar
            r[3, s] = ar + 1
            src[:, s] = -1
            lens[P + b] = T - at
            h0[b] = 0
            b += 1
        h0[b:] = 0
        reads[nread:] = 0
        cu[0] = 0
        cu[1:] = np.cumsum(lens)
        nch = -(-lens // CHUNK)
        offsets[0] = 0
        offsets[1:] = np.cumsum(nch)
        total = int(offsets[-1])
        slot = np.repeat(np.arange(S), nch)
        chunks[:total, 0] = slot
        chunks[:total, 1] = np.arange(total) - offsets[:-1][slot]
        chunks[total:, 0] = S - 1
        chunks[total:, 1] = 0
        n = lay["size"]
        self.flat_in[:n].copy_(self.host_flat[:n], non_blocking=True)
        self.staged.record()
        dev_ids = self.flat_in[lay["ids"][0]: lay["ids"][1]]
        torch.index_select(self.lm.embed_tokens.weight, 0, dev_ids, out=self.emb_flat[:T])
        for j, (seq, _, media, _) in enumerate(items):
            if media is not None:
                s = slice(starts[j], starts[j] + len(seq))
                self.emb_flat[s] = self.bb.encode_vision(dev_ids[s], self.emb_flat[s], [media[0]],
                                                          graphs=self.bb.vision_graphs(), host_ids=seq)

    @torch.inference_mode()
    def run_flat(self, reqs):
        """Final-norm hidden states at the reads of a round of (prefix, branches[, media]) requests, one replay: per
        request in order, the last token of each branch. None when it does not fit."""
        T = self.flat_shape(reqs)
        if T is None or not self.capture(("flat", T)):
            return None
        self._stage_flat(reqs, T)
        self.graphs[("flat", T)].replay()
        n = sum(max(1, len(r[1])) for r in reqs)
        return self.out_flat[:n].clone()


    @staticmethod
    def _sflat_parts(req):
        """(sequence the prefix part holds, branches, prefix length, entry, start, end of level 1) of a session request
        (prefix, branches, entry or None, start, mark or None); one branch runs with the prefix as one sequence (with the
        batch-invariant products a branch of its own)."""
        pre, sufs, entry, start, mark = req
        seq, sufs = (pre + sufs[0], []) if len(sufs) == 1 and not settings.batch_invariant else (pre, sufs)
        return seq, sufs, len(pre), entry, start, (mark if mark and mark > start else len(seq))

    def sflat_shape(self, reqs) -> int | None:
        """Length of the session flat graph a round of session requests fits, or None."""
        if not reqs or len(reqs) > SESSION_REQUESTS:
            return None
        tokens = nb = chunks = ctx = 0
        for req in reqs:
            seq, sufs, _, _, start, m = self._sflat_parts(req)
            tokens += len(seq) - start + sum(map(len, sufs))
            nb += len(sufs)
            chunks += -(-(m - start) // CHUNK) + -(-(len(seq) - m) // CHUNK) + sum(-(-len(x) // CHUNK) for x in sufs)
            ctx += len(seq)
        if ctx >= SESSION_CTX:
            return None
        for T in FLAT_LADDER:
            if T < tokens:
                continue
            _, B, C = sflat_dims(T)
            if nb + 2 <= B and chunks + -(-(T - tokens) // CHUNK) <= C:
                return T
        return None

    def _stage_sflat(self, reqs, T: int) -> list:
        """Write a session round into the session flat graph's inputs (as `_stage_flat`, with the prefixes in two levels:
        every request's tokens from its resume point to its save point, then every request's rest), the context buffer's
        metadata and saved keys, the resumed states and convolution windows. Returns per request (context start,
        save point or None)."""
        import numpy as np

        sb = self._session_buffers()
        R = SESSION_REQUESTS
        P, B, C = sflat_dims(T)
        S = P + B
        lay = _flat_layout(T, True)
        self.staged.synchronize()
        buf = self.host_flat.numpy()
        col = lambda k: buf[lay[k][0]: lay[k][1]]
        ids, pos, job, br = col("ids"), col("pos"), col("job"), col("branch")
        pos3 = col("pos3").reshape(3, T)
        r = col("ranges").reshape(4, T)
        src = col("src").reshape(3, T)
        cu, offsets, h0, reads = col("cu"), col("offsets"), col("h0"), col("reads")
        chunks = col("chunks").reshape(C, 2)
        ctx, resume, snap, snapconv = col("ctx"), col("resume"), col("snap"), col("snapconv")
        meta = sb["host_meta"].numpy()
        dump = SESSION_CTX - 1
        parts = [self._sflat_parts(req) for req in reqs]
        lens = np.zeros(S, np.int64)
        c_at, starts1, starts2, cstart = 0, [], [], []
        at = 0
        for j, (seq, _, _, entry, start, m) in enumerate(parts):
            starts1.append(at)
            at += m - start
            lens[j] = m - start
        for j, (seq, _, _, entry, start, m) in enumerate(parts):
            starts2.append(at)
            at += len(seq) - m
            lens[R + j] = len(seq) - m
        for j, (seq, *_rest) in enumerate(parts):
            cstart.append(c_at)
            meta[0, c_at: c_at + len(seq)] = j
            meta[1, c_at: c_at + len(seq)] = -1
            meta[2, c_at: c_at + len(seq)] = np.arange(len(seq))
            c_at += len(seq)
        meta[0, c_at:] = -(2 ** 30)

        def where(j, q):
            """Stream index of position q of request j's prefix sequence; T + saved-window row before the resume point;
            -1 before the start."""
            seq, _, _, entry, start, m = parts[j]
            if q < 0:
                return -1
            if q < start:
                return T + 3 * j + (start - q - 1) if entry is not None and start - q <= 3 else -1
            if q < m:
                return starts1[j] + q - start
            return starts2[j] + q - m

        for j, (seq, _, _, entry, start, m) in enumerate(parts):
            for lo, hi, base in ((start, m, starts1[j]), (m, len(seq), starts2[j])):
                if hi <= lo:
                    continue
                s_ = slice(base, base + hi - lo)
                ar = np.arange(lo, hi)
                ids[s_] = seq[lo:hi]
                pos[s_] = ar
                pos3[:, s_] = ar
                job[s_] = j
                br[s_] = -1
                r[0, s_] = cstart[j]
                r[1, s_] = cstart[j] + ar + 1
                r[2:, s_] = 0
                ctx[s_] = cstart[j] + ar
                for d in (1, 2, 3):
                    src[d - 1, s_] = [where(j, q - d) for q in range(lo, hi)]
        nread = b = 0
        for j, (seq, sufs, Pj, entry, start, m) in enumerate(parts):
            n = len(seq)
            if not sufs:
                last = where(j, n - 1)
                reads[nread] = last
                nread += 1
            for sfx in sufs:
                k_ = len(sfx)
                s_ = slice(at, at + k_)
                ar = np.arange(k_)
                ids[s_] = sfx
                pos[s_] = ar + n
                pos3[:, s_] = ar + n
                job[s_] = j
                br[s_] = b
                r[0, s_] = cstart[j]
                r[1, s_] = cstart[j] + n
                r[2, s_] = at
                r[3, s_] = ar + at + 1
                ctx[s_] = dump
                for d in (1, 2, 3):
                    back = ar - d
                    src[d - 1, s_] = [at + x if x >= 0 else where(j, n + x) for x in back]
                lens[P + b] = k_
                h0[R + b] = R + j if len(seq) > m else j
                reads[nread] = at + k_ - 1
                nread += 1
                at += k_
                b += 1
        if at < T:
            s_ = slice(at, T)
            ar = np.arange(at, T)
            ids[s_] = 0
            pos[s_] = 0
            pos3[:, s_] = 0
            job[s_] = -1 - ar
            br[s_] = -2
            r[0:2, s_] = 0
            r[2, s_] = ar
            r[3, s_] = ar + 1
            src[:, s_] = -1
            ctx[s_] = dump
            lens[P + b] = T - at
            h0[R + b] = 0
            b += 1
        h0[:R] = np.arange(R)
        h0[R + b:] = 0
        reads[nread:] = 0
        resume[:] = R
        snap[:] = 0
        snapconv[:] = T + 3 * R
        out = []
        for j, (seq, _, _, entry, start, m) in enumerate(parts):
            if entry is not None:
                resume[j] = j
            mark = reqs[j][4] if reqs[j][4] and reqs[j][4] > start else None
            if mark:
                snap[j] = j
                snapconv[4 * j: 4 * j + 4] = [where(j, mark - d) if where(j, mark - d) >= 0 else T + 3 * R
                                              for d in (1, 2, 3, 4)]
            out.append((cstart[j], mark))
        cu[0] = 0
        cu[1:] = np.cumsum(lens)
        nch = -(-lens // CHUNK)
        offsets[0] = 0
        offsets[1:] = np.cumsum(nch)
        total = int(offsets[-1])
        slot = np.repeat(np.arange(S), nch)
        chunks[:total, 0] = slot
        chunks[:total, 1] = np.arange(total) - offsets[:-1][slot]
        chunks[total:, 0] = S - 1
        chunks[total:, 1] = 0
        n_in = lay["size"]
        self.flat_in[:n_in].copy_(self.host_flat[:n_in], non_blocking=True)
        sb["meta"][:, :SESSION_CTX].copy_(sb["host_meta"], non_blocking=True)
        self.staged.record()
        dev_ids = self.flat_in[lay["ids"][0]: lay["ids"][1]]
        torch.index_select(self.lm.embed_tokens.weight, 0, dev_ids, out=self.emb_flat[:T])
        pairs = []
        for j, (seq, _, _, entry, start, m) in enumerate(parts):
            if entry is None:
                continue
            st = entry.states[start]
            for i in sb["S"]:
                pairs.append((sb["S"][i][j], st["S"][i].view_as(sb["S"][i][j])))
                at = 0
                for c in st["conv"][i]:
                    c2 = c.reshape(c.shape[-2], c.shape[-1])
                    pairs.append((sb["convT"][i][j, at: at + c2.shape[0], 4 - c2.shape[1]:], c2))
                    at += c2.shape[0]
            for i in sb["K"]:
                K0, V0 = entry.kv[i]
                pairs += [(sb["K"][i][cstart[j]: cstart[j] + start], K0[:start]),
                          (sb["V"][i][cstart[j]: cstart[j] + start], V0[:start])]
        _grouped_copy(pairs)
        return out

    def _sflat_entry(self, j: int, cstart: int, mark: int, pairs: list):
        """(states, kv) at request j's save point, in the Entry layout, as views of one device allocation; the copies
        that fill it are appended to `pairs` (a round's go in a few grouped copies)."""
        sb = self.sess
        parts = [sb["snapS"][i][j] for i in sb["snapS"]]
        for i in sb["snapconvT"]:
            win = sb["snapconvT"][i][j]
            kd = self.conv[i][0].shape[1]
            parts += [win[:kd], win[kd: 2 * kd], win[2 * kd:]]
        parts += [t[cstart: cstart + mark] for i in sb["K"] for t in (sb["K"][i], sb["V"][i])]
        offsets, at = [], 0
        for p_ in parts:
            offsets.append(at)
            at += (p_.numel() * p_.element_size() + 255) // 256 * 256
        mem = torch.empty(at, dtype=torch.uint8, device=self.device)
        views = [mem[o: o + p_.numel() * p_.element_size()].view(p_.dtype).view(p_.shape) for p_, o in zip(parts, offsets)]
        pairs += zip(views, parts)
        it = iter(views)
        S = {i: next(it) for i in sb["snapS"]}
        conv = {i: [next(it) for _ in range(3)] for i in sb["snapconvT"]}
        kv = {i: (next(it), next(it)) for i in sb["K"]}
        return {mark: {"S": S, "conv": conv}}, kv

    @torch.inference_mode()
    def run_sflat(self, reqs):
        """One replay for a round of session requests (prefix, branches, entry or None, start, mark or None): (final-norm
        hidden states at the last token of every branch, in request and branch order; per request (states, kv) at its
        save point or None). None when the round does not fit."""
        T = self.sflat_shape(reqs)
        if T is None or not self.capture(("sflat", T)):
            return None
        where = self._stage_sflat(reqs, T)
        self.graphs[("sflat", T)].replay()
        n = sum(max(1, len(req[1])) for req in reqs)
        hidden = self.out_flat[:n].clone()
        pairs = []
        snaps = [self._sflat_entry(j, c, mark, pairs) if mark else None for j, (c, mark) in enumerate(where)]
        _grouped_copy(pairs)
        return hidden, snaps

    def capture_sflat(self, log_fn=None) -> None:
        """Capture the session flat graphs (the session cache on)."""
        t0 = time.perf_counter()
        for T in sorted(FLAT_LADDER, reverse=True):
            self.capture(("sflat", T))
        (log_fn or log.info)(f"captured {len(FLAT_LADDER)} session flat graphs in {time.perf_counter() - t0:.1f}s")


    def _cont_chunks(self, n: int) -> list[tuple[int, int]]:
        """(tokens, width rung) per continuation step for n tokens."""
        out, top = [], self.cont_ladder[-1]
        while n > 0:
            k = min(n, top)
            out.append((k, _rung(self.cont_ladder, k)))
            n -= k
        return out

    def reset(self) -> None:
        for i in self.S:
            self.S[i].zero_()
            for c in self.conv[i]:
                c.zero_()
        self.P.fill_(0)
        self.pos_start.fill_(0)

    def restore(self, entry, pos: int) -> None:
        """Load the state an Entry saved at `pos` into the buffers."""
        st = entry.states[pos]
        for i in self.S:
            self.S[i].copy_(st["S"][i].view_as(self.S[i]))
            for c, t in zip(self.conv[i], st["conv"][i]):
                c.copy_(t.view_as(c))
        for i in self.K:
            K0, V0 = entry.kv[i]
            self.K[i][:pos].copy_(K0[:pos])
            self.V[i][:pos].copy_(V0[:pos])
        self.P.fill_(pos)
        self.pos_start.fill_(pos)

    def snapshot(self, pos: int):
        """(states, kv) of the buffers at P = pos, in the Entry layout the packed path writes."""
        return self._saved(pos, self.S, self.conv)

    def _saved(self, pos: int, S: dict, conv: dict):
        """Copies of a recurrent state, its convolution windows and the keys and values before `pos`, as views of one
        device allocation: the session cache grows by an entry per request, and every fresh device allocation can
        stall for 100s of ms, so an entry takes one (carved from the memory the engine reserves for the cache)."""
        parts = [S[i][0] for i in S] + [c[0] for i in conv for c in conv[i]]
        parts += [t[:pos] for i in self.K for t in (self.K[i], self.V[i])]
        offsets, at = [], 0
        for p in parts:
            offsets.append(at)
            at += (p.numel() * p.element_size() + 255) // 256 * 256
        buf = torch.empty(at, dtype=torch.uint8, device=self.device)
        views = [buf[o: o + p.numel() * p.element_size()].view(p.dtype).view(p.shape) for p, o in zip(parts, offsets)]
        torch._foreach_copy_(views, parts)
        it = iter(views)
        states = {pos: {"S": {i: next(it) for i in S}, "conv": {i: [next(it) for _ in conv[i]] for i in conv}}}
        kv = {i: (next(it), next(it)) for i in self.K}
        return states, kv

    def snapshot_mark(self, pos: int):
        """(states, kv) at the end of segment A of the last two-segment continuation, which ended at `pos`."""
        return self._saved(pos, self.S_mark, self.conv_mark)

    def _stage_cont(self, a: list[int], W: int, b: list[int] = (), Wb: int = 0) -> None:
        self.staged.synchronize()
        if a:
            self.host_cont[: len(a)] = torch.tensor(a, dtype=torch.long)
        if b:
            self.host_cont[W: W + len(b)] = torch.tensor(b, dtype=torch.long)
        self.ids_cont[: W + Wb].copy_(self.host_cont[: W + Wb], non_blocking=True)
        self.staged.record()
        self.clen.fill_(len(a))
        self.clen2.fill_(len(b))

    def extend(self, tokens: list[int]) -> None:
        at = 0
        for n, W in self._cont_chunks(len(tokens)):
            self._stage_cont(tokens[at: at + n], W)
            self.graphs[("cont", W)].replay()
            at += n

    @torch.inference_mode()
    def run_session(self, prefix: list[int], suffixes: list[list[int]], entry, start: int, mark: int | None):
        """A request whose prefix resumes from `entry` at `start` (entry None: from nothing, start 0) and, with `mark`,
        saves its state there. One pass where the shapes allow: the prefix up to the save point (segment A) and what
        follows it (segment B: the rest of the prefix, and with one branch the branch, whose last token is read) in
        one two-segment continuation; with several branches the branch graph follows from the end of the prefix.
        Returns (final-norm hidden states at every branch's last token, (states, kv) at the mark or None), or None
        when a shape does not fit."""
        upto = mark if mark else start
        if upto < start:
            return None
        one = len(suffixes) == 1
        tail = prefix[upto:] + suffixes[0] if one else prefix[upto:]
        tb = _rung(self.tail_ladder, len(tail))
        if tb is None or upto + len(tail) > self.kmax:
            return self._run_session_steps(prefix, suffixes, entry, start, mark)
        lead = self._cont_chunks(upto - start)
        if mark:
            *front, (na, _) = lead
            last = ("cont2", _rung(self.mark_ladder, na), tb)
        else:
            front, last = lead, ("cont", _rung(self.cont_ladder, len(tail)))
        chunks, shapes = [], []
        if not one:
            chunks = [suffixes[i: i + self.bmax] for i in range(0, len(suffixes), self.bmax)]
            shapes = [self._branch_shape(len(c), max(len(s) for s in c)) for c in chunks]
            if any(sh is None for sh in shapes):
                return None
        need = [("cont", W) for _, W in front] + [last] + [("branch", B, Wd) for B, Wd in shapes]
        if not all(self.capture(s) for s in need):
            return None
        if entry is None:
            self.reset()
        else:
            self.restore(entry, start)
        at = start
        for n, W in front:
            self._stage_cont(prefix[at: at + n], W)
            self.graphs[("cont", W)].replay()
            at += n
        if mark:
            self._stage_cont(prefix[at: upto], last[1], tail, tb)
        else:
            self._stage_cont(tail, last[1])
        self.graphs[last].replay()
        snap = self.snapshot_mark(upto) if mark else None
        hidden = self.out[:1].clone() if one else self._run_branches(chunks, shapes)
        return hidden, snap

    def _run_session_steps(self, prefix: list[int], suffixes: list[list[int]], entry, start: int, mark: int | None):
        """run_session for tails too long for the two-segment graphs: continue to the save point, snapshot, then the
        rest of the prefix in front of the one branch (or, when that is too long for the branch graphs, through a
        continuation with the branch up to its last token), or with several branches through one more continuation."""
        upto = mark if mark else start
        if upto < start:
            return None
        rest = prefix[upto:]
        if len(suffixes) > 1:
            more, branches = rest, suffixes
        elif self._branch_shape(1, len(rest) + len(suffixes[0])) is not None:
            more, branches = [], [rest + suffixes[0]]
        else:
            seq = rest + suffixes[0]
            more, branches = seq[:-1], [seq[-1:]]
        end = upto + len(more)
        if end > self.kmax:
            return None
        chunks = [branches[i: i + self.bmax] for i in range(0, len(branches), self.bmax)]
        shapes = [self._branch_shape(len(c), max(len(s) for s in c)) for c in chunks]
        if any(sh is None for sh in shapes):
            return None
        cont = [("cont", W) for _, W in self._cont_chunks(upto - start) + self._cont_chunks(len(more))]
        if not all(self.capture(s) for s in cont + [("branch", B, Wd) for B, Wd in shapes]):
            return None
        if entry is None:
            self.reset()
        else:
            self.restore(entry, start)
        self.extend(prefix[start:upto])
        snap = self.snapshot(upto) if mark else None
        if more:
            self.extend(more)
        hidden = self._run_branches(chunks, shapes)
        return hidden, snap

    @torch.inference_mode()
    def run_media(self, prefix: list[int], suffixes: list[list[int]], vision: dict, positions, offset: int,
                  lazy: bool = True):
        """As `run`, for a prefix holding images or videos: the vision tower and the embedding run eagerly, the
        decoder from the graphs with the M-RoPE positions of the prefix; suffix tokens continue at position
        P + offset."""
        n, P = len(suffixes), len(prefix)
        dev = self.device
        N = self.ids_seq.shape[0]
        seq = prefix + suffixes[0] if n == 1 else prefix
        L = _rung(self.seq_ladder, len(seq))
        if L is None or (n > 1 and L > self.kmax):
            return None
        chunks = [suffixes[i: i + self.bmax] for i in range(0, n, self.bmax)] if n > 1 else []
        shapes = [self._branch_shape(len(c), max(len(s) for s in c)) for c in chunks]
        if any(sh is None for sh in shapes):
            return None
        if not all(self._have(s, lazy) for s in [("mseq", L)] + [("branch", B, Wd) for B, Wd in shapes]):
            return None
        ids = torch.tensor(seq, dtype=torch.long, device=dev)
        h = self.bb.encode_vision(ids, self.lm.embed_tokens(ids), [vision], graphs=self.bb.vision_graphs(),
                                  host_ids=seq)
        rope = torch.as_tensor(positions, dtype=torch.long).reshape(3, P)
        if n == 1:
            tail = torch.arange(P + offset, P + offset + len(suffixes[0])).expand(3, -1)
            rope = torch.cat([rope, tail], 1)
        self.emb_seq[N - len(seq):].copy_(h)
        self.pos3_seq[:, N - len(seq):].copy_(rope.to(dev))
        self.P.fill_(len(seq))
        self.graphs[("mseq", L)].replay()
        if n == 1:
            return self.out[:1].clone()
        self.pos_start.fill_(P + offset)
        return self._run_branches(chunks, shapes)

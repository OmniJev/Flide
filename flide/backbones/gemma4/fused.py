"""The Gemma 4 decoder on fewer, larger kernels.

The weights stay the checkpoint's transformers modules. Projections that read the same input are one matrix product
(q, k and v; gate and up), the modules' weights become views into the merged matrix, so the modules still run and no
memory is added. Between the products run the kernels of kernels/pointwise.py: per-head RMSNorm with the rotary
embedding over the whole head (q, k) and the value norm, then after the attention and after the MLP (and the per-layer
input) one pass for the post-norm, the residual add, the layer scalar and the next pre-norm, and GELU-tanh times the
up projection. Each kernel computes in float32 and rounds once at its output; the rotary tables are float32.
Keys-shared layers read the keys and values of the layer they share with, as transformers does. The mixture-of-experts
blocks (26B-A4B) keep transformers' modules.
"""
from __future__ import annotations

import torch

from ... import settings
from ...kernels import attention as attn_kernel
from ...kernels import pointwise as pw
from ...kernels.fp8 import mm


def rmm(x, w):
    """The products of the last layer at the read rows: the batch-invariant kernel, whose sums do not depend on the
    number of rows (a replay holds as many read rows as its length allows, so the count changes with the round)."""
    from ...kernels.invariant import mm as imm

    return imm(x, w)


def _merge(linears):
    """One weight for several bias-free nn.Linear that read the same input; each module's weight becomes a row slice
    of it."""
    if len(linears) == 1:
        return linears[0].weight
    for m in linears:
        if m.bias is not None:
            raise ValueError("fused projections assume bias-free linear layers")
    w = torch.cat([m.weight.data for m in linears], 0)
    at = 0
    for m in linears:
        n = m.weight.shape[0]
        m.weight.data = w[at: at + n]
        at += n
    return w


class FusedDecoder:
    def __init__(self, lm):
        self.lm = lm
        self.cfg = lm.config
        self.ple = bool(self.cfg.hidden_size_per_layer_input)
        self.dtype = lm.layers[0].mlp.gate_proj.weight.dtype
        self.w = []
        for layer in lm.layers:
            a, mlp = layer.self_attn, layer.mlp
            mods = ([a.q_proj] if a.is_kv_shared_layer else
                    [a.q_proj, a.k_proj] + ([a.v_proj] if a.v_proj is not None else []))
            w = {"qkv": _merge(mods), "o": a.o_proj.weight, "gate_up": _merge([mlp.gate_proj, mlp.up_proj]),
                 "down": mlp.down_proj.weight, "inter": mlp.gate_proj.weight.shape[0], "hd": a.head_dim,
                 "nh": a.q_proj.weight.shape[0] // a.head_dim,
                 "nkv": 0 if a.is_kv_shared_layer else a.k_proj.weight.shape[0] // a.head_dim,
                 "window": int(a.sliding_window or 0)}
            if self.ple:
                w["ple_gate"] = layer.per_layer_input_gate.weight
                w["ple_proj"] = layer.per_layer_projection.weight
            self.w.append(w)

    def tables(self, pos: torch.Tensor) -> dict:
        """Float32 rotary tables [T, hd] per layer type."""
        x = torch.empty(0, device=pos.device, dtype=torch.float32)
        return {t: tuple(c[0] for c in self.lm.rotary_emb(x, pos[None], t)) for t in self.lm.unique_layer_types}

    def per_layer_inputs(self, llm: torch.Tensor, emb: torch.Tensor):
        if not self.ple:
            return None
        lm = self.lm
        ids = lm.get_per_layer_inputs(llm[None], None)[0]
        if not settings.batch_invariant:
            return lm.project_per_layer_inputs(emb[None], ids[None])[0]
        T, n = ids.shape[0], ids.shape[-1]
        norm = lm.per_layer_projection_norm
        p = mm(emb, lm.per_layer_model_projection.weight) * lm.per_layer_model_projection_scale
        p = pw.rms_norm(p.view(-1, n), norm.weight, norm.eps, offset=False).view(T, -1, n)
        return (p + ids) * lm.per_layer_input_scale

    def __call__(self, emb, llm, pos, masks, top: int | None = None, reads: dict | None = None, rows=None):
        """Hidden states after `top` layers (all by default) of the stream emb [T, hidden] (raw ids llm, positions
        pos, the packed metadata per layer type in `masks`), before the final norm; `reads` {layer: rows} collects
        the residual stream after those layers (0: the embeddings). With `rows` (no sessions or reads), the last layer runs its queries, attention, output projection and MLP at those
        rows only and the result holds those rows."""
        lm, cfg = self.lm, self.cfg
        layers = lm.layers
        top = len(layers) if top is None else top
        T = emb.shape[0]
        tabs = self.tables(pos)
        pli = self.per_layer_inputs(llm, emb)
        eps = cfg.rms_norm_eps
        out = {}
        h = emb
        if reads and 0 in reads:
            out[0] = h[reads[0]]
        x = pw.rms_norm(h, layers[0].input_layernorm.weight, layers[0].input_layernorm.eps, offset=False)
        shared: dict[str, tuple] = {}
        for i in range(top):
            layer, w = layers[i], self.w[i]
            a = layer.self_attn
            meta = masks[a.layer_type]
            nh, nkv, hd = w["nh"], w["nkv"], w["hd"]
            cos, sin = tabs[a.layer_type]
            last = rows is not None and i == len(layers) - 1
            if last:
                Tq = rows.shape[0]
                q = pw.gemma_norm_rope(rmm(x.index_select(0, rows), w["qkv"][: nh * hd]).view(Tq, nh, hd),
                                       cos.index_select(0, rows), sin.index_select(0, rows), a.q_norm.weight,
                                       a.q_norm.eps)
                kv, c0 = None if a.is_kv_shared_layer else mm(x, w["qkv"][nh * hd:]), 0
            else:
                Tq = T
                kv, c0 = mm(x, w["qkv"]), nh * hd
                q = pw.gemma_norm_rope(kv[:, : nh * hd].view(T, nh, hd), cos, sin, a.q_norm.weight, a.q_norm.eps)
            if a.is_kv_shared_layer:
                k, v = shared[a.layer_type]
            else:
                kr = kv[:, c0: c0 + nkv * hd].view(T, nkv, hd)
                vr = kv[:, c0 + nkv * hd:].view(T, nkv, hd) if a.v_proj is not None else kr
                k = pw.gemma_norm_rope(kr, cos, sin, a.k_norm.weight, a.k_norm.eps)
                v = pw.gemma_norm_rope(vr, None, None, None, a.v_norm.eps)
                if a.store_full_length_kv:
                    shared[a.layer_type] = (k, v)
            k1, v1 = (k, v) if meta.ctx is None else meta.ctx.keys(a, k, v)
            ac = attn_kernel.config(q.device, q.dtype, hd)
            if last:
                rr = meta.ranges.index_select(0, rows)
                R = attn_kernel.block_ranges(rr[:, 0], rr[:, 1], rr[:, 2], rr[:, 3], ac["BLOCK_M"], ac["BLOCK_N"],
                                             exact=True, groups=Tq)
                o = attn_kernel.attention(q, k1, v1, k, v, tuple(t.index_select(0, rows) for t in meta.tok), meta.k1,
                                          meta.tok, R, 1.0, w["window"], ac, qranges=rr.T)
                h = h.index_select(0, rows)
                pli = pli.index_select(0, rows) if pli is not None else None
            else:
                o = attn_kernel.attention(q, k1, v1, k, v, meta.tok, meta.k1, meta.tok, meta.R(ac), 1.0, w["window"],
                                          ac)
            m_ = rmm if last else mm
            d = m_(o.reshape(Tq, nh * hd), w["o"])
            h, x = pw.sandwich(d, h, layer.post_attention_layernorm.weight, eps,
                               w2=layer.pre_feedforward_layernorm.weight)
            n = w["inter"]
            gu = m_(x, w["gate_up"])
            d = m_(pw.geglu(gu[:, :n], gu[:, n:]), w["down"])
            nxt = layers[i + 1].input_layernorm.weight if i + 1 < top else None
            if self.ple:
                h, _ = pw.sandwich(d, h, layer.post_feedforward_layernorm.weight, eps)
                g = pw.geglu(m_(h, w["ple_gate"]), pli[:, i])
                h, x = pw.sandwich(m_(g, w["ple_proj"]), h, layer.post_per_layer_input_norm.weight, eps,
                                   scale=layer.layer_scalar, w2=nxt)
            else:
                h, x = pw.sandwich(d, h, layer.post_feedforward_layernorm.weight, eps, scale=layer.layer_scalar,
                                   w2=nxt)
            if reads and i + 1 in reads:
                out[i + 1] = h[reads[i + 1]]
        return h, out

    def final(self, h: torch.Tensor) -> torch.Tensor:
        norm = self.lm.norm
        return pw.rms_norm(h, norm.weight, norm.eps, offset=False)

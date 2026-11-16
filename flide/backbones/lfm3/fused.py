"""The LFM3 decoder on flide's kernels.

As LFM2's (lfm2/fused.py: the short convolution over gather indices, merged projections), with what LFM3 adds:

    attention:  q, k = per-head RMSNorm then rotary; o = attention(q, k, v) * sigmoid(gate_proj(x)) per head
    rotary:     per layer type: full attention rotates the first partial_rotary_factor of each head (the rest of the
                frequencies zero, so those dimensions pass), sliding attention the whole head, each with its own theta
    windows:    sliding layers see the last `sliding_window` keys
    norms:      with use_sandwich_norm an RMSNorm after each sublayer as well as before it

Under attention "set" (formats.base.Plan) the full-attention layers read the batch's set key ranges, where a state's
tokens see its whole sequence and a question's the state and itself; the sliding layers and the convolutions read
every sequence causally.
"""
from __future__ import annotations

import torch

from ... import lora
from ...kernels import attention as attn_kernel
from ...kernels import lfm as lk
from ...kernels import pointwise as pw
from ...kernels.fp8 import mm
from ..lfm2.fused import _merge


def inv_freq(cfg, layer_type: str, hd: int, device) -> torch.Tensor:
    """LFM3's inverse frequencies for one layer type over the full head (rotate-half pairs): the first
    partial_rotary_factor * hd / 2 follow the full head's progression, the rest are zero."""
    p = cfg.rope_parameters[layer_type]
    base = float(p["rope_theta"])
    angles = int(float(p.get("partial_rotary_factor", 1.0)) * hd) // 2
    f = 1.0 / (base ** (torch.arange(0, 2 * angles, 2, dtype=torch.float32) / hd))
    return torch.cat([f, torch.zeros(hd // 2 - angles, dtype=torch.float32)]).to(device)


class FusedDecoder:
    def __init__(self, lm):
        self.lm = lm
        self.cfg = lm.config
        self.dtype = lm.layers[0].feed_forward.w1.weight.dtype
        self.eps = self.cfg.norm_eps
        self.sandwich = bool(getattr(self.cfg, "use_sandwich_norm", False))
        self.w = []
        hd = None
        for layer in lm.layers:
            ff = layer.feed_forward
            gu = [ff.w1, ff.w3]
            w = {"gate_up": lora.attached(_merge(gu), gu), "down": lora.attached(ff.w2.weight, [ff.w2]),
                 "inter": ff.w1.weight.shape[0]}
            if layer.is_attention_layer:
                a = layer.self_attn
                mods = [a.q_proj, a.k_proj, a.v_proj, a.gate_proj]
                hd = a.head_dim
                w.update({"qkvg": lora.attached(_merge(mods), mods), "o": lora.attached(a.o_proj.weight, [a.o_proj]),
                          "hd": hd, "nh": a.q_proj.weight.shape[0] // hd, "nkv": a.k_proj.weight.shape[0] // hd,
                          "scale": a.scaling, "window": int(a.sliding_window or 0), "type": a.layer_type})
            else:
                c = layer.conv
                if c.bias:
                    raise ValueError("lfm3: convolution layers with biases are not supported")
                w.update({"in": lora.attached(c.in_proj.weight, [c.in_proj]),
                          "out": lora.attached(c.out_proj.weight, [c.out_proj]),
                          "conv": c.conv.weight.squeeze(1).contiguous(), "C": c.in_proj.weight.shape[1]})
            self.w.append(w)
        self.keep = max((w["conv"].shape[1] - 1 for w in self.w if "conv" in w), default=0)
        dev = lm.embed_tokens.weight.device   # on the device, so a graph capture copies nothing from the host
        types = sorted({w["type"] for w in self.w if "type" in w})
        self.inv_freq = {t: inv_freq(self.cfg, t, hd, dev) for t in types}

    def tables(self, pos: torch.Tensor) -> dict:
        """Float32 rotary tables [T, hd] per attention layer type (rotate-half pairs)."""
        out = {}
        for t, f in self.inv_freq.items():
            e = pos.float()[:, None] * f[None]
            e = torch.cat([e, e], dim=-1)
            out[t] = (e.cos().contiguous(), e.sin().contiguous())
        return out

    def norm(self, h: torch.Tensor, module) -> torch.Tensor:
        return pw.rms_norm(h, module.weight, module.variance_epsilon, offset=False)

    def __call__(self, emb, pos, meta, prev, top: int | None = None, reads: dict | None = None, set_meta=None,
                 sl_meta=None):
        """Hidden states after `top` layers (all by default) of the stream emb [T, hidden] (positions pos, the packed
        attention metadata `meta`, `set_meta` for the full-attention layers of a batch with "set" sequences, `sl_meta`
        for the sliding layers (ranges cut to the window), the convolution's gather indices `prev`), before the final
        norm; `reads` {layer: rows} collects the residual
        stream after those layers (0: the embeddings)."""
        layers = self.lm.layers
        top = len(layers) if top is None else top
        T = emb.shape[0]
        rope = self.tables(pos)
        out = {}
        h = emb
        if reads and 0 in reads:
            out[0] = h[reads[0]]
        x = self.norm(h, layers[0].operator_norm) if top else None
        for i in range(top):
            layer, w = layers[i], self.w[i]
            if "qkvg" in w:
                a = layer.self_attn
                nh, nkv, hd = w["nh"], w["nkv"], w["hd"]
                cos, sin = rope[w["type"]]
                y = mm(x, w["qkvg"])
                q = pw.gemma_norm_rope(y[:, : nh * hd].view(T, nh, hd), cos, sin, a.q_norm.weight,
                                       a.q_norm.variance_epsilon)
                k = pw.gemma_norm_rope(y[:, nh * hd: (nh + nkv) * hd].view(T, nkv, hd), cos, sin, a.k_norm.weight,
                                       a.k_norm.variance_epsilon)
                v = y[:, (nh + nkv) * hd: (nh + 2 * nkv) * hd].reshape(T, nkv, hd).contiguous()
                ac = attn_kernel.config(q.device, q.dtype, hd)
                if set_meta is not None and not w["window"]:
                    o = attn_kernel.attention(q, k, v, k, v, meta.tok, meta.k1, meta.tok, set_meta.R(ac), w["scale"], 0,
                                              ac, qranges=set_meta.ranges.T)
                elif w["window"]:
                    if sl_meta is None:
                        raise ValueError("lfm3: a sliding layer needs its windowed ranges (sl_meta)")
                    o = attn_kernel.attention(q, k, v, k, v, meta.tok, meta.k1, meta.tok, sl_meta.R(ac), w["scale"],
                                              w["window"], ac)
                else:
                    o = attn_kernel.attention(q, k, v, k, v, meta.tok, meta.k1, meta.tok, meta.R(ac), w["scale"], 0, ac)
                lk.sigmoid_gate_(o, y[:, (nh + 2 * nkv) * hd:])
                d = mm(o.reshape(T, nh * hd), w["o"])
            else:
                d = mm(lk.short_conv(mm(x, w["in"]), w["conv"], prev), w["out"])
            nxt = layers[i + 1].operator_norm.weight if i + 1 < top else None
            if self.sandwich:   # h += post_norm(d), and the next pre-norm, in one pass each
                h, x = pw.sandwich(d, h, layer.post_operator_norm.weight, self.eps, w2=layer.ffn_norm.weight)
                f = mm(pw.swiglu(mm(x, w["gate_up"]), w["inter"]), w["down"])
                h, x = pw.sandwich(f, h, layer.post_ffn_norm.weight, self.eps, w2=nxt)
            else:
                h = h + d
                f = mm(pw.swiglu(mm(self.norm(h, layer.ffn_norm), w["gate_up"]), w["inter"]), w["down"])
                h = h + f
                x = self.norm(h, layers[i + 1].operator_norm) if nxt is not None else None
            if reads and i + 1 in reads:
                out[i + 1] = h[reads[i + 1]]
        return h, out

    def final(self, h: torch.Tensor) -> torch.Tensor:
        return self.norm(h, self.lm.embedding_norm)

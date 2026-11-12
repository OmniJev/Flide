"""The LFM2 decoder on flide's kernels.

The weights stay the checkpoint's transformers modules; projections that read the same input are one matrix product
(q, k and v; w1 and w3), the modules' weights becoming views into the merged matrix. A layer is either a gated short
convolution or full attention, then a SwiGLU feed-forward, each behind its own RMSNorm (operator_norm, ffn_norm):

    conv:       B, C, x = in_proj(norm(h)); y = C * conv(B * x); h += out_proj(y)
    attention:  q, k = per-head RMSNorm then rotary; h += out_proj(attention(q, k, v))
    ffn:        h += w2(silu(w1(norm(h))) * w3(norm(h)))

The convolution is depthwise and causal over each token's own sequence: a branch's first tokens read the last inputs
of its job's prefix, never the token packed before it. Each token's previous tokens are gather indices (`conv_prev`),
so one product of the stream serves every packing, and the sums run in float32 with one rounding at the output.
"""
from __future__ import annotations

import torch

from ... import lora
from ...kernels import attention as attn_kernel
from ...kernels import pointwise as pw
from ...kernels.fp8 import mm


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


def conv_prev(T: int, Tp: int, cu_prefix: torch.Tensor, cu_suffix: torch.Tensor, branch: torch.Tensor,
              branch_job: torch.Tensor, keep: int) -> torch.Tensor:
    """[T, keep] stream indices of each token's 1st..keep-th previous token in its own sequence (a branch continues its
    job's prefix), T where there is none (the convolution's zero padding). The stream holds every prefix, then every
    branch (runtime/pack.py)."""
    dev = branch.device
    t = torch.arange(T, device=dev)
    k = torch.arange(1, keep + 1, device=dev)
    out = torch.full((T, keep), T, dtype=torch.long, device=dev)
    if Tp:   # prefix tokens: earlier tokens of the same prefix
        jp = torch.bucketize(t[:Tp], cu_prefix[1:], right=True)
        prev = t[:Tp, None] - k[None]
        out[:Tp] = torch.where(prev >= cu_prefix[jp][:, None], prev, T)
    if T > Tp:   # branch tokens: earlier tokens of the branch, then the end of the job's prefix
        br = branch[Tp:]
        off = t[Tp:] - Tp - cu_suffix[br]
        job = branch_job[br]
        lo, hi = cu_prefix[job], cu_prefix[job + 1]
        back = off[:, None] - k[None]
        inside = t[Tp:, None] - k[None]
        pre = hi[:, None] + back
        out[Tp:] = torch.where(back >= 0, inside, torch.where(pre >= lo[:, None], pre, T))
    return out


def short_conv(bx: torch.Tensor, weight: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
    """Depthwise causal convolution of bx [T, C] with weight [C, K] (the last tap reads the token itself), each token's
    earlier inputs gathered through `prev` [T, K - 1]; float32 sums, rounded once."""
    T, C = bx.shape
    K = weight.shape[1]
    padded = torch.cat([bx, bx.new_zeros(1, C)])
    w = weight.float()
    acc = bx.float() * w[:, K - 1]
    for j in range(1, K):
        acc += padded.index_select(0, prev[:, j - 1]).float() * w[:, K - 1 - j]
    return acc.to(bx.dtype)


class FusedDecoder:
    def __init__(self, lm):
        self.lm = lm
        self.cfg = lm.config
        self.dtype = lm.layers[0].feed_forward.w1.weight.dtype
        self.eps = self.cfg.norm_eps
        self.w = []
        for layer in lm.layers:
            ff = layer.feed_forward
            gu = [ff.w1, ff.w3]
            w = {"gate_up": lora.attached(_merge(gu), gu), "down": lora.attached(ff.w2.weight, [ff.w2]),
                 "inter": ff.w1.weight.shape[0]}
            if layer.is_attention_layer:
                a = layer.self_attn
                mods = [a.q_proj, a.k_proj, a.v_proj]
                w.update({"qkv": lora.attached(_merge(mods), mods), "o": lora.attached(a.out_proj.weight, [a.out_proj]),
                          "hd": a.head_dim, "nh": a.q_proj.weight.shape[0] // a.head_dim,
                          "nkv": a.k_proj.weight.shape[0] // a.head_dim, "scale": a.scaling})
            else:
                c = layer.conv
                if c.bias:
                    raise ValueError("lfm2: convolution layers with biases are not supported")
                w.update({"in": lora.attached(c.in_proj.weight, [c.in_proj]),
                          "out": lora.attached(c.out_proj.weight, [c.out_proj]),
                          "conv": c.conv.weight.squeeze(1).contiguous(), "C": c.in_proj.weight.shape[1]})
            self.w.append(w)
        self.keep = max((w["conv"].shape[1] - 1 for w in self.w if "conv" in w), default=0)
        theta = (getattr(self.cfg, "rope_parameters", None) or {}).get("rope_theta") or getattr(self.cfg, "rope_theta")
        hd = next((w["hd"] for w in self.w if "hd" in w), self.cfg.hidden_size // self.cfg.num_attention_heads)
        dev = lm.embed_tokens.weight.device   # on the device, so a graph capture copies nothing from the host
        self.inv_freq = (1.0 / (float(theta) ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))).to(dev)

    def tables(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Float32 rotary tables [T, hd] (rotate-half pairs)."""
        f = pos.float()[:, None] * self.inv_freq[None]
        e = torch.cat([f, f], dim=-1)
        return e.cos().contiguous(), e.sin().contiguous()

    def norm(self, h: torch.Tensor, module) -> torch.Tensor:
        return pw.rms_norm(h, module.weight, module.variance_epsilon, offset=False)

    def __call__(self, emb, pos, meta, prev, top: int | None = None, reads: dict | None = None):
        """Hidden states after `top` layers (all by default) of the stream emb [T, hidden] (positions pos, the packed
        attention metadata `meta`, the convolution's gather indices `prev`), before the final norm; `reads`
        {layer: rows} collects the residual stream after those layers (0: the embeddings)."""
        layers = self.lm.layers
        top = len(layers) if top is None else top
        T = emb.shape[0]
        cos, sin = self.tables(pos)
        out = {}
        h = emb
        if reads and 0 in reads:
            out[0] = h[reads[0]]
        for i in range(top):
            layer, w = layers[i], self.w[i]
            x = self.norm(h, layer.operator_norm)
            if "qkv" in w:
                a = layer.self_attn
                nh, nkv, hd = w["nh"], w["nkv"], w["hd"]
                qkv = mm(x, w["qkv"])
                q = pw.gemma_norm_rope(qkv[:, : nh * hd].view(T, nh, hd), cos, sin, a.q_layernorm.weight,
                                       a.q_layernorm.variance_epsilon)
                k = pw.gemma_norm_rope(qkv[:, nh * hd: (nh + nkv) * hd].view(T, nkv, hd), cos, sin,
                                       a.k_layernorm.weight, a.k_layernorm.variance_epsilon)
                v = qkv[:, (nh + nkv) * hd:].reshape(T, nkv, hd).contiguous()
                ac = attn_kernel.config(q.device, q.dtype, hd)
                o = attn_kernel.attention(q, k, v, k, v, meta.tok, meta.k1, meta.tok, meta.R(ac), w["scale"], 0, ac)
                d = mm(o.reshape(T, nh * hd), w["o"])
            else:
                C = w["C"]
                bcx = mm(x, w["in"])
                bx = bcx[:, :C] * bcx[:, 2 * C:]
                y = bcx[:, C: 2 * C] * short_conv(bx, w["conv"], prev)
                d = mm(y, w["out"])
            h = h + d
            x = self.norm(h, layer.ffn_norm)
            n = w["inter"]
            h = h + mm(pw.swiglu(mm(x, w["gate_up"]), n), w["down"])
            if reads and i + 1 in reads:
                out[i + 1] = h[reads[i + 1]]
        return h, out

    def final(self, h: torch.Tensor) -> torch.Tensor:
        lm = self.lm
        return self.norm(h, lm.embedding_norm if hasattr(lm, "embedding_norm") else lm.norm)

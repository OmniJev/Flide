"""Packed forward for the qwen3_5 family (Qwen3.5 / Qwen3.8 decoders: Gated DeltaNet linear-attention
layers plus gated full-attention layers, 3 to 1).

The weights are the HuggingFace modules of a loaded `Qwen3_5ForCausalLM`. The two token mixers:

- Gated DeltaNet runs twice per layer. The first call covers every prefix of the stream (varlen,
  `cu_prefix`) and returns each job's final conv window and recurrent state; the second covers every
  suffix (`cu_suffix`) starting from the state of the suffix's own job. A suffix therefore sees its
  prefix and nothing of the other suffixes, as in a forked cache.
- Full attention is one call of the Triton kernel (kernels/attention.py) over the whole stream, each
  query seeing its job's prefix and its own branch, keys not after the query.

Hidden states are gathered where the batch's reads point: after any layer (raw residual stream) or
after the last layer with the final norm. The forward stops after the highest layer a read needs.

The forward works on the HF weights with few, large kernels: projections that read the same input are
one matrix product (the HF weights become views into the merged matrix, so no memory is added), the
pointwise work between products runs in a few Triton kernels, fla applies the decay gate, the beta
sigmoid and grouped value heads inside its kernels, and the convolution output is never concatenated
and split again. The last layer runs at the read rows only.
"""
from __future__ import annotations


import numpy as np
import torch

from ... import lora
from ...formats.base import FINAL
from ...kernels.fla import pin_fla_size_buckets
from ...runtime.pack import PackedBatch
from ..base import Backbone
from ...kernels import attention as attn_kernel
from ...kernels.fp8 import mm, mm_into
from .ops import add_rms, final_norm, gated_rms, last_qkv, merge, qk_norm_rope, rms, rows_ranges, swiglu
from .ops import gate as gate_out


_FP8_RE = None


def _fp8_target(name: str) -> bool:
    """Decoder projections that go to float8: the MLP, the attention projections, Gated DeltaNet's input
    (but its small gate projections) and output projections."""
    global _FP8_RE
    if _FP8_RE is None:
        import re

        _FP8_RE = re.compile(r"(^|\.)layers\.\d+\.(mlp\.(gate_proj|up_proj|down_proj)|linear_attn\.(in_proj_qkv|"
                             r"in_proj_z|out_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj))\.weight$")
    return "visual" not in name and _FP8_RE.search(name) is not None


class _Headless(torch.nn.Module):
    """A Qwen3_5Model or Qwen3_5TextModel where the backbone expects a model with an output head; it has none."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_output_embeddings(self):
        return None


class Qwen35(Backbone):
    model_types = ("qwen3_5", "qwen3_5_text")
    block_attention = True

    @classmethod
    def load(cls, model_dir: str, device: str = "cuda:0", dtype: str = "bfloat16", vision: bool | None = None,
             loader: str = "direct", quantize: str | None = None, tp: tuple[int, int] | None = None,
             merge_lora: bool = False, **options) -> "Qwen35":
        """`vision` None loads the vision tower when the checkpoint has one. `loader` "direct" reads the safetensors
        straight onto the device (flide.backbones.load); "hf" uses transformers' from_pretrained (needs accelerate);
        "none" reads no weights (zeros).
        `quantize` "fp8": the decoder's projections are read as float8 (quantized while read, or as an FP8 checkpoint
        stores them), never held in 16 bits. `tp` (rank, size): this rank's shard of a tensor parallel model
        (runtime/tp.py; read in 16 bits, cut, then quantized). `merge_lora`: a LoRA adapter merged into the weights as
        they are read (flide/lora.py)."""
        import transformers

        config = transformers.AutoConfig.from_pretrained(model_dir, local_files_only=True)
        if vision is None:
            vision = getattr(config, "vision_config", None) is not None
        archs = getattr(config, "architectures", None) or []
        headless = "Qwen3_5Model" in archs or "Qwen3_5TextModel" in archs
        if headless:
            arch = transformers.Qwen3_5Model if "Qwen3_5Model" in archs else transformers.Qwen3_5TextModel
            cfg, key_map, skip = config, [], ("mtp.",)
        elif vision:
            arch, cfg, key_map, skip = transformers.Qwen3_5ForConditionalGeneration, config, [], ("mtp.",)
        else:
            arch, cfg = transformers.Qwen3_5ForCausalLM, config.get_text_config()
            key_map, skip = [(r"^model\.language_model\.", "model.")], ("model.visual.", "mtp.")
        model = None
        lean_fp8 = quantize == "fp8" and tp is None
        store = {} if lean_fp8 else None
        lean = _fp8_target if lean_fp8 else None
        if loader == "none":
            from ..load import blank, build

            model = build(arch, cfg, torch.device(device), getattr(torch, dtype), lean=lean)
            blank(model, torch.device(device), lean=lean, fp8=store)
        elif loader == "direct":
            from ..load import build, load_into

            try:
                model = build(arch, cfg, torch.device(device), getattr(torch, dtype), lean=lean)
                load_into(model, model_dir, device, key_map=key_map, skip=skip, lean=lean, fp8=store, merge=merge_lora)
            except (ValueError, FileNotFoundError) as exc:
                import logging

                logging.getLogger("flide.load").warning("direct load failed (%s); using transformers", exc)
                model, store = None, ({} if lean_fp8 else None)
                torch.cuda.empty_cache()
        if model is None:
            store = None
            model = arch.from_pretrained(model_dir, config=cfg, dtype=getattr(torch, dtype),
                                         device_map={"": str(device)}, low_cpu_mem_usage=True, local_files_only=True)
            model.eval()
        names = {id(m): n for n, m in model.named_modules()}
        if headless:
            model = _Headless(model)
        return cls(model, quantize=quantize, fp8_store=(store, names) if store else None, tp=tp, **options)

    def __init__(self, causal_lm, cuda_graphs: bool = False, quantize: str | None = None, fp8_store=None,
                 tp: tuple[int, int] | None = None):
        self.causal_lm = causal_lm
        inner = causal_lm.model
        self.vlm = inner if hasattr(inner, "visual") else None
        self.lm = inner.language_model if self.vlm is not None else inner
        self.config = self.lm.config
        self.layers = self.lm.layers[: self.config.num_hidden_layers]
        self.num_layers = len(self.layers)
        self.layer_types = list(self.config.layer_types)
        p0 = next(causal_lm.parameters())
        self.device, self.dtype = p0.device, p0.dtype
        self.cuda_graphs = cuda_graphs
        pin_fla_size_buckets()
        self.fp8 = False
        self.tp = tp if tp is not None and tp[1] > 1 else None
        if self.tp is not None:
            self.shard(*self.tp)
        self._prepare_fused()
        if fp8_store is not None:
            self._fp8_from_store(*fp8_store)
        elif quantize == "fp8":
            self.quantize_fp8()
        elif quantize is not None:
            raise ValueError(f"quantize {quantize!r}: only fp8")
        if self.tp is not None:
            from ...runtime.tp import RowParallel

            for w in self._w:
                for key in ("down", "out", "o"):
                    if key in w:
                        w[key] = RowParallel(w[key])

    def shard(self, rank: int, size: int) -> None:
        """Keep this rank's part of every decoder projection (tensor parallelism, runtime/tp.py): its heads of the
        attention and Gated DeltaNet layers and its columns of the MLP in the products that read the layer input,
        the matching input columns of the products that write the residual stream. Attention heads and their key
        and value heads split evenly; Gated DeltaNet keeps whole groups (a key head with its value heads)."""
        from ...runtime.tp import split

        def take(t, idx, dim=0):
            return (t.index_select(dim, idx) if isinstance(idx, torch.Tensor)
                    else t.narrow(dim, idx.start, idx.stop - idx.start)).contiguous()

        def keep(param, idx, dim=0):
            return torch.nn.Parameter(take(param.data, idx, dim), requires_grad=False)

        def rows(lin, idx):
            if lin.bias is not None:
                raise ValueError("tensor parallel: projections with a bias")
            lin.weight = keep(lin.weight, idx)
            if getattr(lin, "lora", None) is not None:
                lin.lora = (lin.lora[0], take(lin.lora[1], idx))

        def cols(lin, idx):
            lin.weight = keep(lin.weight, idx, 1)
            if getattr(lin, "lora", None) is not None:
                lin.lora = (take(lin.lora[0], idx, 1), lin.lora[1])

        for i, layer in enumerate(self.layers):
            mlp = layer.mlp
            part = split(mlp.gate_proj.weight.shape[0], rank, size)
            rows(mlp.gate_proj, part)
            rows(mlp.up_proj, part)
            cols(mlp.down_proj, part)
            if self.layer_types[i] == "linear_attention":
                m = layer.linear_attn
                kd, vd, nv = m.key_dim, m.value_dim, m.num_v_heads
                split(m.num_k_heads, rank, size)
                ks, vs, hs = split(kd, rank, size), split(vd, rank, size), split(nv, rank, size)
                ar = lambda sl, base=0: torch.arange(sl.start, sl.stop, device=self.device) + base
                qkv = torch.cat([ar(ks), ar(ks, kd), ar(vs, 2 * kd)])
                rows(m.in_proj_qkv, qkv)
                m.conv1d.weight = keep(m.conv1d.weight, qkv)
                rows(m.in_proj_z, vs)
                rows(m.in_proj_b, hs)
                rows(m.in_proj_a, hs)
                m.A_log = keep(m.A_log, hs)
                m.dt_bias = keep(m.dt_bias, hs)
                cols(m.out_proj, vs)
                m.num_k_heads //= size
                m.num_v_heads //= size
                m.key_dim //= size
                m.value_dim //= size
                m.conv_dim = 2 * m.key_dim + m.value_dim
            else:
                m = layer.self_attn
                hd = m.head_dim
                nh, nkv = m.q_proj.weight.shape[0] // (2 * hd), m.k_proj.weight.shape[0] // hd
                hq, hk = split(nh, rank, size), split(nkv, rank, size)
                rows(m.q_proj, slice(hq.start * 2 * hd, hq.stop * 2 * hd))
                rows(m.k_proj, slice(hk.start * hd, hk.stop * hd))
                rows(m.v_proj, slice(hk.start * hd, hk.stop * hd))
                cols(m.o_proj, slice(hq.start * hd, hq.stop * hd))
        torch.cuda.empty_cache()

    def request_graphs(self):
        """CUDA graphs for one request at a time (graphs.py), built on first use; None without CUDA graphs."""
        if not self.cuda_graphs or self.device.type != "cuda":
            return None
        if getattr(self, "_graphs", None) is None:
            from .graphs import RequestGraphs

            self._graphs = RequestGraphs(self)
        return self._graphs

    def output_head(self):
        head = self.causal_lm.get_output_embeddings()
        if head is None:
            raise ValueError("this checkpoint has no output head")
        return lora.output_head(head)

    def _stop_layer(self, b: PackedBatch) -> int:
        ints = [k for k in b.reads if k is not FINAL]
        if FINAL in b.reads:
            return self.num_layers
        top = max(ints) if ints else 0
        if top > self.num_layers:
            raise ValueError(f"read after layer {top}, the model has {self.num_layers}")
        return top

    def vision_positions(self, ids: list[int], vision: dict):
        """Rotary positions [3, P] of a prefix holding media, and the offset of the first token after it
        (position P + offset): the checkpoint's own M-RoPE indexing (`get_rope_index`), vectorized (vision.py)."""
        if self.vlm is None:
            raise ValueError("this checkpoint was loaded without its vision tower")
        from .vision import mrope_positions

        types = vision.get("mm_token_type_ids")
        types = types.reshape(-1).numpy() if types is not None else np.zeros(len(ids), np.int64)
        return mrope_positions(types, vision.get("image_grid_thw"), vision.get("video_grid_thw"),
                               self.vlm.config.vision_config.spatial_merge_size)

    def start_vision(self, vision: dict) -> None:
        """Queue one request's vision tower on the device now, from its CUDA graphs, so it runs while the host still
        plans and stages the request; `encode_vision` with graphs takes these features for that same request."""
        graphs = self.vision_graphs()
        early = getattr(self, "_early_vision", None)
        if graphs is None or self.vlm is None or (early is not None and early[0] is vision):
            return
        self._early_vision = None
        feats = {}
        for pix, grid, token in self._vision_kinds():
            if self._carries(vision, pix):
                feats[token] = graphs(self._pixels(vision, pix), vision[grid])
        self._early_vision = (vision, feats)

    def _vision_kinds(self):
        cfg = self.vlm.config
        return (("pixel_values", "image_grid_thw", cfg.image_token_id),
                ("pixel_values_videos", "video_grid_thw", cfg.video_token_id))

    @staticmethod
    def _carries(v: dict, pix: str) -> bool:
        return (v.get(pix) is not None or v.get(pix + "_u8") is not None
                or (pix == "pixel_values" and v.get("pixel_images_u8") is not None))

    def vision_graphs(self):
        """The vision tower from CUDA graphs, one per set of image grids (vision.py); None without graphs."""
        if not self.cuda_graphs or self.vlm is None or self.device.type != "cuda":
            return None
        if getattr(self, "_vision_graphs", None) is None:
            from .vision import VisionGraphs

            self._vision_graphs = VisionGraphs(self.vlm)
        return self._vision_graphs

    @torch.inference_mode()
    def encode_vision(self, ids: torch.Tensor, h: torch.Tensor, visions: list, graphs=None,
                      host_ids=None) -> torch.Tensor:
        """Token embeddings with the vision features written into the image and video placeholders, as the
        checkpoint's own forward does; `visions` are the jobs' inputs in the order their prefixes appear.
        `graphs` (one request) runs the tower from VisionGraphs. `host_ids`, the same tokens on the host, lets the
        placeholder count be checked there, so nothing here waits for the device."""
        early = getattr(self, "_early_vision", None)
        early = early[1] if graphs is not None and early is not None and len(visions) == 1 and visions[0] is early[0] \
            else None
        if early is not None:
            self._early_vision = None
        for pix, grid, token in self._vision_kinds():
            items = [v for v in visions if self._carries(v, pix)]
            if not items:
                continue
            if early is not None and token in early:
                feats = early[token].to(h.dtype)
            elif graphs is not None:
                pv = torch.cat([self._pixels(v, pix) for v in items])
                feats = graphs(pv, torch.cat([v[grid] for v in items])).to(h.dtype)
            else:
                pv = torch.cat([self._pixels(v, pix) for v in items])
                gr = torch.cat([v[grid] for v in items]).to(self.device)
                feats = torch.cat(self.vlm.get_image_features(pv, gr, return_dict=True).pooler_output, 0).to(h.dtype)
            mask = ids == token
            n = list(host_ids).count(token) if host_ids is not None else int(mask.sum())
            if n != feats.shape[0]:
                raise ValueError(f"{n} placeholder tokens for {feats.shape[0]} vision features")
            h = h.masked_scatter(mask[:, None], feats)
        return h

    def _pixels(self, v: dict, pix: str) -> torch.Tensor:
        """A request's pixel patches on the device, as the image processor gives them (vision.pixels)."""
        from .vision import pixels

        return pixels(v, pix, self.device)

    def _inputs(self, b: PackedBatch):
        T = b.n_tokens
        h = b.embeds if b.embeds is not None else self.lm.embed_tokens(b.input_ids)
        if b.vision and b.embeds is None:
            h = self.encode_vision(b.input_ids, h, b.vision)
        pos3 = b.pos3.view(3, 1, T) if b.pos3 is not None else b.positions.view(1, 1, T).expand(3, 1, T)
        cos, sin = self.lm.rotary_emb(h[None], pos3)
        return h, cos, sin

    def _prepare_fused(self):
        """Per layer, the weights of every matrix product (projections that read the same input merged into one) and
        the sizes the forward needs, so the products go through `mm` whatever the weights are stored as."""
        def merged(mods):
            return None if any(m.weight.is_meta for m in mods) else lora.attached(merge(mods), mods)

        def single(mod):
            return None if mod.weight.is_meta else lora.attached(mod.weight, [mod])

        self._w = []
        for i, layer in enumerate(self.layers):
            mlp = layer.mlp
            w = {"gate_up": merged([mlp.gate_proj, mlp.up_proj]), "inter": mlp.gate_proj.weight.shape[0],
                 "down": single(mlp.down_proj)}
            if self.layer_types[i] == "linear_attention":
                m = layer.linear_attn
                w["in"] = merged([m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a])
                w["conv"] = m.conv1d.weight.squeeze(1)
                w["out"] = single(m.out_proj)
            else:
                m = layer.self_attn
                w["qkv"] = merged([m.q_proj, m.k_proj, m.v_proj])
                w["o"] = single(m.o_proj)
                w["heads"] = (m.q_proj.weight.shape[0] // (2 * m.head_dim), m.k_proj.weight.shape[0] // m.head_dim)
            self._w.append(w)

    def _fp8_from_store(self, store: dict, names: dict) -> None:
        """The fused weights from the float8 copies the loader read (rows of merged projections concatenated, with
        their scales; the gate projections of Gated DeltaNet's input projection as 16-bit tail rows)."""
        from ...kernels import fp8

        def get(mod):
            return store[names[id(mod)] + ".weight"]

        def joined(mods, tail=None, tail_mods=None):
            qs, ss = zip(*(get(m) for m in mods))
            return lora.attached(fp8.Fp8Weight(torch.cat(qs), torch.cat(ss), tail, self.dtype), mods + (tail_mods or []))

        for i, layer in enumerate(self.layers):
            w, mlp = self._w[i], layer.mlp
            w["gate_up"] = joined([mlp.gate_proj, mlp.up_proj])
            w["down"] = joined([mlp.down_proj])
            if self.layer_types[i] == "linear_attention":
                m = layer.linear_attn
                tail = torch.cat([m.in_proj_b.weight.data, m.in_proj_a.weight.data])
                w["in"] = joined([m.in_proj_qkv, m.in_proj_z], tail, [m.in_proj_b, m.in_proj_a])
                w["out"] = joined([m.out_proj])
            else:
                m = layer.self_attn
                w["qkv"] = joined([m.q_proj, m.k_proj, m.v_proj])
                w["o"] = joined([m.o_proj])
        store.clear()
        self.fp8 = True

    def quantize_fp8(self) -> int:
        """Every decoder matrix product to float8 weights (kernels/fp8.py): the merged projections, the
        output projections and the MLP; the small gate projections of Gated DeltaNet (the last rows of its merged
        input projection), the embedding, the output head and the vision tower stay in 16 bits. The 16-bit copies
        are freed. Returns the number of weights converted."""
        from ...kernels import fp8

        if not fp8.supported(self.device):
            raise RuntimeError("FP8 needs compute capability 8.9 or newer (Ada, Hopper, Blackwell)")
        n = 0
        for i, layer in enumerate(self.layers):
            w = self._w[i]
            for key in ("gate_up", "down", "in", "out", "qkv", "o"):
                if key in w:
                    tail = 2 * layer.linear_attn.num_v_heads if key == "in" else 0
                    if isinstance(w[key], lora.LowRank):
                        w[key] = w[key].with_base(fp8.Fp8Weight.from_weight(w[key].base, tail))
                    else:
                        w[key] = fp8.Fp8Weight.from_weight(w[key], tail)
                    n += 1
            mods = [layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj]
            if self.layer_types[i] == "linear_attention":
                m = layer.linear_attn
                mods += [m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a, m.out_proj]
            else:
                m = layer.self_attn
                mods += [m.q_proj, m.k_proj, m.v_proj, m.o_proj]
            for mod in mods:
                mod.weight.data = torch.empty(0, device=self.device, dtype=self.dtype)
        self.fp8 = True
        torch.cuda.empty_cache()
        return n

    @torch.inference_mode()
    def forward(self, b: PackedBatch, pad_to: int | None = None) -> dict:
        """{layer key: hidden states [rows, hidden]} at the batch's read indices."""
        top = self._stop_layer(b)
        h, cos, sin = self._inputs(b)
        cos, sin = cos[0], sin[0]
        out = {0: h[b.reads[0]]} if 0 in b.reads else {}
        if top == 0:
            return out
        mask = None
        last = (top - 1 if top == self.num_layers and set(b.reads) <= {FINAL, 0}
                and self.layer_types[top - 1] != "linear_attention" else -1)
        n0 = self.layers[0].input_layernorm
        x = rms(h, n0.weight, n0.eps)
        for i, layer in enumerate(self.layers[:top]):
            w = self._w[i]
            if self.layer_types[i] == "linear_attention":
                if len(b.levels) > 1 or b.resumes:
                    d = self._gated_delta_levels(layer.linear_attn, w, x, b, i)
                else:
                    d = self._gated_delta_fused(layer.linear_attn, w, x, b)
            else:
                if mask is None:
                    mask = self._attn_meta(b)
                if i == last:
                    rows = b.reads[FINAL]
                    d = self._attention_fused(layer.self_attn, w, x, cos, sin, mask, b, i, rows=rows)
                    n2 = layer.post_attention_layernorm
                    h, x = add_rms(h.index_select(0, rows), d, n2.weight, n2.eps)
                    h = h + mm(swiglu(mm(x, w["gate_up"]), w["inter"]), w["down"])
                    out[FINAL] = final_norm(self.lm.norm, h)
                    return out
                d = self._attention_fused(layer.self_attn, w, x, cos, sin, mask, b, i)
            n2 = layer.post_attention_layernorm
            h, x = add_rms(h, d, n2.weight, n2.eps)
            d = mm(swiglu(mm(x, w["gate_up"]), w["inter"]), w["down"])
            if i + 1 < top:
                n1 = self.layers[i + 1].input_layernorm
                h, x = add_rms(h, d, n1.weight, n1.eps)
            else:
                h = h + d
            if i + 1 in b.reads:
                out[i + 1] = h[b.reads[i + 1]]
        if FINAL in b.reads:
            out[FINAL] = final_norm(self.lm.norm, h[b.reads[FINAL]])
        return out

    def _gated_delta_fused(self, m, w, x, b: PackedBatch):
        from fla.modules.convolution import causal_conv1d
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        T, Tp = x.shape[0], b.n_prefix_tokens
        kd, vd, nv = m.key_dim, m.value_dim, m.num_v_heads
        cd = 2 * kd + vd
        p = mm(x, w["in"])
        z = p[:, cd: cd + vd]
        beta = p[:, cd + vd: cd + vd + nv]
        a = p[:, cd + vd + nv:]
        act = m.activation
        parts = []
        for lo, hi, heads, dim in ((0, kd, m.num_k_heads, m.head_k_dim), (kd, 2 * kd, m.num_k_heads, m.head_k_dim),
                                   (2 * kd, cd, nv, m.head_v_dim)):
            src = p[None, :, lo:hi]
            cw = w["conv"][lo:hi]
            yp, st = causal_conv1d(src[:, :Tp], cw, None, activation=act, output_final_state=True,
                                   cu_seqlens=b.cu_prefix, cu_seqlens_cpu=b.cu_prefix_cpu)
            ys, _ = causal_conv1d(src[:, Tp:], cw, None, activation=act, initial_state=st.index_select(0, b.branch_job),
                                  cu_seqlens=b.cu_suffix, cu_seqlens_cpu=b.cu_suffix_cpu)
            parts.append((yp.view(1, Tp, heads, dim), ys.view(1, T - Tp, heads, dim)))
        (qp, qs), (kp, ks), (vp, vs) = parts
        gate = dict(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, A_log=m.A_log, dt_bias=m.dt_bias,
                    use_beta_sigmoid_in_kernel=True)
        op, state = chunk_gated_delta_rule(qp, kp, vp, g=a[None, :Tp], beta=beta[None, :Tp], output_final_state=True,
                                           cu_seqlens=b.cu_prefix, cu_seqlens_cpu=b.cu_prefix_cpu, **gate)
        os_, _ = chunk_gated_delta_rule(qs, ks, vs, g=a[None, Tp:], beta=beta[None, Tp:],
                                        initial_state=state.index_select(0, b.branch_job), output_final_state=False,
                                        cu_seqlens=b.cu_suffix, cu_seqlens_cpu=b.cu_suffix_cpu, **gate)
        out = torch.empty(T, w["out"].shape[0], device=x.device, dtype=x.dtype)
        mm_into(gated_rms(op[0], z[:Tp], m.norm.weight, m.norm.variance_epsilon), w["out"], out[:Tp])
        mm_into(gated_rms(os_[0], z[Tp:], m.norm.weight, m.norm.variance_epsilon), w["out"], out[Tp:])
        return out

    def _gated_delta_levels(self, m, w, x, b: PackedBatch, i: int):
        """Gated DeltaNet when prefixes resume from saved states or are cut at marks: one call per prefix level,
        each starting from the conv window and recurrent state (float32) the level before left, then the suffixes.
        States at the marks are saved into `b.saved`."""
        from fla.modules.convolution import causal_conv1d
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        T, Tp, J = x.shape[0], b.n_prefix_tokens, b.n_jobs
        kd, vd, nv = m.key_dim, m.value_dim, m.num_v_heads
        cd = 2 * kd + vd
        p = mm(x, w["in"])
        z = p[:, cd: cd + vd]
        beta = p[:, cd + vd: cd + vd + nv]
        a = p[:, cd + vd + nv:]
        act = m.activation
        spec = ((0, kd, m.num_k_heads, m.head_k_dim), (kd, 2 * kd, m.num_k_heads, m.head_k_dim),
                (2 * kd, cd, nv, m.head_v_dim))
        gate = dict(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, A_log=m.A_log, dt_bias=m.dt_bias,
                    use_beta_sigmoid_in_kernel=True)
        conv: list = [None, None, None]
        S = None
        if b.resumes:
            entry0 = b.resumes[0][1]
            ref = entry0.states[b.resumes[0][2]]
            conv = [x.new_zeros((J,) + tuple(t.shape), dtype=t.dtype) for t in ref["conv"][i]]
            S = torch.zeros((J,) + tuple(ref["S"][i].shape), device=x.device, dtype=torch.float32)
            for j, entry, start in b.resumes:
                st = entry.states[start]
                S[j].copy_(st["S"][i])
                for c, t in zip(conv, st["conv"][i]):
                    c[j].copy_(t)
        outs = []
        for lv in b.levels:
            lo, hi, jobs = lv["lo"], lv["hi"], lv["jobs"]
            ys = []
            for part, (plo, phi, heads, dim) in enumerate(spec):
                init = conv[part].index_select(0, jobs) if conv[part] is not None else None
                y, st = causal_conv1d(p[None, lo:hi, plo:phi], w["conv"][plo:phi], None, activation=act,
                                      initial_state=init, output_final_state=True, cu_seqlens=lv["cu"],
                                      cu_seqlens_cpu=lv["cu_cpu"])
                if conv[part] is None:
                    conv[part] = st.new_zeros((J,) + tuple(st.shape[1:]))
                conv[part].index_copy_(0, jobs, st)
                ys.append(y.view(1, hi - lo, heads, dim))
            q, k, v = ys
            init = S.index_select(0, jobs) if S is not None else None
            o, st = chunk_gated_delta_rule(q, k, v, g=a[None, lo:hi], beta=beta[None, lo:hi], initial_state=init,
                                           output_final_state=True, cu_seqlens=lv["cu"], cu_seqlens_cpu=lv["cu_cpu"],
                                           **gate)
            if S is None:
                S = torch.zeros((J,) + tuple(st.shape[1:]), device=x.device, dtype=torch.float32)
            S.index_copy_(0, jobs, st.float())
            for _, j, e in lv["ends"]:
                rec = b.saved.setdefault(j, {"states": {}, "kv": {}})["states"].setdefault(e, {"S": {}, "conv": {}})
                rec["S"][i] = S[j].clone()
                rec["conv"][i] = [c[j].clone() for c in conv]
            outs.append(o)
        op = torch.cat(outs, 1) if len(outs) > 1 else outs[0]
        bj = b.branch_job
        ys = []
        for part, (plo, phi, heads, dim) in enumerate(spec):
            y, _ = causal_conv1d(p[None, Tp:, plo:phi], w["conv"][plo:phi], None, activation=act,
                                 initial_state=conv[part].index_select(0, bj), cu_seqlens=b.cu_suffix,
                                 cu_seqlens_cpu=b.cu_suffix_cpu)
            ys.append(y.view(1, T - Tp, heads, dim))
        qs, ks, vs = ys
        os_, _ = chunk_gated_delta_rule(qs, ks, vs, g=a[None, Tp:], beta=beta[None, Tp:],
                                        initial_state=S.index_select(0, bj), output_final_state=False,
                                        cu_seqlens=b.cu_suffix, cu_seqlens_cpu=b.cu_suffix_cpu, **gate)
        out = torch.empty(T, w["out"].shape[0], device=x.device, dtype=x.dtype)
        mm_into(gated_rms(op[0], z[:Tp], m.norm.weight, m.norm.variance_epsilon), w["out"], out[:Tp])
        mm_into(gated_rms(os_[0], z[Tp:], m.norm.weight, m.norm.variance_epsilon), w["out"], out[Tp:])
        return out

    def _attn_meta(self, b: PackedBatch):
        """Token metadata and per-block key ranges for the Triton kernel, built once per batch."""
        meta = b.cache.get("attn")
        if meta is None:
            hd = self.config.head_dim
            cfg = attn_kernel.config(self.device, self.dtype, hd)
            tok = (b.job.int(), b.branch.int(), b.positions.int())
            r = b.key_ranges
            R = attn_kernel.block_ranges(r[:, 0], r[:, 1], r[:, 2], r[:, 3], cfg["BLOCK_M"], cfg["BLOCK_N"],
                                         exact=True, groups=b.attn_runs)
            k1 = b.ctx["meta"] if b.ctx is not None else tok
            meta = b.cache["attn"] = ("triton", tok, R, cfg, k1, r.T)
        return meta

    def _attention_fused(self, m, w, x, cos, sin, mask, b: PackedBatch | None = None, i: int = -1, rows=None):
        """`rows`: the last layer read only there (Triton kernel): queries and output of those rows, keys and values
        of every row."""
        T = x.shape[0]
        hd = m.head_dim
        nh, nkv = w["heads"]
        if rows is not None:
            _, tok, R, cfg, k1, qr = mask
            q, gate, k, v = last_qkv(m, w, x, rows, cos, sin)
            R, qr = rows_ranges(qr, rows, cfg)
            tq = tuple(t.index_select(0, rows) for t in tok)
        else:
            p = mm(x, w["qkv"])
            qg = p[:, : nh * 2 * hd].view(T, nh, 2 * hd)
            q, gate = qg[..., :hd], qg[..., hd:]
            k = p[:, nh * 2 * hd: nh * 2 * hd + nkv * hd].view(T, nkv, hd)
            v = p[:, nh * 2 * hd + nkv * hd:].view(T, nkv, hd)
            _, tok, R, cfg, k1, qr = mask
            q, k = qk_norm_rope(q, k, cos, sin, m.q_norm.weight, m.k_norm.weight, m.q_norm.eps)
            tq = tok
        if b is None or b.ctx is None:
            o = attn_kernel.attention(q, k, v, k, v, tq, tok, tok, R, m.scaling, 0, cfg, qranges=qr)
            return mm(gate_out(o, gate), w["o"])
        C, Tp = b.ctx, b.n_prefix_tokens
        Kc = k.new_empty((C["len"], nkv, hd))
        Vc = v.new_empty((C["len"], nkv, hd))
        Kc.index_copy_(0, C["index"], k[:Tp])
        Vc.index_copy_(0, C["index"], v[:Tp])
        for j, entry, start in b.resumes:
            c0 = C["start"][j]
            K0, V0 = entry.kv[i]
            Kc[c0: c0 + start].copy_(K0[:start])
            Vc[c0: c0 + start].copy_(V0[:start])
        o = attn_kernel.attention(q, Kc, Vc, k, v, tq, k1, tok, R, m.scaling, 0, cfg, qranges=qr)
        for j in b.marks:
            c0, P = C["start"][j], C["P"][j]
            b.saved.setdefault(j, {"states": {}, "kv": {}})["kv"][i] = (Kc[c0: c0 + P].clone(), Vc[c0: c0 + P].clone())
        return mm(gate_out(o, gate), w["o"])

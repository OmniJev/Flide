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
from ...runtime.pack import PackedBatch
from ..base import Backbone
from ...kernels import attention as attn_kernel
from ...kernels.fp8 import mm, mm_into
from .ops import add_rms, ffn, final_norm, gated_rms, last_qkv, lerp, merge, qk_norm_rope, rms, rows_ranges, swiglu
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


_TP_RE = None


def _tp_target(name: str) -> bool:
    """Decoder tensors that tensor parallelism cuts (`_tp_cuts`): the MLP and attention projections, and Gated
    DeltaNet's projections, convolution and per-head decay."""
    global _TP_RE
    if _TP_RE is None:
        import re

        _TP_RE = re.compile(r"(^|\.)layers\.\d+\.(mlp\.(gate_proj|up_proj|down_proj)\.weight|linear_attn\.(in_proj_qkv|"
                            r"in_proj_z|in_proj_b|in_proj_a|out_proj|conv1d)\.weight|linear_attn\.(A_log|dt_bias)|"
                            r"self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight)$")
    return "visual" not in name and _TP_RE.search(name) is not None


def _tp_cuts(layers, layer_types, rank: int, size: int) -> list[dict]:
    """Per decoder layer, {tensor path in the layer: (dimension, the part this rank keeps)} for tensor parallelism
    (runtime/tp.py): its heads of the attention and Gated DeltaNet layers and its columns of the MLP in the products
    that read the layer input, the matching input columns of the products that write the residual stream. Attention
    heads and their key and value heads split evenly; Gated DeltaNet keeps whole groups (a key head with its value
    heads)."""
    from ...runtime.tp import split

    out = []
    for i, layer in enumerate(layers):
        mlp = layer.mlp
        part = split(mlp.gate_proj.weight.shape[0], rank, size)
        cuts = {"mlp.gate_proj.weight": (0, part), "mlp.up_proj.weight": (0, part), "mlp.down_proj.weight": (1, part)}
        if layer_types[i] == "linear_attention":
            m = layer.linear_attn
            kd, vd, nv = m.key_dim, m.value_dim, m.num_v_heads
            split(m.num_k_heads, rank, size)
            ks, vs, hs = split(kd, rank, size), split(vd, rank, size), split(nv, rank, size)
            qkv = (ks, slice(kd + ks.start, kd + ks.stop), slice(2 * kd + vs.start, 2 * kd + vs.stop))   # q, k, v rows
            cuts.update({"linear_attn.in_proj_qkv.weight": (0, qkv), "linear_attn.conv1d.weight": (0, qkv),
                         "linear_attn.in_proj_z.weight": (0, vs), "linear_attn.in_proj_b.weight": (0, hs),
                         "linear_attn.in_proj_a.weight": (0, hs), "linear_attn.A_log": (0, hs),
                         "linear_attn.dt_bias": (0, hs), "linear_attn.out_proj.weight": (1, vs)})
        else:
            m = layer.self_attn
            hd = m.head_dim
            nh, nkv = m.q_proj.weight.shape[0] // (2 * hd), m.k_proj.weight.shape[0] // hd
            hq, hk = split(nh, rank, size), split(nkv, rank, size)
            kv = slice(hk.start * hd, hk.stop * hd)
            cuts.update({"self_attn.q_proj.weight": (0, slice(hq.start * 2 * hd, hq.stop * 2 * hd)),
                         "self_attn.k_proj.weight": (0, kv), "self_attn.v_proj.weight": (0, kv),
                         "self_attn.o_proj.weight": (1, slice(hq.start * hd, hq.stop * hd))})
        out.append(cuts)
    return out


def _read_tp(arch, cfg, model_dir: str, device: str, dtype: str, tp: tuple[int, int], to_fp8: bool, key_map, skip,
             merge: bool):
    """(model, float8 store, the parts): the model holding only this rank's part of the tensors `_tp_cuts` cuts, each
    cut as it is read (load.py), so a rank never holds the whole model; with `to_fp8` the projections FP8 takes go into
    the store as the lean FP8 load puts them, the rest stay in 16 bits."""
    import re

    from ..load import build, load_into

    dev = torch.device(device)
    model = build(arch, cfg, dev, getattr(torch, dtype), lean=_tp_target)
    text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    layers = next(m for n, m in model.named_modules() if (n == "layers" or n.endswith(".layers"))
                  and "visual" not in n and "mtp" not in n and isinstance(m, torch.nn.ModuleList))
    cuts = _tp_cuts(list(layers)[: text.num_hidden_layers], list(text.layer_types), *tp)
    where = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")

    def cut(name):
        m = where.search(name)
        i, path = int(m.group(1)), m.group(2)
        if i >= len(cuts) or path not in cuts[i]:
            raise ValueError(f"tensor parallel: no part for {name}")
        return (*cuts[i][path], to_fp8 and _fp8_target(name))

    store, held = {}, {}
    load_into(model, model_dir, device, key_map=key_map, skip=skip, lean=_tp_target, fp8=store, merge=merge, cut=cut,
              held=held)
    for name, t in held.items():
        mod_name, pn = name.rsplit(".", 1)
        setattr(model.get_submodule(mod_name), pn, torch.nn.Parameter(t, requires_grad=False))
    for name, (q, _) in store.items():   # a float8 part's 16-bit weight stays on the meta device, at the part's shape
        mod_name, pn = name.rsplit(".", 1)
        mod = model.get_submodule(mod_name)
        meta = torch.empty(q.shape, dtype=getattr(mod, pn).dtype, device="meta")
        setattr(mod, pn, torch.nn.Parameter(meta, requires_grad=False))
    return model.eval(), store, cuts


class _Headless(torch.nn.Module):
    """A Qwen3_5Model or Qwen3_5TextModel where the backbone expects a model with an output head; it has none."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_output_embeddings(self):
        return None


class Qwen35(Backbone):
    model_types = ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text")
    block_attention = True
    prefix_attention = True
    midloop = None   # (passes, blocks) once set_midloop is called

    def set_midloop(self, cfg: dict) -> None:
        """Recurrent depth, a checkpoint's decision_config "midloop" (its parameter-free damped Euler form): each block
        [a, b] of layers (inclusive) runs `passes` times over the whole sequence, every pass from the stream x that
        entered it, x <- x + (G(x) - x) / passes in float32, rounded back; blocks in order, apart."""
        c = dict(cfg or {})
        if c.get("strategy", "euler") != "euler" or c.get("conv", "A") != "A":
            raise ValueError(f"midloop {c}: only the damped Euler form (strategy euler, conv A)")
        if int(c.get("n_gate") or 0) or int(c.get("n_lora") or 0):
            raise ValueError("midloop: only the parameter-free form (n_gate = n_lora = 0)")
        if float(c.get("gamma", 1.0)) != 1.0:
            raise ValueError(f"midloop gamma {c['gamma']}: only 1")
        blocks = sorted(((int(a), int(b)) for a, b in (c.get("blocks") or [[c["a"], c["b"]]])), key=lambda w: w[1])
        passes = int(c.get("passes") or 1)
        if passes < 1 or any(not 0 <= a <= b < self.num_layers for a, b in blocks) or \
                any(blocks[k][0] <= blocks[k - 1][1] for k in range(1, len(blocks))):
            raise ValueError(f"midloop {c}: blocks apart and inside the {self.num_layers} layers, passes >= 1")
        if self.tp is not None:
            raise ValueError("midloop: no tensor parallelism")
        self.midloop = (passes, blocks) if passes > 1 else None

    def layer_order(self, top: int) -> list[tuple[int, bool, bool]]:
        """(layer, a midloop pass starts there, one ends there) in the order layers 0 .. top - 1 run."""
        if self.midloop is None:
            return [(i, False, False) for i in range(top)]
        passes, blocks = self.midloop
        out, i = [], 0
        for a, b in blocks:
            if a >= top:
                break
            if b >= top:
                raise ValueError(f"midloop: a read inside the block of layers {a} to {b}")
            out += [(j, False, False) for j in range(i, a)]
            out += [(j, j == a, j == b) for _ in range(passes) for j in range(a, b + 1)]
            i = b + 1
        return out + [(j, False, False) for j in range(i, top)]

    def next_input(self, h, d, nxt: int | None, start=None):
        """The stream after a layer's feed-forward output d and layer `nxt`'s normed input (None after the last one);
        `start`: the stream that entered the midloop pass ending here, which the pass's output steps from."""
        if start is None:
            if nxt is None:
                return h + d, None
            n1 = self.layers[nxt].input_layernorm
            return add_rms(h, d, n1.weight, n1.eps, self.xdtype)
        h = lerp(start, h + d, 1.0 / self.midloop[0])
        if nxt is None:
            return h, None
        n1 = self.layers[nxt].input_layernorm
        return h, rms(h, n1.weight, n1.eps, self.xdtype)

    @classmethod
    def load(cls, model_dir: str, device: str = "cuda:0", dtype: str = "bfloat16", vision: bool | None = None,
             loader: str = "direct", quantize: str | None = None, tp: tuple[int, int] | None = None,
             merge_lora: bool = False, **options) -> "Qwen35":
        """`vision` None loads the vision tower when the checkpoint has one. `loader` "direct" reads the safetensors
        straight onto the device (flide.backbones.load); "hf" uses transformers' from_pretrained (needs accelerate);
        "none" reads no weights (zeros).
        `quantize` "fp8": the decoder's projections are read as float8 (quantized while read, or as an FP8 checkpoint
        stores them), never held in 16 bits. `tp` (rank, size): this rank's shard of a tensor parallel model
        (runtime/tp.py; with the "direct" loader each tensor is cut to this rank's part as it is read, then quantized,
        so a rank never holds the whole model; otherwise read in 16 bits, cut, then quantized). `merge_lora`: a LoRA adapter merged into the weights as
        they are read (flide/lora.py)."""
        import transformers

        config = transformers.AutoConfig.from_pretrained(model_dir, local_files_only=True)
        if vision is None:
            vision = getattr(config, "vision_config", None) is not None
        archs = getattr(config, "architectures", None) or []
        moe = config.model_type.startswith("qwen3_5_moe")   # a mixture of experts in every layer (kernels/moe.py)
        if moe and tp is not None:
            raise ValueError("Qwen3.5 MoE: no tensor parallelism yet")
        pre = "Qwen3_5Moe" if moe else "Qwen3_5"
        headless = f"{pre}Model" in archs or f"{pre}TextModel" in archs
        if headless:   # a backbone without an output head (decision checkpoints that ship their own readout)
            arch = getattr(transformers, f"{pre}Model" if f"{pre}Model" in archs else f"{pre}TextModel")
            cfg, key_map, skip = config, [], ("mtp.",)
        elif vision:
            arch, cfg, key_map, skip = getattr(transformers, f"{pre}ForConditionalGeneration"), config, [], ("mtp.",)
        else:
            arch, cfg = getattr(transformers, f"{pre}ForCausalLM"), config.get_text_config()
            key_map, skip = [(r"^model\.language_model\.", "model.")], ("model.visual.", "mtp.")
        model, cut = None, None
        lean_fp8 = quantize == "fp8" and tp is None and not moe   # experts are read in 16 bits, then quantized
        store = {} if lean_fp8 else None
        lean = _fp8_target if lean_fp8 else None
        if loader == "none":
            from ..load import blank, build

            model = build(arch, cfg, torch.device(device), getattr(torch, dtype), lean=lean)
            blank(model, torch.device(device), lean=lean, fp8=store)
        elif loader == "direct":
            from ..load import build, load_into

            try:
                if tp is not None and tp[1] > 1:   # each rank reads its part only
                    model, store, cut = _read_tp(arch, cfg, model_dir, device, dtype, tp, quantize == "fp8", key_map,
                                                 skip, merge_lora)
                else:
                    model = build(arch, cfg, torch.device(device), getattr(torch, dtype), lean=lean)
                    load_into(model, model_dir, device, key_map=key_map, skip=skip, lean=lean, fp8=store,
                              merge=merge_lora)
            except (ValueError, FileNotFoundError) as exc:
                import logging

                logging.getLogger("flide.load").warning("direct load failed (%s); using transformers", exc)
                model, store, cut = None, ({} if lean_fp8 else None), None
                torch.cuda.empty_cache()
        if model is None:
            store = None   # transformers loads 16 bits; the forward quantizes them after
            model = arch.from_pretrained(model_dir, config=cfg, dtype=getattr(torch, dtype),
                                         device_map={"": str(device)}, low_cpu_mem_usage=True, local_files_only=True)
            model.eval()
        names = {id(m): n for n, m in model.named_modules()}
        if headless:
            model = _Headless(model)
        return cls(model, quantize=quantize, fp8_store=(store, names) if store else None, tp=tp, tp_cut=cut, **options)

    def __init__(self, causal_lm, cuda_graphs: bool = False, quantize: str | None = None, fp8_store=None,
                 tp: tuple[int, int] | None = None, float32_residual: bool = False, tp_cut: list | None = None,
                 float32_vision: bool = False):
        """`float32_residual`: the residual stream stays in float32 between the layers; each norm rounds the input of
        the products that read it once, and the final norm's output stays in float32 (off by default). `tp_cut`: the
        parts the loader already cut the weights to (`_read_tp`). `float32_vision`: the vision tower in float32, its
        products on TF32 tensor cores in the vision graphs, its features rounded once as they enter the decoder (off by
        default)."""
        self.causal_lm = causal_lm
        inner = causal_lm.model
        self.vlm = inner if hasattr(inner, "visual") else None   # Qwen3_5Model: vision tower and M-RoPE indexing
        self.lm = inner.language_model if self.vlm is not None else inner
        self.config = self.lm.config
        self.layers = self.lm.layers[: self.config.num_hidden_layers]
        self.num_layers = len(self.layers)
        self.layer_types = list(self.config.layer_types)
        p0 = next(causal_lm.parameters())
        self.device, self.dtype = p0.device, p0.dtype
        self.float32_vision = bool(float32_vision) and self.vlm is not None
        if self.float32_vision:
            self.vlm.visual.float()
        self.float32_residual = bool(float32_residual)
        self.xdtype = self.dtype if self.float32_residual else None   # the norms' output type (None: the stream's)
        self.cuda_graphs = cuda_graphs
        self.fp8 = False
        self.tp = tp if tp is not None and tp[1] > 1 else None
        if self.tp is not None:
            self.shard(*self.tp, cuts=tp_cut)
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

    def shard(self, rank: int, size: int, cuts: list | None = None) -> None:
        """Keep this rank's part of every decoder projection (tensor parallelism, runtime/tp.py; the parts are
        `_tp_cuts`). A LoRA keeps the same rows of its B where its weight keeps rows, and the same columns of its A
        where it keeps columns (the ranks' partial sums add up). `cuts`: the parts the loader cut the weights to already
        (`_read_tp`), so only the LoRA parts and the layer sizes change here."""
        from ..load import part

        weights = cuts is None
        if weights:
            cuts = _tp_cuts(self.layers, self.layer_types, rank, size)
        for i, layer in enumerate(self.layers):
            for path, (dim, idx) in cuts[i].items():
                mod_path, pn = path.rsplit(".", 1)
                mod = layer.get_submodule(mod_path)
                if isinstance(mod, torch.nn.Linear):
                    if dim == 0 and mod.bias is not None:
                        raise ValueError("tensor parallel: projections with a bias")
                    lo = getattr(mod, "lora", None)
                    if lo is not None:
                        mod.lora = (lo[0], part(lo[1], idx)) if dim == 0 else (part(lo[0], idx, 1), lo[1])
                if weights:
                    setattr(mod, pn, torch.nn.Parameter(part(getattr(mod, pn).data, idx, dim), requires_grad=False))
            if self.layer_types[i] == "linear_attention":
                m = layer.linear_attn
                m.num_k_heads //= size
                m.num_v_heads //= size
                m.key_dim //= size
                m.value_dim //= size
                m.conv_dim = 2 * m.key_dim + m.value_dim
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
        if self.midloop is not None and (any(ints) or b.resumes or b.marks):
            raise ValueError("midloop: reads after the final layer only, no session states")
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

    def vision_token_ids(self) -> set[int]:
        """The placeholder ids the vision features replace."""
        return {token for _, _, token in self._vision_kinds()}

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

            self._vision_graphs = VisionGraphs(self.vlm, tf32=self.float32_vision)
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

    def stream(self, h):
        """The residual stream as the layers carry it (after the rotary tables are made from the embeddings)."""
        return h.float() if self.float32_residual else h

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
        def merged(mods):   # weights read as float8 stay on the meta device; their products come from the FP8 store
            return None if any(m.weight.is_meta for m in mods) else lora.attached(merge(mods), mods)

        def single(mod):
            return None if mod.weight.is_meta else lora.attached(mod.weight, [mod])

        self._w = []
        for i, layer in enumerate(self.layers):
            mlp = layer.mlp
            if hasattr(mlp, "experts"):   # a mixture of experts: router, experts as [E, out, in], shared expert
                se = mlp.shared_expert
                w = {"router": single(mlp.gate), "topk": mlp.gate.top_k, "experts": mlp.experts.gate_up_proj,
                     "experts_down": mlp.experts.down_proj, "inter": mlp.experts.intermediate_dim,
                     "shared_gate_up": merged([se.gate_proj, se.up_proj]), "shared_inter": se.gate_proj.weight.shape[0],
                     "shared_down": single(se.down_proj), "shared_gate": single(mlp.shared_expert_gate)}
            else:
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

        def get(mod):   # each copy leaves the store as it is joined, so the float8 weights are never held twice
            return store.pop(names[id(mod)] + ".weight")

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

    @torch.no_grad()   # weights built by transformers' classes require grad
    def quantize_fp8(self) -> int:
        """Every decoder matrix product to float8 weights (kernels/fp8.py): the merged projections, the
        output projections and the MLP; the small gate projections of Gated DeltaNet (the last rows of its merged
        input projection), the embedding, the output head and the vision tower stay in 16 bits. In a mixture of experts
        only the routed experts go to float8 (kernels/moe.py). The 16-bit copies are freed. Returns the number of
        weights converted."""
        from ...kernels import fp8

        if not fp8.supported(self.device):
            raise RuntimeError("FP8 needs compute capability 8.9 or newer (Ada, Hopper, Blackwell)")
        n = 0
        for i, layer in enumerate(self.layers):
            w = self._w[i]
            if "experts" in w:   # a mixture of experts: its routed experts (kernels/moe.py) alone; the rest of the
                # layer keeps 16 bits, which in float8 moves the answers more than the experts do
                from ...kernels.moe import quantize_experts

                for key in ("experts", "experts_down"):
                    w[key], w[key + "_scale"] = quantize_experts(w[key])
                    n += 1
                ex = layer.mlp.experts
                ex.gate_up_proj.data = torch.empty(0, device=self.device, dtype=self.dtype)
                ex.down_proj.data = torch.empty(0, device=self.device, dtype=self.dtype)
                continue
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
            for mod in mods:   # the modules held views of the 16-bit weights; only the float8 ones are used now
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
        order = self.layer_order(top)
        # the last layer only at the final reads, when nothing else reads after it
        last = (top - 1 if top == self.num_layers and set(b.reads) <= {FINAL, 0}
                and self.layer_types[top - 1] != "linear_attention" and not order[-1][2] else -1)
        h = self.stream(h)
        n0 = self.layers[0].input_layernorm
        x = rms(h, n0.weight, n0.eps, self.xdtype)
        start = None
        for p, (i, first, end) in enumerate(order):
            layer = self.layers[i]
            if first:
                start = h
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
                    h, x = add_rms(h.index_select(0, rows), d, n2.weight, n2.eps, self.xdtype)
                    h = h + ffn(w, x)
                    out[FINAL] = final_norm(self.lm.norm, h)
                    return out
                d = self._attention_fused(layer.self_attn, w, x, cos, sin, mask, b, i)
            n2 = layer.post_attention_layernorm
            h, x = add_rms(h, d, n2.weight, n2.eps, self.xdtype)
            d = ffn(w, x)
            h, x = self.next_input(h, d, order[p + 1][0] if p + 1 < len(order) else None, start if end else None)
            if i + 1 in b.reads:
                out[i + 1] = h[b.reads[i + 1]]
        if FINAL in b.reads:
            out[FINAL] = final_norm(self.lm.norm, h[b.reads[FINAL]])
        return out

    def _gated_delta_fused(self, m, w, x, b: PackedBatch):
        from ...kernels.fla.modules.causal_conv1d.ops import causal_conv1d
        from ...kernels.fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule

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
        from ...kernels.fla.modules.causal_conv1d.ops import causal_conv1d
        from ...kernels.fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule

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
        # prefix keys and values from the context buffer: saved ones for resumed jobs, the stream's for the rest
        C, Tp = b.ctx, b.n_prefix_tokens
        Kc = k.new_empty((C["len"], nkv, hd))
        Vc = v.new_empty((C["len"], nkv, hd))
        Kc.index_copy_(0, C["index"], k[:Tp])
        Vc.index_copy_(0, C["index"], v[:Tp])
        for j, entry, start in b.resumes:
            c0 = C["start"][j]
            K0, V0 = entry.kv[i]
            for t, lo, hi in K0.runs():        # a page at a time, not a saved block at a time
                Kc[c0 + lo: c0 + hi].copy_(t)
            for t, lo, hi in V0.runs():
                Vc[c0 + lo: c0 + hi].copy_(t)
        o = attn_kernel.attention(q, Kc, Vc, k, v, tq, k1, tok, R, m.scaling, 0, cfg, qranges=qr)
        for j in b.marks:
            c0, P = C["start"][j], C["P"][j]
            slot = b.saved.setdefault(j, {"states": {}, "kv": {}})["kv"]
            if hasattr(slot, "write"):
                slot.write(i, Kc[c0: c0 + P], Vc[c0: c0 + P])   # straight into the arena rows reserved before the forward
            else:
                slot[i] = (Kc[c0: c0 + P].clone(), Vc[c0: c0 + P].clone())
        return mm(gate_out(o, gate), w["o"])

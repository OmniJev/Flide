"""Packed forward for Gemma 4 decoders (gemma4, gemma4_text, gemma4_unified: E2B, E4B, 12B, 26B-A4B, 31B).

Gemma 4 has no recurrent layers, so the packed stream needs nothing but attention that knows which tokens see which:
every other step is per token. The decoder runs as the checkpoint's own transformers modules over the stream as one
batch row (per-layer input embeddings, the two rotary tables, keys shared from earlier layers, mixture-of-experts
blocks, sandwich norms: all as transformers computes them), with positions restarting per job, and the attention of
every layer is flide's Triton kernel over the packed metadata ("same job, key in the prefix or in the query's
own branch, key not after the query"; sliding layers add "fewer than `sliding_window` positions back"). The mask
transformers would build is replaced by that metadata, passed where the mask goes.

The output head is the tied embedding; Gemma 4 soft-caps its logits (tanh(x / c) * c), exposed as `logit_softcap`
for families that read rows of the head.

Images and videos, as transformers' Gemma4Model does it: placeholder tokens embed as padding, the vision tower's
features replace them, the per-layer inputs read the padding id there; on the sizes trained with
`use_bidirectional_attention = "vision"` (12B, 31B) the tokens of one image see each other in the sliding layers
(block numbers in the attention metadata), the global layers stay causal.
"""
from __future__ import annotations

import torch

import numpy as np

from ...formats.base import FINAL
from ...kernels import attention as attn_kernel
from ...runtime.pack import PackedBatch
from ..base import Backbone

IMPL = "flide"


class _Meta:
    """Packed attention metadata standing in for transformers' mask; block ranges per query tile size. With sessions,
    the first key source is the context buffer (`ctx`, metadata `k1`). The query tiles start at every request's
    prefix and branches (`groups` bounds their runs; `tiles`, the unwindowed ranges of a sliding layer, decide them), so
    a request's attention sums its keys the same way whatever else shares the round."""

    def __init__(self, tok, ranges, k1=None, ctx=None, groups: int = 0, tiles=None):
        self.tok = tok
        self.ranges = ranges
        self.k1 = tok if k1 is None else k1
        self.ctx = ctx
        self.groups = groups
        self.tiles = tiles
        self._R: dict[tuple, torch.Tensor] = {}

    def R(self, cfg: dict) -> torch.Tensor:
        key = (cfg["BLOCK_M"], cfg["BLOCK_N"])
        R = self._R.get(key)
        if R is None:
            r, t = self.ranges, self.tiles
            R = self._R[key] = attn_kernel.block_ranges(
                r[:, 0], r[:, 1], r[:, 2], r[:, 3], *key, exact=True, groups=self.groups,
                tiles_from=None if t is None else (t[:, 0], t[:, 1], t[:, 2], t[:, 3]))
        return R


def window_ranges(r: torch.Tensor, qpos: torch.Tensor, window: int) -> torch.Tensor:
    """Key ranges of the sliding layers: each query's ranges cut to start no more than `window` positions back, so a
    query tile visits about window + tile keys instead of its whole prefix. The first range starts at the key of
    position 0 (a prefix); the second holds the query's own run of the stream (branch, or padding)."""
    if not window:
        return r
    r = r.clone()
    idx = torch.arange(r.shape[0], device=r.device, dtype=r.dtype)
    r[:, 0] = torch.maximum(r[:, 0], torch.minimum(r[:, 0] + qpos.to(r.dtype) - (window - 1), r[:, 1]))
    r[:, 2] = torch.maximum(r[:, 2], torch.minimum(idx - (window - 1), r[:, 3]))
    return r


class _Ctx:
    """Prefix keys and values of a packed batch whose prefixes resume from saved states or save their own (as qwen3_5's
    full attention reads them): per layer a context buffer holding every job's whole prefix in order, the saved keys
    before the resume point, the stream's after it. A save keeps a global layer's keys of the whole prefix up to the
    save point, a sliding layer's of the last `window` positions only (all a later query can see). Keys-shared layers
    read the context of the layer they share with."""

    def __init__(self, b: PackedBatch, window: int):
        self.b = b
        self.window = window
        self.shared: dict[str, tuple] = {}

    def keys(self, module, k: torch.Tensor, v: torch.Tensor):
        if module.is_kv_shared_layer:
            return self.shared[module.layer_type]
        b, C, Tp, i = self.b, self.b.ctx, self.b.n_prefix_tokens, module.layer_idx
        new = k.new_zeros if module.is_sliding and b.resumes else k.new_empty
        Kc, Vc = new((C["len"],) + k.shape[1:]), new((C["len"],) + v.shape[1:])
        Kc.index_copy_(0, C["index"], k[:Tp])
        Vc.index_copy_(0, C["index"], v[:Tp])
        for j, entry, start in b.resumes:
            K0, V0 = entry.kv[i]
            c = C["start"][j] + start
            Kc[c - K0.shape[0]: c].copy_(K0)
            Vc[c - V0.shape[0]: c].copy_(V0)
        for j, marks in b.marks.items():
            m = max(marks)
            c0 = C["start"][j]
            lo = max(0, m - self.window) if module.is_sliding else 0
            rec = b.saved.setdefault(j, {"states": {m: {}}, "kv": {}})
            rec["kv"][i] = (Kc[c0 + lo: c0 + m].clone(), Vc[c0 + lo: c0 + m].clone())
        if module.store_full_length_kv:
            self.shared[module.layer_type] = (Kc, Vc)
        return Kc, Vc


def _attention(module, query, key, value, attention_mask, scaling=None, sliding_window=None, **kwargs):
    """transformers attention interface: [1, H, T, D] in, [1, T, H, D] out."""
    meta = attention_mask
    if not hasattr(meta, "tok") or not hasattr(meta, "R"):
        raise RuntimeError("the flide attention runs only inside flide's packed forward or graphs")
    q = query[0].transpose(0, 1)
    k = key[0].transpose(0, 1).contiguous()
    v = value[0].transpose(0, 1).contiguous()
    cfg = attn_kernel.config(q.device, q.dtype, q.shape[-1])
    k1, v1 = (k, v) if meta.ctx is None else meta.ctx.keys(module, k, v)
    o = attn_kernel.attention(q.contiguous(), k1, v1, k, v, meta.tok, meta.k1, meta.tok, meta.R(cfg),
                              1.0 if scaling is None else scaling, int(sliding_window or 0), cfg)
    return o[None], None


def vision_runs(ids: np.ndarray, placeholders) -> list[tuple[int, int]]:
    """[start, end) of every run of placeholder tokens in `ids` (the tokens of one image or video frame)."""
    mask = np.isin(ids, list(placeholders)).astype(np.int8)
    if not mask.any():
        return []
    edge = np.diff(np.concatenate([[0], mask, [0]]))
    return list(zip(np.flatnonzero(edge == 1).tolist(), np.flatnonzero(edge == -1).tolist()))


def _register() -> None:
    from transformers import AttentionInterface

    AttentionInterface.register(IMPL, _attention)


class Gemma4(Backbone):
    model_types = ("gemma4", "gemma4_text", "gemma4_unified")

    @classmethod
    def load(cls, model_dir: str, device: str = "cuda:0", dtype: str = "bfloat16", vision: bool | None = None,
             loader: str = "direct", **options) -> "Gemma4":
        """Checkpoints with a vision tower load it (`vision` None) unless `vision` is False; then only the decoder
        loads (vision and audio towers stay on disk).
        `loader` "direct" reads the safetensors straight onto the device (flide.backbones.load); "hf" uses
        transformers' from_pretrained (needs accelerate); "none" reads no weights (zeros)."""
        import transformers

        _register()
        config = transformers.AutoConfig.from_pretrained(model_dir, local_files_only=True)
        if vision is None:
            vision = getattr(config, "vision_config", None) is not None
        if config.model_type == "gemma4_text":
            arch, cfg, key_map = transformers.Gemma4ForCausalLM, config, []
        elif vision:
            arch = (transformers.Gemma4UnifiedForConditionalGeneration if config.model_type == "gemma4_unified"
                    else transformers.Gemma4ForConditionalGeneration)
            cfg, key_map = config, ([(r"^model\.vision_embedder\.", "model.embed_vision."),
                                     (r"^model\.embed_vision\.embedding_projection",
                                      "model.embed_vision.multimodal_embedder.embedding_projection")]
                                    if config.model_type == "gemma4_unified" else [])
        else:
            arch, cfg = transformers.Gemma4ForCausalLM, config.get_text_config()
            key_map = [(r"^model\.language_model\.", "model.")]
        cfg._attn_implementation = IMPL
        model = None
        if loader == "none":
            from ..load import blank, build

            model = build(arch, cfg, torch.device(device), getattr(torch, dtype))
            blank(model)
        elif loader == "direct":
            from ..load import build, load_into

            try:
                model = build(arch, cfg, torch.device(device), getattr(torch, dtype))
                load_into(model, model_dir, device, key_map=key_map)
            except (ValueError, FileNotFoundError) as exc:
                import logging

                logging.getLogger("flide.load").warning("direct load failed (%s); using transformers", exc)
                model = None
                torch.cuda.empty_cache()
        if model is None:
            if config.model_type == "gemma4_text":
                arch = transformers.Gemma4ForCausalLM
            elif config.model_type == "gemma4_unified":
                arch = transformers.Gemma4UnifiedForConditionalGeneration
            else:
                arch = transformers.Gemma4ForConditionalGeneration
            model = arch.from_pretrained(model_dir, config=config, dtype=getattr(torch, dtype), device_map={"": str(device)},
                                         low_cpu_mem_usage=True, local_files_only=True, attn_implementation=IMPL)
        if vision:
            for sub in ("vision_config", "audio_config"):
                c = getattr(model.config, sub, None)
                if c is not None:
                    c._attn_implementation = "sdpa"
        model.config.get_text_config()._attn_implementation = IMPL
        model.eval()
        return cls(model, **options)

    def __init__(self, model, cuda_graphs: bool = False, quantize: str | None = None, tp=None, **_):
        if quantize is not None:
            raise NotImplementedError("gemma4: FP8 is not supported yet")
        if tp is not None and tp[1] > 1:
            raise NotImplementedError("gemma4: tensor parallel is not supported yet")
        _register()
        self.model = model
        inner = model.model
        self.lm = getattr(inner, "language_model", inner)
        self.config = self.lm.config
        self.config._attn_implementation = IMPL
        self.num_layers = self.config.num_hidden_layers
        p0 = next(model.parameters())
        self.device, self.dtype = p0.device, p0.dtype
        self.logit_softcap = getattr(self.config, "final_logit_softcapping", None)
        self.cuda_graphs = cuda_graphs
        top = model.config
        self.vlm = model if hasattr(inner, "get_image_features") else None
        self.placeholders = {i for i in (getattr(top, k, None) for k in ("image_token_id", "video_token_id",
                                                                          "audio_token_id")) if i is not None}
        self.pad_id = self.config.pad_token_id
        self.bidir = getattr(self.config, "use_bidirectional_attention", None) == "vision"
        self.moe = bool(getattr(self.config, "enable_moe_block", False))
        self._experts()
        self._fused = None
        self.decoder()

    def decoder(self):
        """The fused decoder (fused.py) at the weights' current dtype; None for transformers' modules (mixture of
        experts)."""
        from .fused import FusedDecoder

        if self.moe:
            return None
        if self._fused is None or self._fused.dtype != self.lm.layers[0].mlp.gate_proj.weight.dtype:
            self._fused = FusedDecoder(self.lm)
        return self._fused

    def _experts(self) -> str | None:
        """The experts implementation for the current dtype (set on the config the expert modules read)."""
        if not self.moe:
            return None
        want = "grouped_mm" if self.dtype in (torch.bfloat16, torch.float16) else "eager"
        if self.config._experts_implementation != want:
            self.config._experts_implementation = want
        return want

    def vision_positions(self, ids, vision):
        """Gemma 4 places every token at its index (no multimodal rotary positions)."""
        return None, 0

    def vision_graphs(self):
        """The vision tower from CUDA graphs (vision.py), built on first use; None without CUDA graphs."""
        if self.vlm is None or not self.cuda_graphs or self.device.type != "cuda":
            return None
        if getattr(self, "_vision_graphs", None) is None:
            from .vision import GemmaVisionGraphs

            self._vision_graphs = GemmaVisionGraphs(self.vlm)
        return self._vision_graphs

    def embed(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(ids with media placeholders as padding, their embeddings): what the decoder and the per-layer inputs read."""
        mm = torch.isin(ids, torch.tensor(sorted(self.placeholders), device=ids.device))
        llm = torch.where(mm, self.pad_id, ids)
        return llm, self.lm.embed_tokens(llm)

    def encode_vision(self, ids: torch.Tensor, h: torch.Tensor, visions: list, graphs=None) -> torch.Tensor:
        """Embeddings with the vision tower's features in the image and video placeholders; `visions` are the requests'
        processor outputs in the order their tokens appear."""
        inner = self.vlm.model
        top = self.vlm.config
        for pix, pos, token, fn in (("pixel_values", "image_position_ids", top.image_token_id, inner.get_image_features),
                                    ("pixel_values_videos", "video_position_ids", getattr(top, "video_token_id", None),
                                     getattr(inner, "get_video_features", None))):
            items = [v for v in visions if v.get(pix) is not None]
            if not items:
                continue
            feats = []
            for v in items:
                px = v[pix].to(self.device)
                ps = v[pos].to(self.device) if v.get(pos) is not None else None
                if graphs is not None and pix == "pixel_values" and ps is not None:
                    feats.append(graphs(px, ps))
                else:
                    feats.extend(fn(px, ps, return_dict=True).pooler_output)
            feats = torch.cat(feats).to(h.dtype)
            mask = ids == token
            if int(mask.sum()) != feats.shape[0]:
                raise ValueError(f"{int(mask.sum())} placeholder tokens for {feats.shape[0]} vision features")
            h = h.masked_scatter(mask[:, None], feats)
        return h

    def attention_meta(self, tok, ranges, ids: np.ndarray, batch: PackedBatch | None = None) -> dict:
        """transformers' mask mapping for the packed metadata: the global layers causal, the sliding layers cut to
        their window and also letting the tokens of one image see each other on the sizes trained so (their key
        ranges are widened to the end of the image in place). A batch with sessions reads prefix keys from a context
        buffer."""
        ctx = k1 = None
        if batch is not None and batch.ctx is not None:
            ctx, k1 = _Ctx(batch, self.config.sliding_window or 0), batch.ctx["meta"]
        sl_tok, full_ranges = tok, ranges
        if self.bidir:
            runs = vision_runs(ids, self.placeholders)
            if runs:
                blk = np.full(ids.shape[0], -1, np.int32)
                r = ranges.cpu().numpy().copy()
                for k, (a, e) in enumerate(runs):
                    blk[a:e] = k
                    r[a:e, 1] = np.maximum(r[a:e, 1], e)
                ranges = torch.from_numpy(r).to(ranges.device)
                sl_tok = tuple(tok) + (torch.from_numpy(blk).to(ranges.device),)
        sl_ranges = window_ranges(ranges, tok[2], self.config.sliding_window or 0)
        groups = batch.attn_runs if batch is not None and batch.attn_runs else attn_kernel.runs(
            full_ranges.cpu().numpy(), by_row=True)
        return {"full_attention": _Meta(tok, full_ranges, k1, ctx, groups),
                "sliding_attention": _Meta(sl_tok, sl_ranges, k1, ctx, groups, tiles=full_ranges)}

    def request_graphs(self):
        """CUDA graphs for one request at a time (graphs.py), built on first use; None without CUDA graphs."""
        if not self.cuda_graphs or self.device.type != "cuda" or self._experts() == "eager":
            return None
        if getattr(self, "_graphs", None) is None:
            from .graphs import GemmaGraphs

            self._graphs = GemmaGraphs(self)
        return self._graphs

    def output_head(self):
        head = self.model.get_output_embeddings()
        return head.weight, getattr(head, "bias", None)

    @torch.inference_mode()
    def forward(self, b: PackedBatch, pad_to: int | None = None) -> dict:
        if b.vision and self.vlm is None:
            raise ValueError("this checkpoint was loaded without its vision tower")
        self._experts()
        ints = [k for k in b.reads if k is not FINAL]
        need_hidden = bool(ints)
        tok = (b.job.int(), b.branch.int(), b.positions.int())
        ids = b.input_ids
        masks = self.attention_meta(tok, b.key_ranges, ids.cpu().numpy(), b)
        llm, emb = self.embed(ids)
        if b.vision:
            emb = self.encode_vision(ids, emb, b.vision)
        dec = self.decoder()
        if dec is not None:
            top = self.num_layers if FINAL in b.reads else max(ints, default=0)
            rows = (b.reads[FINAL] if not ints and FINAL in b.reads and masks["full_attention"].ctx is None
                    else None)
            h, res = dec(emb, llm, b.positions, masks, top=top, reads={k: b.reads[k] for k in ints}, rows=rows)
            if FINAL in b.reads:
                res[FINAL] = dec.final(h if rows is not None else h[b.reads[FINAL]])
            return res
        pli = self.lm.get_per_layer_inputs(llm[None], None) if self.config.hidden_size_per_layer_input else None
        out = self.lm(inputs_embeds=emb[None], per_layer_inputs=pli, position_ids=b.positions[None],
                      attention_mask=masks, use_cache=False, output_hidden_states=need_hidden, return_dict=True)
        res = {}
        if need_hidden:
            hs = out.hidden_states
            for k in ints:
                res[k] = hs[k][0, b.reads[k]]
        if FINAL in b.reads:
            res[FINAL] = out.last_hidden_state[0, b.reads[FINAL]]
        return res

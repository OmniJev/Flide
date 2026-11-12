"""Packed forward for LFM2 decoders (lfm2, and the language model of lfm2_vl: LFM2-VL, LiquidAI's d1-3B).

An LFM2 layer is a gated short convolution or full attention (fused.py). Neither keeps a recurrent state, so a branch
needs only the keys and values of its prefix and the prefix's last few convolution inputs, both read straight from
the stream: the attention is flide's Triton kernel over the packed metadata ("same job, key in the prefix or in
the query's own branch, key not after the query"), the convolution gathers each token's earlier inputs.

The output head is the tied embedding. Images, as transformers' Lfm2VlModel does it: the vision tower's features,
pixel-unshuffled and projected, replace the image placeholder tokens; positions stay one per token.
"""
from __future__ import annotations

import torch

from ... import lora
from ...formats.base import FINAL
from ...kernels import attention as attn_kernel
from ...runtime.pack import PackedBatch
from ..base import Backbone


class _Meta:
    """Packed attention metadata: (job, branch, position) per token and the key ranges, with block ranges per query
    tile size."""

    def __init__(self, tok, ranges, groups: int = 0):
        self.tok = tok
        self.k1 = tok
        self.ranges = ranges
        self.groups = groups
        self._R: dict[tuple, torch.Tensor] = {}

    def R(self, cfg: dict) -> torch.Tensor:
        key = (cfg["BLOCK_M"], cfg["BLOCK_N"])
        R = self._R.get(key)
        if R is None:
            r = self.ranges
            R = self._R[key] = attn_kernel.block_ranges(r[:, 0], r[:, 1], r[:, 2], r[:, 3], *key, exact=True,
                                                         groups=self.groups)
        return R


class Lfm2(Backbone):
    model_types = ("lfm2", "lfm2_vl")
    session_states = False   # not yet: a resumed prefix needs its saved convolution inputs

    @classmethod
    def load(cls, model_dir: str, device: str = "cuda:0", dtype: str = "bfloat16", vision: bool | None = None,
             loader: str = "direct", merge_lora: bool = False, **options) -> "Lfm2":
        """Checkpoints with a vision tower load it (`vision` None) unless `vision` is False; then only the language
        model loads. `loader` "direct" reads the safetensors straight onto the device (flide.backbones.load), "hf"
        uses transformers' from_pretrained, "none" reads no weights (zeros)."""
        import transformers

        config = transformers.AutoConfig.from_pretrained(model_dir, local_files_only=True)
        if vision is None:
            vision = getattr(config, "vision_config", None) is not None
        if config.model_type == "lfm2":
            arch, cfg, key_map = transformers.Lfm2ForCausalLM, config, []
        elif vision:   # checkpoints keep the tower's SigLIP2 names under vision_model, which transformers drops
            arch, cfg = transformers.Lfm2VlForConditionalGeneration, config
            key_map = [(r"^model\.vision_tower\.vision_model\.", "model.vision_tower.")]
        else:
            arch, cfg = transformers.Lfm2ForCausalLM, config.get_text_config()
            key_map = [(r"^model\.language_model\.", "model.")]
        model = None
        if loader == "none":
            from ..load import blank, build

            model = build(arch, cfg, torch.device(device), getattr(torch, dtype))
            blank(model)
        elif loader == "direct":
            from ..load import build, load_into

            try:
                model = build(arch, cfg, torch.device(device), getattr(torch, dtype))
                load_into(model, model_dir, device, key_map=key_map, merge=merge_lora)
            except (ValueError, FileNotFoundError) as exc:
                import logging

                logging.getLogger("flide.load").warning("direct load failed (%s); using transformers", exc)
                model = None
                torch.cuda.empty_cache()
        if model is None:
            arch = transformers.Lfm2ForCausalLM if config.model_type == "lfm2" else transformers.Lfm2VlForConditionalGeneration
            model = arch.from_pretrained(model_dir, config=config, dtype=getattr(torch, dtype),
                                         device_map={"": str(device)}, low_cpu_mem_usage=True, local_files_only=True)
        for sub in ("vision_config",):   # the tower keeps transformers' own attention
            c = getattr(model.config, sub, None)
            if c is not None:
                c._attn_implementation = "sdpa"
        model.eval()
        return cls(model, **options)

    def __init__(self, model, cuda_graphs: bool = False, quantize: str | None = None, tp=None, **_):
        if quantize is not None:
            raise NotImplementedError("lfm2: --quantize is not supported yet")
        if tp is not None and tp[1] > 1:
            raise NotImplementedError("lfm2: tensor parallel is not supported yet")
        self.model = model
        inner = model.model
        self.lm = getattr(inner, "language_model", inner)
        self.config = self.lm.config
        self.num_layers = self.config.num_hidden_layers
        p0 = next(model.parameters())
        self.device, self.dtype = p0.device, p0.dtype
        self.cuda_graphs = cuda_graphs
        self.vlm = model if hasattr(inner, "get_image_features") else None
        self.image_token = getattr(model.config, "image_token_id", None)
        self._fused = None
        self.decoder()

    def decoder(self):
        from .fused import FusedDecoder

        if self._fused is None or self._fused.dtype != self.lm.layers[0].feed_forward.w1.weight.dtype:
            self._fused = FusedDecoder(self.lm)
        return self._fused

    def vision_positions(self, ids, vision):
        """LFM2-VL places every token at its index (no multimodal rotary positions)."""
        return None, 0

    def encode_vision(self, ids: torch.Tensor, h: torch.Tensor, visions: list) -> torch.Tensor:
        """Embeddings with the image features in the placeholders; `visions` are the requests' processor outputs
        (pixel_values, spatial_shapes, pixel_attention_mask) in the order their tokens appear."""
        inner = self.vlm.model
        feats = []
        for v in visions:
            px = v["pixel_values"].to(self.device, self.dtype)
            shapes = v["spatial_shapes"].to(self.device)
            mask = v["pixel_attention_mask"].to(self.device)
            out = inner.get_image_features(pixel_values=px, spatial_shapes=shapes, pixel_attention_mask=mask)
            out = getattr(out, "pooler_output", out)
            feats.extend(out if isinstance(out, (list, tuple)) else [out])
        feats = torch.cat([f.reshape(-1, h.shape[-1]) for f in feats]).to(h.dtype)
        mask = ids == self.image_token
        if int(mask.sum()) != feats.shape[0]:
            raise ValueError(f"{int(mask.sum())} image tokens for {feats.shape[0]} image features")
        return h.masked_scatter(mask[:, None], feats)

    def request_graphs(self):
        """CUDA graphs for rounds of requests (graphs.py), built on first use; None without CUDA graphs."""
        if not self.cuda_graphs or self.device.type != "cuda":
            return None
        if getattr(self, "_graphs", None) is None:
            from .graphs import Lfm2Graphs

            self._graphs = Lfm2Graphs(self)
        return self._graphs

    def output_head(self):
        return lora.output_head(self.model.get_output_embeddings())

    @torch.inference_mode()
    def forward(self, b: PackedBatch, pad_to: int | None = None) -> dict:
        from .fused import conv_prev

        if b.ctx is not None or b.resumes or b.marks:
            raise NotImplementedError("lfm2: the session cache is not supported yet")
        if any(j.blocks or j.prefix_two_way for j in b.jobs):
            raise NotImplementedError("lfm2: only causal attention")
        if b.vision and self.vlm is None:
            raise ValueError("this checkpoint was loaded without its vision tower")
        ints = [k for k in b.reads if k is not FINAL]
        tok = (b.job.int(), b.branch.int(), b.positions.int())
        meta = _Meta(tok, b.key_ranges, b.attn_runs or attn_kernel.runs(b.key_ranges.cpu().numpy(), by_row=True))
        ids = b.input_ids
        emb = self.lm.embed_tokens(ids)
        if b.vision:
            emb = self.encode_vision(ids, emb, b.vision)
        dec = self.decoder()
        T = ids.shape[0]
        prev = conv_prev(T, b.n_prefix_tokens, b.cu_prefix, b.cu_suffix, b.branch, b.branch_job, dec.keep)
        top = self.num_layers if FINAL in b.reads else max(ints, default=0)
        h, res = dec(emb, b.positions, meta, prev, top=top, reads={k: b.reads[k] for k in ints})
        if FINAL in b.reads:
            res[FINAL] = dec.final(h[b.reads[FINAL]])
        return res

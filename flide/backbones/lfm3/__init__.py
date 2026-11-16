"""Packed forward for LFM3 decoders (lfm3, and the language model of lfm3_vl: LiquidAI's d1-2B), text only for now.

An LFM3 layer is a gated short convolution, full attention or sliding-window attention (fused.py). None keeps a
recurrent state: the attention is flide's Triton kernel over the packed metadata, the convolution gathers each
token's earlier inputs. Besides causal plans it runs attention "set" (formats.base.Plan), the mask d1-2B is trained
with: in the full-attention layers a state reads its questions and each question the state and itself.

LFM3 is not in transformers (its checkpoints ship it as code of their own, which flide does not run): the weights
load into model.py's plain modules, and the config is read from config.json. The output head is the tied embedding.
"""
from __future__ import annotations

import torch

from ... import lora
from ...formats.base import FINAL
from ...kernels import attention as attn_kernel
from ...runtime.pack import PackedBatch
from ..base import Backbone
from ..gemma4 import _Meta, window_ranges


class Lfm3(Backbone):
    model_types = ("lfm3", "lfm3_vl")
    session_states = False   # a resumed prefix would need its saved convolution inputs
    set_attention = True

    @classmethod
    def load(cls, model_dir: str, device: str = "cuda:0", dtype: str = "bfloat16", vision: bool | None = None,
             loader: str = "direct", merge_lora: bool = False, **options) -> "Lfm3":
        """The language model alone (an lfm3_vl checkpoint's vision tower is not loaded yet). `loader` "direct" reads
        the safetensors straight onto the device (flide.backbones.load), "none" reads no weights (zeros)."""
        from ..load import blank, build, load_into
        from .model import Lfm3ForCausalLM, text_config

        if vision:
            raise NotImplementedError("lfm3: images are not supported yet")
        cfg = text_config(model_dir)
        model = build(Lfm3ForCausalLM, cfg, torch.device(device), getattr(torch, dtype))
        if loader == "none":
            blank(model)
        elif loader == "direct":
            load_into(model, model_dir, device, key_map=[(r"^model\.language_model\.", "model.")],
                      skip=("model.vision_tower", "model.multi_modal_projector"), merge=merge_lora)
        else:
            raise ValueError(f"lfm3: loader {loader!r}; it reads checkpoints with the direct loader only")
        model.eval()
        return cls(model, **options)

    def __init__(self, model, cuda_graphs: bool = False, quantize: str | None = None, tp=None, **_):
        if quantize is not None:
            raise NotImplementedError("lfm3: --quantize is not supported yet")
        if tp is not None and tp[1] > 1:
            raise NotImplementedError("lfm3: tensor parallel is not supported yet")
        self.model = model
        self.lm = model.model
        self.config = model.config
        self.num_layers = self.config.num_hidden_layers
        p0 = next(model.parameters())
        self.device, self.dtype = p0.device, p0.dtype
        self.cuda_graphs = cuda_graphs
        self.vlm = None   # text only for now
        self._fused = None
        self.decoder()

    def decoder(self):
        from .fused import FusedDecoder

        if self._fused is None or self._fused.dtype != self.lm.layers[0].feed_forward.w1.weight.dtype:
            self._fused = FusedDecoder(self.lm)
        return self._fused

    def request_graphs(self):
        """CUDA graphs for rounds of requests and for "set" plans (graphs.py), built on first use; None without CUDA
        graphs."""
        if not self.cuda_graphs or self.device.type != "cuda":
            return None
        if getattr(self, "_graphs", None) is None:
            from .graphs import Lfm3Graphs

            self._graphs = Lfm3Graphs(self)
        return self._graphs

    def output_head(self):
        return lora.output_head(self.model.get_output_embeddings())

    @torch.inference_mode()
    def forward(self, b: PackedBatch, pad_to: int | None = None) -> dict:
        from ..lfm2.fused import conv_prev

        if b.ctx is not None or b.resumes or b.marks:
            raise NotImplementedError("lfm3: the session cache is not supported yet")
        if any(j.blocks or j.prefix_two_way for j in b.jobs):
            raise NotImplementedError("lfm3: causal and set attention only")
        if b.vision:
            raise NotImplementedError("lfm3: images are not supported yet")
        ints = [k for k in b.reads if k is not FINAL]
        tok = (b.job.int(), b.branch.int(), b.positions.int())
        groups = b.attn_runs or attn_kernel.runs(b.key_ranges.cpu().numpy(), by_row=True)
        meta = _Meta(tok, b.key_ranges, groups=groups)
        # the sliding layers' ranges cut to their window (the kernel reads whole key blocks inside exact ranges
        # unmasked), their tiles those of the uncut ranges
        sl_meta = _Meta(tok, window_ranges(b.key_ranges, tok[2], int(self.config.sliding_window or 0)), groups=groups,
                        tiles=b.key_ranges)
        set_meta = None
        if b.set_ranges is not None:
            set_meta = _Meta(tok, b.set_ranges, groups=attn_kernel.runs(b.set_ranges.cpu().numpy(), by_row=True))
        ids = b.input_ids
        emb = self.lm.embed_tokens(ids)
        dec = self.decoder()
        T = ids.shape[0]
        prev = conv_prev(T, b.n_prefix_tokens, b.cu_prefix, b.cu_suffix, b.branch, b.branch_job, dec.keep)
        top = self.num_layers if FINAL in b.reads else max(ints, default=0)
        h, res = dec(emb, b.positions, meta, prev, top=top, reads={k: b.reads[k] for k in ints}, set_meta=set_meta,
                     sl_meta=sl_meta)
        if FINAL in b.reads:
            res[FINAL] = dec.final(h[b.reads[FINAL]])
        return res

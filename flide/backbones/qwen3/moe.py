"""Qwen3 expert routing on the packed attention runtime."""
from __future__ import annotations

import torch

from ... import lora
from ...kernels.fp8 import mm
from ...kernels import pointwise as pw
from ...kernels import moe as moe_kernel
from ...kernels.moe import experts
from . import Qwen3


class Qwen3Moe(Qwen3):
    model_types = text_types = ("qwen3_moe",)

    @classmethod
    def load(cls, *args, tp=None, **kwargs):
        if tp is not None and tp[1] > 1:
            raise ValueError("qwen3_moe: tensor parallel is not supported")
        return super().load(*args, tp=tp, **kwargs)

    @staticmethod
    def _architecture(config):
        from transformers import Qwen3MoeForCausalLM, Qwen3MoeModel

        return Qwen3MoeModel if "Qwen3MoeModel" in (config.architectures or ()) else Qwen3MoeForCausalLM

    def _mlp_weights(self, mlp, tensor_parallel):
        if not hasattr(mlp, "experts"):
            return super()._mlp_weights(mlp, tensor_parallel)
        return {"experts": mlp.experts.gate_up_proj, "experts_down": mlp.experts.down_proj,
                "router": lora.attached(mlp.gate.weight, [mlp.gate]), "inter": mlp.experts.intermediate_dim,
                "topk": mlp.gate.top_k, "normalize": mlp.gate.norm_topk_prob}

    def _ffn(self, x, weights):
        if isinstance(weights, tuple):
            return super()._ffn(x, weights)
        if weights["normalize"]:   # in float32 and through the shared kernels, as the other mixtures route
            logits, _ = moe_kernel.router(x, weights["router"])
            top, ids = moe_kernel.route(logits, weights["topk"])
        else:
            probabilities = mm(x, weights["router"]).softmax(dim=-1, dtype=torch.float32)
            top, ids = probabilities.topk(weights["topk"], dim=-1)
        y = experts(x, weights, top.to(x.dtype), ids, pw.swiglu)
        return y.reshape(x.shape[0], weights["topk"], x.shape[-1]).sum(dim=1).to(x.dtype)

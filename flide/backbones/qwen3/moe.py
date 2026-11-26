"""Qwen3 expert routing on the packed attention runtime."""
from __future__ import annotations

import torch

from ... import lora
from ...kernels.fp8 import mm
from ...kernels.moe import experts
from ..qwen2.ops import swiglu
from . import Qwen3


class Qwen3Moe(Qwen3):
    model_types = ("qwen3_moe",)

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
        probabilities = mm(x, weights["router"]).softmax(dim=-1, dtype=torch.float32)
        top, ids = probabilities.topk(weights["topk"], dim=-1)
        if weights["normalize"]:
            top = top / top.sum(dim=-1, keepdim=True)
        y = experts(x, weights, top.to(x.dtype), ids, lambda gate_up, _: swiglu(gate_up))
        return y.reshape(x.shape[0], weights["topk"], x.shape[-1]).sum(dim=1).to(x.dtype)

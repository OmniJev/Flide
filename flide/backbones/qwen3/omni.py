"""The text side of Qwen3-Omni MoE checkpoints: the thinker's decoder and output head, a Qwen3 MoE whose multimodal
rotary embedding is the plain one on text (a token's three positions are the same). Images, audio and video are not
read."""
from __future__ import annotations

from .moe import Qwen3Moe


class Qwen3OmniMoe(Qwen3Moe):
    model_types = ("qwen3_omni_moe",)
    key_map = [(r"^thinker\.(model|lm_head)\.", r"\1.")]

    @staticmethod
    def _config(model_dir: str):
        from transformers import AutoConfig, Qwen3MoeConfig

        text = AutoConfig.from_pretrained(model_dir, local_files_only=True).thinker_config.text_config
        fields = {k: v for k, v in text.to_dict().items() if k not in ("model_type", "rope_parameters", "rope_scaling")}
        config = Qwen3MoeConfig(**fields, rope_parameters={"rope_type": "default",
                                                           "rope_theta": text.rope_parameters["rope_theta"]})
        config.architectures = ["Qwen3MoeForCausalLM"]
        return config

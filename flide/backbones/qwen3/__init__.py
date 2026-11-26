"""Qwen3 text checkpoints on the shared full-attention Qwen runtime."""
from ..qwen2 import Qwen2


class Qwen3(Qwen2):
    model_types = text_types = ("qwen3",)

    @staticmethod
    def _architecture(config):
        from transformers import Qwen3ForCausalLM, Qwen3Model

        return Qwen3Model if "Qwen3Model" in (config.architectures or ()) else Qwen3ForCausalLM

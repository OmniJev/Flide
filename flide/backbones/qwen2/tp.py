"""Load dense Qwen attention and MLP projections by rank, including any Q/K/V biases."""
from __future__ import annotations

import torch

from ...runtime.tp import split
from ..load import build, load_into, part


def read_shard(arch, config, model_dir, device, dtype, tp, merge_lora):
    rank, size = tp
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    q_heads = split(config.num_attention_heads, rank, size)
    kv_heads = split(config.num_key_value_heads, rank, size)
    q_rows = slice(q_heads.start * head_dim, q_heads.stop * head_dim)
    kv_rows = slice(kv_heads.start * head_dim, kv_heads.stop * head_dim)
    mlp_rows = split(config.intermediate_size, rank, size)
    cuts = {"self_attn.o_proj.weight": (1, q_rows), "mlp.down_proj.weight": (1, mlp_rows),
            "mlp.gate_proj.weight": (0, mlp_rows), "mlp.up_proj.weight": (0, mlp_rows)}
    for name, rows in (("q_proj", q_rows), ("k_proj", kv_rows), ("v_proj", kv_rows)):
        cuts[f"self_attn.{name}.weight"] = cuts[f"self_attn.{name}.bias"] = (0, rows)

    def layer_path(name):
        return name.partition("layers.")[2].partition(".")[2]

    def cut(name):
        return (*cuts[layer_path(name)], False)

    def sharded(name):
        return layer_path(name) in cuts

    model = build(arch, config, device, dtype, lean=sharded)
    held = {}
    load_into(model, model_dir, device, lean=sharded, cut=cut, held=held, merge=merge_lora)
    for name, tensor in held.items():
        module_name, field = name.rsplit(".", 1)
        module = model.get_submodule(module_name)
        setattr(module, field, torch.nn.Parameter(tensor, requires_grad=False))
        if field == "weight":
            module.out_features, module.in_features = tensor.shape
            lr = getattr(module, "lora", None)
            if lr is not None:
                a, b = lr
                dim, rows = cuts[layer_path(name)]
                module.lora = (a, part(b, rows)) if dim == 0 else (part(a, rows, dim=1), b)
    return model

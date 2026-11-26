"""Native dense and MoE Qwen3 reads against Transformers, including legacy expert shards."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from test_qwen2 import check_graphs  # also sets the float32 Triton precision for the comparisons

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3Model, Qwen3MoeConfig, Qwen3MoeForCausalLM

from flide.backbones import get
from flide.formats.base import FINAL, Read
from flide.runtime.pack import Job, pack


def save_legacy(model, path):
    """The original Qwen3 MoE releases store separate gate, up and down projections for each expert."""
    tensors, expert_tensors = {}, {}
    for name, weight in model.state_dict().items():
        prefix, sep, projection = name.rpartition(".experts.")
        if not sep:
            tensors[name] = weight.contiguous().clone()
            continue
        for i, tensor in enumerate(weight):
            parts = {"down_proj": tensor} if projection == "down_proj" else dict(zip(
                ("gate_proj", "up_proj"), tensor.chunk(2), strict=True))
            for part, rows in parts.items():
                expert_tensors[f"{prefix}.experts.{i}.{part}.weight"] = rows.contiguous().clone()
    weights_to_file = {}
    for i, shard in enumerate((tensors, expert_tensors), 1):
        name = f"model-{i:05d}-of-00002.safetensors"
        save_file(shard, path / name)
        weights_to_file.update(dict.fromkeys(shard, name))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights_to_file}))
    model.config.architectures = [type(model).__name__]
    model.config.save_pretrained(path)


def compare(reference, device, *, legacy=False):
    decoder = getattr(reference, "model", reference)
    for layer in decoder.layers:
        layer.self_attn.q_norm.weight.uniform_(0.7, 1.3)
        layer.self_attn.k_norm.weight.uniform_(0.7, 1.3)
        for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj, layer.self_attn.o_proj):
            if proj.bias is not None:
                proj.bias.normal_(std=0.05)
        if hasattr(layer.mlp, "experts"):
            layer.mlp.gate.weight.normal_(std=0.05)

    with TemporaryDirectory() as path:
        if legacy:
            save_legacy(reference, Path(path))
        else:
            reference.save_pretrained(path)
        backbone = get(reference.config.model_type).load(path, dtype="float32")

    reference.to(device)
    head = reference.get_output_embeddings()
    if head is not None:
        weight, bias = backbone.output_head()
        assert bias is None
        torch.testing.assert_close(weight, head.weight, atol=0, rtol=0)
    positions = (-1, 0, -2)
    reads = (Read(None, FINAL, positions), Read(None, 0, positions))
    jobs = [Job([13, 4, 9], [[4, 6, 8], [7, 5, 12]], [reads] * 2),
            Job(list(range(1, 79)), [[4, 9, 7], [33, 21, 28]], [reads] * 2, blocks=True),
            Job([1, 4, 5], [[38, 19, 7]], [reads], prefix_two_way=True)]
    batch = pack(jobs, device)
    expected = []
    for job in jobs:
        for suffix in job.suffixes:
            ids = torch.tensor([job.prefix + suffix], device=device)
            length = ids.shape[-1]
            prefix = len(job.prefix)
            allowed = torch.ones((length, length), dtype=torch.bool, device=device).tril()
            if job.blocks or job.prefix_two_way:
                allowed[:prefix, :prefix] = True
            if job.blocks:
                allowed[prefix:, prefix:] = True
            mask = allowed[None, None]
            if reference.config.model_type == "qwen3":
                mask = {"full_attention": mask}
            output = decoder(ids, attention_mask=mask, use_cache=False)
            expected.append(output.last_hidden_state[0, list(positions)])
    actual = backbone.forward(batch)[FINAL]
    torch.testing.assert_close(actual, torch.cat(expected), atol=5e-5, rtol=5e-5)
    batch.reads[backbone.num_layers] = batch.reads[FINAL]
    torch.testing.assert_close(backbone.forward(batch)[FINAL], actual, atol=5e-5, rtol=5e-5)
    check_graphs(backbone, device)


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    common = dict(vocab_size=128, hidden_size=256, intermediate_size=384, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=128, attn_implementation="sdpa")
    compare(Qwen3ForCausalLM(Qwen3Config(**common, tie_word_embeddings=True)).eval(), device)
    compare(Qwen3Model(Qwen3Config(**common, attention_bias=True)).eval(), device)
    for legacy in (False, True):
        config = Qwen3MoeConfig(**common, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=128,
                                norm_topk_prob=legacy, mlp_only_layers=[0])
        compare(Qwen3MoeForCausalLM(config).eval(), device, legacy=legacy)
    print("PASS Qwen3 dense, headless and MoE packed reads, CUDA graphs, and cached KV")


if __name__ == "__main__":
    main()

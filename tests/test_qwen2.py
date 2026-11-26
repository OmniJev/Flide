"""Native Qwen2 loading and packed reads against independent Transformers forwards on CUDA."""
from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3MoeConfig

from flide import settings
from flide.backbones import get
from flide.backbones.qwen2.graphs import Qwen2Graphs
from flide.formats.base import FINAL, Plan, Read
from flide.runtime.engine import Engine
from flide.runtime.pack import Job, pack
from flide.runtime.session import Entry, SessionCache


def check_rope_validation():
    modes = {
        "dynamic": {"factor": 2.0},
        "longrope": {"factor": 2.0, "original_max_position_embeddings": 64,
                     "short_factor": [1.0] * 16, "long_factor": [2.0] * 16},
    }
    for config_class in (Qwen2Config, Qwen3Config, Qwen3MoeConfig):
        for rope_type, options in modes.items():
            config = config_class(hidden_size=128, num_attention_heads=4, head_dim=32,
                                  max_position_embeddings=128, rope_parameters={"rope_type": rope_type, **options})
            with TemporaryDirectory() as path:
                config.save_pretrained(path)
                with TestCase().assertRaisesRegex(NotImplementedError, rf"length-dependent RoPE \({rope_type}\)"):
                    get(config.model_type).load(path)

    config = Qwen3Config(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                         rope_parameters={"rope_type": "linear", "factor": 2.0, "partial_rotary_factor": 0.5})
    with TemporaryDirectory() as path:
        config.save_pretrained(path)
        with TestCase().assertRaisesRegex(NotImplementedError, "RoPE frequencies must cover the full attention head"):
            get(config.model_type).load(path, loader="none")


def check_checkpoint_validation(reference):
    with TemporaryDirectory() as path:
        reference.save_pretrained(path)
        loader = get("qwen2")
        loaded = loader.load(path, dtype="float32", loader="hf")
        torch.testing.assert_close(loaded.output_head()[0], reference.lm_head.weight, atol=0, rtol=0)

        shard = Path(path, "model.safetensors")
        tensors = load_file(shard)
        missing = "model.layers.0.self_attn.q_proj.weight"
        weight = tensors.pop(missing)
        save_file(tensors, shard)
        for mode in ("direct", "hf"):
            with TestCase().assertRaisesRegex(ValueError, missing):
                loader.load(path, dtype="float32", loader=mode)

        # The default loader may fall back for legacy PyTorch checkpoints, which still require every weight.
        shard.unlink()
        legacy = Path(path, "pytorch_model.bin")
        torch.save(tensors, legacy)
        with TestCase().assertRaisesRegex(ValueError, missing):
            loader.load(path, dtype="float32")
        torch.save({**tensors, missing: weight}, legacy)
        loaded = loader.load(path, dtype="float32")
        torch.testing.assert_close(loaded.lm.layers[0].self_attn.q_proj.weight, weight.cuda(), atol=0, rtol=0)

        adapter = Path(path, "lora")
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}")
        with TestCase().assertRaisesRegex(ValueError, "LoRA adapter requires the direct"):
            loader.load(path, dtype="float32", loader="hf")


def check_graphs(backbone, device):
    graph = Qwen2Graphs(backbone)
    assert graph.flat_rung(1, graph.flat_max, 1) == graph.flat_max
    reads = (Read(None),)
    prefix = list(range(1, 92))
    branches = [[7, 9, 11], [13, 17]]
    other_prefix, other_branches = list(range(99, 120)), [[31, 43, 51]]
    jobs = [Job(prefix, branches, [reads] * 2, blocks=True),
            Job(other_prefix, other_branches, [reads], prefix_two_way=True)]
    expected = backbone.forward(pack(jobs, device))[FINAL]
    actual = graph.run_flat([(prefix, branches, None, True), (other_prefix, other_branches, None, "prefix")])
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    reads = [(Read(None, FINAL, (-1, -2)),), (Read(None),)]
    spots = (1, len(other_prefix) + 1)
    jobs = [Job(other_prefix, other_branches, [(Read(None, FINAL, spots),)]), Job(prefix, branches, reads)]
    expected = backbone.forward(pack(jobs, device))[FINAL]
    actual = graph.run_flat([(other_prefix, other_branches, None, False, spots),
                             (prefix, branches, None, False, ((2, 1), (1,)))])
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    long_prefix = [i % 127 + 1 for i in range(400)]   # a head over every position: more reads than a graph's usual limit
    spots = tuple(range(len(long_prefix) + 3))
    jobs = [Job(long_prefix, [[8, 9, 10]], [(Read(None, FINAL, spots),)]), Job(other_prefix, other_branches, [(Read(None),)])]
    expected = backbone.forward(pack(jobs, device))[FINAL]
    actual = graph.run_flat([(long_prefix, [[8, 9, 10]], None, False, spots), (other_prefix, other_branches, None)])
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    short = (other_prefix, [[8, 9, 10]], None, False, tuple(range(len(other_prefix) + 3)))   # every position of a short one
    assert graph.whole(short) and graph._rung(1, *graph.flat_cost(short), branches=1) == (32, True)
    jobs = [Job(other_prefix, [[8, 9, 10]], [(Read(None, FINAL, short[4]),)])]
    expected = backbone.forward(pack(jobs, device))[FINAL]
    torch.testing.assert_close(graph.run_flat([short]), expected, atol=2e-5, rtol=2e-5)

    jobs = [Job(prefix, branches, [(Read(None),)] * 2),
            Job(other_prefix, other_branches, [(Read(None),)])]
    expected = backbone.forward(pack(jobs, device))[FINAL]
    cache = SessionCache(2 ** 20, min_tokens=64, stride=64, margin=1)
    cache.attach(backbone.kv_arena(cache.budget))
    graph.set_store(cache.store, cache)
    engine = Engine.__new__(Engine)
    engine.backbone, engine.sessions = backbone, cache
    early_read = Plan(prefix, branches, (Read(None, layer=1),))
    assert engine._sessions_for([early_read]) == {}
    split_read = Plan(prefix, branches, [(Read(None, layer=1),), (Read(None),)])
    engine.run_plans([split_read], budget=len(prefix) + len(branches[0]))
    assert cache.peek(prefix) == 0
    actual, snaps = graph.run_sflat([(prefix, branches, None, 0, 64),
                                    (other_prefix, other_branches, None, 0, None)])
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert snaps[1] == []
    mark, (state, rows) = snaps[0][0]
    entry = Entry(prefix[:mark], state, rows)
    cache.insert(entry)
    hit, start = cache.match(prefix)
    assert hit is entry and start == 64
    assert engine._sessions_for([early_read]) == {0: {"resume": hit, "start": 64, "marks": ()}}
    actual, _ = graph.run_sflat([(prefix, branches, hit, start, None),
                                (other_prefix, other_branches, None, 0, None)])
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    jobs[0].start, jobs[0].resume = start, hit
    actual = backbone.forward(pack(jobs, device))[FINAL]
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    prefix = prefix + [97] * 64
    job = Job(prefix, branches, [(Read(None),)] * 2)
    expected = backbone.forward(pack([job], device))[FINAL]
    job.start, job.resume, job.marks = start, hit, (128,)
    batch = pack([job], device)
    save = cache.reserve({"full": 128}, keep=[hit])
    batch.saved[0] = {"states": {128: {}}, "kv": save}
    actual = backbone.forward(batch)[FINAL]
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    newer = Entry(prefix[:128], batch.saved[0]["states"], save)
    job.start, job.resume, job.marks = 128, newer, ()
    actual = backbone.forward(pack([job], device))[FINAL]
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@torch.inference_mode()
def main():
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(23)
    device = torch.device("cuda:0")
    check_rope_validation()
    positions = (-1, 0, -2, -1)
    reads = (Read(None, FINAL, positions), Read(None, 1, positions), Read(None, 0, positions))
    jobs = [
        Job(list(range(1, 67)), [[8, 9, 10], [11, 12, 13, 14]], [reads] * 2),
        Job([41, 5, 7], [[19, 21], [22, 24, 27]], [reads] * 2, blocks=True),
        Job([7, 29, 31], [[32, 34], [35, 37, 39]], [reads] * 2, prefix_two_way=True),
        Job([1], [list(range(2, 113)), list(range(36, 127))], [reads] * 2),
    ]

    for tied in (True, False):
        config = Qwen2Config(vocab_size=128, hidden_size=256, intermediate_size=384, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=tied)
        config._attn_implementation = "sdpa"
        reference = Qwen2ForCausalLM(config).eval()
        for layer in reference.model.layers:
            for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                proj.bias.normal_(std=0.05)
        with TemporaryDirectory() as path:
            reference.save_pretrained(path, max_shard_size="1MB")
            backbone = get("qwen2").load(path, dtype="float32")
        reference.to(device)
        if tied:
            check_checkpoint_validation(reference)
        weight, bias = backbone.output_head()
        assert bias is None
        assert (weight.data_ptr() == backbone.lm.embed_tokens.weight.data_ptr()) == tied
        torch.testing.assert_close(weight, reference.lm_head.weight, atol=0, rtol=0)

        batch = pack(jobs, device)
        actual = backbone.forward(batch)
        expected = {FINAL: [], 1: [], 0: []}
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
                ref = reference.model(ids, attention_mask={"full_attention": allowed[None, None]},
                                      use_cache=False, output_hidden_states=True)
                expected[FINAL].append(ref.last_hidden_state[0, list(positions)])
                for layer in (0, 1):
                    expected[layer].append(ref.hidden_states[layer][0, list(positions)])
        for layer, rows in expected.items():
            torch.testing.assert_close(actual[layer], torch.cat(rows), atol=2e-5, rtol=2e-5)
        batch.reads[backbone.num_layers] = batch.reads[FINAL]
        full = backbone.forward(batch)
        torch.testing.assert_close(full[FINAL], actual[FINAL], atol=2e-5, rtol=2e-5)

    previous = settings.batch_invariant
    settings.batch_invariant = True
    try:
        together = backbone.forward(pack(jobs, device))[FINAL]
        separate = torch.cat([backbone.forward(pack([job], device))[FINAL] for job in jobs])
        torch.testing.assert_close(together, separate, atol=0, rtol=0)
    finally:
        settings.batch_invariant = previous
    check_graphs(backbone, device)
    print("PASS native Qwen2 loading, checkpoint validation, packed reads, CUDA graphs, and cached KV")


if __name__ == "__main__":
    main()

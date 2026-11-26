"""Tensor parallel: each rank reading only its part of the weights (the per-tensor cut as they are read) holds the same
tensors, bit for bit, as reading the whole model and cutting it, in 16 bits and in float8 (no GPU needed)."""
import tempfile
import types
from unittest.mock import Mock, patch

import torch
import transformers

from flide import backbones, settings
from flide.backbones.load import build, load_into
from flide.backbones.qwen35 import decoder
from flide.kernels.fp8 import quantize_rows
from flide.runtime import tp

CPU = torch.device("cpu")
ARCH = transformers.Qwen3_5ForCausalLM
_CKPT = None


def checkpoint():
    global _CKPT
    if _CKPT is None:
        cfg = transformers.Qwen3_5TextConfig(
            vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=4, num_attention_heads=8,
            num_key_value_heads=4, head_dim=32, linear_num_key_heads=4, linear_num_value_heads=8, linear_key_head_dim=32,
            linear_value_head_dim=32, layer_types=["linear_attention"] * 3 + ["full_attention"], tie_word_embeddings=False)
        torch.manual_seed(0)
        model = ARCH(cfg).to(torch.bfloat16)
        with torch.no_grad():
            for p in model.parameters():
                p.normal_(0, 0.05)
        d = tempfile.mkdtemp()
        model.save_pretrained(d)
        _CKPT = cfg, d
    return _CKPT


def holder(model, cfg):
    return types.SimpleNamespace(layers=model.model.layers, layer_types=list(cfg.layer_types), device=CPU)


def check(size):
    cfg, d = checkpoint()
    for rank in range(size):
        whole = build(ARCH, cfg, CPU, torch.bfloat16)
        load_into(whole, d, "cpu")
        decoder.Qwen35.shard(holder(whole, cfg), rank, size)
        ref = whole.state_dict()

        part, store, cuts = decoder._read_tp(ARCH, cfg, d, "cpu", "bfloat16", (rank, size), False, [], ("mtp.",), False)
        decoder.Qwen35.shard(holder(part, cfg), rank, size, cuts=cuts)
        got = part.state_dict()
        assert not store and ref.keys() == got.keys()
        for k in ref:
            assert torch.equal(ref[k], got[k]), k
        for la, lb in zip(whole.model.layers, part.model.layers):
            if hasattr(la, "linear_attn"):
                for att in ("num_k_heads", "num_v_heads", "key_dim", "value_dim", "conv_dim"):
                    assert getattr(la.linear_attn, att) == getattr(lb.linear_attn, att)

        lean, store, cuts = decoder._read_tp(ARCH, cfg, d, "cpu", "bfloat16", (rank, size), True, [], ("mtp.",), False)
        decoder.Qwen35.shard(holder(lean, cfg), rank, size, cuts=cuts)
        held = lean.state_dict()
        for k, t in ref.items():
            assert tuple(held[k].shape) == tuple(t.shape), k   # float8 parts too: the forward reads sizes from them
            if decoder._fp8_target(k):
                q, s = quantize_rows(t)
                q2, s2 = store[k]
                assert torch.equal(q.view(torch.uint8), q2.view(torch.uint8)) and torch.equal(s, s2), k
            else:
                assert torch.equal(t, held[k]), k


def test_two_ranks():
    check(2)


def test_four_ranks():
    check(4)


def test_worker_sets_batch_invariant_before_loading():
    def load(*args, **kwargs):
        assert settings.batch_invariant is True

    conn = Mock()
    conn.recv.return_value = None
    with (patch.object(settings, "batch_invariant", False), patch.object(tp, "_init"),
          patch.object(backbones, "get", return_value=types.SimpleNamespace(load=load)),
          patch("flide.kernels.launcher.install"), patch("flide.runtime.kcache.seed"),
          patch("flide.runtime.kcache.tune")):
        tp._worker(1, 2, 0, "qwen2", "unused", "cuda:1", "bfloat16", {}, True, conn)
    conn.send.assert_called_once_with("ready")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

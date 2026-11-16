"""LFM3's language model as plain modules: the checkpoint's own names and shapes, so the direct loader fills them, and
nothing that runs (fused.py runs the layers). LFM3 ships as code of its own (trust_remote_code), which flide does
not run; its config is read from config.json as it is.

A layer is a gated short convolution (width 7) or attention, full or in a sliding window, each with a SwiGLU
feed-forward; every sublayer has an RMSNorm before it and, with `use_sandwich_norm`, one after it. Attention has
per-head RMSNorm on queries and keys and a sigmoid gate per head on its output. The output head is the tied embedding.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


def text_config(model_dir: str) -> SimpleNamespace:
    """The language model's config (`text_config` of an lfm3_vl checkpoint, else the whole file)."""
    raw = json.loads((Path(model_dir) / "config.json").read_text())
    text = raw.get("text_config") or raw
    cfg = SimpleNamespace(**text)
    cfg.model_type = text.get("model_type", "lfm3")
    return cfg


class RMSNorm(nn.Module):
    def __init__(self, n: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.variance_epsilon = eps


class Attention(nn.Module):
    def __init__(self, cfg, layer_type: str):
        super().__init__()
        H, nh, nkv = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads
        hd = getattr(cfg, "head_dim", None) or H // nh
        self.head_dim, self.scaling = hd, hd ** -0.5
        self.q_proj = nn.Linear(H, nh * hd, bias=False)
        self.k_proj = nn.Linear(H, nkv * hd, bias=False)
        self.v_proj = nn.Linear(H, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nh * hd, H, bias=False)
        self.gate_proj = nn.Linear(H, nh, bias=False)
        self.q_norm = RMSNorm(hd, cfg.norm_eps)
        self.k_norm = RMSNorm(hd, cfg.norm_eps)
        self.layer_type = layer_type
        self.sliding_window = cfg.sliding_window if layer_type == "sliding_attention" else None


class ShortConv(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H, K, bias = cfg.hidden_size, cfg.conv_L_cache, bool(getattr(cfg, "conv_bias", False))
        self.bias = bias
        self.conv = nn.Conv1d(H, H, kernel_size=K, groups=H, bias=bias, padding=K - 1)
        self.in_proj = nn.Linear(H, 3 * H, bias=bias)
        self.out_proj = nn.Linear(H, H, bias=bias)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H, I = cfg.hidden_size, cfg.intermediate_size
        self.w1 = nn.Linear(H, I, bias=False)
        self.w3 = nn.Linear(H, I, bias=False)
        self.w2 = nn.Linear(I, H, bias=False)


class Layer(nn.Module):
    def __init__(self, cfg, i: int):
        super().__init__()
        t = cfg.layer_types[i]
        self.layer_type = t
        self.is_attention_layer = t != "conv"
        if self.is_attention_layer:
            self.self_attn = Attention(cfg, t)
        else:
            self.conv = ShortConv(cfg)
        self.feed_forward = MLP(cfg)
        self.operator_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        if getattr(cfg, "use_sandwich_norm", False):
            self.post_operator_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
            self.post_ffn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)


class Text(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Layer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.embedding_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)


class Lfm3ForCausalLM(nn.Module):
    """The language model under `model`, the names a checkpoint's `model.language_model.*` map onto."""

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.model = Text(cfg)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.model.embed_tokens

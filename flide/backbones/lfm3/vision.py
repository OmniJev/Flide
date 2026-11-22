"""LFM3-VL's vision tower and projector (the pictures of LiquidAI's d1-2B), from transformers' LFM3-VL code at commit
69da1b47 (Apache 2.0), the copy d1-2B ships: a linear patchifier and an RMSNorm, then layers of a gated depthwise 3 x 3
convolution or bidirectional attention over the patches of each picture (axial rotary positions, RMSNorm on the query
and key heads), each followed by a SwiGLU feed-forward; the projector merges each 2 x 2 block of patches into one
language token. The patches arrive grouped by merge block (the processor's order), so a block is four rows in a row.
Module names are the checkpoint's, under model.vision_tower and model.multi_modal_projector.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn


def vision_config(model_dir: str) -> SimpleNamespace:
    """The tower's config and the projector's settings of an lfm3_vl checkpoint."""
    raw = json.loads((Path(model_dir) / "config.json").read_text())
    cfg = SimpleNamespace(**raw["vision_config"])
    cfg.text_hidden_size = raw["text_config"]["hidden_size"]
    for k in ("downsample_factor", "projector_hidden_size", "projector_bias", "projector_hidden_act",
              "projector_layer_norm_eps", "image_token_id"):
        setattr(cfg, k, raw[k])
    return cfg


class RMSNorm(nn.Module):
    def __init__(self, n: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w3 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class Attention(nn.Module):
    """Grouped-query attention over the patches of each picture, both ways."""

    def __init__(self, cfg):
        super().__init__()
        self.heads, self.kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.q_proj = nn.Linear(cfg.hidden_size, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.heads * self.head_dim, cfg.hidden_size, bias=False)
        self.q_layernorm = RMSNorm(self.head_dim, cfg.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, cfg.norm_eps)

    def forward(self, h, cos, sin, lengths: list[int]):
        n = h.shape[0]
        q = self.q_layernorm(self.q_proj(h).view(n, self.heads, self.head_dim))
        k = self.k_layernorm(self.k_proj(h).view(n, self.kv_heads, self.head_dim))
        v = self.v_proj(h).view(n, self.kv_heads, self.head_dim)
        c, s = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        q = ((q.float() * c) + (_rotate_half(q.float()) * s)).to(h.dtype)
        k = ((k.float() * c) + (_rotate_half(k.float()) * s)).to(h.dtype)
        rep = self.heads // self.kv_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        outs = []
        for qi, ki, vi in zip(*(t.split(lengths) for t in (q, k, v))):   # each picture attends to itself
            o = F.scaled_dot_product_attention(qi.transpose(0, 1)[None], ki.transpose(0, 1)[None],
                                               vi.transpose(0, 1)[None], scale=self.head_dim ** -0.5)
            outs.append(o[0].transpose(0, 1))
        return self.out_proj(torch.cat(outs).reshape(n, -1))


class Conv(nn.Module):
    """Gated depthwise 2D convolution over each picture's patch grid."""

    def __init__(self, cfg):
        super().__init__()
        self.merge = cfg.spatial_merge_size
        self.in_proj = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.spatial_conv = nn.Conv2d(cfg.hidden_size, cfg.hidden_size, kernel_size=cfg.conv_kernel_size,
                                      padding=cfg.conv_kernel_size // 2, groups=cfg.hidden_size, bias=False)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, h, shapes: list[tuple[int, int]]):
        b, c, x = self.in_proj(h).chunk(3, dim=-1)
        gated = b * x
        m = self.merge
        outs = []
        for frame, (hh, ww) in zip(gated.split([a * w for a, w in shapes]), shapes):
            # merge-block order -> the raster grid, convolve, back
            f = frame.view(hh // m, ww // m, m, m, -1).permute(4, 0, 2, 1, 3).reshape(1, -1, hh, ww)
            f = self.spatial_conv(f)
            f = f.view(-1, hh // m, m, ww // m, m).permute(1, 3, 2, 4, 0)
            outs.append(f.reshape(hh * ww, -1))
        return self.out_proj(c * torch.cat(outs))


class Layer(nn.Module):
    def __init__(self, cfg, kind: str):
        super().__init__()
        if kind == "full_attention":
            self.self_attn = Attention(cfg)
        else:
            self.conv = Conv(cfg)
        self.feed_forward = MLP(cfg)
        self.operator_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)

    def forward(self, h, cos, sin, shapes):
        x = self.operator_norm(h)
        if hasattr(self, "self_attn"):
            h = h + self.self_attn(x, cos, sin, [a * w for a, w in shapes])
        else:
            h = h + self.conv(x, shapes)
        return h + self.feed_forward(self.ffn_norm(h))


class Tower(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.merge = cfg.spatial_merge_size
        self.patchifier = nn.Linear(cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size ** 2, cfg.hidden_size,
                                    bias=False)
        self.embedding_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.layers = nn.ModuleList(Layer(cfg, kind) for kind in cfg.layer_types)
        dim = cfg.hidden_size // cfg.num_attention_heads // 2   # half the head per axis
        theta = cfg.rope_parameters["rope_theta"]
        # on the CPU in float32, as the release computes them (on the GPU the last bit can differ); moved when used
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device="cpu") / dim))

    def positions(self, grid: list[list[int]], device) -> torch.Tensor:
        """(row, column) of every patch, in merge-block order, frame after frame."""
        m, out = self.merge, []
        for t, h, w in grid:
            rows = torch.arange(h, device=device)[:, None].expand(-1, w)
            cols = torch.arange(w, device=device)[None, :].expand(h, -1)
            pos = [p.reshape(h // m, m, w // m, m).transpose(1, 2).flatten() for p in (rows, cols)]
            out.append(torch.stack(pos, dim=-1).repeat(t, 1))
        return torch.cat(out)

    def forward(self, pixel_values, grid: list[list[int]]):
        shapes = [(h, w) for t, h, w in grid for _ in range(t)]
        h = self.embedding_norm(self.patchifier(pixel_values.to(self.patchifier.weight.dtype)))
        freqs = self.positions(grid, h.device)[..., None].float() * self.inv_freq.to(h.device)   # [N, 2, dim / 2]
        cos, sin = freqs.cos(), freqs.sin()
        cos = torch.cat([cos[:, 0], cos[:, 1]], -1).repeat(1, 2)
        sin = torch.cat([sin[:, 0], sin[:, 1]], -1).repeat(1, 2)
        for layer in self.layers:
            h = layer(h, cos, sin, shapes)
        return h


class Projector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.downsample_factor ** 2
        self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.projector_layer_norm_eps)
        self.linear_1 = nn.Linear(cfg.hidden_size * d, cfg.projector_hidden_size, bias=cfg.projector_bias)
        self.linear_2 = nn.Linear(cfg.projector_hidden_size, cfg.text_hidden_size, bias=cfg.projector_bias)
        if cfg.projector_hidden_act != "gelu":
            raise ValueError(f"lfm3_vl: projector activation {cfg.projector_hidden_act!r}")

    def forward(self, x):
        x = self.layer_norm(x).reshape(-1, self.linear_1.in_features)
        return self.linear_2(F.gelu(self.linear_1(x)))


class Vision(nn.Module):
    """The tower and the projector, named as the checkpoint names them after `model.`."""

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.vision_tower = Tower(cfg)
        self.multi_modal_projector = Projector(cfg)

    def features(self, pixel_values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """One language-sized row per merge block of every picture, pictures in order."""
        return self.multi_modal_projector(self.vision_tower(pixel_values, grid.tolist()))

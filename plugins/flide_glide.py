"""GLiDE (fastino/GLiDE-0.8B, 2B, 9B, 27B: Qwen3.5 and Qwen3.8 whose full-attention layers read each question's whole
prompt both ways, with a linear readout; 9B and 27B run their middle layers twice, the decision_config's "midloop").
Prompts are the shipped configurations formats/known/fastino__glide.json and fastino__glide_midloop.json; this family
claims the checkpoints for them and reads the state as the release does: a state that is a list of parts with image
parts gives its text parts, one per line, and its images, which come before the request's "images".

GLiDE-27B reads images (its vision tower) as its release prepares them: an image is resized, aspect kept, so both
sides are multiples of 32 pixels and the area lies within [min_pixels, max_pixels] (256 x 256 and 1024 x 1024 by
default, the release's GLIDE_MIN_PIXELS and GLIDE_MAX_PIXELS), with PIL's bicubic filter; scaled to [-1, 1] in float32,
cut into 16 x 16 patches over two frames (2 x 2 patches merge into one token) and stored in float16. Each image becomes
<|vision_start|>, one <|image_pad|> per merged patch and <|vision_end|>, all ahead of the state text, which follows on
the next line. The other sizes turn images down."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from flide.api.schema import SystemOneRequest
from flide.formats import register
from flide.formats.base import Plan
from flide.formats.template import Template

REPOS = ("fastino/GLiDE-0.8B", "fastino/GLiDE-2B", "fastino/GLiDE-9B", "fastino/GLiDE-27B")
FILES = ("readout.safetensors", "decision_config.json")
DECISION_CONFIG = {"format_version": 1, "attention_mode": "noncausal_full_attention", "pooling": "last"}
KEYS = ({"attention_mode", "codes", "format_version", "model_name", "pooling", "prompt_format", "temperature", "token_ids"},
        {"attention_mode", "codes", "format_version", "midloop", "model_name", "pooling", "prompt_format", "temperature",
         "token_ids"},
        {"attention_mode", "codes", "format_version", "image_inputs", "midloop", "pooling", "prompt_format", "temperature",
         "token_ids"})
IMAGE_PARTS, TEXT_PARTS = ("image_url", "image", "input_image"), ("text", "input_text")
VISION_START, IMAGE_TOKEN, VISION_END = "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"


def split_state(state: Any) -> tuple[Any, list]:
    """A list of parts holding image parts: its text, one part per line, and its image references; anything else as
    it is, with none."""
    if not isinstance(state, list) or not any(isinstance(p, dict) and p.get("type") in IMAGE_PARTS for p in state):
        return state, []
    texts, refs = [], []
    for p in state:
        if isinstance(p, dict) and p.get("type") in IMAGE_PARTS:
            u = p.get("image_url", p.get("url", p.get("data", p.get("image"))))
            refs.append(u.get("url") if isinstance(u, dict) else u)
        elif isinstance(p, dict) and p.get("type") in TEXT_PARTS:
            texts.append(str(p.get("text", "")))
        elif isinstance(p, str):
            texts.append(p)
        else:
            texts.append(json.dumps(p, ensure_ascii=False))
    return (texts[0] if len(texts) == 1 else "\n".join(texts)), refs


def fit(h: int, w: int, factor: int, lo: int, hi: int) -> tuple[int, int]:
    """Height and width, multiples of `factor`, near the image's, with the area inside [lo, hi] and the aspect kept."""
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if hb * wb > hi:
        beta = math.sqrt(h * w / hi)
        hb, wb = max(factor, math.floor(h / beta / factor) * factor), max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < lo:
        beta = math.sqrt(lo / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return hb, wb


def patches(img, patch: int, temporal: int, merge: int, lo: int, hi: int):
    """An RGB image as float16 patches [n, 3 * temporal * patch * patch], its grid (1, h, w) and its token count."""
    import numpy as np
    from PIL import Image

    w, h = img.size
    hb, wb = fit(h, w, patch * merge, lo, hi)
    x = np.asarray(img.resize((wb, hb), Image.BICUBIC), dtype=np.float32) / 255.0
    x = np.repeat(((x - 0.5) / 0.5).transpose(2, 0, 1)[None], temporal, axis=0)
    gh, gw = hb // patch, wb // patch
    x = x.reshape(1, temporal, 3, gh // merge, merge, patch, gw // merge, merge, patch)
    x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gh * gw, 3 * temporal * patch * patch)
    return x.astype(np.float16), (1, gh, gw), gh * gw // (merge * merge)


@register("glide")
class Glide(Template):
    backbones = ("qwen3_5", "qwen3_5_text")

    def __init__(self, model_dir: str, tokenizer=None, min_pixels: int = 256 * 256, max_pixels: int = 1024 * 1024):
        dc = json.loads((Path(model_dir) / "decision_config.json").read_text())
        super().__init__(model_dir, tokenizer=tokenizer,
                         spec="fastino__glide_midloop" if "midloop" in dc else "fastino__glide")
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        v = cfg.get("vision_config") if "image_inputs" in dc else None
        self.patch = None if v is None else (int(v["patch_size"]), int(v["temporal_patch_size"]),
                                             int(v["spatial_merge_size"]))
        self.pixels = (int(min_pixels), int(max_pixels))
        self.pad = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        d = Path(model_dir)
        parts = {p.lower() for p in d.resolve().parts}
        if any(f"models--{r.lower().replace('/', '--')}" in parts for r in REPOS):
            return True
        return (decision_config is not None and all((d / f).is_file() for f in FILES)
                and all(decision_config.get(k) == v for k, v in DECISION_CONFIG.items())
                and set(decision_config) in KEYS)

    def plan(self, request: SystemOneRequest, **kw) -> Plan:
        state, refs = split_state(request.state)
        refs = refs + list(getattr(request, "images", None) or [])
        if not refs:
            return super().plan(request if state is request.state else request.model_copy(update={"state": state}), **kw)
        if self.patch is None:
            raise ValueError("images: this model reads text only")
        import numpy as np
        import torch

        px = [patches(self._open(r), *self.patch, *self.pixels) for r in refs]
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        text = "".join(VISION_START + IMAGE_TOKEN * n + VISION_END for _, _, n in px) + "\n" + text
        plan = super().plan(request.model_copy(update={"state": text, "images": None}), **kw)
        rows, total = plan.suffixes, sum(n for _, _, n in px)
        last = max(i for i, t in enumerate(rows[0]) if t == self.pad) + 1
        if any(r[:last] != rows[0][:last] or r.count(self.pad) != total for r in rows):
            raise ValueError("the state or a question spells an image placeholder")
        plan.vision = {"pixel_values": torch.from_numpy(np.concatenate([p for p, _, _ in px])),
                       "image_grid_thw": torch.tensor([g for _, g, _ in px]),
                       "mm_token_type_ids": (torch.tensor(rows[0][:last]) == self.pad).long()[None]}
        return plan

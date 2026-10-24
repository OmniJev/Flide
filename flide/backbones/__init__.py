"""Backbones by HuggingFace `model_type`. A backbone owns the weights and the packed forward; it reads
hidden states where the runtime asks and knows nothing of prompts or heads.

Built in: qwen3_5, gemma4. Other packages add backbones through the `flide.backbones` entry point group
(value: the Backbone subclass), or with `register`.
"""
from __future__ import annotations

import logging
from importlib.metadata import entry_points

from .base import Backbone

log = logging.getLogger("flide.backbones")

GROUP = "flide.backbones"
_BACKBONES: dict[str, type[Backbone]] = {}
_BUILTIN = {"qwen3_5": "flide.backbones.qwen35:Qwen35", "qwen3_5_text": "flide.backbones.qwen35:Qwen35",
            "gemma4": "flide.backbones.gemma4:Gemma4", "gemma4_text": "flide.backbones.gemma4:Gemma4",
            "gemma4_unified": "flide.backbones.gemma4:Gemma4"}
_loaded = False


def register(cls: type[Backbone]) -> type[Backbone]:
    for t in cls.model_types:
        _BACKBONES[t] = cls
    return cls


def _import(value: str):
    import importlib

    mod, _, attr = value.partition(":")
    return getattr(importlib.import_module(mod), attr)


def _load() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    for ep in entry_points(group=GROUP):
        try:
            register(ep.load())
        except Exception as exc:
            log.warning("backbone plugin %s (%s) failed to load: %s: %s", ep.name, ep.value, type(exc).__name__, exc)


def get(model_type: str) -> type[Backbone]:
    _load()
    if model_type in _BACKBONES:
        return _BACKBONES[model_type]
    if model_type in _BUILTIN:
        return register(_import(_BUILTIN[model_type]))
    known = sorted(set(_BACKBONES) | set(_BUILTIN))
    raise ValueError(f"no backbone for model_type {model_type!r}; known: {', '.join(known)}")


__all__ = ["Backbone", "get", "register"]

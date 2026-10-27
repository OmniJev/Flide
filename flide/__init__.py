"""flide: inference for decision models. Read the input once, read answers where the model family
says, never generate."""
from __future__ import annotations

__version__ = "0.1.5"


def __getattr__(name):
    if name == "Engine":
        from .runtime.engine import Engine

        return Engine
    raise AttributeError(name)


__all__ = ["Engine", "__version__"]

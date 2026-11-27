"""Quantized products for any backbone. A backbone names the products it lets quantize as slots (how to read and replace
the weight, how many of its last rows stay in 16 bits, which modules held the 16-bit weight), and this module turns them
into float8 or int8 weights with per-row scales (kernels/fp8.py, kernels/int8.py), smooths int8 with SmoothQuant from
calibration requests, and frees the 16-bit copies. The backbone's products already go through kernels.fp8.mm, which
calls a quantized weight for its product: nothing in the forward branches on the format.

A mixture of experts keeps its other products in 16 bits and takes its routed experts to float8 alone (kernels/moe.py):
in float8 the rest of such a layer moves the answers more than the experts do.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .. import lora
from ..kernels import gpu

MODES = ("fp8", "int8")


@dataclass
class Slot:
    key: tuple                          # (layer, name): what calibration statistics are kept by
    get: Callable[[], object]
    set: Callable[[object], None]
    tail: int = 0                       # last rows kept in 16 bits (Gated DeltaNet's small gate projections)
    modules: list = field(default_factory=list)   # modules whose 16-bit weights the slot's weight held or viewed


def table(weights: list[dict], mods: list[dict]) -> list[Slot]:
    """Slots for weights kept as one dict per layer (a fused decoder's): `mods[i]` {key: modules} names the keys of
    layer i to convert and the modules that held their 16-bit weights."""
    return [Slot((i, k), lambda w=w, k=k: w[k], lambda v, w=w, k=k: w.__setitem__(k, v), modules=m)
            for i, (w, layer) in enumerate(zip(weights, mods)) for k, m in layer.items()]


def check(mode: str | None, device, name: str) -> None:
    """Raises when `mode` is not a quantization this part runs."""
    if mode is None:
        return
    if mode not in MODES:
        raise ValueError(f"{name}: quantize {mode!r}, expected fp8 or int8")
    if mode == "fp8" and not gpu.fp8(device):
        raise RuntimeError(f"{name}: FP8 needs float8 products (NVIDIA Ada, Hopper or Blackwell)")
    if mode == "int8" and not gpu.int8(device):
        raise RuntimeError(f"{name}: int8 needs int8 products (NVIDIA Ampere or newer, AMD CDNA)")


def _put(slot: Slot, make) -> None:
    w = slot.get()
    base = w.base if isinstance(w, lora.LowRank) else w
    q = make(base)
    slot.set(w.with_base(q) if isinstance(w, lora.LowRank) else q)
    for m in slot.modules:   # the modules held (views of) the 16-bit weight; only the quantized one is used now
        m.weight.data = torch.empty(0, device=m.weight.device, dtype=m.weight.dtype)


@torch.no_grad()
def fp8(slots: list[Slot]) -> int:
    """Every slot to float8 weights. Returns the number converted."""
    from ..kernels import fp8 as f8

    for s in slots:
        _put(s, lambda base, s=s: f8.Fp8Weight.from_weight(base, s.tail))
    torch.cuda.empty_cache()
    return len(slots)


@torch.no_grad()
def int8(slots: list[Slot], input_max: dict | None = None, alpha: float = 0.75) -> int:
    """Every slot to int8 weights; `input_max` {key: per-channel max |input|} smooths each one's input (SmoothQuant,
    kernels/int8.smoothing). Returns the number converted."""
    from ..kernels import int8 as i8

    def make(base, s):
        amax = (input_max or {}).get(s.key)
        smooth = None if amax is None else i8.smoothing(amax, base[: base.shape[0] - s.tail], alpha)
        return i8.Int8Weight.from_weight(base, s.tail, smooth)

    for s in slots:
        _put(s, lambda base, s=s: make(base, s))
    torch.cuda.empty_cache()
    return len(slots)


@torch.no_grad()
def calibrate_int8(slots: list[Slot], run: Callable[[], object], alpha: float = 0.75,
                   rewire: Callable[[], None] | None = None) -> int:
    """int8 with SmoothQuant: each slot keeps the per-channel max |input| while `run()` forwards calibration requests
    in 16 bits, then is quantized smoothed by it. `rewire()` rebuilds what the backbone derives from the slots (the last
    layer's split projections), called after the slots are wrapped and again after they are quantized."""
    from ..kernels import int8 as i8

    for s in slots:
        s.set(i8.InputMax(s.get()))
    if rewire is not None:
        rewire()
    try:
        run()
        stats = {s.key: s.get().amax for s in slots}
    finally:
        for s in slots:
            s.set(s.get().w)
    n = int8(slots, stats, alpha)
    if rewire is not None:
        rewire()
    return n


@torch.no_grad()
def experts_fp8(weights: dict, mlp) -> None:
    """A mixture-of-experts layer's routed experts (`weights` as kernels/moe.py reads them) to float8, freeing the
    16-bit expert tensors of `mlp.experts`."""
    from ..kernels.moe import quantize_experts

    for key in ("experts", "experts_down"):
        weights[key], weights[key + "_scale"] = quantize_experts(weights[key])
    for p in (mlp.experts.gate_up_proj, mlp.experts.down_proj):
        p.data = torch.empty(0, device=p.device, dtype=p.dtype)

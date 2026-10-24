"""Weights straight from safetensors onto the device, without transformers' loader (which needs accelerate for any
device placement and reads through several layers of conversion).

The model is built on the device with initialization skipped, then every checkpoint tensor is copied into the
parameter or buffer of the same name (cast to the model's dtype), one tensor at a time, so device memory never holds
two copies. `key_map` rewrites checkpoint names that differ from the module names (a text-only decoder read from a
vision checkpoint). Checkpoint tensors with no home are skipped and reported; module tensors the checkpoint does not
fill raise, except output heads tied to the embedding.

FP8: the weights `lean(name)` picks are never allocated in 16 bits (they stay on the meta device, shape only); each is
read into `fp8` as (float8 weight, per-row scale), quantized on the device as it is read, or taken as stored when
the checkpoint already holds it in float8 with a `weight_scale` (compressed-tensors). Device memory
then holds the float8 copy only.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import torch

log = logging.getLogger("flide.load")


def checkpoint_files(model_dir: str) -> list[Path]:
    d = Path(model_dir)
    index = d / "model.safetensors.index.json"
    if index.is_file():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [d / f for f in files]
    single = d / "model.safetensors"
    if single.is_file():
        return [single]
    found = sorted(d.glob("*.safetensors"))
    if not found:
        raise FileNotFoundError(f"no safetensors weights in {model_dir}")
    return found


def _set_dtype(config, dtype) -> None:
    """transformers builds sub-models from their own config's dtype; make every level say the run dtype."""
    seen = set()
    stack = [config]
    while stack:
        c = stack.pop()
        if id(c) in seen:
            continue
        seen.add(id(c))
        for attr in ("dtype", "torch_dtype"):
            if hasattr(c, attr):
                try:
                    setattr(c, attr, dtype)
                except Exception:
                    pass
        for v in vars(c).values():
            if hasattr(v, "to_dict") and hasattr(v, "model_type"):
                stack.append(v)


def build(cls, config, device, dtype, lean=None):
    """The model on `device` with initialization skipped; parameters `lean(name)` picks stay on the meta device."""
    from transformers.initialization import no_init_weights

    _set_dtype(config, dtype)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with no_init_weights(), torch.device(device if lean is None else "meta"):
            model = cls(config)
    finally:
        torch.set_default_dtype(prev)
    if lean is not None:
        for mod_name, mod in model.named_modules():
            for pn, p in list(mod._parameters.items()):
                if p is not None and not lean(f"{mod_name}.{pn}" if mod_name else pn):
                    mod._parameters[pn] = torch.nn.Parameter(torch.empty(p.shape, device=device, dtype=p.dtype),
                                                             requires_grad=False)
            for bn, b in list(mod._buffers.items()):
                if b is not None:
                    mod._buffers[bn] = torch.empty(b.shape, device=device, dtype=b.dtype)
    return model.eval()


def _init_buffers(model) -> None:
    """Non-persistent buffers of parameter-free modules (rotary inverse frequencies, positional tables) set the way
    from_pretrained sets them: by the model's own _init_weights, computed on the CPU and copied. Computed on the GPU
    while the model is built they can differ in the last bit."""
    init = getattr(model, "_init_weights", None)
    if init is None:
        return
    for mod in model.modules():
        if getattr(mod, "_non_persistent_buffers_set", None) and next(mod.parameters(recurse=False), None) is None:
            init(mod)


@torch.inference_mode()
def blank(model, device=None, lean=None, fp8: dict | None = None) -> None:
    """Zeros for every parameter, buffers as loading sets them: a model that runs every kernel of the real one
    without reading its weights (kernels depend on shapes only; `flide kernels export --no-weights`). With `lean`
    (a model built lean, see `build`), the parameters it picks go into `fp8` as float8 zeros, as `load_into` would
    put them."""
    for name, p in model.named_parameters():
        if lean is not None and lean(name):
            from ..kernels.fp8 import FP8

            fp8[name] = (torch.zeros(p.shape, dtype=FP8, device=device), torch.ones(p.shape[0], 1, device=device))
        else:
            p.zero_()
    _init_buffers(model)


@torch.inference_mode()
def load_into(model, model_dir: str, device, key_map: list[tuple[str, str]] | None = None,
              skip: tuple[str, ...] = (), lean=None, fp8: dict | None = None) -> dict:
    from safetensors import safe_open

    t0 = time.perf_counter()
    if lean is not None:
        from ..kernels.fp8 import FP8, quantize_rows
    targets = model.state_dict(keep_vars=True)
    rules = [(re.compile(a), b) for a, b in (key_map or [])]
    filled, unused, nbytes = set(), [], 0
    for path in checkpoint_files(model_dir):
        with safe_open(str(path), framework="pt", device=str(device)) as f:
            for key in f.keys():
                name = key
                for rx, rep in rules:
                    name = rx.sub(rep, name)
                if name.startswith(skip) or key.endswith(".weight_scale"):
                    continue
                t = targets.get(name)
                if t is None:
                    unused.append(key)
                    continue
                if lean is not None and lean(name):
                    src = f.get_tensor(key)
                    if tuple(src.shape) != tuple(t.shape):
                        raise ValueError(f"{key}: checkpoint shape {tuple(src.shape)}, model {tuple(t.shape)}")
                    if src.dtype == FP8:
                        fp8[name] = (src, f.get_tensor(key + "_scale").float().reshape(-1, 1))
                    else:
                        fp8[name] = quantize_rows(src.to(t.dtype))
                    nbytes += src.numel() * src.element_size()
                    filled.add(name)
                    continue
                src = f.get_tensor(key)
                if src.dtype == torch.float8_e4m3fn:
                    raise ValueError(f"{key} is stored in float8; load this checkpoint with FP8 (--quantize fp8)")
                if tuple(src.shape) != tuple(t.shape):
                    raise ValueError(f"{key}: checkpoint shape {tuple(src.shape)}, model {tuple(t.shape)}")
                t.data.copy_(src)
                nbytes += src.numel() * src.element_size()
                filled.add(name)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    _init_buffers(model)
    tied = set()
    for a, b in (getattr(model, "_tied_weights_keys", None) or {}).items():
        tied.add(a)
    missing = [k for k in targets if k not in filled and k not in tied and not k.startswith(skip)]
    if missing:
        raise ValueError(f"{len(missing)} tensors of the model are not in the checkpoint, e.g. {missing[:5]}")
    dt = time.perf_counter() - t0
    log.info("read %.2f GB of weights in %.1fs (%d checkpoint tensors unused)", nbytes / 1e9, dt, len(unused))
    return {"bytes": nbytes, "seconds": dt, "unused": unused}

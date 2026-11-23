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
then holds the float8 copy only. A float8 tensor of such a checkpoint that FP8 keeps in 16 bits (Gated DeltaNet's gate
projections) is read back to 16 bits through its scale.

Tensor parallel: with `cut`, the tensors `lean` picks are cut to this rank's part as they are read (`cut(name)` gives
the dimension and the part), then quantized into `fp8` (when `cut` says so) or kept in 16 bits in `held`, so a rank
never holds the whole model.

A LoRA adapter under lora/ of the directory (flide/lora.py) is read with the checkpoint (modules saved whole and
an embedding's LoRA go into the tensors as they are read) and its projections' LoRA attached to the modules after it,
unmerged, or with `merge` merged into their weights as they are read; with one, a failed read raises instead of
falling back to transformers, which would drop it.
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


def tensor_files(model_dir: str) -> dict[str, Path]:
    """The shard of each tensor the checkpoint's index lists (empty without an index): a float8 weight's scale may sit
    in another shard than the weight."""
    d = Path(model_dir)
    index = d / "model.safetensors.index.json"
    if not index.is_file():
        return {}
    return {k: d / f for k, f in json.loads(index.read_text())["weight_map"].items()}


def _set_dtype(config, dtype) -> None:
    """transformers builds sub-models from their own config's dtype; make every level say the run dtype."""
    seen = set()
    stack = [config]
    while stack:
        c = stack.pop()
        if id(c) in seen:
            continue
        seen.add(id(c))
        attr = "dtype" if hasattr(c, "dtype") else "torch_dtype" if "torch_dtype" in vars(c) else None
        if attr:
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
        # buffers a module with parameters sets in __init__ (Gemma 4's embedding scale) were lost on the meta device: the
        # model's own _init_weights sets them again (the parameters it also fills are read from the checkpoint after)
        init = getattr(model, "_init_weights", None)
        if init is not None:
            for mod in model.modules():
                if getattr(mod, "_non_persistent_buffers_set", None) and next(mod.parameters(recurse=False), None) is not None:
                    init(mod)
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

            scales = p.shape[:2] if p.ndim == 3 else (p.shape[0], 1)
            fp8[name] = (torch.zeros(p.shape, dtype=FP8, device=device), torch.ones(scales, device=device))
        else:
            p.zero_()
    _init_buffers(model)


def part(t, idx, dim: int = 0):
    """The part `idx` (a slice, or a tuple of slices joined in order) of `t` along `dim`, in memory of its own (a view
    would keep the whole of `t` alive)."""
    if isinstance(idx, tuple):
        return torch.cat([t.narrow(dim, s.start, s.stop - s.start) for s in idx], dim)
    return t.narrow(dim, idx.start, idx.stop - idx.start).clone(memory_format=torch.contiguous_format)


def _read_part(f, key: str, idx, dim: int):
    """The part `idx` of checkpoint tensor `key`, read from the file alone (the rest never reaches the device)."""
    sl = f.get_slice(key)
    pieces = [sl[s.start: s.stop] if dim == 0 else sl[:, s.start: s.stop] for s in (idx if isinstance(idx, tuple) else (idx,))]
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim)


@torch.inference_mode()
def load_into(model, model_dir: str, device, key_map: list[tuple[str, str]] | None = None,
              skip: tuple[str, ...] = (), lean=None, fp8: dict | None = None, merge: bool = False,
              cut=None, held: dict | None = None) -> dict:
    from .. import lora

    adapter = lora.load(model_dir, device, merge)
    if adapter is None:
        return _load_into(model, model_dir, device, key_map, skip, lean, fp8, None, cut, held)
    try:
        out = _load_into(model, model_dir, device, key_map, skip, lean, fp8, adapter, cut, held)
        adapter.attach(model)
        adapter.check()
    except (ValueError, FileNotFoundError) as exc:
        raise RuntimeError(f"{model_dir}: with its LoRA adapter: {exc}") from exc
    log.info("a LoRA adapter %s the weights (%d tensors)", "merged into" if merge else "beside", len(adapter))
    return out


def _expert_slice(name, targets):
    match = re.fullmatch(r"(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", name)
    if match is None:
        return None
    prefix, index, projection = match.groups()
    index = int(index)
    fused = prefix + (".down_proj" if projection == "down_proj" else ".gate_up_proj")
    target = targets.get(fused)
    if target is None:
        return None
    if target.ndim != 3 or not 0 <= index < target.shape[0]:
        raise ValueError(f"{name}: invalid expert index or fused shape {tuple(target.shape)}")
    rows = slice(None)
    if projection != "down_proj":
        if target.shape[1] % 2:
            raise ValueError(f"{fused}: gate/up rows must divide evenly")
        half = target.shape[1] // 2
        rows = slice(0, half) if projection == "gate_proj" else slice(half, None)
    return fused, (index, projection), rows, target[index, rows]


class FP8CheckpointError(RuntimeError):
    """A recognized FP8 tensor cannot be interpreted safely by this loader."""


def _check_fp8_dtype(src, key):
    if str(src.dtype).startswith("torch.float8") and src.dtype != torch.float8_e4m3fn:
        raise FP8CheckpointError(f"{key}: unsupported FP8 dtype {src.dtype}; expected float8_e4m3fn")


def _row_scales(weight, scale, key):
    """Normalize tensor/expert/row scales without changing the FP8 values."""
    scale = scale.float()
    if weight.ndim == 2:
        n = weight.shape[0]
        if scale.numel() == 1:
            return scale.reshape(1, 1).expand(n, 1).contiguous()
        if tuple(scale.shape) in ((n,), (n, 1), (1, n)):
            return scale.reshape(n, 1).contiguous()
    elif weight.ndim == 3:
        e, n, _ = weight.shape
        if scale.numel() == 1:
            return scale.reshape(1, 1).expand(e, n).contiguous()
        if tuple(scale.shape) in ((e, n), (e, n, 1)):
            return scale.reshape(e, n).contiguous()
        if tuple(scale.shape) in ((e,), (e, 1), (e, 1, 1)):
            return scale.reshape(e, 1).expand(e, n).contiguous()
        if key.endswith(".gate_up_proj") and n % 2 == 0 and tuple(scale.shape) == (e, 2):
            return scale.repeat_interleave(n // 2, dim=1).contiguous()
    raise FP8CheckpointError(
        f"{key}: unsupported FP8 scale shape {tuple(scale.shape)} for weight {tuple(weight.shape)}; "
        "only scalar, per-expert, or per-output-row scales are supported (not block scales)")


def _quantize_weight(src):
    if src.ndim == 3:
        from ..kernels.moe import quantize_experts

        return quantize_experts(src)
    from ..kernels.fp8 import quantize_rows

    return quantize_rows(src)


def _load_into(model, model_dir, device, key_map, skip, lean, fp8, adapter, cut=None, held=None) -> dict:
    from safetensors import safe_open

    t0 = time.perf_counter()
    if lean is not None:
        from ..kernels.fp8 import FP8, quantize_rows
    targets = model.state_dict(keep_vars=True)
    rules = [(re.compile(a), b) for a, b in (key_map or [])]
    filled, unused, nbytes = set(), [], 0
    expert_parts, seen = {}, set()
    where = tensor_files(model_dir)
    files = checkpoint_files(model_dir)
    if not where and len(files) > 1:
        for path in files:
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in where:
                        raise ValueError(f"duplicate checkpoint tensor: {key}")
                    where[key] = path
    for path in files:
        with safe_open(str(path), framework="pt", device=str(device)) as f:
            here = set(f.keys())
            duplicates = seen & here
            if duplicates:
                raise ValueError(f"duplicate checkpoint tensor: {min(duplicates)}")
            seen.update(here)

            def scale_of(key):
                """A float8 weight's scale, from this shard or the one the index names."""
                k = key + "_scale"
                if k in here:
                    return f.get_tensor(k)
                if k not in where:
                    raise FP8CheckpointError(f"{key}: missing {k}; block/inverse-scale formats are not supported")
                with safe_open(str(where[k]), framework="pt", device=str(device)) as g:
                    if k not in g.keys():
                        raise FP8CheckpointError(f"{key}: missing {k} in {where[k]}")
                    return g.get_tensor(k)

            for key in f.keys():
                name = key
                for rx, rep in rules:
                    name = rx.sub(rep, name)
                if name.startswith(skip) or (name not in targets and key.endswith("_scale")
                                            and (key[:-6] in here or key[:-6] in where)):
                    continue
                t = targets.get(name)
                expert = _expert_slice(name, targets) if t is None else None
                if expert is not None:
                    name, part_id, rows, t = expert
                    if name in filled:
                        raise ValueError(f"{name}: both fused and split expert weights are present")
                    parts = expert_parts.setdefault(name, set())
                    if part_id in parts:
                        raise ValueError(f"duplicate expert projection: {key}")
                    parts.add(part_id)
                elif name in expert_parts:
                    raise ValueError(f"{name}: both fused and split expert weights are present")
                if t is None:
                    unused.append(key)
                    continue
                if expert is not None and lean is not None and lean(name):
                    if adapter is not None or cut is not None:
                        raise ValueError("split FP8 experts with LoRA or tensor parallelism are not supported")
                    src = f.get_tensor(key)
                    _check_fp8_dtype(src, key)
                    if tuple(src.shape) != tuple(t.shape):
                        raise ValueError(f"{key}: checkpoint shape {tuple(src.shape)}, model {tuple(t.shape)}")
                    if src.dtype == FP8:
                        q, scale = src, _row_scales(src, scale_of(key), key)
                    else:
                        q, scale = _quantize_weight(src.to(t.dtype))
                    if name not in fp8:
                        shape = targets[name].shape
                        fp8[name] = (torch.empty(shape, device=src.device, dtype=FP8),
                                     torch.empty(shape[:2], device=src.device, dtype=torch.float32))
                    weights, scales = fp8[name]
                    weights[part_id[0], rows].copy_(q)
                    scales[part_id[0], rows].copy_(scale[:, 0])
                    nbytes += src.numel() * src.element_size()
                    continue
                if cut is not None and lean(name) and adapter is None and "F8" not in f.get_slice(key).get_dtype():
                    # tensor parallel: this rank's part read alone, then float8 or 16 bits
                    shape = tuple(f.get_slice(key).get_shape())
                    if shape != tuple(t.shape):
                        raise ValueError(f"{key}: checkpoint shape {shape}, model {tuple(t.shape)}")
                    dim, idx, to_fp8 = cut(name)
                    p = _read_part(f, key, idx, dim).to(device=device, dtype=t.dtype)
                    if to_fp8:
                        fp8[name] = quantize_rows(p)
                    else:
                        held[name] = p
                    nbytes += p.numel() * p.element_size()
                    filled.add(name)
                    continue
                if lean is not None and lean(name):
                    src = f.get_tensor(key)
                    _check_fp8_dtype(src, key)
                    if tuple(src.shape) != tuple(t.shape):
                        raise ValueError(f"{key}: checkpoint shape {tuple(src.shape)}, model {tuple(t.shape)}")
                    if cut is not None:   # tensor parallel with a float8 checkpoint or a LoRA: read whole, then cut
                        dim, idx, to_fp8 = cut(name)
                        if src.dtype == FP8:
                            scale = _row_scales(src, scale_of(key), key)
                            if to_fp8:
                                fp8[name] = (part(src, idx, dim), part(scale, idx) if dim == 0 else scale)
                            else:
                                held[name] = part((src.float() * scale).to(t.dtype), idx, dim)
                        else:
                            if adapter is not None:
                                src = adapter.apply(key, src)
                            p = part(src.to(t.dtype), idx, dim)
                            if to_fp8:
                                fp8[name] = quantize_rows(p)
                            else:
                                held[name] = p
                        nbytes += src.numel() * src.element_size()
                        filled.add(name)
                        continue
                    if adapter is not None and src.dtype != FP8:
                        src = adapter.apply(key, src)
                    if src.dtype == FP8:
                        fp8[name] = (src, _row_scales(src, scale_of(key), key))
                    else:
                        fp8[name] = _quantize_weight(src.to(t.dtype))
                    nbytes += src.numel() * src.element_size()
                    filled.add(name)
                    continue
                src = f.get_tensor(key)
                _check_fp8_dtype(src, key)
                if src.dtype == torch.float8_e4m3fn:
                    if lean is None:
                        raise ValueError(f"{key} is stored in float8; load this checkpoint with FP8 (--quantize fp8)")
                    scale = _row_scales(src, scale_of(key), key)
                    src = (src.float() * (scale.unsqueeze(-1) if src.ndim == 3 else scale)).to(t.dtype)
                if adapter is not None:
                    src = adapter.apply(key, src)
                if tuple(src.shape) != tuple(t.shape):
                    raise ValueError(f"{key}: checkpoint shape {tuple(src.shape)}, model {tuple(t.shape)}")
                t.data.copy_(src)
                nbytes += src.numel() * src.element_size()
                if expert is None:
                    filled.add(name)
    for name, parts in expert_parts.items():
        projections = ("down_proj",) if name.endswith(".down_proj") else ("gate_proj", "up_proj")
        expected = {(i, p) for i in range(targets[name].shape[0]) for p in projections}
        if parts != expected:
            raise ValueError(f"{name}: missing expert projections {sorted(expected - parts)[:5]}")
        filled.add(name)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    head = model.get_output_embeddings() if adapter is not None and adapter.embedding is not None else None
    if head is not None and head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr():
        head.weight = torch.nn.Parameter(adapter.embedding, requires_grad=False)   # peft leaves a tied head as it was
    _init_buffers(model)
    tied = set()
    for a, b in (getattr(model, "_tied_weights_keys", None) or {}).items():
        tied.add(a)
    missing = [k for k in targets if k not in filled and k not in tied and not k.startswith(skip)]
    # non-persistent buffers are not in state_dict; persistent ones the checkpoint lacks are an error like weights
    if missing:
        raise ValueError(f"{len(missing)} tensors of the model are not in the checkpoint, e.g. {missing[:5]}")
    dt = time.perf_counter() - t0
    log.info("read %.2f GB of weights in %.1fs (%d checkpoint tensors unused)", nbytes / 1e9, dt, len(unused))
    return {"bytes": nbytes, "seconds": dt, "unused": unused}

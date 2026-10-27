"""LoRA adapters (peft format): a model directory that holds an adapter and no checkpoint of its own runs from its base
checkpoint with the adapter beside the weights, unmerged as peft runs it: each projection keeps its 16-bit weight and
adds x A^T (scaling B)^T (`attach`; the fused decoders read it through `attached`, every other path through a forward
hook), so the answers carry no rounding of a merged weight. A family whose release serves its adapters merged
(Format.merge_lora) gets them merged as the weights are read: W + scaling B A in float32, rounded once, as peft's
merge_and_unload over a float32 product. modules_to_save replace their module's tensors and an embedding's LoRA is
merged into its table as they are read. It serves as the full model it stands for, on every path, with the kernel
bundles as they are.

`view` makes the directory the engine loads: links to the base checkpoint's files, the adapter's own files over them
(tokenizer, chat template, decision configuration, readout heads), the adapter itself under lora/. It sits under
FLIDE_CACHE (~/.cache/flide) by the adapter's Hub name (models--org--name), so configurations matched by
repository still find it. `Adapter` is what the loader reads.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import torch

CONFIG = "adapter_config.json"
WEIGHTS = "adapter_model.safetensors"
SUB = "lora"
_PREFIXES = ("base_model.model.",)
_ROOTS = ("model.language_model.", "language_model.model.", "language_model.", "model.")


def _peft(d: Path) -> bool:
    return (d / CONFIG).is_file() and (d / WEIGHTS).is_file()


def adapter_dir(model_dir: str) -> Path | None:
    """The peft adapter of a directory without a checkpoint of its own: at its top, or in its one subdirectory that
    holds one (a release that keeps its own files beside it)."""
    d = Path(model_dir)
    if (d / "config.json").is_file():
        return None
    if (d / CONFIG).is_file():
        return d
    subs = [s for s in d.iterdir() if s.is_dir() and _peft(s)] if d.is_dir() else []
    return subs[0] if len(subs) == 1 else None


def is_adapter(model_dir: str) -> bool:
    return adapter_dir(model_dir) is not None


def _hub_name(d: Path) -> str:
    for p in [d, *d.parents]:
        if p.name.startswith("models--"):
            return p.name
    return d.name


def view(model_dir: str, resolve) -> str:
    """The directory an adapter runs from (made once); `resolve` turns the adapter's base model into a local directory."""
    src = Path(model_dir).resolve()
    peft = adapter_dir(str(src)).resolve()
    cfg = json.loads((peft / CONFIG).read_text())
    if not (peft / WEIGHTS).is_file():
        raise FileNotFoundError(f"{peft}: {CONFIG} without {WEIGHTS}")
    base = Path(resolve(cfg["base_model_name_or_path"])).resolve()
    key = hashlib.sha256(f"{src}\n{base}".encode()).hexdigest()[:16]
    out = Path(os.environ.get("FLIDE_CACHE", Path.home() / ".cache" / "flide")) / SUB / _hub_name(src) / key
    if (out / ".done").is_file():
        return str(out)
    out.mkdir(parents=True, exist_ok=True)
    links = {f.name: f for f in base.iterdir() if f.name != CONFIG}
    links.update({f.name: f for f in src.iterdir() if f.name not in (CONFIG, WEIGHTS) and not f.name.startswith(".")})
    for name, target in links.items():
        link = out / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    (out / SUB).mkdir(exist_ok=True)
    for name in (CONFIG, WEIGHTS):
        link = out / SUB / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(peft / name)
    (out / ".done").write_text(f"{src}\n{base}\n")
    return str(out)


def canonical(key: str) -> str:
    """A tensor name with the wrapper and the language model's own prefix taken off: layers.N...., embed_tokens...,
    lm_head...; adapters trained on a text model and on its multimodal wrapper name the same weight alike."""
    for p in _PREFIXES:
        if key.startswith(p):
            key = key[len(p):]
    key = key.replace(".modules_to_save.default", "").replace(".original_module", "").replace(".base_layer", "")
    for p in _ROOTS:
        if key.startswith(p):
            return key[len(p):]
    return key


class Adapter:
    """The adapter of a loaded directory: `apply(checkpoint key, tensor)` returns the tensor the model holds (a
    projection's own weight, a module saved whole, an embedding with its LoRA merged); `attach(model)` gives the
    projections their LoRA; `check()` raises if a part of the adapter found no tensor or module of the checkpoint."""

    def __init__(self, model_dir: str, device, merge: bool = False):
        from safetensors import safe_open

        d = Path(model_dir) / SUB
        cfg = json.loads((d / CONFIG).read_text())
        if cfg.get("use_dora") or cfg.get("fan_in_fan_out") or cfg.get("lora_bias"):
            raise NotImplementedError(f"{d / CONFIG}: DoRA, fan_in_fan_out and LoRA biases are not supported")
        self.alpha = float(cfg.get("lora_alpha", 8))
        self.patterns = dict(cfg.get("alpha_pattern") or {})
        self.r = cfg.get("r")
        self.rank_patterns = dict(cfg.get("rank_pattern") or {})
        self.rslora = bool(cfg.get("use_rslora"))
        parts: dict[str, dict] = {}
        self.full = {}
        with safe_open(str(d / WEIGHTS), framework="pt", device=str(device)) as f:
            for key in f.keys():
                name = canonical(key)
                m = re.match(r"(.+)\.(lora_A|lora_B)(?:\.default)?\.weight$", name) or \
                    re.match(r"(.+)\.(lora_embedding_A|lora_embedding_B)(?:\.default)?$", name)
                if m:
                    parts.setdefault(m.group(1), {})[m.group(2)] = f.get_tensor(key)
                else:
                    self.full[name] = f.get_tensor(key)
        self.pairs = {}
        for module, p in parts.items():
            if set(p) == {"lora_A", "lora_B"}:
                self.pairs[module + ".weight"] = (p["lora_A"], p["lora_B"], False)
            elif set(p) == {"lora_embedding_A", "lora_embedding_B"}:
                self.pairs[module + ".weight"] = (p["lora_embedding_A"], p["lora_embedding_B"], True)
            else:
                raise ValueError(f"{d / WEIGHTS}: {module} has {sorted(p)}")
        self.merge = merge
        self.applied = set()
        self.attached = set()
        self.embedding = None

    def _scaling(self, name: str, r: int) -> float:
        """alpha / r of the adapter's configuration (its patterns first); `r`, the tensors' rank, when it names none. A
        converted adapter may hold LoRAs of lower rank than its r under one alpha: they keep that alpha / r."""
        alpha = next((float(a) for pat, a in self.patterns.items() if re.search(pat, name)), self.alpha)
        r = next((int(x) for pat, x in self.rank_patterns.items() if re.search(pat, name)),
                 int(self.r) if self.r else r)
        return alpha / (r ** 0.5 if self.rslora else r)

    def __len__(self) -> int:
        return len(self.pairs) + len(self.full)

    def apply(self, key: str, t):
        name = canonical(key)
        if name in self.full:
            new = self.full[name]
            if tuple(new.shape) != tuple(t.shape):
                raise ValueError(f"adapter {name}: shape {tuple(new.shape)}, checkpoint {tuple(t.shape)}")
            self.applied.add(name)
            t = new.to(device=t.device, dtype=t.dtype)
        if name in self.pairs:
            A, B, embedding = self.pairs[name]
            if not embedding:
                if (B.shape[0], A.shape[1]) != tuple(t.shape):
                    raise ValueError(f"adapter {name}: B A {(B.shape[0], A.shape[1])}, checkpoint {tuple(t.shape)}")
                self.applied.add(name)
                if not self.merge:
                    return t
                delta = (B.float() @ A.float()).to(t.device) * self._scaling(name, A.shape[0])
                return (t.float() + delta).to(t.dtype)
            delta = (B.float() @ A.float()).T
            self.embedding = t.clone()
            if tuple(delta.shape) != tuple(t.shape):
                raise ValueError(f"adapter {name}: delta {tuple(delta.shape)}, checkpoint {tuple(t.shape)}")
            self.applied.add(name)
            return (t.float() + self._scaling(name, A.shape[0]) * delta.to(t.device)).to(t.dtype)
        return t

    def attach(self, model) -> None:
        """Every projection with a LoRA gets `lora` = (A [r, in], scaling B [out, r]) in its weight's dtype, and a
        forward hook that adds x A^T B^T when the module itself runs."""
        for mname, mod in model.named_modules():
            name = canonical(mname + ".weight")
            if name in self.pairs and not self.pairs[name][2] and not self.merge and isinstance(mod, torch.nn.Linear):
                A, B, _ = self.pairs[name]
                dt = mod.weight.dtype
                mod.lora = (A.to(dt), (B.float() * self._scaling(name, A.shape[0])).to(dt))
                mod.register_forward_hook(_hook)
                self.attached.add(name)

    def check(self) -> None:
        missed = sorted((set(self.pairs) | set(self.full)) - self.applied)
        if missed:
            raise ValueError(f"{len(missed)} adapter tensors found no checkpoint tensor, e.g. {missed[:4]}")
        loose = sorted(k for k, p in self.pairs.items() if not p[2] and not self.merge and k not in self.attached)
        if loose:
            raise ValueError(f"{len(loose)} adapter projections found no linear module, e.g. {loose[:4]}")


class HeadRows:
    """An output head whose LoRA stays unmerged, as the families' heads read it: the rows of the token ids they select
    (`index_select`), merged in float32 (W[ids] + scaling B[ids] A); never the whole vocabulary."""

    def __init__(self, weight, lora):
        self.weight = weight
        self.A, self.B = lora
        self.device = weight.device

    def detach(self) -> "HeadRows":
        return self

    def index_select(self, dim: int, idx):
        if dim != 0:
            raise ValueError("an output head is read by rows")
        return self.weight.index_select(0, idx).float() + self.B.index_select(0, idx).float() @ self.A.float()


def output_head(head):
    """(weight, bias) of an output head module; its weight as HeadRows when the adapter gave it a LoRA."""
    lr = getattr(head, "lora", None)
    return (head.weight if lr is None else HeadRows(head.weight, lr)), getattr(head, "bias", None)


def _hook(mod, args, out):
    A, B = mod.lora
    x = args[0]
    return out + torch.nn.functional.linear(torch.nn.functional.linear(x, A.to(x.dtype)), B.to(x.dtype))


class LowRank:
    """A projection weight (a tensor or an Fp8Weight; the merged weight of projections that read the same input, their
    rows in order) with their LoRA left unmerged, called as its matrix product: x W^T + (x A^T) B^T. A stacks their A
    [sum r, in]; B holds each one's scaled B at its own rows and columns [out, sum r]. Outside the batch-invariant
    mode the LoRA product is accumulated in place onto the weight's, in float32, rounded once."""

    def __init__(self, base, A: torch.Tensor, B: torch.Tensor):
        self.base, self.A, self.B = base, A, B
        self.shape = tuple(base.shape)

    def __call__(self, x: torch.Tensor, prod=None) -> torch.Tensor:
        from . import settings
        from .kernels.fp8 import mm

        if prod is not None or settings.batch_invariant:
            p = prod or mm
            return p(x, self.base).add_(p(p(x, self.A), self.B))
        return mm(x, self.base).addmm_(mm(x, self.A), self.B.T)

    def into(self, x: torch.Tensor, out: torch.Tensor) -> None:
        from . import settings
        from .kernels.fp8 import mm, mm_into

        mm_into(x, self.base, out)
        if settings.batch_invariant:
            out.add_(mm(mm(x, self.A), self.B))
        else:
            out.addmm_(mm(x, self.A), self.B.T)

    def __getitem__(self, rows):
        return LowRank(self.base[rows], self.A, self.B[rows])

    def split(self, n: int):
        top, rest = (self.base[:n], self.base[n:]) if isinstance(self.base, torch.Tensor) else self.base.split(n)
        return LowRank(top, self.A, self.B[:n]), LowRank(rest, self.A, self.B[n:])

    def with_base(self, base) -> "LowRank":
        return LowRank(base, self.A, self.B)


def attached(base, mods):
    """`base`, the weight of `mods` (one projection, or several merged by rows in this order), as a LowRank when any of
    them carries a LoRA; else `base` itself. The modules' `lora` become views of the stacked A and B."""
    if base is None or not any(getattr(m, "lora", None) is not None for m in mods):
        return base
    loras = [m.lora for m in mods if getattr(m, "lora", None) is not None]
    A = torch.cat([a for a, _ in loras])
    out = sum(m.weight.shape[0] for m in mods)
    if out != base.shape[0]:
        raise ValueError(f"LoRA rows {out} against a weight of {base.shape[0]}")
    B = torch.zeros(out, A.shape[0], device=A.device, dtype=A.dtype)
    r0 = c0 = 0
    for m in mods:
        n = m.weight.shape[0]
        if getattr(m, "lora", None) is not None:
            a, b = m.lora
            B[r0: r0 + n, c0: c0 + a.shape[0]] = b
            m.lora = (A[c0: c0 + a.shape[0]], B[r0: r0 + n, c0: c0 + a.shape[0]])
            c0 += a.shape[0]
        r0 += n
    return LowRank(base, A, B)


def load(model_dir: str, device, merge: bool = False) -> Adapter | None:
    return Adapter(model_dir, device, merge) if (Path(model_dir) / SUB / CONFIG).is_file() else None

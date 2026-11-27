"""Flide's side of integrations.vllm_connector: a finished vLLM request's keys and values as a saved session state.

A request that names `vllm_ref` (the `flide_ref` of a vLLM response) has its questions asked after that conversation:
formats.next_token reads the conversation's tokens (`tokens`, a small file, on the thread that plans), and the engine,
on the thread that runs the GPU, copies the request's keys and values from vLLM's blocks into its session cache before
the plan runs (`lend`) and gives the blocks back to vLLM. The plan's prefix then resumes from that entry: only the
questions are computed. A ref can be asked about again while its entry stays in the cache; once evicted, the
conversation is computed whole.

`flide serve --vllm-kv DIR` (or FLIDE_VLLM_KV) names the connector's directory. `VllmKV(dir)` opens the KV cache
memory the connector exported from the vLLM process on the same GPU. vLLM's layout per layer is [blocks, key heads,
block size, 2 * head dim], keys then values on the last axis (its FlashAttention backend), every layer a view of one
allocation; keys are stored after RoPE, as Flide keeps them. The answers read vLLM's keys and values, so they follow
vLLM's rounding of the conversation, not Flide's.
"""
from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path

ENV = "FLIDE_VLLM_KV"
KEEP_LENT = 4096   # tokens of lent requests kept for questions asked again, newest first


def directory() -> Path:
    """The connector's directory, from FLIDE_VLLM_KV."""
    d = os.environ.get(ENV)
    if not d:
        raise ValueError("vllm_ref needs the vLLM connector's directory (flide serve --vllm-kv DIR)")
    return Path(d)


def _record(dir: Path, ref: str) -> dict:
    """The connector's record of request `ref`, before or after it was lent."""
    if not re.fullmatch(r"[A-Za-z0-9._:-]+", ref):
        raise ValueError(f"vllm_ref {ref!r}: not a request id")
    for sub in ("refs", "lent"):
        path = dir / sub / f"{ref}.json"
        if path.is_file():
            return json.loads(path.read_text())
    raise ValueError(f"vllm_ref {ref}: no such vLLM request (it needs \"kv_transfer_params\": {{\"flide\": true}})")


def tokens(ref: str, dir: Path | None = None) -> list[int]:
    """The conversation of vLLM request `ref`: its prompt and reply tokens, the last sampled one included."""
    r = _record(dir or directory(), ref)
    return r["tokens"] + r["rest"]


class VllmKV:
    """The KV cache of a vLLM process on this GPU, by its connector's directory (`kv.pt`, `refs/`, `done/`)."""

    def __init__(self, dir: str | Path, views: dict | None = None, block_size: int | None = None):
        import torch

        self.dir = Path(dir)
        (self.dir / "lent").mkdir(exist_ok=True)
        if views is None:
            with (self.dir / "kv.pt").open("rb") as f:
                info = pickle.load(f)
            rebuild, args = info["handle"]
            base = rebuild(*args)
            views = {}
            for name, lay in info["layers"].items():
                dt = getattr(torch, lay["dtype"])
                elem = torch.empty(0, dtype=dt).element_size()
                views[name] = torch.as_strided(base.view(dt), lay["shape"], lay["stride"], lay["offset"] // elem)
            block_size = info["block_size"]
            self._base = base
        self.block_size = int(block_size)
        self.views = {}
        for name, v in views.items():
            m = re.search(r"layers\.(\d+)\.", name)
            if m is None:
                raise ValueError(f"{name}: not a decoder layer's cache")
            if v.dim() != 4 or v.shape[2] != self.block_size or v.shape[3] % 2:
                raise ValueError(f"{name}: layout {tuple(v.shape)} is not [blocks, heads, {self.block_size}, 2 * dim]")
            self.views[int(m[1])] = v

    def ref(self, ref: str) -> dict:
        """The request's tokens with keys (`tokens`), the tokens after them (`rest`) and its block ids."""
        return _record(self.dir, ref)

    def held(self, ref: str) -> bool:
        """Whether vLLM still holds the request's blocks for Flide (not lent yet)."""
        return (self.dir / "refs" / f"{ref}.json").is_file()

    def kv(self, ref: dict) -> dict:
        """Per layer index, (K, V) of the request's `tokens`, each [tokens, key heads, head dim]: one gather per layer out
        of the blocks (each token's block and slot), K and V the two halves of it (views; `lend` writes them into the
        session cache's rows)."""
        import torch

        n = len(ref["tokens"])
        if not n or not self.views:
            return {}
        device = next(iter(self.views.values())).device
        pos = torch.arange(n)
        block = torch.tensor(ref["blocks"], dtype=torch.long)[pos // self.block_size].to(device)
        slot = (pos % self.block_size).to(device)
        out = {}
        for layer, v in self.views.items():
            x = v[block, :, slot]   # [tokens, key heads, 2 * head dim]
            d = v.shape[3] // 2
            out[layer] = (x[..., :d], x[..., d:])
        return out

    def release(self, ref: str) -> None:
        """vLLM may free the request's blocks (at its next step); its record moves to lent/, the newest kept. The copies
        out of the blocks queued on this thread's stream finish first."""
        import torch

        view = next(iter(self.views.values()), None)
        if view is not None and view.is_cuda:
            torch.cuda.current_stream(view.device).synchronize()
        held = self.dir / "refs" / f"{ref}.json"
        if held.is_file():
            held.replace(self.dir / "lent" / held.name)
        (self.dir / "done" / ref).touch()
        lent = sorted(self.dir.joinpath("lent").glob("*.json"), key=lambda p: p.stat().st_mtime)
        for p in lent[:-KEEP_LENT]:
            p.unlink(missing_ok=True)


_source: VllmKV | None = None


def source() -> VllmKV:
    """The VllmKV of FLIDE_VLLM_KV, opened once per process (on the GPU thread)."""
    global _source
    if _source is None:
        _source = VllmKV(directory())
    return _source


def lend(engine, src: VllmKV, ref: str) -> list[int]:
    """The request `ref`'s keys and values as an entry of `engine`'s session cache (its rows in the cache's store when
    it has one), and its blocks given back to vLLM; nothing to do for a ref already lent. Returns the conversation's
    tokens, the last sampled one included."""
    from ..runtime.session import Entry

    sessions = getattr(engine, "sessions", None)
    r = src.ref(ref)
    if sessions is None:
        src.release(ref)
        raise ValueError("the session cache is off (--session-cache-gb)")
    if not src.held(ref):
        return r["tokens"] + r["rest"]
    n = len(r["tokens"])
    try:
        kv = src.kv(r)
        if sessions.store is not None:
            rows = sessions.reserve(engine.backbone.session_rows(n))
            if rows is None:
                raise ValueError(f"vllm_ref {ref}: {n} tokens do not fit the session cache")
            for layer, (k, v) in kv.items():
                rows.write(layer, k, v)
            kv = rows
        else:   # the entry holds the tensors themselves
            kv = {layer: (k.contiguous(), v.contiguous()) for layer, (k, v) in kv.items()}
    finally:
        src.release(ref)
    sessions.insert(Entry(r["tokens"], {n: {}}, kv))   # over the cache's budget it is dropped, and the request recomputes
    return r["tokens"] + r["rest"]


def decide(engine, src: VllmKV, ref: str, request, **options):
    """`engine.decide(request)` with the request's questions asked after the conversation of vLLM request `ref`, its
    keys and values lent to the session cache first (`lend`)."""
    request.context_tokens = lend(engine, src, ref)
    return engine.decide(request, **options)

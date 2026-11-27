"""A vLLM KV connector that lends a finished request's keys and values to a Flide process on the same GPU.

vLLM loads it with

    vllm serve MODEL --kv-transfer-config '{"kv_connector": "FlideConnector",
        "kv_connector_module_path": "flide.integrations.vllm_connector", "kv_role": "kv_both",
        "kv_connector_extra_config": {"dir": "/path/shared/with/flide"}}'

A request opts in with `"kv_transfer_params": {"flide": true}` in its body. When it finishes, its blocks are kept
instead of freed, its tokens and block ids go to `dir/refs/<id>.json`, and its response carries
`"kv_transfer_params": {"flide_ref": "<id>"}`; Flide reads the blocks (integrations.vllm_kv) and writes
`dir/done/<id>` when it no longer needs them, and the blocks go back to vLLM at its next step. At start the worker
exports the KV cache's memory once by CUDA IPC to `dir/kv.pt`, with each layer's view of it.

Requests without the flag are untouched. No keys or values are copied on vLLM's side, and nothing runs per layer, so
CUDA graphs and the scheduler run as they do without a connector.
"""
from __future__ import annotations

import json
import os
import pickle
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (KVConnectorBase_V1, KVConnectorMetadata,
                                                               KVConnectorRole)

FLAG, REF = "flide", "flide_ref"


@dataclass
class FlideMetadata(KVConnectorMetadata):
    """Scheduler to worker, once a step: the requests whose blocks the scheduler kept since the last step."""

    held: list[str] = field(default_factory=list)


class FlideConnector(KVConnectorBase_V1):
    """Keeps the blocks of flagged requests after they finish until Flide releases them (module docstring)."""

    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        default = Path(tempfile.gettempdir()) / "flide_vllm"
        self.dir = Path(extra.get("dir") or os.environ.get("FLIDE_VLLM_DIR") or default)
        (self.dir / "refs").mkdir(parents=True, exist_ok=True)
        (self.dir / "done").mkdir(parents=True, exist_ok=True)
        self.block_size = vllm_config.cache_config.block_size
        self._new_held: list[str] = []   # scheduler: kept since the last metadata
        self._held: set[str] = set()     # worker: kept and not yet returned
        self._finished: set[str] = set()  # worker: ids get_finished has been given

    # ---- scheduler side ----------------------------------------------------------------------------------------

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int):
        """Nothing is loaded from outside vLLM."""
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int) -> None:
        """Nothing to track at allocation."""

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        """The requests kept since the last step, for the worker to watch for Flide's release."""
        meta, self._new_held = FlideMetadata(held=self._new_held), []
        return meta

    def request_finished(self, request, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        """A flagged request keeps its blocks: its computed tokens and block ids go to refs/<id>.json and its
        response names it. The last sampled token has no keys yet, so it is listed apart (`rest`)."""
        params = request.kv_transfer_params or {}
        if not params.get(FLAG):
            return False, None
        n = min(request.num_computed_tokens, len(block_ids) * self.block_size)
        tokens = list(request.all_token_ids)
        ref = {"id": request.request_id, "tokens": tokens[:n], "rest": tokens[n:], "blocks": list(block_ids),
               "block_size": self.block_size}
        path = self.dir / "refs" / f"{request.request_id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ref))
        tmp.replace(path)   # whole or not at all for a reader
        self._new_held.append(request.request_id)
        return True, {REF: request.request_id}

    # ---- worker side -------------------------------------------------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Exports the memory under every layer's cache by CUDA IPC, with each layer's view of it, to kv.pt."""
        from torch.multiprocessing.reductions import reduce_tensor

        first = next(iter(kv_caches.values()))
        storage = first.untyped_storage()
        base = torch.empty(0, dtype=torch.uint8, device=first.device).set_(storage)
        layers = {}
        for name, t in kv_caches.items():
            if t.untyped_storage().data_ptr() != storage.data_ptr():
                raise ValueError(f"{name}: its cache is not in the shared allocation")
            layers[name] = {"offset": t.storage_offset() * t.element_size(), "shape": list(t.shape),
                            "stride": list(t.stride()), "dtype": str(t.dtype).removeprefix("torch.")}
        info = {"handle": reduce_tensor(base), "device": first.device.index, "nbytes": storage.nbytes(),
                "layers": layers, "block_size": self.block_size, "pid": os.getpid()}
        path = self.dir / "kv.pt"
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(info, f)
        tmp.replace(path)
        self._base = base   # held for the process's life, so the exported memory stays valid

    def start_load_kv(self, forward_context, **kwargs) -> None:
        """Nothing is loaded."""

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Nothing is loaded."""

    def save_kv_layer(self, layer_name: str, kv_layer, attn_metadata, **kwargs) -> None:
        """Nothing is copied: Flide reads the blocks in place."""

    def wait_for_save(self) -> None:
        """Nothing is copied."""

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        """The kept requests Flide has released (done/<id>), which the scheduler then frees."""
        if self.has_connector_metadata():
            meta = self._get_connector_metadata()
            if isinstance(meta, FlideMetadata):
                self._held.update(meta.held)
        self._finished.update(finished_req_ids)
        done = {r for r in self._held & self._finished if (self.dir / "done" / r).exists()}
        for r in done:
            self._held.discard(r)
            self._finished.discard(r)
            for p in (self.dir / "done" / r, self.dir / "refs" / f"{r}.json"):
                p.unlink(missing_ok=True)
        return (done or None), None

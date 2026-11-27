"""Read-only allocator and graph diagnostics. Never synchronize, collect, or empty caches."""
from __future__ import annotations

import time
from collections import Counter


def diagnostics(engine, *, memory_details: bool = False) -> dict:
    import torch

    backbone = engine.backbone
    # Calling request_graphs() here could lazily allocate a manager and its buffers.
    # The engine records the already-created manager during initialization instead.
    graphs = getattr(engine, "_diagnostic_graphs", None)
    captured = getattr(graphs, "graphs", {}) if graphs is not None else {}
    shapes = tuple(captured)
    pool = getattr(graphs, "pool", None)
    graph_stats = {
        "count": len(shapes),
        "by_kind": dict(Counter(str(s[0]) if isinstance(s, tuple) else "other" for s in shapes)),
        "failed_count": len(getattr(graphs, "failed", ())),
        "pool_id": list(pool) if pool is not None else None,
    }
    if memory_details:
        graph_stats["shapes"] = [list(s) if isinstance(s, tuple) else str(s) for s in shapes]
    device = torch.device(backbone.device)
    memory = {"device": str(device), "sampled_at": time.time(), "available": False}
    result = {"memory": memory, "cuda_graphs": graph_stats}
    if device.type != "cuda" or not torch.cuda.is_initialized():
        return result
    index = device.index if device.index is not None else torch.cuda.current_device()
    stats = torch.cuda.memory_stats(index)
    free, total = torch.cuda.mem_get_info(index)
    allocated = stats.get("allocated_bytes.all.current", 0)
    reserved = stats.get("reserved_bytes.all.current", 0)
    memory.update(
        available=True, allocated_bytes=allocated, reserved_bytes=reserved,
        reserved_unallocated_bytes=reserved - allocated,
        active_bytes=stats.get("active_bytes.all.current", 0),
        inactive_split_bytes=stats.get("inactive_split_bytes.all.current", 0),
        peak_allocated_bytes=stats.get("allocated_bytes.all.peak", 0),
        peak_reserved_bytes=stats.get("reserved_bytes.all.peak", 0),
        allocation_retries=stats.get("num_alloc_retries", 0),
        oom_count=stats.get("num_ooms", 0),
        device_free_bytes=free, device_total_bytes=total, device_used_bytes=total - free,
        allocator_backend=torch.cuda.get_allocator_backend(),
        allocator_limit_bytes=int(total * torch.cuda.get_per_process_memory_fraction(index)),
    )
    if memory_details and pool is not None:
        # Pool-filtered, trace-free snapshots are supported by recent PyTorch. Older
        # releases can still filter their public full snapshot by segment_pool_id.
        try:
            segments = torch.cuda.memory_snapshot(mempool_id=pool, include_traces=False)
        except TypeError:
            segments = torch.cuda.memory_snapshot()
        segments = [s for s in segments if s["device"] == index
                    and tuple(s.get("segment_pool_id", ())) == tuple(pool)]
        pool_reserved = sum(s["total_size"] for s in segments)
        pool_allocated = sum(s["allocated_size"] for s in segments)
        pool_active = sum(s["active_size"] for s in segments)
        graph_stats["pool_memory"] = {
            "source": "allocator_snapshot", "segments": len(segments),
            "reserved_bytes": pool_reserved, "allocated_bytes": pool_allocated,
            "active_bytes": pool_active, "inactive_bytes": pool_reserved - pool_active,
        }
    return result

"""Tensor parallelism over the GPUs of one machine.

Every rank holds the whole model except the decoder's projections, which are cut by heads (attention and Gated
DeltaNet) and by columns (MLP): the products that read a layer's input keep their rank's output rows, the products
that write the residual stream (MLP down, Gated DeltaNet out, attention o) keep the matching input columns and add
their partial sums across ranks (one all-reduce each, inside the CUDA graphs too). Everything else (embeddings,
norms, the vision tower, the head) is the same on every rank, so every rank computes the same residual stream.

Rank 0 is the engine. Its backbone and request graphs are wrapped so that every call that runs the model (forward,
graph replays, captures) is first sent to the other ranks, which make the same call on their own shard with the same
arguments, then run on rank 0; the collectives inside meet in the same order. A packed forward is sent as its jobs
(each rank packs them again). Calls are serialized by a lock, so threads cannot reorder them between ranks.
"""
from __future__ import annotations

import logging
import socket
import threading
import traceback

import torch

log = logging.getLogger("flide.tp")

MIRRORED_GRAPH_CALLS = {"run", "run_flat", "run_sflat", "run_media", "run_session", "capture", "capture_all",
                        "capture_flat", "capture_sessions", "capture_sflat", "reset", "restore", "extend"}


def world() -> int:
    import torch.distributed as dist

    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def capture_mode() -> str:
    """CUDA graph capture mode: NCCL's watchdog thread queries events while a graph with collectives is captured."""
    return "thread_local" if world() > 1 else "global"


class RowParallel:
    """A product whose input columns are split across ranks: this rank's partial sum, added across ranks."""

    def __init__(self, w):
        self.w = w
        self.shape = w.shape

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        from ..kernels.fp8 import mm

        y = mm(x, self.w)
        dist.all_reduce(y)
        return y

    def into(self, x: torch.Tensor, out: torch.Tensor) -> None:
        import torch.distributed as dist

        from ..kernels.fp8 import mm_into

        if not out.is_contiguous():
            out.copy_(self(x))
            return
        mm_into(x, self.w, out)
        dist.all_reduce(out)


def split(n: int, rank: int, size: int) -> slice:
    if n % size:
        raise ValueError(f"{n} heads or columns do not split over {size} GPUs")
    return slice(rank * n // size, (rank + 1) * n // size)


class _Mirror:
    """Rank 0's object (backbone or request graphs), with the calls in `calls` sent to the other ranks first."""

    def __init__(self, local, target: str, calls, link: "_Link"):
        object.__setattr__(self, "_local", local)
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_calls", calls)
        object.__setattr__(self, "_link", link)

    def __getattr__(self, name):
        attr = getattr(self._local, name)
        if name not in self._calls or not callable(attr):
            return attr
        link, target = self._link, self._target

        def call(*args, **kwargs):
            with link.lock:
                link.send(target, name, args, kwargs)
                return attr(*args, **kwargs)
        return call

    def __setattr__(self, name, value):
        setattr(self._local, name, value)


class TPBackbone(_Mirror):
    def __init__(self, local, link: "_Link"):
        super().__init__(local, "bb", (), link)
        object.__setattr__(self, "_graphs", None)

    def forward(self, b, pad_to=None):
        with self._link.lock:
            self._link.send("bb", "forward", (b.jobs,), {"pad_to": pad_to})
            return self._local.forward(b, pad_to=pad_to)

    def request_graphs(self):
        if self._graphs is None:
            with self._link.lock:
                self._link.send("bb", "request_graphs", (), {})
                g = self._local.request_graphs()
            object.__setattr__(self, "_graphs", None if g is None else _Mirror(g, "graphs", MIRRORED_GRAPH_CALLS,
                                                                                self._link))
        return self._graphs

    def close(self) -> None:
        self._link.close()


class _Link:
    def __init__(self, conns, procs):
        self.conns = conns
        self.procs = procs
        self.lock = threading.RLock()

    def send(self, target, name, args, kwargs) -> None:
        for c in self.conns:
            c.send((target, name, args, kwargs))

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(None)
            except (OSError, BrokenPipeError):
                pass
        for p in self.procs:
            p.join(timeout=30)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init(rank: int, size: int, port: int, device: torch.device) -> None:
    import torch.distributed as dist

    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=size,
                            device_id=device)


def _worker(rank, size, port, model_type, model_dir, device, dtype, options, conn):
    from .. import backbones
    from ..kernels import launcher
    from .kcache import seed, tune
    from .pack import pack

    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s tp{rank} %(name)s %(message)s")
    launcher.install()
    seed()
    try:
        device = torch.device(device)
        _init(rank, size, port, device)
        bb = backbones.get(model_type).load(model_dir, device=str(device), dtype=dtype, tp=(rank, size), **options)
        tune()
        conn.send("ready")
    except Exception:
        conn.send(("error", traceback.format_exc()))
        raise
    objs = {"bb": bb}
    with torch.inference_mode():
        while True:
            msg = conn.recv()
            if msg is None:
                break
            target, name, args, kwargs = msg
            try:
                if target == "bb" and name == "forward":
                    bb.forward(pack(args[0], bb.device), **kwargs)
                else:
                    out = getattr(objs[target], name)(*args, **kwargs)
                    if name == "request_graphs":
                        objs["graphs"] = out
            except Exception:
                log.error("tensor parallel rank %d failed in %s.%s:\n%s", rank, target, name, traceback.format_exc())
                raise


def launch(model_type: str, model_dir: str, device: str, dtype: str, size: int, options: dict) -> TPBackbone:
    """Rank 0's backbone on `device` and size - 1 worker processes on the next GPUs, every one holding its shard."""
    import torch.multiprocessing as mp

    from .. import backbones

    dev0 = torch.device(device)
    first = dev0.index or 0
    if first + size > torch.cuda.device_count():
        raise ValueError(f"tensor parallel over {size} GPUs from cuda:{first}: only {torch.cuda.device_count()} visible")
    port = _free_port()
    ctx = mp.get_context("spawn")
    conns, procs = [], []
    for r in range(1, size):
        a, b = ctx.Pipe()
        p = ctx.Process(target=_worker, args=(r, size, port, model_type, model_dir, f"cuda:{first + r}", dtype, options,
                                             b), daemon=True)
        p.start()
        conns.append(a)
        procs.append(p)
    _init(0, size, port, dev0)
    local = backbones.get(model_type).load(model_dir, device=device, dtype=dtype, tp=(0, size), **options)
    for r, c in enumerate(conns, 1):
        msg = c.recv()
        if msg != "ready":
            raise RuntimeError(f"tensor parallel rank {r} did not start:\n{msg[1] if isinstance(msg, tuple) else msg}")
    log.info("tensor parallel over %d GPUs (cuda:%d to cuda:%d)", size, first, first + size - 1)
    return TPBackbone(local, _Link(conns, procs))

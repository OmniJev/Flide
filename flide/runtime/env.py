"""Startup checks that turn the usual CUDA setup failures into one line the user can act on."""
from __future__ import annotations

import os
import re
from importlib.metadata import PackageNotFoundError, requires
from pathlib import Path

TORCH_INDEX = "https://download.pytorch.org/whl/"

GIB = 2 ** 30
WORKER_BYTES = GIB   # memory one planner process may take (about 0.55 GiB resident, measured on Qwen3.5)


def _cgroup_dirs(cgroup_root: str = "/sys/fs/cgroup", proc: str = "/proc/self/cgroup") -> list[Path]:
    """This process's cgroup v2 directory and its parents up to the root, innermost first; [] without cgroup v2."""
    root = Path(cgroup_root)
    try:
        rel = next(line[3:].strip() for line in Path(proc).read_text().splitlines() if line.startswith("0::"))
    except (OSError, StopIteration):
        return []
    d = root / rel.lstrip("/")
    dirs = []
    while True:
        dirs.append(d)
        if d == root or root not in d.parents:
            break
        d = d.parent
    return [x for x in dirs if x.is_dir()]


def _read(p: Path) -> str | None:
    try:
        return p.read_text().strip()
    except OSError:
        return None


def usable_cpus(cgroup_root: str = "/sys/fs/cgroup", proc: str = "/proc/self/cgroup") -> int:
    """CPUs this process may run on: its affinity mask, cut to the tightest cgroup CPU quota (v2 cpu.max or v1
    cfs_quota_us), so a container that sees the host's cores counts only what it was given."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    quotas = []
    for d in _cgroup_dirs(cgroup_root, proc):
        parts = (_read(d / "cpu.max") or "").split()
        if len(parts) == 2 and parts[0] != "max":
            quotas.append(int(parts[0]) / int(parts[1]))
    v1 = Path(cgroup_root) / "cpu"
    q, period = _read(v1 / "cpu.cfs_quota_us"), _read(v1 / "cpu.cfs_period_us")
    if q and period and q.lstrip("-").isdigit() and int(q) > 0 and int(period) > 0:
        quotas.append(int(q) / int(period))
    if quotas:
        n = min(n, max(1, int(min(quotas))))
    return max(1, n)


def free_memory(cgroup_root: str = "/sys/fs/cgroup", proc: str = "/proc/self/cgroup",
                meminfo: str = "/proc/meminfo") -> int | None:
    """Bytes this process can still take: MemAvailable, cut to the room left under the tightest cgroup memory limit
    (v2 memory.max less memory.current, or v1 limit_in_bytes less usage_in_bytes); None when nothing says."""
    room = []
    for line in (_read(Path(meminfo)) or "").splitlines():
        if line.startswith("MemAvailable:"):
            room.append(int(line.split()[1]) * 1024)
    for d in _cgroup_dirs(cgroup_root, proc):
        limit, used = _read(d / "memory.max"), _read(d / "memory.current")
        if limit and limit.isdigit() and used and used.isdigit():
            room.append(int(limit) - int(used))
    v1 = Path(cgroup_root) / "memory"
    limit, used = _read(v1 / "memory.limit_in_bytes"), _read(v1 / "memory.usage_in_bytes")
    if limit and limit.isdigit() and used and used.isdigit() and int(limit) < 2 ** 60:
        room.append(int(limit) - int(used))
    return max(0, min(room)) if room else None


def default_workers(cpus: int | None = None, free: int | None = -1) -> int:
    """Planner processes for `run` when the caller names none: the usable CPUs less two (one for the GPU loop, one
    spare), and no more than the free memory holds at `WORKER_BYTES` each."""
    cpus = usable_cpus() if cpus is None else cpus
    free = free_memory() if free == -1 else free
    n = cpus - 2
    if free is not None:
        n = min(n, free // WORKER_BYTES)
    return max(1, int(n))


def driver_version() -> str | None:
    """The loaded NVIDIA kernel driver's version, or None when no driver is loaded."""
    p = Path("/proc/driver/nvidia/version")
    if not p.is_file():
        return None
    for word in p.read_text().split():
        parts = word.split(".")
        if len(parts) >= 2 and all(x.isdigit() for x in parts):
            return word
    return "unknown"


def driver_cuda() -> tuple[int, int] | None:
    """The newest CUDA version the loaded driver runs (cuDriverGetVersion: 12090 -> (12, 9)), or None."""
    import ctypes

    try:
        v = ctypes.c_int()
        if ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(v)) != 0:
            return None
    except (OSError, AttributeError):
        return None
    return v.value // 1000, v.value % 1000 // 10


def torch_install(have: tuple[int, int] | None) -> str:
    """The command that installs a CUDA build of PyTorch the driver runs (cu132 or cu130 on a CUDA 13 driver, else
    cu126), within the torch range this package requires, so the index's newest torch cannot fall outside it."""
    tag = "cu132" if have is not None and have >= (13, 2) else "cu130" if have is not None and have >= (13, 0) else "cu126"
    try:
        deps = requires(__name__.split(".")[0]) or []
    except PackageNotFoundError:
        deps = []
    spec = next((d.split(";")[0].replace(" ", "") for d in deps if re.match(r"torch(?![\w.-])", d)), "torch")
    return f'pip install --force-reinstall "{spec}" --index-url {TORCH_INDEX}{tag}'


def check_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        return
    drv = driver_version()
    if drv is None:
        raise RuntimeError("No NVIDIA driver is loaded, and flide's PyTorch path runs on NVIDIA GPUs.")
    if torch.version.cuda is None:
        raise RuntimeError(f"PyTorch {torch.__version__} is a CPU build. Install a CUDA build: "
                           f"{torch_install(driver_cuda())}")
    need = int(torch.version.cuda.split(".")[0])
    have = driver_cuda()
    if have is not None and have[0] < need:   # within a major version, newer builds run on older drivers
        fix = (f"Install the CUDA 12.6 build, which this driver runs: {torch_install(have)}" if have[0] == 12
               else "Update the NVIDIA driver to 525 or newer.")
        raise RuntimeError(f"PyTorch {torch.__version__} is built for CUDA {torch.version.cuda}, and this machine's "
                           f"NVIDIA driver ({drv}) runs CUDA up to {have[0]}.{have[1]}. {fix}")
    try:
        torch.cuda.init()
        reason = "no GPU is visible to this process"
    except Exception as e:
        reason = str(e).strip().splitlines()[0]
    runs = f" although the NVIDIA driver ({drv}) runs its CUDA {torch.version.cuda}" if have else ""
    raise RuntimeError(f"PyTorch {torch.__version__} cannot use the GPU{runs}: {reason}")

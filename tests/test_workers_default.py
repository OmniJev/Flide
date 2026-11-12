"""The default number of planning workers for `run`, from fake cgroup and meminfo files, without a GPU.

    python tests/test_workers_default.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flide.runtime.env import GIB, default_workers, free_memory, usable_cpus


def fake(cpu_max=None, mem_max=None, mem_current=None, mem_available_gib=200, inner=None, v1=None):
    """A cgroup root, a /proc/self/cgroup and a meminfo in a temporary directory; `inner` puts the process one level
    down with its own cpu.max; `v1` writes cgroup v1 files {name: text} instead."""
    d = Path(tempfile.mkdtemp())
    root = d / "cg"
    root.mkdir()
    rel = "/"
    for name, text in (("cpu.max", cpu_max), ("memory.max", mem_max), ("memory.current", mem_current)):
        if text is not None:
            (root / name).write_text(text + "\n")
    if inner is not None:
        (root / "job").mkdir()
        (root / "job" / "cpu.max").write_text(inner + "\n")
        rel = "/job"
    for name, text in (v1 or {}).items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text + "\n")
    (d / "cgroup").write_text(f"0::{rel}\n")
    (d / "meminfo").write_text(f"MemTotal: 999999999 kB\nMemAvailable: {int(mem_available_gib * 2 ** 20)} kB\n")
    return str(root), str(d / "cgroup"), str(d / "meminfo")


def test_container_quota():
    # a pod that sees 96 host cores and is given 10.2 CPUs and 31 GB: 10 CPUs, 8 workers
    root, proc, meminfo = fake(cpu_max="1020000 100000", mem_max=str(31 * 10 ** 9), mem_current=str(2 * GIB))
    assert usable_cpus(root, proc) == min(10, usable_cpus("/nonexistent", proc))
    free = free_memory(root, proc, meminfo)
    assert free == 31 * 10 ** 9 - 2 * GIB, free
    assert default_workers(10, free) == 8


def test_no_quota():
    root, proc, meminfo = fake(cpu_max="max 100000", mem_max="max", mem_current=str(GIB))
    assert usable_cpus(root, proc) == usable_cpus("/nonexistent", proc)
    assert free_memory(root, proc, meminfo) == 200 * GIB


def test_parent_quota_counts():
    root, proc, _ = fake(cpu_max="400000 100000", inner="max 100000")
    assert usable_cpus(root, proc) <= 4


def test_cgroup_v1():
    root, proc, meminfo = fake(v1={"cpu/cpu.cfs_quota_us": "600000", "cpu/cpu.cfs_period_us": "100000",
                                   "memory/memory.limit_in_bytes": str(16 * GIB),
                                   "memory/memory.usage_in_bytes": str(4 * GIB)})
    assert usable_cpus(root, proc) <= 6
    assert free_memory(root, proc, meminfo) == 12 * GIB


def test_memory_bounds_workers():
    assert default_workers(224, 5 * GIB + 1) == 5      # many CPUs, little memory
    assert default_workers(24, 500 * GIB) == 22        # CPUs less two
    assert default_workers(2, 500 * GIB) == 1          # never below one
    assert default_workers(64, 0) == 1
    assert default_workers(16, None) == 14             # memory unknown: CPUs decide


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

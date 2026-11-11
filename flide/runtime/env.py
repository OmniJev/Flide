"""Startup checks that turn the usual CUDA setup failures into one line the user can act on."""
from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, requires
from pathlib import Path

TORCH_INDEX = "https://download.pytorch.org/whl/"


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

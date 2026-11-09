"""Startup checks that turn the usual CUDA setup failures into one line the user can act on."""
from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, requires
from pathlib import Path
from shlex import quote

TORCH_WHEEL_INDEX = "https://download.pytorch.org/whl"


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


def _install_hint(have: tuple[int, int] | None) -> str:
    have = have or (12, 0)
    if have < (12, 0):
        return "Update the NVIDIA driver to 525 or newer."
    major, minor = (12, 6)
    if have >= (13, 0):
        major, minor = (13, 2) if have >= (13, 2) else (13, 0)
    try:
        deps = requires("flide") or ()
    except PackageNotFoundError:
        deps = ()
    torch_req = next((dep for dep in deps if re.match(r"torch(?![\w.-])", dep, re.IGNORECASE)), None)
    index = f"{TORCH_WHEEL_INDEX}/cu{major}{minor}"
    hint = f"Install the CUDA {major}.{minor} build"
    if torch_req:
        hint += f": pip install --force-reinstall {quote(torch_req)} --index-url {index}"
    else:
        hint += f" from {index}"
    if major == 13:
        hint += f". For older GPUs, use the CUDA 12.6 index at {TORCH_WHEEL_INDEX}/cu126."
    return hint


def check_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        return
    drv = driver_version()
    if drv is None:
        raise RuntimeError("No NVIDIA driver is loaded, and flide's PyTorch path runs on NVIDIA GPUs.")
    have = driver_cuda()
    if torch.version.cuda is None:
        raise RuntimeError(f"PyTorch {torch.__version__} is a CPU build. {_install_hint(have)}")
    need = int(torch.version.cuda.split(".")[0])
    if have is not None and have[0] < need:
        raise RuntimeError(f"PyTorch {torch.__version__} is built for CUDA {torch.version.cuda}, and this machine's "
                           f"NVIDIA driver ({drv}) reports CUDA {have[0]}.{have[1]}. {_install_hint(have)}")
    try:
        torch.cuda.init()
        reason = "no GPU is visible to this process"
    except Exception as e:
        reason = str(e).strip().splitlines()[0]
    driver = f" (NVIDIA driver {drv} reports CUDA {have[0]}.{have[1]})" if have else ""
    raise RuntimeError(f"PyTorch {torch.__version__} cannot use the GPU{driver}: {reason}")

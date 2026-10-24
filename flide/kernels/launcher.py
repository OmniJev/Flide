"""Triton without a C compiler.

From Triton 3.7 on, the only C code Triton compiles at run time is its CUDA driver module (driver.c, one
source per Triton release; GPU code it compiles with its own bundled compiler). flide ships that
module prebuilt for the Triton releases it supports, every CPython from 3.10 (free-threaded builds too),
x86_64 and aarch64, built in manylinux_2_28 containers. `install()` points Triton's
build hook at those files. A combination they do not cover falls back to Triton's own compiler call; with
no compiler on the machine it fails with a message that says what to install.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import sysconfig
from pathlib import Path

STORE = Path(__file__).parent / "prebuilt" / "cuda_utils"
_installed = False


def _compiler() -> str | None:
    return os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")


def prebuilt_path(name: str, source: bytes) -> Path:
    ext = sysconfig.get_config_var("EXT_SUFFIX")
    return STORE / hashlib.sha256(source).hexdigest() / f"{name}{ext}"


def _build(name, src, srcdir, library_dirs, include_dirs, libraries):
    from triton import knobs
    from triton.runtime import build

    ext = sysconfig.get_config_var("EXT_SUFFIX")
    pre = prebuilt_path(name, Path(src).read_bytes())
    if pre.is_file():
        dst = Path(srcdir) / f"{name}{ext}"
        shutil.copyfile(pre, dst)
        return str(dst)
    if _compiler():
        knobs.build.impl = None
        try:
            return build._build(name, src, srcdir, library_dirs, include_dirs, libraries, [])
        finally:
            knobs.build.impl = _build
    import triton

    raise RuntimeError(
        f"flide ships no prebuilt {name} for Triton {triton.__version__} on Python "
        f"{sys.version.split()[0]} ({ext}), and no C compiler (gcc or clang) is installed. Install a compiler "
        f"and the Python headers, or a Triton release flide ships for (3.7, 3.8).")


def install() -> None:
    """Route Triton's run-time C builds through the prebuilt modules. Safe to call more than once; leaves a
    build hook someone else installed alone."""
    global _installed
    if _installed:
        return
    _installed = True
    try:
        from triton import knobs
    except Exception:
        return
    if getattr(knobs.build, "impl", None) is None:
        knobs.build.impl = _build

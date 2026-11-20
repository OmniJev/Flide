"""Automatic CuTe producer/quantization fusion with ordinary-tensor fallback.

The pair (FP8 activation, per-row FP32 scale) is consumed by grouped_fp8 without
requantization. CUDA execution errors propagate. No model/checkpoint names are used.
"""
import logging
from functools import lru_cache
import torch
from .. import settings

log = logging.getLogger(__name__)
_logged = set()


def _eligible(x, kind):
    return (settings.fp8_fusion in ("auto", kind) and not settings.batch_invariant
            and x.is_cuda and x.ndim == 2 and x.dtype == torch.bfloat16
            and x.shape[0] > 0 and x.stride(1) == 1 and x.data_ptr() % 16 == 0
            and x.stride(0) % 8 == 0
            and torch.cuda.get_device_capability(x.device) == (12, 0))


def _selected(kind):
    if kind not in _logged:
        log.info("using CuTe FP8 fusion: %s", kind)
        _logged.add(kind)


@lru_cache(maxsize=1)
def _load():
    # Cache both successful imports and missing/incompatible optional dependencies.
    try:
        from .cute_swiglu_quant import swiglu_quant as run
        return run, None
    except (ImportError, AttributeError) as exc:
        return None, exc


def _backend():
    run, error = _load()
    if run is None:
        if settings.fp8_fusion == "swiglu":
            raise RuntimeError("CuTe SwiGLU dependencies are unavailable; install '.[cute]' "
                               "or use FLIDE_FP8_FUSION=auto/off") from error
        if "unavailable" not in _logged:
            log.info("CuTe SwiGLU unavailable; using original activation/quantization: %s", error)
            _logged.add("unavailable")
    return run


def _threads_per_row(n):
    # Bound register use; wide rows share the maximum through shared memory.
    # The upper bound limits the sizes qualified by the current kernel.
    if n < 32 or n > 32768 or n % 32:
        return None
    if n <= 512:
        return 32
    if n <= 4096:
        return 128
    if n <= 16384:
        return 256
    return 512


def swiglu_quant(gu, n):
    threads = _threads_per_row(n)
    if threads is None or not _eligible(gu, "swiglu") or gu.shape[1] != 2 * n:
        return None
    run = _backend()
    if run is None:
        return None
    _selected("swiglu")
    with torch.cuda.device(gu.device):
        return run(gu, warps=4, vector=1, threads_per_row=threads)

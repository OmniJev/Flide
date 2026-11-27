"""What the GPU this process runs on can do, asked in one place. Kernels and backbones ask these questions instead of
reading compute capabilities or the vendor's build of torch themselves, so supporting another part, or another
vendor's, is a change in this file.

NVIDIA parts answer as their compute capability says. AMD parts (a ROCm build of torch) answer from what they are:
ROCm reports an MI210 (gfx90a) as compute capability 9.0, which every NVIDIA test below would read as Hopper.
"""
from __future__ import annotations

import torch

AMD = torch.version.hip is not None

# compile options of the kernels whose masked and unmasked loops must give the same bits: ROCm's compiler contracts
# multiplies and adds into FMAs differently in the two, NVIDIA's the same way in both
SAME_BITS = {"enable_fp_fusion": False} if AMD else {}


def _cap(device=None) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def smem(device=None) -> int:
    """Shared memory (AMD: LDS) one block may ask for; the opt-in limit where the part has one."""
    p = torch.cuda.get_device_properties(device)
    n = getattr(p, "shared_memory_per_block_optin", None)
    return n if n is not None else getattr(p, "shared_memory_per_block", 0)


def sms(device=None) -> int:
    """Streaming multiprocessors (AMD: compute units), the count persistent kernels size their grids by."""
    return torch.cuda.get_device_properties(device).multi_processor_count


def big_smem(device=None) -> bool:
    """The shared memory of H100, H200 and B200 (about 227 KB), which the widest tiles need."""
    return smem(device) >= 160 * 1024


def fit_stages(stages: int, device=None) -> int:
    """A kernel's software-pipelining stages, at most two on a part with under 96 KB of shared memory (AMD's MI200 and
    MI300 have 64 KB; every NVIDIA part from Ampere on has 99 KB or more). The stages only overlap loads with
    products: a row's sums stay the same."""
    return min(stages, 2) if smem(device) < 96 * 1024 else stages


def tma(device=None) -> bool:
    """NVIDIA's tensor memory accelerator: Hopper and later."""
    return not AMD and _cap(device)[0] >= 9


def hopper_or_later(device=None) -> bool:
    return not AMD and _cap(device)[0] >= 9


def blackwell(device=None) -> bool:
    """B200-class and later (compute capability 10 and up)."""
    return torch.cuda.is_available() and not AMD and _cap(device)[0] >= 10


def consumer_blackwell(device=None) -> bool:
    """RTX 50, RTX PRO 6000 (sm_120 and up)."""
    return not AMD and _cap(device) >= (12, 0)


def a100(device=None) -> bool:
    return not AMD and _cap(device) == (8, 0)


def fp8(device=None) -> bool:
    """float8 (e4m3fn) products: NVIDIA Ada and later. MI210 has no float8, and MI300's is the fnuz format."""
    return torch.cuda.is_available() and not AMD and _cap(device) >= (8, 9)


def int8(device=None) -> bool:
    """int8 products with int32 sums: NVIDIA Ampere and later, AMD CDNA."""
    return torch.cuda.is_available() and (AMD or _cap(device) >= (8, 0))


def choose_blas() -> None:
    """torch's library for the matmuls of this process. On ROCm rocBLAS: hipBLASLt (ROCm's default) sums a row's
    products in an order that depends on how many rows the product has, so a question answered in a graph and the same
    question packed with others differ, and on MI210 it lands further from a float32 referee (Qwen2.5-1.5B, 16 rows
    each of MultiRC, RACE and QuALITY: 199, 63 and 1 questions differing between the two, against 0, 30 and 0 with
    rocBLAS, at the same speed). NVIDIA builds are left as they are."""
    if AMD:
        torch.backends.cuda.preferred_blas_library("cublas")   # the name stands for rocBLAS on ROCm

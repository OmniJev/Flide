"""Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li; MIT license (THIRD_PARTY_NOTICES)."""
from flide.kernels.fla.utils.decorators import input_guard, tensor_cache
from flide.kernels.fla.utils.env import TRITON_ABOVE_3_8_0
from flide.kernels.fla.utils.hardware import (
    IS_AMD,
    IS_GATHER_SUPPORTED,
    IS_INTEL,
    IS_NVIDIA,
    IS_NVIDIA_BLACKWELL,
    IS_NVIDIA_HOPPER,
    IS_TF32_SUPPORTED,
    check_shared_mem,
)

__all__ = ["IS_AMD", "IS_GATHER_SUPPORTED", "IS_INTEL", "IS_NVIDIA", "IS_NVIDIA_BLACKWELL", "IS_NVIDIA_HOPPER",
           "IS_TF32_SUPPORTED", "TRITON_ABOVE_3_8_0", "check_shared_mem", "input_guard", "tensor_cache"]

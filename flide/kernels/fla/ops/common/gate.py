"""Copied from flash-linear-attention 0.6.0 (fla/ops/common/gate.py), forward path only. Copyright (c) 2023-2026
Songlin Yang, Yu Zhang, Zhiyuan Li; MIT license (THIRD_PARTY_NOTICES)."""

import torch
import triton
import triton.language as tl

from flide.kernels.fla.utils import input_guard


@triton.jit
def fused_beta_sigmoid_fwd_kernel(
    x,
    y,
    scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements
    b_x = tl.load(x + offs, mask=mask, other=0).to(tl.float32)
    b_y = scale * tl.sigmoid(b_x)
    tl.store(y + offs, b_y.to(y.dtype.element_ty), mask=mask)


_BETA_SIGMOID_BLOCK_SIZE = 2048


_BETA_SIGMOID_NUM_WARPS = 8


def fused_beta_sigmoid_fwd(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    y = torch.empty_like(x, dtype=torch.float32)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, _BETA_SIGMOID_BLOCK_SIZE),)
    fused_beta_sigmoid_fwd_kernel[grid](
        x,
        y,
        scale,
        n_elements,
        BLOCK_SIZE=_BETA_SIGMOID_BLOCK_SIZE,
        num_warps=_BETA_SIGMOID_NUM_WARPS,
    )
    return y


@input_guard
def fused_beta_sigmoid(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """fla's fused_beta_sigmoid (BetaSigmoidFunction.forward) for inference."""
    return fused_beta_sigmoid_fwd(x, scale)

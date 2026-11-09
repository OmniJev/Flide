"""fla kernels with the size-bucket constexpr pinned, so packed micro-batches of new lengths do not recompile."""
from __future__ import annotations


class _PinnedNB:
    """A Triton kernel whose `NB` argument is replaced by a constant at launch.

    Some FLA kernels use NB = cdiv(length, bucket) only as an autotune key.
    Packed micro-batches vary in length. Pinning NB avoids tuning and compiling
    for each bucket without changing the arithmetic."""

    def __init__(self, kernel, nb: int = 1):
        self.kernel = kernel
        self.nb = nb

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            kwargs["NB"] = self.nb
            return launch(*args, **kwargs)

        return run


def pin_fla_size_buckets() -> None:
    import fla.modules.conv.triton.ops as conv_ops
    import fla.modules.l2norm as l2norm

    for module, name in ((conv_ops, "causal_conv1d_fwd_kernel"), (l2norm, "l2norm_fwd_kernel")):
        kernel = getattr(module, name)
        if not isinstance(kernel, _PinnedNB) and "NB" in getattr(kernel, "arg_names", ()):
            setattr(module, name, _PinnedNB(kernel))


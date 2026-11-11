"""fla kernels with the size-bucket constexpr pinned, so packed micro-batches of new lengths do not recompile."""
from __future__ import annotations


class _PinnedNB:
    """A Triton kernel whose `NB` argument is replaced by a constant at launch.

    fla's causal conv and l2norm kernels take NB = cdiv(length, bucket) as a constexpr that the
    kernel body never reads; it only keys the autotuner. A packed micro-batch has a new length every
    time, so every new NB recompiles and re-autotunes 16 configs, which takes far longer than the
    forward itself. Pinning NB
    keeps the arithmetic identical and compiles each kernel once."""

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
    import importlib

    # (fla <= 0.5 module, fla 0.6 module) of each kernel; fla 0.6 moved them and dropped NB from the conv kernel
    for paths, name in ((("fla.modules.conv.triton.ops", "fla.modules.causal_conv1d.ops"), "causal_conv1d_fwd_kernel"),
                        (("fla.modules.l2norm", "fla.modules.norm.l2norm"), "l2norm_fwd_kernel")):
        for path in paths:
            try:
                module = importlib.import_module(path)
            except ImportError:
                continue
            kernel = getattr(module, name, None)
            if kernel is not None and not isinstance(kernel, _PinnedNB) and "NB" in getattr(kernel, "arg_names", ()):
                setattr(module, name, _PinnedNB(kernel))
            break

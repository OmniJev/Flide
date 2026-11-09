import sys
from types import ModuleType

import pytest

from flide.kernels.fla import pin_fla_size_buckets


class Kernel:
    def __init__(self, *arg_names):
        self.arg_names = arg_names

    def __getitem__(self, grid):
        return lambda **kwargs: (grid, kwargs)


@pytest.mark.parametrize("conv_has_nb", [True, False], ids=["fla-0.5.2", "fla-0.6.0"])
def test_pin_fla_size_buckets(monkeypatch, conv_has_nb):
    modules = (
        "fla",
        "fla.modules",
        "fla.modules.conv",
        "fla.modules.conv.triton",
        "fla.modules.conv.triton.ops",
        "fla.modules.l2norm",
    )
    for name in modules:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    conv, norm = (sys.modules[name] for name in modules[-2:])
    conv_kernel = Kernel("NB") if conv_has_nb else Kernel("T")
    conv.causal_conv1d_fwd_kernel = conv_kernel
    norm.l2norm_fwd_kernel = Kernel("NB")

    pin_fla_size_buckets()
    pinned = norm.l2norm_fwd_kernel
    pin_fla_size_buckets()
    assert norm.l2norm_fwd_kernel is pinned
    assert pinned[(2,)](NB=9) == ((2,), {"NB": 1})
    if conv_has_nb:
        assert conv.causal_conv1d_fwd_kernel[(2,)](NB=9) == ((2,), {"NB": 1})
    else:
        assert conv.causal_conv1d_fwd_kernel is conv_kernel
        assert conv_kernel[(2,)](T=12) == ((2,), {"T": 12})

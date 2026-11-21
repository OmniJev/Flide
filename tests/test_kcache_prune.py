"""Kernel bundles keep an autotuned function's kernels only for the configs its picks name (no GPU needed)."""
from types import SimpleNamespace

from flide.runtime.kcache import _kernel_config, _picked

TUNING = {"m:k": [[[16, "torch.bfloat16"], {"BK": 64, "BV": 128, "num_warps": 4, "num_ctas": 1, "num_stages": 2}],
                  [[32, "torch.bfloat16"], {"BK": 128, "BV": 64, "num_warps": 8, "num_ctas": 1, "num_stages": 3}]]}


def entry(fn="m:k", **cfg):
    base = {"num_warps": 4, "num_ctas": 1, "num_stages": 2, "maxnreg": None, "BK": 64, "BV": 128, "H": 16}
    return {"fn": fn, "config": {**base, **cfg}}


def test_a_picked_config_is_kept():
    assert _picked(entry(), TUNING)
    assert _picked(entry(BK=128, BV=64, num_warps=8, num_stages=3), TUNING)


def test_a_losing_candidate_is_dropped():
    assert not _picked(entry(BK=32), TUNING)
    assert not _picked(entry(num_warps=8), TUNING)


def test_untuned_functions_and_untagged_kernels_are_kept():
    assert _picked(entry(fn="m:other", BK=32), TUNING)
    assert _picked({"fn": "m:k"}, TUNING)


def test_kernel_config_reads_constexprs_by_name():
    jit = SimpleNamespace(arg_names=["x", "H", "BK", "BV", "dtype"])
    consts = {(1,): 16, (2,): SimpleNamespace(value=64), (3,): 128, (4,): object()}
    cfg = _kernel_config(jit, consts, {"num_warps": 4, "num_stages": 2, "num_ctas": 1, "maxnreg": None, "arch": 90})
    assert cfg == {"num_warps": 4, "num_stages": 2, "num_ctas": 1, "maxnreg": None, "H": 16, "BK": 64, "BV": 128}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

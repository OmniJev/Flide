import sys
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from flide.runtime import env


@pytest.mark.parametrize(
    ("have", "built_for", "hint"),
    [
        ((13, 4), None, "cu132"),
        ((13, 2), None, "cu132"),
        ((13, 0), None, "cu130"),
        ((12, 9), None, "cu126"),
        (None, None, "cu126"),
        ((11, 8), None, "Update the NVIDIA driver"),
        ((12, 9), "13.0", "cu126"),
    ],
)
def test_check_cuda_install_hint(monkeypatch, have, built_for, hint):
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False), version=SimpleNamespace(cuda=built_for), __version__="2.14.1"
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(env, "driver_version", lambda: "test driver")
    monkeypatch.setattr(env, "driver_cuda", lambda: have)
    monkeypatch.setattr(env, "requires", lambda _: ["torchvision", "torch<2.14,>=2.12"])

    with pytest.raises(RuntimeError) as exc:
        env.check_cuda()
    message = str(exc.value)
    if hint.startswith("cu"):
        assert f"'torch<2.14,>=2.12' --index-url https://download.pytorch.org/whl/{hint}" in message
    else:
        assert hint in message


def test_install_hint_without_package_metadata(monkeypatch):
    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(env, "requires", missing)
    assert env._install_hint(None) == "Install the CUDA 12.6 build from https://download.pytorch.org/whl/cu126"


def test_check_cuda_reports_init_error_with_older_minor_driver(monkeypatch):
    def init():
        raise RuntimeError("No CUDA GPUs are available")

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, init=init),
        version=SimpleNamespace(cuda="13.2"),
        __version__="2.14.1",
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(env, "driver_version", lambda: "test driver")
    monkeypatch.setattr(env, "driver_cuda", lambda: (13, 1))

    with pytest.raises(RuntimeError, match=r"reports CUDA 13\.1\): No CUDA GPUs are available"):
        env.check_cuda()

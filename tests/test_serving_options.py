"""Serving options reach single and multi-frontend deployments without loading a model."""
import argparse
import os
from unittest.mock import MagicMock

import pytest

from flide.api import frontend, serve, server
from flide import formats
from flide.runtime import engine, online


class _Family:
    name = "fake"

    def __init__(self):
        pass


def parse(*options):
    parser = argparse.ArgumentParser()
    serve.add_parser(parser.add_subparsers())
    return parser.parse_args(["serve", "--model", "unused", *options])


@pytest.fixture
def deployment(monkeypatch):
    import uvicorn

    monkeypatch.setenv("FLIDE_API_KEY", "")
    monkeypatch.setenv("FLIDE_NO_META", "")
    monkeypatch.setattr(engine, "resolve_model", lambda model: model)
    monkeypatch.setattr(engine, "model_config", lambda model: None)
    monkeypatch.setattr(formats, "detect", lambda *args: _Family)
    mocks = {}
    for module, name in ((engine, "Engine"), (online, "Batcher"), (server, "create_app"),
                         (frontend, "start"), (frontend, "serve"), (uvicorn, "run"), (serve, "freeze_heap")):
        mocks[name] = MagicMock()
        monkeypatch.setattr(module, name, mocks[name])
    return mocks


@pytest.mark.parametrize("frontends", [0, 2])
def test_controls_reach_http_and_keep_tensor_parallel(deployment, frontends):
    args = parse("--max-running-requests", "2", "--max-queued-requests", "3", "--no-meta",
                 "--api-key", "test-key", "--tp", "2", "--frontends", str(frontends))
    serve.run(args)
    assert deployment["Engine"].call_args.kwargs["backbone_options"]["tp"] == 2
    if frontends:
        options = deployment["start"].call_args.kwargs
        assert options == {"admission": True}
        admission = deployment["serve"].call_args.args[2]
        deployment["create_app"].assert_not_called()
        deployment["run"].assert_not_called()
    else:
        options = deployment["create_app"].call_args.kwargs
        assert "api_key" not in options and "with_meta" not in options
        admission = options["admission"]
        deployment["start"].assert_not_called()
        deployment["serve"].assert_not_called()
    assert (admission.max_running, admission.max_queued) == (2, 3)
    assert os.environ["FLIDE_API_KEY"] == "test-key"
    assert os.environ["FLIDE_NO_META"] == "1"


def test_default_controls_remain_optional(deployment):
    serve.run(parse())
    options = deployment["create_app"].call_args.kwargs
    assert options["admission"] is None
    assert "api_key" not in options and "with_meta" not in options


def test_running_limit_can_have_unlimited_queue(deployment):
    serve.run(parse("--max-running-requests", "2"))
    admission = deployment["create_app"].call_args.kwargs["admission"]
    assert admission.max_running == 2 and admission.max_queued is None


@pytest.mark.parametrize("explicit", [False, True])
def test_api_key_environment_and_cli_precedence(monkeypatch, deployment, explicit):
    monkeypatch.setenv("FLIDE_API_KEY", "env-key")
    serve.run(parse(*(["--api-key", "cli-key"] if explicit else [])))
    assert os.environ["FLIDE_API_KEY"] == ("cli-key" if explicit else "env-key")


def test_queue_requires_running_limit(deployment):
    with pytest.raises(SystemExit, match="requires --max-running-requests"):
        serve.run(parse("--max-queued-requests", "0"))
    deployment["Engine"].assert_not_called()


@pytest.mark.parametrize("option,value", [("--max-running-requests", "0"), ("--max-running-requests", "-1"),
                                         ("--max-queued-requests", "-1")])
def test_invalid_request_limits(option, value):
    with pytest.raises(SystemExit) as error:
        parse(option, value)
    assert error.value.code == 2

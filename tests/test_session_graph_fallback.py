"""The engine accepts both session graph snapshot layouts without a GPU."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flide.formats.base import Plan, Read
from flide.runtime.engine import Engine
from flide.runtime.session import SessionCache


def engine_for(graphs):
    engine = Engine.__new__(Engine)
    engine.backbone = SimpleNamespace(request_graphs=lambda: graphs)
    engine.eager_tokens = None
    engine.sessions = SessionCache(1 << 20)
    engine.heads = {"test": Mock(side_effect=lambda hidden: hidden)}
    return engine


@pytest.mark.parametrize("has_sflat", [False, True])
@pytest.mark.parametrize("save", [False, True])
def test_session_fallback_snapshot_and_resume(has_sflat, save):
    hidden = object()
    calls = []

    def run_session(prefix, suffixes, entry, start, mark):
        calls.append((entry, start, mark))
        return hidden, ({mark: {"S": {}}}, {}) if mark is not None else None

    graphs = SimpleNamespace(run_session=Mock(side_effect=run_session))
    if has_sflat:
        graphs.run_sflat = Mock(return_value=None)
    engine = engine_for(graphs)
    if not save:
        engine.sessions.mark = Mock(return_value=None)
    plan = Plan(list(range(4096)), [[10], [11]], (Read("test"),))

    assert engine._graph_hidden(plan) is hidden
    assert calls[0][0] is None and calls[0][1] == 0
    if save:
        mark = calls[0][2]
        cached, start = engine.sessions.match(plan.prefix)
        assert start == mark
        assert cached.tokens == plan.prefix[:mark]
        assert list(cached.states) == [mark]
        assert cached.kv == {}
        # With no new save point, a resumed request returns no snapshot.
        assert engine._graph_hidden(plan) is hidden
        assert calls[-1] == (cached, mark, None)
    else:
        assert not engine.sessions.entries
    assert engine.heads["test"].call_count == (2 if save else 1)


@pytest.mark.parametrize("save", [False, True])
def test_sflat_snapshot_layout_is_preserved(save):
    hidden = object()
    mark = 3840
    snaps = [(mark, ({mark: {"S": {}}}, {}))] if save else []
    graphs = SimpleNamespace(run_sflat=Mock(return_value=(hidden, [snaps])),
                             run_session=Mock(side_effect=AssertionError("unexpected fallback")))
    engine = engine_for(graphs)
    plan = Plan(list(range(4096)), [[10]], (Read("test"),))

    assert engine._graph_hidden(plan) is hidden
    assert len(engine.sessions.entries) == int(save)
    if save:
        cached, start = engine.sessions.match(plan.prefix)
        assert start == mark and cached.tokens == plan.prefix[:mark]
    graphs.run_session.assert_not_called()


def test_unsupported_session_graph_returns_to_packed_path():
    graphs = SimpleNamespace(run_sflat=Mock(return_value=None), run_session=Mock(return_value=None))
    engine = engine_for(graphs)
    plan = Plan(list(range(4096)), [[10]], (Read("test"),))

    assert engine._graph_hidden(plan) is None
    assert not engine.sessions.entries
    engine.heads["test"].assert_not_called()

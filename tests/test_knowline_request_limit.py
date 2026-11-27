"""KnowLine refuses oversized plans before they enter the GPU queue."""
import asyncio
from types import SimpleNamespace

import httpx
import pytest

from flide.api.server import create_app
from flide.api.schema import SystemOneRequest
from flide_knowline import KnowLine


def family(limit):
    f = KnowLine.__new__(KnowLine)
    f.max_questions = 255
    f.max_request_tokens = limit
    f.read_twice = False
    f.header_when_long = 0
    f.vision = False
    f.assets = False
    f.labels = ["A", "B"]
    f._scaffold = lambda *args: ("prefix", None, "ending", [])
    f._encode = lambda prefix, texts: ([1] * 7, [], [[2] * 3 for _ in texts])
    return f


def request(n=2):
    return SystemOneRequest(state="input", questions={f"q{i}": {"type": "noul"} for i in range(n)})


@pytest.mark.parametrize("limit", [None, 13, 14])
def test_unlimited_and_boundary_requests(limit):
    assert family(limit).plan(request()).tokens == 13


def test_limit_counts_all_question_branches():
    f = family(12)
    assert f.plan(request(1)).tokens == 10
    with pytest.raises(ValueError, match="request is 13 tokens, limit is 12"):
        f.plan(request(2))


@pytest.mark.parametrize("batched", [False, True])
def test_oversized_request_returns_422_without_gpu_execution(batched):
    f = family(12)
    calls = []

    def decide(req, **options):
        f.plan(req)
        calls.append(req)
        raise AssertionError("oversized request reached inference")

    app = create_app(SimpleNamespace(format=f, decide=decide), model_name="m")
    item = request().model_dump()
    item["model"] = "m"
    body = {"model": "m", "requests": [item, item]} if batched else item

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/systemone", json=body)
        assert response.status_code == 422
        assert "limit is 12" in response.text

    asyncio.run(run())
    assert not calls

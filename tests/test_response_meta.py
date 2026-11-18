"""Deployment metadata settings apply to single and batched decision responses."""
import asyncio
from types import SimpleNamespace

import httpx
import pytest

from flide.api.schema import NoulAnswer, SystemOneResponse, Usage
from flide.api.server import create_app


class _Engine:
    def __init__(self, meta_key):
        self.format = SimpleNamespace(name="fake", aliases=(), meta_key=meta_key)

    def decide(self, request, **options):
        response = SystemOneResponse(model="", answers={"flag": NoulAnswer(noul=0.5)},
                                     usage=Usage(input_tokens=5, output_tokens=1))
        return response, {"format": "fake", "timing": {"total_s": 0.1}}


@pytest.mark.parametrize("include_meta", [None, True, False])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("meta_key", ["meta", "fake_meta"])
def test_decision_response_metadata(include_meta, batched, meta_key):
    options = {} if include_meta is None else {"include_meta": include_meta}
    app = create_app(_Engine(meta_key), model_name="example", **options)
    state = {"model": "example", "state": "input", "questions": {
        "flag": {"type": "noul", "instructions": "A question"}}}
    body = {"model": "example", "requests": [state, state]} if batched else state

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/systemone", json=body)
        assert response.status_code == 200
        payload = response.json()
        assert payload["model"] == "example"
        results = payload["results"] if batched else [payload]
        assert len(results) == (2 if batched else 1)
        for result in results:
            expected = {"answers", "usage"} if batched else {"model", "answers", "usage"}
            if include_meta is not False:
                expected.add(meta_key)
                assert result[meta_key]["format"] == "fake"
            assert set(result) == expected
            assert result["answers"]["flag"]["noul"] == 0.5
            assert result["usage"] == {"input_tokens": 5, "output_tokens": 1}

    asyncio.run(check())

import base64
import io
import json
import wave
from types import SimpleNamespace

import pytest
import torch
import transformers
from PIL import Image

from flide import settings, verify
from flide.api.schema import SystemOneRequest, SystemOneResponse
from flide.formats.base import Format, check_limits
from flide.runtime import offline
from flide.runtime.engine import Engine, model_config
from flide_d1 import D1


@pytest.fixture(params=["vl", "omni"])
def checkpoint(tmp_path, request):
    kind = request.param
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": f"lfm3_{kind}",
                "auto_map": {"AutoModel": f"modeling_d1_lfm3_{kind}.D1Model", "AutoConfig": "missing.CustomConfig"},
            }
        )
    )
    return tmp_path


def _request(kind="noul", **payload):
    return SystemOneRequest.model_validate(
        {
            "questions": {"q": {"type": kind, "instructions": "What is in the state?"}},
            **payload,
        }
    )


def _audio(rate=8000):
    file = io.BytesIO()
    with wave.open(file, "wb") as wav:
        wav.setparams((2, 2, rate, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x00\x40\x00\xc0")
    return {"type": "audio", "data": base64.b64encode(file.getvalue()).decode()}


def test_plan_media(checkpoint, monkeypatch):
    family = D1(str(checkpoint))
    image = Image.new("RGB", (8, 8), "red")
    media = [{"type": "video", "frames": [image], "fps": 2}, {"type": "image", "image": image}]
    if family.omni:
        media.insert(1, _audio())
    state, _, items = family.plan(_request(state="context", media=media)).inputs
    assert state == "context"
    assert items[0]["fps"] == 2 and items[0]["video"][0].size == (8, 8)
    assert items[-1].getpixel((0, 0)) == (255, 0, 0)
    if family.omni:
        assert items[1]["sampling_rate"] == 8000 and items[1]["audio"].tolist() == [0.0]
        request = _request(questions={"q": {"type": "noul", "instructions": _audio()}})
        assert family.plan(request).inputs[1]["q"]["instructions"]["audio"].tolist() == [0.0]
        with pytest.raises(ValueError, match="local paths"):
            family.plan(_request(media=[{"type": "audio", "path": "clip.wav"}]))
        with pytest.raises(ValueError, match="16-bit PCM WAV"):
            family.plan(_request(media=[{"type": "audio", "data": "YWJj"}]))
        monkeypatch.setattr(settings, "max_media_bytes", 64)
        with pytest.raises(ValueError, match="resampled audio is larger"):
            family.plan(_request(media=[_audio(rate=100)]))
    else:
        with pytest.raises(ValueError, match="requires d1-Omni"):
            family.plan(_request(media=[_audio()]))

    for kind in ("image", "video"):
        request = _request(questions={"q": {"type": "noul", "instructions": {"type": kind, "path": "/outside"}}})
        with pytest.raises(ValueError, match=f"put {kind} inputs in the request's media list"):
            family.plan(request)


@pytest.mark.parametrize(
    "extra",
    [
        {"step": "1"},
        {"step": False},
        {"step": 0},
        {"unit": 1},
        {"range": [-1e308, 1e308]},
        {"step": 1e-320},
    ],
)
def test_invalid_estimate(extra):
    with pytest.raises(ValueError):
        _request(questions={"q": {"type": "number", "instructions": "How many?", "range": [0, 10], **extra}})


def test_number_answer_keeps_large_integer():
    number = 2**53 + 1
    answer = {"type": "number", "number": number, "text": str(number), "span": None, "found": 1, "unit": None}
    response = SystemOneResponse(model="d1", answers={"q": answer}, usage={"input_tokens": 1, "output_tokens": 0})
    assert response.model_dump()["answers"]["q"]["number"] == number


def test_explicit_cuda_is_checked_before_model_resolution(monkeypatch):
    def check():
        raise RuntimeError("CUDA unavailable")

    def resolve(model):
        pytest.fail(f"resolved {model} before checking CUDA")

    monkeypatch.setattr("flide.runtime.env.check_cuda", check)
    monkeypatch.setattr("flide.runtime.engine.resolve_model", resolve)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        Engine("owner/model", device="cuda:0")


def test_engine_d1(checkpoint, monkeypatch):
    assert D1.detect(str(checkpoint), model_config(str(checkpoint)), None)
    assert not D1.detect(str(checkpoint), SimpleNamespace(model_type="lfm3_vl"), None)
    with pytest.raises(ValueError, match="trust_remote_code=True"):
        Engine(str(checkpoint), device="cpu")
    with pytest.raises(ValueError, match="session caching"):
        Engine(str(checkpoint), device="cpu", session_cache_gb=1)

    boxes = {
        "type": "boxes",
        "boxes": [{"box": {"top": 0.5}, "found": 0.9, "confidence": 0.8}],
        "count": 1,
        "presence": [0.9],
        "at_least": True,
    }
    runner = SimpleNamespace()

    def answer(state, questions, media):
        if state == "invalid":
            raise ValueError("invalid question")
        assert questions["q"]["type"] == "boxes"
        return {"answers": {"q": boxes}, "usage": {"input_tokens": 13, "output_tokens": 0}}

    model = SimpleNamespace(
        engine=runner,
        device=torch.device("cpu"),
        system_one_batch=lambda inputs: [answer(*x) for x in inputs],
        system_one=answer,
    )
    model.to = lambda _: model
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *args, **kwargs: model)
    engine = Engine(str(checkpoint), device="cpu", trust_remote_code=True, budget=42)
    request = _request("boxes")
    result, meta = engine.decide(request)
    assert result.model_dump()["answers"] == {"q": boxes}
    assert meta["path"] == "runner" and runner.token_budget == 42
    engine.warmup()

    good, bad = (engine.format.plan(_request("boxes", state=state)) for state in ("valid", "invalid"))
    outputs, stats = engine.run_plans([None, bad, good], budget=32)
    assert outputs[0] is None and stats["tokens"] == good.tokens == 13
    assert outputs[1].__traceback__ is None
    assert runner.token_budget == 32
    with pytest.raises(ValueError, match="invalid question"):
        engine.format.finish(request, bad, outputs[1])
    assert engine.format.finish(request, good, outputs[2])[0] == result


def test_extended_questions_are_format_specific():
    with pytest.raises(ValueError, match="does not support date"):
        check_limits(Format, _request("date"))
    check_limits(D1, _request("date"))
    choice = _request(questions={"q": {"type": "choice", "multiple": True, "criteria": {"a": "A"}}})
    with pytest.raises(ValueError, match="does not support multiple choice"):
        check_limits(Format, choice)
    check_limits(D1, choice)


def test_offline_media_stays_in_request(tmp_path):
    path = tmp_path / "rows.jsonl"
    payload = {"questions": {"q": {"type": "noul"}}, "media": [_audio()]}
    path.write_text(json.dumps({"payload": payload, "media": payload["media"], "id": "clip"}) + "\n")
    (row,) = offline.read_rows([path])
    assert offline._request(row).media == payload["media"]
    assert offline._request({**row, "media": {}}).media == {}
    result = offline._result(row, "error", "invalid question", None, None, None, None, {}, 0)
    assert result["id"] == "clip" and "media" not in result


def test_verify_skips_runner_rejections(checkpoint, monkeypatch, capsys):
    rows = checkpoint / "rows.jsonl"
    questions = {"q": {"type": "noul", "instructions": "Is this valid?"}}
    rows.write_text("".join(json.dumps({"state": s, "questions": questions}) + "\n" for s in ("a", "invalid", "b")))
    family = D1(str(checkpoint))

    def forward(plans, mode):
        response = {"answers": {"q": {"type": "noul", "noul": 0.5}}, "usage": {"input_tokens": 1, "output_tokens": 0}}
        return [ValueError("invalid question") if p.inputs[0] == "invalid" else response for p in plans], []

    engine = SimpleNamespace(format=family, sessions=None, forward=forward)
    monkeypatch.setattr("flide.runtime.engine.Engine", lambda *args, **kwargs: engine)
    verify.run(
        SimpleNamespace(
            model=str(checkpoint),
            format="d1",
            device="cpu",
            session_cache_gb=0,
            batch_invariant=False,
            format_arg=None,
            requests=str(rows),
            limit=32,
        )
    )
    assert "2 requests, 2 questions" in capsys.readouterr().out

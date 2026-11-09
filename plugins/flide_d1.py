"""LiquidAI's LFM3 decision checkpoints and their System One runner."""

from __future__ import annotations

import io
import json
import math
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from flide import settings
from flide.api.schema import SystemOneResponse
from flide.formats import Format, register
from flide.formats.base import media_bytes
from flide_onejev import media as visual

_MODELS = {
    "lfm3_vl": "modeling_d1_lfm3_vl.D1Model",
    "lfm3_omni": "modeling_d1_lfm3_omni.D1Model",
}
_AUDIO_RATE = 16000


def _waveform(raw: bytes):
    import numpy as np

    try:
        with wave.open(io.BytesIO(raw), "rb") as wav:
            if wav.getsampwidth() != 2:
                raise ValueError("audio must be 16-bit PCM WAV")
            frames, channels, rate = wav.getnframes(), wav.getnchannels(), wav.getframerate()
            pcm = wav.readframes(frames)
    except (EOFError, wave.Error, ZeroDivisionError) as exc:
        raise ValueError("audio must be 16-bit PCM WAV") from exc
    if not frames or len(pcm) != frames * channels * 2:
        raise ValueError("audio is empty or truncated")
    # The upstream sinc resampler allocates one filter per phase of this ratio.
    if not rate or rate * _AUDIO_RATE // math.gcd(rate, _AUDIO_RATE) ** 2 > 1_000_000:
        raise ValueError("unsupported WAV sample rate; resample the clip to 16000 Hz")
    if frames * _AUDIO_RATE > rate * (settings.max_media_bytes // 2):
        raise ValueError(f"resampled audio is larger than {settings.max_media_bytes / 2**20:g} MiB as 16-bit PCM")
    signal = np.frombuffer(pcm, dtype="<i2").reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return {"type": "audio", "audio": signal / 32768, "sampling_rate": rate}


@dataclass
class _Plan:
    inputs: tuple
    tokens: int
    data: None = None  # RemoteBatcher strips local plan data before sending.


@register("d1")
class D1(Format):
    backbones = tuple(_MODELS)
    extended_answers = True

    @classmethod
    def detect(cls, model_dir, config, decision_config):
        expected = _MODELS.get(getattr(config, "model_type", None))
        return expected is not None and (getattr(config, "auto_map", None) or {}).get("AutoModel") == expected

    def __init__(
        self,
        model_dir: str,
        trust_remote_code: bool = False,
        media_root: str | None = None,
        allow_media_urls: bool = False,
    ):
        self.model_dir = model_dir
        config = json.loads((Path(model_dir) / "config.json").read_text())
        if not self.detect(model_dir, SimpleNamespace(**config), None):
            raise ValueError(f"{model_dir} is not an LFM3 d1 decision checkpoint")
        self.omni = config["model_type"] == "lfm3_omni"
        self.trust_remote_code = trust_remote_code
        self.media_root = media_root
        self.allow_media_urls = allow_media_urls

    def _audio(self, item):
        if not self.omni:
            raise ValueError("audio requires d1-Omni")
        if "data" in item:
            if not isinstance(item["data"], str):
                raise ValueError("audio data must be base64")
            raw = media_bytes(item["data"], "audio")
        else:
            ref = item.get("url", item.get("path"))
            raw = visual.read_bytes(
                ref, self.media_root, self.media_root is not None, allow_urls=self.allow_media_urls, what="audio"
            )
        return _waveform(raw)

    def plan(self, request, **options):
        media = getattr(request, "media", None)
        if media is None:
            media = []
        if not isinstance(media, list) or any(not isinstance(item, dict) for item in media):
            raise ValueError("media must be a list of image, video or audio objects")
        items = []
        for item in media:
            if item.get("type") == "audio":
                items.append(self._audio(item))
            else:
                images, videos, meta = visual.load(
                    visual.check_items([item]), self.media_root, self.media_root is not None, self.allow_media_urls
                )
                items.append(images[0] if images else {"type": "video", "video": videos[0], "fps": meta[0].fps})
        questions = {name: q.model_dump(exclude_unset=True) for name, q in request.questions.items()}
        # Flide's queue needs a cost before the checkpoint's runner tokenizes the request.
        cost = len(json.dumps((request.state, questions), ensure_ascii=False))
        for q in questions.values():
            instructions = q.get("instructions")
            if isinstance(instructions, dict):
                kind = instructions.get("type")
                if kind == "audio":
                    q["instructions"] = self._audio(instructions)
                elif kind in ("image", "video"):
                    raise ValueError(f"put {kind} inputs in the request's media list")
        return _Plan((request.state, questions, items), cost)

    def load_runner(self, device, dtype):
        if self.trust_remote_code is not True:
            raise ValueError(
                "d1 uses Python code from the checkpoint; set trust_remote_code=True "
                "(CLI: --format-arg trust_remote_code=true) to load it"
            )
        import torch
        from transformers import AutoModel
        from transformers.utils.versions import require_version

        require_version("transformers>=5.19", "LiquidAI d1 checkpoints require Transformers 5.19 or later.")
        model = AutoModel.from_pretrained(
            self.model_dir, trust_remote_code=True, local_files_only=True, dtype=getattr(torch, dtype)
        ).to(device)
        return _Runner(model)

    def finish(self, request, plan, outputs, **options):
        if isinstance(outputs, ValueError):
            raise outputs
        return SystemOneResponse(model=request.model, **outputs), {}


class _Runner:
    def __init__(self, model):
        self.model = model

    def __call__(self, plans, budget):
        from contextlib import nullcontext

        import torch

        start = time.perf_counter()
        indices = [i for i, p in enumerate(plans) if p is not None]
        outputs = [None] * len(plans)
        calls = tokens = 0
        device = self.model.device
        with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
            if indices:
                self.model.engine.token_budget = budget
                calls = 1
                try:
                    results = self.model.system_one_batch([plans[i].inputs for i in indices])
                except ValueError as exc:
                    # Keep a malformed request from failing its neighbors in the round.
                    if len(indices) == 1:
                        results = [ValueError(str(exc))]
                    else:
                        results = []
                        for i in indices:
                            try:
                                results.append(self.model.system_one(*plans[i].inputs))
                            except ValueError as error:
                                results.append(ValueError(str(error)))
                        calls += len(indices)
                for i, result in zip(indices, results):
                    outputs[i] = result
                    plans[i].tokens = 0 if isinstance(result, ValueError) else result["usage"]["input_tokens"]
                    tokens += plans[i].tokens
        return outputs, {"tokens": tokens, "micro_batches": calls, "forward_s": time.perf_counter() - start}

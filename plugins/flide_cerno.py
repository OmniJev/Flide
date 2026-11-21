"""Cerno 27B v1.1 (apertixai/cerno-27b-v1.1, Qwen3.8-27B with a letter readout, served in FP8) on the shipped
configuration formats/known/apertixai__cerno.json, which serves the other Cerno checkpoints as they are. Its
decider_config.json sets read_twice_tokens (3,000), and the release (evals/decision_index/engine.py, CernoEngine) reads
a short request twice: when the request is estimated at no more than that many tokens, the state becomes the state,
the questions with their options, "Read the same input again before answering:" and the state again. The estimate is
the length of the state's and the questions' JSON divided by 3.5, or for 24,000 characters and more their token count
times 1.08."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flide.api.schema import SystemOneRequest
from flide.formats import register
from flide.formats.base import Plan
from flide.formats.template import Template

REPOS = ("apertixai/cerno-27b-v1.1",)
KEYS = {"attention", "read_twice_tokens", "temperature", "temperature_fit"}
SPEC = "apertixai__cerno"


def release_questions(request: SystemOneRequest) -> dict:
    """The questions as the release keeps them: type, instructions and criteria, each only when the request gave it."""
    out = {}
    for k, q in request.questions.items():
        given = q.model_fields_set
        d: dict[str, Any] = {"type": q.type}
        if "instructions" in given:
            d["instructions"] = q.instructions
        if "criteria" in given:
            c = q.criteria
            d["criteria"] = c.model_dump(exclude_unset=True) if hasattr(c, "model_dump") else c
        out[k] = d
    return out


def read_twice(state: Any, qs: dict) -> str:
    """The release's _read_twice: the state, the questions with their options, the line asking to read again, the
    state again."""
    st = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    lines = []
    for q in qs.values():
        if q["type"] == "choice":
            c = q.get("criteria") or {}
            opts = "; ".join(f"{k}: {v}" if v and v != k else str(k)
                             for k, v in (c.items() if isinstance(c, dict) else ((x, "") for x in c)))
            lines.append(f"- {q.get('instructions', '')} Options: {opts}")
        elif q["type"] == "score":
            lines.append(f"- {q.get('instructions', '')} Levels: " + "; ".join(str(x) for x in (q.get("criteria") or [])))
        else:
            lines.append(f"- {q.get('instructions', '')} (yes or no)")
    return f"{st}\n\nQuestions:\n" + "\n".join(lines) + f"\n\nRead the same input again before answering:\n\n{st}"


@register("cerno")
class Cerno(Template):
    backbones = ("qwen3_5", "qwen3_5_text")

    def __init__(self, model_dir: str, tokenizer=None):
        super().__init__(model_dir, tokenizer=tokenizer, spec=SPEC)
        dc = json.loads((Path(model_dir) / "decider_config.json").read_text())
        if dc.get("attention", "causal") != "causal":
            raise ValueError(f"{model_dir}: decider_config.json has attention={dc.get('attention')!r}; Cerno 27B v1.1 "
                             "reads causally")
        self.read_twice_tokens = int(dc.get("read_twice_tokens", 0))

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        d = Path(model_dir)
        parts = {p.lower() for p in d.resolve().parts}
        if any(f"models--{r.lower().replace('/', '--')}" in parts for r in REPOS):
            return True
        p = d / "decider_config.json"
        try:
            dc = json.loads(p.read_text()) if p.is_file() else None
        except ValueError:
            return False
        return isinstance(dc, dict) and set(dc) == KEYS and dc.get("attention") == "causal"

    def estimate(self, state: Any, qs: dict) -> float:
        """The release's _est_tokens."""
        txt = json.dumps(state, ensure_ascii=False) + json.dumps(qs, ensure_ascii=False)
        if len(txt) < 24000:
            return len(txt) / 3.5
        return len(self.tokenizer(txt, add_special_tokens=False)["input_ids"]) * 1.08

    def plan(self, request: SystemOneRequest, **kw) -> Plan:
        if self.read_twice_tokens:
            qs = release_questions(request)
            if self.estimate(request.state, qs) <= self.read_twice_tokens:
                request = request.model_copy(update={"state": read_twice(request.state, qs)})
        return super().plan(request, **kw)

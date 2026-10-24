"""Open-Jev family for flide (ZefanCai/Open-Jev-2B / 9B / 27B): every candidate is its own prompt, read at its
last token by a float32 scalar head; Choice and Score take a softmax over the candidates' scores at the release's
temperature, Noul a softmax over (0, score).

Prompts are the release's own (jev/api.py `candidate_prompts`, chat template with thinking off), tokenized whole,
so the tokens match the author's runtime exactly. The candidates of a request share their tokens up to where they
first differ, and that shared part is the plan's prefix: computed once, every candidate branches from its state.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from flide.api.answers import make_choice, make_noul, make_score
from flide.api.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneResponse, Usage
from flide.formats import register
from flide.formats.base import Format, Plan, Read

from .prepare import FORMAT_ID

SCORE = Read("score")


def _render(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class ScalarHead:
    def __init__(self, weight, bias):
        self.weight = weight.float().contiguous()
        self.bias = bias.float()

    def __call__(self, hidden):
        return hidden.float() @ self.weight.T + self.bias


@register("open_jev", format_ids=(FORMAT_ID,))
class OpenJev(Format):
    backbones = ("qwen3_5", "qwen3_5_text")
    aliases = ("jev-latest",)

    def __init__(self, model_dir: str, tokenizer=None, max_length: int | None = None):
        import transformers

        self.model_dir = model_dir
        self.config = json.loads((Path(model_dir) / "decision_config.json").read_text())
        self.temperature = float(self.config["temperature"])
        self.max_length = max_length or int(self.config.get("max_length", 4096))
        self.tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self._wrap = None

    def load_heads(self, backbone) -> dict[str, Any]:
        from safetensors.torch import load_file

        t = load_file(str(Path(self.model_dir) / "head.safetensors"), device=str(backbone.device))
        return {"score": ScalarHead(t["weight"], t["bias"])}


    def _records(self, request):
        state = request.state if isinstance(request.state, (str, dict, list)) else request.model_dump()["state"]
        out = []
        for qid, q in request.questions.items():
            question = _render(q.instructions)
            if isinstance(q, ChoiceQuestion):
                keys = list(q.criteria)
                options = [k if v is None else f"{k}: {_render(v)}" for k, v in q.criteria.items()]
                out.append((qid, "choice", question, keys, options, None))
            elif isinstance(q, ScoreQuestion):
                options = [_render(level) for level in q.criteria]
                out.append((qid, "score", question, [str(i) for i in range(len(options))], options, list(q.criteria)))
            elif isinstance(q, NoulQuestion):
                crit = q.criteria
                for key, label in (("true", "Yes"), ("false", "No")):
                    v = getattr(crit, key, None) if crit is not None else None
                    if v is not None:
                        question += f"\n{label} means: {_render(v)}"
                out.append((qid, "noul", question, ["false", "true"], None, None))
            else:
                raise ValueError(f"unknown question type {type(q).__name__}")
        return state, out

    def _prompt(self, text: str) -> list[int]:
        chat = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        return self.tokenizer(chat)["input_ids"]

    def _chat_wrap(self) -> tuple[str, str]:
        """The chat template's text before and after the user message."""
        if self._wrap is None:
            mark = "\x00flide\x00"
            chat = self.tokenizer.apply_chat_template([{"role": "user", "content": mark}], tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False)
            self._wrap = tuple(chat.split(mark)) if chat.count(mark) == 1 else ("", "")
        return self._wrap

    def _prompts(self, context: str, tails: list[str]) -> list[list[int]]:
        """Token ids of the prompts context + tail with the long context tokenized once. The context ends with a
        newline run and every tail starts with a letter, where the pre-tokenizer always splits; checked on the end
        of the context with the first tail, and on failure every prompt is tokenized whole."""
        pre, post = self._chat_wrap()
        enc = lambda text, first=False: self.tokenizer(text, add_special_tokens=first)["input_ids"]
        if pre or post:
            window, first_tail = context[-256:], tails[0] + post
            if enc(window + first_tail) == enc(window) + enc(first_tail):
                shared = enc(pre + context, first=True)
                return [shared + enc(t + post) for t in tails]
        return [self._prompt(context + t) for t in tails]

    def plan(self, request, **_) -> Plan:
        state, records = self._records(request)
        tails, owners = [], []
        for qid, kind, question, keys, options, levels in records:
            head = f"Question: {question}\n"
            if kind == "noul":
                tails.append(head + "Is the answer to this question yes? Answer Yes or No.")
                owners.append(qid)
                continue
            for opt in options:
                tails.append(head + f"Proposed answer: {opt}\nIs this proposed answer correct? Answer Yes or No.")
                owners.append(qid)
        seqs = self._prompts(f"Context:\n{_render(state)}\n\n", tails)
        longest = max(len(s) for s in seqs)
        if longest > self.max_length:
            raise ValueError(f"input length {longest} exceeds max_length={self.max_length}")
        n = min(len(s) for s in seqs) - 1
        first = seqs[0]
        p = 0
        while p < n:
            k = min(n, p + 256)
            if all(s[p:k] == first[p:k] for s in seqs):
                p = k
                continue
            while p < n and all(s[p] == first[p] for s in seqs):
                p += 1
            break
        return Plan(first[:p], [s[p:] for s in seqs], (SCORE,), {"records": records, "owners": owners})

    def finish(self, request, plan: Plan, outputs, debug: bool = False, **_):
        scores: dict[str, list[float]] = {}
        for qid, out in zip(plan.data["owners"], outputs):
            scores.setdefault(qid, []).append(float(out[0][0][0]))
        T = self.temperature
        answers = {}
        for qid, kind, question, keys, options, levels in plan.data["records"]:
            s = scores[qid]
            logits = [0.0, s[0]] if kind == "noul" else s
            m = max(logits)
            e = [math.exp((x - m) / T) for x in logits]
            probs = [x / sum(e) for x in e]
            if kind == "noul":
                answers[qid] = make_noul(probs[1])
            elif kind == "choice":
                answers[qid] = make_choice(keys, probs)
            else:
                answers[qid] = make_score(levels, probs)
        usage = Usage(input_tokens=plan.tokens, output_tokens=len(answers))
        meta = {"prefix_tokens": len(plan.prefix), "branches": len(plan.suffixes),
                "logical_tokens": len(plan.prefix) * len(plan.suffixes) + sum(len(x) for x in plan.suffixes)}
        if debug:
            meta["scores"] = scores
        return SystemOneResponse(model="", answers=answers, usage=usage), meta

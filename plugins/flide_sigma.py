"""Logits Sigma (logitshq/Logits-Sigma-1: Gemma 4 31B with a merged LoRA, causal attention, a letter readout): its
release's prompt layout v5 (code/logits_decision/prompt.py), which the checkpoint's merge.json names.

Every question is one prompt in the model's chat template, thinking off. The user message starts with a block that
depends only on the request: an instruction, "Questions about this state:" with every question of the request
numbered in request order (its text only, each within an even share of 16,000 characters, at most 80 listed), and the
state. When the template head and that block, read once, take at most 3,000 tokens, the block is written twice, the
second copy after "Read the questions and the state again.". The question then follows with its options coded A, B,
..., and the model turn opens with "Answer:"; the answer is the next-token distribution over the codes shown (Gemma's
soft-capped logits, temperature 1). The block and the question are tokenized separately and joined, as the release
joins them, so every question forks from the block. A noul question shows "true" (or "Yes") as A and "false" (or "No")
as B; its probability is A's. The release answers choice and noul questions; it reads text only, up to 131,072 tokens."""
from __future__ import annotations

import json
import math
import string
from pathlib import Path
from typing import Any

from flide.api.answers import make_choice, make_noul
from flide.api.schema import SystemOneRequest, SystemOneResponse, Usage
from flide.formats import register
from flide.formats.base import Format, Plan, Read
from flide.formats.next_token import TokenRows

REPOS = ("logitshq/Logits-Sigma-1",)
INSTRUCTION = ("Read the questions and the state below, then decide the question at the end. "
               "Reply with the letter of one option.")
HEADER = "Questions about this state:"
AGAIN = "Read the questions and the state again."
ANSWER_PREFIX = "Answer:"
HEADER_MAX_CHARS = 16_000
HEADER_MIN_CHARS = 200
TWICE_MAX_TOKENS = 3000
MAX_OPTIONS = 255
SENTINEL = "decision-prompt"
READ = Read("codes")


def _text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def header(questions: list[str]) -> str:
    """The numbered list of the request's questions within the character budget (the release's header_v5)."""
    n = max(len(questions), 1)
    per = max(HEADER_MAX_CHARS // n, HEADER_MIN_CHARS)
    shown = questions[: HEADER_MAX_CHARS // per]
    lines = [HEADER]
    for i, question in enumerate(shown, 1):
        text = question.strip()
        if len(text) > per:
            text = text[:per].rstrip() + "…"
        first, *rest = text.split("\n")
        lines.append(f"{i}. {first}")
        lines += [f"   {line}" if line.strip() else "" for line in rest]
    if len(shown) < len(questions):
        lines.append(f"(and {len(questions) - len(shown)} more questions)")
    return "\n".join(lines)


def shared_message(state: str, questions: list[str], twice: bool = False) -> str:
    """The request's block, ending in a blank line (the release's shared_message_v5)."""
    state = state.strip() if state and state.strip() else "(empty)"
    block = "\n".join([header(questions), "", "State:", state])
    parts = [INSTRUCTION, "", block]
    if twice:
        parts += ["", AGAIN, "", block]
    return "\n".join(parts) + "\n\n"


def question_message(question: str, options: list[str], codes: list[str]) -> str:
    lines = [f"Question: {question}", "Options:"] + [f"{code}. {option}" for code, option in zip(codes, options)]
    return "\n".join(lines).rstrip()


def _options(q) -> tuple[list[str], list[str]]:
    """The keys of a question's options and the text shown for each (the release's _question)."""
    if q.type == "noul":
        c = q.criteria
        return ["true", "false"], [_text(getattr(c, "true", None)) or "Yes", _text(getattr(c, "false", None)) or "No"]
    keys = list(q.criteria)
    return keys, [_text(q.criteria[k]) or k for k in keys]


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    w = [math.exp(x - m) for x in xs]
    s = sum(w)
    return [x / s for x in w]


@register("sigma")
class Sigma(Format):
    backbones = ("gemma4", "gemma4_text", "gemma4_unified")

    def __init__(self, model_dir: str, tokenizer=None, max_len: int = 131072, twice_max_tokens: int = TWICE_MAX_TOKENS):
        import transformers

        merge = json.loads((Path(model_dir) / "merge.json").read_text())
        dc = merge.get("decision_config", {})
        if dc.get("bidirectional"):
            raise ValueError(f"{model_dir}: merged from a bidirectional adapter; this family reads causally")
        if dc.get("layout", "v1") != "v5":
            raise ValueError(f"{model_dir}: layout {dc.get('layout', 'v1')!r}; this family reads layout v5")
        self.model_dir = model_dir
        self.tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.max_len = int(max_len)
        self.twice_max_tokens = int(twice_max_tokens)
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": SENTINEL}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        self.head, sep, self.tail = text.partition(SENTINEL)
        if not sep or SENTINEL in self.tail:
            raise ValueError("the chat template does not render the user message verbatim once")
        self.codes, self.code_ids = self._codes()

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        d = Path(model_dir)
        parts = {p.lower() for p in d.resolve().parts}
        if any(f"models--{r.lower().replace('/', '--')}" in parts for r in REPOS):
            return True
        p = d / "merge.json"
        try:
            merge = json.loads(p.read_text()) if p.is_file() else None
        except ValueError:
            return False
        return (isinstance(merge, dict) and {"base", "adapter", "scale", "lora", "decision_config"} <= set(merge)
                and merge["decision_config"].get("layout") == "v5")

    def _enc(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _codes(self) -> tuple[list[str], list[int]]:
        """A..Z, then two-letter codes that are one token right after "Answer:" (the release's option_codes)."""
        base = self._enc(ANSWER_PREFIX)
        letters = string.ascii_uppercase
        codes, ids = [], []
        for code in list(letters) + [a + b for a in letters for b in letters]:
            enc = self._enc(ANSWER_PREFIX + " " + code)
            if enc[: len(base)] == base and len(enc) == len(base) + 1:
                codes.append(code)
                ids.append(enc[-1])
            if len(codes) == MAX_OPTIONS:
                break
        if len(codes) < MAX_OPTIONS or len(set(ids)) != len(ids):
            raise ValueError("the tokenizer does not give 255 distinct single-token option codes")
        return codes, ids

    def token_heads(self) -> dict[str, list[int]]:
        return {"codes": list(self.code_ids)}

    def load_heads(self, backbone) -> dict[str, Any]:
        weight, bias = backbone.output_head()
        return {"codes": TokenRows(weight, self.code_ids, bias, softcap=getattr(backbone, "logit_softcap", None))}

    def prefix(self, state: str, texts: list[str]) -> list[int]:
        """The request's shared tokens: template head and block, the block twice when once fits the limit."""
        ids = self._enc(self.head + shared_message(state, texts))
        if self.twice_max_tokens and len(ids) <= self.twice_max_tokens:
            ids = self._enc(self.head + shared_message(state, texts, twice=True))
        return ids

    def plan(self, request: SystemOneRequest, **_) -> Plan:
        if getattr(request, "images", None):
            raise ValueError("images: this model reads text only")
        state = _text(request.state)
        texts = [_text(q.instructions) for q in request.questions.values()]
        base = self.prefix(state, texts)
        questions, suffixes = [], []
        for (qid, q), text in zip(request.questions.items(), texts):
            if q.type not in ("choice", "noul"):
                raise ValueError(f"question '{qid}': this model answers choice and noul questions")
            keys, options = _options(q)
            if len(options) > MAX_OPTIONS:
                raise ValueError(f"question '{qid}': {len(options)} options, limit {MAX_OPTIONS}")
            tail = self._enc(question_message(text, options, self.codes[: len(options)]) + self.tail + ANSWER_PREFIX)
            if len(base) + len(tail) > self.max_len:
                raise ValueError(f"maximum context length {self.max_len} exceeded: {len(base) + len(tail)} tokens")
            questions.append({"qid": qid, "type": q.type, "keys": keys, "n": len(options)})
            suffixes.append(tail)
        return Plan(base, suffixes, (READ,), {"questions": questions})

    def finish(self, request: SystemOneRequest, plan: Plan, outputs, **_):
        answers = {}
        for q, out in zip(plan.data["questions"], outputs):
            p = _softmax(out[0][0][: q["n"]])
            answers[q["qid"]] = make_noul(p[0]) if q["type"] == "noul" else make_choice(q["keys"], p)
        usage = Usage(input_tokens=plan.tokens, output_tokens=len(plan.data["questions"]))
        return SystemOneResponse(model="", answers=answers, usage=usage), {"format": "sigma",
                                                                          "prefix_tokens": len(plan.prefix),
                                                                          "branches": len(plan.suffixes)}

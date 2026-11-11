"""this-that-model (flock-io/this-that-model-1.0 on Qwen3.5): the release's own prompt and head.

One request is one sequence, no chat template: `Context:\n{state}`, then per question
`\n\nQuestion k: {text}\nOptions:\n(A) ...\n(B) ...`, then `\nAnswer k: (`. The answer is read at each
`Answer k: (` token, the last token of that block, against the lm_head rows of the option labels, exactly
as thisthat/prompt.py and thisthat/backends/torch_backend.py do:

    p_k(j | x) = softmax_j( <lm_head[label_j], h_k> / tau )

so one pass answers every question and no answer is ever written into the prompt (the release's
conditional-independence argument). The layout, the truncation and the option rendering follow the
release; the temperature default is the one its inference code uses.
"""
from __future__ import annotations

import json
import math
import string
from typing import Any, Sequence

from flide.api.answers import make_choice, make_noul, make_score
from flide.api.schema import SystemOneRequest, SystemOneResponse, Usage
from flide.formats import register
from flide.formats.base import Format, Plan, Read
from flide.formats.next_token import TokenRows

LETTERS = "ABCDEFGHIJ"          # options past these need the two-letter labels
MAX_OPTIONS = 255
DEFAULT_MAX_STATE_TOKENS = 1536
DEFAULT_TEMPERATURE = 1.99241824
READ_HEAD = "slots"


def label_table(tok) -> tuple[list[str], list[int]]:
    """The single-token labels A..Z then two-letter combinations, in the release's order."""
    names = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    labels: list[str] = []
    ids: list[int] = []
    for name in names:
        got = tok.encode(name, add_special_tokens=False)
        if len(got) == 1:
            labels.append(name)
            ids.append(got[0])
        if len(labels) == MAX_OPTIONS:
            break
    if len(labels) < MAX_OPTIONS or len(set(ids)) != MAX_OPTIONS:
        raise ValueError(f"this tokenizer yields {len(labels)} distinct single-token labels; {MAX_OPTIONS} are needed")
    return labels, ids


def _options_ids(tok, options: Sequence[str], label_ids: Sequence[int]) -> list[int]:
    if len(options) <= len(LETTERS):
        return tok.encode("".join(f"\n({LETTERS[j]}) {o}" for j, o in enumerate(options)), add_special_tokens=False)
    out: list[int] = []
    for j, option in enumerate(options):
        out += tok.encode("\n(", add_special_tokens=False) + [label_ids[j]] + tok.encode(f") {option}",
                                                                                     add_special_tokens=False)
    return out


def build(tok, state: str, questions: Sequence[tuple[str, Sequence[str]]], *,
          max_state_tokens: int = DEFAULT_MAX_STATE_TOKENS) -> dict:
    """The release's state-first layout: (ids, slot index per question, option count per question)."""
    multi = len(questions) > 1
    labels, label_ids = label_table(tok)
    lead = "\n\n"
    ids = tok.encode("Context:\n" + state, add_special_tokens=False)[: max_state_tokens + 3]
    slots: list[int] = []
    for k, (text, options) in enumerate(questions):
        head = f"{lead}Question{' ' + str(k + 1) if multi else ''}: {text}\nOptions:"
        ids += tok.encode(head, add_special_tokens=False) + _options_ids(tok, options, label_ids)
        ids += tok.encode(f"\nAnswer{' ' + str(k + 1) if multi else ''}: (", add_special_tokens=False)
        slots.append(len(ids) - 1)
    return {"ids": ids, "slots": slots, "n_options": [len(o) for _, o in questions]}


def _legend(lines: Sequence[tuple[str, str | None]], heading: str) -> str:
    return heading + "\n" + "\n".join(f"- {lab}: {desc}" if desc else f"- {lab}" for lab, desc in lines)


def _given(model) -> dict:
    """The fields a request set on a pydantic object, as the request wrote them."""
    if model is None:
        return {}
    return {k: getattr(model, k) for k in (getattr(model, "model_fields_set", None) or ())} | dict(
        getattr(model, "model_extra", None) or {})


def typed(name: str, q) -> tuple[str, list[str], list[str], str]:
    """(question text, labels, options, kind) for one request question, as thisthat/systemone_protocol.py reads it.

    The options the head scores are the *labels* -- the caller's keys for a choice, "0".."n-1" for a score,
    no/yes -- and the criteria's descriptions go into the question text as a legend, which is the release's
    layout.
    """
    given = _given(q)
    instructions = given.get("instructions") or ""
    if not isinstance(instructions, str):
        raise ValueError(f"question {name!r}: instructions must be a string")
    criteria = given.get("criteria")
    if q.type == "noul":
        crit = _given(criteria)
        lines = [("no", crit.get("false")), ("yes", crit.get("true"))]
        text = _legend(lines, instructions + "\n\nOptions:") if any(d for _, d in lines) else instructions
        return text, ["no", "yes"], ["no", "yes"], "noul"
    if q.type == "choice":
        labels = [str(k) for k in criteria]
        text = _legend([(str(k), str(v)) for k, v in criteria.items()], instructions + "\n\nOptions:")
        return text, labels, list(labels), "choice"
    if q.type == "score":
        labels = [str(i) for i in range(len(criteria))]
        text = _legend([(lab, str(d)) for lab, d in zip(labels, criteria)], instructions + "\n\nLevels:")
        return text, labels, list(labels), "score"
    raise ValueError(f"question {name!r} has type {q.type!r}; this model answers noul, choice and score")


@register("thisthat")
class ThisThat(Format):
    """flock-io/this-that-model: one state, N questions, N read positions, one pass."""

    name = "thisthat"
    backbones = ("qwen3_5", "qwen3_5_text")

    def __init__(self, model_dir: str, tokenizer=None, temperature: float | None = None,
                 max_state_tokens: int = DEFAULT_MAX_STATE_TOKENS, **_):
        import transformers

        self.model_dir = model_dir
        self.tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.labels, self.label_ids = label_table(self.tokenizer)
        self.temperature = DEFAULT_TEMPERATURE if temperature is None else float(temperature)
        self.max_state_tokens = int(max_state_tokens)
        self.max_choice_options = len(self.label_ids)

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        """False on purpose: the checkpoint carries no marker of its own (stock Qwen files and no
        decision_config.json), so this family is chosen with `--format thisthat`."""
        return False

    def token_heads(self) -> dict[str, list[int]]:
        return {READ_HEAD: list(self.label_ids)}

    def load_heads(self, backbone) -> dict[str, Any]:
        weight, bias = backbone.output_head()
        return {READ_HEAD: TokenRows(weight, self.label_ids, bias,
                                     softcap=getattr(backbone, "logit_softcap", None))}

    def plan(self, request: SystemOneRequest, **_) -> Plan:
        state = request.state if isinstance(request.state, str) else json.dumps(request.state, ensure_ascii=False)
        questions, data = [], []
        for name, q in request.questions.items():
            text, labels, options, kind = typed(name, q)
            if not 2 <= len(labels) <= MAX_OPTIONS:
                raise ValueError(f"question {name!r} needs 2 to {MAX_OPTIONS} options, got {len(labels)}")
            if len(set(labels)) != len(labels):
                raise ValueError(f"question {name!r}: labels must be distinct")
            questions.append((text, options))
            data.append((name, kind, tuple(labels), tuple(options)))
        built = build(self.tokenizer, state, questions, max_state_tokens=self.max_state_tokens)
        ids = built["ids"]
        offsets = tuple(slot - len(ids) for slot in built["slots"])
        return Plan([], [ids], (Read(READ_HEAD, positions=offsets),), data)

    def finish(self, request: SystemOneRequest, plan: Plan, outputs, **_):
        rows = outputs[0][0]                 # the one read's rows: one per answer slot, in question order
        answers = {}
        for (name, kind, labels, options), logits in zip(plan.data, rows):
            z = [logits[i] / self.temperature for i in range(len(labels))]
            top = max(z)
            w = [math.exp(x - top) for x in z]
            total = sum(w)
            probs = [x / total for x in w]
            if kind == "noul":
                answers[name] = make_noul(probs[1])
            elif kind == "choice":
                answers[name] = make_choice(list(labels), probs)
            else:
                best = max(range(len(labels)), key=probs.__getitem__)
                answers[name] = make_score(list(request.questions[name].criteria), probs).model_copy(
                    update={"score": float(best)})
        usage = Usage(input_tokens=plan.tokens, output_tokens=len(plan.data))
        return SystemOneResponse(model="", answers=answers, usage=usage), {"format": "thisthat",
                                                                          "tokens": plan.tokens}
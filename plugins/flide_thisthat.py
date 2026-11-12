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


def _question_block(tok, k: int, text: str, options: Sequence[str], multi: bool, lead: str,
                    label_ids: Sequence[int]) -> list[int]:
    head = f"{lead}Question{' ' + str(k + 1) if multi else ''}: {text}\nOptions:"
    return tok.encode(head, add_special_tokens=False) + _options_ids(tok, options, label_ids)


def build(tok, state: str, questions: Sequence[tuple[str, Sequence[str]]], *, layout: str = "state_first",
          max_state_tokens: int = DEFAULT_MAX_STATE_TOKENS) -> dict:
    """The release's layout: (ids, slot index per question, option count per question, state-free prefix length).

    `state_first` interleaves each question with its answer slot after the state; `schema_first` puts the
    questions first, so their tokens are a prefix no state depends on -- the release's argument for reusing
    that prefix across a stream of states.
    """
    if layout not in ("state_first", "schema_first"):
        raise ValueError(f"layout must be 'state_first' or 'schema_first', got {layout!r}")
    questions = list(questions)
    if not questions:
        raise ValueError("no questions")
    multi = len(questions) > 1
    labels, label_ids = label_table(tok)
    if layout == "schema_first":
        ids: list[int] = []
        for k, (text, options) in enumerate(questions):
            ids += _question_block(tok, k, text, options, multi, "\n\n" if k else "", label_ids)
        prefix_len = len(ids)
        ids += tok.encode("\n\nContext:\n", add_special_tokens=False)
        ids += tok.encode(state, add_special_tokens=False)[:max_state_tokens]
        slots: list[int] = []
        for k in range(len(questions)):
            lead = "\n\n" if k == 0 else "\n"
            ids += tok.encode(f"{lead}Answer{' ' + str(k + 1) if multi else ''}: (", add_special_tokens=False)
            slots.append(len(ids) - 1)
        return {"ids": ids, "slots": slots, "n_options": [len(o) for _, o in questions],
                "prefix_len": prefix_len}
    ids = tok.encode("Context:\n" + state, add_special_tokens=False)[: max_state_tokens + 3]
    slots = []
    for k, (text, options) in enumerate(questions):
        ids += _question_block(tok, k, text, options, multi, "\n\n", label_ids)
        ids += tok.encode(f"\nAnswer{' ' + str(k + 1) if multi else ''}: (", add_special_tokens=False)
        slots.append(len(ids) - 1)
    return {"ids": ids, "slots": slots, "n_options": [len(o) for _, o in questions], "prefix_len": 0}


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


def _slot_rows(outputs) -> list:
    """The rows of the one read under schema_first, through whatever single-branch wrapper the engine hands."""
    value = outputs[0]
    while (isinstance(value, list) and len(value) == 1 and isinstance(value[0], list)
           and value[0] and isinstance(value[0][0], list)):
        value = value[0]
    return value


@register("thisthat")
class ThisThat(Format):
    """flock-io/this-that-model: one state, N questions, N read positions, one pass."""

    name = "thisthat"
    backbones = ("qwen3_5", "qwen3_5_text")

    def __init__(self, model_dir: str, tokenizer=None, temperature: float | None = None,
                 max_state_tokens: int = DEFAULT_MAX_STATE_TOKENS, layout: str = "state_first", **_):
        import transformers

        self.model_dir = model_dir
        self.tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.labels, self.label_ids = label_table(self.tokenizer)
        self.temperature = DEFAULT_TEMPERATURE if temperature is None else float(temperature)
        self.max_state_tokens = int(max_state_tokens)
        if layout not in ("state_first", "schema_first"):
            raise ValueError(f"layout must be 'state_first' or 'schema_first', got {layout!r}")
        self.layout = layout
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
        built = build(self.tokenizer, state, questions, layout=self.layout,
                      max_state_tokens=self.max_state_tokens)
        ids = built["ids"]
        offsets = tuple(slot - len(ids) for slot in built["slots"])
        if self.layout == "schema_first":
            cut = built["prefix_len"]
            return Plan(ids[:cut], [ids[cut:]], (Read(READ_HEAD, positions=offsets),), data)
        return Plan([], [ids], (Read(READ_HEAD, positions=offsets),), data)

    def finish(self, request: SystemOneRequest, plan: Plan, outputs, **_):
        rows = _slot_rows(outputs) if self.layout == "schema_first" \
            else [out[0][0] for out in outputs]   # one row per answer slot, in question order
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
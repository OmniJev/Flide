"""Turion Jev (turion-ai/turion-jev-31b: Gemma 4 31B, causal, an option-code readout) as its release runs the Decision
Index (code/turion_jev: predictor.py with prompt.py, the official options order_avg_max 2, second_pass_max_questions 1,
repeat_max_questions 1, recheck off; layout, slots and read-twice length from prompt_config.json).

Every question is one prompt in the chat template, thinking off: a system line, then the user turn with every question
of the request listed before the state ("Questions about this state:"), the state, the question with a hint for its
type, its options coded A..Z and then two-letter codes (skipping codes that read as English words), and a closing line.
A request with one question whose user turn has at most `repeat_max_chars` characters (12,000) has that turn written
twice. The answer is the next-token distribution over the codes shown, divided by the temperature calibration.json
gives its type. A noul question shows "yes" and "no" whatever its criteria; its probability is "yes"'s. A choice
question with two options in a one-question request is also read with its options reversed, and the two are averaged;
a question with one option answers it with certainty. Each prompt is tokenized whole, as the release tokenizes it,
and the questions of a request fork from the tokens their prompts share. Prompts longer than 65,536 tokens are refused.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from flide.api.answers import make_choice, make_noul, make_score
from flide.api.schema import SystemOneRequest, SystemOneResponse, Usage
from flide.formats import register
from flide.formats.base import Format, Plan, Read
from flide.formats.next_token import TokenRows
from flide.formats.tokens import Pieces, common_start

REPOS = ("turion-ai/turion-jev-31b",)
FILES = ("turion_jev.json", "prompt_config.json")
SYSTEM_V2 = ("You are a decision engine. Read the state, then answer the question by choosing "
             "exactly one option. Reply with the option code only.")
HINT = {"choice": "Choose the single best option.", "noul": "Answer yes or no.",
        "score": "Choose the score that fits best (0 = lowest, 5 = highest)."}
WORDS = {"AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "HI", "IF", "IN", "IS", "IT", "ME", "MY", "NO", "OF",
         "OH", "OK", "ON", "OR", "OX", "SO", "TO", "UP", "US", "WE", "YE", "YO"}
YES_NO = [("yes", ""), ("no", "")]
MAX_CODES = 255
READ = Read("codes")


def _string(x: Any) -> str:
    """The release's as_instruction_string and as_state_string."""
    if x is None:
        return ""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def _options(q) -> list[tuple[str, str]]:
    """(key, description) of each option, as the release's _parse_question reads them."""
    if q.type == "noul":
        return list(YES_NO)
    if q.type == "choice":
        return [(str(k), _string(v) if v is not None else "") for k, v in q.criteria.items()]
    out = []
    for c in q.criteria:
        if isinstance(c, dict):
            out.append((str(c.get("key") or c.get("id") or c.get("name")),
                        _string(c.get("desc") or c.get("description") or "")))
        else:
            out.append((str(c), ""))
    return out


def render_user(kind: str, state: str, instructions: str, options: list[tuple[str, str]], codes: list[str],
                asked: list[str]) -> str:
    """The release's render_user, layout v2."""
    state_block = state if state.strip() else "(empty)"
    question = instructions.strip() or "(no instructions)"
    lines = []
    for code, (key, desc) in zip(codes, options):
        lines.append(f"{code}) {key}: {desc}" if desc and desc.strip() != key.strip() else f"{code}) {key}")
    shown = [a.strip() for a in asked if a and a.strip()] or [question]
    header = "\n".join(f"- {a}" for a in shown)
    return (f"Questions about this state:\n{header}\n\n"
            f"State:\n{state_block}\n\n"
            f"Question:\n{question}\n"
            f"{HINT.get(kind, HINT['choice'])}\n\n"
            "Options:\n" + "\n".join(lines) + "\n\n"
            f"Answer with the code of one option ({codes[0]} to {codes[len(options) - 1]}).")


def _softmax(xs: list[float], t: float) -> list[float]:
    z = [x / t for x in xs]
    m = max(z)
    w = [math.exp(x - m) for x in z]
    s = sum(w)
    return [x / s for x in w]


@register("turion")
class Turion(Format):
    backbones = ("gemma4", "gemma4_text", "gemma4_unified")

    def __init__(self, model_dir: str, tokenizer=None, max_len: int = 65536, order_avg_max: int = 2,
                 second_pass_max_questions: int = 1, repeat_max_questions: int = 1):
        import transformers

        d = Path(model_dir)
        pcfg = {"repeat_max_chars": 0, "layout": "v1", "max_slots": 16,
                **json.loads((d / "prompt_config.json").read_text())}
        if pcfg["layout"] != "v2":
            raise ValueError(f"{model_dir}: prompt layout {pcfg['layout']!r}; this family reads layout v2")
        self.repeat_max_chars = int(pcfg["repeat_max_chars"])
        self.n_slots = int(pcfg["max_slots"])
        cal = d / "calibration.json"
        temps = json.loads(cal.read_text()).get("temperatures", {}) if cal.is_file() else {}
        self.temperature = {"choice": 1.0, "noul": 1.0, "score": 1.0, **{k: float(v) for k, v in temps.items()}}
        self.max_len = int(max_len)
        self.order_avg_max = int(order_avg_max)
        self.second_pass_max_questions = int(second_pass_max_questions)
        self.repeat_max_questions = int(repeat_max_questions)
        self.tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self._pieces = Pieces(self.tokenizer)
        self.codes, self.code_ids = self._codes(self.n_slots)

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        d = Path(model_dir)
        parts = {p.lower() for p in d.resolve().parts}
        if any(f"models--{r.lower().replace('/', '--')}" in parts for r in REPOS):
            return True
        return all((d / f).is_file() for f in FILES) and (d / "code" / "turion_jev").is_dir()

    def _enc(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _codes(self, n: int) -> tuple[list[str], list[int]]:
        """The release's option_codes: A..Z, then two-letter codes, each one token alone and distinct."""
        singles = [chr(ord("A") + i) for i in range(26)]
        codes, ids = [], []
        for c in singles + [a + b for a in singles for b in singles if a + b not in WORDS]:
            if len(codes) == n:
                break
            toks = self.tokenizer.encode(c, add_special_tokens=False)
            if len(toks) == 1 and toks[0] not in ids:
                codes.append(c)
                ids.append(toks[0])
            elif len(c) == 1:
                raise ValueError(f"letter {c!r} is not a single token for this tokenizer")
        if len(codes) < n:
            raise ValueError(f"only {len(codes)} single-token option codes, need {n}")
        return codes, ids

    def token_heads(self) -> dict[str, list[int]]:
        return {"codes": list(self.code_ids)}

    def load_heads(self, backbone) -> dict[str, Any]:
        weight, bias = backbone.output_head()
        return {"codes": TokenRows(weight, self.code_ids, bias, softcap=getattr(backbone, "logit_softcap", None))}

    def prompt(self, kind: str, state: str, instructions: str, options: list[tuple[str, str]], asked: list[str],
               repeat: int) -> str:
        """The release's build_prompt, layout v2."""
        user = render_user(kind, state, instructions, options, self.codes, asked)
        if repeat and len(user) <= repeat:
            user = f"{user}\n\nThe same request again:\n\n{user}"
        return self.tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_V2},
                                                   {"role": "user", "content": user}],
                                                  tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def plan(self, request: SystemOneRequest, **_) -> Plan:
        if getattr(request, "images", None):
            raise ValueError("images: this model reads text only")
        state = _string(request.state)
        qs = []
        for qid, q in request.questions.items():
            opts = _options(q)
            if len(opts) > MAX_CODES:
                raise ValueError(f"question '{qid}' has {len(opts)} options (max {MAX_CODES})")
            qs.append({"qid": qid, "type": q.type, "text": _string(q.instructions or ""), "options": opts, "n": len(opts)})
        asked = [q["text"] for q in qs]
        repeat = 0 if self.repeat_max_questions and len(asked) > self.repeat_max_questions else self.repeat_max_chars
        second = not (self.second_pass_max_questions and len(qs) > self.second_pass_max_questions)
        jobs = []   # (question index, reversed): the release's first round, then its reversed-order round
        for i, q in enumerate(qs):
            if q["n"] > 1:
                jobs.append((i, False))
        for i, q in enumerate(qs):
            if second and q["type"] == "choice" and 2 <= q["n"] <= self.n_slots and q["n"] <= self.order_avg_max:
                jobs.append((i, True))
        if not jobs:   # every question has one option: one read, so that the round is not empty; its answer is set
            jobs.append((0, False))
        texts = []
        for i, rev in jobs:
            q = qs[i]
            opts = list(reversed(q["options"])) if rev else q["options"]
            texts.append(self.prompt(q["type"], state, q["text"], opts, asked, repeat))
        head, tails = self._pieces.rows(self.tokenizer, texts)   # the state is tokenized once
        for t in tails:
            if len(head) + len(t) > self.max_len:
                raise ValueError(f"prompt of {len(head) + len(t)} tokens exceeds max_len {self.max_len}")
        m = common_start(tails) if len(tails) > 1 else len(tails[0]) - 1
        return Plan(head + tails[0][:m], [t[m:] for t in tails], (READ,), {"questions": qs, "jobs": jobs})

    def finish(self, request: SystemOneRequest, plan: Plan, outputs, **_):
        qs, jobs = plan.data["questions"], plan.data["jobs"]
        probs: dict[int, list[float]] = {}
        for (i, rev), out in zip(jobs, outputs):
            q = qs[i]
            if q["n"] == 1:
                continue
            p = _softmax(list(out[0][0][: q["n"]]), self.temperature.get(q["type"], 1.0))
            if rev:
                probs[i] = [0.5 * (a + b) for a, b in zip(probs[i], p[::-1])]
            else:
                probs[i] = p
        answers = {}
        for i, q in enumerate(qs):
            p = probs.get(i, [1.0])
            if q["type"] == "noul":
                answers[q["qid"]] = make_noul(p[0])
            elif q["type"] == "choice":
                answers[q["qid"]] = make_choice([k for k, _ in q["options"]], p)
            else:
                answers[q["qid"]] = make_score(list(request.questions[q["qid"]].criteria), p)
        usage = Usage(input_tokens=plan.tokens, output_tokens=len(jobs))
        return SystemOneResponse(model="", answers=answers, usage=usage), {"format": "turion",
                                                                          "prefix_tokens": len(plan.prefix),
                                                                          "branches": len(plan.suffixes)}

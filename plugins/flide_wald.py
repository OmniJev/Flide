"""Wald family (org2ai/Wald-26B-A4B on Gemma 4 26B-A4B): the release's own reader, reference/eval/systemone_vllm.py in
the setting serving.json and run.sh give it (--template chat --prompt-format plain --wide knockout, effort none, no
temperature table, --noul-decisive-floor 0.81), reading the letters as its vLLM server scores them.

The request is flattened as its kev/api.py renders it: a string, number or boolean as it is, an object as "key: value"
lines (nested values indented below their key), an array as "- item" lines; a choice option is its name, or "name:
description"; a noul's options are "no" and "yes" with the descriptions the request gives them; a score's are its
levels. A choice whose keys only number the options (a, b, c or o27) drops the "a: " before each description. Every
"<|name|>" in the text becomes "<¦name¦>". Every question is its own chat prompt, thinking off: a fixed system line,
then one user turn with "State:" and the state (left out when blank), "Question:" and the instructions, the options
lettered A. to Z., and "Answer with the letter of one option."

A letter's logit is the log-sum-exp of its one-token forms ("L", " L", and L after "(" when that is a token of its
own), and the answer is the softmax over the question's letters. A choice of more than 26 options is read by
knockout: chunks of near-equal size, at most 26, each read alone, then a final read over the chunk winners; an option's
probability is its chunk's final probability times its own within the chunk, renormalized. A noul's probability of yes
strictly between 0.19 and 0.81 moves to 0.81 or 0.19 on its side (0.5 to 0.19). A score's confidence is the
probability of its most likely level. The state comes first, so the questions fork from it.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from flide.api.answers import _round, make_choice, make_noul, make_score
from flide.api.schema import SystemOneRequest, SystemOneResponse, Usage
from flide.formats import register
from flide.formats.base import Format, Plan, Read
from flide.formats.next_token import TokenRows
from flide.formats.tokens import Pieces

ENGINE = "systemone-vllm-gemma-chat"
SYSTEM = ("You are a calibration engine inside a decision system. You are given a state, a question and lettered "
          "options. Choose exactly one option and reply with its letter only: one capital letter, no words, no "
          "punctuation, no explanation.")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_OPTIONS = 255
NOUL_FLOOR = 0.81
SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
POSITIONAL = re.compile(r"[a-z]|o[0-9]+")
READ = Read("letters")


def render(v: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}"
                     for k, x in v.items())


def option_text(name: str, desc: Any) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def plain(text: Any) -> str:
    return SPECIAL.sub(r"<¦\1¦>", str(text))


def options_of(q) -> tuple[list[str], list[str]]:
    """(the keys the answer reports, the option strings after the letters)."""
    if q.type == "noul":
        c = q.criteria
        return ["false", "true"], [option_text("no", None if c is None else c.false),
                                   option_text("yes", None if c is None else c.true)]
    if q.type == "choice":
        keys = [str(k) for k in q.criteria]
        texts = [option_text(k, q.criteria[k]) for k in keys]
        if all(POSITIONAL.fullmatch(k) for k in keys):
            texts = [t[len(k) + 2:] if t.startswith(f"{k}: ") else t for k, t in zip(keys, texts)]
        return keys, texts
    return [str(i) for i in range(len(q.criteria))], [render(x) for x in q.criteria]


def user_message(state: str, instructions: str, options: list[str]) -> str:
    head = f"State:\n{plain(state)}\n\n" if str(state).strip() else ""
    lines = [f"{LETTERS[j]}. {plain(o)}" for j, o in enumerate(options)]
    return (head + f"Question: {plain(instructions)}\nOptions:\n" + "\n".join(lines)
            + "\n\nAnswer with the letter of one option.")


def chunks(n: int, size: int = 26) -> list[list[int]]:
    k = math.ceil(n / size)
    base, extra = divmod(n, k)
    out, i = [], 0
    for c in range(k):
        m = base + (1 if c < extra else 0)
        out.append(list(range(i, i + m)))
        i += m
    return out


@register("wald")
class Wald(Format):
    backbones = ("gemma4", "gemma4_text", "gemma4_unified")
    CUT_CHECKS = 8

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        p = Path(model_dir) / "serving.json"
        try:
            return p.is_file() and json.loads(p.read_text()).get("engine") == ENGINE
        except ValueError:
            return False

    def __init__(self, model_dir: str, tokenizer=None, max_model_len: int = 131072, noul_floor: float = NOUL_FLOOR):
        import transformers

        self.model_dir = model_dir
        self.tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.max_len = int(max_model_len)
        self.floor = float(noul_floor)
        self.forms = self._letter_forms()
        self.row_ids = sorted({t for f in self.forms for t in f})
        index = {t: i for i, t in enumerate(self.row_ids)}
        self.form_rows = [[index[t] for t in f] for f in self.forms]
        self._pieces = Pieces(self.tokenizer)
        self._cut_checks = 0

    def _letter_forms(self) -> list[list[int]]:
        """The token ids read for each letter, as its Client finds them: "L" and " L" when one token each, and L as
        it encodes after "(" when that is another token."""
        enc = lambda s: self.tokenizer.encode(s, add_special_tokens=False)
        paren = enc("(")
        out = []
        for L in LETTERS:
            ids = []
            for v in (L, " " + L):
                t = enc(v)
                if len(t) == 1 and t[0] not in ids:
                    ids.append(t[0])
            t = enc("(" + L)
            if len(t) == len(paren) + 1 and t[:-1] == paren and t[-1] not in ids:
                ids.append(t[-1])
            if not ids:
                raise ValueError(f"no single-token encoding of letter {L!r}")
            out.append(ids)
        return out

    def token_heads(self) -> dict[str, list[int]]:
        return {"letters": list(self.row_ids)}

    def load_heads(self, backbone) -> dict[str, Any]:
        weight, bias = backbone.output_head()
        return {"letters": TokenRows(weight, self.row_ids, bias, softcap=getattr(backbone, "logit_softcap", None))}

    def _text(self, state: str, instructions: str, options: list[str]) -> str:
        """The prompt as its vLLM server renders it: messages through the chat template, each content a list of text
        parts (vLLM's choice for a template that walks its parts)."""
        part = lambda t: [{"type": "text", "text": t}]
        chat = [{"role": "system", "content": part(SYSTEM)},
                {"role": "user", "content": part(user_message(state, instructions, options))}]
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)

    def _plan(self, request: SystemOneRequest, rows: list[tuple], data: dict) -> Plan:
        """rows: (question id, option indices or None, what the read stands for); one branch each, forked from the
        state."""
        state = render(request.state)
        texts = []
        for qid, idx, _ in rows:
            q = request.questions[qid]
            _, opts = options_of(q)
            if idx is not None:
                opts = [opts[i] for i in idx]
            texts.append(self._text(state, render(q.instructions), opts))
        shared = texts[0]
        for t in texts[1:]:
            n = 0
            while n < min(len(shared), len(t)) and shared[n] == t[n]:
                n += 1
            shared = shared[:n]
        c = self._pieces.last_cut(shared, len(shared) - 1) if len(shared) > 1 else 0
        head, tail, *rest = self._pieces.encode(self.tokenizer, [shared[:c], shared[c:]] + [t[c:] for t in texts])
        if c and self._cut_checks < self.CUT_CHECKS:
            self._cut_checks += 1
            if head + rest[0] != self.tokenizer.backend_tokenizer.encode(texts[0], add_special_tokens=False).ids:
                self._pieces.ok = False
                head, tail, *rest = self._pieces.encode(self.tokenizer, ["", ""] + texts)
        longest = len(head) + max(len(r) for r in rest)
        if longest + 1 > self.max_len:
            raise ValueError(f"prompt of {longest} tokens is longer than the maximum context length {self.max_len}")
        m = min(len(r) for r in rest) - 1
        for i in range(m):
            if any(r[i] != rest[0][i] for r in rest):
                m = i
                break
        return Plan(head + rest[0][:m], [r[m:] for r in rest], (READ,), {**data, "rows": rows})

    def plan(self, request: SystemOneRequest, **_) -> Plan:
        if getattr(request, "images", None):
            raise ValueError("images: the wald family reads text only")
        rows, wide = [], {}
        for qid, q in request.questions.items():
            keys, _ = options_of(q)
            if not 1 <= len(keys) <= MAX_OPTIONS:
                raise ValueError(f"{len(keys)} options, limit is 1 to {MAX_OPTIONS}")
            if len(keys) > len(LETTERS):
                if q.type != "choice":
                    raise ValueError(f"{len(keys)} levels: the letter readout has {len(LETTERS)} letters")
                wide[qid] = {"groups": chunks(len(keys)), "within": None, "final": None}
                rows += [(qid, g, ("chunk", qid, i)) for i, g in enumerate(wide[qid]["groups"])]
            else:
                rows.append((qid, None, ("plain", qid)))
        return self._plan(request, rows, {"wide": wide, "plain": {}, "tokens": 0})

    def _probs(self, out, n: int) -> list[float]:
        logits = out[0][0]
        z = []
        for rows in self.form_rows[:n]:
            m = max(logits[r] for r in rows)
            z.append(m + math.log(sum(math.exp(logits[r] - m) for r in rows)))
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        return [x / s for x in e]

    def _absorb(self, request: SystemOneRequest, plan: Plan, outputs) -> dict:
        d = plan.data
        for (qid, idx, ref), out in zip(d["rows"], outputs):
            n = len(idx) if idx is not None else len(options_of(request.questions[qid])[0])
            p = self._probs(out, n)
            if ref[0] == "plain":
                d["plain"][qid] = p
            elif ref[0] == "chunk":
                node = d["wide"][qid]
                node["within"] = node["within"] or [None] * len(node["groups"])
                node["within"][ref[2]] = p
            else:
                d["wide"][qid]["final"] = p
        d["tokens"] += plan.tokens
        return d

    def follow_up(self, request: SystemOneRequest, plan: Plan, outputs, **_) -> Plan | None:
        d = self._absorb(request, plan, outputs)
        rows = []
        for qid, node in d["wide"].items():
            if node["final"] is None and node["within"] is not None and all(w is not None for w in node["within"]):
                winners = [g[max(range(len(g)), key=lambda j: w[j])] for g, w in zip(node["groups"], node["within"])]
                node["winners"] = winners
                rows.append((qid, winners, ("final", qid)))
        if not rows:
            d["done"] = True
            return None
        return self._plan(request, rows, d)

    def finish(self, request: SystemOneRequest, plan: Plan, outputs, **_):
        d = plan.data if plan.data.get("done") else self._absorb(request, plan, outputs)
        answers = {}
        for qid, q in request.questions.items():
            keys, _ = options_of(q)
            if qid in d["wide"]:
                node = d["wide"][qid]
                p = [0.0] * len(keys)
                for c, (g, w) in enumerate(zip(node["groups"], node["within"])):
                    for j, i in enumerate(g):
                        p[i] = node["final"][c] * w[j]
                s = sum(p)
                p = [x / s for x in p]
            else:
                p = d["plain"][qid]
            if q.type == "noul":
                y = p[1]
                low = round(1.0 - self.floor, 12)
                if self.floor and low < y < self.floor:
                    y = self.floor if y > 0.5 else low
                answers[qid] = make_noul(y)
            elif q.type == "choice":
                answers[qid] = make_choice(keys, p)
            else:
                a = make_score(list(q.criteria), p)
                answers[qid] = a.model_copy(update={"confidence": _round(max(p))})
        usage = Usage(input_tokens=d["tokens"], output_tokens=len(answers))
        return SystemOneResponse(model="", answers=answers, usage=usage), {"format": "wald",
                                                                          "prefix_tokens": len(plan.prefix)}

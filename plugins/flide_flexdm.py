"""FlexDM family (1000luca/FlexDM-LoRA-v2.1, a LoRA adapter on google/gemma-4-31B-it): the release's VllmLoraReader as
its README runs it, in the header layout, read twice up to 3,000 tokens, options reversed below 0.6.

Every question is its own prompt in Gemma's chat template, thinking off, one user turn: the list of the request's
questions ("Questions that will be asked about the context:", one "- " line each), "Context:" and the state (a string as
it is, anything else JSON, arrays of 8 or more elements written with their positions), then the question and its
options, (A) to (J), or past 10 options each option's single-token label (A..Z, then two letters) tokenized apart. The
reply's "Answer: (" closes it, and the answer is the softmax over the option labels' soft-capped logits there, divided
by T = 10.9757 - 1.7016 ln n for a choice of n options and by 10.732 for a noul (no, then yes). A request whose prompts,
read once, come to at most 3,000 tokens is read with every user turn written twice, the copy opened by "Read the same
input again before answering:". A question whose top probability is then below 0.6 is read once more with its options
in reverse order, written the same way, and the two distributions are averaged. Everything through the state is the
same for every question, so the questions fork from it, and the reversed round forks from it again. Text only; choice
(2 to 255 options) and noul questions; prompts of up to 131,071 tokens.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import string
from pathlib import Path
from typing import Any

from flide import lora
from flide.api.answers import make_choice, make_noul
from flide.api.schema import SystemOneRequest, SystemOneResponse, Usage
from flide.formats import register
from flide.formats.base import Format, Plan, Read
from flide.formats.next_token import TokenRows

REPOS = ("1000luca/FlexDM-LoRA-v2.1",)
LETTERS = "ABCDEFGHIJ"
MAX_OPTIONS = 255
ANNOTATE_MIN = 8
NOUL_WITHOUT_INSTRUCTIONS = "Which answer fits the context?"
READ_AGAIN = "\n\nRead the same input again before answering:\n"
QUESTIONS_HEAD = "Questions that will be asked about the context:\n"
SENTINEL = "@@FLEXDM_USER_CONTENT@@"
THINKING_BLOCKS = (("<think>\n", "</think>\n\n"), ("<think>", "</think>\n\n"), ("<|channel>thought\n", "<channel|>"))
T_CHOICE, T_CHOICE_SLOPE, T_NOUL, NOUL_BIAS = 10.9757, -1.7016, 10.732, 0.0
READ = Read("labels")
log = logging.getLogger(__name__)


def _txt(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def annotate_indices(x: Any, min_len: int = ANNOTATE_MIN) -> Any:
    if isinstance(x, list):
        if len(x) >= min_len:
            return [{"_index": i, **annotate_indices(v, min_len)} if isinstance(v, dict)
                    else {"_index": i, "value": annotate_indices(v, min_len)} for i, v in enumerate(x)]
        return [annotate_indices(v, min_len) for v in x]
    if isinstance(x, dict):
        return {k: annotate_indices(v, min_len) for k, v in x.items()}
    return x


def render_state(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(annotate_indices(state), ensure_ascii=False)


def render_question(q) -> tuple[str, list[str], list[str]]:
    """(question text, option texts, option keys); a noul is the options no and yes."""
    raw = q.instructions if "instructions" in q.model_fields_set else ""
    if q.type == "choice":
        if not 2 <= len(q.criteria) <= MAX_OPTIONS:
            raise ValueError(f"options per choice: a map of 2..{MAX_OPTIONS} options is required")
        keys = list(q.criteria)
        return _txt(raw), [k if d in (None, "") else f"{k}: {_txt(d)}" for k, d in q.criteria.items()], keys
    if q.type == "noul":
        c = q.criteria
        no, yes = (None, None) if c is None else (c.false, c.true)
        text = NOUL_WITHOUT_INSTRUCTIONS if raw in (None, "") else _txt(raw)
        return text, ["no" if no in (None, "") else f"no: {_txt(no)}",
                      "yes" if yes in (None, "") else f"yes: {_txt(yes)}"], ["false", "true"]
    raise ValueError(f"question type {q.type}: the flexdm family reads choice and noul")


def _softmax(x: list[float]) -> list[float]:
    m = max(x)
    w = [math.exp(v - m) for v in x]
    s = sum(w)
    return [v / s for v in w]


def _common(seqs: list[list[int]]) -> int:
    """Tokens every sequence starts with, leaving at least one token in each."""
    m = min(len(s) for s in seqs) - 1
    for i in range(m):
        t = seqs[0][i]
        if any(s[i] != t for s in seqs):
            return i
    return max(m, 0)


@register("flexdm")
class FlexDM(Format):
    backbones = ("gemma4",)

    def __init__(self, model_dir: str, tokenizer=None, read_twice: bool = True, twice_max_tokens: int = 3000,
                 reverse_below: float = 0.6, max_tokens: int = 131071):
        import transformers

        self.model_dir = model_dir
        self.tokenizer = tok = tokenizer or transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.read_twice, self.twice_max_tokens = bool(read_twice), int(twice_max_tokens)
        self.reverse_below, self.limit = float(reverse_below), int(max_tokens)
        if not getattr(tok, "chat_template", None):
            raise ValueError(f"{model_dir}: the tokenizer has no chat template")
        kw = {"enable_thinking": False} if "enable_thinking" in tok.chat_template else {}
        head, tail = tok.apply_chat_template([{"role": "user", "content": SENTINEL}], tokenize=False,
                                             add_generation_prompt=True, **kw).split(SENTINEL)
        for opened, closed in THINKING_BLOCKS:
            if tail.endswith(opened):
                tail += closed
        self.head, self.tail = self._enc(head), self._enc(tail)
        self.answer = self._enc("Answer: (")
        self.open = self._enc("\n(")
        names = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
        single = [(n, t[0]) for n, t in ((n, self._enc(n)) for n in names) if len(t) == 1][:MAX_OPTIONS]
        self.labels, self.label_ids = [n for n, _ in single], [i for _, i in single]
        if len(set(self.label_ids)) != MAX_OPTIONS:
            raise ValueError(f"{model_dir}: the tokenizer does not have {MAX_OPTIONS} distinct single-token labels")
        for n, i in single[:len(LETTERS)]:
            if self._enc("Answer: (" + n) != self.answer + [i]:
                raise ValueError(f"{model_dir}: the label {n!r} is not one token after 'Answer: ('")
        if self._enc(tail + "Answer: (") != self.tail + self.answer:
            raise ValueError(f"{model_dir}: the chat template's tail merges with the answer slot")

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        path = str(Path(model_dir).resolve()).lower()
        return any(f"models--{r.lower().replace('/', '--')}" in path for r in REPOS)

    @classmethod
    def run_dir(cls, model_dir: str) -> str:
        """The adapter as transformers reads it: the release's served copy repeats each k_proj LoRA as v_proj on Gemma
        4's full-attention layers, which read their keys as values and have no v_proj (vLLM gives the V slice its own
        LoRA). Those copies, each checked equal to its layer's k_proj LoRA, are left out of a copy linked once under
        FLIDE_CACHE; a directory without them runs as it is."""
        src = Path(model_dir).resolve()
        peft = src / lora.SUB
        copies = cls._value_copies(src, peft) if (peft / lora.WEIGHTS).is_file() else set()
        if not copies:
            return model_dir
        key = hashlib.sha256(f"{src}\nk_eq_v".encode()).hexdigest()[:16]
        out = Path(os.environ.get("FLIDE_CACHE", Path.home() / ".cache" / "flide")) / lora.SUB / \
            lora._hub_name(src) / key
        if (out / ".done").is_file():
            return str(out)
        out.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.name == lora.SUB or f.name.startswith("."):
                continue
            link = out / f.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(f.resolve())
        (out / lora.SUB).mkdir(exist_ok=True)
        for name in (lora.CONFIG, lora.WEIGHTS):
            link = out / lora.SUB / name
            if link.is_symlink() or link.exists():
                link.unlink()
            if name == lora.WEIGHTS:
                from safetensors import safe_open
                from safetensors.torch import save_file

                with safe_open(str(peft / name), framework="pt") as h:
                    save_file({k: h.get_tensor(k) for k in h.keys() if k not in copies}, str(link), metadata=h.metadata())
                log.info("%s: %d v_proj LoRA tensors of key-as-value layers left out (copies of k_proj's)", src.name,
                         len(copies))
            else:
                link.symlink_to((peft / name).resolve())
        (out / ".done").write_text(f"{src}\n")
        return str(out)

    @staticmethod
    def _value_copies(src: Path, peft: Path) -> set[str]:
        """The adapter's v_proj LoRA tensors on the layers that read their keys as values (attention_k_eq_v, the
        full-attention layers), each equal to its layer's k_proj LoRA, or a ValueError."""
        cfg = json.loads((src / "config.json").read_text())
        text = cfg.get("text_config") or cfg
        if not text.get("attention_k_eq_v"):
            return set()
        import torch
        from safetensors import safe_open

        full = {i for i, t in enumerate(text.get("layer_types") or []) if t == "full_attention"}
        copies = set()
        with safe_open(str(peft / lora.WEIGHTS), framework="pt") as h:
            keys = set(h.keys())
            for k in sorted(keys):
                m = re.search(r"\.layers\.(\d+)\.self_attn\.v_proj\.lora_[AB]\.weight$", k)
                if m and int(m[1]) in full:
                    kk = k.replace(".v_proj.", ".k_proj.")
                    if kk not in keys or not torch.equal(h.get_tensor(k), h.get_tensor(kk)):
                        raise ValueError(f"{peft}: {k} is not its layer's k_proj LoRA, and that layer reads its keys as "
                                         "values")
                    copies.add(k)
        return copies

    def _enc(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def token_heads(self) -> dict[str, list[int]]:
        return {"labels": list(self.label_ids)}

    def load_heads(self, backbone) -> dict[str, Any]:
        weight, bias = backbone.output_head()
        return {"labels": TokenRows(weight, self.label_ids, bias, softcap=getattr(backbone, "logit_softcap", None))}

    @staticmethod
    def listing(texts: list[str]) -> str:
        return QUESTIONS_HEAD + "\n".join("- " + t for t in texts) + "\n\n"

    def question(self, text: str, options: list[str], again: list[int] | None = None) -> list[int]:
        """The ids after the state: the question, its labeled options (with `again`, the user turn written a second
        time), the template's tail and the answer slot."""
        ids = self._enc(f"\n\nQuestion: {text}\nOptions:")
        if len(options) <= len(LETTERS):
            ids += self._enc("".join(f"\n({LETTERS[j]}) {o}" for j, o in enumerate(options)))
        else:
            for j, o in enumerate(options):
                ids += self.open + [self.label_ids[j]] + self._enc(f") {o}")
        if again:
            ids = ids + again + ids
        return ids + self.tail + self.answer

    def _check(self, ctx: list[int], piece: list[int]) -> None:
        if len(ctx) + len(piece) > self.limit:
            raise ValueError(f"prompt is {len(ctx) + len(piece)} tokens, limit is {self.limit}")

    def plan(self, request: SystemOneRequest, **_) -> Plan:
        if getattr(request, "images", None):
            raise ValueError("images: the flexdm family reads text only")
        rendered = [(qid, q.type, *render_question(q)) for qid, q in request.questions.items()]
        texts = [text for _, _, text, _, _ in rendered]
        state = render_state(request.state)
        ctx = self.head + self._enc(self.listing(texts)) + self._enc("Context:\n" + state)
        again = self._enc(READ_AGAIN + self.listing(texts) + "Context:\n" + state) if self.read_twice else None
        once = [self.question(text, options) for _, _, text, options, _ in rendered]
        twice = [self.question(text, options, again) for _, _, text, options, _ in rendered] if again else None
        whole = (again is not None and all(len(ctx) + len(t) <= self.limit for t in twice)
                 and (self.twice_max_tokens <= 0 or len(ctx) + sum(map(len, once)) <= self.twice_max_tokens))
        first = twice if whole else once
        for piece in first:
            self._check(ctx, piece)
        questions = [{"qid": qid, "type": kind, "text": text, "options": options, "keys": keys, "n": len(keys)}
                     for qid, kind, text, options, keys in rendered]
        data = {"round": 1, "ctx": ctx, "again": again if whole else None, "whole": whole, "questions": questions}
        return self._plan(ctx, first, data)

    def _plan(self, ctx: list[int], pieces: list[list[int]], data: dict) -> Plan:
        m = _common(pieces)
        return Plan(ctx + pieces[0][:m], [p[m:] for p in pieces], (READ,), {**data, "pieces": pieces})

    def _probs(self, logits, kind: str, reverse: bool = False) -> list[float]:
        n = len(logits)
        if kind == "noul":
            t, bias = T_NOUL, NOUL_BIAS
        else:
            t, bias = max(T_CHOICE + T_CHOICE_SLOPE * math.log(n), 0.05), 0.0
        x = [float(s) / t for s in logits]
        if bias:
            x[0 if reverse else 1] += bias
        return _softmax(x)

    def follow_up(self, request: SystemOneRequest, plan: Plan, outputs, **_) -> Plan | None:
        d = plan.data
        if d["round"] != 1 or self.reverse_below <= 0:
            return None
        first = [self._probs(out[0][0][:q["n"]], q["type"]) for q, out in zip(d["questions"], outputs)]
        low = [i for i, p in enumerate(first) if max(p) < self.reverse_below]
        if not low:
            return None
        rev = [self.question(d["questions"][i]["text"], d["questions"][i]["options"][::-1], d["again"]) for i in low]
        for piece in rev:
            self._check(d["ctx"], piece)
        return self._plan(d["ctx"], rev, {**d, "round": 2, "first": first, "low": low, "first_tokens": plan.tokens})

    def finish(self, request: SystemOneRequest, plan: Plan, outputs, **_):
        d = plan.data
        qs = d["questions"]
        if d["round"] == 1:
            probs = [self._probs(out[0][0][:q["n"]], q["type"]) for q, out in zip(qs, outputs)]
        else:
            probs = list(d["first"])
            for i, out in zip(d["low"], outputs):
                back = self._probs(out[0][0][:qs[i]["n"]], qs[i]["type"], reverse=True)[::-1]
                probs[i] = [(a + b) / 2 for a, b in zip(probs[i], back)]
        answers = {}
        for q, p in zip(qs, probs):
            answers[q["qid"]] = make_noul(p[1]) if q["type"] == "noul" else make_choice(q["keys"], p)
        reversed_ = len(d.get("low", ()))
        usage = Usage(input_tokens=d.get("first_tokens", 0) + plan.tokens, output_tokens=len(qs) + reversed_)
        return SystemOneResponse(model="", answers=answers, usage=usage), {"format": "flexdm",
                                                                          "read_twice": d["whole"],
                                                                          "prefix_tokens": len(plan.prefix),
                                                                          "branches": len(plan.suffixes),
                                                                          "reversed": reversed_}

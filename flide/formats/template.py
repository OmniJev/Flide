"""template: a decision checkpoint described by a configuration instead of code.

The configuration is the checkpoint's decision_config.json with "format_id": "flide-template-v1", or one that ships
with Flide for a known checkpoint (formats/known/*.json, matched by its Hub repository, or by its own files and the
fields of its own decision_config.json). It says how
the model is asked and read:

  messages     chat turns [{"role", "content"}], each content a Jinja template, rendered through the checkpoint's chat
               template with the generation prompt (thinking off unless "thinking" is true); "reply" is appended
  prompt       instead of messages, one Jinja template for the whole prompt, no chat template
  labels       a label set for every question type, or one per type {"choice": ..., "noul": ..., "score": ...}; a set
               is "A-Z+" (A to Z, then two-letter labels), "A-Z", "pairs" (two-letter labels only) or a list, or tiers
               [{"upto": 26, "labels": ...}, {"labels": ...}] chosen by the number of options; two-letter labels that are
               not one distinct token where they are read are left out, up to 256 labels a set
  label_ids    the token ids of one label list, when the release fixed them; else each label's token after the
               prompt, or with "label_tokens": "standalone" the label encoded alone (one token that decodes back to it)
  label_forms  several token forms per label, e.g. ["{label}", " {label}"]: each form that is one token alone is read
               and the label's logit is the log-sum-exp of its forms; a longer label is kept when its first form is
               new and is also its token after the prompt
  head         "output" (the output head, with its soft cap unless "softcap" is false) or a safetensors file of
               readout rows, one per label ("head_key" names the tensor, "weight" by default)
  temperature  one number, or one per question type ("choice", "noul", "score") over a "default"; a type's value
               {"a", "b"} makes it depend on the number of options n: log T = a + b ln n
  noul         the order of a noul's two options in the prompt, of "false" and "true" (["false", "true"])
  noul_text    their descriptions when a request gives none
  types, max_options, max_prompt_tokens
  state        the state as the prompt shows it, cut to a token budget: "text" (a Jinja template of state, the prompt
               reads it as state_text), "max_tokens", "min_tokens", "keep" ("head" or "tail", or one per state shape
               {"list": ..., "default": ...}) and "marks" (the text before and after the kept tokens, per side); while
               the longest prompt is over max_prompt_tokens the budget shrinks by the excess plus "slack", "tries" times
A value "@file#a.b" is key a.b of that JSON file of the checkpoint. A checkpoint the match does not find (a copy in
a directory of another name) is served with --format template --format-arg spec=<a file, or a shipped name>.

Template variables: state, state_text, question (key, type, instructions, criteria, given: the fields the request
wrote), options (label, key, description, index) in the prompt's order, questions (every question of the request, in
its order, as question is). Filters: describe (a string as it is, anything else
JSON), text (the same with compact separators) and json (indent, compact); test: matching(pattern); functions:
refuse(message), which turns the request down, boundary(), where the text before and after is tokenized apart, and
labels_for(type, n), the labels a question of that type and n options reads. Every question is its own prompt and the state comes first, so the questions fork where their
prompts part. The answer is the softmax over the question's labels at the last prompt position, divided by the
temperature of its type.
"""
from __future__ import annotations

import json
import math
import re
import string
import uuid
from pathlib import Path
from typing import Any

from ..api.answers import make_choice, make_noul, make_score
from ..api.schema import ChoiceQuestion, NoulQuestion, SystemOneRequest, SystemOneResponse, Usage
from . import register
from .base import Format, Plan, Read
from .next_token import TokenRows

FORMAT_ID = "flide-template-v1"
KNOWN = Path(__file__).resolve().parent / "known"
LETTERS = string.ascii_uppercase
READ = Read("labels")
TYPES = ("choice", "noul", "score")


def _describe(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _json(value: Any, indent: int | None = None, compact: bool = False) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent, separators=(",", ":") if compact else None)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _temp(v: Any) -> float | tuple[float, float]:
    return (float(v["a"]), float(v.get("b", 0.0))) if isinstance(v, dict) else float(v)


def _logsumexp(xs: list[float]) -> float:
    top = max(xs)
    return top + math.log(sum(math.exp(x - top) for x in xs))


def _refuse(message: str):
    raise ValueError(message)


def _given(model) -> dict:
    """The fields a request set on a pydantic object, as the request wrote them."""
    if model is None:
        return {}
    if isinstance(model, dict):
        return model
    return {k: getattr(model, k) for k in model.model_fields_set} | dict(model.model_extra or {})


def _value(v: Any, model_dir: str) -> Any:
    """A configuration value, with "@file#a.b" read from the checkpoint's own JSON files."""
    if isinstance(v, str) and v.startswith("@"):
        name, _, key = v[1:].partition("#")
        data = json.loads((Path(model_dir) / name).read_text())
        for k in key.split(".") if key else []:
            data = data[k]
        return data
    if isinstance(v, dict):
        return {k: _value(x, model_dir) for k, x in v.items()}
    if isinstance(v, list):
        return [_value(x, model_dir) for x in v]
    return v


def known(model_dir: str, decision_config: dict | None) -> dict | None:
    """The shipped configuration that claims a checkpoint: by its Hub repository in the path, else by its files."""
    path = str(Path(model_dir).resolve()).lower()
    for f in sorted(KNOWN.glob("*.json")):
        spec = json.loads(f.read_text())
        m = spec.get("match", {})
        if any(f"models--{r.lower().replace('/', '--')}" in path for r in m.get("repos", ())):
            return spec
        files, dc, keys = m.get("files", ()), m.get("decision_config"), m.get("decision_config_keys")
        if files and all((Path(model_dir) / x).is_file() for x in files) and (
                dc is None or (decision_config is not None and all(decision_config.get(k) == v for k, v in dc.items()))) \
                and (keys is None or (decision_config is not None and set(decision_config) == set(keys))):
            return spec
    return None


@register("template", format_ids=(FORMAT_ID,))
class Template(Format):
    backbones = ("qwen3_5", "qwen3_5_text", "gemma4", "gemma4_text", "gemma4_unified")

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        return known(model_dir, decision_config) is not None

    def __init__(self, model_dir: str, tokenizer=None, spec: dict | str | None = None):
        import jinja2
        import jinja2.sandbox
        import transformers

        self.model_dir = model_dir
        if isinstance(spec, str):
            spec = json.loads((Path(spec) if Path(spec).is_file() else KNOWN / f"{spec}.json").read_text())
        if spec is None:
            p = Path(model_dir) / "decision_config.json"
            dc = json.loads(p.read_text()) if p.is_file() else None
            spec = dc if dc and dc.get("format_id") == FORMAT_ID else known(model_dir, dc)
        if spec is None:
            raise ValueError(f"{model_dir}: no template configuration")
        self.spec = spec = {k: (v if k in ("messages", "prompt", "match") else _value(v, model_dir))
                            for k, v in spec.items()}
        self.tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(keep_trailing_newline=True, undefined=jinja2.StrictUndefined)
        env.filters["describe"] = _describe
        env.filters["json"] = _json
        env.filters["text"] = _text
        env.globals["refuse"] = _refuse
        env.globals["labels_for"] = lambda kind, n: self._labels(kind, n)
        self.boundary = f"\x00{uuid.uuid4().hex}\x00"
        env.globals["boundary"] = lambda: self.boundary
        env.tests["matching"] = lambda s, pattern: isinstance(s, str) and re.search(pattern, s) is not None
        if ("messages" in spec) == ("prompt" in spec):
            raise ValueError("a template configuration has either messages or prompt")
        self.messages = [(m["role"], env.from_string(m["content"])) for m in spec.get("messages", ())]
        self.prompt = env.from_string(spec["prompt"]) if "prompt" in spec else None
        self.reply = spec.get("reply", "")
        self.thinking = bool(spec.get("thinking", False))
        self.noul = list(spec.get("noul", ["false", "true"]))
        if sorted(self.noul) != ["false", "true"]:
            raise ValueError("noul lists false and true once each")
        self.noul_text = dict(spec.get("noul_text", {}))
        self.types = tuple(spec.get("types", TYPES))
        self.max_prompt_tokens = int(spec.get("max_prompt_tokens", 32768))
        st = spec.get("state")
        self.state_text = env.from_string(st["text"]) if st else None
        if st:
            keep = st.get("keep", "head")
            self.state_keep = keep if isinstance(keep, dict) else {"default": keep}
            self.state_marks = {k: tuple(v) for k, v in st.get("marks", {}).items()}
            self.state_budget = (int(st["max_tokens"]), int(st.get("min_tokens", 0)), int(st.get("slack", 0)),
                                 int(st.get("tries", 1)))
        t = spec.get("temperature", 1.0)
        self.temperature = {k: _temp(t.get(k, t.get("default", 1.0))) for k in TYPES} if isinstance(t, dict) \
            else {k: float(t) for k in TYPES}
        self.head = spec.get("head", "output")
        self.softcap = bool(spec.get("softcap", True))
        self.sets, self.row_ids = self._label_sets(spec)
        self.max_choice_options = min(len(self.sets["choice"][-1][1]), int(spec.get("max_options", 256)))
        self.max_score_levels = min(len(self.sets["score"][-1][1]), int(spec.get("max_options", 256)))

    def _render(self, request_state: Any, kind: str, q: Any, keys: list[str], descs: list[Any], labels: list[str],
                state_text: str | None = None, qid: str | None = None, questions: list | None = None) -> str:
        ctx = {"state": request_state, "state_text": state_text, "questions": questions or [],
               "question": {"key": qid, "type": kind, "instructions": q.instructions if q is not None else "s",
                            "given": sorted(q.model_fields_set) if q is not None else ["criteria", "instructions", "type"],
                            "criteria": None if q is None else (_given(q.criteria) if kind == "noul" else q.criteria)},
               "options": [{"label": lab, "key": k, "description": d, "index": i}
                           for i, (lab, k, d) in enumerate(zip(labels, keys, descs))]}
        if self.prompt is not None:
            return self.prompt.render(**ctx)
        msgs = [{"role": role, "content": tpl.render(**ctx)} for role, tpl in self.messages]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=self.thinking) + self.reply

    def _encode(self, texts: list[str]) -> list[list[int]]:
        """Token ids of rendered prompts; the pieces between boundaries are tokenized apart and joined."""
        if not any(self.boundary in t for t in texts):
            return self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [[i for piece in t.split(self.boundary) if piece
                 for i in self.tokenizer(piece, add_special_tokens=False)["input_ids"]] for t in texts]

    @staticmethod
    def _expand(v) -> tuple[list[str], bool]:
        """A label set as candidates, and whether candidates that are not one token may be left out."""
        pairs = [a + b for a in LETTERS for b in LETTERS]
        if v == "A-Z+":
            return list(LETTERS) + pairs, True
        if v == "A-Z":
            return list(LETTERS), False
        if v == "pairs":
            return pairs, True
        return list(v), False

    def _token(self, label: str, probe: list[int] | None, text: str | None) -> int | None:
        tok = self.tokenizer
        if probe is None:
            ids = tok.encode(label, add_special_tokens=False)
            return ids[0] if len(ids) == 1 and tok.decode(ids, clean_up_tokenization_spaces=False) == label else None
        ids = self._encode([text + label])[0]
        return ids[-1] if len(ids) == len(probe) + 1 and ids[:-1] == probe else None

    def _label_sets(self, spec) -> tuple[dict, list[int]]:
        """{question type: [(upto, labels)]} and the token ids the head reads, every label once."""
        raw = spec.get("labels", "A-Z+")
        if spec.get("label_ids") is not None:
            labels, ids = list(raw), [int(i) for i in spec["label_ids"]]
            if len(labels) != len(ids):
                raise ValueError("labels and label_ids differ in length")
            self.row_of = {lab: i for i, lab in enumerate(labels)}
            return {k: [(None, labels)] for k in TYPES}, ids
        per = raw if isinstance(raw, dict) else {k: raw for k in TYPES}
        if spec.get("label_forms"):
            return self._form_sets(per, list(spec["label_forms"]))
        standalone = spec.get("label_tokens") == "standalone"
        text = None if standalone else self._render("s", "choice", None, ["x", "y"], ["x", "y"], ["A", "B"], "s")
        probe = None if standalone else self._encode([text])[0]
        ids, self.row_of, sets = [], {}, {}
        for kind in TYPES:
            v = per.get(kind, per.get("default", "A-Z+"))
            tiers = v if isinstance(v, list) and v and isinstance(v[0], dict) else [{"labels": v}]
            sets[kind] = []
            for t in tiers:
                cands, optional = self._expand(t["labels"])
                kept, seen = [], set()
                for label in cands:
                    if len(kept) == 256:
                        break
                    tid = self._token(label, probe, text)
                    if tid is None or tid in seen:
                        if optional and len(label) > 1:
                            continue
                        raise ValueError(f"label {label!r} is not one distinct token where it is read")
                    if label not in self.row_of:
                        self.row_of[label] = len(ids)
                        ids.append(tid)
                    elif ids[self.row_of[label]] != tid:
                        raise ValueError(f"label {label!r} reads two different tokens")
                    seen.add(tid)
                    kept.append(label)
                sets[kind].append((t.get("upto"), kept))
        return sets, ids

    def _form_sets(self, per: dict, forms: list[str]) -> tuple[dict, list[int]]:
        """Label sets whose labels read the log-sum-exp of several token forms; row_of maps a label to its rows."""
        text = self._render("s", "choice", None, ["x", "y"], ["x", "y"], ["A", "B"], "s")
        probe = self._encode([text])[0]
        ids, self.row_of, sets = [], {}, {}
        for kind in TYPES:
            v = per.get(kind, per.get("default", "A-Z+"))
            tiers = v if isinstance(v, list) and v and isinstance(v[0], dict) else [{"labels": v}]
            sets[kind] = []
            for t in tiers:
                cands, optional = self._expand(t["labels"])
                kept, seen = [], set()
                for label in cands:
                    if len(kept) == 256:
                        break
                    tids = []
                    for f in forms:
                        enc = self.tokenizer.encode(f.format(label=label), add_special_tokens=False)
                        if len(enc) == 1 and enc[0] not in tids:
                            tids.append(enc[0])
                    if not tids or tids[0] in seen or (len(label) > 1 and self._token(label, probe, text) != tids[0]):
                        if optional and len(label) > 1:
                            continue
                        raise ValueError(f"label {label!r} has no distinct one-token form where it is read")
                    seen.update(tids)
                    if label not in self.row_of:
                        self.row_of[label] = []
                        for tid in tids:
                            self.row_of[label].append(len(ids))
                            ids.append(tid)
                    kept.append(label)
                sets[kind].append((t.get("upto"), kept))
        return sets, ids

    def _labels(self, kind: str, n: int) -> list[str]:
        for upto, labels in self.sets[kind]:
            if upto is None or n <= upto:
                if n > len(labels):
                    break
                return labels[:n]
        raise ValueError(f"{n} options, more than this model's labels for a {kind}")

    def load_heads(self, backbone) -> dict[str, Any]:
        if self.head == "output":
            weight, bias = backbone.output_head()
            cap = getattr(backbone, "logit_softcap", None) if self.softcap else None
            return {"labels": TokenRows(weight, self.row_ids, bias, softcap=cap)}
        from safetensors.torch import load_file

        rows = load_file(str(Path(self.model_dir) / self.head))[self.spec.get("head_key", "weight")]
        if rows.shape[0] != len(self.row_ids):
            raise ValueError(f"{self.head}: {rows.shape[0]} readout rows for {len(self.row_ids)} labels")
        return {"labels": TokenRows(rows.to(backbone.device), list(range(rows.shape[0])))}

    def _question(self, q) -> tuple[str, list[str], list[Any]]:
        if isinstance(q, NoulQuestion):
            given = _given(q.criteria)
            return "noul", list(self.noul), [given.get(k) if given.get(k) is not None else self.noul_text.get(k)
                                             for k in self.noul]
        if isinstance(q, ChoiceQuestion):
            return "choice", [str(k) for k in q.criteria], list(q.criteria.values())
        return "score", [str(i) for i in range(len(q.criteria))], list(q.criteria)

    def _cut_state(self, state: Any, budget: int) -> tuple[str, int]:
        """The state text within `budget` tokens, and its length before the cut."""
        text = self.state_text.render(state=state)
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= budget:
            return text, len(ids)
        side = self.state_keep.get("list" if isinstance(state, list) else "default", self.state_keep["default"])
        before, after = self.state_marks.get(side, ("", ""))
        return before + self.tokenizer.decode(ids[-budget:] if side == "tail" else ids[:budget]) + after, len(ids)

    def plan(self, request: SystemOneRequest, **_) -> Plan:
        if getattr(request, "images", None):
            raise ValueError("images: this template family reads text only")
        questions = []
        for qid, q in request.questions.items():
            kind, keys, descs = self._question(q)
            if kind not in self.types:
                raise ValueError(f"question '{qid}': this model answers {', '.join(self.types)} questions")
            questions.append((qid, q, kind, keys, descs, self._labels(kind, len(keys))))
        data = [(qid, kind, keys, [self.row_of[lab] for lab in labels]) for qid, _, kind, keys, _, labels in questions]
        everyone = [{"key": qid, "type": kind, "instructions": q.instructions, "given": sorted(q.model_fields_set),
                     "criteria": _given(q.criteria) if kind == "noul" else q.criteria}
                    for qid, q, kind, _, _, _ in questions]
        budget, floor, slack, tries = self.state_budget if self.state_text is not None else (0, 0, 0, 1)
        for _ in range(tries):
            state_text, n = self._cut_state(request.state, budget) if self.state_text is not None else (None, 0)
            texts = [self._render(request.state, kind, q, keys, descs, labels, state_text, qid, everyone)
                     for qid, q, kind, keys, descs, labels in questions]
            rows = self._encode(texts)
            over = max(len(r) for r in rows) - self.max_prompt_tokens
            if over <= 0:
                break
            if self.state_text is None:
                raise ValueError(f"prompt is {over + self.max_prompt_tokens} tokens, limit is {self.max_prompt_tokens}")
            budget = min(budget, n) - over - slack
            if budget < floor:
                raise ValueError(f"the questions leave no room for the state within {self.max_prompt_tokens} tokens")
        else:
            raise ValueError(f"prompt is over {self.max_prompt_tokens} tokens")
        m = min(len(r) for r in rows) - 1
        for i in range(m):
            if any(r[i] != rows[0][i] for r in rows):
                m = i
                break
        return Plan(rows[0][:m], [r[m:] for r in rows], (READ,), data)

    def finish(self, request: SystemOneRequest, plan: Plan, outputs, **_):
        answers = {}
        for (qid, kind, keys, rows), out in zip(plan.data, outputs):
            t = self.temperature[kind]
            t = math.exp(t[0] + t[1] * math.log(len(rows))) if isinstance(t, tuple) else t
            z = [(_logsumexp([out[0][0][i] for i in r]) if isinstance(r, list) else out[0][0][r]) / t for r in rows]
            top = max(z)
            w = [math.exp(x - top) for x in z]
            s = sum(w)
            probs = [x / s for x in w]
            if kind == "noul":
                answers[qid] = make_noul(probs[keys.index("true")])
            elif kind == "choice":
                answers[qid] = make_choice(keys, probs)
            else:
                answers[qid] = make_score(list(request.questions[qid].criteria), probs)
        usage = Usage(input_tokens=plan.tokens, output_tokens=len(plan.data))
        return SystemOneResponse(model="", answers=answers, usage=usage), {"format": "template",
                                                                          "prefix_tokens": len(plan.prefix),
                                                                          "branches": len(plan.suffixes)}

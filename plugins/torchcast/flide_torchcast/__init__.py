"""Torchcast family (torchcast-ai/torchcast-decision-27b on Qwen3.5, format StartLux-Decision-v1): the release's own
torchcast_decision package (jevfmt.py renders, model.py reads).

Every question is its own chat prompt, thinking off: a fixed system line, then "Evidence:" with the state, the question
and its options lettered A) to Z) in the request's order. The answer is the softmax over the listed letters of the
next-token logits at the last prompt position, divided by the temperature of the question type. The evidence comes
first, so the questions fork from it. A choice with one option is answered without the model. A choice with more than
26 options is read in rounds: near-equal groups of at most `group` options, then the top `keep` of every group in a
final round (grouped again when they are still more than 26); options left out keep `residual` of the mass, shared in
proportion to their first-round probabilities.

Images (data URLs or base64), as the release takes them: in a string state "<image>" marks where each goes, else they
come first ("Image k: " before each when there are several); the image processor works at most 1M pixels and at least
65,536 per image, and each image token stands for its merged patches.
"""
from __future__ import annotations

import json
import math
import re
import string
from pathlib import Path
from typing import Any

from flide.api.answers import make_choice, make_noul, make_score
from flide.api.schema import SystemOneRequest, SystemOneResponse, Usage
from flide.formats import register
from flide.formats.base import Format, Plan, Read
from flide.formats.next_token import TokenRows
from flide.formats.tokens import Pieces

FORMAT = "StartLux-Decision-v1"
SYSTEM = "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only."
LETTERS = string.ascii_uppercase
MAX_OPTIONS = 26
MAX_LEVELS = 10
THINK_OFF_SUFFIX = "<think>\n\n</think>\n\n"
ANNOTATE_MIN = 8
BARE = re.compile(r"^(?:\(?[A-Za-z][\).]?|\(?\d{1,3}[\).]?|option[_ ]?\d{1,3}|opt[_ ]?\d{1,3})$", re.I)
READ = Read("letters")
IMAGE = "<image>"
SLOT = "\x00IMAGE\x00"
MAX_PIXELS, MIN_PIXELS = 1 << 20, 65536


def annotate(x):
    """Long arrays carry their element index."""
    if isinstance(x, list):
        if len(x) >= ANNOTATE_MIN:
            return [({"_index": i, **annotate(v)} if isinstance(v, dict) else {"_index": i, "value": annotate(v)})
                    for i, v in enumerate(x)]
        return [annotate(v) for v in x]
    if isinstance(x, dict):
        return {k: annotate(v) for k, v in x.items()}
    return x


def state_text(state) -> str:
    if isinstance(state, str):
        return state
    if state is None or state == {} or state == []:
        return ""
    return json.dumps(annotate(state), ensure_ascii=False)


def value_text(v) -> str | None:
    if v is None:
        return None
    s = (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)).strip()
    return s or None


def _norm(s) -> str:
    return re.sub(r"[\s_\-]+", " ", str(s)).strip().lower()


def options_of(q, keys: list[str] | None = None) -> list[tuple[str, str | None]]:
    """(id, criterion) of every option shown; `keys` picks some of a choice's options, in that order."""
    if q.type == "choice":
        keys = list(q.criteria) if keys is None else keys
        return [(str(k), value_text(q.criteria[k])) for k in keys]
    if q.type == "score":
        return [(str(i), value_text(c)) for i, c in enumerate(q.criteria)]
    c = q.criteria
    return [("true", value_text(c.true if c is not None else None)),
            ("false", value_text(c.false if c is not None else None))]


def question_text(q) -> str:
    ins = value_text(q.instructions) or ""
    if q.type == "noul" and not ins:
        ins = "Which answer fits the evidence?"
    if not ins:
        raise ValueError("instructions required")
    return ins


def option_lines(kind: str, opts: list[tuple[str, str | None]]) -> list[str]:
    hide = kind == "choice" and all(BARE.match(i) for i, _ in opts) and all(c for _, c in opts)
    lines = []
    for k, (i, c) in enumerate(opts):
        name = ("yes" if i == "true" else "no") if kind == "noul" else i
        if hide:
            body = c
        elif c is None or not c.strip() or _norm(c) == _norm(name):
            body = name
        else:
            body = f"{name}: {c}"
        lines.append(f"{LETTERS[k]}) {body}")
    return lines


def groups_of(keys: list[str], size: int) -> list[list[str]]:
    n = -(-len(keys) // size)
    base, extra = divmod(len(keys), n)
    out, start = [], 0
    for g in range(n):
        end = start + base + (1 if g < extra else 0)
        out.append(keys[start:end])
        start = end
    return out


@register("torchcast")
class Torchcast(Format):
    backbones = ("qwen3_5", "qwen3_5_text")
    CUT_CHECKS = 8

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        return bool(decision_config) and decision_config.get("format") == FORMAT

    def __init__(self, model_dir: str, tokenizer=None, max_length: int = 262144):
        import transformers

        self.model_dir = model_dir
        self.config = json.loads((Path(model_dir) / "decision_config.json").read_text())
        self.tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.letter_ids = self._letters()
        if self.letter_ids != self.config["letter_token_ids"]:
            raise ValueError("the tokenizer's letter ids differ from decision_config.json")
        self.temperature = {k: float(v) for k, v in self.config["temperature_by_type"].items()}
        wide = self.config.get("wide_choice", {})
        self.group, self.keep, self.residual = wide.get("group", 25), wide.get("keep", 3), wide.get("residual", 1e-3)
        self.max_length = int(max_length)
        self._pieces = Pieces(self.tokenizer)
        self._cut_checks = 0

    @property
    def image_processor(self):
        ip = getattr(self, "_image_processor", None)
        if ip is None:
            import transformers

            try:
                ip = transformers.AutoImageProcessor.from_pretrained(self.model_dir, local_files_only=True)
            except ImportError as e:
                if "orchvision" not in str(e):
                    raise
                raise ImportError("images need torchvision: pip install torchvision") from e
            ip.size = {"longest_edge": MAX_PIXELS, "shortest_edge": MIN_PIXELS}
            self._image_processor = ip
            self.image_token = json.loads((Path(self.model_dir) / "config.json").read_text())["image_token_id"]
        return ip

    @staticmethod
    def _media_state(text: str, n: int) -> str:
        """The state as text with the places of n images: at its "<image>" marks, else first."""
        marks = text.count(IMAGE)
        if marks == n:
            return text.replace(IMAGE, SLOT)
        if marks:
            raise ValueError(f"the state has {marks} {IMAGE} marks for {n} images")
        head = SLOT if n == 1 else "\n".join(f"Image {k + 1}: {SLOT}" for k in range(n))
        return head + ("\n\n" + text if text.strip() else "")

    def _template(self, user: str) -> str:
        if SLOT in user:
            parts, content = user.split(SLOT), []
            for k, part in enumerate(parts):
                if part:
                    content.append({"type": "text", "text": part})
                if k < len(parts) - 1:
                    content.append({"type": "image"})
            chat = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}, {"role": "user", "content": content}]
        else:
            chat = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if not text.endswith(THINK_OFF_SUFFIX):
            raise ValueError("the chat template lacks the thinking-off assistant prefix")
        return text

    def _letters(self) -> list[int]:
        text = self._template("Evidence:\ns\n\nQuestion: q\nOptions:\nA) yes\nB) no")
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        out = []
        for L in LETTERS:
            t = self.tokenizer.encode(L, add_special_tokens=False)
            if len(t) != 1 or self.tokenizer.encode(text + L, add_special_tokens=False) != ids + t:
                raise ValueError(f"letter {L} is not one token after the prompt")
            out.append(t[0])
        if len(set(out)) != len(out):
            raise ValueError("letter ids are not distinct")
        return out

    def token_heads(self) -> dict[str, list[int]]:
        return {"letters": list(self.letter_ids)}

    def load_heads(self, backbone) -> dict[str, Any]:
        weight, bias = backbone.output_head()
        return {"letters": TokenRows(weight, self.letter_ids, bias, softcap=getattr(backbone, "logit_softcap", None))}

    def _text(self, state: str, q, opts) -> str:
        user = ("Evidence:\n" + (state if state.strip() else "(none)") + "\n\nQuestion: " + question_text(q)
                + "\nOptions:\n" + "\n".join(option_lines(q.type, opts)))
        return self._template(user)

    def _plan(self, request: SystemOneRequest, rows: list[tuple], data: dict) -> Plan:
        """rows: (question id, option keys or None, what the read stands for); one branch each, forked from the
        evidence."""
        refs = getattr(request, "images", None) or []
        state = state_text(request.state)
        if refs:
            state = self._media_state(state, len(refs))
        texts = []
        for qid, keys, _ in rows:
            q = request.questions[qid]
            opts = options_of(q, keys)
            if not 2 <= len(opts) <= MAX_OPTIONS or (q.type == "score" and len(opts) > MAX_LEVELS):
                raise ValueError(f"{len(opts)} options, limit is 2 to {MAX_OPTIONS}")
            texts.append(self._text(state, q, opts))
        if refs:
            return self._media_plan(texts, refs, {**data, "rows": rows})
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
        if longest > self.max_length:
            raise ValueError(f"prompt is {longest} tokens, limit is {self.max_length}")
        m = min(len(r) for r in rest) - 1
        for i in range(m):
            if any(r[i] != rest[0][i] for r in rest):
                m = i
                break
        return Plan(head + rest[0][:m], [r[m:] for r in rest], (READ,), {**data, "rows": rows})

    def _media_plan(self, texts: list[str], refs: list, data: dict) -> Plan:
        """Prompts with images: each tokenized whole, every image token repeated once per merged patch, forked where
        the prompts part (the images sit in the evidence, before every question)."""
        import torch

        from flide.formats.next_token import NextToken

        ip = self.image_processor
        enc = ip(images=[NextToken._open(r) for r in refs], return_tensors="pt")
        counts = (enc["image_grid_thw"].prod(-1) // ip.merge_size ** 2).tolist()
        rows = []
        for text in texts:
            ids, k = [], 0
            for t in self.tokenizer.encode(text, add_special_tokens=False):
                if t == self.image_token:
                    if k == len(counts):
                        raise ValueError("more image places than images")
                    ids.extend([t] * counts[k])
                    k += 1
                else:
                    ids.append(t)
            if k != len(counts):
                raise ValueError("fewer image places than images")
            if len(ids) > self.max_length:
                raise ValueError(f"prompt is {len(ids)} tokens, limit is {self.max_length}")
            rows.append(ids)
        m = min(len(r) for r in rows) - 1
        for i in range(m):
            if any(r[i] != rows[0][i] for r in rows):
                m = i
                break
        prefix = rows[0][:m]
        if sum(t == self.image_token for t in prefix) != sum(counts):
            raise ValueError("the images must sit in the evidence, before every question")
        vision = {"pixel_values": enc["pixel_values"], "image_grid_thw": enc["image_grid_thw"],
                  "mm_token_type_ids": (torch.tensor(prefix) == self.image_token).long()[None]}
        return Plan(prefix, [r[m:] for r in rows], (READ,), data, vision=vision)

    def _probs(self, out, kind: str, n: int) -> list[float]:
        t = self.temperature.get(kind, 1.0)
        z = [x / t for x in out[0][0][:n]]
        m = max(z)
        w = [math.exp(x - m) for x in z]
        s = sum(w)
        return [x / s for x in w]

    def _wide_rows(self, qid: str, node: dict) -> list[tuple]:
        """The reads a wide choice still needs: its group reads, or the reads of its final round."""
        if node["first"] is None:
            return [(qid, g, ("group", qid, node["path"], i)) for i, g in enumerate(node["groups"])]
        if node["child"] is not None:
            return self._wide_rows(qid, node["child"])
        return [(qid, node["finalists"], ("final", qid, node["path"]))]

    def _node(self, keys: list[str], path: tuple) -> dict:
        return {"keys": keys, "groups": groups_of(keys, self.group), "first": None, "finalists": None, "child": None,
                "final": None, "path": path}

    def plan(self, request: SystemOneRequest, **_) -> Plan:
        fixed, wide, rows = {}, {}, []
        for qid, q in request.questions.items():
            if q.type == "choice" and len(q.criteria) == 1:
                fixed[qid] = next(iter(q.criteria))
            elif q.type == "choice" and len(q.criteria) > MAX_OPTIONS:
                wide[qid] = self._node([str(k) for k in q.criteria], ())
                rows += self._wide_rows(qid, wide[qid])
            else:
                rows.append((qid, None, ("plain", qid)))
        data = {"fixed": fixed, "wide": wide, "plain": {}, "tokens": 0}
        if not rows:
            return Plan([], [[self.letter_ids[0]]], (READ,), {**data, "rows": []})
        return self._plan(request, rows, data)

    def _find(self, node: dict, path: tuple) -> dict:
        while node["path"] != path:
            node = node["child"]
        return node

    def _settle(self, node: dict) -> bool:
        """Combine what a wide choice's reads gave, deepest first; True when its probabilities are known."""
        if node["final"] is not None:
            return True
        if node["first"] is None or (node["child"] is not None and not self._settle(node["child"])):
            return False
        if node["child"] is not None:
            final = node["child"]["final"]
        elif node.get("final_read") is not None:
            final = dict(zip(node["finalists"], node["final_read"]))
        else:
            return False
        rest = [k for k in node["keys"] if k not in set(node["finalists"])]
        mass = sum(node["first"][k] for k in rest) or 1.0
        probs = {k: final[k] * (1 - self.residual) for k in node["finalists"]}
        probs.update({k: self.residual * node["first"][k] / mass for k in rest})
        z = sum(probs.values())
        node["final"] = {k: v / z for k, v in probs.items()}
        return True

    def _absorb(self, request: SystemOneRequest, plan: Plan, outputs) -> dict:
        d = plan.data
        firsts: dict = {}
        for (qid, keys, ref), out in zip(d["rows"], outputs):
            q = request.questions[qid]
            n = len(keys) if keys is not None else len(options_of(q))
            p = self._probs(out, q.type, n)
            if ref[0] == "plain":
                d["plain"][qid] = p
            elif ref[0] == "group":
                firsts.setdefault((qid, ref[2]), {}).update(zip(keys, p))
            else:
                self._find(d["wide"][qid], ref[2])["final_read"] = p
        for (qid, path), first in firsts.items():
            node = self._find(d["wide"][qid], path)
            if len(first) == len(node["keys"]):
                node["first"] = first
                node["finalists"] = [k for g in node["groups"]
                                     for k, _ in sorted(((k, first[k]) for k in g), key=lambda x: -x[1])[:self.keep]]
                if len(node["finalists"]) > MAX_OPTIONS:
                    node["child"] = self._node(node["finalists"], path + (len(path),))
        d["tokens"] += plan.tokens
        return d

    def follow_up(self, request: SystemOneRequest, plan: Plan, outputs, **_) -> Plan | None:
        if not plan.data["rows"]:
            return None
        d = self._absorb(request, plan, outputs)
        rows = [r for qid, node in d["wide"].items() if not self._settle(node) for r in self._wide_rows(qid, node)]
        if not rows:
            d["done"] = True
            return None
        return self._plan(request, rows, d)

    def finish(self, request: SystemOneRequest, plan: Plan, outputs, **_):
        d = plan.data if plan.data.get("done") or not plan.data["rows"] else self._absorb(request, plan, outputs)
        for node in d["wide"].values():
            self._settle(node)
        answers = {}
        for qid, q in request.questions.items():
            if qid in d["fixed"]:
                answers[qid] = make_choice([d["fixed"][qid]], [1.0])
            elif qid in d["wide"]:
                p = d["wide"][qid]["final"]
                answers[qid] = make_choice(list(p), list(p.values()))
            elif q.type == "noul":
                answers[qid] = make_noul(d["plain"][qid][0])
            elif q.type == "choice":
                answers[qid] = make_choice([str(k) for k in q.criteria], d["plain"][qid])
            else:
                answers[qid] = make_score(list(q.criteria), d["plain"][qid])
        usage = Usage(input_tokens=d["tokens"], output_tokens=len(answers))
        return SystemOneResponse(model="", answers=answers, usage=usage), {"format": "torchcast",
                                                                          "prefix_tokens": len(plan.prefix)}

"""Kev family (jaredpalmer/kev-0.8b, kev-4b, kev-9b, kev-27b) and JevAny (SimpleJev/JevAny-Qwen3.5-4B-LoRA and
JevAny-Qwen3.8-27B-LoRA) on Qwen3.5: a pointer readout over the state and a branch per question, as kev.serve and
jevany.serve read them.

A request has no chat template. The prefix is <|fim_prefix|> and the state; each question is a branch of its own,
<|fim_middle|> and its instructions, every option between <|box_start|> and <|box_end|>, then <|fim_suffix|>, with
positions running on from the state. Text is flattened as the releases do (objects as "key: value" lines, arrays as
"- item" lines, a choice option as "name: description", a noul as the options "no" and "yes" with the criteria's
descriptions), and every `<|name|>` in it becomes `<¦name¦>`, so no text makes a delimiter. A branch is read at its
last token (the query) and at each <|box_end|> (the options) through the checkpoint's pointer head (head.pt): q and k
projections, an option's logit k.q / sqrt(256), plus JevAny's residual MLP over k * q, divided by the checkpoint's
temperature; the answer is their softmax. Kev's adapters run unmerged on their base checkpoint and JevAny's merge into
it as the weights are read, as jevany.serve merges them. Kev tokenizes with its base checkpoint's tokenizer at the
revision head.pt names, JevAny with the one it saved, as their releases load them.

Kev takes states of up to 65,536 tokens (the first delimiter included) and question rows, the state and one branch, of
up to 73,728, and text only. JevAny takes 8,192 tokens for the state, for a row and for the whole request, 64
questions, nouls whose criteria name only false and true, and scores of two levels or more. A JevAny request may carry
up to 32 images (data URLs or base64) with one question: then the whole request is one text, each image's placeholder
between <|vision_start|> and <|vision_end|> ahead of it, through the checkpoint's processor, the images sharing a
budget of 512 x 512 pixels as the release sizes them (each at least 128 x 128), and special tokens left in the text
get the brackets ‹ and ［.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from flide.api.answers import make_choice, make_noul, make_score
from flide.api.schema import ChoiceQuestion, NoulQuestion, SystemOneRequest, SystemOneResponse, Usage
from flide.formats import register
from flide.formats.base import Format, Plan, Read
from flide.formats.next_token import TokenRows

HEAD = "head.pt"
READ = "pointer"
LEGACY = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
V1 = ["<|jev_state|>", "<|jev_question|>", "<|jev_option|>", "<|jev_end_option|>", "<|jev_decide|>"]
_SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def _meta(model_dir: str) -> dict | None:
    path = Path(model_dir) / HEAD
    if not path.is_file():
        return None
    import torch

    try:
        meta = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        return None
    head = meta.get("head") if isinstance(meta, dict) else None
    if not isinstance(head, dict) or not {"q.weight", "k.weight"} <= set(head) or "head_dim" not in meta:
        return None
    return meta


def _saved_tokenizer(model_dir: str):
    """The checkpoint's own tokenizer, of the class its tokenizer_config.json names, as it loads from its own directory
    (the directory an adapter runs from also holds its base's config.json, which would choose the base's class)."""
    import json

    import transformers

    cfg = json.loads((Path(model_dir) / "tokenizer_config.json").read_text())
    cls = getattr(transformers, cfg.get("tokenizer_class") or "", None) or transformers.AutoTokenizer
    return cls.from_pretrained(model_dir, local_files_only=True)


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


def _options(q) -> tuple[str, list[str]]:
    if isinstance(q, NoulQuestion):
        c = q.criteria
        return "noul", [option_text("no", None if c is None else c.false), option_text("yes", None if c is None else c.true)]
    if isinstance(q, ChoiceQuestion):
        return "choice", [option_text(k, v) for k, v in q.criteria.items()]
    return "score", [render(x) for x in q.criteria]


@register("kev")
class Kev(Format):
    backbones = ("qwen3_5",)
    max_choice_options = 255
    max_score_levels = 255
    max_state = 65536
    max_row = 65536 + 8192
    max_tokens: int | None = None
    max_questions: int | None = None
    images = False

    def __init__(self, model_dir: str, tokenizer=None, temperature: float | None = None):
        import transformers

        self.model_dir = model_dir
        self.meta = _meta(model_dir)
        if self.meta is None:
            raise ValueError(f"{model_dir}: no pointer head ({HEAD})")
        if tokenizer is None:
            if self.meta.get("tokenizer_saved") or not self.meta.get("base"):
                tokenizer = _saved_tokenizer(model_dir)
            else:
                tokenizer = transformers.AutoTokenizer.from_pretrained(self.meta["base"],
                                                                       revision=self.meta.get("base_revision"))
        self.tokenizer = tokenizer
        self.delimiters = self._alphabet(self.tokenizer)
        ids = [self.tokenizer.convert_tokens_to_ids(t) for t in self.delimiters]
        if any(i is None or i == self.tokenizer.unk_token_id for i in ids):
            raise ValueError(f"{model_dir}: the tokenizer lacks the delimiters {self.delimiters}")
        self.state_id, self.question_id, self.open_id, self.close_id, self.decide_id = ids
        self.dim = int(self.meta["head_dim"])
        self.scale = 1 / math.sqrt(self.dim)
        t = self.meta.get("temperature", 1.0) if temperature is None else temperature
        self.temperature = float(t)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        head = self.meta["head"]
        self.residual = None
        if "residual.0.weight" in head:
            self.residual = tuple(head[k].float() for k in ("residual.0.weight", "residual.0.bias", "residual.2.weight",
                                                             "residual.2.bias"))

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        meta = _meta(model_dir)
        return meta is not None and "decision_mode" not in meta and "head_type" not in meta

    @staticmethod
    def _alphabet(tok) -> list[str]:
        return LEGACY

    def load_heads(self, backbone) -> dict[str, Any]:
        import torch

        h = self.meta["head"]
        w = torch.cat([h["q.weight"], h["k.weight"]]).to(backbone.device)
        b = torch.cat([h["q.bias"], h["k.bias"]]).to(backbone.device)
        return {READ: TokenRows(w, list(range(w.shape[0])), bias=b)}

    def _safe(self, text: str) -> str:
        text = _SPECIAL.sub(r"<¦\1¦>", text)
        for d in self.delimiters:
            text = text.replace(d, d.replace("<", "<¦", 1).replace(">", "¦>", 1))
        return text

    def _tokens(self, text: str) -> list[int]:
        return self.tokenizer(self._safe(text), add_special_tokens=False).input_ids

    def _check(self, request: SystemOneRequest) -> None:
        if getattr(request, "images", None) and not self.images:
            raise ValueError(f"images: the {self.name} family reads text only")

    def plan(self, request: SystemOneRequest, **_) -> Plan:
        self._check(request)
        state = self._tokens(render(request.state))
        if len(state) + 1 > self.max_state:
            raise ValueError(f"state is {len(state) + 1} tokens, over the {self.max_state}-token limit")
        prefix = [self.state_id] + state
        suffixes, reads, fields = [], [], []
        for name, q in request.questions.items():
            kind, texts = _options(q)
            branch = [self.question_id] + self._tokens(render(q.instructions))
            ends = []
            for t in texts:
                branch += [self.open_id] + self._tokens(t) + [self.close_id]
                ends.append(len(branch) - 1)
            branch.append(self.decide_id)
            if len(prefix) + len(branch) > self.max_row:
                raise ValueError(f"question {name!r}: its row is {len(prefix) + len(branch)} tokens, over the "
                                 f"{self.max_row}-token limit")
            suffixes.append(branch)
            reads.append((Read(READ, positions=(-1, *(e - len(branch) for e in ends))),))
            fields.append((name, kind, len(texts)))
        total = len(prefix) + sum(map(len, suffixes))
        if self.max_tokens is not None and total > self.max_tokens:
            raise ValueError(f"request is {total} tokens, over the {self.max_tokens}-token limit")
        return Plan(prefix, suffixes, reads, fields)

    def logits(self, rows):
        """Option logits of one question from its head rows: the query's, then each option's."""
        import torch

        x = torch.tensor(rows, dtype=torch.float32)
        q, k = x[0, : self.dim], x[1:, self.dim:]
        z = (k @ q) * self.scale
        if self.residual is not None:
            w1, b1, w2, b2 = self.residual
            z = z + (torch.nn.functional.gelu((k * q) @ w1.T + b1) @ w2.T + b2).squeeze(-1)
        return z if self.temperature == 1.0 else z / self.temperature

    def finish(self, request: SystemOneRequest, plan: Plan, outputs, **_):
        import torch

        answers = {}
        for (name, kind, n), branch in zip(plan.data, outputs):
            probs = torch.softmax(self.logits(branch[0]), -1).tolist()
            q = request.questions[name]
            if kind == "noul":
                answers[name] = make_noul(probs[1])
            elif kind == "choice":
                answers[name] = make_choice([str(k) for k in q.criteria], probs)
            else:
                answers[name] = make_score(list(q.criteria), probs)
        usage = Usage(input_tokens=plan.tokens, output_tokens=len(plan.data))
        return SystemOneResponse(model="", answers=answers, usage=usage), {"format": self.name, "tokens": plan.tokens}


@register("jevany")
class JevAny(Kev):
    merge_lora = True
    max_choice_options = 256
    max_state = 8192
    max_row = 8192
    max_tokens = 8192
    max_questions = 64
    images = True
    max_images = 32
    pixel_budget = 512 * 512

    @property
    def processor(self):
        """The checkpoint's processor with this family's tokenizer (images only)."""
        p = getattr(self, "_processor", None)
        if p is None:
            import transformers

            try:
                p = transformers.AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
            except ImportError as e:
                if "orchvision" not in str(e):
                    raise
                raise ImportError("images need torchvision: pip install torchvision") from e
            p.tokenizer = self.tokenizer
            self._processor = p
        return p

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        meta = _meta(model_dir)
        return meta is not None and meta.get("decision_mode") == "pointer"

    @staticmethod
    def _alphabet(tok) -> list[str]:
        kw = getattr(tok, "init_kwargs", {})
        explicit = kw.get("jevany_decision_tokens")
        if explicit is not None:
            if not isinstance(explicit, list) or len(explicit) != 5 or len(set(explicit)) != 5:
                raise ValueError("jevany_decision_tokens must be five distinct strings")
            return explicit
        schema = kw.get("jevany_token_schema", "legacy")
        if schema not in ("legacy", "v1"):
            raise ValueError(f"unknown JevAny tokenizer schema {schema!r}")
        return LEGACY if schema == "legacy" else V1

    def _check(self, request: SystemOneRequest) -> None:
        super()._check(request)
        if len(request.questions) > self.max_questions:
            raise ValueError(f"{len(request.questions)} questions, maximum is {self.max_questions}")
        for name, q in request.questions.items():
            if isinstance(q, NoulQuestion) and q.criteria is not None and q.criteria.model_extra:
                raise ValueError(f"question {name!r}: noul criteria only accepts false and true")
            if q.type == "score" and len(q.criteria) < 2:
                raise ValueError(f"question {name!r}: a score needs two levels or more")
        refs = getattr(request, "images", None) or []
        if len(refs) > self.max_images:
            raise ValueError(f"images: {len(refs)}, maximum is {self.max_images}")
        if refs and len(request.questions) != 1:
            raise ValueError("images: a request with images holds one question")

    def plan(self, request: SystemOneRequest, **options) -> Plan:
        refs = getattr(request, "images", None) or []
        if not refs:
            return super().plan(request, **options)
        self._check(request)
        from flide.formats.next_token import NextToken

        p = self.processor
        escapes = [t for t in self.tokenizer.all_special_tokens if t.startswith(("<", "["))]

        def text(v: str) -> str:
            v = self._safe(v)
            for t in escapes:
                v = v.replace(t, t.replace("<", "‹").replace("[", "［"))
            return v

        state = render(request.state)
        if len(self._tokens(state)) + 1 > self.max_state:
            raise ValueError(f"state is {len(self._tokens(state)) + 1} tokens, over the {self.max_state}-token limit")
        ((name, q),) = request.questions.items()
        kind, texts = _options(q)
        d = self.delimiters
        parts = [d[0], text(state), d[1], text(render(q.instructions))]
        for t in texts:
            parts += [d[2], text(t), d[3]]
        parts.append(d[4])
        lead = (p.vision_start_token + p.image_token + p.vision_end_token) * len(refs)
        size = {"shortest_edge": 32 * 32, "longest_edge": max(128 * 128, self.pixel_budget // len(refs))}
        out = p(text=[lead + "".join(parts)], images=[NextToken._open(r) for r in refs], size=size, return_tensors="pt")
        ids = out["input_ids"][0].tolist()
        if ids.count(self.question_id) != 1 or ids.count(self.decide_id) != 1 or ids[-1] != self.decide_id:
            raise ValueError("images: the delimiters do not match the question")
        start = ids.index(self.question_id)
        ends = [i for i in range(start, len(ids)) if ids[i] == self.close_id]
        if len(ends) != len(texts):
            raise ValueError("images: the delimiters do not match the options")
        if len(ids) > min(self.max_row, self.max_tokens):
            raise ValueError(f"request is {len(ids)} tokens, over the {min(self.max_row, self.max_tokens)}-token limit")
        vision = {"pixel_values": out["pixel_values"], "image_grid_thw": out["image_grid_thw"],
                  "mm_token_type_ids": (out["input_ids"][:, :start] == p.image_token_id).long()}
        reads = [(Read(READ, positions=(-1, *(e - len(ids) for e in ends))),)]
        return Plan(ids[:start], [ids[start:]], reads, [(name, kind, len(texts))], vision=vision)

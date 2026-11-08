"""Clef family (Cloudflare/clef-flash on Qwen3.5-9B and Cloudflare/clef on Qwen3.8-27B): the release's
joint_schema_model.py, its joint schema head over every hidden state of one causal sequence.

A request is one sequence, tokenized piece by piece as encode_record tokenizes it: a system turn, "STATE:" and the
state, then "SCHEMA FIELDS:" with every question in order (its id, type, instructions and allowed options, each
option the JSON of its id and description), then the assistant turn with thinking off. A choice's options are in the
order of their names, a noul's are true and false with the release's descriptions unless the request gives its own,
and a score's are its levels. The state is cut so the sequence holds at most 16,384 tokens. A request's images (data
URLs or base64) go after the system turn as the release places them, at its processor's sizes.

The backbone reads the sequence causally and the head reads its final hidden states at every position (through
JointSchemaHead.hidden_norm, kept on the device): span means of each question's instructions and each option, the
lm_head rows of each option's tokens, evidence routing over all positions, joint layers over the questions, and one
logit per option. The answer is the softmax over each question's options. This is JointSchemaHead.forward for one
record with its loops over questions and options done at once.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from flide.api.answers import make_choice, make_noul, make_score
from flide.api.schema import ChoiceQuestion, NoulQuestion, SystemOneRequest, SystemOneResponse, Usage
from flide.formats import register
from flide.formats.base import Format, Plan, Read

CONFIG = "joint_head_config.json"
WEIGHTS = "joint_head.safetensors"
READ = "hidden"
SYSTEM_PROMPT = ("Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of "
                 "that field's allowed options.")
PREFIX = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
SUFFIX = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
NOUL = {"true": "The proposition is true or the answer is yes.",
        "false": "The proposition is false or the answer is no."}
TYPES = {"noul": 0, "choice": 1, "score": 2}
MAX_LENGTH = 16384


def render(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                                           sort_keys=True)


def options(q) -> list[tuple[str, Any]]:
    """Each option's id and description, in the order the head scores them."""
    if isinstance(q, NoulQuestion):
        c = dict(NOUL)
        if q.criteria is not None:
            c.update(q.criteria.model_dump(exclude_unset=True))
        return [(k, c[k]) for k in ("true", "false")]
    if isinstance(q, ChoiceQuestion):
        return sorted((str(k), v) for k, v in q.criteria.items())
    return [(str(i), v) for i, v in enumerate(q.criteria)]


def _head_module(cfg: dict):
    import torch
    from torch import nn

    class EvidenceRoutingLayer(nn.Module):
        def __init__(self, width, heads, feedforward):
            super().__init__()
            self.query_norm = nn.LayerNorm(width)
            self.memory_norm = nn.LayerNorm(width)
            self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
            self.feedforward_norm = nn.LayerNorm(width)
            self.feedforward = nn.Sequential(nn.Linear(width, feedforward), nn.GELU(), nn.Identity(),
                                             nn.Linear(feedforward, width), nn.Identity())

        def forward(self, queries, memory):
            routed, _ = self.attention(self.query_norm(queries), self.memory_norm(memory), self.memory_norm(memory),
                                       need_weights=False)
            queries = queries + routed
            return queries + self.feedforward(self.feedforward_norm(queries))

    class JointSchemaHead(nn.Module):
        def __init__(self, hidden_size, width, routing_layers, layers, heads, feedforward):
            super().__init__()
            self.hidden_norm = nn.LayerNorm(hidden_size)
            self.memory_projection = nn.Linear(hidden_size, width, bias=False)
            self.question_projection = nn.Linear(hidden_size, width, bias=False)
            self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
            self.global_projection = nn.Linear(hidden_size, width, bias=False)
            self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
            self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
            self.type_embedding = nn.Embedding(3, width)
            self.evidence_layers = nn.ModuleList(EvidenceRoutingLayer(width, heads, feedforward)
                                                 for _ in range(routing_layers))
            self.option_summary_norm = nn.LayerNorm(width)
            self.layers = nn.ModuleList(nn.TransformerDecoderLayer(d_model=width, nhead=heads,
                                                                   dim_feedforward=feedforward, dropout=0.0,
                                                                   activation="gelu", batch_first=True,
                                                                   norm_first=True) for _ in range(layers))
            self.field_norm = nn.LayerNorm(width)
            self.option_norm = nn.LayerNorm(width)
            self.residual_scorer = nn.Sequential(nn.Linear(width * 4, width), nn.GELU(), nn.Identity(),
                                                 nn.Linear(width, 1))
            self.prior_logit_scale = nn.Parameter(torch.zeros(()))
            self.joint_logit_scale = nn.Parameter(torch.zeros(()))
            self.residual_gate = nn.Parameter(torch.zeros(()))

    return JointSchemaHead(**cfg)


class _Normed:
    """The head's LayerNorm over the final hidden states, its output left on the device for `finish`."""

    on_device = True

    def __init__(self, norm):
        self.norm = norm

    def __call__(self, h):
        return self.norm(h)


@register("clef")
class Clef(Format):
    backbones = ("qwen3_5",)
    max_choice_options = 256
    max_score_levels = 256

    def __init__(self, model_dir: str, tokenizer=None, max_length: int = MAX_LENGTH):
        import transformers

        self.model_dir = model_dir
        self.config = json.loads((Path(model_dir) / CONFIG).read_text())
        self.tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.max_length = int(max_length)
        self.head = None
        self.lm_head = None
        self.prefix = self._tokens(PREFIX)
        self.suffix = self._tokens(SUFFIX)

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        d = Path(model_dir)
        return (d / CONFIG).is_file() and (d / WEIGHTS).is_file()

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

    def load_heads(self, backbone) -> dict[str, Any]:
        from safetensors.torch import load_file

        head = _head_module(self.config)
        head.load_state_dict(load_file(str(Path(self.model_dir) / WEIGHTS)), strict=True)
        self.head = head.to(device=backbone.device, dtype=backbone.dtype).eval().requires_grad_(False)
        self.lm_head = backbone.output_head()[0]
        return {READ: _Normed(self.head.hidden_norm)}

    def _tokens(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def plan(self, request: SystemOneRequest, **_) -> Plan:
        if getattr(request, "videos", None):
            raise ValueError("videos: the clef family reads text and images")
        texts, opts = [render(request.state), "\n\nSCHEMA FIELDS:\n"], []
        for i, (name, q) in enumerate(request.questions.items()):
            instructions = q.instructions
            texts += [f"\nFIELD {i + 1}\nID: {name}\nTYPE: {q.type}\nINSTRUCTION: ",
                      render(str(name) if instructions is None or instructions == "" else instructions),
                      "\nALLOWED OPTIONS:\n"]
            opts.append(options(q))
            for j, (key, desc) in enumerate(opts[-1]):
                texts += [f"OPTION {j + 1}: ", render({"option_id": key} if desc is None else
                                                      {"option_id": key, "description": desc}), "\n"]
            texts.append("END FIELD\n")
        pieces = iter(self.tokenizer(texts, add_special_tokens=False).input_ids)
        state = next(pieces)
        schema = next(pieces)
        fields = []
        for (name, q), qo in zip(request.questions.items(), opts):
            schema += next(pieces)
            start = len(schema)
            schema += next(pieces)
            span = (start, len(schema))
            if span[0] == span[1]:
                raise ValueError(f"question {name!r}: its instructions and its id are empty")
            schema += next(pieces)
            spans, ids = [], []
            for key, _ in qo:
                schema += next(pieces)
                at = len(schema)
                schema += next(pieces)
                spans.append((at, len(schema)))
                ids.append(key)
                schema += next(pieces)
            schema += next(pieces)
            fields.append((name, q.type, span, spans, ids))
        prefix, vision = list(self.prefix), None
        refs = getattr(request, "images", None) or []
        if refs:
            from flide.formats.next_token import NextToken

            import torch

            p = self.processor
            out = p(text=[IMAGE * len(refs) + "\n"], images=[NextToken._open(r) for r in refs], return_tensors="pt")
            media = out["input_ids"][0].tolist()
            types = torch.zeros(1, len(prefix) + len(media), dtype=torch.long)
            types[0, len(prefix):] = (out["input_ids"][0] == p.image_token_id).long()
            prefix += media
            vision = {"pixel_values": out["pixel_values"], "image_grid_thw": out["image_grid_thw"],
                      "mm_token_type_ids": types}
        fixed = len(prefix) + len(schema) + len(self.suffix)
        if fixed > self.max_length:
            raise ValueError(f"the questions need {fixed} tokens before the state; the maximum is {self.max_length}")
        state = state[: self.max_length - fixed]
        shift = len(prefix) + len(state)
        rest = state + schema + self.suffix
        n = len(prefix) + len(rest)
        data = {"ids": prefix + rest,
                "fields": [(name, kind, (a + shift, b + shift), [(s + shift, e + shift) for s, e in spans], ids)
                           for name, kind, (a, b), spans, ids in fields]}
        return Plan(prefix, [rest], (Read(READ, positions=tuple(range(n))),), data, vision=vision)

    def logits(self, normed, data) -> list:
        """Every question's option logits (float32, on the device), as JointSchemaHead.forward gives them."""
        import torch
        import torch.nn.functional as F

        h, dev, dt = self.head, normed.device, normed.dtype
        fields = data["fields"]
        spans = [s for _, _, q, opts, _ in fields for s in [q, *opts]]
        lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
        avg = torch.zeros(len(spans), hi - lo, dtype=torch.float32)
        for k, (a, b) in enumerate(spans):
            avg[k, a - lo: b - lo] = 1.0 / (b - a)
        avg = avg.to(dev, non_blocking=True)
        ids = torch.tensor(data["ids"][lo:hi], device=dev)
        means = (avg @ normed[lo:hi].float()).to(dt)
        lex_all = (avg @ self.lm_head.index_select(0, ids).float()).to(dt)
        qi, oi, owner, kinds = [], [], [], []
        for i, (_, kind, _, opts, _) in enumerate(fields):
            qi.append(len(qi) + len(oi))
            oi += range(qi[-1] + 1, qi[-1] + 1 + len(opts))
            owner += [i] * len(opts)
            kinds.append(TYPES[kind])
        qi, oi, owner = (torch.tensor(x, device=dev) for x in (qi, oi, owner))
        counts = [len(f[3]) for f in fields]
        question, context, lexical = means[qi], means[oi], lex_all[oi]

        memory = h.memory_projection(normed).unsqueeze(0)
        glob = normed[-1]
        routed = (h.option_context_projection(context) + h.option_lexical_projection(lexical)
                  + h.option_question_projection(question)[owner]).unsqueeze(0)
        for layer in h.evidence_layers:
            routed = layer(routed, memory)
        routed = routed[0]

        base = h.question_projection(question)
        Q, widest, W = len(fields), max(counts), routed.shape[-1]
        slot = torch.tensor([j for c in counts for j in range(c)], device=dev)
        score = torch.matmul(routed.unsqueeze(1), base[owner].unsqueeze(2)).view(-1) / math.sqrt(W)
        padded = torch.full((Q, widest), float("-inf"), dtype=dt, device=dev)
        padded[owner, slot] = score
        weights = torch.softmax(padded, dim=1)
        grid = routed.new_zeros(Q, widest, W)
        grid[owner, slot] = routed
        summaries = torch.sum(weights.unsqueeze(-1) * grid, dim=1)
        f = (base + h.option_summary_norm(summaries) + h.global_projection(glob).unsqueeze(0)
             + h.type_embedding(torch.tensor(kinds, device=dev)))
        f = f.unsqueeze(0)
        for layer in h.layers:
            f = layer(f, memory)
        f = h.field_norm(f[0])

        anchor = F.normalize(question + glob, dim=-1)
        prior_scale = h.prior_logit_scale.clamp(max=math.log(100.0)).exp()
        prior = prior_scale * torch.matmul(F.normalize(lexical, dim=-1).unsqueeze(1), anchor[owner].unsqueeze(2)).view(-1)
        opts = h.option_norm(routed)
        rep = f[owner]
        cosine = F.cosine_similarity(rep, opts, dim=-1)
        features = torch.cat([rep, opts, rep * opts, torch.abs(rep - opts)], dim=-1)
        residual = h.residual_scorer(features).squeeze(-1)
        joint_scale = h.joint_logit_scale.clamp(max=math.log(100.0)).exp()
        z = prior + torch.sigmoid(h.residual_gate) * (joint_scale * cosine + residual)
        return list(torch.split(z.float(), counts))

    def finish(self, request: SystemOneRequest, plan: Plan, outputs, **_):
        import torch

        with torch.inference_mode():
            zs = self.logits(outputs[0][0], plan.data)
            probs = [p.tolist() for p in torch.split(torch.cat([z.softmax(-1) for z in zs]).cpu(),
                                                     [len(z) for z in zs])]
        answers = {}
        for (name, kind, _, _, ids), p in zip(plan.data["fields"], probs):
            by_id = dict(zip(ids, p))
            q = request.questions[name]
            if kind == "noul":
                answers[name] = make_noul(by_id["true"])
            elif kind == "choice":
                answers[name] = make_choice([str(k) for k in q.criteria], [by_id[str(k)] for k in q.criteria])
            else:
                answers[name] = make_score(list(q.criteria), [by_id[str(i)] for i in range(len(q.criteria))])
        usage = Usage(input_tokens=plan.tokens, output_tokens=len(answers))
        return SystemOneResponse(model="", answers=answers, usage=usage), {"format": self.name, "tokens": plan.tokens}

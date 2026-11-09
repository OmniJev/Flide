"""Intern-Decision family (internlm/Intern-Decision-4B, -2B, -0.8B on Qwen3.5): the release's inference.py.

A request is one causal sequence in the model's chat template: a system line, a user turn with the state (indented
JSON) and a decision schema that lists every question with its options marked by symbols A..Z, a..z, 0..9, then an
assistant turn holding a JSON object with one "<decision>" marker per question. A request's images (1 to 8, data URLs or
base64) go ahead of that text in the user turn, as the release places them, at the processor's own resolution. Each
question's answer is the softmax
over its symbols of the logits one position before its marker, rescaled by the checkpoint's temperature (the
DEFAULT_TEMPERATURE of its inference.py). Each question is a branch that ends one token before its marker and is read
at its last token, the logits of that position in the one pass; the branches share everything before the first marker.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from flide.api.answers import make_choice, make_noul, make_score
from flide.api.schema import SystemOneRequest, SystemOneResponse, Usage
from flide.formats import register
from flide.formats.base import Format, Plan, Read
from flide.formats.next_token import TokenRows

DECISION = "<decision>"
SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
SYSTEM_PROMPT = ("You are a careful decision assistant. Use the state and decision schema in the user message to make "
                 "the requested decisions. For every field, choose exactly one answer symbol (e.g. A, B, C, ...) from its "
                 "listed options and return one valid JSON object mapping each field name to its chosen symbol. Use the "
                 "field names and symbols exactly as given. Do not include explanations, Markdown, or extra text.")
ASK = "Return one answer for every field using the supplied answer symbols.\n\n## State\n"
YES = "The answer is yes (affirmative, or align with the claim)."
NO = "The answer is no (negative, or disagree with the claim)."
MAX_QUESTIONS = 16
MAX_IMAGES = 8
READ_HEAD = "symbols"


def _given(model) -> dict:
    """The fields a request set on a pydantic object, as the request wrote them."""
    if model is None:
        return {}
    return {k: getattr(model, k) for k in model.model_fields_set} | dict(model.model_extra or {})


def _options(q) -> list[tuple[str, str]]:
    if q.type == "choice":
        return [(str(k), str(v)) for k, v in q.criteria.items()]
    if q.type == "score":
        return [(str(i), str(v)) for i, v in enumerate(q.criteria)]
    given = _given(q.criteria)
    yes = next((str(given[k]) for k in given if str(k).lower() in {"yes", "true", "1"}), YES)
    no = next((str(given[k]) for k in given if str(k).lower() in {"no", "false", "0"}), NO)
    return [("no", no), ("yes", yes)]


@register("intern_decision")
class InternDecision(Format):
    backbones = ("qwen3_5", "qwen3_5_text")

    def __init__(self, model_dir: str, tokenizer=None, temperature: float | None = None, max_length: int = 262144):
        import transformers

        self.model_dir = model_dir
        self.tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        if DECISION not in self.tokenizer.get_added_vocab():
            raise ValueError(f"{model_dir}: the tokenizer has no {DECISION} token")
        self.marker = self.tokenizer.convert_tokens_to_ids(DECISION)
        self.symbol_ids = []
        for s in SYMBOLS:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"answer symbol {s!r} is not one token")
            self.symbol_ids.append(ids[0])
        self.temperature = float(temperature) if temperature is not None else self._release_temperature()
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.max_length = int(max_length)

    def _release_temperature(self) -> float:
        path = Path(self.model_dir) / "inference.py"
        m = re.search(r"^DEFAULT_TEMPERATURE\s*=\s*([0-9.eE+-]+)", path.read_text() if path.is_file() else "", re.M)
        if m is None:
            raise ValueError(f"{self.model_dir}: no DEFAULT_TEMPERATURE in inference.py; pass the temperature")
        return float(m.group(1))

    @property
    def processor(self):
        """The checkpoint's processor with this family's tokenizer, as the release builds it (images only)."""
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
        added, script = Path(model_dir) / "added_tokens.json", Path(model_dir) / "inference.py"
        if not (added.is_file() and script.is_file()):
            return False
        try:
            return DECISION in json.loads(added.read_text()) and "DECISION_TOKEN" in script.read_text()
        except (ValueError, OSError):
            return False

    def token_heads(self) -> dict[str, list[int]]:
        return {READ_HEAD: list(self.symbol_ids)}

    def load_heads(self, backbone) -> dict[str, Any]:
        weight, bias = backbone.output_head()
        return {READ_HEAD: TokenRows(weight, self.symbol_ids, bias, softcap=getattr(backbone, "logit_softcap", None))}

    def plan(self, request: SystemOneRequest, **_) -> Plan:
        refs = getattr(request, "images", None) or []
        if len(refs) > MAX_IMAGES:
            raise ValueError(f"{len(refs)} images, limit is {MAX_IMAGES}")
        if not 1 <= len(request.questions) <= MAX_QUESTIONS:
            raise ValueError(f"{len(request.questions)} questions, limit is {MAX_QUESTIONS}")
        lines, fields = [], []
        for name, q in request.questions.items():
            options = _options(q)
            if not 1 <= len(options) <= len(SYMBOLS):
                raise ValueError(f"{len(options)} options, limit is {len(SYMBOLS)}")
            instructions = q.instructions if "instructions" in q.model_fields_set else ""
            lines.append(f"{name}: {instructions}")
            lines += [f"    {s} = {value}: {text}" for s, (value, text) in zip(SYMBOLS, options)]
            fields.append((name, q.type, options))
        state = json.dumps(request.state, ensure_ascii=False, indent=2, sort_keys=False)
        user = ASK + state + "\n## Decision schema\n" + "\n".join(lines)
        if DECISION in user:
            raise ValueError("the reserved decision marker appears in the request")
        skeleton = json.dumps(dict.fromkeys([n for n, _, _ in fields], DECISION), ensure_ascii=False, indent=4)
        content = [{"type": "image"}] * len(refs) + [{"type": "text", "text": user}] if refs else user
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content},
                    {"role": "assistant", "content": skeleton}]
        template = self.processor if refs else self.tokenizer
        text = template.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                            enable_thinking=False, add_vision_id=True)
        vision = None
        if refs:
            from flide.formats.next_token import NextToken

            proc = self.processor
            out = proc(text=[text], images=[NextToken._open(r) for r in refs], return_tensors="pt", padding=False)
            ids = out["input_ids"][0].tolist()
            vision = {"pixel_values": out["pixel_values"], "image_grid_thw": out["image_grid_thw"]}
        else:
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > self.max_length:
            raise ValueError(f"request is {len(ids)} tokens, limit is {self.max_length}")
        at = [i - 1 for i, t in enumerate(ids) if t == self.marker]
        if len(at) != len(fields) or at[0] < 1:
            raise ValueError("decision markers do not match the questions")
        cut = at[0]
        if vision is not None:
            vision["mm_token_type_ids"] = (out["input_ids"][:, :cut] == self.processor.image_token_id).long()
        return Plan(ids[:cut], [ids[cut: p + 1] for p in at], (Read(READ_HEAD),), fields, vision=vision)

    def finish(self, request: SystemOneRequest, plan: Plan, outputs, **_):
        rows = [out[0][0] for out in outputs]
        answers = {}
        for (name, kind, options), logits in zip(plan.data, rows):
            z = [x / self.temperature for x in logits[: len(options)]]
            m = max(z)
            w = [math.exp(x - m) for x in z]
            s = sum(w)
            probs = [x / s for x in w]
            if kind == "noul":
                answers[name] = make_noul(probs[1])
            elif kind == "choice":
                answers[name] = make_choice([v for v, _ in options], probs)
            else:
                answers[name] = make_score(list(request.questions[name].criteria), probs)
        usage = Usage(input_tokens=plan.tokens, output_tokens=len(plan.data))
        return SystemOneResponse(model="", answers=answers, usage=usage), {"format": "intern_decision",
                                                                          "tokens": plan.tokens}

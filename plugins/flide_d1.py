"""d1 family (LiquidAI/d1-3B on LFM2-VL): the release's prompt.py and runner.py.

A request is the state then each question in the checkpoint's chat turns, read at the answer slot right after the
assistant header: `<|startoftext|><|im_start|>user\\n`, the pictures' markup, the state (a string as it is, anything
else as indented JSON), a blank line, `QUESTION:`, then per question its text, its options and a reply line,
`<|im_end|>\\n<|im_start|>assistant\\n`. The state is the shared prefix and each question a branch; a request of one
question is tokenized whole, as the release reads it in one row. Each option scores the best float32 logit among its token
forms (yes, Yes, YES; a choice's code and the code after a space; a score's digit), and the answer is the softmax over
the options: a noul's P(yes), a choice's pick, a score's expected level. Pictures are capped at 1024 x 1024 pixels
(bicubic) and go through the checkpoint's processor, as the release does.
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

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
YES_FORMS, NO_FORMS = ("yes", "Yes", "YES"), ("no", "No", "NO")
MAX_PIXELS = 1024 * 1024
HEAD = "logits"
# codes tried, in order, when an option's own code is not one token or is taken (the release's fallback pool)
POOL = ([chr(c) for c in range(ord("A"), ord("Z") + 1)] + [f"{i:02d}" for i in range(100)]
        + [chr(c) for c in range(ord("a"), ord("z") + 1)] + [f"#{i}" for i in range(200)]
        + [chr(a) + chr(b) for a in range(ord("A"), ord("Z") + 1) for b in range(ord("A"), ord("Z") + 1)])
PLACEHOLDER = re.compile(r"^opt\d+$")


def _given(model) -> dict:
    """The fields a request set on a pydantic object, as the request wrote them."""
    if model is None:
        return {}
    if isinstance(model, dict):
        return model
    return {k: getattr(model, k) for k in model.model_fields_set} | dict(model.model_extra or {})


def option_codes(labels: list) -> list[str]:
    labs = [str(x).strip() for x in labels]
    if labs and all(len(k) == 1 and k.isalpha() for k in labs):
        return labs
    if len(labs) <= 26:
        return [chr(ord("A") + i) for i in range(len(labs))]
    return [f"{i:02d}" for i in range(len(labs))]


class _Hidden:
    """The final-norm hidden states at the read rows, left on the device: `finish` takes the logits of the few tokens
    each question reads, in float32 (the release rounds its logits to 16 bits, where two close options can tie)."""

    on_device = True

    def __init__(self, weight):
        self.weight = weight.detach()

    def __call__(self, h):
        return h


@register("d1")
class D1(Format):
    backbones = ("lfm2_vl", "lfm2")
    max_choice_options = 255

    def __init__(self, model_dir: str, tokenizer=None, max_request_tokens: int = 262144):
        import transformers

        self.model_dir = model_dir
        self.tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        bos = getattr(self.tokenizer, "bos_token", None)
        self.bos = bos if isinstance(bos, str) else ""
        self.max_request_tokens = int(max_request_tokens)
        self._ids: dict[str, int | None] = {}
        self.yes, self.no = self._forms(YES_FORMS), self._forms(NO_FORMS)
        if not self.yes or not self.no:
            raise ValueError("the tokenizer has no single-token yes/no")

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        d = Path(model_dir)
        if getattr(config, "model_type", None) not in cls.backbones:
            return False
        try:
            auto = json.loads((d / "config.json").read_text()).get("auto_map") or {}
        except (OSError, ValueError):
            return False
        return str(auto.get("AutoModel", "")).startswith("modeling_d1.") and (d / "prompt.py").is_file()

    @property
    def processor(self):
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

    # ------------------------------------------------------------------ tokens

    def _one(self, text: str) -> int | None:
        if text not in self._ids:
            enc = self.tokenizer.encode(text, add_special_tokens=False)
            self._ids[text] = enc[0] if len(enc) == 1 else None
        return self._ids[text]

    def _forms(self, texts) -> list[int]:
        out = []
        for t in texts:
            i = self._one(t)
            if i is not None and i not in out:
                out.append(i)
        return out

    def aliases(self, labels: list) -> list[tuple[str, int]]:
        used, out = set(), []

        def take(raw: str) -> bool:
            i = self._one(raw)
            if i is None or i in used:
                return False
            out.append((raw, i))
            used.add(i)
            return True

        for code in option_codes(labels):
            if not take(code) and not any(take(raw) for raw in POOL):
                raise ValueError(f"no single-token code left for {len(labels)} options")
        return out

    def groups(self, q) -> list[list[int]]:
        """Token ids per option whose best logit scores it."""
        if q.type == "noul":
            return [self.yes, self.no]
        if q.type == "score":
            out = [self._forms([str(i)]) for i in range(len(q.criteria))]
            if any(not g for g in out):
                raise ValueError(f"a score of {len(q.criteria)} levels needs single-token digits")
            return out
        return [[tid] + [i for i in self._forms([f" {code}"]) if i != tid] for code, tid in self.aliases(list(q.criteria))]

    # ------------------------------------------------------------------ text

    def state_block(self, state) -> str:
        if isinstance(state, str):
            return f"{state}\n\n"
        return json.dumps(state, ensure_ascii=False, indent=2) + "\n\n"

    def question_block(self, q) -> str:
        instructions = getattr(q, "instructions", None) or ""
        if q.type == "choice":
            labels = list(q.criteria)
            lines = []
            for (code, _), lab in zip(self.aliases(labels), labels):
                desc = q.criteria[lab]
                lines.append(f"{code} {desc or str(lab).replace('_', ' ')}")
            return f"{instructions}\n\nOptions:\n" + "\n".join(lines) + "\n\nReply with the option code only."
        if q.type == "noul":
            crit = _given(q.criteria)
            extra = f"\nYes: {crit.get('true')}\nNo: {crit.get('false')}" if crit else ""
            return f"{instructions}{extra}\n\nReply with yes or no only."
        legend = "\n".join(f"{i} {name}" for i, name in enumerate(q.criteria))
        return f"{instructions}\n\n{legend}\n\nReply with a single digit 0-{len(q.criteria) - 1} only."

    def prefix_text(self, state, images: str = "") -> str:
        body = "" if state is None else f"{self.state_block(state)}\nQUESTION:\n"
        return f"{self.bos}{IM_START}user\n{images}{body}"

    def suffix_text(self, q) -> str:
        return f"{self.question_block(q)}{IM_END}\n{IM_START}assistant\n"

    def image_markup(self, n: int) -> str:
        msgs = [{"role": "user", "content": [*([{"type": "image"}] * n), {"type": "text", "text": "\x00"}]}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
        head = f"{IM_START}user\n"
        return text[text.index(head) + len(head): text.index("\x00")]

    # ------------------------------------------------------------------ plan

    def load_heads(self, backbone) -> dict[str, Any]:
        from flide import settings

        if settings.batch_invariant:
            raise ValueError("the d1 family does not run with --batch-invariant yet")
        weight, bias = backbone.output_head()
        if bias is not None or not hasattr(weight, "T"):
            raise ValueError("d1: an output head with a bias or an adapter is not supported")
        self._head = _Hidden(weight)
        return {HEAD: self._head}

    def plan(self, request: SystemOneRequest, **_) -> Plan:
        qs = list(request.questions.items())
        if not qs:
            raise ValueError("no questions")
        groups = [self.groups(q) for _, q in qs]
        refs = getattr(request, "images", None) or []
        state = request.state
        suffixes = [self.suffix_text(q) for _, q in qs]
        enc = lambda s: self.tokenizer.encode(s, add_special_tokens=False)
        vision = None
        if not refs:
            prefix = enc(self.prefix_text(state))
            if len(qs) == 1:   # the release reads a lone question as one row, tokenized whole
                whole = enc(self.prefix_text(state) + suffixes[0])
                cut = 0
                while cut < min(len(prefix), len(whole) - 1) and prefix[cut] == whole[cut]:
                    cut += 1
                prefix, branches = whole[:cut], [whole[cut:]]
            else:
                branches = [enc(s) for s in suffixes]
        else:
            from flide.formats.next_token import NextToken

            pics = [self._cap(NextToken._open(r)) for r in refs]
            head = self.prefix_text(state, self.image_markup(len(pics)))
            text = head + suffixes[0] if len(qs) == 1 else head
            out = self.processor(text=[text], images=[pics], return_tensors="pt", add_special_tokens=False)
            if "pixel_attention_mask" in out:   # the release cuts the padding patches every picture is given
                n = int(out["pixel_attention_mask"].sum(1).max())
                out["pixel_values"] = out["pixel_values"][:, :n]
                out["pixel_attention_mask"] = out["pixel_attention_mask"][:, :n]
            ids = out["input_ids"][0].tolist()
            vision = {k: out[k] for k in ("pixel_values", "spatial_shapes", "pixel_attention_mask") if k in out}
            if len(qs) == 1:
                last = max(i for i, t in enumerate(ids) if t == self.processor.image_token_id)
                cut = max(last + 1, len(ids) - len(enc(suffixes[0])))
                prefix, branches = ids[:cut], [ids[cut:]]
            else:
                prefix, branches = ids, [enc(s) for s in suffixes]
        total = len(prefix) + sum(len(b) for b in branches)
        if total > self.max_request_tokens:
            raise ValueError(f"request is {total} tokens, limit is {self.max_request_tokens}")
        data = [(name, q.type, groups[k]) for k, (name, q) in enumerate(qs)]
        return Plan(prefix, branches, (Read(HEAD),), data, vision=vision)

    @staticmethod
    def _cap(image):
        image = image.convert("RGB") if hasattr(image, "convert") else image
        w, h = image.size
        if w * h <= MAX_PIXELS:
            return image
        from PIL import Image

        s = math.sqrt(MAX_PIXELS / (w * h))
        return image.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.BICUBIC)

    def finish(self, request: SystemOneRequest, plan: Plan, outputs, **_):
        import torch

        flat = sorted({i for _, _, gs in plan.data for g in gs for i in g})
        at = {t: k for k, t in enumerate(flat)}
        h = torch.stack([out[0][0] for out in outputs]).float()
        w = self._head.weight.index_select(0, torch.tensor(flat, device=h.device)).float()
        picked = (h @ w.T).tolist()
        answers = {}
        for (name, kind, gs), logits in zip(plan.data, picked):
            scores = [max(logits[at[i]] for i in g) for g in gs]
            m = max(scores)
            w = [math.exp(s - m) for s in scores]
            z = sum(w)
            probs = [x / z for x in w]
            q = request.questions[name]
            if kind == "noul":
                answers[name] = make_noul(probs[0])
            elif kind == "choice":
                answers[name] = make_choice([str(k) for k in q.criteria], probs)
            else:
                answers[name] = make_score(list(q.criteria), probs)
        usage = Usage(input_tokens=plan.tokens, output_tokens=0)
        return SystemOneResponse(model="", answers=answers, usage=usage), {"format": "d1", "tokens": plan.tokens}

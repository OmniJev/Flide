"""next_token: any chat model, read at the first token of its reply.

The user turn holds the state, then the question and its options lettered A, B, C, ..., then an instruction
to answer with one letter; thinking is off where the chat template has a switch. The answer is the softmax
of the output head over the letter tokens at the position where the reply starts. Every question of a
request shares the state as its prefix.

This is the family for chat models that were never trained as decision models. It is built in and it
serves a checkpoint that no installed family claims; it reads up to 26 options.

`style="vllm"` renders and reads as vLLM's /v1/systemone does for chat models (NextTokenStrategy, vLLM 0.32): the
state as given (JSON without indentation unless it is a string), a blank line, then "Question: ..." and one line per
option labeled A, B, ... ("A: name - description"), yes / no for a noul, 0, 1, ... for a score, and that type's reply
instruction; each label read as the reply's first token after the chat template's generation prompt. The same
requests then give the same probabilities on both engines (up to arithmetic). `style="sglang"` renders as SGLang's
/v1/systemone does (0.5.21, prompt format version 1): the same as vllm except that JSON is compact, a noul opens with
"Is the following true?" and lists only the outcomes that have a description.

Images: a request's `images` (data URLs or base64, as vLLM's /v1/systemone takes them) go ahead of the text in the user
turn, as vLLM places them, at the image processor's own resolution; they belong to the shared prefix.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import string
from typing import Any

from ..api.answers import make_choice, make_noul, make_score
from ..api.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest, SystemOneResponse, Usage
from . import register
from .base import Format, Plan, Read
from .tokens import Pieces

log = logging.getLogger("flide.next_token")

LETTERS = string.ascii_uppercase
MARK = "FLIDE_SPLIT_MARK"
READ = Read("letters")
VLLM_LABELS = {"choice": tuple(LETTERS), "noul": ("yes", "no"), "score": tuple(string.digits)}
VLLM_REPLY = {"choice": "Answer with the letter of one option only.", "noul": "Answer with yes or no only.",
              "score": "Answer with the number of one level only."}


def _text(entry: Any) -> str:
    if entry is None:
        return ""
    if isinstance(entry, str):
        return entry.strip()
    return json.dumps(entry, ensure_ascii=False, indent=2)


class TokenRows:
    """Output-head rows of some token ids, in float32: their logits for final-norm hidden states."""

    def __init__(self, weight, ids: list[int], bias=None, softcap: float | None = None):
        import torch

        idx = torch.tensor(ids, device=weight.device)
        self.weight = weight.detach().index_select(0, idx).float().contiguous()
        self.bias = None if bias is None else bias.detach().index_select(0, idx).float()
        self.softcap = softcap

    def __call__(self, hidden):
        from .. import settings

        if settings.batch_invariant:
            from ..kernels.invariant import mm

            out = mm(hidden.float(), self.weight)
        else:
            out = hidden.float() @ self.weight.T
        out = out if self.bias is None else out + self.bias
        if self.softcap:
            out = (out / self.softcap).tanh() * self.softcap
        return out


@register("next_token")
class NextToken(Format):
    backbones = ()

    def __init__(self, model_dir: str, tokenizer=None, max_branch_tokens: int = 32768, max_request_tokens: int = 65536,
                 style: str = "flide"):
        import transformers

        if style not in ("flide", "vllm", "sglang"):
            raise ValueError(f"next_token style {style!r}: flide, vllm or sglang")
        self.style = style
        self.model_dir = model_dir
        self.tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        if self.tokenizer.chat_template is None:
            raise ValueError(f"{model_dir} has no chat template; next_token needs one")
        self.max_branch_tokens = max_branch_tokens
        self.max_request_tokens = max_request_tokens
        self.template_kwargs = {"enable_thinking": False}
        self._pieces = Pieces(self.tokenizer)
        self._mark_ok, self._mark_checks, self._cut_checks = True, 0, 0
        if style != "flide":
            self.labels = [label for kind in ("choice", "noul", "score") for label in VLLM_LABELS[kind]]
            self.letter_ids = self._reply_label_ids(self.labels)
            return
        self.letter_ids = []
        for letter in LETTERS:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                break
            self.letter_ids.append(ids[0])
        if len(self.letter_ids) < 2:
            raise ValueError("the tokenizer does not hold A and B as single tokens")

    def _reply_label_ids(self, labels) -> list[int]:
        """Each label's token as the reply's first token, as vLLM finds it: the generation prompt's ids after its last
        added token, decoded, the label appended, encoded again; the label must add exactly one distinct token."""
        tok = self.tokenizer
        probe = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True,
                                        **self.template_kwargs)
        ids = tok.encode(probe, add_special_tokens=False)
        added = set(tok.get_added_vocab().values())
        tail = ids[max((i + 1 for i, t in enumerate(ids) if t in added), default=0):]
        text = tok.decode(tail)
        out: list[int] = []
        for label in labels:
            ext = tok.encode(text + label, add_special_tokens=False)
            if ext[:-1] != tail or ext[-1] in out:
                raise ValueError(f"label {label!r} is not one distinct token after this model's chat prompt")
            out.append(ext[-1])
        return out

    def load_heads(self, backbone) -> dict[str, Any]:
        weight, bias = backbone.output_head()
        return {"letters": TokenRows(weight, self.letter_ids, bias, softcap=getattr(backbone, "logit_softcap", None))}


    def _plain(self, entry: Any) -> str:
        if self.style == "sglang" and entry is not None and not isinstance(entry, str):
            return json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        return "" if entry is None else str(entry)

    def _question_vllm(self, q) -> tuple[str, list[str], str, list[int]]:
        """(kind, option names, the question as vLLM or SGLang writes it, the read positions of its labels)."""
        if isinstance(q, NoulQuestion):
            kind, c = "noul", q.criteria
            names = ["yes", "no"]
            descs = [None if c is None else self._plain(c.true), None if c is None else self._plain(c.false)]
        elif isinstance(q, ChoiceQuestion):
            kind, names = "choice", [str(n) for n in q.criteria]
            descs = [self._plain(v) for v in q.criteria.values()]
        elif isinstance(q, ScoreQuestion):
            kind, names = "score", [self._plain(level) if self.style == "sglang" else str(level) for level in q.criteria]
            descs = [None] * len(names)
        else:
            raise TypeError(f"unknown question type {type(q).__name__}")
        labels = VLLM_LABELS[kind]
        if len(names) > len(labels):
            raise ValueError(f"{len(names)} options; a {kind} reads at most {len(labels)}")
        raw = q.instructions
        blank = self.style == "sglang" and not (raw.strip() if isinstance(raw, str) else raw)
        ins = "" if blank else self._plain(raw)
        if kind == "noul" and self.style == "sglang":
            lines = [f"Is the following true? {ins}" if ins else "Is the following true?"]
        else:
            lines = [f"Question: {ins}"] if ins else []
        for label, name, desc in zip(labels, names, descs):
            if kind == "noul":
                if desc or self.style == "vllm":
                    lines.append(f"{label}: {desc}" if desc else label)
            else:
                lines.append(f"{label}: {name} - {desc}" if desc else f"{label}: {name}")
        lines.append(VLLM_REPLY[kind])
        return kind, names, "\n".join(lines), [self.labels.index(label) for label in labels[: len(names)]]

    def _question(self, q) -> tuple[str, list[str], list[str]]:
        """(kind, option names, suffix text)."""
        if isinstance(q, NoulQuestion):
            c = q.criteria
            yes = _text(c.true) if c and c.true is not None else ""
            no = _text(c.false) if c and c.false is not None else ""
            names = ["yes", "no"]
            descs = [yes, no]
            kind = "noul"
            instructions = _text(q.instructions) or "Is the statement true?"
        elif isinstance(q, ChoiceQuestion):
            names = list(q.criteria)
            descs = [_text(q.criteria[n]) for n in names]
            kind = "choice"
            instructions = _text(q.instructions) or "Which option applies?"
        elif isinstance(q, ScoreQuestion):
            names = [str(i) for i in range(len(q.criteria))]
            descs = [_text(c) for c in q.criteria]
            kind = "score"
            instructions = _text(q.instructions) or "Which level applies?"
        else:
            raise TypeError(f"unknown question type {type(q).__name__}")
        if len(names) > len(self.letter_ids):
            raise ValueError(f"{len(names)} options; the next_token format reads at most {len(self.letter_ids)}")
        lines = []
        for letter, name, desc in zip(LETTERS, names, descs):
            label = name if kind != "score" else f"level {name}"
            lines.append(f"{letter}. {label}: {desc}" if desc else f"{letter}. {label}")
        suffix = (f"Question: {instructions}\nOptions:\n" + "\n".join(lines)
                  + f"\nAnswer with the letter of one option ({', '.join(LETTERS[: len(names)])}).")
        return kind, names, suffix

    def _render(self, state: Any, suffix: str, images: int = 0) -> str:
        if images:
            user = (state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)) + "\n\n" + suffix
            content = [{"type": "image"}] * images + [{"type": "text", "text": user}]
            return self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                      add_generation_prompt=True, **self.template_kwargs)
        if self.style == "sglang":
            user = self._plain(state) + "\n\n" + suffix
        elif self.style == "vllm":
            user = (state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)) + "\n\n" + suffix
        else:
            user = (_text(state) + "\n\n" if state is not None else "") + suffix
        try:
            return self.tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                                      add_generation_prompt=True, **self.template_kwargs)
        except TypeError:
            return self.tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                                      add_generation_prompt=True)

    @property
    def processor(self):
        p = getattr(self, "_processor", None)
        if p is None:
            import transformers

            try:
                p = self._processor = transformers.AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
            except ImportError as e:
                if "orchvision" not in str(e):
                    raise
                raise ImportError("images and video need torchvision; install it from the same index as torch, e.g. "
                                  "pip install torchvision --index-url https://download.pytorch.org/whl/cu126") from e
            if p.tokenizer.chat_template is None:
                p.tokenizer.chat_template = p.chat_template
        return p

    @staticmethod
    def _open(ref: Any):
        from PIL import Image

        if not isinstance(ref, str):
            raise ValueError("images: each image is a data URL or base64 string")
        raw = base64.b64decode(ref.split(",", 1)[1] if ref.startswith("data:") else ref)
        with Image.open(io.BytesIO(raw)) as im:
            return im.convert("RGB")

    def _vision(self, prefix: list[int], images: list, text: str, suffix: list[int]) -> tuple[list[int], dict]:
        """The prefix with each image placeholder repeated once per merged patch, and the pixel tensors. Processors
        other than Qwen's (Gemma 4: begin and end tokens around each image's soft tokens) expand the first prompt
        `text` whole, which must end with that prompt's `suffix`."""
        import torch

        proc = self.processor
        ip = proc.image_processor
        if not type(ip).__name__.startswith("Qwen2VL"):
            out = proc(text=[text], images=images, return_tensors="pt")
            ids = out["input_ids"][0].tolist()
            n = len(ids) - len(suffix)
            if ids[n:] != suffix:
                raise ValueError("the images must sit in the state, before every question")
            vision = {k: out[k] for k in ("pixel_values", "image_grid_thw", "image_position_ids") if k in out}
            if "mm_token_type_ids" in out:
                vision["mm_token_type_ids"] = out["mm_token_type_ids"][:, :n]
            return ids[:n], vision
        out = ip(images=images, return_tensors="pt")
        counts = (out["image_grid_thw"].prod(-1) // ip.merge_size ** 2).tolist()
        pad = self.processor.image_token_id
        expanded, k = [], 0
        for t in prefix:
            if t == pad:
                expanded.extend([pad] * counts[k])
                k += 1
            else:
                expanded.append(t)
        if k != len(counts):
            raise ValueError(f"{k} image placeholders in the shared prefix for {len(counts)} images")
        vision = {"pixel_values": out["pixel_values"], "image_grid_thw": out["image_grid_thw"],
                  "mm_token_type_ids": (torch.tensor(expanded) == pad).long()[None]}
        return expanded, vision

    MARK_CHECKS = 32
    CUT_CHECKS = 8

    def _prompts(self, state: Any, marked: str, end: int, suffixes: list[str], images: int = 0) -> list[str]:
        """Every question's chat prompt, from the rendering with MARK in place of the question. The first prompt of
        every plan, and all of them in the first MARK_CHECKS plans, are also rendered whole and compared; one
        mismatch renders every prompt whole from then on."""
        if not self._mark_ok:
            return [self._render(state, x, images) for x in suffixes]
        head, tail = marked[:end], marked[end + len(MARK):]
        texts = [head + x + tail for x in suffixes]
        check = len(texts) if self._mark_checks < self.MARK_CHECKS else 1
        self._mark_checks += 1
        if any(self._render(state, x, images) != t for x, t in zip(suffixes[:check], texts)):
            self._mark_ok = False
            log.info("prompt marker shortcut off: the chat template does not place the question verbatim")
            return [self._render(state, x, images) for x in suffixes]
        return texts

    def plan(self, request: SystemOneRequest, **_) -> Plan:
        if self.style != "flide":
            questions = [(qid, kind, (names, at), text)
                         for qid, (kind, names, text, at) in ((qid, self._question_vllm(q))
                                                              for qid, q in request.questions.items())]
        else:
            questions = [(qid, *self._question(q)) for qid, q in request.questions.items()]
        refs = getattr(request, "images", None) or []
        if refs and self.style == "flide":
            raise ValueError("images: the next_token format takes them in style vllm")
        marked = self._render(request.state, MARK, len(refs))
        end = marked.index(MARK)
        texts = self._prompts(request.state, marked, end, [suffix for _, _, _, suffix in questions], len(refs))
        shared = marked[:end]
        c = self._pieces.last_cut(marked, end - 1) if all(t.startswith(shared) for t in texts) else 0
        head, prefix_tail, *tails = self._pieces.encode(self.tokenizer,
                                                        [marked[:c], shared[c:]] + [t[c:] for t in texts])
        if c and self._cut_checks < self.CUT_CHECKS:
            self._cut_checks += 1
            if head + tails[0] != self.tokenizer.backend_tokenizer.encode(texts[0], add_special_tokens=False).ids:
                self._pieces.ok = False
                log.warning("tokenizing in pieces off: a cut changed the tokens of a prompt")
                head, prefix_tail, *tails = self._pieces.encode(self.tokenizer, ["", shared] + texts)
        m = len(prefix_tail)
        while m > 0 and not all(t[:m] == prefix_tail[:m] for t in tails):
            m -= 1
        n = len(head) + m
        suffixes = [t[m:] for t in tails]
        if any(len(s) == 0 for s in suffixes):
            raise ValueError("a question renders to nothing after the shared state")
        longest = n + max(len(s) for s in suffixes)
        total = n + sum(len(s) for s in suffixes)
        if longest > self.max_branch_tokens:
            raise ValueError(f"state plus longest question is {longest} tokens, limit is {self.max_branch_tokens}")
        if total > self.max_request_tokens:
            raise ValueError(f"request is {total} tokens, limit is {self.max_request_tokens}")
        data = [(qid, kind, names) for qid, kind, names, _ in questions]
        if refs:
            prefix, vision = self._vision(head + tails[0][:m], [self._open(r) for r in refs], texts[0], suffixes[0])
            return Plan(prefix, suffixes, (READ,), data, vision=vision)
        return Plan(head + tails[0][:m], suffixes, (READ,), data)

    def finish(self, request: SystemOneRequest, plan: Plan, outputs, **_):
        import math

        answers = {}
        for (qid, kind, names), out in zip(plan.data, outputs):
            if self.style != "flide":
                names, at = names
                logits = [out[0][0][i] for i in at]
            else:
                logits = out[0][0][: len(names)]
            m = max(logits)
            w = [math.exp(x - m) for x in logits]
            s = sum(w)
            probs = [x / s for x in w]
            if kind == "noul":
                answers[qid] = make_noul(probs[0])
            elif kind == "choice":
                answers[qid] = make_choice(names, probs)
            else:
                levels = list(request.questions[qid].criteria)
                answers[qid] = make_score(levels, probs)
        usage = Usage(input_tokens=plan.tokens, output_tokens=len(request.questions))
        return SystemOneResponse(model="", answers=answers, usage=usage), {"format": "next_token",
                                                                          "prefix_tokens": len(plan.prefix),
                                                                          "branches": len(plan.suffixes)}

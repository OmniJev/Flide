"""Decision 2.0 family (vllm-sr/Decision-2.0-Kai-0.6B, vLLM Semantic Router's decision models): a text backbone under
backbone/ and a candidate head (decision_head.safetensors), read as the release's exact path reads them.

Every question is one causal sequence, "Context:", the state, the task type, the question and "Options:", then each
option as {"description", "key"} JSON between <option> tags and a closing line ending in "Decision:"; states,
instructions and descriptions that are not strings go in as compact JSON with sorted keys. The text up to "Options:",
each option and the closing line are tokenized apart, and a request's questions share the tokens they start with. The
head reads the last token of every option and of the sequence in float32: normed, an option's key times the query over
sqrt(head_dim), plus a one-layer MLP over both. Probabilities are the softmax of those logits; a score whose number of
levels score_bias.json names has that row added first.

A question the release answers with an error (an option list or criteria it does not read, a sequence over the
checkpoint's max_input_tokens) refuses the request here. A noul whose criteria write both outcomes lists true first
(the request's own order is not kept); with fewer, "No" and "Yes" stand in and false comes first. Text only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from flide import settings
from flide.api.answers import make_choice, make_noul, make_score
from flide.api.schema import ChoiceQuestion, NoulQuestion, SystemOneRequest, SystemOneResponse, Usage
from flide.formats import register
from flide.formats.base import Format, Plan, Read
from flide.formats.tokens import Pieces

FORMAT = "vllm-sr-decision"
PROMPT_VERSION = "decision2-segmented-options-global-query-v1"
BACKBONE, HEAD, MANIFEST, BIAS = "backbone", "decision_head.safetensors", "MODEL_MANIFEST.json", "score_bias.json"
FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json",
         "decision_config.json", HEAD, MANIFEST, BIAS)
CLOSING = "\n\nSelect the single option best supported by the context and instructions.\nDecision:"
READ = "decision"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def payload(value: Any) -> str:
    return value if isinstance(value, str) else canonical(value)


def _payload_ok(value: Any, nullable: bool = False) -> bool:
    """Text, an object or an array, as the release takes it."""
    return nullable if value is None else isinstance(value, (str, dict, list))


def options(q) -> tuple[str, list[tuple[str, Any]]]:
    """A question's task type and its options as (key, description), as the release orders them."""
    if isinstance(q, NoulQuestion):
        c = q.criteria
        given = {} if c is None else {k: getattr(c, k) for k in ("true", "false") if k in c.model_fields_set}
        if c is not None and c.model_extra:
            raise ValueError("noul criteria only name false and true")
        if len(given) < 2:
            given = {"false": given.get("false", "No"), "true": given.get("true", "Yes")}
        pairs, kind = list(given.items()), "noul"
    elif isinstance(q, ChoiceQuestion):
        pairs, kind = [(str(k), v) for k, v in q.criteria.items()], "choice"
        if not 2 <= len(pairs) <= 255 or not all(k for k, _ in pairs):
            raise ValueError("choice criteria need 2 to 255 options with nonempty keys")
    else:
        pairs, kind = [(str(i), v) for i, v in enumerate(q.criteria)], "score"
        if not 2 <= len(pairs) <= 10:
            raise ValueError("score criteria are 2 to 10 levels")
    if not all(_payload_ok(d, kind == "choice") for _, d in pairs):
        raise ValueError(f"{kind} criteria descriptions must be text, an object or an array")
    return kind, pairs


class Part:
    """One side of the candidate head over hidden rows, in float32: normed, then its projection to `head_dim` and its
    share of the MLP's input, side by side."""

    def __init__(self, t: dict, side: str, project: str, device):
        self.norm = (t[f"{side}_norm.weight"].float().to(device), t[f"{side}_norm.bias"].float().to(device))
        mlp = f"{side}_mlp"
        self.weight = [t[f"{project}.weight"].float().to(device), t[f"{mlp}.weight"].float().to(device)]
        self.bias = t[f"{mlp}.bias"].float().to(device) if f"{mlp}.bias" in t else None

    def __call__(self, hidden):
        import torch

        x = torch.nn.functional.layer_norm(hidden.float(), hidden.shape[-1:], *self.norm, eps=1e-5)
        if settings.batch_invariant:
            from flide.kernels.invariant import mm

            a, b = (mm(x, w) for w in self.weight)
        else:
            a, b = (x @ w.T for w in self.weight)
        return torch.cat([a, b if self.bias is None else b + self.bias], dim=-1)


class Both:
    """Both sides of the head over every row, the option's outputs first and the query's after them: a row is an option's
    last token or the question's, and each branch reads all of its rows in one read (as the CUDA graphs take them)."""

    def __init__(self, option: Part, query: Part):
        self.option, self.query = option, query

    def __call__(self, hidden):
        import torch

        return torch.cat([self.option(hidden), self.query(hidden)], dim=-1)


@register("decision2")
class Decision2(Format):
    backbones = ("qwen3",)
    max_choice_options = 255
    max_score_levels = 10

    def __init__(self, model_dir: str, tokenizer=None):
        import transformers

        self.model_dir = model_dir
        d = Path(model_dir)
        self.spec = json.loads((d / "decision_config.json").read_text())
        if (self.spec.get("prompt_version") != PROMPT_VERSION or self.spec.get("head_variant", "shared") != "shared"
                or self.spec.get("readout") or self.spec.get("dec_residual") is not None):
            raise ValueError(f"{model_dir}: not a Decision 2.0 checkpoint with the shared candidate head")
        manifest = json.loads((d / MANIFEST).read_text())
        if manifest.get("calibration") or manifest.get("storage"):
            raise ValueError(f"{model_dir}: calibrated or packed Decision 2.0 checkpoints are not supported")
        self.max_tokens = int(manifest["max_input_tokens"])
        self.dim = int(self.spec["head_dim"])
        self.bias = {}
        entry = manifest.get("score_bias")
        if entry:
            report = json.loads((d / entry["file"]).read_text())
            self.bias = {int(k): [float(x) for x in v] for k, v in report["offsets"].items()}
            if report["model_sha256"] != manifest["identity"]["model_sha256"]:
                raise ValueError(f"{model_dir}: {BIAS} was fitted for another checkpoint")
        self.tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self._pieces = Pieces(self.tokenizer)
        self._scalar = None

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        return bool(decision_config and decision_config.get("prompt_version") == PROMPT_VERSION
                    and (Path(model_dir) / HEAD).is_file())

    @classmethod
    def run_dir(cls, model_dir: str) -> str:
        """The backbone's config and weights next to the tokenizer, head and configuration of the package, linked once
        under FLIDE_CACHE; a directory that is laid out so already runs as it is."""
        src = Path(model_dir).resolve()
        if not (src / BACKBONE).is_dir():
            return model_dir
        key = hashlib.sha256(f"{src}\ndecision2".encode()).hexdigest()[:16]
        out = Path(os.environ.get("FLIDE_CACHE", Path.home() / ".cache" / "flide")) / "decision2" / key
        if (out / ".done").is_file():
            return str(out)
        out.mkdir(parents=True, exist_ok=True)
        links = {f.name: f for f in (src / BACKBONE).iterdir() if not f.name.startswith(".")}
        links.update({n: src / n for n in FILES if (src / n).is_file()})
        for name, target in links.items():
            link = out / name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target.resolve())
        (out / ".done").write_text(f"{src}\n")
        return str(out)

    def load_heads(self, backbone) -> dict[str, Any]:
        from safetensors.torch import load_file

        t = load_file(str(Path(self.model_dir) / HEAD))
        self._scalar = t["scalar.weight"].float().reshape(-1)
        return {READ: Both(Part(t, "candidate", "key", backbone.device), Part(t, "query", "query", backbone.device))}

    def _encode(self, texts: list[str]) -> list[list[int]]:
        return self.tokenizer(texts, add_special_tokens=False)["input_ids"]

    def _opening(self, texts: list[str]) -> list[list[int]]:
        """Token ids of the questions' openings; the text they start with is tokenized once."""
        if len(texts) < 2 or not self._pieces.ok:
            return self._encode(texts)
        a, b = min(texts), max(texts)
        n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        cut = self._pieces.last_cut(texts[0], n - 1) if n else 0
        if not cut:
            return self._encode(texts)
        head, *tails = self._pieces.encode(self.tokenizer, [texts[0][:cut]] + [t[cut:] for t in texts])
        return [head + t for t in tails]

    def plan(self, request: SystemOneRequest, **_) -> Plan:
        if getattr(request, "images", None):
            raise ValueError(f"images: the {self.name} family reads text only")
        if not _payload_ok(request.state):
            raise ValueError("state must be text, an object or an array")
        state = payload(request.state)
        openings, rest, fields = [], [], []
        for name, q in request.questions.items():
            kind, pairs = options(q)
            if not _payload_ok(q.instructions) or q.instructions == "":
                raise ValueError(f"question {name!r}: instructions are required")
            openings.append(f"Context:\n{state}\n\nTask type: {kind}\nQuestion:\n{payload(q.instructions)}\nOptions:")
            rest.append(["\n<option>\n" + canonical({"key": k, "description": d}) + "\n</option>" for k, d in pairs]
                        + [CLOSING])
            fields.append((name, kind, [k for k, _ in pairs]))
        flat = iter(self._encode([t for parts in rest for t in parts]))
        rows, reads, opened = [], [], []
        for ids, parts, (name, _, _) in zip(self._opening(openings), rest, fields):
            opened.append(len(ids))
            ids, ends = list(ids), []
            for _ in parts:
                ids += next(flat)
                ends.append(len(ids) - 1)
            if len(ids) > self.max_tokens:
                raise ValueError(f"question {name!r} is {len(ids)} tokens, over the {self.max_tokens}-token limit")
            rows.append(ids)
            reads.append((Read(READ, positions=tuple(e - len(ids) for e in ends)),))
        head = min(self._shared(rows), *opened)   # every read lies after it
        return Plan(rows[0][:head], [r[head:] for r in rows], reads, fields)

    @staticmethod
    def _shared(rows: list[list[int]]) -> int:
        """The tokens every row starts with (0 for a single question: it runs as one sequence)."""
        if len(rows) < 2:
            return 0
        a, b = min(rows), max(rows)
        return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)) - 1)

    def logits(self, rows, kind: str) -> list[float]:
        """A question's option logits from its rows: an option's, then the question's own last."""
        import torch

        x = torch.tensor(rows, dtype=torch.float32)
        k, q = x[:-1, : 2 * self.dim], x[-1, 2 * self.dim:]
        z = (k[:, : self.dim] * q[: self.dim]).sum(-1) / math.sqrt(self.dim)
        z = z + torch.nn.functional.gelu(k[:, self.dim:] + q[self.dim:]) @ self._scalar
        out = z.tolist()
        bias = self.bias.get(len(out)) if kind == "score" else None
        return out if bias is None else [x + b for x, b in zip(out, bias)]

    def finish(self, request: SystemOneRequest, plan: Plan, outputs, **_):
        answers = {}
        for (name, kind, keys), branch in zip(plan.data, outputs):
            z = self.logits(branch[0], kind)
            top = max(z)
            w = [math.exp(x - top) for x in z]
            probs = [x / sum(w) for x in w]
            q = request.questions[name]
            if kind == "noul":
                answers[name] = make_noul(probs[keys.index("true")])
            elif kind == "choice":
                answers[name] = make_choice(keys, probs)
            else:
                answers[name] = make_score(list(q.criteria), probs)
        usage = Usage(input_tokens=plan.tokens, output_tokens=len(plan.data))
        return SystemOneResponse(model="", answers=answers, usage=usage), {"format": self.name, "tokens": plan.tokens}

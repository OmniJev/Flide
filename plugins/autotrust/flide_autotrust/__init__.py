"""AutoTrust family (autotrust/JEV-9B, JEV-27B, JEV-27B-VL, JEV-35B, GEV-26B-Decide): System 1 as the release's
serve_decide.py serves it (POST /v1/decide, thinking off).

Every question is its own raw prompt, no chat template: "[kind] <kind>\\n[state] <state>\\n[question] <instructions>\\n
[options]\\n<lines>\\n[decision]:", after "<bos>" on Gemma. A string state goes in as it is, an object as JSON, a list part by
part (strings as they are, {"image": ...} parts as images, anything else as JSON); the request's images come first. A
choice lists its options (each one's description, else its name) as "A) <option>" lines, labeled A to Z, then AA, AB,
... while each is one token after a newline, up to 256; a noul lists false and true, a score 0 to 5 (six levels). The
answer is read at the prompt's last token over the labels' tokens through the output head and its LoRA: each logit plus
the head's bias of its slot (none past the trained slots), over the kind's temperature (calibration.json), softmaxed.
A choice with more options than the trained slots is read in one pass on Qwen and as a tournament on Gemma (groups of
at most 16, then a final of 16), as the release does by default; `strategy` (single, tournament, permute) picks another.

The questions of one kind share everything up to the end of the state, so they fork from it; each kind is a round of its
own, and a tournament's final a further one. adapter_vllm/ runs beside the weights (JEV-35B ships its head merged).
On Gemma 4's layers that read their keys as values (attention_k_eq_v), vLLM loads the keys' weight as the values' and
adapter_vllm/ repeats the layer's k_proj LoRA as v_proj; the checkpoint has no v_proj there and the keys serve as the
values with their LoRA, so those copies are left out (each checked equal to its k_proj LoRA).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import string
import zlib
from pathlib import Path
from typing import Any

from flide import lora
from flide.api.answers import make_choice, make_noul, make_score
from flide.api.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest, SystemOneResponse, Usage
from flide.formats import register
from flide.formats.base import Format, Plan, Read
from flide.formats.next_token import NextToken, TokenRows
from flide.formats.tokens import Pieces

log = logging.getLogger("flide.autotrust")

ADAPTER = "adapter_vllm"
HEAD = "decision_head.json"
CALIBRATION = "calibration.json"
MAX_OPTIONS = 256
MAX_IMAGES = 8
GROUP = 16
PERMUTATIONS = 3
CUT_CHECKS = 8
PROFILES = {"qwen": {"prefix": "", "image": "<|vision_start|><|image_pad|><|vision_end|>", "strategy": "single"},
            "gemma": {"prefix": "<bos>", "image": "<|image|>", "strategy": "tournament"}}
CONTEXT = {"gemma4": 65536, "qwen3_5_moe": 65536}
READS = (Read("decide"),)


def _plain(entry: Any) -> str:
    if entry is None:
        return ""
    return entry if isinstance(entry, str) else json.dumps(entry, ensure_ascii=False)


def _softmax(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def _groups(n: int, k: int = GROUP) -> list[list[int]]:
    g = math.ceil(n / k)
    base, extra = divmod(n, g)
    out, i = [], 0
    for j in range(g):
        size = base + (1 if j < extra else 0)
        out.append(list(range(i, i + size)))
        i += size
    return out


@register("autotrust")
class AutoTrust(Format):
    backbones = ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "gemma4")
    max_score_levels = 6

    processor = NextToken.processor
    _open = staticmethod(NextToken._open)
    _vision = NextToken._vision

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        d = Path(model_dir)
        return (d / ADAPTER / HEAD).is_file() and (d / CALIBRATION).is_file()

    @classmethod
    def run_dir(cls, model_dir: str) -> str:
        """The checkpoint with adapter_vllm/ as its LoRA (lora/), linked once under FLIDE_CACHE; a release with
        its head merged into the weights runs from its own directory."""
        src = Path(model_dir).resolve()
        peft = src / ADAPTER
        if not (peft / lora.CONFIG).is_file():
            return model_dir
        if not (peft / lora.WEIGHTS).is_file():
            raise FileNotFoundError(f"{peft}: {lora.CONFIG} without {lora.WEIGHTS}")
        key = hashlib.sha256(f"{src}\n{ADAPTER}\nk_eq_v".encode()).hexdigest()[:16]
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
            link.symlink_to(f)
        (out / lora.SUB).mkdir(exist_ok=True)
        copies = cls._value_copies(src, peft)
        for name in (lora.CONFIG, lora.WEIGHTS):
            link = out / lora.SUB / name
            if link.is_symlink() or link.exists():
                link.unlink()
            if name == lora.WEIGHTS and copies:
                from safetensors import safe_open
                from safetensors.torch import save_file

                with safe_open(str(peft / name), framework="pt") as h:
                    save_file({k: h.get_tensor(k) for k in h.keys() if k not in copies}, str(link), metadata=h.metadata())
                log.info("%s: %d v_proj LoRA tensors of key-as-value layers left out (copies of k_proj's)", src.name,
                         len(copies))
            else:
                link.symlink_to(peft / name)
        (out / ".done").write_text(f"{src}\n")
        return str(out)

    @staticmethod
    def _value_copies(src: Path, peft: Path) -> set[str]:
        """adapter_vllm/'s v_proj LoRA tensors on the layers that read their keys as values (Gemma 4's
        attention_k_eq_v, its full-attention layers), each equal to its layer's k_proj LoRA, or a ValueError."""
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

    def __init__(self, model_dir: str, tokenizer=None, strategy: str | None = None, calibration: str | None = None,
                 max_branch_tokens: int | None = None, max_request_tokens: int | None = None):
        import transformers

        d = Path(model_dir)
        self.model_dir = model_dir
        self.head = json.loads((d / ADAPTER / HEAD).read_text())
        cal = Path(calibration) if calibration else d / CALIBRATION
        self.temps = json.loads(cal.read_text())["per_kind"] if cal.is_file() else \
            {"noul": 1.0, "score": 1.0, "choice": 1.0}
        self.temps.setdefault("score", 1.0)
        config = json.loads((d / "config.json").read_text())
        model_type = config["model_type"]
        self.prof = PROFILES["gemma" if "gemma" in model_type else "qwen"]
        self.strategy = strategy or self.prof["strategy"]
        if self.strategy not in ("single", "tournament", "permute"):
            raise ValueError(f"autotrust strategy {self.strategy!r}: single, tournament or permute")
        self.vision = config.get("vision_config") is not None
        self.tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.labels, self.label_ids = self._labels()
        lo, hi = self.head["slots"]["ranges"]["choice"]
        if self.label_ids[: hi - lo] != self.head["verbalizer_ids"][lo:hi]:
            raise ValueError(f"{model_dir}: the first option labels are not the decision head's trained choice slots")
        self.native = hi - lo
        self.max_choice_options = len(self.labels)
        context = CONTEXT.get(model_type, 32768)
        self.max_branch_tokens = max_branch_tokens or context - 1
        self.max_request_tokens = max_request_tokens or 2 * context
        ids = list(self.label_ids)
        for kind in ("noul", "score"):
            lo, hi = self.head["slots"]["ranges"][kind]
            ids += [t for t in self.head["verbalizer_ids"][lo:hi] if t not in ids]
        self.ids = ids
        self.row = {t: i for i, t in enumerate(ids)}
        self._pieces = Pieces(self.tokenizer)
        self._cut_checks = 0

    def _labels(self) -> tuple[list[str], list[int]]:
        """A, B, ..., Z, AA, AB, ... that are one token, and that token in "x\\n<label>) y", up to 256."""
        tok = self.tokenizer
        labels, ids = [], []
        for lab in list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]:
            t = tok.encode(lab, add_special_tokens=False)
            if len(t) == 1 and t[0] in tok.encode(f"x\n{lab}) y", add_special_tokens=False):
                labels.append(lab)
                ids.append(t[0])
            if len(labels) == MAX_OPTIONS:
                break
        return labels, ids

    def load_heads(self, backbone) -> dict[str, Any]:
        weight, bias = backbone.output_head()
        return {"decide": TokenRows(weight, self.ids, bias, softcap=getattr(backbone, "logit_softcap", None))}


    def _question(self, q) -> dict:
        ins = _plain(q.instructions)
        if isinstance(q, NoulQuestion):
            return {"kind": "noul", "question": ins, "options": ["false", "true"]}
        if isinstance(q, ScoreQuestion):
            if len(q.criteria) != 6:
                raise ValueError(f"score question has {len(q.criteria)} levels; this model reads six (0 to 5)")
            return {"kind": "score", "question": ins, "options": [str(i) for i in range(6)]}
        if isinstance(q, ChoiceQuestion):
            names = [str(n) for n in q.criteria]
            if not 2 <= len(names) <= len(self.labels):
                raise ValueError(f"choice needs 2-{len(self.labels)} options, got {len(names)}")
            opts = [_plain(v) or n for n, v in zip(names, q.criteria.values())]
            return {"kind": "choice", "question": ins, "options": opts, "names": names}
        raise TypeError(f"unknown question type {type(q).__name__}")

    def _parts(self, state: Any, refs: list) -> list[tuple[str, Any]]:
        parts: list[tuple[str, Any]] = [("image", r) for r in refs]
        if state is None or isinstance(state, str):
            parts.append(("text", state or ""))
        elif isinstance(state, dict):
            parts.append(("text", json.dumps(state, ensure_ascii=False)))
        elif isinstance(state, list):
            for p in state:
                if isinstance(p, str):
                    parts.append(("text", p))
                elif isinstance(p, dict) and "image" in p:
                    parts.append(("image", p["image"]))
                elif isinstance(p, dict) and p.get("type") == "image_url":
                    url = p.get("image_url")
                    parts.append(("image", url.get("url") if isinstance(url, dict) else url))
                elif isinstance(p, dict) and p.get("type") == "text":
                    parts.append(("text", _plain(p.get("text"))))
                else:
                    parts.append(("text", json.dumps(p, ensure_ascii=False)))
        else:
            raise ValueError("state: a string, an object or a list")
        images = sum(1 for k, _ in parts if k == "image")
        if images and not self.vision:
            raise ValueError("images: this checkpoint reads text only")
        if images > MAX_IMAGES:
            raise ValueError(f"{images} images; at most {MAX_IMAGES} per request")
        return parts

    def _passes(self, d: dict, kind: str) -> list[tuple]:
        """The first round of a kind: (question id, step, its option indices) for every read."""
        out = []
        for qid, q in d["qs"].items():
            if q["kind"] != kind:
                continue
            n = len(q["options"])
            if kind != "choice" or n <= self.native or self.strategy == "single":
                out.append((qid, "one", list(range(n))))
            elif self.strategy == "permute":
                rng = random.Random(zlib.crc32(q["question"].encode()))
                orders = [list(range(n))] + [rng.sample(range(n), n) for _ in range(PERMUTATIONS)]
                d["acc"][qid] = {"p": [0.0] * n, "left": len(orders)}
                out += [(qid, "permute", o) for o in orders]
            else:
                groups = _groups(n)
                d["acc"][qid] = {"groups": groups, "in_group": {}, "left": len(groups)}
                out += [(qid, "group", g) for g in groups]
        return out

    def _lines(self, kind: str, q: dict, sub: list[int]) -> tuple[list[str], list[int], list[float]]:
        lo, hi = self.head["slots"]["ranges"][kind]
        if kind == "choice":
            bias = [self.head["bias"][lo + i] if lo + i < hi else 0.0 for i in range(len(sub))]
            lines = [f"{self.labels[i]}) {q['options'][j]}" for i, j in enumerate(sub)]
            return lines, self.label_ids[: len(sub)], bias
        return list(q["options"]), self.head["verbalizer_ids"][lo:hi], self.head["bias"][lo:hi]

    def _round(self, d: dict, kind: str, stage: str, passes: list[tuple]) -> Plan:
        refs = [v for k, v in d["parts"] if k == "image"]
        shared = self.prof["prefix"] + f"[kind] {kind}\n[state] " + \
            "".join(self.prof["image"] if k == "image" else v for k, v in d["parts"])
        reads = []
        texts = []
        for qid, _, sub in passes:
            q = d["qs"][qid]
            lines, ids, bias = self._lines(kind, q, sub)
            reads.append((ids, bias))
            texts.append(shared + f"\n[question] {q['question']}\n[options]\n" + "\n".join(lines) + "\n[decision]:")
        c = self._pieces.last_cut(texts[0], len(shared) - 1) if shared else 0
        head, prefix_tail, *tails = self._pieces.encode(self.tokenizer, [shared[:c], shared[c:]] + [t[c:] for t in texts])
        if c and self._cut_checks < CUT_CHECKS:
            self._cut_checks += 1
            if head + tails[0] != self.tokenizer.backend_tokenizer.encode(texts[0], add_special_tokens=False).ids:
                self._pieces.ok = False
                log.warning("tokenizing in pieces off: a cut changed the tokens of a prompt")
                head, prefix_tail, *tails = self._pieces.encode(self.tokenizer, ["", shared] + texts)
        m = len(prefix_tail)
        while m > 0 and not all(t[:m] == prefix_tail[:m] for t in tails):
            m -= 1
        prefix, suffixes = head + tails[0][:m], [t[m:] for t in tails]
        if any(len(s) == 0 for s in suffixes):
            raise ValueError("a question renders to nothing after the shared state")
        vision = None
        if refs:
            prefix, vision = self._vision(prefix, [self._open(r) for r in refs], texts[0], suffixes[0])
        longest = len(prefix) + max(len(s) for s in suffixes)
        total = len(prefix) + sum(len(s) for s in suffixes)
        if longest > self.max_branch_tokens:
            raise ValueError(f"a prompt is {longest} tokens, limit is {self.max_branch_tokens}")
        if total > self.max_request_tokens:
            raise ValueError(f"request is {total} tokens, limit is {self.max_request_tokens}")
        data = {"d": d, "kind": kind, "stage": stage, "passes": passes, "reads": reads, "absorbed": False}
        return Plan(prefix, suffixes, READS, data, vision=vision)

    def plan(self, request: SystemOneRequest, **_) -> Plan:
        qs = {qid: self._question(q) for qid, q in request.questions.items()}
        d = {"qs": qs, "parts": self._parts(request.state, getattr(request, "images", None) or []),
             "kinds": list(dict.fromkeys(q["kind"] for q in qs.values())), "probs": {}, "acc": {}, "tokens": 0,
             "branches": 0, "rounds": 0}
        kind = d["kinds"][0]
        return self._round(d, kind, "first", self._passes(d, kind))

    def _absorb(self, plan: Plan, outputs) -> dict:
        data = plan.data
        d = data["d"]
        if data["absorbed"]:
            return d
        data["absorbed"] = True
        d["tokens"] += plan.tokens
        d["branches"] += len(plan.suffixes)
        d["rounds"] += 1
        T = self.temps[data["kind"]]
        for (qid, step, sub), (ids, bias), out in zip(data["passes"], data["reads"], outputs):
            row = out[0][0]
            p = _softmax([(row[self.row[t]] + b) / T for t, b in zip(ids, bias)])
            if step in ("one", "final"):
                if step == "final":
                    p = self._final(d["acc"][qid], sub, p, len(d["qs"][qid]["options"]))
                d["probs"][qid] = p
            elif step == "permute":
                acc = d["acc"][qid]
                for i, v in zip(sub, p):
                    acc["p"][i] += v / (PERMUTATIONS + 1)
                acc["left"] -= 1
                if not acc["left"]:
                    d["probs"][qid] = acc["p"]
            else:
                acc = d["acc"][qid]
                acc["in_group"].update(zip(sub, p))
                acc["left"] -= 1
        return d

    @staticmethod
    def _final(acc: dict, fin: list[int], p: list[float], n: int) -> list[float]:
        groups, in_group = acc["groups"], acc["in_group"]
        final = dict(zip(fin, p))
        group_of = {o: gi for gi, g in enumerate(groups) for o in g}
        share, cap = [0.0] * len(groups), [0.0] * len(groups)
        for f in fin:
            share[group_of[f]] += final[f]
            cap[group_of[f]] += in_group[f]
        among = sum(a * b for a, b in zip(share, cap))
        out = [final[o] * among if o in final else share[group_of[o]] * in_group[o] for o in range(n)]
        s = sum(out)
        return [x / s for x in out]

    def follow_up(self, request: SystemOneRequest, plan: Plan, outputs, **_) -> Plan | None:
        d = self._absorb(plan, outputs)
        kind = plan.data["kind"]
        if plan.data["stage"] == "first":
            finals = []
            for qid, acc in d["acc"].items():
                if d["qs"][qid]["kind"] == kind and "groups" in acc and qid not in d["probs"]:
                    groups, in_group = acc["groups"], acc["in_group"]
                    chosen = [max(g, key=lambda o: (in_group[o], -o)) for g in groups]
                    taken = set(chosen)
                    rest = sorted((o for g in groups for o in g if o not in taken), key=lambda o: (-in_group[o], o))
                    finals.append((qid, "final", sorted(chosen + rest[: max(0, GROUP - len(chosen))])))
            if finals:
                return self._round(d, kind, "final", finals)
        i = d["kinds"].index(kind) + 1
        if i < len(d["kinds"]):
            return self._round(d, d["kinds"][i], "first", self._passes(d, d["kinds"][i]))
        return None

    def finish(self, request: SystemOneRequest, plan: Plan, outputs, **_):
        d = self._absorb(plan, outputs)
        answers = {}
        for qid, q in d["qs"].items():
            p = d["probs"][qid]
            if q["kind"] == "noul":
                answers[qid] = make_noul(p[1])
            elif q["kind"] == "choice":
                answers[qid] = make_choice(q["names"], p)
            else:
                answers[qid] = make_score(list(request.questions[qid].criteria), p)
        usage = Usage(input_tokens=d["tokens"], output_tokens=len(request.questions))
        return SystemOneResponse(model="", answers=answers, usage=usage), {
            "format": "autotrust", "prefix_tokens": len(plan.prefix), "branches": d["branches"], "rounds": d["rounds"]}

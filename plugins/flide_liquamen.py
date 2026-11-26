"""Liquamen family (plantwaterer/liquamen, a full fine-tune of Gemma 4 31B): the release's board engine
(lenavg_bayes_balanced_k5_engine and the engines under it) with its Latent01 head (vbll_head.safetensors, schema
jev-l01post-head-v1), which the checkpoint ships.

Every question is one user turn in the chat template, thinking off: the state, "Question:" with the instructions, the
options lettered A..Z then a..z, and a closing line, tokenized whole. A choice shows its options in a non-identity order
drawn from a hash of the state, the question key and the option keys; a yes/no question shows Yes, then No, each with
its definition when the criteria give one. Each prompt is a sequence of its own, read both ways whole in the global
layers and causally in the sliding-window layers (attention "blocks", nothing shared), as the release's attention
policy B reads it.

The read is the final-norm state at the last token: the soft-capped logits of the 52 letters and the state itself. The
head runs on the CPU in float64 (garas_vbll_engine, vbll_core): the letter logits centred over the options shown, on a
yes/no question shifted by the head's mean and widened by its posterior variance along the letter-orthogonal state,
then a 16-node Gauss-Hermite integral at the head's temperature.

Averaging, for a short request (the release's 939 characters, or one question of at most 6,000): a choice is read again
in three more orders (fixed-seed permutations up to 4 options, cyclic shifts by round(j K / 4) from 5 to 52 options)
and the four head outputs are averaged at T_HEAD; a yes/no question is read again with No first, its native letter
probabilities averaged at T_NOUL and moved by the head's log-odds correction on the averaged logits. A choice of more
than 52 options is split into chunks by a hash of the request and its best options asked again in a second round.
Score questions, choices of one option and prompts that do not fit 32,768 tokens with the whole state are refused.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import string
from pathlib import Path
from typing import Any

from flide.api.answers import make_choice, make_noul
from flide.api.schema import SystemOneRequest, SystemOneResponse, Usage
from flide.formats import register
from flide.formats.base import Format, Plan, Read
from flide.formats.next_token import TokenRows
from flide.formats.tokens import Pieces

SCHEMA = "jev-l01post-head-v1"
HEAD_FILE, HEAD_META = "vbll_head.safetensors", "vbll_head.json"
LABELS = string.ascii_uppercase + string.ascii_lowercase
MAX_K = 52
PROMPT = ("{state}\n\nQuestion: {instructions}\n\nOptions:\n{options}\n\n"
          "Answer with the letter of the single best option and nothing else.")
MAX_LEN, MARGIN = 32768, 16
L_RAW, L_ONE = 939, 6000
SEED, K_AVG = 20261021, 4
T_HEAD, T_NOUL = 0.95438, 0.93738
K52_EPS = 0.001
KIND = {"choice": 0, "noul": 1, "score": 2}
HEAD = "liquamen"
READ = Read(HEAD)


def _dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)


def _text(x: Any) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else _dumps(x)


def _state(state: Any) -> str:
    return state if isinstance(state, str) else _dumps(state)


def _length(state: Any, questions: dict) -> int:
    """The release's request_length: per question, the state, the instructions and every option's key and value, in
    characters of the values as sent."""
    st = state or ""
    tot = 0
    for q in questions.values():
        tot += len(st) + len(q.get("instructions") or "") + sum(len(str(k)) + len(str(v))
                                                                for k, v in (q.get("criteria") or {}).items())
    return tot


def _request_id(state: Any, questions: dict) -> str:
    return "eng-" + hashlib.sha1(json.dumps({"state": state, "questions": questions}, sort_keys=True,
                                            ensure_ascii=False, default=str).encode("utf-8")).hexdigest()[:16]


def _seed(state_text: str, key: str, keys: list[str]) -> int:
    return int(hashlib.sha256(_dumps([state_text, key, list(keys)]).encode("utf-8")).hexdigest()[:16], 16)


def _base_order(k: int, seed: int) -> list[int]:
    """A choice's order: the first permutation of a generator seeded by the question that is not the identity."""
    import numpy as np

    if k < 2:
        return list(range(k))
    rng = np.random.default_rng(seed)
    while True:
        o = [int(x) for x in rng.permutation(k)]
        if o != list(range(k)):
            return o


def _reletterings(k: int) -> list[list[int]]:
    """The three further letterings of a choice: fixed-seed permutations up to 4 options (repeats kept), cyclic shifts
    by round(j K / 4) from 5 options."""
    import numpy as np

    if k <= 4:
        return [[int(x) for x in np.random.default_rng(SEED + 1000 * j + k).permutation(k)] for j in range(1, K_AVG)]
    used = []
    for s in (int(round(j * k / K_AVG)) % k for j in range(K_AVG)):
        while s in used:
            s = (s + 1) % k
        used.append(s)
    return [[(i + s) % k for i in range(k)] for s in used[1:]]


def _chunks(run_id: str, key: str, keys: list[str]) -> list[list[str]]:
    c = math.ceil(len(keys) / MAX_K)
    order = list(keys)
    random.Random(int(hashlib.sha256(f"di-k52:{run_id}|{key}".encode()).hexdigest()[:16], 16)).shuffle(order)
    pos = {k: i for i, k in enumerate(keys)}
    return [sorted(order[j::c], key=pos.__getitem__) for j in range(c)]


def _noul_definitions(crit: Any) -> dict | None:
    if not isinstance(crit, dict) or not crit:
        return None
    low = {str(k).strip().lower(): v for k, v in crit.items()}

    def side(names):
        return next((str(low[k]).strip() for k in names if low.get(k) is not None and str(low[k]).strip()), "")
    d = {"yes": side(("yes", "true")), "no": side(("no", "false"))}
    return d if d["yes"] or d["no"] else None


def _combine(prob_lists, t: float):
    import numpy as np

    p = np.asarray(prob_lists, dtype=np.float64).mean(0)
    q = np.where(p > 0, np.power(np.maximum(p, 1e-300), 1.0 / t), 0.0)
    return q / q.sum()


def _logodds(p0: float, p1: float) -> float:
    return math.log(max(p0, 1e-300)) - math.log(max(p1, 1e-300))


def _norm(d: dict) -> dict:
    z = sum(d.values())
    if not z > 0 or not math.isfinite(z):
        return {k: 1.0 / len(d) for k in d}
    return {k: v / z for k, v in d.items()}


# ---- the head (vbll_core's lloc_hperp, l01_extract, l01_probs and l01_ovr, float64 on the CPU)

def _gauss_hermite():
    import numpy as np
    import torch

    x, w = np.polynomial.hermite.hermgauss(16)
    return torch.from_numpy(x * math.sqrt(2.0)).double(), torch.from_numpy(w / math.sqrt(math.pi)).double()


def _hperp(h, ve):
    hp = h - (h @ ve) @ ve.T
    return hp * (hp.pow(2).mean(-1, keepdim=True) + 1e-12).rsqrt()


def _extract(h, ell, ks, kinds, st, c):
    """l01_extract; the letter rows' term only when alpha weighs it and the residual norm only when sig does (both are
    multiplied by zero otherwise), the noul terms for the noul rows."""
    import torch

    K = ell.shape[1]
    mask = torch.arange(K)[None] < ks[:, None]
    m = mask.double()
    centre = lambda x: (x - (x * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)) * m
    ell_c = centre(ell)
    n = int(ks.shape[0])
    d = {"ell_c": ell_c, "mask": mask, "kinds": kinds, "k": ks,
         "row_rms": ((ell_c.pow(2) * m).sum(1) / m.sum(1).clamp_min(1.0) + 1.0).sqrt(),
         "dm_noul": torch.zeros(n, dtype=torch.float64), "var_noul": torch.zeros(n, dtype=torch.float64),
         "res": torch.zeros(n, dtype=torch.float64), "dz_c": torch.zeros_like(ell_c)}
    if c["alpha"] != 0.0:
        d["dz_c"] = centre((h @ st["E"].T)[:, :K]) - ell_c
    if c["sig"] > 0.0:
        hp = _hperp(h, st["V_E"]) - st["mu_hp"]
        u = hp @ st["V_perp"]
        d["res"] = (hp - u @ st["V_perp"].T).pow(2).sum(-1).sqrt()
    idx = (kinds == KIND["noul"]).nonzero().flatten()
    if len(idx):
        un = _hperp(h[idx], st["V_E"]) @ st["V_perp"]
        d["dm_noul"][idx] = (un @ st["mu"]) / 2.0
        d["var_noul"][idx] = ((un @ st["S"]) * un).sum(-1).clamp_min(0.0) / 4.0
    return d


def _ovr(mu_scaled, sigma, m, gz, gw):
    import torch

    z = mu_scaled.masked_fill(~m, -1e9)
    lse = torch.logsumexp(z, dim=-1, keepdim=True)
    p0 = torch.softmax(z, dim=-1).masked_fill(~m, 0.0)
    margin = (z - lse) - torch.log1p(-p0.clamp_max(1.0 - 1e-12))
    kb = m.double().sum(dim=-1, keepdim=True).clamp_min(2.0)
    sig_eff = sigma * (2.0 / kb).sqrt() * m.double()
    f = torch.sigmoid(margin[None] + sig_eff[None] * gz[:, None, None]).masked_fill(~m[None], 0.0)
    i_k = torch.einsum("q,qbk->bk", gw, f).masked_fill(~m, 0.0)
    return i_k / i_k.sum(dim=-1, keepdim=True).clamp_min(1e-15)


def _probs(d, st, t, c, gh):
    """l01_probs: [n, K] float64 (the score-interval step never runs: score questions are refused)."""
    import torch

    m, kinds = d["mask"], d["kinds"]
    mu = (d["ell_c"] + c["alpha"] * d["dz_c"]).clone()
    is_noul = kinds == KIND["noul"]
    if c["beta"] > 0.0 and bool(is_noul.any()):
        dm = c["beta"] * d["dm_noul"][is_noul]
        mu[is_noul, 0] += dm
        mu[is_noul, 1] -= dm
    if c["tau"] > 0.0:
        r = torch.ones(len(mu), dtype=torch.float64)
        for kd in ("noul", "choice", "score"):
            r[kinds == KIND[kd]] = 1.0 + c["tau"] * (st["rel_s"][kd] - 1.0)
        t_row = t * r[:, None]
    else:
        t_row = t
    if c["sig"] <= 0.0 and not (c["kern"] > 0.0 or c["lam_uni"] > 0.0) and c["tau"] <= 0.0:
        return torch.softmax((mu / t).masked_fill(~m, -1e9), -1).masked_fill(~m, 0.0)
    rr = d["res"] / st["res_mean"] if c["dnorm"] == "ratio" else torch.ones_like(d["res"])
    sigma = (c["sig"] * rr * d["row_rms"])[:, None].expand_as(mu).clone()
    if bool(is_noul.any()):
        sigma[is_noul, :2] = (sigma[is_noul, :2].pow(2) + c["infl"] * d["var_noul"][is_noul, None]).sqrt()
    return _ovr(mu / t_row, (sigma / t_row) * m.double(), m, *gh)


class _Letters:
    """The 52 letters' soft-capped logits in float32 and the final-norm state, side by side, kept on the device."""

    on_device = True

    def __init__(self, rows: TokenRows):
        self.rows = rows

    def __call__(self, hidden):
        import torch

        return torch.cat([self.rows(hidden), hidden.float()], -1)


@register("liquamen")
class Liquamen(Format):
    backbones = ("gemma4",)
    max_choice_options = 256
    backbone_options = {"vision": False}

    def __init__(self, model_dir: str, tokenizer=None):
        import torch
        import transformers
        from safetensors.torch import load_file

        self.model_dir = model_dir
        root = Path(model_dir)
        meta = json.loads((root / HEAD_META).read_text())
        if meta.get("schema") != SCHEMA:
            raise ValueError(f"{model_dir}: {HEAD_META} has schema {meta.get('schema')!r}, this family reads {SCHEMA}")
        policy = (json.loads((root / "config.json").read_text()).get("text_config") or {}).get("jev_attention_policy")
        if policy != "B":
            raise ValueError(f"{model_dir}: attention policy {policy!r}, this family reads policy B")
        raw = (root / HEAD_FILE).read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta["tensor_sha256"]:
            raise ValueError(f"{model_dir}: {HEAD_FILE} differs from the sha256 in {HEAD_META}")
        self.tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self._pieces = Pieces(self.tokenizer)
        ids = []
        for ch in LABELS:
            t = self._enc(ch)
            if len(t) != 1:
                raise ValueError(f"{model_dir}: the letter {ch!r} is not one token")
            ids.append(t[0])
        if ids != list(meta["label_ids"]):
            raise ValueError(f"{model_dir}: the letters' token ids differ from {HEAD_META}'s label_ids")
        self.label_ids = ids
        t = load_file(str(root / HEAD_FILE))
        self.head = {k: t[k].double() for k in ("E", "V_E", "V_perp", "mu", "S")}
        self.head["mu_hp"] = t["mu_hp"].double().reshape(1, -1)
        self.head["rel_s"] = {k: float(v) for k, v in meta["rel_s"].items()}
        self.head["res_mean"] = float(meta["res_mean"])
        self.c = dict(meta["constants"])
        self.temperature = float(meta.get("temperature", 1.0))
        self.gh = _gauss_hermite()
        torch.set_num_threads(max(1, torch.get_num_threads()))

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        path = Path(model_dir) / HEAD_META
        if not path.is_file():
            return False
        try:
            return json.loads(path.read_text()).get("schema") == SCHEMA
        except ValueError:
            return False

    def _enc(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _render(self, state: str, instructions: str, options: str) -> str:
        msg = PROMPT.format(state=state, instructions=instructions, options=options)
        return self.tokenizer.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)

    def token_heads(self) -> dict[str, list[int]]:
        return {HEAD: list(self.label_ids)}

    def load_heads(self, backbone) -> dict[str, Any]:
        weight, bias = backbone.output_head()
        return {HEAD: _Letters(TokenRows(weight, self.label_ids, bias, softcap=getattr(backbone, "logit_softcap", None)))}

    # ---- prompts (garas_core.build_prompt, garas_engine.prompt, oavg_engine.prompt)

    def _build(self, st: str, n_state, unit: dict, order: list[int]) -> str | None:
        """The prompt in this order; the state cut to the tokens left when it does not fit (the caller refuses that)."""
        opts = "\n".join(f"{LABELS[j]}) {unit['texts'][o]}" for j, o in enumerate(order))
        budget = MAX_LEN - len(self._enc(self._render("", unit["instructions"], opts))) - MARGIN
        if budget <= 0:
            return None
        if n_state() > budget:
            off = self.tokenizer(st, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            st = st[: int(off[budget - 1][1])]
        return self._render(st, unit["instructions"], opts)

    def _prompts(self, st: str, n_state, unit: dict, averaged: bool) -> list[tuple[list[int], str]]:
        k = len(unit["keys"])
        base = _base_order(k, _seed(st, unit["hkey"], unit["keys"])) if unit["type"] == "choice" else list(range(k))
        orders = [base]
        if averaged and not unit["k52"]:
            rs = [[1, 0]] if unit["type"] == "noul" else _reletterings(k)
            orders += [[base[j] for j in r] for r in rs]
        out = []
        for i, order in enumerate(orders):
            text = self._build(st, n_state, unit, order)
            if text is None or (i == 0 and st.lstrip() not in text):
                raise ValueError(f"maximum context length: question {unit['hkey']!r} does not fit {MAX_LEN} tokens "
                                 f"without cutting the state")
            out.append((order, text))
        return out

    def _unit(self, key: str, hkey: str, kind: str, keys: list[str], texts: list[str], instructions: str,
              k52: bool = False) -> dict:
        return {"key": key, "hkey": hkey, "type": kind, "keys": keys, "texts": texts, "instructions": instructions,
                "k52": k52}

    def _plan(self, st: str, n_state, units: list[dict], averaged: bool, data: dict) -> Plan:
        texts, prompts = [], []
        for ui, unit in enumerate(units):
            for order, text in self._prompts(st, n_state, unit, averaged):
                prompts.append((ui, order))
                texts.append(text)
        head, tails = self._pieces.rows(self.tokenizer, texts)   # the state is tokenized once
        return Plan([], [head + t for t in tails], (READ,), {**data, "units": units, "prompts": prompts}, attention="blocks")

    def plan(self, request: SystemOneRequest, **_) -> Plan:
        if getattr(request, "images", None):
            raise ValueError("images: the liquamen family reads text only")
        raw = {key: q.model_dump(exclude_unset=True) for key, q in request.questions.items()}
        try:
            l_raw, l_ren = _length(request.state, raw), _length(_state(request.state), raw)
        except (TypeError, AttributeError) as e:
            raise ValueError(f"request length: {e}") from None
        for key, q in raw.items():
            if q.get("type") not in ("choice", "noul"):
                raise ValueError(f"question type {q.get('type')!r}: the liquamen family answers choice and noul")
            if q["type"] == "choice" and len(q.get("criteria") or {}) < 2:
                raise ValueError("a choice needs at least two options")
            if "|" in key:
                raise ValueError(f"question key with '|': {key!r}")
        st = _state(request.state)
        averaged = l_raw <= L_RAW or (len(raw) == 1 and l_ren <= L_ONE)
        counted = []

        def n_state():
            if not counted:
                counted.append(len(self._enc(st)))
            return counted[0]

        units, k52, rid = [], {}, None
        for key, q in raw.items():
            instructions = _text(q.get("instructions"))
            if q["type"] == "noul":
                d = _noul_definitions(q.get("criteria"))
                texts = [f"{t}: {d[k]}" if d and d.get(k) else t for t, k in (("Yes", "yes"), ("No", "no"))]
                units.append(self._unit(key, key, "noul", ["yes", "no"], texts, instructions))
                continue
            crit = {str(k): _text(v) for k, v in q["criteria"].items()}
            keys = list(crit)
            if len(keys) <= MAX_K:
                units.append(self._unit(key, key, "choice", keys, [crit[k] or k for k in keys], instructions))
                continue
            rid = rid or _request_id(request.state, raw)
            parts = _chunks(rid, key, keys)
            k52[key] = {"keys": keys, "texts": crit, "per": MAX_K // len(parts), "units": []}
            for j, ck in enumerate(parts):
                k52[key]["units"].append(len(units))
                units.append(self._unit(key, f"{key}|c{j}of{len(parts)}", "choice", ck, [crit[k] or k for k in ck],
                                        instructions, k52=True))
        return self._plan(st, n_state, units, averaged, {"round": 1, "averaged": averaged, "k52": k52,
                                                        "state": st, "tokens": 0})

    # ---- the readout (garas_vbll_engine.run_prompts, lenavg_bayes_engine.run_prompts)

    def _read(self, plan: Plan, outputs) -> list[list[float]]:
        """Every unit's probabilities over its options, in their order."""
        import numpy as np
        import torch

        d = plan.data
        y = torch.stack([out[0][0] for out in outputs]).double().cpu()
        ell, h = y[:, :MAX_K], y[:, MAX_K:]
        ks = torch.tensor([len(o) for _, o in d["prompts"]])
        kinds = torch.tensor([KIND[d["units"][ui]["type"]] for ui, _ in d["prompts"]])
        p = _probs(_extract(h, ell, ks, kinds, self.head, self.c), self.head, self.temperature, self.c, self.gh)
        native = torch.softmax(ell.masked_fill(torch.arange(MAX_K)[None] >= ks[:, None], float("-inf")), -1)
        p, native = p.numpy(), native.numpy()
        rows: list[list[tuple]] = [[] for _ in d["units"]]
        for b, (ui, order) in enumerate(d["prompts"]):
            k = len(order)
            pb = p[b, :k] / p[b, :k].sum()
            q, nq = np.zeros(k), np.zeros(k)
            q[np.asarray(order)] = pb
            nq[np.asarray(order)] = native[b, :k]
            rows[ui].append((b, order, q, nq))
        out = []
        for unit, rs in zip(d["units"], rows):
            if len(rs) == 1:
                out.append(rs[0][2].tolist())
            elif unit["type"] == "choice":
                out.append(_combine([r[2] for r in rs], T_HEAD).tolist())
            else:
                out.append(self._noul_averaged(h, ell, rs))
        return out

    def _noul_averaged(self, h, ell, rs) -> list[float]:
        """The release's V4: the native probabilities of both orders averaged at T_NOUL, plus the head's log-odds
        correction (shift and variance on, against both off) on the base prompt's state and the averaged logits."""
        import numpy as np
        import torch

        (b, order, _, _), nats = rs[0], [r[3] for r in rs]
        p_len = _combine(nats, T_NOUL)
        avg = np.asarray(nats, dtype=np.float64).mean(0)
        letters = torch.tensor([[math.log(max(avg[order[0]], 1e-300)), math.log(max(avg[order[1]], 1e-300))]],
                               dtype=torch.float32).double()
        ks, kinds = torch.tensor([2]), torch.tensor([KIND["noul"]])
        off = dict(self.c, beta=0.0, infl=0.0)
        p_on = _probs(_extract(h[b:b + 1], letters, ks, kinds, self.head, self.c), self.head, self.temperature,
                      self.c, self.gh)[0].tolist()
        p_off = _probs(_extract(h[b:b + 1], letters, ks, kinds, self.head, off), self.head, self.temperature, off,
                       self.gh)[0].tolist()
        lo = _logodds(p_len[order[0]], p_len[order[1]]) + _logodds(p_on[0], p_on[1]) - _logodds(p_off[0], p_off[1])
        pa = 1.0 / (1.0 + math.exp(-lo)) if lo > -700 else 0.0
        p = [0.0, 0.0]
        p[order[0]], p[order[1]] = pa, 1.0 - pa
        return p

    def follow_up(self, request: SystemOneRequest, plan: Plan, outputs, **_) -> Plan | None:
        d = plan.data
        if d["round"] != 1 or not d["k52"]:
            return None
        first = self._read(plan, outputs)
        finals = []
        for key, info in d["k52"].items():
            chosen = set()
            for ui in info["units"]:
                ks, p = d["units"][ui]["keys"], first[ui]
                chosen.update(ks[i] for i in sorted(range(len(ks)), key=lambda i: (-p[i], i))[: info["per"]])
            fin = [k for k in info["keys"] if k in chosen]
            u0 = d["units"][info["units"][0]]
            finals.append(self._unit(key, f"{key}|final", "choice", fin, [info["texts"][k] or k for k in fin],
                                     u0["instructions"], k52=True))
        counted = []
        st = d["state"]

        def n_state():
            if not counted:
                counted.append(len(self._enc(st)))
            return counted[0]

        return self._plan(st, n_state, finals, d["averaged"],
                          {"round": 2, "averaged": d["averaged"], "k52": d["k52"], "state": st,
                           "tokens": plan.tokens, "first": first, "first_units": d["units"]})

    def finish(self, request: SystemOneRequest, plan: Plan, outputs, **_):
        d = plan.data
        probs = self._read(plan, outputs)
        units = d["units"] if d["round"] == 1 else d["first_units"]
        first = probs if d["round"] == 1 else d["first"]
        final = {} if d["round"] == 1 else {u["key"]: (u["keys"], p) for u, p in zip(d["units"], probs)}
        by_key = {u["key"]: p for u, p in zip(units, first) if not u["k52"]}
        answers = {}
        for key, q in request.questions.items():
            if key in d["k52"]:
                info = d["k52"][key]
                p1 = {}
                for ui in info["units"]:
                    p1.update({k: v / len(info["units"])
                               for k, v in _norm(dict(zip(units[ui]["keys"], first[ui]))).items()})
                if key in final:
                    p2 = _norm(dict(zip(*final[key])))
                    rest = {k: v for k, v in p1.items() if k not in p2}
                    zr = sum(rest.values()) or 1.0
                    dist = {k: (1 - K52_EPS) * p2[k] if k in p2 else K52_EPS * rest[k] / zr for k in info["keys"]}
                else:
                    dist = {k: p1[k] for k in info["keys"]}
                dist = _norm(dist)
            else:
                keys = ["yes", "no"] if q.type == "noul" else [str(k) for k in q.criteria]
                dist = _norm(dict(zip(keys, by_key[key])))
            if q.type == "noul":
                answers[key] = make_noul(dist["yes"])
            else:
                answers[key] = make_choice(list(dist), list(dist.values()))
        usage = Usage(input_tokens=d["tokens"] + plan.tokens, output_tokens=len(request.questions))
        return SystemOneResponse(model="", answers=answers, usage=usage), {"format": "liquamen",
                                                                          "prompts": len(d["prompts"]),
                                                                          "averaged": d["averaged"],
                                                                          "rounds": d["round"]}

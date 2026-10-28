"""`flide verify`: the guarantees of exact state forking checked on your model, your requests and this GPU.

fork     every question of every request against the same question asked alone (the state and that question only),
         each in a round of its own: the largest gap between their probabilities, the top answers that agree, and the
         questions whose head outputs are the same bits
load     every request alone against the same requests shuffled into rounds of 2, 8 and 32, as a loaded server
         batches them: the requests whose head outputs keep their bits, and the largest gap
session  (with --session-cache-gb) each state grown in four steps, every step answered with the session cache on,
         resuming from the step before, and off: the tokens resumed, the largest gap, the top answers that agree

A top answer that moves counts against a check only where the reference's top two options are at least 0.05 apart.
With --batch-invariant, any bit that moves in load fails it too: that mode promises the same bits at any load.

    flide verify --model M --requests requests.jsonl [--limit 32] [--batch-invariant] [--session-cache-gb 4]
"""
from __future__ import annotations

import json
import random
from pathlib import Path

TIE = 0.05
ROUNDS = (2, 8, 32)
STEPS = 4


def _requests(path: str, limit: int) -> list[dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.append(r.get("payload", r))
        if len(out) == limit:
            break
    return out


def _same(a, b) -> bool:
    """Head outputs bit for bit: nested lists of floats, or tensors for a head that keeps them on the device."""
    if hasattr(a, "shape") or hasattr(b, "shape"):
        import torch

        return torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _probs(answer) -> list[float]:
    a = answer.model_dump()
    return [1 - a["noul"], a["noul"]] if a["type"] == "noul" else list(a["probabilities"].values())


def _gap(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def _moved(ref: list[float], other: list[float]) -> bool:
    """The top answer moved where the reference's top two options are at least TIE apart."""
    s = sorted(ref, reverse=True)
    top = lambda p: max(range(len(p)), key=p.__getitem__)
    return len(ref) > 1 and s[0] - s[1] >= TIE and top(ref) != top(other)


class _Check:
    def __init__(self, name: str):
        self.name, self.n, self.same_bits, self.gap, self.moved = name, 0, 0, 0.0, 0

    def add(self, ref: list[float], other: list[float], bits: bool) -> None:
        self.n += 1
        self.same_bits += bits
        self.gap = max(self.gap, _gap(ref, other))
        self.moved += _moved(ref, other)


def _answers(eng, request, plan, outputs) -> dict:
    response, _ = eng.format.finish(request, plan, outputs)
    return response.answers


def run(args) -> None:
    from . import formats
    from .api.schema import SystemOneRequest
    from .api.serve import _family_options
    from .runtime.engine import Engine, model_config, resolve_model
    from .runtime.offline import parse_kv

    model_dir = resolve_model(args.model)
    family = formats.detect(model_dir, model_config(model_dir), args.format)
    eng = Engine(model_dir, format=family.name, device=args.device, backbone_options={"cuda_graphs": True},
                 session_cache_gb=args.session_cache_gb, batch_invariant=args.batch_invariant,
                 **_family_options(family, parse_kv(args.format_arg)))
    reqs, plans = [], []
    for r in _requests(args.requests, args.limit):
        req = SystemOneRequest.model_validate(r)
        try:
            plans.append(eng.format.plan(req))
        except ValueError:
            continue
        reqs.append(req)
    if not reqs:
        raise SystemExit("no request the model takes")
    invariant = args.batch_invariant
    print(f"{family.name} on {model_dir}: {len(reqs)} requests, {sum(len(r.questions) for r in reqs)} questions, "
          f"batch-invariant {'on' if invariant else 'off'}", flush=True)

    alone = [eng.forward([p], "batch")[0][0] for p in plans]
    lines, failed = [], False

    fork = _Check("fork")
    for req, plan, outs in zip(reqs, plans, alone):
        if len(plan.suffixes) != len(req.questions):
            continue
        full = _answers(eng, req, plan, outs)
        for b, (name, q) in enumerate(req.questions.items()):
            one = req.model_copy(update={"questions": {name: q}})
            p1 = eng.format.plan(one)
            o1 = eng.forward([p1], "batch")[0][0]
            fork.add(_probs(_answers(eng, one, p1, o1)[name]), _probs(full[name]), _same(o1[0], outs[b]))
    if fork.n:
        bad = fork.moved
        failed |= bool(bad)
        lines.append(f"fork     {'FAIL' if bad else 'PASS'}  {fork.n} questions against each asked alone: largest "
                     f"probability gap {fork.gap:.2e}, {fork.n - fork.moved}/{fork.n} top answers kept, "
                     f"{fork.same_bits}/{fork.n} the same bits")
    else:
        lines.append("fork     skip  this family reads the questions of a request together")

    load = _Check("load")
    order = list(range(len(plans)))
    random.Random(0).shuffle(order)
    for size in ROUNDS:
        if size > len(plans):
            break
        for i in range(0, len(order), size):
            sel = order[i: i + size]
            outs, _ = eng.forward([plans[k] for k in sel], "batch")
            for k, o in zip(sel, outs):
                ref = _answers(eng, reqs[k], plans[k], alone[k])
                got = _answers(eng, reqs[k], plans[k], o)
                for name in ref:
                    load.add(_probs(ref[name]), _probs(got[name]), _same(o, alone[k]))
    if load.n:
        bad = load.moved or (invariant and load.same_bits < load.n)
        failed |= bool(bad)
        lines.append(f"load     {'FAIL' if bad else 'PASS'}  {load.n} answers in rounds of "
                     f"{', '.join(str(s) for s in ROUNDS if s <= len(plans))} against each request alone: largest "
                     f"probability gap {load.gap:.2e}, {load.n - load.moved}/{load.n} top answers kept, "
                     f"{load.same_bits}/{load.n} the same bits")

    cache = eng.sessions
    if cache is None:
        lines.append("session  skip  run with --session-cache-gb to check the session cache")
    else:
        sess = _Check("session")
        before = cache.saved_tokens
        for r in _requests(args.requests, args.limit):
            text = r.get("state")
            text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, indent=2)
            rows = text.split("\n")
            cuts = sorted({max(1, round(len(rows) * k / STEPS)) for k in range(1, STEPS + 1)})
            for c in cuts:
                req = SystemOneRequest.model_validate({**r, "state": "\n".join(rows[:c])})
                try:
                    eng.sessions = None
                    off, _ = eng.decide(req)
                    eng.sessions = cache
                    on, _ = eng.decide(req)
                except ValueError:
                    eng.sessions = cache
                    continue
                for name in off.answers:
                    sess.add(_probs(off.answers[name]), _probs(on.answers[name]), off.answers[name] == on.answers[name])
        resumed = cache.saved_tokens - before
        if not resumed:
            lines.append(f"session  skip  no state resumed: states of {eng.sessions.min_tokens} tokens or more use "
                         f"the cache")
        else:
            bad = sess.moved
            failed |= bool(bad)
            lines.append(f"session  {'FAIL' if bad else 'PASS'}  {sess.n} answers over {STEPS} growing steps, "
                         f"{resumed} state tokens resumed: largest probability gap {sess.gap:.2e} against the cache "
                         f"off, {sess.n - sess.moved}/{sess.n} top answers kept")
    print("\n".join(lines), flush=True)
    if failed:
        raise SystemExit(1)


def add_parser(sub) -> None:
    v = sub.add_parser("verify", help="check exact forking, load invariance and the session cache on your requests",
                       description=__doc__)
    v.add_argument("--model", required=True, help="local directory or Hub repository id")
    v.add_argument("--requests", required=True, help="JSON lines, one System One request each")
    v.add_argument("--format", default=None, help="model family (default: detected from the checkpoint)")
    v.add_argument("--format-arg", action="append", metavar="KEY=VALUE", help="family option")
    v.add_argument("--limit", type=int, default=32, help="requests to read")
    v.add_argument("--device", default="cuda:0")
    v.add_argument("--batch-invariant", action="store_true", help="check the batch-invariant mode, bit for bit")
    v.add_argument("--session-cache-gb", type=float, default=0.0, help="also check the session cache, this large")
    v.set_defaults(func=run)

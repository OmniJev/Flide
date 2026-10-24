"""Offline run over a file of System One requests.

    flide run --model M --rows R.jsonl[.gz] ... --out DIR

Three phases, timed separately: plan (CPU workers render and tokenize with the family's own code),
forward (packed micro-batches on the GPU), finish (the family turns head outputs into answers).

A row is a request (`state`, `questions`) or a results row that carries one under `payload`. Every other
key of the row is kept and written back next to the answer, so evaluation kits (Decision Index:
`_evaluation`) can score the output file as it is.
"""
from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .. import formats

_worker = None
_DROP = ("payload", "response", "raw_output", "status", "error", "started_utc", "completed_utc", "total_wall_ms",
         "model_request_wall_ms", "engine")


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_rows(paths, every: int = 1, limit: int | None = None) -> list[dict]:
    rows, n = [], 0
    for path in paths:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if n % every == 0:
                    r = json.loads(line)
                    if "_evaluation" not in r and "payload" in r:
                        ev = {k: v for k, v in r.items() if k not in _DROP}
                        r = {"state": r["payload"]["state"], "questions": r["payload"]["questions"], "_evaluation": ev}
                    rows.append(r)
                    if limit and len(rows) >= limit:
                        return rows
                n += 1
    return rows


def parse_kv(items) -> dict:
    out = {}
    for item in items or []:
        k, _, v = item.partition("=")
        try:
            out[k.replace("-", "_")] = json.loads(v)
        except json.JSONDecodeError:
            out[k.replace("-", "_")] = v
    return out


def _request(row):
    from ..api.schema import SystemOneRequest

    req = {"model": "flide", "state": row.get("state"), "questions": row["questions"]}
    if row.get("media"):
        req["media"] = row["media"]
    return SystemOneRequest.model_validate(req)


def _init_worker(name: str, model_dir: str, format_args: dict, plan_args: dict):
    global _worker
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _worker = (formats.get(name)(model_dir, **format_args), plan_args)


def _plan_row(row: dict):
    family, plan_args = _worker
    try:
        return "ok", family.plan(_request(row), **plan_args), None
    except ValueError as exc:
        status = "unsupported" if ("limit is" in str(exc) or "slots" in str(exc)) else "error"
        return status, None, str(exc)
    except Exception as exc:
        return "error", None, f"{type(exc).__name__}: {exc}"


def plan_iter(rows, name, model_dir, workers, format_args, plan_args):
    """(status, plan, error) per row, in row order, planned by `workers` processes that run ahead of the consumer.
    Returns (iterator, pool or None); close the pool when done."""
    args = (name, model_dir, format_args, plan_args)
    if workers <= 1:
        _init_worker(*args)
        return map(_plan_row, rows), None
    pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=args)
    return pool.imap(_plan_row, rows, chunksize=8), pool


def plan_all(rows, name, model_dir, workers, format_args, plan_args, log):
    plans, status = [None] * len(rows), [("ok", None)] * len(rows)
    args = (name, model_dir, format_args, plan_args)
    step = max(1, len(rows) // 10)
    if workers <= 1:
        _init_worker(*args)
        it, pool = map(_plan_row, rows), None
    else:
        pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=args)
        it = pool.imap(_plan_row, rows, chunksize=8)
    try:
        for i, (st, plan, err) in enumerate(it):
            if st == "ok":
                plans[i] = plan
            else:
                status[i] = (st, err)
            if (i + 1) % step == 0:
                log(f"plan {i + 1}/{len(rows)}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return plans, status


def _result(row, st, err, plan, output, engine, args, plan_args, ms):
    result = {k: v for k, v in row.items() if k not in ("state", "questions", "_evaluation")}
    result.update(row.get("_evaluation", {}))
    result.update(started_utc=stamp(), engine="flide")
    if st == "ok":
        try:
            resp, _ = engine.format.finish(_request(row), plan, output, **plan_args)
            result.update(status="ok", response={"model": args.name, "answers": resp.model_dump()["answers"]})
        except Exception as exc:
            result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    else:
        result.update(status=st, error=err)
    result.update(completed_utc=stamp(), total_wall_ms=ms, model_request_wall_ms=ms)
    return result


def run_streamed(args, rows, name, model_dir, format_args, plan_args, out, log) -> dict:
    """Plan, forward and finish overlapped: the planning pool starts before the model loads and keeps running ahead,
    the GPU takes the rows block by block as soon as a block is planned, a writer thread finishes and writes each
    block in row order while the next one is on the GPU."""
    import concurrent.futures as cf

    from .engine import Engine

    t_wall = time.perf_counter()
    it, pool = plan_iter(rows, name, model_dir, args.workers, format_args, plan_args)
    t0 = time.perf_counter()
    engine = Engine(model_dir, format=name, device=args.device, dtype=args.dtype, budget=args.budget,
                    backbone_options={"quantize": args.quantize}, capture=False, batch_invariant=args.batch_invariant,
                    **format_args)
    t_load = time.perf_counter() - t0
    log(f"model loaded in {t_load:.1f}s (planning meanwhile)")
    counts: dict[str, int] = {}
    tokens = micro = 0
    t_forward = t_wait = 0.0
    writer = cf.ThreadPoolExecutor(1)
    f = (out / "results.jsonl").open("w", encoding="utf-8")

    def write(block, statuses, plans, outputs, fwd, ntok):
        for row, (st, err), plan, output in zip(block, statuses, plans, outputs):
            ms = (plan.tokens / ntok) * fwd * 1000 if plan is not None and ntok else 0.0
            r = _result(row, st, err, plan, output, engine, args, plan_args, ms)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            counts[r["status"]] = counts.get(r["status"], 0) + 1

    pending = None
    try:
        for lo in range(0, len(rows), args.stream_block):
            block = rows[lo: lo + args.stream_block]
            t0 = time.perf_counter()
            planned = [next(it) for _ in block]
            t_wait += time.perf_counter() - t0
            plans = [p if st == "ok" else None for st, p, _ in planned]
            statuses = [("ok", None) if st == "ok" else (st, err) for st, _, err in planned]
            outputs, stats = engine.run_plans(plans)
            t_forward += stats["forward_s"]
            tokens += stats["tokens"]
            micro += stats["micro_batches"]
            if pending is not None:
                pending.result()
            pending = writer.submit(write, block, statuses, plans, outputs, stats["forward_s"], stats["tokens"])
            log(f"rows {lo + len(block)}/{len(rows)}: waited {t_wait:.1f}s for plans, forward {t_forward:.1f}s, "
                f"{stats['tokens'] / max(stats['forward_s'], 1e-9):.0f} tokens/s in this block")
        t0 = time.perf_counter()
        if pending is not None:
            pending.result()
        t_tail = time.perf_counter() - t0
    finally:
        f.close()
        writer.shutdown()
        if pool is not None:
            pool.close()
            pool.join()
    total = time.perf_counter() - t_wall
    return {"rows": len(rows), "tokens": tokens, "micro_batches": micro, "budget": args.budget, "streamed": True,
            "stream_block": args.stream_block, "load_s": t_load, "plan_wait_s": t_wait, "forward_s": t_forward,
            "finish_tail_s": t_tail, "wall_s": total, "total_excl_load_s": total - t_load,
            "tokens_per_s_forward": tokens / max(t_forward, 1e-9), "workers": args.workers, "counts": counts}


def run(args) -> None:
    import torch

    from .engine import Engine, model_config, resolve_model

    format_args, plan_args = parse_kv(args.format_arg), parse_kv(args.plan_arg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    def log(msg):
        print(f"[{time.perf_counter() - t_start:8.1f}s] {msg}", flush=True)

    rows = read_rows(args.rows, args.every, args.limit)
    log(f"{len(rows)} rows from {len(args.rows)} file(s)")
    model_dir = resolve_model(args.model)
    name = formats.detect(model_dir, model_config(model_dir), args.format).name
    if args.stream_block > 0:
        timing = run_streamed(args, rows, name, model_dir, format_args, plan_args, out, log)
        env = {"engine": "flide", "format": name, "quantize": args.quantize, "batch_invariant": args.batch_invariant,
               "model": model_dir, "dtype": args.dtype, "format_args": format_args, "plan_args": plan_args,
               "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               "torch": torch.__version__, "rows": [str(x) for x in args.rows], "every": args.every,
               "limit": args.limit, "latency": "batched; per-request times are the block's forward time split by "
               "token share", "argv": sys.argv}
        (out / "timing.json").write_text(json.dumps(timing, indent=2))
        (out / "environment.json").write_text(json.dumps(env, indent=2))
        log(json.dumps(timing))
        return

    t0 = time.perf_counter()
    plans, status = plan_all(rows, name, model_dir, args.workers, format_args, plan_args, log)
    t_plan = time.perf_counter() - t0
    log(f"plan done in {t_plan:.1f}s, {sum(p is not None for p in plans)} ok")

    t0 = time.perf_counter()
    engine = Engine(model_dir, format=name, device=args.device, dtype=args.dtype, budget=args.budget,
                    backbone_options={"quantize": args.quantize}, batch_invariant=args.batch_invariant, **format_args)
    t_load = time.perf_counter() - t0
    log(f"model loaded in {t_load:.1f}s")
    outputs, stats = engine.run_plans(plans, log_fn=log, warmup=args.warmup)
    tokens, t_forward = stats["tokens"], stats["forward_s"]
    log(f"forward done in {t_forward:.1f}s, {tokens / t_forward:.0f} tokens/s")

    t0 = time.perf_counter()
    counts: dict[str, int] = {}
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            result = {k: v for k, v in row.items() if k not in ("state", "questions", "_evaluation")}
            result.update(row.get("_evaluation", {}))
            result.update(started_utc=stamp(), engine="flide")
            st, err = status[i]
            if st == "ok":
                try:
                    resp, _ = engine.format.finish(_request(row), plans[i], outputs[i], **plan_args)
                    result.update(status="ok", response={"model": args.name, "answers": resp.model_dump()["answers"]})
                except Exception as exc:
                    result.update(status="error", error=f"{type(exc).__name__}: {exc}")
            else:
                result.update(status=st, error=err)
            share = (plans[i].tokens / tokens) if plans[i] is not None and tokens else 0.0
            ms = share * t_forward * 1000
            result.update(completed_utc=stamp(), total_wall_ms=ms, model_request_wall_ms=ms)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            counts[result["status"]] = counts.get(result["status"], 0) + 1
    t_finish = time.perf_counter() - t0
    log(f"finish done in {t_finish:.1f}s, {counts}")

    timing = {"rows": len(rows), "tokens": tokens, "micro_batches": stats["micro_batches"], "budget": args.budget,
              "plan_s": t_plan, "load_s": t_load, "forward_s": t_forward, "finish_s": t_finish,
              "total_excl_load_s": t_plan + t_forward + t_finish, "tokens_per_s_forward": tokens / t_forward,
              "workers": args.workers, "counts": counts}
    env = {"engine": "flide", "format": name, "quantize": args.quantize, "batch_invariant": args.batch_invariant,
           "model": model_dir, "dtype": args.dtype, "format_args": format_args, "plan_args": plan_args,
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "torch": torch.__version__,
           "rows": [str(x) for x in args.rows], "every": args.every, "limit": args.limit,
           "latency": "batched; per-request times are the forward wall time split by token share", "argv": sys.argv}
    (out / "timing.json").write_text(json.dumps(timing, indent=2))
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    log(json.dumps(timing))


def add_parser(sub) -> None:
    p = sub.add_parser("run", help="answer a file of requests offline", description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--rows", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--format", default=None, help="model family (default: detected from the checkpoint)")
    p.add_argument("--format-arg", action="append", metavar="KEY=VALUE", help="family option, e.g. max_branch_tokens=131072")
    p.add_argument("--plan-arg", action="append", metavar="KEY=VALUE", help="per-request option, e.g. debias=1")
    p.add_argument("--name", default="flide", help="model name written into each response")
    p.add_argument("--every", type=int, default=1, help="keep every n-th row")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--budget", type=int, default=65536, help="tokens per micro-batch")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-invariant", action="store_true",
                   help="batch-invariant products: every request gets the same bits whatever it is batched with")
    p.add_argument("--quantize", default=None, choices=["fp8"],
                   help="decoder weights in float8 with per-row scales, inputs scaled per token (compute capability 8.9+)")
    p.add_argument("--warmup", type=int, default=0, help="micro-batches run once before the timed forward")
    p.add_argument("--stream-block", type=int, default=20000,
                   help="rows per GPU block while planning continues (0: plan everything, then forward, then finish)")
    p.set_defaults(func=run)

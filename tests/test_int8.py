"""int8 decoder weights (kernels/int8.py): the token quantization kernel against the same arithmetic in torch, the
product against its exact int32 sum scaled in float32 (both kernels, bit for bit), each row's bits whatever rows share
its product, then a model, one request at a time from the CUDA graphs: int8 (and fp8 beside it) against 16 bits and
float32 (same top answer, |dp|), int8 graphs against the int8 packed forward.

    PYTHONPATH=flide:flide/plugins/onejev FLIDE_PLUGINS=flide_onejev \\
        python flide/tests/test_int8.py --model MODEL --rows sample760.jsonl.gz --n 200
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys

import torch


def probs(resp) -> dict:
    out = {}
    for q, a in resp.model_dump()["answers"].items():
        out[q] = a.get("probabilities") or {"yes": a["noul"], "no": 1 - a["noul"]}
    return out


def kernels() -> bool:
    from flide.kernels import int8

    g = torch.Generator(device="cuda").manual_seed(0)
    x = (torch.randn(77, 3000, device="cuda", generator=g) * 3).to(torch.bfloat16)
    x[5] = 0
    x[9, 17] = 400.0   # an outlier channel
    q, s = int8.quantize_tokens(x)
    s_ref = x.float().abs().amax(1).clamp(min=1e-12) * (1.0 / int8.INT8_MAX)
    q_ref = (x.float() / s_ref[:, None]).round().clamp(-127, 127).to(torch.int8)
    ok = torch.equal(s, s_ref) and torch.equal(q, q_ref)
    print(f"[int8 kernel] token quantization {'bit for bit' if ok else 'DIFFERENT'} against torch "
          f"(zero row stays zero: {bool((q[5] == 0).all())})")

    def ref(xq, sx, wq, sw, dtype):
        acc = xq.double() @ wq.double().T   # exact: |sum| < 2**53
        return (acc.float() * sx[:, None] * sw[None, :]).to(dtype)

    for N, K in ((6144, 1024), (1024, 3584), (5120, 4096), (2560, 9216)):
        w = torch.randn(N, K, device="cuda", generator=g, dtype=torch.bfloat16) * 0.02
        wq, ws = int8.quantize_rows(w)
        ws = ws.reshape(-1).contiguous()
        big = (torch.randn(1500, K, device="cuda", generator=g) * 2).to(torch.bfloat16)
        bq, bs = int8.quantize_tokens(big)
        full = int8.mm_scaled(bq, bs, wq, ws, torch.bfloat16)
        exact = torch.equal(full, ref(bq, bs, wq, ws, torch.bfloat16))
        rows = True
        for M in (1, 5, 64, 65, 257):
            part = int8.mm_scaled(bq[:M], bs[:M], wq, ws, torch.bfloat16)
            rows &= torch.equal(part, full[:M])
            buf = torch.empty(M, N + 1, device="cuda", dtype=torch.bfloat16)   # misaligned: the portable kernel
            int8.mm_scaled(bq[:M], bs[:M], wq, ws, torch.bfloat16, out=buf[:, 1:])
            rows &= torch.equal(buf[:, 1:], full[:M])
            for split in (1, 4):   # the reduction in one piece and in four
                rows &= torch.equal(int8.mm_scaled(bq[:M], bs[:M], wq, ws, torch.bfloat16, split=split), full[:M])
                int8.mm_scaled(bq[:M], bs[:M], wq, ws, torch.bfloat16, out=buf[:, 1:], split=split)
                rows &= torch.equal(buf[:, 1:], full[:M])
        print(f"[int8 product] N {N} K {K}: 1500 rows {'exact' if exact else 'NOT EXACT'} against int32 sums, "
              f"rows alone, split and in the portable kernel {'the same bits' if rows else 'DIFFERENT'}")
        ok &= exact and rows

    w = torch.randn(4100, 1024, device="cuda", generator=g, dtype=torch.bfloat16) * 0.02
    wt = int8.Int8Weight.from_weight(w, tail_rows=4)
    xs = (torch.randn(33, 1024, device="cuda", generator=g)).to(torch.bfloat16)
    y = wt(xs)
    xq, sx = int8.quantize_tokens(xs)
    tail_ok = (y.shape == (33, 4100) and torch.equal(y[:, :4096], ref(xq, sx, wt.q, wt.scale, torch.bfloat16))
               and torch.equal(y[:, 4096:], xs @ w[4096:].T))
    rel = ((y.float() - (xs @ w.T).float()).norm() / (xs @ w.T).float().norm()).item()
    print(f"[int8 weight] tail rows in 16 bits {'right' if tail_ok else 'WRONG'}; relative error against bf16 {rel:.2e}")
    return ok and tail_ok


def calibrate(eng, calib, alpha: float) -> None:
    """Every product's per-channel max |input| over the calibration requests (packed, eager), then int8 with
    SmoothQuant from them."""
    from flide.kernels import int8

    bb = eng.backbone
    keys = ("gate_up", "down", "in", "out", "qkv", "o")
    saved = [{k: w[k] for k in keys if k in w} for w in bb._w]
    for w, orig in zip(bb._w, saved):
        for k, t in orig.items():
            w[k] = int8.InputMax(t)
    for r in calib:
        eng.run_plans([eng.format.plan(r)])
    stats = {}
    for i, (w, orig) in enumerate(zip(bb._w, saved)):
        for k, t in orig.items():
            stats[(i, k)] = w[k].amax
            w[k] = t
    n = bb.quantize_int8(stats, alpha)
    print(f"[int8s] {n} weights smoothed (alpha {alpha}) from {len(calib)} calibration requests", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="also a model's answers (without it, the kernels alone)")
    ap.add_argument("--rows")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--no-fp8", action="store_true", help="skip the fp8 rows (a GPU without FP8)")
    ap.add_argument("--no-int8", action="store_true", help="skip the int8 rows (a model int8 does not cover yet)")
    ap.add_argument("--smooth", type=int, default=0, help="also int8 with SmoothQuant, calibrated on this many rows "
                    "after the tested ones")
    ap.add_argument("--alpha", type=float, default=0.75)
    args = ap.parse_args(argv)
    from flide.api.schema import SystemOneRequest
    from flide.runtime.engine import Engine

    ok = kernels()
    if not args.model:
        print("->", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)

    rows = []
    for line in gzip.open(args.rows, "rt") if args.rows.endswith(".gz") else open(args.rows):
        r = json.loads(line)
        rows.append(SystemOneRequest.model_validate({"state": r["state"], "questions": r["questions"]}))
        if len(rows) >= args.n + args.smooth:
            break
    rows, calib = rows[:args.n], rows[args.n:]
    opts = dict(max_branch_tokens=131072, max_request_tokens=262144)

    def run(engine, graphs: bool):
        if graphs:
            return [probs(engine.decide(r)[0]) for r in rows]
        out = []
        for r in rows:
            plan = engine.format.plan(r)
            out.append(probs(engine.format.finish(r, plan, engine.run_plans([plan])[0][0])[0]))
        return out

    runs = [("fp32", "float32", None), ("bf16", "bfloat16", None)] + ([] if args.no_int8 else [("int8", "bfloat16", "int8")])
    if not args.no_fp8:
        runs.append(("fp8", "bfloat16", "fp8"))
    if calib:
        runs.append(("int8s", "bfloat16", "smooth"))
    res = {}
    for name, dtype, quant in runs:   # the float32 truth packed, without graph memory (a 27B fits one H200)
        eng = Engine(args.model, dtype=dtype, capture=False, **opts,
                     backbone_options={"cuda_graphs": name != "fp32", "quantize": None if quant == "smooth" else quant})
        if quant == "smooth":
            calibrate(eng, calib, args.alpha)
        print(f"[{name}] running", flush=True)
        if name != "fp32":
            res[name + " graphs"] = run(eng, True)
        res[name + " packed"] = run(eng, False)
        del eng
        torch.cuda.empty_cache()

    def compare(a, b):
        n = same = 0
        dps = []
        for x, y in zip(a, b):
            for k in x:
                n += 1
                same += max(x[k], key=x[k].get) == max(y[k], key=y[k].get)
                dps.append(max(abs(x[k][c] - y[k].get(c, 0)) for c in x[k]))
        return same / n, statistics.median(dps), max(dps), n

    def hits(a, b):   # whether each question's top answer is the other's, in order
        return [max(x[k], key=x[k].get) == max(y[k], key=y[k].get) for x, y in zip(a, b) for k in x]

    truth = res["fp32 packed"]
    for name in [k for k in res if k != "fp32 packed"]:
        agree, med, mx, n = compare(res[name], truth)
        print(f"[{name:11} vs fp32 truth] {n} questions: same top {agree:.4f}, |dp| median {med:.4f} max {mx:.4f}")
    pairs = [(f"{q} graphs", f"{q} packed") for q in ("bf16", "int8", "fp8", "int8s")] + [("int8 graphs", "bf16 graphs")]
    for a, b in [(a, b) for a, b in pairs if a in res and b in res]:   # each path against the other: the spread
        agree, med, mx, _ = compare(res[a], res[b])           # 8-bit rounding adds between two 16-bit paths
        print(f"[{a} vs {b}] same top {agree:.4f}, |dp| median {med:.4f} max {mx:.4f}")
    if "int8 graphs" in res:   # fp8, shipped before, is printed beside it as the reference
        g_t = compare(res["int8 graphs"], truth)
        p_t = compare(res["int8 packed"], truth)
        # the two paths are equally far from the truth (neither drops a layer to 16 bits or mis-scales one): the questions
        # only one of them gets right balance within 4 standard deviations, and so do the median distances
        pairs = [(a, b) for a, b in zip(hits(res["int8 graphs"], truth), hits(res["int8 packed"], truth))]
        only_g, only_p = sum(a and not b for a, b in pairs), sum(b and not a for a, b in pairs)
        print(f"[int8 graphs alone right {only_g}, int8 packed alone right {only_p}]")
        ok &= abs(only_g - only_p) <= 4 * max(1, only_g + only_p) ** 0.5 and abs(g_t[1] - p_t[1]) <= 0.005
    if "int8 graphs" in res and "fp8 graphs" in res:   # int8 keeps as many top answers as fp8 does, within 4 standard
        # deviations of the questions only one of them gets right (a fixed line is one deviation wide on a thousand questions)
        pairs = list(zip(hits(res["int8 graphs"], truth), hits(res["fp8 graphs"], truth)))
        only_i, only_f = sum(a and not b for a, b in pairs), sum(b and not a for a, b in pairs)
        print(f"[int8 graphs alone right {only_i}, fp8 graphs alone right {only_f}]")
        ok &= only_f - only_i <= 4 * max(1, only_i + only_f) ** 0.5
    print("->", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

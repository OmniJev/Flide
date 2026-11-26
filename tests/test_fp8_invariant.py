"""Batch-invariant float8 products (kernels/fp8_invariant.py): every row's bits the same whatever rows share its
product and whichever tile shape is picked, and the result as close to the float32 sum of the same float8 operands as
cuBLAS's scaled matmul. With --model, answers on DI requests in float32, in float8 through cuBLAS and in float8 through
the batch-invariant products (same top answer as float32, largest probability gap).

    python tests/test_fp8_invariant.py [--model MODEL --rows ROWS.jsonl[.gz] --n 64]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys

import torch


def kernels() -> bool:
    from flide.kernels import fp8, fp8_invariant as fi

    g = torch.Generator(device="cuda").manual_seed(0)
    ok = True
    for M, N, K in ((1, 4096, 2560), (7, 9216, 2560), (300, 2560, 9216), (1500, 6144, 4096), (2049, 1024, 1024)):
        x = torch.randn(M, K, device="cuda", generator=g).to(torch.bfloat16)
        w = (torch.randn(N, K, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
        wq, ws = fp8.quantize_rows(w)
        xq, sx = fp8.quantize_tokens(x)
        y = fi.mm_scaled(xq, sx, wq, ws, torch.bfloat16)
        ref = (xq.float() @ wq.float().T) * sx.reshape(-1, 1) * ws.reshape(1, -1)
        cub = torch._scaled_mm(xq, wq.t(), scale_a=sx, scale_b=ws.reshape(1, -1).contiguous(), out_dtype=torch.bfloat16)
        scale = ref.abs().max().item()
        err, cerr = (y.float() - ref).abs().max().item() / scale, (cub.float() - ref).abs().max().item() / scale
        rows = sorted({0, M // 2, M - 1})
        alone = torch.cat([fi.mm_scaled(xq[i:i + 1], sx[i:i + 1], wq, ws, torch.bfloat16) for i in rows])
        same_rows = torch.equal(alone, y[rows])
        shapes = [fi.mm_scaled(xq, sx, wq, ws, torch.bfloat16, cfg=c) for c in (fi.WIDE, fi.NARROW, fi.SMALL)
                  if fi._needs(c, 2) <= fi.budget(xq.device)]
        same_shapes = all(torch.equal(s, y) for s in shapes)
        good = same_rows and same_shapes and err <= max(2 * cerr, 1e-2)
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} M={M} N={N} K={K}: rows alone {same_rows}, {len(shapes)} tile shapes "
              f"{same_shapes}, error against float32 {err:.2e} (cuBLAS {cerr:.2e})", flush=True)
    return ok


def probs(resp) -> dict:
    out = {}
    for q, a in resp.model_dump()["answers"].items():
        out[q] = a.get("probabilities") or {"yes": a["noul"], "no": 1 - a["noul"]}
    return out


def model_check(model: str, rows_path: str, n: int) -> None:
    from flide.api.schema import SystemOneRequest
    from flide.runtime.engine import Engine

    rows = []
    for line in (gzip.open(rows_path, "rt") if rows_path.endswith(".gz") else open(rows_path)):
        r = json.loads(line)
        rows.append(SystemOneRequest.model_validate({"state": r["state"], "questions": r["questions"]}))
        if len(rows) >= n:
            break
    res = {}
    for name, dtype, quant, inv in (("float32", "float32", None, False), ("fp8 cuBLAS", "bfloat16", "fp8", False),
                                    ("fp8 invariant", "bfloat16", "fp8", True)):   # invariant last: it is process-wide
        graphs = dtype != "float32"
        eng = Engine(model, dtype=dtype, capture=4096 if graphs else False, batch_invariant=inv,
                     backbone_options={"cuda_graphs": graphs, "quantize": quant})
        out = []
        for r in rows:
            if graphs:
                out.append(probs(eng.decide(r)[0]))
            else:
                plan = eng.format.plan(r)
                out.append(probs(eng.format.finish(r, plan, eng.run_plans([plan])[0][0])[0]))
        res[name] = out
        del eng
        torch.cuda.empty_cache()
    truth = res["float32"]
    for name in ("fp8 cuBLAS", "fp8 invariant"):
        n_q = same = 0
        gap = []
        for x, y in zip(res[name], truth):
            for k in x:
                n_q += 1
                same += max(x[k], key=x[k].get) == max(y[k], key=y[k].get)
                gap.append(max(abs(x[k][c] - y[k].get(c, 0)) for c in x[k]))
        gap.sort()
        print(f"{name} against float32: same top {same}/{n_q}, |dp| median {gap[len(gap) // 2]:.4f}, "
              f"max {gap[-1]:.4f}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--rows")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args(argv)
    from flide.kernels import fp8

    if not fp8.supported(torch.device("cuda:0")):
        print("SKIP: this GPU has no float8 products (compute capability below 8.9)")
        return 0
    ok = kernels()
    if args.model:
        model_check(args.model, args.rows, args.n)
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

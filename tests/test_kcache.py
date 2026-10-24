"""Kernel bundles without a GPU: a kernel compiled for a target is filed by `kcache.seed` under the very key
Triton's compile() looks up, so compiling it again in a fresh cache directory is a cache hit.

    python tests/test_kcache.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import triton
import triton.language as tl
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, make_backend

from flide.runtime import kcache


@triton.jit
def _axpy(x, y, n, a: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(y + i, tl.load(x + i, mask=m) * a + tl.load(y + i, mask=m), mask=m)


@triton.autotune(configs=[triton.Config({"BLOCK": b}, num_warps=w) for b in (64, 128) for w in (2, 4)], key=["n"])
@triton.jit
def _tuned(x, y, n, a: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y + i, tl.load(x + i, mask=i < n) * a, mask=i < n)


class _Launch:
    def __init__(self, kernel):
        self.kernel = kernel


_wrapped = _Launch(_tuned)


def compile_once(target, launch):
    src = ASTSource(_axpy, {"x": "*fp32", "y": "*fp32", "n": "i32", "a": "constexpr", "BLOCK": "constexpr"},
                    {"a": 2.0, "BLOCK": 128}, {(0,): [["tt.divisibility", 16]], (1,): [["tt.divisibility", 16]]})
    opts = make_backend(target).parse_options(dict(launch)).__dict__
    hits = []
    knobs.compilation.listener = lambda **kw: hits.append(kw["cache_hit"])
    k = triton.compile(src, target=target, options=opts)
    knobs.compilation.listener = None
    return src, opts, k, hits[-1]


def main():
    target = GPUTarget("cuda", 90, 32)
    tdict = {"backend": "cuda", "arch": 90, "warp_size": 32}
    fails = 0
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        kcache.SHIPPED = Path(b) / "shipped"
        os.environ["TRITON_CACHE_DIR"] = a
        launch = {"num_warps": 4, "num_stages": 2}
        src, opts, k, hit = compile_once(target, launch)
        options = json.loads(json.dumps(kcache._portable(opts)))
        h = kcache._hash(src.hash(), options, make_backend(target), kcache._env_vars())
        ok = h == k.hash and not hit
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} portable recipe gives compile()'s key ({h[:12]} vs {k.hash[:12]})")

        from triton.runtime.cache import get_cache_manager

        group = get_cache_manager(h).get_group("_axpy.json")
        meta = json.loads(Path(group["_axpy.json"]).read_text())
        meta.pop("extern_libs", None)
        man = {"format": kcache.FORMAT, "triton": triton.__version__, "target": tdict, "tuning": {},
               "kernels": [{"fn": "t:_axpy", "src": src.hash(), "options": options, "meta": "_axpy.json",
                            "dir": "k00000", "files": ["_axpy.cubin", "_axpy.json"]}]}
        bundle = Path(b) / "k.tgz"
        kcache._write(bundle, man, {"k00000/_axpy.cubin": Path(group["_axpy.cubin"]).read_bytes(),
                                    "k00000/_axpy.json": json.dumps(meta).encode()})
        stored = json.loads(json.dumps(man))
        leaked = [k for k in ("extern_libs",) if k in json.dumps(stored["kernels"][0]["options"])]
        ok = not leaked
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} no library paths in the bundle")

        fresh = Path(b) / "cache"
        os.environ["TRITON_CACHE_DIR"] = str(fresh)
        n = kcache.seed(str(bundle), target=tdict)
        again = kcache.seed(str(bundle), target=tdict)
        _, _, k2, hit = compile_once(target, launch)
        ok = n == 1 and again == 0 and hit and k2.hash == h and k2.asm["cubin"] == k.asm["cubin"]
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} seeded cache hits ({n} seeded, {again} on the second seed, hit {hit})")

        other = Path(b) / "other"
        os.environ["TRITON_CACHE_DIR"] = str(other)
        n = kcache.seed(str(bundle), target={**tdict, "arch": 80})
        ok = n == 0 and not other.exists()
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} a bundle for another GPU is skipped")
    pick = {"BLOCK": 128, "num_warps": 2, "num_ctas": 1, "num_stages": 3}
    kcache._picks[f"{__name__}:_wrapped"] = [[[4096, "torch.float32", "torch.float32"], pick]]
    n = kcache.tune()
    got = _tuned.cache.get((4096, "torch.float32", "torch.float32"))
    ok = n == 1 and got is not None and kcache._config(got) == pick and got in _tuned.configs and not kcache._picks
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} an autotuner pick reaches the autotuner through a launch wrapper")
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

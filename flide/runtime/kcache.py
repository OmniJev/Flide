"""Compiled kernels carried between machines, so a first start neither compiles nor tunes.

Triton files a compiled kernel under a hash of the kernel's source and specialization, its options, the GPU target
and the Triton install itself (its shared library and its ptxas build), so a copy of one machine's cache directory
hits only where the very same Triton wheel is installed: another Python version or CPU architecture misses every
kernel. A bundle keeps per kernel the parts of that hash that travel (the source and specialization hash, the
options) with the binary (cubin and Triton's metadata), and files each kernel under the key this machine's Triton
computes. It also keeps every autotuner's picks, which go into the autotuners once their modules are loaded, so
nothing is benchmarked at start either.

`flide kernels export --model M [--model M2 ...] --out K.tgz` warms the models on a GPU and packs every kernel
they ran (`--no-weights` builds each model from its config alone: kernels depend on shapes, not on values).
Bundles are applied at engine start from the
package's `kernels/prebuilt/gpu`; entries for another Triton version or GPU are skipped.
"""
from __future__ import annotations

import gc
import hashlib
import importlib
import io
import json
import logging
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path

log = logging.getLogger("flide.kernels")

FORMAT = 2
MANIFEST = "flide-kernels.json"
SHIPPED = Path(__file__).resolve().parent.parent / "kernels" / "prebuilt" / "gpu"

_picks: dict[str, list] = {}
_matched = False


def cache_dir() -> Path:
    d = os.environ.get("TRITON_CACHE_DIR")
    if d:
        return Path(d)
    try:
        from triton import knobs

        if knobs.cache.dir:
            return Path(knobs.cache.dir)
    except Exception:
        pass
    return Path.home() / ".triton" / "cache"


class _Src:
    def __init__(self, digest: str):
        self.digest = digest

    def hash(self) -> str:
        return self.digest


def _target() -> dict:
    from triton.runtime import driver

    t = driver.active.get_current_target()
    return {"backend": t.backend, "arch": t.arch, "warp_size": t.warp_size}


def _backend(target: dict):
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import make_backend

    return make_backend(GPUTarget(target["backend"], target["arch"], target["warp_size"]))


def _portable(options: dict) -> dict:
    """Options as stored: without the library paths (each machine fills in its own copy, hashed by content)."""
    return {k: v for k, v in options.items() if k != "extern_libs"}


def _hash(src: str, options: dict, backend, env_vars) -> str:
    """The hash compile() files a kernel under on this machine."""
    from triton.compiler.compiler import get_cache_key

    opts = backend.parse_options({k: tuple(v) if isinstance(v, list) else v for k, v in options.items()})
    return hashlib.sha256(get_cache_key(_Src(src), backend, opts, env_vars).encode("utf-8")).hexdigest()


def _env_vars():
    from triton._C.libtriton import get_cache_invalidating_env_vars

    return get_cache_invalidating_env_vars()


def _name(jit) -> str:
    return f"{jit.module}:{jit.fn.__qualname__}"


def _jit(obj):
    from triton.runtime.jit import JITFunction

    while obj is not None and not isinstance(obj, JITFunction):
        obj = getattr(obj, "fn", None)
    return obj


def _autotuner(obj):
    """The autotuner inside a module attribute: through Triton's wrappers (`.fn`) and launch wrappers that keep the
    kernel as `.kernel` (kernels/fla.py)."""
    from triton.runtime.autotuner import Autotuner

    while obj is not None and not isinstance(obj, Autotuner):
        obj = getattr(obj, "fn", None) or getattr(obj, "kernel", None)
    return obj


def _config(cfg) -> dict | None:
    kw = {k: v for k, v in cfg.all_kwargs().items() if k != "pre_hook"}
    try:
        json.dumps(kw)
    except TypeError:
        return None
    return kw


def bundles(extra: str | None = None) -> list[Path]:
    out = []
    for part in (extra or "").split(":"):
        if not part:
            continue
        p = Path(part)
        if p.is_dir():
            out += sorted(p.glob("*.tgz"))
        elif p.is_file():
            out.append(p)
        else:
            log.warning("kernel bundle %s not found; kernels compile on first use", part)
    if SHIPPED.is_dir():
        out += sorted(SHIPPED.glob("*.tgz"))
    return out


def seed(bundle: str | None = None, target: dict | None = None) -> int:
    """File every kernel of the bundles that fit this machine into its Triton cache (kernels already there are
    kept) and hold their autotuner picks for `tune()`. Returns the kernels added."""
    paths = bundles(bundle)
    if not paths:
        return 0
    try:
        import triton
        from triton.runtime.cache import get_cache_manager, triton_key

        target = target or _target()
    except Exception as e:
        log.debug("no kernel seeding: %s", e)
        return 0
    added = 0
    backend = env_vars = None
    install = hashlib.sha256(triton_key().encode()).hexdigest()[:12]
    for path in paths:
        t0 = time.perf_counter()
        try:
            with tarfile.open(path, "r:gz") as tar:
                man = json.load(tar.extractfile(MANIFEST))
                if man.get("format") != FORMAT or man["triton"] != triton.__version__ or man["target"] != target:
                    continue
                global _matched
                _matched = True
                for fn, picks in man.get("tuning", {}).items():
                    _picks.setdefault(fn, []).extend(picks)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                marker = cache_dir() / f".flide-bundle-{digest}-{install}"
                if marker.exists():
                    continue
                if backend is None:
                    backend, env_vars = _backend(target), _env_vars()
                n = 0
                for e in man["kernels"]:
                    h = _hash(e["src"], e["options"], backend, env_vars)
                    mgr = get_cache_manager(h)
                    if mgr.get_group(e["meta"]) is not None:
                        continue
                    group = {}
                    for fname in e["files"]:
                        data = tar.extractfile(f"{e['dir']}/{fname}").read()
                        if fname == e["meta"]:
                            meta = json.loads(data)
                            meta["hash"] = h
                            meta.pop("extern_libs", None)
                            group[fname] = mgr.put(json.dumps(meta), fname, binary=False)
                        else:
                            group[fname] = mgr.put(data, fname)
                    mgr.put_group(e["meta"], group)
                    n += 1
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(str(path))
        except (OSError, KeyError, ValueError, tarfile.TarError) as e:
            log.warning("kernel bundle %s not applied: %s", path, e)
            continue
        added += n
        log.info("seeded %s with %d kernels from %s in %.1fs", cache_dir(), n, path.name, time.perf_counter() - t0)
    tune()
    return added


def first_compile() -> bool:
    """Whether this start compiles its kernels: no bundle fits this GPU and Triton, and the Triton cache is empty."""
    d = cache_dir()
    return not _matched and not (d.is_dir() and any(d.iterdir()))


def tune() -> int:
    """Put held autotuner picks into their autotuners. Kernel modules of flide and fla are imported for it (some
    are imported only when graphs are captured, after the first kernels have run); others take their picks at a later
    call once loaded."""
    done = 0
    for fn in list(_picks):
        mod, qual = fn.split(":", 1)
        m = sys.modules.get(mod)
        if m is None and mod.split(".")[0] in ("flide", "fla"):
            try:
                m = importlib.import_module(mod)
            except Exception as e:
                log.debug("autotuner picks for %s not applied: %s", fn, e)
                _picks.pop(fn)
                continue
        if m is None:
            continue
        obj = m
        for part in qual.split("."):
            obj = getattr(obj, part, None)
        a = _autotuner(obj)
        picks = _picks.pop(fn)
        if a is None:
            continue
        configs = [(_config(c), c) for c in a.configs]
        for key, kw in picks:
            key = tuple(key)
            cfg = next((c for k, c in configs if k == kw), None)
            if cfg is not None and key not in a.cache:
                a.cache[key] = cfg
                done += 1
    return done


class _Recorder:
    """Every kernel the JIT compiles in this process, with what it takes to find it in another machine's cache."""

    def __init__(self):
        self.seen = {}

    def __call__(self, *, key, repr, fn, compile, is_manual_warmup, already_compiled):
        from triton.compiler.compiler import ASTSource

        jit = fn.jit_function
        spec = json.loads(compile["specialization_data"])
        src = ASTSource(jit, compile["signature"], compile["constants"], compile["configs"][0])
        self.seen[(src.hash(), json.dumps(_portable(spec["options"]), sort_keys=True))] = (_name(jit), src.name)
        return None


def _pack(rec: _Recorder, out: Path) -> tuple[int, int]:
    import triton
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.cache import get_cache_manager

    target = _target()
    backend, env_vars = _backend(target), _env_vars()
    kernels, missing = [], 0
    buf = {}
    for i, ((src, opts), (fn, name)) in enumerate(sorted(rec.seen.items())):
        options = json.loads(opts)
        group = get_cache_manager(_hash(src, options, backend, env_vars)).get_group(f"{name[:150]}.json")
        if group is None:
            missing += 1
            continue
        meta = f"{name[:150]}.json"
        files = sorted(f for f in group if f == meta or f.endswith(".cubin"))
        d = f"k{i:05d}"
        for f in files:
            data = Path(group[f]).read_bytes()
            if f == meta:
                m = json.loads(data)
                m.pop("extern_libs", None)
                data = json.dumps(m).encode()
            buf[f"{d}/{f}"] = data
        kernels.append({"fn": fn, "src": src, "options": options, "meta": meta, "dir": d, "files": files})
    tuning: dict[str, list] = {}
    for a in gc.get_objects():
        if isinstance(a, Autotuner) and a.cache:
            jit = _jit(a)
            for key, cfg in a.cache.items():
                kw = _config(cfg)
                try:
                    json.dumps(list(key))
                except TypeError:
                    continue
                if kw is not None:
                    tuning.setdefault(_name(jit), []).append([list(key), kw])
    man = {"format": FORMAT, "triton": triton.__version__, "target": target, "kernels": kernels, "tuning": tuning}
    _write(out, man, buf)
    return len(kernels), missing


def _write(out: Path, man: dict, buf: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for name, data in [(MANIFEST, json.dumps(man).encode())] + sorted(buf.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))


def export(args) -> None:
    """Warm each model in a fresh cache directory and pack every kernel that ran."""
    tmp = tempfile.mkdtemp(prefix="flide-kernels-")
    os.environ["TRITON_CACHE_DIR"] = tmp
    os.environ["TRITON_STORE_BINARY_ONLY"] = "1"
    os.environ["TRITON_DISABLE_LINE_INFO"] = "1"
    import torch
    from triton import knobs

    from .engine import Engine

    rec = _Recorder()
    knobs.runtime.jit_post_compile_hook = rec
    for model in args.model:
        t0 = time.perf_counter()
        opts = {"cuda_graphs": True, "quantize": args.quantize}
        if args.no_weights:
            opts["loader"] = "none"
        eng = Engine(model, format=args.format, device=args.device, dtype=args.dtype, capture=True,
                     session_cache_gb=1.0 if args.sessions else 0.0, backbone_options=opts)
        log.info("%s warm in %.1fs, %d kernels so far", model, time.perf_counter() - t0, len(rec.seen))
        del eng
        gc.collect()
        torch.cuda.empty_cache()
    knobs.runtime.jit_post_compile_hook = None
    n, missing = _pack(rec, Path(args.out))
    out = Path(args.out)
    print(f"{out}: {n} kernels, {out.stat().st_size / 1e6:.1f} MB" + (f", {missing} not found in the cache" if missing else ""))
    marks = {str(Path.home()).encode(), b"site-packages", os.getcwd().encode()} - {b"/", b""}
    with tarfile.open(out, "r:gz") as tar:
        named = [m.name for m in tar.getmembers() if m.isfile() and any(x in tar.extractfile(m).read() for x in marks)]
    if named:
        print(f"{len(named)} files name a path of this machine, e.g. {named[0]}")


def merge(args) -> None:
    """One bundle from several made with the same Triton on the same GPU target (kernels and picks united)."""
    man, buf, seen = None, {}, set()
    for path in args.bundles:
        with tarfile.open(path, "r:gz") as tar:
            m = json.load(tar.extractfile(MANIFEST))
            if man is None:
                man = {**m, "kernels": [], "tuning": {}}
            elif (m["triton"], m["target"]) != (man["triton"], man["target"]):
                raise SystemExit(f"{path}: Triton {m['triton']} on {m['target']}, not {man['triton']} on {man['target']}")
            for e in m["kernels"]:
                k = (e["src"], json.dumps(e["options"], sort_keys=True))
                if k in seen:
                    continue
                seen.add(k)
                d = f"k{len(man['kernels']):05d}"
                for f in e["files"]:
                    buf[f"{d}/{f}"] = tar.extractfile(f"{e['dir']}/{f}").read()
                man["kernels"].append({**e, "dir": d})
            for fn, picks in m["tuning"].items():
                have = man["tuning"].setdefault(fn, [])
                have += [p for p in picks if p not in have]
    _write(Path(args.out), man, buf)
    print(f"{args.out}: {len(man['kernels'])} kernels")


def info(args) -> None:
    with tarfile.open(args.bundle, "r:gz") as tar:
        m = json.load(tar.extractfile(MANIFEST))
    t = m["target"]
    picks = sum(len(v) for v in m["tuning"].values())
    print(f"Triton {m['triton']}, {t['backend']} sm_{t['arch']}: {len(m['kernels'])} kernels, "
          f"{picks} autotuner picks over {len(m['tuning'])} kernels")


def add_parser(sub) -> None:
    k = sub.add_parser("kernels", help="kernel bundles: compile once, start without compiling elsewhere")
    ks = k.add_subparsers(dest="kernels_command", required=True)
    e = ks.add_parser("export", help="warm models in a fresh cache and pack the compiled kernels")
    e.add_argument("--model", required=True, action="append")
    e.add_argument("--out", required=True)
    e.add_argument("--format", default=None)
    e.add_argument("--device", default="cuda:0")
    e.add_argument("--dtype", default="bfloat16")
    e.add_argument("--sessions", action="store_true", help="also compile the session (save / resume) path")
    e.add_argument("--no-weights", action="store_true", help="build each model from its config, weights unread")
    e.add_argument("--quantize", default=None, choices=["fp8"], help="the kernels of FP8 serving")
    e.set_defaults(func=export)
    s = ks.add_parser("seed", help="file a bundle's kernels into this machine's kernel cache")
    s.add_argument("bundle")
    s.set_defaults(func=lambda a: print(f"{seed(a.bundle)} kernels added to {cache_dir()}"))
    m = ks.add_parser("merge", help="unite bundles made with the same Triton on the same GPU")
    m.add_argument("out")
    m.add_argument("bundles", nargs="+")
    m.set_defaults(func=merge)
    i = ks.add_parser("info", help="what a bundle holds")
    i.add_argument("bundle")
    i.set_defaults(func=info)

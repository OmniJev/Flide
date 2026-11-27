"""Compiled kernels carried between machines, so a first start neither compiles nor tunes.

Triton files a compiled kernel under a hash of the kernel's source and specialization, its options, the GPU target
and the Triton install itself (its shared library and its ptxas build), so a copy of one machine's cache directory
hits only where the very same Triton wheel is installed: another Python version or CPU architecture misses every
kernel. A bundle keeps per kernel the parts of that hash that travel (the source and specialization hash, the
options) with the binary (cubin, hsaco on AMD, and Triton's metadata), and files each kernel under the key this
machine's Triton computes. It also keeps every autotuner's picks, which go into the autotuners once their modules are
loaded, so nothing is benchmarked at start either.

`flide kernels export --model M [--model M2 ...] --out K.tar.xz` warms the models on a GPU and packs every kernel
they ran (`--no-weights` builds each model from its config alone: kernels depend on shapes, not on values);
`--tuning` starts its autotuners from an earlier bundle's picks, so a new bundle keeps the configs (and so the bits)
of the one it replaces wherever the keys recur. Bundles are xz-compressed tar files; gzip ones (`.tgz`) still load.
Bundles are applied at engine start from the
package's `kernels/prebuilt/gpu`; entries for another Triton version or GPU are skipped. `flide kernels pack` puts the
bundles of several GPUs and Triton versions into one archive, the same kernel's files side by side: they compress
against each other, and a start reads past the ones that are not its own.
"""
from __future__ import annotations

import gc
import hashlib
import importlib
import io
import json
import logging
import lzma
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
PATTERNS = ("*.tar.xz", "*.tgz")   # bundle files in a directory
SOLID = "flide-bundles.json"   # the first member of an archive of several bundles: {"format", "bundles": {name: manifest}}
SOLID_FORMAT = 1

_picks: dict[str, list] = {}   # autotuner picks from applied bundles, by kernel, until its module is loaded
_matched = False   # a bundle for this GPU and Triton was found


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
    kernel as `.kernel`."""
    from triton.runtime.autotuner import Autotuner

    while obj is not None and not isinstance(obj, Autotuner):
        obj = getattr(obj, "fn", None) or getattr(obj, "kernel", None)
    return obj


LAUNCH = ("num_warps", "num_stages", "num_ctas", "maxnreg")   # the launch options an autotuner config may set


def _kernel_config(jit, constants: dict, options: dict) -> dict:
    """The constexpr values and launch options a kernel was compiled with, in an autotuner config's terms."""
    out = {k: options[k] for k in LAUNCH if k in options}
    names = getattr(jit, "arg_names", None) or []
    for key, value in constants.items():
        if isinstance(key, tuple) and len(key) == 1 and key[0] < len(names):
            v = getattr(value, "value", value)
            if v is None or isinstance(v, (bool, int, float, str)):
                out[names[key[0]]] = v
    return out


def _picked(entry: dict, tuning: dict) -> bool:
    """False for an autotuner's losing candidate: a kernel of a tuned function whose config no pick names (the
    export compiled it only to time it). Kernels without a recorded config are kept."""
    picks, cfg = tuning.get(entry["fn"]), entry.get("config")
    if not picks or cfg is None:
        return True
    return any(all(cfg.get(k) == v for k, v in kw.items()) for _, kw in picks)


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
            out += sorted(x for pat in PATTERNS for x in p.glob(pat))
        elif p.is_file():
            out.append(p)
        else:
            log.warning("kernel bundle %s not found; kernels compile on first use", part)
    if SHIPPED.is_dir():
        out += sorted(x for pat in PATTERNS for x in SHIPPED.glob(pat))
    return out


def _index(path: Path) -> dict[str, dict]:
    """{name: manifest} of the bundles in a file, the name "" for a file that is one bundle."""
    with tarfile.open(path, "r:*") as tar:
        first = tar.next()
        if first is None:
            raise ValueError("empty archive")
        if first.name != SOLID:
            return {"": json.load(tar.extractfile(MANIFEST))}
        index = json.load(tar.extractfile(first))
        if index.get("format") != SOLID_FORMAT:
            raise ValueError(f"archive format {index.get('format')}")
        return index["bundles"]


def _files(path: Path, name: str) -> dict[str, bytes]:
    """The kernel files of one bundle of `path` ({"dir/file": bytes}). An xz file is decompressed whole, several times
    faster than member by member (179 MB for the ten shipped bundles, released when this returns)."""
    data = path.read_bytes()
    tar = tarfile.open(fileobj=io.BytesIO(lzma.decompress(data)), mode="r:") if data[:6] == b"\xfd7zXZ\0" \
        else tarfile.open(path, "r:*")
    out = {}
    with tar:
        for m in tar:
            top, _, rest = m.name.partition("/")
            if name:
                if top != name:
                    continue
            else:
                rest = m.name
            if m.isfile() and rest not in (MANIFEST, SOLID):
                out[rest] = tar.extractfile(m).read()
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
    except Exception as e:   # no GPU or no Triton: nothing to seed
        log.debug("no kernel seeding: %s", e)
        return 0
    added = 0
    backend = env_vars = None
    install = hashlib.sha256(triton_key().encode()).hexdigest()[:12]
    for path in paths:
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            index = _index(path)
        except (OSError, KeyError, ValueError, tarfile.TarError) as e:
            log.warning("kernel bundle %s not applied: %s", path, e)
            continue
        for name, man in index.items():
            t0 = time.perf_counter()
            try:
                if man.get("format") != FORMAT or man["triton"] != triton.__version__ or man["target"] != target:
                    continue
                global _matched
                _matched = True
                for fn, picks in man.get("tuning", {}).items():
                    _picks.setdefault(fn, []).extend(picks)
                marker = cache_dir() / f".flide-bundle-{digest}-{name + '-' if name else ''}{install}"
                if marker.exists():
                    continue
                if backend is None:
                    backend, env_vars = _backend(target), _env_vars()
                files, n = None, 0
                for e in man["kernels"]:
                    h = _hash(e["src"], e["options"], backend, env_vars)
                    mgr = get_cache_manager(h)
                    if mgr.get_group(e["meta"]) is not None:
                        continue
                    if files is None:
                        files = _files(path, name)
                    group = {}
                    for fname in e["files"]:
                        data = files[f"{e['dir']}/{fname}"]
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
            log.info("seeded %s with %d kernels from %s in %.1fs", cache_dir(), n, name or path.name,
                     time.perf_counter() - t0)
    tune()
    return added


def hold_picks(extra: str, target: dict | None = None) -> int:
    """Hold the autotuner picks of the bundles in `extra` (files or directories, ':'-separated) made for this GPU and
    Triton, their kernels left out, and put them into the autotuners now. An export started this way keeps an earlier
    bundle's configs wherever its keys recur, so answers do not move between releases for a tuner's pick alone; keys
    it lacks are tuned as usual. Returns the picks held."""
    import triton

    target, n = target or _target(), 0
    for path in bundles(extra):
        if path.parent == SHIPPED:
            continue
        for man in _index(path).values():
            if man.get("format") != FORMAT or man["triton"] != triton.__version__ or man["target"] != target:
                continue
            for fn, picks in man.get("tuning", {}).items():
                _picks.setdefault(fn, []).extend(picks)
                n += len(picks)
    tune()
    return n


def no_bundle() -> str | None:
    """This GPU and Triton ("sm_120 with Triton 3.8.0") when no bundle fits them, so kernels missing from the Triton
    cache compile on first use; None when one fits, or with no GPU or no Triton."""
    if _matched:
        return None
    try:
        import triton

        t = _target()
    except Exception:
        return None
    return f"sm_{t['arch']} with Triton {triton.__version__}" if t["backend"] == "cuda" else \
        f"{t['backend']} {t['arch']} with Triton {triton.__version__}"


def tune() -> int:
    """Put held autotuner picks into their autotuners. Kernel modules of flide are imported for it (some are
    imported only when graphs are captured, after the first kernels have run); others take their picks at a later call
    once loaded. Picks recorded under fla's module names (bundles made while fla was a dependency) go to the copy in
    kernels/fla, which keeps fla's module layout."""
    done = 0
    for fn in list(_picks):
        mod, qual = fn.split(":", 1)
        if mod.split(".")[0] == "fla":
            mod = f"{__name__.split('.')[0]}.kernels.{mod}"
        m = sys.modules.get(mod)
        if m is None and mod.split(".")[0] == __name__.split(".")[0]:
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


class _Progress:
    """Counts the kernels the JIT builds and says how many every few seconds, so a first start that compiles them
    shows it is moving; a hook that was there (the exporter's recorder) still runs."""

    def __init__(self, inner, every: float = 10.0):
        self.inner, self.every, self.n = inner, every, 0
        self.t0 = self.last = time.perf_counter()

    def __call__(self, **kw):
        self.n += 1
        now = time.perf_counter()
        if now - self.last >= self.every:
            self.last = now
            log.info("compiling kernels: %d so far, %.0f s", self.n, now - self.t0)
        return self.inner(**kw) if self.inner is not None else None


def progress(on: bool) -> None:
    """Start counting the kernels the JIT builds, or stop and say how many were built, if any."""
    try:
        from triton import knobs
    except ImportError:
        return
    hook = knobs.runtime.jit_post_compile_hook
    if on and not isinstance(hook, _Progress):
        knobs.runtime.jit_post_compile_hook = _Progress(hook)
    elif not on and isinstance(hook, _Progress):
        knobs.runtime.jit_post_compile_hook = hook.inner
        if hook.n:
            log.info("compiled %d kernels in %.0f s", hook.n, time.perf_counter() - hook.t0)


class _Recorder:
    """Every kernel the JIT compiles in this process, with what it takes to find it in another machine's cache."""

    def __init__(self):
        self.seen = {}
        self.configs = {}

    def __call__(self, *, key, repr, fn, compile, is_manual_warmup, already_compiled):
        from triton.compiler.compiler import ASTSource

        jit = fn.jit_function
        spec = json.loads(compile["specialization_data"])
        src = ASTSource(jit, compile["signature"], compile["constants"], compile["configs"][0])
        k = (src.hash(), json.dumps(_portable(spec["options"]), sort_keys=True))
        self.seen[k] = (_name(jit), src.name)
        self.configs[k] = _kernel_config(jit, compile["constants"], spec["options"])
        return None


def _pack(rec: _Recorder, out: Path) -> tuple[int, int]:
    import triton
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.cache import get_cache_manager

    target = _target()
    backend, env_vars = _backend(target), _env_vars()
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
    kernels, missing, dropped = [], 0, 0
    buf = {}
    for i, ((src, opts), (fn, name)) in enumerate(sorted(rec.seen.items())):
        cfg = rec.configs.get((src, opts))
        if not _picked({"fn": fn, "config": cfg}, tuning):
            dropped += 1
            continue
        options = json.loads(opts)
        group = get_cache_manager(_hash(src, options, backend, env_vars)).get_group(f"{name[:150]}.json")
        if group is None:   # the portable recipe does not give back the key this kernel was filed under
            missing += 1
            continue
        meta = f"{name[:150]}.json"
        files = sorted(f for f in group if f == meta or f.endswith((".cubin", ".hsaco")))
        d = f"k{i:05d}"
        for f in files:
            data = Path(group[f]).read_bytes()
            if f == meta:
                m = json.loads(data)
                m.pop("extern_libs", None)
                data = json.dumps(m).encode()
            buf[f"{d}/{f}"] = data
        kernels.append({"fn": fn, "src": src, "options": options, "meta": meta, "dir": d, "files": files,
                        **({"config": cfg} if cfg is not None else {})})
    if dropped:
        log.info("left out %d autotuner candidates that no pick names", dropped)
    man = {"format": FORMAT, "triton": triton.__version__, "target": target, "kernels": kernels, "tuning": tuning}
    _write(out, man, buf)
    return len(kernels), missing


def _write(out: Path, man: dict, buf: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:xz", preset=9 | lzma.PRESET_EXTREME) as tar:
        for name, data in [(MANIFEST, json.dumps(man).encode())] + sorted(buf.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))


def warm_blocks(eng) -> None:
    """Rounds of block-attention plans on a backbone that reads them (a family such as Sieve asks for them; the engine's
    own warm-up is causal), through every path a served round takes, so their kernels go into the bundle too: a
    prefixless plan of whole blocks and plans whose questions fork from block 0."""
    import random

    from ..formats.base import Plan, Read

    if not getattr(eng.backbone, "block_attention", False):
        return
    rng = random.Random(1)
    tok = lambda n: [rng.randrange(100, 30000) for _ in range(n)]
    plans = [Plan([], [tok(200), tok(220)], (Read(None),), attention="blocks"),
             Plan(tok(300), [tok(40), tok(25), tok(60)], (Read(None),), attention="blocks"),
             Plan(tok(1500), [tok(50)], (Read(None),), attention="blocks")]
    for mode in ("batch", "graphs", "packed"):
        eng.forward(plans, mode=mode)
    for p in plans:
        eng.forward([p], mode="alone")


def warm_requests(eng) -> None:
    """A few requests through the checkpoint's own family (its plans: forks, whole blocks, sets), whole and in rounds on
    every path a served round takes, at token counts of both parities, so a family's own attention layouts and the
    lengths its plans reach go into the bundle as well as the engine's causal warm-up; each one-question request also runs
    alone on the packed path, a micro-batch of one read (the last of an offline run often is), where backbones that
    compute their last layers only at the read rows run them on a single row. Requests the family turns down are
    skipped."""
    from ..api.schema import SystemOneRequest

    choice = {"type": "choice", "instructions": "Pick one.", "criteria": {"a": "first", "b": "second", "c": "third"}}
    asks = [{"q": choice}, {"q": choice, "n": {"type": "noul", "instructions": "Is it so?"},
                            "s": {"type": "score", "instructions": "How much?", "criteria": ["none", "some", "much"]}}]
    plans, alone = [], []
    for words in (61, 250, 1200):
        for questions in asks:
            try:
                req = SystemOneRequest.model_validate({"state": "word " * words, "questions": questions})
                eng.decide(req)
                plans.append(eng.format.plan(req))
            except ValueError:
                continue
            if questions is asks[0]:
                alone.append(plans[-1])
    if plans:
        for mode in ("batch", "graphs", "packed"):
            eng.forward(plans, mode=mode)
    for p in alone:
        eng.forward([p], mode="packed")


class _Calls:
    """Notes how the backbones call the kernels whose compiled forms follow a round's counts while models warm: the
    attention tile scan (`attention.block_ranges`) and the grouped expert products (kernels/moe.py), for `warm_counts`."""

    def __enter__(self):
        from .. import settings
        from ..kernels import attention, moe

        self.tiles, self.experts, ids = set(), {}, [None]
        self.real = [(attention, "block_ranges"), (moe, "align"), (moe, "grouped"), (moe, "grouped_fp8")]
        self.real = [(m, name, getattr(m, name)) for m, name in self.real]
        block_ranges, align, grouped, grouped_fp8 = (f for _, _, f in self.real)

        def held(x):   # rows of one tensor, columns of one, or vectors of their own
            own = x[0].untyped_storage().data_ptr() != x[1].untyped_storage().data_ptr()
            return (None if x[0].stride(0) not in (1, 4) else "apart" if own and x[0].stride(0) == 1
                    else "rows" if x[0].stride(0) == 1 else "columns")

        def noted_ranges(lo1, hi1, lo2, hi2, block_m, block_n=0, exact=False, groups=0, tiles_from=None):
            how = held((lo1, hi1)), None if tiles_from is None else held(tiles_from)
            if None not in how[:1 if tiles_from is None else 2]:
                self.tiles.add((block_m, block_n, bool(exact), bool(groups), *how, bool(settings.batch_invariant),
                                str(lo1.device)))
            return block_ranges(lo1, hi1, lo2, hi2, block_m, block_n, exact, groups, tiles_from)

        def noted_align(i, *args):
            ids[0] = i.dtype
            return align(i, *args)

        def note(a, w, scale, out, topk, by_token, weights):
            key = (tuple(w.shape), w.dtype, a.dtype, a.shape[1], out.dtype, topk, by_token,
                   None if weights is None else weights.dtype, ids[0])
            self.experts.setdefault(key, (w, scale))

        def noted_grouped(a, w, out, rows, blocks, used, topk, by_token, weights=None, tile=None):
            note(a, w, None, out, topk, by_token, weights)
            return grouped(a, w, out, rows, blocks, used, topk, by_token, weights, tile)

        def noted_fp8(a, w, w_scale, out, rows, blocks, used, topk, by_token, weights=None, tile=None):
            note(a, w, w_scale, out, topk, by_token, weights)
            return grouped_fp8(a, w, w_scale, out, rows, blocks, used, topk, by_token, weights, tile)

        for (m, name, _), f in zip(self.real, (noted_ranges, noted_align, noted_grouped, noted_fp8)):
            setattr(m, name, f)
        return self

    def __exit__(self, *exc):
        for m, name, f in self.real:
            setattr(m, name, f)


def warm_counts(calls: _Calls) -> None:
    """The kernels that take a round's counts, in every form Triton compiles them: a kernel differs with whether a count
    is one or a multiple of 16 and whether a vector starts on 16 bytes, all of which follow the rows and tokens of a
    packed round, and the expert products change their tiles with the round. A family or a round of other counts than
    the warm-up's would compile one while it answers, so each kind of call the models made runs at counts of every class."""
    import torch

    from .. import settings
    from ..kernels import attention, moe

    def vectors(T, how, device):
        if how == "apart":
            return [torch.zeros(T, dtype=torch.int32, device=device) for _ in range(4)]
        x = torch.zeros((4, T) if how == "rows" else (T, 4), dtype=torch.int32, device=device)
        return list(x if how == "rows" else x.T)

    was = settings.batch_invariant
    try:
        for bm, bn, exact, grouped, how, by, invariant, device in sorted(calls.tiles, key=str):
            settings.batch_invariant = invariant
            if grouped:
                shapes = [(T, g) for T in (1, 2, 3, 4, 16) for g in (1, 3, 15)]
            else:
                shapes = [(T, 0) for T in (1, 2, 3, 4, 16)] + [(k * bm + r, 0) for k in (1, 3, 15) for r in (1, 2, 4, 16)]
            for T, groups in shapes:
                attention.block_ranges(*vectors(T, how, device), bm, bn, exact, groups,
                                       None if by is None else vectors(T, by, device))
    finally:
        settings.batch_invariant = was
    for key, (w, scale) in calls.experts.items():
        (E, N, KD), _, dtype, width, out_dtype, K, by_token, weight_dtype, ids_dtype = key
        fp8, done = scale is not None, set()
        for T in (*range(1, 600), *range(28000 // K - 600, 28000 // K + 1200), *range(15000 // K, 15000 // K + 600)):
            n = T * K   # every tile the count picks, pairs and row blocks a multiple of 16 or not
            t = moe.tile(n, fp8, w.device)
            blocks_n = -(-(n + E * (t[0] - 1)) // t[0])
            kind = (t, n % 16 == 0, blocks_n % 16 == 0)
            if kind in done:
                continue
            done.add(kind)
            rows, blocks, used = moe.align(torch.zeros((T, K), dtype=ids_dtype, device=w.device), E, t[0])
            a = torch.zeros((T if by_token else n, width), dtype=dtype, device=w.device)
            out = torch.empty((n, N), dtype=out_dtype, device=w.device)
            weights = None if weight_dtype is None else torch.zeros(n, dtype=weight_dtype, device=w.device)
            if fp8:
                moe.grouped_fp8(a, w, scale, out, rows, blocks, used, K, by_token, weights, tile=t)
            else:
                moe.grouped(a, w, out, rows, blocks, used, K, by_token, weights, tile=t)


def export(args) -> None:
    """Warm each model in a fresh cache directory and pack every kernel that ran."""
    tmp = tempfile.mkdtemp(prefix="flide-kernels-")
    os.environ["TRITON_CACHE_DIR"] = tmp
    os.environ["TRITON_STORE_BINARY_ONLY"] = "1"
    os.environ["TRITON_DISABLE_LINE_INFO"] = "1"   # line tables name this machine's source paths
    import torch
    from triton import knobs

    from .engine import Engine

    if args.tuning:
        held = hold_picks(args.tuning)
        if not held:
            raise SystemExit(f"--tuning {args.tuning}: no bundle there for this GPU and Triton")
        log.info("%d autotuner picks held from %s", held, args.tuning)
    rec = _Recorder()
    knobs.runtime.jit_post_compile_hook = rec
    for model in args.model:
        t0 = time.perf_counter()
        opts = {"cuda_graphs": True, "quantize": args.quantize}
        if args.no_weights:
            opts["loader"] = "none"
        with _Calls() as calls:
            eng = Engine(model, format=args.format, device=args.device, dtype=args.dtype, capture=True,
                         session_cache_gb=1.0 if args.sessions else 0.0, backbone_options=opts)
            warm_blocks(eng)
            warm_requests(eng)
        n = len(rec.seen)
        warm_counts(calls)
        log.info("%s warm in %.1fs, %d kernels so far (%d of them at other counts)", model, time.perf_counter() - t0,
                 len(rec.seen), len(rec.seen) - n)
        del eng, calls
        gc.collect()
        torch.cuda.empty_cache()
    knobs.runtime.jit_post_compile_hook = None
    n, missing = _pack(rec, Path(args.out))
    out = Path(args.out)
    print(f"{out}: {n} kernels, {out.stat().st_size / 1e6:.1f} MB" + (f", {missing} not found in the cache" if missing else ""))
    marks = {str(Path.home()).encode(), b"site-packages", os.getcwd().encode()} - {b"/", b""}
    with tarfile.open(out, "r:*") as tar:
        named = [m.name for m in tar.getmembers() if m.isfile() and any(x in tar.extractfile(m).read() for x in marks)]
    if named:
        print(f"{len(named)} files name a path of this machine, e.g. {named[0]}")


def merge(args) -> None:
    """One bundle from several made with the same Triton on the same GPU target (kernels and picks united), without
    the autotuner candidates that none of the united picks names."""
    man, entries, seen = None, [], set()
    for path in args.bundles:
        with tarfile.open(path, "r:*") as tar:
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
                entries.append((e, {f: tar.extractfile(f"{e['dir']}/{f}").read() for f in e["files"]}))
            for fn, picks in m["tuning"].items():
                have = man["tuning"].setdefault(fn, [])
                have += [p for p in picks if p not in have]
    buf, dropped = {}, 0
    for e, files in entries:
        if not _picked(e, man["tuning"]):
            dropped += 1
            continue
        d = f"k{len(man['kernels']):05d}"
        for f, data in files.items():
            buf[f"{d}/{f}"] = data
        man["kernels"].append({**e, "dir": d})
    _write(Path(args.out), man, buf)
    print(f"{args.out}: {len(man['kernels'])} kernels" + (f", {dropped} autotuner candidates left out" if dropped else ""))


def pack(out: Path, paths, manifest: str = MANIFEST, solid: str = SOLID) -> int:
    """One archive of the bundles in `paths` (each for one GPU and Triton, none a solid archive), with the same kernel's
    files of every target side by side. `manifest` and `solid` are the names the package reads, for a package built under
    another name. Returns the kernels written."""
    bundles_, files, n = {}, [], 0
    for path in paths:
        with tarfile.open(path, "r:*") as tar:
            man = json.load(tar.extractfile(manifest))
            t = man["target"]
            name = f"{_arch(t)}-triton{man['triton']}"
            if name in bundles_:
                raise SystemExit(f"two bundles for {name}")
            bundles_[name] = man
            for e in man["kernels"]:
                for f in e["files"]:
                    files.append(((e["fn"].rpartition(":")[2], f, name, e["dir"]), f"{name}/{e['dir']}/{f}",
                                  tar.extractfile(f"{e['dir']}/{f}").read()))
                n += 1
    files.sort(key=lambda x: x[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    index = json.dumps({"format": SOLID_FORMAT, "bundles": bundles_}).encode()
    with tarfile.open(out, "w:xz", preset=9 | lzma.PRESET_EXTREME) as tar:
        for name, data in [(solid, index)] + [(x[1], x[2]) for x in files]:
            info_ = tarfile.TarInfo(name)
            info_.size = len(data)
            info_.mode = 0o644
            tar.addfile(info_, io.BytesIO(data))
    return n


def _arch(t: dict) -> str:
    """A target's GPU as bundles are named: sm90 for NVIDIA, gfx90a for AMD."""
    return f"sm{t['arch']}" if t["backend"] == "cuda" else str(t["arch"])


def info(args) -> None:
    for name, m in _index(Path(args.bundle)).items():
        t = m["target"]
        picks = sum(len(v) for v in m["tuning"].values())
        gpu = f"sm_{t['arch']}" if t["backend"] == "cuda" else t["arch"]
        print(f"{name + ': ' if name else ''}Triton {m['triton']}, {t['backend']} {gpu}: {len(m['kernels'])} "
              f"kernels, {picks} autotuner picks over {len(m['tuning'])} kernels")


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
    e.add_argument("--quantize", default=None, choices=["fp8", "int8"], help="the kernels of 8-bit serving")
    e.add_argument("--tuning", default=None, metavar="BUNDLES",
                   help="start the autotuners from these bundles' picks (files or directories, ':'-separated)")
    e.set_defaults(func=export)
    s = ks.add_parser("seed", help="file a bundle's kernels into this machine's kernel cache")
    s.add_argument("bundle")
    s.set_defaults(func=lambda a: print(f"{seed(a.bundle)} kernels added to {cache_dir()}"))
    m = ks.add_parser("merge", help="unite bundles made with the same Triton on the same GPU")
    m.add_argument("out")
    m.add_argument("bundles", nargs="+")
    m.set_defaults(func=merge)
    p = ks.add_parser("pack", help="one archive from the bundles of several GPUs and Triton versions")
    p.add_argument("out")
    p.add_argument("bundles", nargs="+")
    p.set_defaults(func=lambda a: print(f"{a.out}: {pack(Path(a.out), a.bundles)} kernels, "
                                        f"{Path(a.out).stat().st_size / 1e6:.1f} MB"))
    i = ks.add_parser("info", help="what a bundle holds")
    i.add_argument("bundle")
    i.set_defaults(func=info)

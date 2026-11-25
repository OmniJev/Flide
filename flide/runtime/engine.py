"""Engine: a checkpoint, its family and its backbone, and the packed forward between them."""
from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .. import backbones, formats, settings
from ..formats.base import FINAL, Plan, check_limits
from .pack import pack
from .schedule import schedule

log = logging.getLogger("flide.engine")



def resolve_model(name: str) -> str:
    """A local directory, or a Hub repository id downloaded (or found in the cache) as a snapshot; a LoRA adapter
    becomes the directory it runs from, its base checkpoint with the adapter (lora.view)."""
    from .. import lora

    if Path(name).is_dir():
        d = name
    else:
        from huggingface_hub import snapshot_download

        d = snapshot_download(name)
    return lora.view(d, resolve_model) if lora.is_adapter(d) else d


class FileConfig:
    """A config.json transformers has no class for (a model that ships its code, LFM3), its fields as attributes."""

    def __init__(self, d: dict):
        self.__dict__.update({k: FileConfig(v) if k.endswith("_config") and isinstance(v, dict) else v
                              for k, v in d.items()})

    def get_text_config(self):
        return getattr(self, "text_config", self)


def model_config(model_dir: str):
    import json

    import transformers

    try:
        # trust_remote_code=False refuses a checkpoint's own code at once instead of asking on the terminal
        return transformers.AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    except ValueError:   # a model type of its own code (trust_remote_code), which flide does not run
        return FileConfig(json.loads((Path(model_dir) / "config.json").read_text()))


SMALL_CARD = 25 * 2 ** 30
# requests of one round that share a start of this many tokens or more (in 64-token blocks) compute it once
MERGE_MIN_TOKENS = 512


def default_budget(total_bytes: int) -> int:
    """Tokens per micro-batch when the caller names none: 65536, and 16384 on a card of 24 GB or less, where a 9B
    model's 65536-token micro-batch does not fit beside its weights. `total_bytes` 0 (no CUDA device) keeps 65536."""
    return 16384 if 0 < total_bytes <= SMALL_CARD else 65536


def _total_memory(device) -> int:
    import torch

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_properties(device).total_memory


def _rooted(plan: Plan) -> Plan:
    """A causal text plan without a prefix as the same computation with one: the start its sequences share (each keeps
    at least the tokens its reads look at, and one token) becomes the prefix, so the flat graphs and the session cache
    take it (a family whose prompt is one stream read at several positions). Other plans as they are."""
    if (plan is None or plan.prefix or plan.attention != "causal" or plan.vision is not None or plan.positions is not None
            or plan.offset or not plan.suffixes):
        return plan
    need = 1
    for b in range(len(plan.suffixes)):
        for r in plan.branch_reads(b):
            for pos in r.positions:
                if pos >= 0:   # counted from the branch's start: a cut would move it
                    return plan
                need = max(need, -pos)
    s0 = plan.suffixes[0]
    cut = min(len(s) for s in plan.suffixes) - need
    n = 0
    while n < cut and all(s[n] == s0[n] for s in plan.suffixes[1:]):
        n += 1
    if n < 1:
        return plan
    import dataclasses

    return dataclasses.replace(plan, prefix=list(s0[:n]), suffixes=[s[n:] for s in plan.suffixes])


def _blocks(plan: Plan) -> bool:
    """Whether a plan reads any of its prefix both ways, attention "blocks" or "prefix" (formats.base.Plan); ValueError
    for an attention no backbone runs."""
    if plan.attention not in ("causal", "blocks", "prefix", "set"):
        raise ValueError(f"unknown attention {plan.attention!r}")
    if plan.attention == "prefix" and not plan.prefix:
        raise ValueError("attention \"prefix\" needs a prefix")
    if plan.attention == "set" and (plan.prefix or plan.segments is None or len(plan.segments) != len(plan.suffixes)):
        raise ValueError("attention \"set\" takes no prefix and one segments tuple per sequence")
    return plan.attention != "causal"


def _capability(plan: Plan) -> str:
    """The backbone or graphs attribute that says it runs the plan's attention."""
    return {"prefix": "prefix_attention", "set": "set_attention"}.get(plan.attention, "block_attention")


def _flag(plan: Plan):
    """The attention the flat graphs take: True for "blocks", "prefix" for "prefix" (sessions and other graphs take
    neither)."""
    return "prefix" if plan.attention == "prefix" else True


def _split(rows: list, counts: list[int] | None) -> list:
    """Graph head outputs laid out per branch and read: a row per branch, or `counts[b]` rows in branch b's one read."""
    if counts is None:
        return [[[row]] for row in rows]
    out, at = [], 0
    for n in counts:
        out.append([rows[at: at + n]])
        at += n
    return out


class _Kept:
    """Head outputs left on the device for a head with `on_device`: `tolist` hands back the tensor itself."""

    __slots__ = ("y",)

    def __init__(self, y):
        self.y = y

    def tolist(self):
        return self.y


def eager_tokens(model_type: str, backbone) -> int | None:
    """The length from which a request runs faster packed, at its exact length, than from the CUDA graphs, which pad it
    to their ladder and run its recurrence in fewer, longer programs (bench/path_cross.py on an H200: the dense Qwen3.5
    models of 9B and more); None where the graphs stay ahead at every length served."""
    if model_type not in ("qwen3_5", "qwen3_5_text"):
        return None
    layers = getattr(backbone, "layers", None)
    if layers is None:
        return None
    params = sum(p.numel() for p in layers.parameters())
    return 3072 if params >= 20e9 else 4096 if params >= 6e9 else None


def session_saves(snaps, mark: int | None) -> list:
    """The saves of a session run as a list of (position, (states, kv)), the earlier mark first. The continuation graphs
    give that list, or for a tail too long for them one (states, kv) at `mark`, or None when nothing is saved."""
    if snaps is None:
        return []
    if isinstance(snaps, list):
        return snaps
    return [(mark, snaps)] if mark else []


def session_floor(backbone) -> int:
    """The shortest prefix the session cache saves when `session_min_tokens` is not given. A prefix is worth saving when
    computing it again takes about as long as 2048 tokens on a model with 2B parameters in its layers (the floor the
    small models are served with), so the floor falls with the size of the layers, in strides of 256 down to one stride
    from 16B on; 2048 where the layers are not at hand (tensor parallel)."""
    layers = getattr(backbone, "layers", None)
    if layers is None:
        layers = getattr(getattr(backbone, "lm", None), "layers", None)
    params = sum(p.numel() for p in layers.parameters()) if layers is not None else 0
    if params <= 0:
        return 2048
    return max(256, min(2048, int(2048 * 2e9 / params) // 256 * 256))


def _gib() -> tuple[float, float, float]:
    """GPU memory in GiB: allocated by torch, reserved by its caching allocator (graph pools included), used on the
    device by every process."""
    import torch

    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    torch.cuda.synchronize()
    torch.cuda.empty_cache()   # cached free blocks would make the differences between snapshots meaningless
    free, total = torch.cuda.mem_get_info()
    g = 2 ** 30
    return torch.cuda.memory_allocated() / g, torch.cuda.memory_reserved() / g, (total - free) / g


def refuse_invariant_quantized(batch_invariant: bool, backbone_options: dict | None) -> None:
    """`--quantize` and `--batch-invariant` cannot both hold: the quantized products come from a scaled matmul whose
    split depends on the batch, so the promise `--batch-invariant` makes does not survive it. Refused rather than
    silently answering with batch-dependent numbers."""
    quantize = (backbone_options or {}).get("quantize")
    if batch_invariant and quantize:
        raise ValueError(
            f"--quantize {quantize} cannot be combined with --batch-invariant: the quantized products come from a "
            "scaled matmul whose split depends on the batch, so the same question can answer differently in another "
            "round. Drop one of the two.")


class Engine:
    def __init__(self, model: str, format: str | None = None, device: str = "cuda:0", dtype: str = "bfloat16",
                 budget: int | None = None, backbone_options: dict | None = None, capture: bool | int = 4096,
                 session_cache_gb: float = 0.0, session_min_tokens: int | None = None, batch_invariant: bool = False,
                 **format_options):
        """`batch_invariant`: the batch-invariant products for every engine of the process (`settings`)."""
        refuse_invariant_quantized(batch_invariant, backbone_options)
        if batch_invariant:
            settings.batch_invariant = True
        from ..kernels import launcher
        from .env import check_cuda

        if str(device).startswith("cuda"):
            check_cuda()
        launcher.install()
        from .kcache import cache_dir, no_bundle, progress, seed, tune

        seed()
        missing = no_bundle()
        if missing:   # said on every start, a warm Triton cache or not, so a slow first start never looks hung
            log.info("no precompiled kernels ship for %s: kernels not yet in %s compile on first use, so the first start "
                     "on this machine takes a few minutes and later ones read them from the cache", missing, cache_dir())
        progress(True)   # kernels a bundle lacks (or all, without one) are counted as they compile
        t0 = time.perf_counter()
        self.model_dir = resolve_model(model)
        config = model_config(self.model_dir)
        family = formats.detect(self.model_dir, config, format)
        self.model_dir = family.run_dir(self.model_dir)
        if family.backbones and config.model_type not in family.backbones:
            raise ValueError(f"format {family.name!r} runs on {family.backbones}, {self.model_dir} is {config.model_type}")
        self.format = family(self.model_dir, **format_options)
        options = {**getattr(self.format, "backbone_options", family.backbone_options), **(backbone_options or {})}
        if family.merge_lora:
            options["merge_lora"] = True
        tp_size = int(options.pop("tp", 1) or 1)
        if tp_size > 1:   # rank 0 here, the other ranks in worker processes (runtime/tp.py)
            if session_cache_gb == "auto" or session_cache_gb > 0:
                raise ValueError("tensor parallel runs without the session cache")
            from . import tp

            self.backbone = tp.launch(config.model_type, self.model_dir, device, dtype, tp_size, options)
        else:
            self.backbone = backbones.get(config.model_type).load(self.model_dir, device=device, dtype=dtype, **options)
        tune()   # autotuner picks from the kernel bundles, now that the kernels' modules are loaded
        needs = getattr(self.format, "attention", "causal")
        if needs != "causal" and not getattr(self.backbone, "prefix_attention" if needs == "prefix" else "block_attention",
                                             False):
            raise ValueError(f"{self.model_dir} reads with block attention, which the {config.model_type} backbone "
                             "does not run")
        midloop = getattr(self.format, "midloop", None)
        if midloop:   # layers run again inside the pass (the checkpoint's recurrent depth)
            if tp_size > 1 or not hasattr(self.backbone, "set_midloop"):
                raise ValueError(f"{self.model_dir} runs layers again (midloop), which the {config.model_type} backbone "
                                 "does not" + (" under tensor parallelism" if tp_size > 1 else ""))
            self.backbone.set_midloop(midloop)
        self.heads = self.format.load_heads(self.backbone)
        self.eager_tokens = eager_tokens(config.model_type, self.backbone) if tp_size == 1 else None
        mem = _gib()
        self.budget = budget or default_budget(_total_memory(device))
        self._budget_given = budget is not None
        if budget is None and self.budget != 65536:
            log.info("budget %d tokens per micro-batch on a card of 24 GB or less (--budget sets it)", self.budget)
        self.sessions = None
        auto = session_cache_gb == "auto"   # sized from the memory left once the model is up and warm
        if (auto or session_cache_gb > 0) and not getattr(self.backbone, "session_states", True):
            log.info("%s keeps no session states yet: the session cache is off", config.model_type)
            auto, session_cache_gb = False, 0.0
        if auto or session_cache_gb > 0:
            from .session import SessionCache

            if session_min_tokens is None:
                session_min_tokens = session_floor(self.backbone)
                log.info("session cache saves prefixes of %d tokens or more (--session-min-tokens sets it)",
                         session_min_tokens)
            self.sessions = SessionCache(0 if auto else int(session_cache_gb * 2 ** 30), min_tokens=session_min_tokens)
        graphs = getattr(self.backbone, "request_graphs", lambda: None)()
        tune()
        if graphs is not None and capture is not False:
            limit = None if capture is True else capture
            # flat graphs take text and media requests alone or together; the sequence, branch and continuation
            # graphs are for sessions (and requests the flat graphs do not take, captured when first needed)
            if hasattr(graphs, "capture_flat"):
                graphs.capture_flat(max_tokens=limit)
            if self.sessions is not None or not hasattr(graphs, "capture_flat"):
                graphs.capture_all(max_tokens=limit)
            if self.sessions is not None and hasattr(graphs, "capture_sessions"):
                graphs.capture_sessions()
            if self.sessions is not None and hasattr(graphs, "capture_sflat"):
                graphs.capture_sflat()
        n_graphs = len(getattr(graphs, "graphs", None) or {}) if graphs is not None else 0
        mem_graphs = _gib()
        self._capture_on = capture is not False
        if self.sessions is not None and not auto:
            if hasattr(self.backbone, "kv_arena"):
                self._attach_store(self.sessions.budget)
            else:
                self.sessions.attach_prefix_store(self.backbone.device)
        if capture is not False:
            tune()
            self.warmup()
        if auto:
            self.sessions.budget = self._auto_cache()
        progress(False)
        log.info("loaded %s as %s on %s in %.1fs", self.model_dir, family.name, config.model_type,
                 time.perf_counter() - t0)
        end = _gib()
        log.info("memory: model %.2f GiB, graphs %.2f GiB (%d), sessions and work space %.2f GiB, reserved %.2f GiB, "
                 "device used %.2f GiB",
                 mem[0], mem_graphs[1] - mem[1], n_graphs, end[1] - mem_graphs[1], end[1], end[2])

    def warmup(self) -> None:
        """One packed forward over made-up requests, so the kernels of the packed path (requests the graphs do not
        take, rounds of several requests) are compiled before the first real one instead of stalling it."""
        import random

        import torch

        from ..formats.base import Read

        t0 = time.perf_counter()
        rng = random.Random(0)
        tok = lambda n: [rng.randrange(100, 30000) for _ in range(n)]
        plans = [Plan(tok(300), [tok(40), tok(25), tok(60)], (Read(None),)), Plan(tok(1500), [tok(50)], (Read(None),)),
                 Plan(tok(70), [tok(9)], (Read(None),))]
        full = self.sessions is not None and hasattr(self.backbone, "kv_arena")
        saved, self.sessions = self.sessions, None
        try:
            self.run_plans(plans)
            if full:
                self._warm_budget(tok)
            if saved is not None:   # the level-by-level path that saves and resumes session states
                from .pack import Job
                from .session import Entry

                p = plans[1]
                b = pack([Job(p.prefix, p.suffixes, [p.reads], marks=(1024,))], self.backbone.device)
                self.backbone.forward(b)
                entry = Entry(p.prefix[:1024], b.saved[0]["states"], b.saved[0]["kv"])
                b = pack([Job(p.prefix, p.suffixes, [p.reads], start=1024, resume=entry)], self.backbone.device)
                self.backbone.forward(b)
        except NotImplementedError:
            pass
        finally:
            self.sessions = saved
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        log.info("packed path warmed up in %.1fs", time.perf_counter() - t0)

    def _warm_budget(self, tok) -> None:
        """A packed micro-batch as large as the budget, run before the session store takes the memory left, so its work
        space counts. When it does not fit beside the model, its graphs and the store, a budget the caller did not set
        halves until it does."""
        import torch

        from ..formats.base import Read

        while True:
            try:
                self.run_plans([Plan(tok(1000), [tok(20)] * 4, (Read(None),)) for _ in range(self.budget // 1080)])
                return
            except torch.OutOfMemoryError:
                if self._budget_given or self.budget <= 4096:
                    raise
            gc.collect()
            torch.cuda.empty_cache()
            self.budget //= 2
            log.info("budget %d tokens per micro-batch: a larger one does not fit beside the model, its graphs and the "
                     "session cache (--budget sets it)", self.budget)

    def run_plans(self, plans: Sequence[Plan | None], budget: int | None = None,
                  log_fn: Callable[[str], None] | None = None, warmup: int = 0) -> tuple[list, dict]:
        """Head outputs for every plan: out[plan][branch][read] is a list of rows, one per read position."""
        import torch

        budget = budget or self.budget
        plans = [_rooted(p) for p in plans]
        for p in plans:
            if p is not None and _blocks(p) and not getattr(self.backbone, _capability(p), False):
                raise ValueError(f"this backbone does not run attention {p.attention!r}")
            if p is not None and p.vision is not None and p.positions is None:
                p.positions, p.offset = self._media_positions(p)
        outputs: list[Any] = [None if p is None else [None] * len(p.suffixes) for p in plans]
        sessions = self._sessions_for(plans)
        if sessions and self.sessions.store is not None:   # no save of this call evicts what another of its jobs reads
            self.sessions.pinned.update(id(s["resume"]) for s in sessions.values() if s["resume"] is not None)
        batches = schedule(plans, budget, sessions)
        tokens = sum(b.tokens for b in batches)
        if log_fn:
            log_fn(f"forward: {len(batches)} micro-batches, {tokens} tokens, budget {budget}")

        def launch(mb):
            """Queue a micro-batch on the GPU; its head outputs come back to the host without waiting."""
            batch = pack([p.job for p in mb.pieces], self.backbone.device)
            shape = budget
            while shape < mb.tokens:
                shape *= 2
            store = getattr(self.sessions, "store", None) or getattr(self.sessions, "pstore", None) \
                if self.sessions is not None else None
            if store is not None and batch.marks:
                # the store's rows for every save point, before the forward writes them (a save without room is skipped)
                keep = [e for _, e, _ in batch.resumes]
                for j in list(batch.marks):
                    m = max(batch.marks[j])
                    need = self.backbone.session_rows(m)
                    rows = (self.sessions.reserve(need, keep=keep) if getattr(self.sessions, "store", None) is not None
                            else self.sessions.reserve_prefix(need, dtype=self.backbone.dtype))
                    if rows is None:
                        del batch.marks[j]
                    else:
                        batch.saved[j] = {"states": {m: {}}, "kv": rows}
            try:
                hidden = self.backbone.forward(batch, pad_to=shape)
            except BaseException:
                for saved in batch.saved.values():
                    getattr(saved["kv"], "free", lambda: None)()
                raise
            if self.sessions is not None and batch.saved:
                from .session import Entry

                for j, saved in batch.saved.items():
                    pre = mb.pieces[j].job.prefix
                    if self.sessions.store is not None:   # the tokens it keeps, so an agent's next save replaces it
                        pre = pre[:max(saved["states"])]
                    self.sessions.insert(Entry(pre, saved["states"], saved["kv"]), parent=mb.pieces[j].job.resume)
            ys = []
            for g in batch.groups:
                h = hidden[g.layer][g.start: g.end]
                y = self.heads[g.head](h) if g.head is not None else h
                ys.append(self._host(y, g.head))
            done_ev = None
            if self.backbone.device.type == "cuda":
                done_ev = torch.cuda.Event()
                done_ev.record()
            return mb, batch.groups, ys, done_ev

        def collect(item):
            mb, groups, ys, done_ev = item
            if done_ev is not None:
                done_ev.synchronize()
            branch_rows: dict[int, dict[int, list]] = {}
            for g, y in zip(groups, ys):
                rows = y.tolist()
                at = 0
                for br, r, n in g.owners:
                    branch_rows.setdefault(br, {})[r] = rows[at: at + n]
                    at += n
            br = 0
            for piece in mb.pieces:
                for b in piece.branches:
                    reads = branch_rows.get(br, {})
                    outputs[piece.request][b] = [reads[r] for r in range(len(reads))]
                    br += 1

        if warmup:
            t0 = time.perf_counter()
            for mb in batches[len(batches) // 2: len(batches) // 2 + warmup]:
                collect(launch(mb))
            torch.cuda.synchronize()
            if log_fn:
                log_fn(f"warmup: {warmup} micro-batches in {time.perf_counter() - t0:.1f}s (not timed below)")
        t_start = time.perf_counter()
        step = max(1, len(batches) // 10)
        done = 0
        pending = None
        # the next micro-batch is queued before the last one's outputs are read, so the GPU never waits for the host
        for k, mb in enumerate(batches):
            item = launch(mb)
            if pending is not None:
                collect(pending)
            pending = item
            done += mb.tokens
            if log_fn and (k + 1) % step == 0:
                log_fn(f"forward {k + 1}/{len(batches)} micro-batches, {done / max(tokens, 1):.0%} of tokens")
        if pending is not None:
            collect(pending)
        if self.sessions is not None:
            self.sessions.pinned.clear()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stats = {"tokens": tokens, "micro_batches": len(batches), "forward_s": time.perf_counter() - t_start}
        return outputs, stats

    def _kept(self, head) -> bool:
        return head is not None and getattr(self.heads[head], "on_device", False)

    def _host(self, y, head):
        """Head outputs copied to the host without waiting, or left on the device for a head with `on_device`."""
        return _Kept(y) if self._kept(head) else y.float().to("cpu", non_blocking=True)

    @staticmethod
    def _spots(plan: Plan):
        """Where a plan of one branch, read once after the final layer at positions other than its last token, is read:
        positions in its one sequence (the prefix, then the branch); else None."""
        if len(plan.suffixes) != 1:
            return None
        read = plan.branch_reads(0)
        if len(read) != 1 or read[0].layer is not FINAL or read[0].positions == (-1,):
            return None
        L = len(plan.prefix) + len(plan.suffixes[0])
        at = tuple(L + p if p < 0 else p for p in read[0].positions)
        return at if all(0 <= a < L for a in at) else None

    @staticmethod
    def _branch_spots(plan: Plan):
        """Where a plan of several branches, each read once after the final layer by the same head, is read when some
        branch is read at positions other than its last token: per branch, positions in it; else None."""
        if len(plan.suffixes) < 2:
            return None
        reads = [plan.branch_reads(b) for b in range(len(plan.suffixes))]
        if any(len(r) != 1 or r[0].layer is not FINAL for r in reads) or len({r[0].head for r in reads}) != 1:
            return None
        if all(r[0].positions == (-1,) for r in reads):
            return None
        out = []
        for s, r in zip(plan.suffixes, reads):
            at = tuple(len(s) + p if p < 0 else p for p in r[0].positions)
            if not all(0 <= a < len(s) for a in at):
                return None
            out.append(at)
        return tuple(out)

    def _counts(self, plan: Plan) -> list[int] | None:
        """Graph head outputs per branch of a plan read at several positions (`_spots`, `_branch_spots`, attention
        "set"), else None."""
        if plan.attention == "set":
            return [len(plan.branch_reads(b)[0].positions) for b in range(len(plan.suffixes))]
        spots = self._spots(plan)
        if spots is not None:
            return [len(spots)]
        spots = self._branch_spots(plan)
        return None if spots is None else [len(s) for s in spots]

    def _layout(self, plan: Plan, rows: list) -> list:
        """Graph head outputs per branch and read: a row per branch, or every row of a branch in its one read when the
        plan is read at several positions (`_counts`)."""
        return _split(rows, self._counts(plan))

    def _eager(self, plan: Plan) -> bool:
        """A plan run packed rather than from the graphs: a state and its branches of `eager_tokens` or more that the
        session cache does not take (the session graphs resume it, which saves far more), not under the batch-invariant
        products. Sequences of their own (no prefix) keep the graphs: each is short however many there are."""
        if self.eager_tokens is None or settings.batch_invariant or not plan.prefix or plan.tokens < self.eager_tokens:
            return False
        return not (self.sessions is not None and len(plan.prefix) >= self.sessions.min_tokens and self._session_ok(plan))

    def _session_ok(self, plan: Plan) -> bool:
        """Sessions take text-only plans whose reads all fall inside the branches."""
        if self.sessions is None or plan.vision is not None or plan.positions is not None or plan.offset:
            return False
        if not getattr(self.backbone, "session_states", True):
            return False
        if getattr(self.backbone, "midloop", None) is not None:
            return False
        if _blocks(plan):
            return False
        if not plan.prefix:
            return False
        shortest = min(len(s) for s in plan.suffixes)
        return all(p < 0 and -p <= shortest for b in range(len(plan.suffixes)) for r in plan.branch_reads(b)
                   for p in r.positions)

    def cost(self, plan: Plan) -> int:
        """Tokens a plan computes: all of them, less the start of its prefix a saved session state covers."""
        plan = _rooted(plan)
        if self.sessions is None or not self._session_ok(plan):
            return plan.tokens
        return plan.tokens - self.sessions.peek(plan.prefix)

    def _sessions_for(self, plans) -> dict | None:
        """Per plan index: where its prefix resumes from and where to save states."""
        if self.sessions is None:
            return None
        out = {}
        for i, p in enumerate(plans):
            if p is None or not self._session_ok(p):
                continue
            entry, start = self.sessions.match(p.prefix)
            marks = self.sessions.marks(len(p.prefix), start)
            if start or marks:
                out[i] = {"start": start, "resume": entry, "marks": marks}
        return out

    def _auto_cache(self) -> int:
        """The "auto" session cache's bytes, held at once (`session.reserve`), and said in the log."""
        import torch

        from .session import auto_bytes, reserve

        dev = self.backbone.device
        if torch.device(dev).type != "cuda":
            return 0
        torch.cuda.synchronize(dev)
        free, total = torch.cuda.mem_get_info(dev)
        n = auto_bytes(free, total, torch.cuda.memory_reserved(dev), torch.cuda.memory_allocated(dev),
                       torch.cuda.max_memory_allocated(dev))
        log.info("session cache %.2f GiB (auto)%s", n / 2 ** 30, "" if n else ": too little memory left, so it saves nothing")
        if n and hasattr(self.backbone, "kv_arena"):
            return self._attach_store(n)
        if n and self.sessions is not None:
            self.sessions.budget = n
            return self.sessions.attach_prefix_store(dev)
        reserve(n, dev)
        return n

    def _attach_store(self, nbytes: int) -> int:
        """The session cache's store in `nbytes` (a backbone whose saved state is its keys and values alone: the
        entries keep rows of it, and its graphs read and write them in place). Returns the budget it gives."""
        store = self.backbone.kv_arena(nbytes)
        self.sessions.attach(store)
        graphs = getattr(self.backbone, "request_graphs", lambda: None)()
        if graphs is not None and hasattr(graphs, "set_store"):
            graphs.set_store(store, self.sessions, capture=self._capture_on)
        return self.sessions.budget

    def _graph_hidden(self, plan: Plan, lazy: bool = True):
        """Head outputs at the last token of every branch from CUDA graphs (still on the device): a flat graph (one
        replay), else the sequence and branch graphs, or the session graphs for long prefixes the cache takes; None when
        the plan or its shape does not fit them (reads other than the last token after the final layer, or lengths
        beyond the graph ladders). `lazy=False` uses sequence and branch graphs only when they were captured already."""
        graphs = getattr(self.backbone, "request_graphs", lambda: None)()
        set_media = plan.attention == "set" and getattr(graphs, "set_media", False)   # pictures heading set sequences
        if graphs is None or (plan.vision is not None and not graphs.media and not set_media) or self._eager(plan):
            return None
        if plan.attention == "set":   # one replay of all its sequences, read at every slot, on graphs that run "set"
            reads = [plan.branch_reads(b) for b in range(len(plan.suffixes))]
            if (not getattr(graphs, "set_attention", False) or (plan.vision is not None and not set_media)
                    or any(len(r) != 1 or r[0].layer is not FINAL or r[0].head != reads[0][0].head for r in reads)):
                return None
            media = {} if plan.vision is None else {"media": plan.vision}
            hidden = graphs.run_set(plan.suffixes, plan.segments, [r[0].positions for r in reads], lazy=lazy, **media)
            if hidden is None:
                return None
            head = reads[0][0].head
            return self.heads[head](hidden) if head is not None else hidden
        if not plan.prefix:   # sequences of their own: flat graphs under block attention (`_alone_items`), else none
            items = self._alone_items(plan, graphs) if hasattr(graphs, "run_flat") else None
            if items is None:
                return None
            hs = []
            for sel in self._flat_rounds(graphs, items):
                h = graphs.run_flat([items[k] for k in sel])
                if h is None:
                    return None
                hs.append(h)
            import torch

            hidden = hs[0] if len(hs) == 1 else torch.cat(hs)
            head = plan.branch_reads(0)[0].head
            return self.heads[head](hidden) if head is not None else hidden
        blocks = _blocks(plan)
        if blocks and not getattr(graphs, _capability(plan), False):
            return None
        if self._branch_spots(plan) is not None:   # flat graphs only
            item = self._flat_item(plan, graphs) if hasattr(graphs, "run_flat") else None
            hidden = None if item is None else graphs.run_flat([item])
            if hidden is None:
                return None
            head = plan.branch_reads(0)[0].head
            return self.heads[head](hidden) if head is not None else hidden
        reads = {plan.branch_reads(b) for b in range(len(plan.suffixes))}
        if len(reads) != 1:
            return None
        (read,) = reads
        spots = self._spots(plan)
        if len(read) != 1 or read[0].layer is not FINAL or read[0].positions != (-1,) and spots is None:
            return None
        if (spots is None and self.sessions is not None
                and (hasattr(graphs, "run_session") or hasattr(graphs, "run_sflat"))
                and len(plan.prefix) >= self.sessions.min_tokens and self._session_ok(plan)):
            # long text prefixes resume from and save session states
            entry, start = self.sessions.match(plan.prefix)
            marks = self.sessions.marks(len(plan.prefix), start)
            mark = marks[-1] if marks else None
            mark2 = marks[0] if len(marks) > 1 else None
            out = None
            if hasattr(graphs, "run_sflat"):
                out = graphs.run_sflat([(plan.prefix, plan.suffixes, entry, start, mark)])
                if out is not None:
                    out = (out[0], out[1][0])   # a list of (position, snapshot), the earlier mark first
            if out is None and hasattr(graphs, "run_session"):
                out = graphs.run_session(plan.prefix, plan.suffixes, entry, start, mark, mark2)
            if out is None:
                return None
            hidden, snaps = out
            for pos, snap in session_saves(snaps, mark):
                from .session import Entry

                self.sessions.insert(Entry(plan.prefix[:pos], *snap), parent=entry)
            head = read[0].head
            return self.heads[head](hidden) if head is not None else hidden
        item = self._flat_item(plan, graphs) if hasattr(graphs, "run_flat") else None
        if item is not None:
            hidden = graphs.run_flat([item])
            if hidden is not None:
                head = read[0].head
                return self.heads[head](hidden) if head is not None else hidden
        if spots is not None or blocks or not graphs.fits(len(plan.prefix), [len(s) for s in plan.suffixes]):
            return None
        if plan.vision is not None:
            if not hasattr(graphs, "run_media"):
                return None
            if plan.positions is None:
                plan.positions, plan.offset = self.backbone.vision_positions(plan.prefix, plan.vision)
            hidden = graphs.run_media(plan.prefix, plan.suffixes, plan.vision, plan.positions, plan.offset, lazy=lazy)
        else:
            hidden = graphs.run(plan.prefix, plan.suffixes, lazy=lazy)
        if hidden is None:
            return None
        head = read[0].head
        return self.heads[head](hidden) if head is not None else hidden

    def _flat_item(self, plan: Plan, graphs):
        """(prefix, branches, media) of a plan the flat graphs take, else None: every branch read at its last token
        after the final layer by the same head, text or media the backbone embeds, not one the session cache takes. A
        plan with block attention gets a fourth field, True, on graphs with `block_attention`; a plan read at several
        positions of its one branch (`_spots`) or of its branches (`_branch_spots`), on graphs with `read_spots`, a
        fourth (block attention) and a fifth (the positions)."""
        if not plan.suffixes or not plan.prefix or self._eager(plan):
            return None
        blocks = _blocks(plan)
        if blocks and not getattr(graphs, _capability(plan), False):
            return None
        bspots = self._branch_spots(plan)
        if bspots is not None:   # the session cache reads last tokens only
            if not getattr(graphs, "read_spots", False):
                return None
            tail = (_flag(plan) if blocks else False, bspots)
        else:
            reads = {plan.branch_reads(b) for b in range(len(plan.suffixes))}
            if len(reads) != 1:
                return None
            (read,) = reads
            spots = self._spots(plan)
            if spots is not None and (settings.batch_invariant or not getattr(graphs, "read_spots", False)):
                return None
            if len(read) != 1 or read[0].layer is not FINAL or read[0].positions != (-1,) and spots is None:
                return None
            if self.sessions is not None and len(plan.prefix) >= self.sessions.min_tokens and self._session_ok(plan):
                return None
            tail = (_flag(plan) if blocks else False, spots) if spots is not None else (_flag(plan),) if blocks else ()
        if plan.vision is None:
            if plan.positions is not None or plan.offset:
                return None
            return (plan.prefix, plan.suffixes, None) + tail
        if not graphs.media:
            return None
        if plan.positions is None:
            plan.positions, plan.offset = self.backbone.vision_positions(plan.prefix, plan.vision)
        return (plan.prefix, plan.suffixes, (plan.vision, plan.positions, plan.offset)) + tail

    def _alone_items(self, plan: Plan, graphs):
        """Flat graph items of a plan without a prefix under block attention, one per branch: its whole sequence as the
        prefix, a copy of its last token as the branch, read at the sequence's last token (on graphs with
        `read_spots` or `alone_reads`); None when the plan is not one of those or reads elsewhere."""
        if plan.prefix or not _blocks(plan) or plan.attention == "set" or settings.batch_invariant:
            return None
        if not ((getattr(graphs, "read_spots", False) or getattr(graphs, "alone_reads", False))
                and getattr(graphs, "block_attention", False)):
            return None
        if plan.vision is not None and not graphs.media:
            return None
        reads = {plan.branch_reads(b) for b in range(len(plan.suffixes))}
        if len(reads) != 1:
            return None
        (read,) = reads
        if len(read) != 1 or read[0].layer is not FINAL or read[0].positions != (-1,):
            return None
        if plan.vision is None:
            return [(s, [s[-1:]], None, True, (len(s) - 1,)) for s in plan.suffixes]
        if plan.positions is None:
            plan.positions, plan.offset = self._media_positions(plan)
        return [(s, [s[-1:]], (plan.vision, pos, off), True, (len(s) - 1,))
                for s, pos, off in zip(plan.suffixes, plan.positions, plan.offset)]

    def _media_positions(self, plan: Plan):
        """Rotary positions of a plan holding media: its prefix's [3, P] and the offset after it; for a plan without a
        prefix, a list of each per sequence, the media in the head they share (the token types `vision` gives cover
        that head, the rest is text)."""
        if plan.prefix:
            return self.backbone.vision_positions(plan.prefix, plan.vision)
        types = plan.vision.get("mm_token_type_ids")
        pos, off = [], []
        for s in plan.suffixes:
            v = plan.vision
            if types is not None:
                head = types.reshape(1, -1)
                v = dict(v, mm_token_type_ids=head.new_zeros(1, len(s)))
                v["mm_token_type_ids"][:, :head.shape[1]] = head
            p, o = self.backbone.vision_positions(s, v)
            pos.append(p)
            off.append(o)
        return pos, off

    @staticmethod
    def _flat_rounds(graphs, items) -> list[list[int]]:
        """Indices of flat graph items cut into rounds, in order, as many per replay as fit."""
        rounds, cur, size = [], [], None
        whole = getattr(graphs, "whole", lambda item: False)   # a request read at every position runs alone
        for k, item in enumerate(items):
            cost = graphs.flat_cost(item)
            grown = cost if size is None else tuple(x + y for x, y in zip(size, cost))
            if cur and (graphs.flat_rung(len(cur) + 1, *grown) is None or whole(item) or whole(items[cur[0]])):
                rounds.append(cur)
                cur, grown = [], cost
            cur.append(k)
            size = grown
        if cur:
            rounds.append(cur)
        return rounds

    def _run_flat(self, plans, skip=()) -> dict:
        """Plans of a round through flat graphs: as many per replay as fit, in order (a plan without a prefix under block
        attention as one item per branch, `_alone_items`); plans in `skip` are left out."""
        graphs = getattr(self.backbone, "request_graphs", lambda: None)()
        if graphs is None or not hasattr(graphs, "run_flat"):
            return {}
        entries = []
        for i, p in enumerate(plans):
            if i in skip:
                continue
            item = self._flat_item(p, graphs)
            if item is not None:
                entries.append((i, item))
            else:
                entries.extend((i, it) for it in self._alone_items(p, graphs) or ())
        parts, failed = {}, set()
        for sel in self._flat_rounds(graphs, [it for _, it in entries]):
            h = graphs.run_flat([entries[k][1] for k in sel])
            if h is None:
                failed.update(entries[k][0] for k in sel)
                continue
            at = 0
            for k in sel:
                i, item = entries[k]
                n = len(plans[i].suffixes) if len(item) < 5 else len(item[4]) if len(item[1]) == 1 else sum(map(len, item[4]))
                parts.setdefault(i, []).append(h[at: at + n])
                at += n
        import torch

        out = {}
        for i, hs in parts.items():
            if i in failed:
                continue
            y = hs[0] if len(hs) == 1 else torch.cat(hs)
            head = plans[i].branch_reads(0)[0].head
            out[i] = self.heads[head](y) if head is not None else y
        return out

    def _sflat_reads_ok(self, p: Plan) -> bool:
        """A plan the session flat graphs can read: text-only, every branch read once, at its last token, after the
        final layer, by the same head."""
        if not p.suffixes or not self._session_ok(p):
            return False
        reads = {p.branch_reads(b) for b in range(len(p.suffixes))}
        if len(reads) != 1:
            return False
        (read,) = reads
        return len(read) == 1 and read[0].layer is FINAL and read[0].positions == (-1,)

    def _merge_groups(self, plans, skip=()) -> dict[int, tuple[int, int]]:
        """Plans of a batch round that share their first MERGE_MIN_TOKENS tokens or more, in whole 64-token blocks:
        {plan: (carrier, length)}. The carrier (the first plan of its group) runs first and saves its state at that
        length, and the other plans of the group resume from it there, so the shared start is computed once; the length
        is the one that saves the most tokens, and plans that share less wait for a group of their own. Only with
        the session cache (its graphs carry the states). Under the batch-invariant products a merged request keeps the
        bits it gets alone, with or without the session cache: the cut falls on a 64-token block, as a request's own
        save points do (bench/merge_check.py --invariant)."""
        graphs = getattr(self.backbone, "request_graphs", lambda: None)()
        if self.sessions is None or graphs is None or not hasattr(graphs, "run_sflat"):
            return {}
        from .session import BLOCK, _Hashes

        cand = [i for i, p in enumerate(plans) if i not in skip and self._sflat_reads_ok(p)]
        if len(cand) < 2:
            return {}
        chains = {i: _Hashes(plans[i].prefix).blocks for i in cand}
        out: dict[int, tuple[int, int]] = {}
        for a in cand:
            if a in out:
                continue
            members = []
            for b in cand:
                if b == a or b in out:
                    continue
                ca, cb = chains[a], chains[b]
                k = 0
                while k + 1 < min(len(ca), len(cb)) and ca[k + 1] == cb[k + 1]:
                    k += 1
                n = k * BLOCK   # both prefixes run on past it, so each has tokens of its own after the shared start
                if n >= MERGE_MIN_TOKENS and n < len(plans[a].prefix) and n < len(plans[b].prefix) \
                        and plans[a].prefix[:n] == plans[b].prefix[:n]:
                    members.append((b, n))
            if members:   # the cut that saves the most tokens: the length times the members that share that much
                n = max({m for _, m in members}, key=lambda c: (c * sum(m >= c for _, m in members), c))
                out[a] = (a, n)
                out.update((b, (a, n)) for b, m in members if m >= n)
        return out

    def _run_sflat(self, plans, skip, merge=None) -> dict:
        """Plans of a round that the session cache takes (resume from a saved state, save their own) through session
        flat graphs, as many per replay as fit, in order; their save points go into the cache. `merge` (from
        `_merge_groups`): the carriers run in the first replays with their save point at the shared length, and the
        other plans of their groups then resume from the carrier's saved state."""
        graphs = getattr(self.backbone, "request_graphs", lambda: None)()
        if self.sessions is None or graphs is None or not hasattr(graphs, "run_sflat"):
            return {}
        merge = merge or {}
        items, later = {}, {}
        for i, p in enumerate(plans):
            if i in skip or not self._sflat_reads_ok(p):
                continue
            carrier, n = merge.get(i, (i, None))
            if carrier != i:
                later[i] = (carrier, n)
                continue
            if n is None and len(p.prefix) < self.sessions.min_tokens:
                continue
            entry, start = self.sessions.match(p.prefix)
            mark = n if n is not None and start < n else self.sessions.mark(len(p.prefix), start)
            items[i] = (p.prefix, p.suffixes, entry, start, mark)
        # a save in one replay must not evict what a later replay of this round resumes from
        self.sessions.pinned.update(id(it[2]) for it in items.values() if it[2] is not None)
        out, trunks = {}, {}
        self._sflat_rounds(graphs, plans, items, out, trunks)
        if later:
            items = {}
            for i, (carrier, n) in later.items():
                entry = trunks.get(carrier)
                if entry is None or len(entry.tokens) != n:   # the carrier did not run or did not save there
                    continue
                p = plans[i]
                self.sessions.hits += 1
                self.sessions.saved_tokens += n
                items[i] = (p.prefix, p.suffixes, entry, n, self.sessions.mark(len(p.prefix), n))
            self._sflat_rounds(graphs, plans, items, out, trunks)
        self.sessions.pinned.clear()
        return out

    def _sflat_rounds(self, graphs, plans, items, out, trunks) -> None:
        """Replay `items` (plan -> session flat item) in rounds that fit, in order: head outputs into `out`, each save
        point into the cache and into `trunks` (plan -> the Entry it saved)."""
        from .session import Entry

        rounds, cur = [], []
        for i in items:
            if cur and graphs.sflat_shape([items[k] for k in cur + [i]]) is None:
                rounds.append(cur)
                cur = []
            cur.append(i)
        if cur:
            rounds.append(cur)
        for sel in rounds:
            res = graphs.run_sflat([items[i] for i in sel])
            if res is None:
                continue
            h, snaps = res
            at = 0
            for i, snap in zip(sel, snaps):
                n = len(plans[i].suffixes)
                head = plans[i].branch_reads(0)[0].head
                out[i] = self.heads[head](h[at: at + n]) if head is not None else h[at: at + n]
                at += n
                for pos, snap in snap or ():
                    entry = Entry(plans[i].prefix[:pos] if self.sessions.store is not None else plans[i].prefix, *snap)
                    self.sessions.insert(entry, parent=items[i][2])
                    trunks[i] = entry
                    self.sessions.pinned.add(id(entry))   # the other plans of its merge group resume from it

    def _graph_outputs(self, plan: Plan):
        y = self._graph_hidden(plan)
        if y is None:
            return None
        return self._layout(plan, y if self._kept(plan.branch_reads(0)[0].head) else y.float().cpu().tolist())

    def forward(self, plans: Sequence[Plan], mode: str = "graphs") -> tuple[list, list[str]]:
        """Head outputs of several plans in one round. "graphs" replays each plan's graphs back to back (one host sync
        at the end) and packs only the plans that do not fit them; "batch" runs the plans the flat graphs take together,
        one replay for as many as fit (no padding between them), and the rest as "graphs"; "packed" runs one packed
        forward over all of them; "alone" is "graphs" with every plan that does not fit packed by itself, so no plan's
        numbers depend on which others share the round (bit for bit the same answers at any load)."""
        return self.forward_collect(self.forward_launch(plans, mode))

    def forward_launch(self, plans: Sequence[Plan], mode: str = "graphs"):
        """The first half of `forward`: the round's graphs are queued on the GPU and their head outputs start coming
        back to the host, without waiting (plans that do not fit the graphs are computed here, packed). The online
        batcher queues the next round before collecting this one, so the GPU does not wait for the host."""
        import torch

        plans = [_rooted(p) for p in plans]
        outs: list[Any] = [None] * len(plans)
        paths = ["packed"] * len(plans)
        rest = list(range(len(plans)))
        copies = None
        if mode in ("graphs", "alone", "batch"):
            ys, rest = {}, []
            merge = self._merge_groups(plans) if mode == "batch" else {}
            batched = self._run_flat(plans, skip=merge) if mode == "batch" else {}
            if mode == "batch":
                batched.update(self._run_sflat(plans, batched, merge))
            ys.update(batched)
            for i, p in enumerate(plans):
                if i in batched:
                    continue
                # with the batch-invariant products a plan the flat graphs do not take is packed: the sequence and
                # branch graphs cut their key blocks where their padding ends
                y = None if settings.batch_invariant else self._graph_hidden(p, lazy=mode != "batch")
                if y is None:
                    rest.append(i)
                else:
                    ys[i] = y
            if ys:
                copies = {i: self._host(y, plans[i].branch_reads(0)[0].head) for i, y in ys.items()}
                done = torch.cuda.Event() if torch.cuda.is_available() else None
                if done is not None:
                    done.record()
                copies = (copies, done, {i: self._counts(plans[i]) for i in ys})
                for i in ys:
                    paths[i] = "graph"
        if rest and mode == "alone":
            for i in rest:
                outs[i] = self.run_plans([plans[i]])[0][0]
        elif rest:
            packed, _ = self.run_plans([plans[i] for i in rest])
            for i, br in zip(rest, packed):
                outs[i] = br
        return outs, paths, copies

    @staticmethod
    def forward_collect(launched) -> tuple[list, list[str]]:
        """The second half of `forward`: wait for the round's head outputs and lay them out per branch and read."""
        outs, paths, copies = launched
        if copies is not None:
            host, done, counts = copies
            if done is not None:
                done.synchronize()
            for i, y in host.items():
                outs[i] = _split(y.tolist(), counts[i])
        return outs, paths

    def _run_one(self, plan):
        plan = _rooted(plan)
        if settings.batch_invariant:
            outs, paths = self.forward([plan], "batch")
            return outs[0], paths[0]
        branches = self._graph_outputs(plan)
        if branches is None:
            return self.run_plans([plan])[0][0], "packed"
        return branches, "graph"

    def decide(self, request, **options):
        t0 = time.perf_counter()
        # the vision tower is queued as soon as the format has the images (one request at a time: the online batcher's
        # planner threads do not touch the device), so it runs while the host tokenizes and stages the request
        start = getattr(self.backbone, "start_vision", None)
        check_limits(self.format, request)
        plan = self.format.plan(request, start_vision=start, **options)
        if start is not None and plan.vision is not None:
            start(plan.vision)
        t1 = time.perf_counter()
        branches, path = self._run_one(plan)
        while (nxt := self.format.follow_up(request, plan, branches, **options)) is not None:
            plan = nxt
            branches, path = self._run_one(plan)
        t2 = time.perf_counter()
        response, meta = self.format.finish(request, plan, branches, **options)
        meta["timing"] = {"plan_s": t1 - t0, "forward_s": t2 - t1, "total_s": time.perf_counter() - t0}
        meta["path"] = path
        return response, meta

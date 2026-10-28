"""Engine: a checkpoint, its family and its backbone, and the packed forward between them."""
from __future__ import annotations

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


def model_config(model_dir: str):
    import transformers

    return transformers.AutoConfig.from_pretrained(model_dir, local_files_only=True)


def _blocks(plan: Plan) -> bool:
    """Whether a plan reads with block attention (formats.base.Plan); ValueError for an attention no backbone runs."""
    if plan.attention not in ("causal", "blocks"):
        raise ValueError(f"unknown attention {plan.attention!r}")
    return plan.attention == "blocks"


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


def _gib() -> tuple[float, float, float]:
    """GPU memory in GiB: allocated by torch, reserved by its caching allocator (graph pools included), used on the
    device by every process."""
    import torch

    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    g = 2 ** 30
    return torch.cuda.memory_allocated() / g, torch.cuda.memory_reserved() / g, (total - free) / g


class Engine:
    def __init__(self, model: str, format: str | None = None, device: str = "cuda:0", dtype: str = "bfloat16",
                 budget: int = 65536, backbone_options: dict | None = None, capture: bool | int = 4096,
                 session_cache_gb: float = 0.0, session_min_tokens: int = 2048, batch_invariant: bool = False,
                 **format_options):
        """`batch_invariant`: the batch-invariant products for every engine of the process (`settings`)."""
        if batch_invariant:
            settings.batch_invariant = True
        from ..kernels import launcher
        from .env import check_cuda

        if str(device).startswith("cuda"):
            check_cuda()
        launcher.install()
        from .kcache import cache_dir, first_compile, seed, tune

        seed()
        if first_compile():
            log.info("no precompiled kernels ship for this GPU and Triton version, so they compile now; only the first "
                     "start takes these minutes, later ones read them from %s", cache_dir())
        t0 = time.perf_counter()
        self.model_dir = resolve_model(model)
        config = model_config(self.model_dir)
        family = formats.detect(self.model_dir, config, format)
        if family.backbones and config.model_type not in family.backbones:
            raise ValueError(f"format {family.name!r} runs on {family.backbones}, {self.model_dir} is {config.model_type}")
        self.format = family(self.model_dir, **format_options)
        options = dict(backbone_options or {})
        if family.merge_lora:
            options["merge_lora"] = True
        tp_size = int(options.pop("tp", 1) or 1)
        if tp_size > 1:
            if session_cache_gb > 0:
                raise ValueError("tensor parallel runs without the session cache")
            from . import tp

            self.backbone = tp.launch(config.model_type, self.model_dir, device, dtype, tp_size, options)
        else:
            self.backbone = backbones.get(config.model_type).load(self.model_dir, device=device, dtype=dtype, **options)
        tune()
        needs = getattr(self.format, "attention", "causal")
        if needs != "causal" and not getattr(self.backbone, "block_attention", False):
            raise ValueError(f"{self.model_dir} reads with block attention, which the {config.model_type} backbone "
                             "does not run")
        self.heads = self.format.load_heads(self.backbone)
        self.eager_tokens = eager_tokens(config.model_type, self.backbone) if tp_size == 1 else None
        mem = _gib()
        self.budget = budget
        self.sessions = None
        if session_cache_gb > 0:
            from .session import SessionCache

            self.sessions = SessionCache(int(session_cache_gb * 2 ** 30), min_tokens=session_min_tokens)
        graphs = getattr(self.backbone, "request_graphs", lambda: None)()
        tune()
        if graphs is not None and capture is not False:
            limit = None if capture is True else capture
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
        if self.sessions is not None:
            from .session import reserve

            reserve(self.sessions.budget, self.backbone.device)
        if capture is not False:
            tune()
            self.warmup()
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
        saved, self.sessions = self.sessions, None
        try:
            self.run_plans(plans)
            if saved is not None:
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

    def run_plans(self, plans: Sequence[Plan | None], budget: int | None = None,
                  log_fn: Callable[[str], None] | None = None, warmup: int = 0) -> tuple[list, dict]:
        """Head outputs for every plan: out[plan][branch][read] is a list of rows, one per read position."""
        import torch

        budget = budget or self.budget
        for p in plans:
            if p is not None and _blocks(p) and not getattr(self.backbone, "block_attention", False):
                raise ValueError("this backbone does not run block attention")
            if p is not None and p.vision is not None and p.positions is None:
                p.positions, p.offset = self.backbone.vision_positions(p.prefix, p.vision)
        outputs: list[Any] = [None if p is None else [None] * len(p.suffixes) for p in plans]
        sessions = self._sessions_for(plans)
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
            hidden = self.backbone.forward(batch, pad_to=shape)
            if self.sessions is not None and batch.saved:
                from .session import Entry

                for j, saved in batch.saved.items():
                    self.sessions.insert(Entry(mb.pieces[j].job.prefix, saved["states"], saved["kv"]))
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
        """Graph head outputs per branch of a plan read at several positions (`_spots`, `_branch_spots`), else None."""
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
        if _blocks(plan):
            return False
        if not plan.prefix:
            return False
        shortest = min(len(s) for s in plan.suffixes)
        return all(p < 0 and -p <= shortest for b in range(len(plan.suffixes)) for r in plan.branch_reads(b)
                   for p in r.positions)

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

    def _graph_hidden(self, plan: Plan, lazy: bool = True):
        """Head outputs at the last token of every branch from CUDA graphs (still on the device): a flat graph (one
        replay), else the sequence and branch graphs, or the session graphs for long prefixes the cache takes; None when
        the plan or its shape does not fit them (reads other than the last token after the final layer, or lengths
        beyond the graph ladders). `lazy=False` uses sequence and branch graphs only when they were captured already."""
        graphs = getattr(self.backbone, "request_graphs", lambda: None)()
        if graphs is None or (plan.vision is not None and not graphs.media) or self._eager(plan):
            return None
        if not plan.prefix:
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
        if blocks and not getattr(graphs, "block_attention", False):
            return None
        if self._branch_spots(plan) is not None:
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
        if (spots is None and self.sessions is not None and hasattr(graphs, "run_session")
                and len(plan.prefix) >= self.sessions.min_tokens and self._session_ok(plan)):
            entry, start = self.sessions.match(plan.prefix)
            mark = self.sessions.mark(len(plan.prefix), start)
            out = None
            if hasattr(graphs, "run_sflat"):
                out = graphs.run_sflat([(plan.prefix, plan.suffixes, entry, start, mark)])
                if out is not None:
                    out = (out[0], out[1][0])
            if out is None:
                out = graphs.run_session(plan.prefix, plan.suffixes, entry, start, mark)
            if out is None:
                return None
            hidden, snap = out
            if snap is not None:
                from .session import Entry

                self.sessions.insert(Entry(plan.prefix[:mark], *snap))
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
        if blocks and not getattr(graphs, "block_attention", False):
            return None
        bspots = self._branch_spots(plan)
        if bspots is not None:
            if not getattr(graphs, "read_spots", False):
                return None
            tail = (blocks, bspots)
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
            tail = (blocks, spots) if spots is not None else (True,) if blocks else ()
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
        `read_spots`); None when the plan is not one of those or reads elsewhere."""
        if plan.prefix or not _blocks(plan) or plan.vision is not None or settings.batch_invariant:
            return None
        if not (getattr(graphs, "read_spots", False) and getattr(graphs, "block_attention", False)):
            return None
        reads = {plan.branch_reads(b) for b in range(len(plan.suffixes))}
        if len(reads) != 1:
            return None
        (read,) = reads
        if len(read) != 1 or read[0].layer is not FINAL or read[0].positions != (-1,):
            return None
        return [(s, [s[-1:]], None, True, (len(s) - 1,)) for s in plan.suffixes]

    @staticmethod
    def _flat_rounds(graphs, items) -> list[list[int]]:
        """Indices of flat graph items cut into rounds, in order, as many per replay as fit."""
        rounds, cur, size = [], [], None
        whole = getattr(graphs, "whole", lambda item: False)
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

    def _run_flat(self, plans) -> dict:
        """Plans of a round through flat graphs: as many per replay as fit, in order (a plan without a prefix under block
        attention as one item per branch, `_alone_items`)."""
        graphs = getattr(self.backbone, "request_graphs", lambda: None)()
        if graphs is None or not hasattr(graphs, "run_flat"):
            return {}
        entries = []
        for i, p in enumerate(plans):
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

    def _run_sflat(self, plans, skip) -> dict:
        """Plans of a round that the session cache takes (resume from a saved state, save their own) through session
        flat graphs, as many per replay as fit, in order; their save points go into the cache."""
        graphs = getattr(self.backbone, "request_graphs", lambda: None)()
        if self.sessions is None or graphs is None or not hasattr(graphs, "run_sflat"):
            return {}
        items = {}
        for i, p in enumerate(plans):
            if i in skip or not p.suffixes or len(p.prefix) < self.sessions.min_tokens or not self._session_ok(p):
                continue
            reads = {p.branch_reads(b) for b in range(len(p.suffixes))}
            if len(reads) != 1:
                continue
            (read,) = reads
            if len(read) != 1 or read[0].layer is not FINAL or read[0].positions != (-1,):
                continue
            entry, start = self.sessions.match(p.prefix)
            items[i] = (p.prefix, p.suffixes, entry, start, self.sessions.mark(len(p.prefix), start))
        rounds, cur = [], []
        for i in items:
            if cur and graphs.sflat_shape([items[k] for k in cur + [i]]) is None:
                rounds.append(cur)
                cur = []
            cur.append(i)
        if cur:
            rounds.append(cur)
        out = {}
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
                if snap is not None:
                    from .session import Entry

                    mark = next(iter(snap[0]))
                    self.sessions.insert(Entry(plans[i].prefix[:mark], *snap))
        return out

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

        outs: list[Any] = [None] * len(plans)
        paths = ["packed"] * len(plans)
        rest = list(range(len(plans)))
        copies = None
        if mode in ("graphs", "alone", "batch"):
            ys, rest = {}, []
            batched = self._run_flat(plans) if mode == "batch" else {}
            if mode == "batch":
                batched.update(self._run_sflat(plans, batched))
            ys.update(batched)
            for i, p in enumerate(plans):
                if i in batched:
                    continue
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
        if settings.batch_invariant:
            outs, paths = self.forward([plan], "batch")
            return outs[0], paths[0]
        branches = self._graph_outputs(plan)
        if branches is None:
            return self.run_plans([plan])[0][0], "packed"
        return branches, "graph"

    def decide(self, request, **options):
        t0 = time.perf_counter()
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

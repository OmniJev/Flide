"""OneJev: onejev-labels-v2 prompts, answer slots read from the output head at the last token of each
question, optional cyclic-shift debiasing, and chunking of choices with more options than slots.

`plan` builds the branches of a request and `finish` turns their slot logits into typed answers.

Checkpoints trained with OneJev's readout variants carry them in readout.json, the full-attention mask also under
`modeling_overrides` in config.json. "noncausal" (block_bidirectional) attention: the prefix (system turn and state
block) and each question read bidirectionally inside themselves and the question sees the prefix, the plan's
attention "blocks"; such a checkpoint reads text only, as its release does. The learned "head" readout is not read.
"""
from __future__ import annotations

import bisect
import hashlib
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from flide.api.answers import make_choice, make_noul, make_score
from flide.api.schema import Answer, ChoiceQuestion, SystemOneRequest, SystemOneResponse, Usage
from flide.formats import Format, Plan, Read, register

from . import media as mmedia
from .calibrate import Calibration, softmax
from .prompt import DEFAULT_STYLE, LETTERS, PROMPT_VERSION, SLOT_LABELS, PromptStyle, RenderedQuestion, build_messages, \
    render_question

log = logging.getLogger("flide.onejev")

NONE_OPTION = "none of the options listed here"
NONE_DESCRIPTION = "The correct option is not among the options listed in this question."
PREFIX_MARK = "FLIDE_PREFIX_MARK"
NAME_HINTS = ("onejev", "omnijev")
SLOTS = Read("slots")
READOUT_KEYS = {"attention", "readout", "head_dim", "n_slots"}
FULL_ATTENTION_MASK = {"causal": "causal", "block_bidirectional": "noncausal"}


def readout_config(model_dir: str) -> dict | None:
    """The readout variant a checkpoint was trained with (readout.json, the attention also under `modeling_overrides`
    in config.json, which must agree), or None when it carries neither."""
    import json

    path, cfg = Path(model_dir) / "readout.json", Path(model_dir) / "config.json"
    fields = json.loads(path.read_text()) if path.is_file() else None
    overrides = json.loads(cfg.read_text()).get("modeling_overrides") if cfg.is_file() else None
    if fields is None and not overrides:
        return None
    fields = dict(fields or {})
    if not set(fields) <= READOUT_KEYS:
        return None
    if overrides:
        mask = overrides.get("attention", {}).get("full_attention")
        attention = FULL_ATTENTION_MASK.get(mask)
        if attention is None:
            raise ValueError(f"unsupported full_attention mask {mask!r} in {cfg}")
        if fields.get("attention", attention) != attention:
            raise ValueError(f"{path} says attention {fields['attention']!r} but {cfg} says {mask!r}")
        fields["attention"] = attention
    fields.setdefault("attention", "causal")
    fields.setdefault("readout", "slot_lm")
    return fields


class _DeferredResize:
    """Stands in for the image processor's resize and patch reshuffle: the images it would resize as they are, with the
    size its own `smart_resize` gives them (in `sizes`) and the grid of that size; the device resizes them bit for bit
    (kernels/vision.resize_u8) and reshuffles the patches (backbones/qwen35/vision.pixels)."""

    def __init__(self, ip):
        import sys

        self.smart_resize = sys.modules[type(ip).resize.__module__].smart_resize
        self.sizes: list = []

    def resize(self, images, size, resample, factor, **_):
        h, w = images.shape[-2:]
        self.sizes.append(self.smart_resize(h, w, factor=factor, min_pixels=size.shortest_edge,
                                            max_pixels=size.longest_edge))
        return images

    def patchify(self, images, patch_size, merge_size, temporal_patch_size):
        h, w = self.sizes[-1]
        return images, h // patch_size, w // patch_size


def common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix, by bisection on slices (compared in C; os.path.commonprefix walks the
    characters in Python, ~1 µs each, which made a lookup cost tens of milliseconds on long prompts)."""
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n
    while hi - lo > 32:
        mid = (lo + hi) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid
    while lo < hi and a[lo] == b[lo]:
        lo += 1
    return lo


class TokenCache:
    """Token ids (with character offsets) of recent prompts, to tokenize a new prompt only after the point where it
    stops agreeing with one of them. The reuse point is just after a newline that both prompts share, before a shared
    character that is not whitespace: the pre-tokenizer (GPT-style regex) always starts a new piece there and BPE
    never merges across pieces, so the result is the whole-prompt tokenization. Shared by a family's per-thread
    twins; `selfcheck` turns it off for a tokenizer that does not split after newlines."""

    def __init__(self, size: int = 16, min_chars: int = 512):
        self.items: OrderedDict[int, tuple[str, list[int], list[tuple[int, int]]]] = OrderedDict()
        self.size = size
        self.min_chars = min_chars
        self.lock = threading.Lock()
        self.enabled = True
        self.hits = self.misses = self.saved_chars = 0
        self._next = 0

    def lookup(self, text: str):
        if not self.enabled:
            return None
        with self.lock:
            items = list(self.items.values())
        best = None
        for t, ids, offs in items:
            if t[: self.min_chars] != text[: self.min_chars]:
                continue
            c = common_prefix_len(t, text)
            if c >= self.min_chars and (best is None or c > best[0]):
                best = (c, ids, offs)
        if best is None:
            return None
        c, ids, offs = best
        b = text.rfind("\n", 0, c - 1)
        while b > 0 and text[b + 1: b + 2].isspace():
            b = text.rfind("\n", 0, b)
        if b <= 0:
            return None
        b += 1
        starts = [o[0] for o in offs]
        i = bisect.bisect_left(starts, b)
        if i >= len(starts) or starts[i] != b:
            return None
        return ids[:i], offs[:i], b

    def add(self, text: str, ids: list[int], offs: list[tuple[int, int]]) -> None:
        if not self.enabled or len(text) < self.min_chars:
            return
        with self.lock:
            self.items[self._next] = (text, ids, offs)
            self._next += 1
            while len(self.items) > self.size:
                self.items.popitem(last=False)


class SlotHead:
    """The output-head rows of the answer slots, in float32: slot logits for final-norm hidden states."""

    def __init__(self, weight, slot_ids: list[int], bias=None, dtype=None, softcap: float | None = None):
        import torch

        ids = torch.tensor(slot_ids, device=weight.device)
        dt = dtype or torch.float32
        self.weight = weight.detach().index_select(0, ids).to(dt).contiguous()
        self.bias = None if bias is None else bias.detach().index_select(0, ids).to(dt)
        self.softcap = softcap

    def __call__(self, hidden):
        out = hidden.to(self.weight.dtype) @ self.weight.T
        out = out if self.bias is None else out + self.bias
        if self.softcap:
            out = (out / self.softcap).tanh() * self.softcap
        return out


@register("onejev", format_ids=("onejev-slot-v2",))
class OneJev(Format):
    backbones = ("qwen3_5", "qwen3_5_text", "gemma4", "gemma4_text", "gemma4_unified")
    meta_key = "onejev"
    aliases = ("jev-latest", "jev-preview")
    max_choice_options = 256

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        if decision_config is not None:
            return False
        if getattr(config, "model_type", None) not in cls.backbones:
            return False
        path = str(Path(model_dir).resolve()).lower()
        if any(h in path for h in NAME_HINTS):
            return True
        rc = readout_config(model_dir)
        return rc is not None and rc["readout"] in ("slot_lm", "head")

    def __init__(self, model_dir: str, tokenizer=None, calibration: Calibration | str | None = None,
                 style: PromptStyle = DEFAULT_STYLE, max_branch_tokens: int = 32768, max_request_tokens: int = 65536,
                 media_root: str | None = None, allow_local_paths: bool | None = None):
        if tokenizer is None:
            import transformers

            tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.tokenizer = tokenizer
        if calibration is None:
            found = Path(model_dir) / "calibration.json"
            calibration = Calibration.load(found) if found.is_file() else Calibration()
        elif isinstance(calibration, (str, Path)):
            calibration = Calibration.load(calibration)
        self.calibration = calibration
        self.style = style
        self.max_branch_tokens = max_branch_tokens
        self.max_request_tokens = max_request_tokens
        import transformers

        config = transformers.AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.template_kwargs = {"enable_thinking": False} if config.model_type.startswith(("qwen", "gemma4")) else {}
        self.slot_ids = self._verify_slots()
        self.max_score_levels = self.max_slots
        self.token_cache = TokenCache()
        self._token_cache_selfcheck()
        self.model_dir = model_dir
        self.media_root = media_root
        self.allow_local_paths = media_root is not None if allow_local_paths is None else allow_local_paths
        self.vision_ids = {i for i in (getattr(config, k, None) for k in ("image_token_id", "video_token_id",
                           "vision_start_token_id", "vision_end_token_id", "boi_token_id", "eoi_token_id",
                           "audio_token_id")) if i is not None}
        self.has_vision = getattr(config, "vision_config", None) is not None
        rc = readout_config(model_dir) or {"attention": "causal", "readout": "slot_lm"}
        if rc["readout"] != "slot_lm":
            raise ValueError(f"{model_dir} reads its answers with a learned {rc['readout']!r} readout; Flide reads "
                             "the slot letters' output-head logits only (slot_lm)")
        if rc["attention"] not in ("causal", "noncausal"):
            raise ValueError(f"unknown attention {rc['attention']!r} in {model_dir}")
        self.attention = "blocks" if rc["attention"] == "noncausal" else "causal"
        self._processor = None
        self._marker_checks = 0
        self._marker_ok = True

    def for_thread(self):
        twin = super().for_thread()
        twin._processor = None
        twin._raw_ip = None
        return twin

    @property
    def processor(self):
        if self._processor is None:
            import transformers

            try:
                self._processor = transformers.AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
            except ImportError as e:
                if "orchvision" not in str(e):
                    raise
                raise ImportError("images and video need torchvision; install it from the same index as torch, e.g. "
                                  "pip install torchvision --index-url https://download.pytorch.org/whl/cu126") from e
            if self._processor.tokenizer.chat_template is None:
                self._processor.tokenizer.chat_template = self._processor.chat_template
        return self._processor


    def _verify_slots(self) -> list[int]:
        """Slot token ids for every label the tokenizer holds as one round-trip token. A..Z are required. The
        two-letter labels are all-or-nothing: a tokenizer that splits any of them keeps 26 slots, and choices
        with more options go through the chunked path."""
        def one(label):
            enc = self.tokenizer.encode(label, add_special_tokens=False)
            return enc[0] if len(enc) == 1 and self.tokenizer.decode(enc) == label else None

        ids = [one(l) for l in LETTERS]
        if any(i is None for i in ids):
            bad = [l for l, i in zip(LETTERS, ids) if i is None]
            raise ValueError(f"answer slots {bad} are not single round-trip tokens")
        extra = [one(l) for l in SLOT_LABELS[len(LETTERS):]]
        if all(i is not None for i in extra):
            ids += extra
        else:
            log.warning("tokenizer splits some two-letter labels; %d slots, larger choices are chunked", len(ids))
        if len(set(ids)) != len(ids):
            raise ValueError("answer slot tokens collide")
        self.max_slots = len(ids)
        return ids

    def _render_prompt(self, state: Any, suffix: str) -> str:
        return self.tokenizer.apply_chat_template(build_messages(state, suffix, self.style), tokenize=False,
                                                  add_generation_prompt=True, **self.template_kwargs)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _encode_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Token ids and character offsets of a whole prompt, tokenizing only what follows the part it shares with
        a recent prompt (TokenCache)."""
        hit = self.token_cache.lookup(text)
        if hit is None:
            enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
            self.token_cache.misses += 1
        else:
            head_ids, head_offs, b = hit
            enc = self.tokenizer(text[b:], add_special_tokens=False, return_offsets_mapping=True)
            ids = head_ids + list(enc["input_ids"])
            offs = head_offs + [(x + b, y + b) for x, y in enc["offset_mapping"]]
            self.token_cache.hits += 1
            self.token_cache.saved_chars += b
        self.token_cache.add(text, ids, offs)
        return ids, offs

    def _token_cache_selfcheck(self) -> None:
        """Turn the token cache off for a tokenizer whose pieces do not restart after newlines."""
        probe = ("{\n  \"a\": \"x y z\",\n\n  \"b\": [1, 22, 333],\n  \"ccc\": \"日本語 テスト\"\n}\n"
                 "Line two\t tabs   spaces\n\n\n   indented\nend\n \nblank\n") * 20
        whole = self.tokenizer(probe, add_special_tokens=False, return_offsets_mapping=True)
        starts = {o[0] for o in whole["offset_mapping"]}
        for b in range(1, len(probe)):
            if probe[b - 1] == "\n" and not probe[b].isspace():
                if b not in starts or (self._encode(probe[:b]) + self._encode(probe[b:]) != list(whole["input_ids"])):
                    self.token_cache.enabled = False
                    log.info("token cache off: this tokenizer does not split after newline runs")
                    return

    BACKOFF = 16
    MARKER_CHECKS = 32

    def _branch_prompts(self, render, suffixes: list[str]) -> tuple[list[str], str, int]:
        """Every branch's chat prompt from one rendering with PREFIX_MARK where the branch text goes, plus that
        rendering and the marker's offset. The template trims the user turn, so a branch enters without trailing
        whitespace. The first MARKER_CHECKS plans also render their first branch whole and compare; one mismatch
        turns the shortcut off for this instance (every branch rendered whole, as before)."""
        marked = render(PREFIX_MARK)
        cut = marked.index(PREFIX_MARK)
        if not self._marker_ok:
            return [render(x) for x in suffixes], marked, cut
        head, tail = marked[:cut], marked[cut + len(PREFIX_MARK):]
        prompts = [head + x.rstrip() + tail for x in suffixes]
        if self._marker_checks < self.MARKER_CHECKS:
            self._marker_checks += 1
            if render(suffixes[0]) != prompts[0]:
                self._marker_ok = False
                log.info("prompt marker shortcut off: the template does not place the branch text verbatim")
                return [render(x) for x in suffixes], marked, cut
        return prompts, marked, cut

    def _encode_shared(self, prompts: list[str], cut: int) -> list[list[int]]:
        """Whole-prompt token ids of prompts that agree on their first `cut` characters, tokenizing the shared part
        once. Falls back to tokenizing each prompt whole when the shortcut does not reproduce the first one."""
        if any(p[:cut] != prompts[0][:cut] for p in prompts[1:]):
            return [self._encode(p) for p in prompts]
        ids0, offs = self._encode_offsets(prompts[0])
        if len(prompts) == 1:
            return [ids0]
        inside = 0
        while inside < len(ids0) and offs[inside][1] <= cut:
            inside += 1
        n0 = inside - self.BACKOFF
        if n0 <= 0:
            return [self._encode(p) for p in prompts]
        off = offs[n0][0]
        head = ids0[:n0]
        if head + self._encode(prompts[0][off:]) != ids0:
            return [self._encode(p) for p in prompts]
        return [ids0] + [head + self._encode(p[off:]) for p in prompts[1:]]

    def _prefix_and_suffix_ids(self, state: Any, rendered: list[RenderedQuestion]):
        """The longest common token prefix of all branches that ends at or before the end of the state block, and each
        branch's tokens after it, as if every branch were tokenized whole.

        Tokenizing every branch whole costs (branches x state) tokens, the bulk of planning for long states with many
        questions. The branches share their text up to the end of the state block, so the first branch is tokenized
        whole and the others only from a token boundary 16 tokens before that point: tokens never merge across that
        far, and the first branch, rebuilt the same way, is checked against its whole tokenization."""
        prompts, prefix_text, cut = self._branch_prompts(lambda x: self._render_prompt(state, x),
                                                         [r.suffix for r in rendered])
        full = self._encode_shared(prompts, cut)
        prefix_ids = self._encode_offsets(prefix_text[:cut])[0]
        n = len(prefix_ids)
        while n > 0 and not all(ids[:n] == prefix_ids[:n] for ids in full):
            n -= 1
        if self.attention == "blocks" and n < len(prefix_ids):
            for p in prompts:
                ends = [e for _, e in self._encode_offsets(p)[1]]
                k = bisect.bisect_right(ends, cut)
                if k != n:
                    raise ValueError(f"prefix block ends at token {k} but the shared cache at {n}")
        prefix_ids = prefix_ids[:n]
        suffixes = [ids[n:] for ids in full]
        if any(len(s) == 0 for s in suffixes):
            raise ValueError("a branch has an empty suffix; state block boundary mismatch")
        return prefix_ids, suffixes, prompts[0]

    def _prefix_and_suffix_media(self, state: Any, rendered: list[RenderedQuestion], items: list[dict],
                                 start_vision=None):
        """Multimodal preparation without the model: tokenize every branch with its vision blocks
        unexpanded, take the longest common prefix that ends at or before the end of the state block (all media
        sit inside it), then expand the vision blocks of the prefix and build the pixel tensors. Images on Qwen: the
        image processor alone, each placeholder repeated once per merged patch (what the processor's text expansion
        gives, without tokenizing the expanded prompt again). Otherwise (videos, whose timestamps go between frames;
        Gemma's begin / end tokens around each image): the processor on the first branch, whose tokens after the
        expanded prefix must equal its unexpanded suffix. `start_vision(vision)`, when given, gets the images' inputs
        before the prompts are tokenized (the engine queues the vision tower on the device then)."""
        import torch

        proc = self.processor
        qwen_images = (all(m["type"] == "image" for m in items)
                       and type(proc.image_processor).__name__.startswith("Qwen2VL"))
        if qwen_images:
            imgs, sizes, grid, images = self._qwen_images(items, self._raw_image_processor())
            vision = {"image_grid_thw": grid}
            if imgs is not None:
                ip = proc.image_processor
                vision.update(pixel_images_u8=imgs, pixel_resize=sizes, pixel_scale=self._pixel_scale(ip),
                              pixel_patch=(ip.patch_size, ip.merge_size, ip.temporal_patch_size))
                if start_vision is not None:
                    start_vision(vision)
        else:
            images, videos, metas = mmedia.load(items, self.media_root, self.allow_local_paths)
        render = lambda suffix: mmedia.render_prompt(proc, self.template_kwargs, build_messages(state, suffix, self.style),
                                                     items)
        prompts, prefix_text, cut = self._branch_prompts(render, [r.suffix for r in rendered])
        full = self._encode_shared(prompts, cut)
        prefix_u = self._encode_offsets(prefix_text[:cut])[0]
        n = len(prefix_u)
        while n > 0 and not all(ids[:n] == prefix_u[:n] for ids in full):
            n -= 1
        suffixes = [ids[n:] for ids in full]
        if any(len(s) == 0 for s in suffixes):
            raise ValueError("a branch has an empty suffix; state block boundary mismatch")
        if any(t in self.vision_ids for s in suffixes for t in s):
            raise ValueError("media placeholders belong in the state, not in a question")
        if qwen_images:
            ip = proc.image_processor
            counts = (grid.prod(-1) // ip.merge_size ** 2).tolist()
            pad = proc.image_token_id
            expanded, k = [], 0
            for t in full[0][:n]:
                if t == pad:
                    expanded.extend([pad] * counts[k])
                    k += 1
                else:
                    expanded.append(t)
            if k != len(counts):
                raise ValueError(f"{k} image placeholders for {len(counts)} images")
            vision["mm_token_type_ids"] = (torch.tensor(expanded) == pad).long()[None]
            if imgs is None:
                vision["pixel_values"] = ip(images, return_tensors="pt")["pixel_values"]
            return expanded, suffixes, prompts[0], vision
        proc.tokenizer.padding_side = "right"
        out = proc(text=[prompts[0]], return_tensors="pt", **mmedia.processor_kwargs(images, videos, metas))
        expanded = out["input_ids"][0].tolist()
        P = n + len(expanded) - len(full[0])
        if expanded[P:] != suffixes[0]:
            raise RuntimeError("expanded prompt does not end with the branch suffix; vision block outside the prefix")
        vision = {k: out[k] for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw",
                                      "image_position_ids", "video_position_ids") if k in out}
        if "mm_token_type_ids" in out:
            vision["mm_token_type_ids"] = out["mm_token_type_ids"][:, :P]
        return expanded[:P], suffixes, prompts[0], vision

    def _raw_image_processor(self):
        """This format's image processor with its resize and patch reshuffle left to the device (it returns the uint8
        image it would resize, `_DeferredResize` keeps the size)."""
        raw = getattr(self, "_raw_ip", None)
        if raw is None:
            import copy

            raw = self._raw_ip = copy.copy(self.processor.image_processor)
            raw.deferred = _DeferredResize(raw)
            raw.resize, raw.patchify = raw.deferred.resize, raw.deferred.patchify
        return raw

    def _qwen_images(self, items: list[dict], raw) -> tuple:
        """(uint8 [3, h, w] per image as the processor would resize it, or None when it does not give uint8; the sizes
        it resizes them to; grids [n, 3]; the decoded images)."""
        import torch

        images, _, _ = mmedia.load(items, self.media_root, self.allow_local_paths)
        imgs, grids, sizes = [], [], []
        for im in images:
            fast = self._rgb_image(im, raw) if getattr(self, "_rgb_fast", True) else None
            if fast is None or getattr(self, "_rgb_fast", None) is None:
                raw.deferred.sizes.clear()
                out = raw(im, do_rescale=False, do_normalize=False, return_tensors="pt")
                x = out["pixel_values"]
                if x.dtype != torch.uint8 or x.dim() != 3:
                    return (None, None, self.processor.image_processor(images, return_tensors="pt")["image_grid_thw"],
                            images)
                got = (x, raw.deferred.sizes[-1], out["image_grid_thw"])
                if fast is not None:
                    self._rgb_fast = (torch.equal(fast[0], got[0]) and tuple(fast[1]) == tuple(got[1])
                                      and torch.equal(fast[2], got[2]))
                    if not self._rgb_fast:
                        log.warning("RGB images without the image processor disagree with it; using the processor")
                fast = got
            imgs.append(fast[0])
            sizes.append(fast[1])
            grids.append(fast[2])
        return imgs, sizes, torch.cat(grids), images

    @staticmethod
    def _rgb_image(im, raw):
        """What the image processor (resize deferred) gives for an RGB image, without running it: the pixels as uint8
        [3, h, w] (channels last in memory, the image's own layout), the size its smart_resize gives and the grid of
        that size; None for other modes. About 1 ms less per image than the processor's conversions."""
        if getattr(im, "mode", None) != "RGB":
            return None
        import numpy as np
        import torch

        edge = lambda k: raw.size[k] if isinstance(raw.size, dict) else getattr(raw.size, k)
        w, h = im.size
        H, W = raw.deferred.smart_resize(h, w, factor=raw.patch_size * raw.merge_size, min_pixels=edge("shortest_edge"),
                                         max_pixels=edge("longest_edge"))
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1)
        return x, (H, W), torch.tensor([[1, H // raw.patch_size, W // raw.patch_size]])

    def _pixel_scale(self, ip):
        """Per patch column, the mean and standard deviation the image processor normalizes with after folding its
        rescale factor in (its own arithmetic: float32 tensors times 1 / factor), so normalizing uint8 patches with them
        gives the processor's values bit for bit."""
        scale = getattr(self, "_scale", None)
        if scale is None:
            import torch

            per = ip.temporal_patch_size * ip.patch_size ** 2
            mean = torch.tensor(ip.image_mean) * (1.0 / ip.rescale_factor)
            std = torch.tensor(ip.image_std) * (1.0 / ip.rescale_factor)
            scale = self._scale = (mean.repeat_interleave(per), std.repeat_interleave(per))
        return scale

    def _plan_chunks(self, q: ChoiceQuestion) -> list[list[str]]:
        """Split a many-option choice into chunks of at most max_slots - 1 options (plus one shared
        "none of the options listed here" slot per chunk), balanced in size."""
        names = list(q.criteria.keys())
        n_chunks = -(-len(names) // (self.max_slots - 1))
        size = -(-len(names) // n_chunks)
        return [names[i:i + size] for i in range(0, len(names), size)]


    def load_heads(self, backbone) -> dict[str, Any]:
        weight, bias = backbone.output_head()
        return {"slots": SlotHead(weight, self.slot_ids, bias, softcap=getattr(backbone, "logit_softcap", None))}

    def plan(self, request: SystemOneRequest, debias: int = 1, start_vision=None, **_) -> Plan:
        rendered: list[RenderedQuestion] = []
        owners: list[tuple[str, list[int], int | None]] = []
        for qid, q in request.questions.items():
            if isinstance(q, ChoiceQuestion) and len(q.criteria) > self.max_slots:
                for ci, names in enumerate(self._plan_chunks(q)):
                    sub = ChoiceQuestion(type="choice", instructions=q.instructions,
                                         criteria={**{n: q.criteria[n] for n in names}, NONE_OPTION: NONE_DESCRIPTION})
                    rendered.append(render_question(qid, sub, style=self.style))
                    owners.append((qid, list(range(len(names) + 1)), ci))
                continue
            base = render_question(qid, q, style=self.style)
            k = base.n_slots
            shifts = range(min(max(debias, 1), k)) if k > 1 else range(1)
            for s in shifts:
                order = [(i + s) % k for i in range(k)]
                rendered.append(render_question(qid, q, order=order if s else None, style=self.style))
                owners.append((qid, order, None))

        items = mmedia.check_items(getattr(request, "media", None))
        vision = None
        if items:
            if self.attention == "blocks":
                raise ValueError("this checkpoint reads text only; the request carries media")
            if not self.has_vision:
                raise ValueError("this checkpoint has no vision tower; the request carries media")
            prefix_ids, suffixes, first_prompt, vision = self._prefix_and_suffix_media(request.state, rendered, items,
                                                                                       start_vision)
        else:
            prefix_ids, suffixes, first_prompt = self._prefix_and_suffix_ids(request.state, rendered)
        longest = len(prefix_ids) + max(len(s) for s in suffixes)
        total = len(prefix_ids) + sum(len(s) for s in suffixes)
        if longest > self.max_branch_tokens:
            raise ValueError(f"state plus longest question is {longest} tokens, limit is {self.max_branch_tokens}")
        if total > self.max_request_tokens:
            raise ValueError(f"request is {total} tokens, limit is {self.max_request_tokens}")
        data = {"rendered": rendered, "owners": owners,
                "prompt_sha256": hashlib.sha256(first_prompt.encode()).hexdigest()}
        return Plan(prefix_ids, suffixes, (SLOTS,), data, vision=vision, attention=self.attention)

    def finish(self, request: SystemOneRequest, plan: Plan, outputs: list[list[Any]], debug: bool = False, **_):
        rendered: list[RenderedQuestion] = plan.data["rendered"]
        owners = plan.data["owners"]
        if len(outputs) != len(rendered):
            raise RuntimeError(f"{len(outputs)} branch outputs for {len(rendered)} branches")

        acc: dict[str, dict[str, Any]] = {}
        chunked: dict[str, dict[str, Any]] = {}
        for r, out, (qid, order, ci) in zip(rendered, outputs, owners):
            logits = list(out[0][0][: r.n_slots])
            temp = self.calibration.temperature(r.kind, len(r.labels))
            probs = softmax(logits, temp)
            if ci is not None:
                c = chunked.setdefault(qid, {"scores": {}, "weights": [], "n": 0, "logits": []})
                none_p = probs[r.labels.index(NONE_OPTION)]
                keep = max(1.0 - none_p, 1e-9)
                for lab, p in zip(r.labels, probs):
                    if lab != NONE_OPTION:
                        c["scores"][lab] = (p / keep, keep)
                c["weights"].append(keep)
                c["n"] += 1
                c["logits"].append(logits)
                continue
            slot = acc.setdefault(qid, {"kind": r.kind, "probs": None, "labels": None, "descriptions": None,
                                        "n": 0, "logits": []})
            if slot["probs"] is None:
                k = len(r.labels)
                slot["probs"] = [0.0] * k
                slot["labels"] = [None] * k
                slot["descriptions"] = [None] * k
            for pos_i, orig_i in enumerate(order):
                slot["probs"][orig_i] += probs[pos_i]
                slot["labels"][orig_i] = r.labels[pos_i]
                slot["descriptions"][orig_i] = r.descriptions[pos_i]
            slot["n"] += 1
            slot["logits"].append(logits)

        answers: dict[str, Answer] = {}
        diag: dict[str, Any] = {}
        for qid, c in chunked.items():
            total_w = sum(c["weights"]) or 1.0
            names = list(request.questions[qid].criteria.keys())
            probs = [c["scores"][n][0] * (c["scores"][n][1] / total_w) for n in names]
            answers[qid] = make_choice(names, probs)
            diag[qid] = {"slot_logits": c["logits"], "chunks": c["n"],
                         "temperature": self.calibration.temperature("choice", self.max_slots)}
        for qid, slot in acc.items():
            probs = [p / slot["n"] for p in slot["probs"]]
            if slot["kind"] == "noul":
                answers[qid] = make_noul(probs[0])
            elif slot["kind"] == "choice":
                answers[qid] = make_choice(slot["labels"], probs)
            else:
                answers[qid] = make_score(slot["descriptions"], probs)
            diag[qid] = {"slot_logits": slot["logits"],
                         "temperature": self.calibration.temperature(slot["kind"], len(probs))}

        usage = Usage(input_tokens=plan.tokens, output_tokens=len(request.questions))
        response = SystemOneResponse(model="", answers=answers, usage=usage)
        meta = {"prompt_version": self.style.name or PROMPT_VERSION, "prompt_sha256": plan.data["prompt_sha256"],
                "prefix_tokens": len(plan.prefix), "branches": len(rendered)}
        if debug:
            meta["diagnostics"] = diag
        return response, meta

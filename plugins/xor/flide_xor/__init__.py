"""Xor family (juspay/Xor-26B-A4B on Gemma 4 26B-A4B): the release's own front end, serving/xn_base_server.py, as its
serving bundle runs it (two-order scoring on; temperatures choice 1.95, noul 1, score 1).

Every question is its own chat prompt, thinking off, its pieces encoded one by one and joined: the template's head,
"Context:\n", the state (a string as it is, anything else as compact JSON), "\n\nQuestion: <instructions>\nOptions:",
one "\n(<label>) <option>" line per option, the template's tail and "Answer: (". The answer is the softmax over the
labels' tokens there: A to Z, then AA, AB, ... while each is a token of its own after "Answer: (", 255 in all. A choice
is read a second time with its options reversed; the two orders' probabilities (temperature 1) are averaged by option
and their logarithms taken as the logits of the type's temperature. Everything up to the end of the state is the same
for every row, so the rows fork from it.

Images (1 to 8 data URLs) and a video (one) go in front of the text as the release's SGLang takes them. A JPEG is
decoded on the GPU (nvJPEG), and then each row's ids are decoded to text and given whole to the processor with the
images, which writes every image's begin token, soft tokens and end token in place of its placeholder and tokenizes
the row anew; other images are opened by PIL and every placeholder is repeated once per soft token of its image, no
other token changed. A video is decoded by torchcodec, up to 32 frames sampled evenly, and each row goes to the
processor the first way, which writes every frame's timestamp and soft tokens. The processor runs on the GPU.
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import math
import string
from pathlib import Path
from typing import Any

from flide.api.answers import _round, make_choice, make_noul, make_score
from flide.api.schema import SystemOneRequest, SystemOneResponse, Usage
from flide.formats import register
from flide.formats.base import Format, Plan, Read
from flide.formats.next_token import TokenRows

MAX_OPTIONS = 255
MAX_IMAGES = 8
SLOT = "@@XN_USER@@"
ANSWER = "Answer: ("
NOUL_DEFAULT = "Which answer fits the context?"
TEMPERATURES = {"choice": 1.95, "noul": 1.0, "score": 1.0}
TEMPLATE_KW = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
READ = (Read("labels"),)


def _txt(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def render_state(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def render_question(qid: str, q) -> tuple[str, list[str], list[str]]:
    """(instructions, option texts, answer keys) as the release renders a question; ValueError where it refuses."""
    ins = q.instructions
    if q.type == "choice":
        crit = q.criteria
        if len(crit) < 2:
            raise ValueError(f"{qid}: a choice needs at least two options")
        if len(crit) > MAX_OPTIONS:
            raise ValueError(f"{qid}: {len(crit)} options; the limit is {MAX_OPTIONS} options per choice")
        keys = list(crit)
        opts = [k if crit[k] in (None, "") else f"{k}: {_txt(crit[k])}" for k in keys]
        if ins in (None, ""):
            raise ValueError(f"{qid}: choice question without instructions")
        return _txt(ins), opts, keys
    if q.type == "noul":
        c = q.criteria
        f, tr = (c.false, c.true) if c is not None else (None, None)
        opts = ["no" if f in (None, "") else f"no: {_txt(f)}", "yes" if tr in (None, "") else f"yes: {_txt(tr)}"]
        if ins in (None, ""):
            if f in (None, "") and tr in (None, ""):
                raise ValueError(f"{qid}: noul question without instructions or criteria")
            ins = NOUL_DEFAULT
        return _txt(ins), opts, ["false", "true"]
    crit = q.criteria
    if not 2 <= len(crit) <= 10:
        raise ValueError(f"{qid}: a score takes 2 to 10 levels, got {len(crit)}")
    if ins in (None, ""):
        raise ValueError(f"{qid}: score question without instructions")
    return _txt(ins), [f"{i}: {_txt(c)}" for i, c in enumerate(crit)], [str(i) for i in range(len(crit))]


def _softmax(x: list[float], t: float = 1.0) -> list[float]:
    m = max(x)
    e = [math.exp((v - m) / t) for v in x]
    s = sum(e)
    return [v / s for v in e]


def _data_url(ref: Any, kind: str) -> bytes:
    if not isinstance(ref, str) or not ref.startswith(f"data:{kind}/"):
        raise ValueError(f"{kind} must be a base64 data URL")
    header, sep, encoded = ref.partition(",")
    if not sep or not header.endswith(";base64") or not encoded:
        raise ValueError(f"{kind} must be a non-empty base64 data URL")
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError(f"{kind} contains invalid base64") from None


def _image(raw: bytes, device):
    """An image as its SGLang opens it: a JPEG (its start and end markers) by nvJPEG on the GPU as a [3, H, W] uint8
    tensor, falling back to PIL; anything else by PIL, converted to RGB. A JPEG of other than three channels (a
    grayscale one) is read as RGB: the release's vision tower fails on it and its server stops, Flide answers."""
    jpeg = raw.startswith(b"\xff\xd8") and raw.endswith(b"\xff\xd9")
    if device is not None and jpeg:
        try:
            import torch
            from torchvision.io import ImageReadMode, decode_jpeg

            data = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            t = decode_jpeg(data, device=device)
            return t if t.shape[0] == 3 else decode_jpeg(data, mode=ImageReadMode.RGB, device=device)
        except Exception:
            pass
    from PIL import Image

    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except OSError as e:
        raise ValueError(f"Could not decode image: {e}") from e
    return im if im.mode == "RGB" else im.convert("RGB")


def _frames(ref: Any, n: int):
    """A video's frames as its SGLang reads them: torchcodec on the CPU, every frame when there are at most n, else n
    at torch.arange(0, total, total / n), as a [frames, 3, H, W] uint8 tensor."""
    import torch

    raw = _data_url(ref, "video")
    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError as e:
        raise ImportError("video needs torchcodec and FFmpeg: pip install torchcodec") from e
    try:
        dec = VideoDecoder(raw, dimension_order="NHWC")
    except RuntimeError as e:
        raise ValueError(f"Could not decode video: {e}") from e
    total = len(dec)
    idx = list(range(total)) if total <= n else torch.arange(0, total, total / n).int().tolist()
    return torch.from_numpy(dec.get_frames_at(idx).data.numpy()).permute(0, 3, 1, 2).contiguous()


@register("xor")
class Xor(Format):
    backbones = ("gemma4", "gemma4_text", "gemma4_unified")

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        return (Path(model_dir) / "serving" / "xn_base_server.py").is_file()

    def __init__(self, model_dir: str, tokenizer=None, max_context: int = 131072, two_order: bool = True,
                 temperatures: dict | None = None):
        import transformers

        self.model_dir = model_dir
        self.tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.max_context = int(max_context)
        self.two_order = bool(two_order)
        self.temperatures = {**TEMPERATURES, **(temperatures or {})}
        head, tail = self.tokenizer.apply_chat_template([{"role": "user", "content": SLOT}], **TEMPLATE_KW).split(SLOT)
        self.head, self.tail = self._e(head), self._e(tail)
        self.answer = self._e(ANSWER)
        self.labels, self.label_ids = self._labels()
        self._heads: dict[tuple, list[int]] = {}
        self.device = None

    def _e(self, s: str) -> list[int]:
        return self.tokenizer.encode(s, add_special_tokens=False)

    def _labels(self) -> tuple[list[str], list[int]]:
        """A..Z, then AA, AB, ...: the token after "Answer: (" of each, skipping those that merge into the prefix or
        repeat a token, up to 255."""
        names, ids = [], []
        for n in list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]:
            t = self._e(ANSWER + n)
            if len(t) <= len(self.answer) or t[-1] in ids:
                continue
            names.append(n)
            ids.append(t[-1])
            if len(ids) == MAX_OPTIONS:
                break
        if len(ids) != MAX_OPTIONS:
            raise ValueError(f"the tokenizer has {len(ids)} label tokens after {ANSWER!r}, the release needs {MAX_OPTIONS}")
        return names, ids

    def token_heads(self) -> dict[str, list[int]]:
        return {"labels": list(self.label_ids)}

    def load_heads(self, backbone) -> dict[str, Any]:
        dev = str(getattr(backbone, "device", "cpu"))
        self.device = dev if dev.startswith("cuda") else None
        weight, bias = backbone.output_head()
        return {"labels": TokenRows(weight, self.label_ids, bias, softcap=getattr(backbone, "logit_softcap", None))}

    @property
    def processor(self):
        p = getattr(self, "_processor", None)
        if p is None:
            import transformers

            try:
                p = transformers.AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
            except ImportError as e:
                if "orchvision" not in str(e):
                    raise
                raise ImportError("images need torchvision: pip install torchvision") from e
            self._processor = p
        return p

    def _media_head(self, n: int, video: bool = False) -> list[int]:
        """The template's head with n image placeholders, or a video's, from the processor's chat template."""
        if (n, video) not in self._heads:
            media = [{"type": "image"} for _ in range(n)] if n else [{"type": "video"}]
            content = media + [{"type": "text", "text": SLOT}]
            head, tail = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                            **TEMPLATE_KW).split(SLOT)
            if self._e(tail) != self.tail:
                raise ValueError("media template suffix differs from the scoring template")
            self._heads[(n, video)] = self._e(head)
        return self._heads[(n, video)]

    def _media_rows(self, head: list[int], rows: list[list[int]], refs: list | None, video) -> tuple[list, dict]:
        """Every row's ids with its media written in, as the release's SGLang builds them, and the vision inputs."""
        p = self.processor
        kw = {"device": self.device} if self.device else {}
        if refs:
            raws = [_data_url(r, "image") for r in refs]
            media = {"images": [_image(raw, self.device) for raw in raws]}
            if self.device is not None:
                anew = any(not hasattr(im, "height") for im in media["images"])
            else:
                anew = any(raw.startswith(b"\xff\xd8") and raw.endswith(b"\xff\xd9") for raw in raws)
            keys = ("pixel_values", "image_position_ids")
        else:
            media = {"videos": [_frames(video, getattr(p.video_processor, "num_frames", 32))], "do_sample_frames": False}
            anew, keys = True, ("pixel_values_videos", "video_position_ids")
        tok = p.tokenizer
        head_text = tok.decode(head)
        tk = {"add_special_tokens": False} if tok.bos_token and head_text.startswith(tok.bos_token) else {}
        px = p(text=[head_text], padding=True, return_tensors="pt", **media, **tk, **kw)
        vision = {k: px[k] for k in keys if k in px}
        if anew:
            hid = px["input_ids"][0].tolist()
            out = []
            for row in rows:
                text = tok.decode(row)
                if not text.startswith(head_text):
                    raise ValueError("a row's text does not begin with the template's head")
                out.append(hid + tok(text[len(head_text):], add_special_tokens=False).input_ids)
            return out, vision
        ims = media["images"]
        counts = [int(c) for c in p._get_num_multimodal_tokens(
            image_sizes=[(im.height, im.width) for im in ims]).num_image_tokens]
        pad = p.image_token_id
        expanded, k = [], 0
        for t in head:
            if t == pad:
                expanded += [pad] * counts[k]
                k += 1
            else:
                expanded.append(t)
        return [expanded + row[len(head):] for row in rows], vision

    def plan(self, request: SystemOneRequest, **_) -> Plan:
        refs = getattr(request, "images", None)
        video = getattr(request, "video", None)
        if refs is not None and (not isinstance(refs, list) or not 1 <= len(refs) <= MAX_IMAGES):
            raise ValueError(f"images must contain 1 to {MAX_IMAGES} image data URLs")
        if video is not None:
            _data_url(video, "video")
            if refs is not None:
                raise ValueError("send images or video, not both in the same request")
        state = request.state
        state_ids = [] if state is None else self._e(render_state(state))
        head = self._media_head(len(refs)) if refs else self._media_head(0, True) if video is not None else self.head
        shared = head + self._e("Context:\n") + state_ids
        rows, data = [], []
        for qid, q in request.questions.items():
            ins, opts, keys = render_question(qid, q)
            rows.append((qid, q.type, keys, opts, ins, False))
        if self.two_order:
            rows += [(qid, kind, keys, opts[::-1], ins, True) for qid, kind, keys, opts, ins, _ in rows
                     if kind == "choice"]
        suffixes = []
        for qid, kind, keys, opts, ins, rev in rows:
            s = self._e(f"\n\nQuestion: {ins}\nOptions:")
            for j, o in enumerate(opts):
                s += self._e("\n(" + self.labels[j] + ") " + o)
            s += self.tail + self.answer
            if len(shared) + len(s) + 1 > self.max_context:
                if rev:
                    continue
                raise ValueError(f"{qid}: prompt of {len(shared) + len(s)} tokens exceeds the maximum context "
                                 f"length of {self.max_context}")
            suffixes.append(s)
            data.append((qid, kind, len(keys), rev))
        prefix, vision = shared, None
        if refs or video is not None:
            full, vision = self._media_rows(head, [shared + s for s in suffixes], refs, video)
            n = min(len(f) for f in full) - 1
            for i in range(n):
                if any(f[i] != full[0][i] for f in full):
                    n = i
                    break
            prefix, suffixes = full[0][:n], [f[n:] for f in full]
        longest = len(prefix) + max(len(s) for s in suffixes)
        if longest > self.max_context:
            raise ValueError(f"a prompt of {longest} tokens with its media exceeds the maximum context length of "
                             f"{self.max_context}")
        m = min(len(s) for s in suffixes) - 1
        for i in range(m):
            if any(s[i] != suffixes[0][i] for s in suffixes):
                m = i
                break
        forward = sum(len(prefix) + len(s) for s, row in zip(suffixes, data) if not row[3])
        return Plan(prefix + suffixes[0][:m], [s[m:] for s in suffixes], READ, {"rows": data, "forward": forward},
                    vision=vision)

    def finish(self, request: SystemOneRequest, plan: Plan, outputs, **_):
        logits, reverse = {}, {}
        for (qid, kind, n, rev), out in zip(plan.data["rows"], outputs):
            (reverse if rev else logits)[qid] = [float(x) for x in out[0][0][:n]]
        answers = {}
        for qid, q in request.questions.items():
            lp = logits[qid]
            if qid in reverse:
                pf, pr = _softmax(lp), _softmax(reverse[qid])
                k = len(lp)
                lp = [math.log(max((pf[j] + pr[k - 1 - j]) / 2, 1e-12)) for j in range(k)]
            p = _softmax(lp, self.temperatures[q.type])
            if q.type == "noul":
                answers[qid] = make_noul(p[1])
            elif q.type == "choice":
                a = make_choice(list(q.criteria), p)
                answers[qid] = a.model_copy(update={"confidence": _round(max(p))})
            else:
                a = make_score(list(q.criteria), p)
                answers[qid] = a.model_copy(update={"score": round(sum(i * x for i, x in enumerate(p)), 4),
                                                    "confidence": _round(max(p))})
        usage = Usage(input_tokens=plan.data["forward"], output_tokens=len(answers))
        return SystemOneResponse(model="", answers=answers, usage=usage), {"format": "xor",
                                                                          "prefix_tokens": len(plan.prefix),
                                                                          "branches": len(plan.suffixes)}

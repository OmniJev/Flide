"""Strands Decider family (StrandsAgents/strands-decider-*-gemma4-v2-*: Gemma 4 from E2B to 31B, dense and mixture of
experts), as the release's core renders and reads a request (strands_decider/core/prompt.py, readout.py).

A prompt is a state part and a question part, both Gemma 4 chat turns written out: `<bos><|turn>user\\n<state>\\n` the
state `</state>\\n`, then `<question type="...">` the kind's header line, the instructions, the options one per line
(`A. name — description`), `</question><turn|>\\n<|turn>model\\n`. The state part is the shared prefix and each question a
branch (attention "blocks": the state sees itself both ways, a question the state and itself both ways), on every layer
with no window (`no_window`). The branch is read at its last token against `readout.safetensors` (255 rows, one per answer
code, in 16 bits), soft-capped and scaled by the release's one temperature, and softmaxed over its options. A score
question's answer is the expected level.

A string is a state, an instruction or a description with its ends stripped (descriptions and names in one line);
structured content is indented JSON. Text that spells a control token stays text: the tokenizer matches no special
tokens. A request's images (data URLs or base64 PNG, JPEG, WebP or GIF) open the state, each as its begin token, one
placeholder per 48 x 48 block of its resized size (at most 280) and its end token. The pixels are the first frame,
upright, transparent areas on white, resized by Pillow's bicubic filter (width, then height), and go through the
checkpoint's image processor with its own resize off.
"""
from __future__ import annotations

import base64
import io
import json
import math
import re
from pathlib import Path
from typing import Any

from flide.api.answers import make_choice, make_noul, make_score
from flide.api.schema import SystemOneRequest, SystemOneResponse, Usage
from flide.formats import register
from flide.formats.base import Format, Plan, Read

INTERFACE = "strands-decider"
NOUL_KEYS = ("false", "true")
NOUL_DESCRIPTIONS = {"false": "the statement does not hold for this state", "true": "the statement holds for this state"}
HEADERS = {
    "noul": "Decide whether the statement is true of the state.",
    "choice": "Select exactly one option.",
    "score": "Rate the state against the ordered levels below (lowest first).",
}
BOS, TURN_OPEN, TURN_CLOSE = "<bos>", "<|turn>", "<turn|>"
IMAGE_OPEN, IMAGE_SOFT, IMAGE_CLOSE = "<|image>", "<|image|>", "<image|>"
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "GIF")
MAX_IMAGE_PIXELS, MAX_IMAGE_BYTES = 4096 * 4096, 16 * 2 ** 20
DATA_URI = re.compile(r"^data:image/[A-Za-z0-9_.+-]+;base64,")
SIDE, MAX_SOFT_TOKENS = 48, 280   # 16-pixel patches pooled 3 x 3 into one token
MAX_OPTIONS = 255
SCORE_LEVELS = (2, 10)
HEAD = "readout"


def content(x: Any) -> str:
    """A string stripped; structured content as indented JSON in its own key order."""
    if x is None:
        return ""
    return x.strip() if isinstance(x, str) else json.dumps(x, indent=2, ensure_ascii=False)


def option_line(code: str, name: str, description: str) -> str:
    name, description = " ".join(name.split()), " ".join(description.split())
    return f"{code}." + (f" {name}" if name else "") + (f" — {description}" if description else "") + "\n"


def resized(width: int, height: int) -> tuple[int, int]:
    """The (width, height) Gemma 4's processor gives an image: its aspect ratio, sides in multiples of 48, 280 tokens."""
    factor = math.sqrt(MAX_SOFT_TOKENS * SIDE * SIDE / (height * width))
    h, w = math.floor(factor * height / SIDE) * SIDE, math.floor(factor * width / SIDE) * SIDE
    if h == 0:
        h, w = SIDE, min(math.floor(width / height) * SIDE, MAX_SOFT_TOKENS * SIDE)
    elif w == 0:
        w, h = SIDE, min(math.floor(height / width) * SIDE, MAX_SOFT_TOKENS * SIDE)
    return w, h


def read_image(ref: Any):
    """(the image at the size the encoder takes, its placeholders): the release's checks, then its pixels."""
    from PIL import Image, ImageOps

    if not isinstance(ref, str):
        raise ValueError("images: each image is a data URL or base64 string")
    try:
        raw = base64.b64decode(DATA_URI.sub("", ref.strip()), validate=True)
    except ValueError as e:
        raise ValueError(f"image is not valid base64: {e}") from None
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"image is {len(raw)} bytes; at most {MAX_IMAGE_BYTES:,} bytes are accepted")
    try:
        im = Image.open(io.BytesIO(raw), formats=IMAGE_FORMATS)
        header = im.size
    except Exception as e:
        raise ValueError(f"image could not be read as {'/'.join(IMAGE_FORMATS)}: {e}") from None
    if min(header) < 1:
        raise ValueError(f"image could not be read: its header gives a size of {header[0]}x{header[1]}")
    if header[0] * header[1] > MAX_IMAGE_PIXELS:
        raise ValueError(f"image is {header[0]}x{header[1]}; at most {MAX_IMAGE_PIXELS:,} pixels are accepted")
    try:
        im.load()
        im = ImageOps.exif_transpose(im)
    except Exception as e:
        raise ValueError(f"image could not be decoded: {e}") from None
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    im = im.convert("RGB")
    if im.size not in (header, header[::-1]):
        raise ValueError(f"image could not be decoded: its pixels are {im.width}x{im.height}, and its header gives "
                         f"{header[0]}x{header[1]}")
    w, h = resized(*header)
    tokens = (w // SIDE) * (h // SIDE)
    w, h = resized(*im.size)
    if im.width != w:
        im = im.resize((w, im.height), Image.Resampling.BICUBIC)
    if im.height != h:
        im = im.resize((w, h), Image.Resampling.BICUBIC)
    return im, tokens


def _given(model: Any) -> dict:
    """The fields a request set on a pydantic object, as the request wrote them."""
    if model is None:
        return {}
    if isinstance(model, dict):
        return model
    return {k: getattr(model, k) for k in model.model_fields_set} | dict(model.model_extra or {})


def normalised(name: str, q: Any) -> tuple[str, Any, list[str], list[Any]]:
    """(kind, instructions, option names, descriptions), the options in the request's order, as the release's schema."""
    given = _given(q)
    kind, instructions, criteria = q.type, given.get("instructions"), given.get("criteria")
    if instructions is None:
        raise ValueError(f"question {name!r}: instructions are required")
    if kind == "noul":
        crit = _given(criteria)
        if set(crit) - set(NOUL_KEYS):
            raise ValueError(f'question {name!r}: noul criteria may only describe "false" and "true"')
        return kind, instructions, list(NOUL_KEYS), [crit.get(k, NOUL_DESCRIPTIONS[k]) for k in NOUL_KEYS]
    if kind == "choice":
        crit = _given(criteria)
        if not 2 <= len(crit) <= MAX_OPTIONS:
            raise ValueError(f"question {name!r}: choice criteria must map 2 to {MAX_OPTIONS} options")
        return kind, instructions, list(crit), list(crit.values())
    levels = list(criteria or [])
    if not SCORE_LEVELS[0] <= len(levels) <= SCORE_LEVELS[1] or any(not isinstance(x, str) for x in levels):
        raise ValueError(f"question {name!r}: score criteria must be 2 to 10 level texts")
    return kind, instructions, [str(i) for i in range(len(levels))], levels


class _Hidden:
    """The final-norm hidden states at the read rows, left on the device: `finish` takes the readout."""

    on_device = True

    def __call__(self, h):
        return h


@register("strands")
class Strands(Format):
    backbones = ("gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text")
    attention = "blocks"
    max_choice_options = MAX_OPTIONS
    max_score_levels = SCORE_LEVELS[1]
    backbone_options = {"no_window": True}

    def __init__(self, model_dir: str, tokenizer=None, max_request_tokens: int = 262144):
        import transformers

        self.model_dir = model_dir
        dc = json.loads((Path(model_dir) / "decision_config.json").read_text())
        self.codes = list(dc["codes"])
        self.temperature, self.softcap = float(dc["temperature"]), float(dc["final_logit_softcapping"])
        self.cap = int(dc["max_length_validated"])
        self.max_request_tokens = int(max_request_tokens)
        if len(self.codes) != MAX_OPTIONS:
            raise ValueError(f"strands: decision_config lists {len(self.codes)} codes; {MAX_OPTIONS} are needed")
        tok = tokenizer or transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True,
                                                                      trust_remote_code=False)
        self.tokenizer = tok
        self._bt = tok.backend_tokenizer
        self._bt.encode_special_tokens = True   # request text that spells a control token stays text
        self._bt.no_padding()
        self._bt.no_truncation()
        ids = {s: self._bt.token_to_id(s) for s in (BOS, TURN_OPEN, TURN_CLOSE, IMAGE_OPEN, IMAGE_SOFT, IMAGE_CLOSE)}
        if None in ids.values():
            raise ValueError("strands: the tokenizer lacks Gemma 4's control tokens")
        self._bos, self._open, self._close = ids[BOS], ids[TURN_OPEN], ids[TURN_CLOSE]
        self._image = ids[IMAGE_OPEN], ids[IMAGE_SOFT], ids[IMAGE_CLOSE]
        self.reads_images = "image" in dc.get("inputs", ())
        self._processor = None

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        return (decision_config or {}).get("interface") == INTERFACE

    @property
    def processor(self):
        """The checkpoint's image processor (images only); the 12B's exists on torchvision alone."""
        if self._processor is None:
            import transformers

            unified = json.loads((Path(self.model_dir) / "config.json").read_text()).get("model_type") == "gemma4_unified"
            cls = transformers.Gemma4UnifiedImageProcessor if unified else transformers.Gemma4ImageProcessorPil
            self._processor = cls.from_pretrained(self.model_dir, local_files_only=True)
        return self._processor

    def _ids(self, text: str) -> list[int]:
        return self._bt.encode(text, add_special_tokens=False).ids

    def state_ids(self, state: Any, images: list[int] = ()) -> list[int]:
        """`images`: the placeholders of each image, which open the state. Text between two control tokens is one encode."""
        text = content(state)
        out, run = [self._bos, self._open], "user\n<state>\n"
        begin, soft, end = self._image
        for n in images:
            out += [*self._ids(run), begin, *[soft] * n, end]
            run = "\n"
        return [*out, *self._ids(run + (f"{text}\n" if text else "") + "</state>\n")]

    def question_ids(self, kind: str, instructions: Any, options: list[str], descriptions: list[Any]) -> list[int]:
        text = content(instructions)
        lines = "".join(option_line(c, o, content(d)) for c, o, d in zip(self.codes, options, descriptions))
        body = (f'<question type="{kind}">\n{HEADERS[kind]}\n' + (f"{text}\n" if text else "") + "<options>\n" + lines
                + "</options>\nAnswer with the code of one option.\n</question>")
        return [*self._ids(body), self._close, *self._ids("\n"), self._open, *self._ids("model\n")]

    def load_heads(self, backbone) -> dict[str, Any]:
        from safetensors.torch import load_file

        from flide import settings

        if settings.batch_invariant:
            raise ValueError("the strands family does not run with --batch-invariant yet")
        rows = load_file(str(Path(self.model_dir) / "readout.safetensors"))["weight"]
        if rows.shape[0] != MAX_OPTIONS:
            raise ValueError(f"strands: readout.safetensors has {rows.shape[0]} rows; {MAX_OPTIONS} are needed")
        self.readout = rows.to(backbone.device).to(backbone.dtype)
        return {HEAD: _Hidden()}

    def plan(self, request: SystemOneRequest, **_) -> Plan:
        if getattr(request, "videos", None) or getattr(request, "audio", None) is not None:
            raise ValueError("strands: audio and video are not supported")
        given = _given(request)
        refs = given.get("images", [])
        if not isinstance(refs, list):
            raise ValueError("images: a list of data URLs or base64 strings")
        if "state" in given and not isinstance(request.state, (str, dict, list)):
            raise ValueError("state: a string, an object or a list")
        if refs and not self.reads_images:
            raise ValueError("this checkpoint does not take images")
        qs = list(request.questions.items())
        if not qs:
            raise ValueError("no questions")
        images = [read_image(r) for r in refs]
        prefix = self.state_ids(request.state, [n for _, n in images])
        branches, data = [], []
        for name, q in qs:
            kind, instructions, options, descriptions = normalised(name, q)
            ids = self.question_ids(kind, instructions, options, descriptions)
            if len(prefix) + len(ids) > self.cap:
                raise ValueError(f"question {name!r}: its prompt has {len(prefix) + len(ids)} tokens, over the "
                                 f"{self.cap} validated")
            branches.append(ids)
            data.append((name, kind, len(options), options))
        total = len(prefix) + sum(map(len, branches))
        if total > self.max_request_tokens:
            raise ValueError(f"request is {total} tokens, limit is {self.max_request_tokens}")
        vision = None
        if images:
            out = self.processor(images=[im for im, _ in images], do_resize=False, return_tensors="pt")
            vision = {k: out[k] for k in ("pixel_values", "image_position_ids")}
        return Plan(prefix, branches, (Read(HEAD),), data, vision=vision, attention="blocks")

    def probabilities(self, out, n: int) -> list[float]:
        """A branch's answer over its `n` codes: the readout in 16 bits, then in float32 the soft cap, the temperature and
        the softmax over the question's own codes."""
        import torch

        h = torch.as_tensor(out[0][0], device=self.readout.device)
        z = (h.to(self.readout.dtype)[None] @ self.readout.T)[0].float()[:n]
        return torch.softmax(torch.tanh(z / self.softcap) * self.softcap / self.temperature, -1).tolist()

    def finish(self, request: SystemOneRequest, plan: Plan, outputs, **_):
        answers = {}
        for (name, kind, n, options), out in zip(plan.data, outputs):
            probs = self.probabilities(out, n)
            if kind == "noul":
                answers[name] = make_noul(probs[NOUL_KEYS.index("true")])
            elif kind == "choice":
                answers[name] = make_choice(options, probs)
            else:
                answers[name] = make_score(list(request.questions[name].criteria), probs)
        usage = Usage(input_tokens=plan.tokens, output_tokens=len(answers))
        return SystemOneResponse(model="", answers=answers, usage=usage), {
            "format": "strands", "prefix_tokens": len(plan.prefix), "branches": len(plan.suffixes)}

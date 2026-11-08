"""Images and videos in a System One request, by the OneJev convention.

A request may carry `media`, a list of items the state refers to by placeholders `<image:N>` / `<video:N>` (1-based,
each item once, in order of appearance). This is the convention of the OneJev training rows, and
the rendering below matches the training prompts byte for byte, so a
request is seen exactly as training rows were:

    {"type": "image", "path" | "url" | "data": ...}
    {"type": "video", "frames": [ref, ...], "fps": 2.0}

A reference is base64 data (a `data:` URI or the bare string under "data"), a local path (relative to the engine's
media root, when it has one), an http(s) URL (when the engine allows them, `--allow-media-urls`: the server then fetches
any address it can reach), or, from Python, a PIL image. Each is at most `settings.max_media_bytes`. Video frames are
pre-extracted stills; `fps` sets the timestamps the model sees (default 1.0, as in training).
"""
from __future__ import annotations

import io
import math
import re
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from flide import settings
from flide.formats.base import media_bytes

PLACEHOLDER = re.compile(r"<(image|video):(\d+)>")
VIDEOS_KWARGS = {"do_sample_frames": False, "cap_pixels_per_frame": True}


class MediaError(ValueError):
    pass


def content_parts(text: str, media: Sequence[dict]) -> list[dict]:
    """Split a user-turn text at the media placeholders into chat content parts."""
    parts, pos, seen = [], 0, []
    for m in PLACEHOLDER.finditer(text):
        if m.start() > pos:
            parts.append({"type": "text", "text": text[pos:m.start()]})
        kind, n = m.group(1), int(m.group(2))
        if n < 1 or n > len(media) or media[n - 1]["type"] != kind:
            raise MediaError(f"placeholder {m.group(0)} does not match media {n} of {len(media)}")
        parts.append({"type": kind})
        seen.append(n)
        pos = m.end()
    if pos < len(text):
        parts.append({"type": "text", "text": text[pos:]})
    if seen != list(range(1, len(media) + 1)):
        raise MediaError(f"media must be referenced once each, in order, by <image:N> / <video:N>; saw {seen} for "
                         f"{len(media)} items")
    return parts


def render_prompt(processor, template_kwargs: dict, messages: list[dict], media: Sequence[dict]) -> str:
    """Chat-rendered prompt with one vision block per media item (pads are expanded later by the processor). Without
    media the text is exactly the text-only prompt."""
    user = messages[-1]["content"]
    msgs = messages[:-1] + [{"role": "user", "content": content_parts(user, media) if media else user}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **template_kwargs)


def check_items(media: Any) -> list[dict]:
    if media is None:
        return []
    if not isinstance(media, list):
        raise MediaError("media must be a list of {type: image | video, ...} items")
    out = []
    for i, m in enumerate(media):
        if not isinstance(m, dict) or m.get("type") not in ("image", "video"):
            raise MediaError(f"media[{i}] must be an object with type image or video")
        if m["type"] == "image" and not any(k in m for k in ("path", "url", "data", "image")):
            raise MediaError(f"media[{i}] (image) needs one of path, url or data")
        if m["type"] == "video" and not (isinstance(m.get("frames"), list) and m["frames"]):
            raise MediaError(f"media[{i}] (video) needs a non-empty frames list")
        out.append(m)
    return out


def _decode(raw: bytes):
    from PIL import Image

    try:
        with Image.open(io.BytesIO(raw)) as im:
            return im.convert("RGB")
    except Exception as exc:
        raise MediaError(f"not a readable image ({type(exc).__name__})") from None


def open_image(ref: Any, root: str | Path | None = None, allow_paths: bool = True, timeout: float = 20.0,
               allow_urls: bool = False):
    """One RGB PIL image from base64 data, a path, a URL or a PIL image (or a dict holding one of those)."""
    if isinstance(ref, dict):
        if "data" in ref:
            return _decode(media_bytes(str(ref["data"])))
        for key in ("image", "url", "path"):
            if key in ref:
                return open_image(ref[key], root, allow_paths, timeout, allow_urls)
        raise MediaError("image reference needs one of path, url or data")
    if hasattr(ref, "convert"):
        return ref.convert("RGB")
    if not isinstance(ref, str):
        raise MediaError(f"unsupported image reference of type {type(ref).__name__}")
    cap = settings.max_media_bytes
    if ref.startswith("data:"):
        return _decode(media_bytes(ref))
    if ref.startswith(("http://", "https://")):
        if not allow_urls:
            raise MediaError("media URLs are not fetched by this server; send base64 data")
        req = urllib.request.Request(ref, headers={"User-Agent": "flide"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read(cap + 1)
        except OSError as exc:
            raise MediaError(f"could not fetch {ref!r}: {exc}") from None
        if len(raw) > cap:
            raise MediaError(f"{ref!r} is larger than {cap / 2 ** 20:g} MiB")
        return _decode(raw)
    if not allow_paths:
        raise MediaError("local paths are not accepted by this server; send base64 data")
    p = Path(ref)
    if root is not None:
        base = Path(root).resolve()
        p = (p if p.is_absolute() else base / p).resolve()
        if base not in p.parents:
            raise MediaError(f"path {ref!r} is outside the media root")
    try:
        if p.stat().st_size > cap:
            raise MediaError(f"{ref!r} is larger than {cap / 2 ** 20:g} MiB")
        raw = p.read_bytes()
    except OSError as exc:
        raise MediaError(f"could not read {ref!r}: {exc.strerror or exc}") from None
    return _decode(raw)


def load(media: Sequence[dict], root: str | Path | None = None, allow_paths: bool = True, allow_urls: bool = False):
    """(images, videos, video metadata) in media order, the way training built them."""
    from transformers.video_utils import VideoMetadata

    images, videos, metas = [], [], []
    for m in media:
        if m["type"] == "image":
            images.append(open_image(m, root, allow_paths, allow_urls=allow_urls))
        else:
            frames = [open_image(f, root, allow_paths, allow_urls=allow_urls) for f in m["frames"]]
            n = len(frames)
            try:
                fps = float(m.get("fps") or 1.0)
            except (TypeError, ValueError):
                fps = math.nan
            if not (math.isfinite(fps) and fps > 0):
                raise MediaError(f"video fps {m.get('fps')!r} is not a positive number")
            videos.append(frames)
            metas.append(VideoMetadata(total_num_frames=n, fps=fps, frames_indices=list(range(n)), duration=n / fps))
    return images, videos, metas


def processor_kwargs(images, videos, metas) -> dict:
    kw: dict[str, Any] = {}
    if images:
        kw["images"] = images
    if videos:
        kw["videos"] = videos
        kw["video_metadata"] = metas
        kw["videos_kwargs"] = dict(VIDEOS_KWARGS)
    return kw

"""Model families are plugins. The core ships one, the generic `next_token` family, and imports no other by name.

A family package registers its Format subclass in the `flide.formats` entry point group:

    [project.entry-points."flide.formats"]
    myjev = "myjev_package:MyJev"

A script can also call `register` directly.

Which family serves a checkpoint, in order: the name given explicitly; the `format_id` in the
checkpoint's decision_config.json; the first family whose `detect` claims it; the built-in `next_token`
family, which reads any chat model at the first token of its reply (logged as such). A checkpoint whose
decision_config.json names a format that is not installed is refused, with the package to install.
"""
from __future__ import annotations

import json
import logging
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from .base import FINAL, Format, Plan, Read

log = logging.getLogger("flide.formats")

GROUP = "flide.formats"
_FORMATS: dict[str, type[Format]] = {}
_FORMAT_IDS: dict[str, str] = {}
_loaded = False


def register(name: str, format_ids: tuple[str, ...] = ()):
    """Class decorator: make a Format available under `name`, and under each decision_config format_id."""
    def wrap(cls):
        if not (isinstance(cls, type) and issubclass(cls, Format)):
            raise TypeError(f"{cls!r} is not a Format subclass")
        if name in _FORMATS and _FORMATS[name] is not cls:
            log.warning("format %r registered twice; keeping %s", name, _FORMATS[name].__module__)
            return cls
        cls.name = name
        _FORMATS[name] = cls
        for fid in format_ids:
            _FORMAT_IDS[fid] = name
        return cls
    return wrap


def load_plugins() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    from . import next_token
    for ep in entry_points(group=GROUP):
        try:
            ep.load()
        except Exception as exc:
            log.warning("format plugin %s (%s) failed to load: %s: %s", ep.name, ep.value, type(exc).__name__, exc)


def get(name: str) -> type[Format]:
    load_plugins()
    try:
        return _FORMATS[name]
    except KeyError:
        raise ValueError(f"unknown format {name!r}; installed: {', '.join(sorted(_FORMATS)) or 'none'}") from None


def names() -> list[str]:
    load_plugins()
    return sorted(_FORMATS)


def read_decision_config(model_dir: str) -> dict | None:
    path = Path(model_dir) / "decision_config.json"
    return json.loads(path.read_text()) if path.is_file() else None


def detect(model_dir: str, config: Any, name: str | None = None) -> type[Format]:
    if name:
        return get(name)
    load_plugins()
    dc = read_decision_config(model_dir)
    if dc and dc.get("format_id") in _FORMAT_IDS:
        return _FORMATS[_FORMAT_IDS[dc["format_id"]]]
    for cls in _FORMATS.values():
        if cls.detect(model_dir, config, dc):
            return cls
    if dc:
        msg = f"no installed format serves {model_dir} (decision_config.json format_id {dc.get('format_id')!r}"
        msg += f", install {dc['package']!r})" if dc.get("package") else ")"
        raise ValueError(f"{msg}; installed: {', '.join(sorted(_FORMATS))}")
    log.warning("no installed family claims %s; reading it as a chat model at the first token of its reply "
                "(format next_token). Pass --format to choose another.", model_dir)
    return _FORMATS["next_token"]


__all__ = ["FINAL", "Format", "Plan", "Read", "detect", "get", "load_plugins", "names", "read_decision_config",
           "register"]

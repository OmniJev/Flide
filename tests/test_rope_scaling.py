"""Rotary settings whose tables follow the longest position of a forward (dynamic NTK, LongRoPE) are refused at load;
static ones (default, linear, YaRN, Llama 3, proportional, Gemma's per layer type) load. CPU only.

    python tests/test_rope_scaling.py
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

from flide.runtime.engine import FileConfig, refuse_length_rope


def refused(cfg) -> bool:
    try:
        refuse_length_rope(cfg)
    except ValueError:
        return True
    return False


def main() -> int:
    cases = [
        ("default", SimpleNamespace(rope_parameters={"rope_type": "default", "rope_theta": 1e7}), False),
        ("yarn", SimpleNamespace(rope_parameters={"rope_type": "yarn", "factor": 4.0}), False),
        ("llama3", SimpleNamespace(rope_scaling={"rope_type": "llama3", "factor": 8.0}), False),
        ("linear, old key", SimpleNamespace(rope_scaling={"type": "linear", "factor": 2.0}), False),
        ("per layer type, static", SimpleNamespace(rope_parameters={
            "full_attention": {"rope_type": "proportional"}, "sliding_attention": {"rope_type": "default"}}), False),
        ("none", SimpleNamespace(), False),
        ("dynamic", SimpleNamespace(rope_parameters={"rope_type": "dynamic", "factor": 2.0}), True),
        ("dynamic, old key", SimpleNamespace(rope_scaling={"type": "dynamic", "factor": 2.0}), True),
        ("longrope", SimpleNamespace(rope_parameters={"rope_type": "longrope", "short_factor": [1.0]}), True),
        ("per layer type, one dynamic", SimpleNamespace(rope_parameters={
            "full_attention": {"rope_type": "dynamic"}, "sliding_attention": {"rope_type": "default"}}), True),
        ("text config inside", SimpleNamespace(get_text_config=lambda: SimpleNamespace(
            rope_parameters={"rope_type": "longrope"})), True),
        ("file config", FileConfig({"text_config": {"rope_scaling": {"type": "dynamic"}}}), True),
    ]
    ok = True
    for name, cfg, want in cases:
        got = refused(cfg)
        ok &= got == want
        print(f"{'PASS' if got == want else 'FAIL'} {name}: {'refused' if got else 'loads'}")
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

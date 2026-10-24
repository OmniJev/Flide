from __future__ import annotations

import time
from typing import Any, ClassVar


class Backbone:
    model_types: ClassVar[tuple[str, ...]] = ()

    device: Any
    dtype: Any
    num_layers: int

    @classmethod
    def load(cls, model_dir: str, device: str = "cuda:0", dtype: str = "bfloat16", **options) -> "Backbone":
        raise NotImplementedError

    def output_head(self) -> tuple[Any, Any]:
        """(weight [vocab, hidden], bias or None) of the output head, for families that read it."""
        raise NotImplementedError

    def forward(self, batch, pad_to: int | None = None) -> dict:
        """Hidden states at `batch.reads`: {layer key: [rows, hidden]}, rows in the order of the index tensor."""
        raise NotImplementedError


def capture_progress(i: int, n: int, t0: float, what: str, log_fn) -> None:
    """A line at every quarter of a capture, so a first start that compiles its kernels shows that it moves."""
    step = max(1, n // 4)
    if i < n and i % step == 0:
        log_fn(f"captured {i} of {n} {what}, {time.perf_counter() - t0:.0f}s so far")

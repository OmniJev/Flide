"""Calibration: temperatures per (primitive, option count) and the softmax that applies them."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class Calibration:
    """Temperature table. Keys are "kind" or "kind:K". Missing keys fall back to the primitive-level
    temperature, then to 1.0."""

    temperatures: dict[str, float] = field(default_factory=dict)
    source: str = "none"

    def temperature(self, kind: str, k: int) -> float:
        return self.temperatures.get(f"{kind}:{k}", self.temperatures.get(kind, 1.0))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text())
        return cls(temperatures={k: float(v) for k, v in data.get("temperatures", {}).items()}, source=str(path))


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    z = [float(x) / temperature for x in logits]
    m = max(z)
    w = [math.exp(x - m) for x in z]
    s = sum(w)
    return [x / s for x in w]

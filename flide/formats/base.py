"""What a model family hands the runtime, and what it gets back.

A family turns a request into a plan: one shared prefix, one suffix per branch, and for each branch
the reads to take. The runtime runs the backbone over packed plans, takes the hidden states each read
names, passes them through the named head and returns every branch's head outputs. The family turns
those into the response. The runtime knows nothing about prompts, answer slots or calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Sequence

FINAL = None


@dataclass(frozen=True)
class Read:
    """Hidden states of one branch, fed to one head.

    head: name of a head the family loaded; None returns the hidden states themselves.
    layer: FINAL (None) reads the output of the last layer after the final norm; an integer k reads the
        raw residual stream after decoder layer k (1-based; 0 is the input embeddings), and the backbone
        stops after the highest layer any read of the micro-batch needs.
    positions: offsets into the branch's own tokens, negative from its end; -1 is its last token.
    """

    head: str | None
    layer: int | None = FINAL
    positions: tuple[int, ...] = (-1,)


@dataclass
class Plan:
    """`reads` is one tuple of reads shared by every branch, or a list with one tuple per branch.
    `data` is the family's own bookkeeping for `finish`; the runtime only carries it.

    Media: `vision` holds the backbone's vision inputs for media inside the prefix (for qwen3_5 the
    processor's pixel tensors and grids); the prefix ids carry the placeholder tokens the features replace.
    `positions` are the prefix's rotary positions [3, P] when they are not 0..P-1, and suffix tokens start
    at position P + `offset`. The backbone fills both from `vision` when the family leaves them empty.

    An empty `prefix` makes every suffix a sequence of its own (nothing shared)."""

    prefix: list[int]
    suffixes: list[list[int]]
    reads: tuple[Read, ...] | list[tuple[Read, ...]]
    data: Any = None
    vision: Any = None
    positions: Any = None
    offset: int = 0

    @property
    def tokens(self) -> int:
        return len(self.prefix) + sum(len(s) for s in self.suffixes)

    def branch_reads(self, b: int) -> tuple[Read, ...]:
        return self.reads if isinstance(self.reads, tuple) else self.reads[b]


class Format:
    """A model family: how its checkpoints are asked and read. Subclasses register with
    `flide.formats.register` and are published to other installs through the `flide.formats`
    entry point group."""

    name: ClassVar[str] = ""
    backbones: ClassVar[tuple[str, ...]] = ()
    meta_key: ClassVar[str] = "meta"
    aliases: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        """Whether this family claims a checkpoint that does not name its format in decision_config.json."""
        return False

    def for_thread(self) -> "Format":
        """A twin that plans and finishes on another thread. It shares everything except the tokenizer, which keeps
        mutable truncation and padding state and fails ("Already borrowed") when two threads encode at once."""
        import copy

        twin = copy.copy(self)
        if getattr(self, "tokenizer", None) is not None:
            twin.tokenizer = copy.deepcopy(self.tokenizer)
        return twin

    def load_heads(self, backbone) -> dict[str, Any]:
        """Heads by name; each maps hidden states [rows, hidden] to outputs [rows, ...]."""
        raise NotImplementedError

    def plan(self, request, **options) -> Plan:
        raise NotImplementedError

    def finish(self, request, plan: Plan, outputs: Sequence[Sequence[Any]], **options):
        """`outputs[b][r]` is the head output of read r of branch b: one row per position, as lists."""
        raise NotImplementedError

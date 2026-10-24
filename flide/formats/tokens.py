"""Whole-text token ids of long prompts, tokenized in pieces.

Byte-level BPE tokenizers that split text with the GPT-style regex (Qwen 2 to 3.5, Llama 3) never merge tokens across
the regex's pieces, and the regex starts a new piece at two kinds of places whatever came before:

    after a newline, before a character that is not whitespace (the whitespace run, which ends at its last newline,
    or punctuation's trailing newlines end there);
    at a space after an ASCII letter or digit (the letter, digit or contraction piece ends at the space).

Both need only the characters just before and at the cut, and no added token holds a space or a newline, so a cut
never falls inside one.

BPE tokenizers that run the whole text as one word (Gemma 4: spaces become "▁", no split) merge two symbols only into
a vocabulary token, so they never merge across two neighbouring characters that no vocabulary token holds side by
side. Their cuts are at whitespace (before a space after another character, before or after a newline run) where the
two characters, spaces as "▁", are such a pair; no added token holds whitespace, so again no cut falls inside one.

Cut at such places, the pieces' ids put together are the whole text's ids, so a long text is tokenized as a batch of
pieces (in parallel where the tokenizers library may use threads) and the texts of one request that share a long
start tokenize the start once. `Pieces.ok` is False for any other tokenizer, which then tokenizes texts whole.
"""
from __future__ import annotations

import json
import re

CUT = re.compile(r"(?<=\n)(?=\S)|(?<=[A-Za-z0-9]) ")
WS_CUT = re.compile(r"(?<=[^ ])(?= )|(?<=\n)(?=[^\n])|(?<=[^\n])(?=\n)")
SPACE = "\u2581"
_HEAD = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
_TAIL = r"|\s*[\r\n]+|\s+(?!\S)|\s+"
PATTERNS = (
    _HEAD + r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*" + _TAIL,
    _HEAD + r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*" + _TAIL,
    _HEAD + r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*" + _TAIL,
)
PIECE = 1024
LONG = 4 * PIECE
SERIAL = 8


def splits_like_gpt(tokenizer) -> bool:
    """Whether the cuts of this module are exact for this tokenizer: NFC or no normalizer, the GPT-style split
    regex (isolated pieces) then byte-level, and no added token that holds a space or newline or strips them."""
    try:
        spec = json.loads(tokenizer.backend_tokenizer.to_str())
    except Exception:
        return False
    norm = spec.get("normalizer")
    if norm is not None and norm.get("type") != "NFC":
        return False
    pre = spec.get("pre_tokenizer") or {}
    steps = pre.get("pretokenizers", []) if pre.get("type") == "Sequence" else [pre]
    if len(steps) != 2:
        return False
    split, byte = steps
    if split.get("type") != "Split" or split.get("behavior") != "Isolated" or split.get("invert"):
        return False
    if (split.get("pattern") or {}).get("Regex") not in PATTERNS:
        return False
    if byte.get("type") != "ByteLevel" or byte.get("add_prefix_space") or byte.get("use_regex"):
        return False
    for t in spec.get("added_tokens", []):
        if t.get("lstrip") or t.get("rstrip") or any(c in t.get("content", "") for c in " \r\n"):
            return False
    return True


def one_word_pairs(tokenizer) -> set[str] | None:
    """For a BPE tokenizer that runs the whole text as one word (no normalizer, or spaces to "▁"; no pre-tokenizer,
    or a split at spaces) and whose added tokens hold no whitespace: every pair of neighbouring characters inside a
    vocabulary token (spaces as "▁"). None for any other tokenizer."""
    try:
        spec = json.loads(tokenizer.backend_tokenizer.to_str())
    except Exception:
        return None
    norm, pre, model = spec.get("normalizer"), spec.get("pre_tokenizer"), spec.get("model") or {}
    if norm is not None and not (norm.get("type") == "Replace" and norm.get("pattern") == {"String": " "}
                                 and norm.get("content") == SPACE):
        return None
    if pre is not None and not (pre.get("type") == "Split" and pre.get("pattern") == {"String": " "}
                                and pre.get("behavior") == "MergedWithPrevious" and not pre.get("invert")):
        return None
    if model.get("type") != "BPE" or model.get("dropout") or model.get("continuing_subword_prefix") \
            or model.get("end_of_word_suffix"):
        return None
    for t in spec.get("added_tokens", []):
        if t.get("lstrip") or t.get("rstrip") or any(c.isspace() for c in t.get("content", "")):
            return None
    return {s[i: i + 2] for s in model.get("vocab", {}) for i in range(len(s) - 1)}


class Pieces:
    def __init__(self, tokenizer):
        self.ok = splits_like_gpt(tokenizer)
        self.pairs = None if self.ok else one_word_pairs(tokenizer)
        if self.pairs is not None:
            self.ok = True

    def cuts(self, text: str, lo: int = 0, hi: int | None = None):
        """The cuts in text[lo:hi], in order (none when `ok` is False)."""
        if not self.ok:
            return
        hi = len(text) if hi is None else hi
        if self.pairs is None:
            for m in CUT.finditer(text, lo, hi):
                yield m.start()
            return
        for m in WS_CUT.finditer(text, lo, hi):
            c = m.start()
            if c and (text[c - 1] + text[c]).replace(" ", SPACE) not in self.pairs:
                yield c

    def last_cut(self, text: str, end: int) -> int:
        """The last cut at or before `end` (0 when there is none). A cut at c reads text[c - 1] and text[c] only, so it
        holds for every text that agrees with this one on its first end + 1 characters."""
        if not self.ok:
            return 0
        lo = max(0, end - 512)
        while True:
            best = 0
            for c in self.cuts(text, lo, end + 1):
                best = c
            if best or lo == 0:
                return best
            lo = max(0, lo - 4096)

    def encode(self, tokenizer, texts: list[str]) -> list[list[int]]:
        """Token ids of each text (no special tokens added), long ones in pieces, all in one batch."""
        bt = tokenizer.backend_tokenizer
        pieces, owner = [], []
        for k, t in enumerate(texts):
            parts = self._split(t) if self.ok and len(t) >= LONG else [t]
            pieces += parts
            owner += [k] * len(parts)
        if len(pieces) == len(texts) and len(texts) <= SERIAL:
            return [bt.encode(t, add_special_tokens=False).ids for t in texts]
        out: list[list[int]] = [[] for _ in texts]
        for k, e in zip(owner, bt.encode_batch(pieces, add_special_tokens=False)):
            out[k] += e.ids
        return out

    def _split(self, text: str) -> list[str]:
        parts, s = [], 0
        while len(text) - s > PIECE * 3 // 2:
            c = next(self.cuts(text, s + PIECE), None)
            if c is None:
                break
            parts.append(text[s: c])
            s = c
        parts.append(text[s:])
        return parts

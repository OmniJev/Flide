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

import bisect
import json
import logging
import re
import threading
from collections import OrderedDict

CUT = re.compile(r"(?<=\n)(?=\S)|(?<=[A-Za-z0-9]) ")
# whitespace places a one-word BPE may cut at (before a space after another character, before or after a newline run)
WS_CUT = re.compile(r"(?<=[^ ])(?= )|(?<=\n)(?=[^\n])|(?<=[^\n])(?=\n)")
SPACE = "\u2581"   # a space as such a tokenizer's vocabulary writes it
# the split patterns this holds for: Qwen 2 to 3.5 as tokenizer.json writes it, as transformers rebuilds it, Llama 3
_HEAD = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
_TAIL = r"|\s*[\r\n]+|\s+(?!\S)|\s+"
PATTERNS = (
    _HEAD + r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*" + _TAIL,
    _HEAD + r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*" + _TAIL,
    _HEAD + r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*" + _TAIL,
)
PIECE = 1024       # characters per piece of a long text
LONG = 4 * PIECE   # texts at least this long are tokenized in pieces
SERIAL = 8         # up to this many short texts are tokenized one by one
log = logging.getLogger("flide.tokens")


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
        if len(pieces) == len(texts) and len(texts) <= SERIAL:   # nothing long: a batch costs more than it saves
            return [bt.encode(t, add_special_tokens=False).ids for t in texts]
        out: list[list[int]] = [[] for _ in texts]
        for k, e in zip(owner, bt.encode_batch(pieces, add_special_tokens=False)):
            out[k] += e.ids
        return out

    def split(self, text: str) -> list[str]:
        """The pieces `encode` tokenizes a text in: a long text cut, any other whole."""
        return self._split(text) if self.ok and len(text) >= LONG else [text]

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


def common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix, by bisection on slices (compared in C; os.path.commonprefix walks the
    characters in Python, ~1 µs each, which made a lookup cost tens of milliseconds on long prompts)."""
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n          # a[:lo] == b[:lo], a[:hi] != b[:hi]
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
    stops agreeing with one of them (an agent's next step repeats the last one's prompt and adds to it). The reuse
    point is just after a newline that both prompts share, before a shared character that is not whitespace: the
    pre-tokenizer (GPT-style regex) always starts a new piece there and BPE never merges across pieces, so the result
    is the whole-prompt tokenization. Shared by a family's per-thread twins; `selfcheck` turns it off for a tokenizer
    that does not split after newlines."""

    def __init__(self, size: int = 16, min_chars: int = 512):
        self.items: OrderedDict[int, tuple[str, list[int], list[tuple[int, int]]]] = OrderedDict()
        self.size = size
        self.min_chars = min_chars
        self.lock = threading.Lock()
        self.enabled = True
        self.hits = self.misses = self.saved_chars = 0
        self._next = 0

    def selfcheck(self, tokenizer) -> None:
        """Turn the cache off for a tokenizer whose pieces do not restart after newline runs."""
        probe = ("{\n  \"a\": \"x y z\",\n\n  \"b\": [1, 22, 333],\n  \"ccc\": \"日本語 テスト\"\n}\n"
                 "Line two\t tabs   spaces\n\n\n   indented\nend\n \nblank\n") * 20
        whole = tokenizer(probe, add_special_tokens=False, return_offsets_mapping=True)
        starts = {o[0] for o in whole["offset_mapping"]}
        enc = lambda t: tokenizer.encode(t, add_special_tokens=False)   # noqa: E731
        for b in range(1, len(probe)):
            if probe[b - 1] == "\n" and not probe[b].isspace():
                if b not in starts or enc(probe[:b]) + enc(probe[b:]) != list(whole["input_ids"]):
                    self.enabled = False
                    log.info("token cache off: this tokenizer does not split after newline runs")
                    return

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
        b = text.rfind("\n", 0, c - 1)      # the boundary b must leave text[b] shared (b < c)
        while b > 0 and text[b + 1: b + 2].isspace():   # whitespace up to a later newline is one piece with it
            b = text.rfind("\n", 0, b)
        if b <= 0:
            return None
        b += 1                                # just after a newline; text[b] is not whitespace
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


def tokenize_offsets(tokenizer, pieces: Pieces, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Token ids and character offsets of a text; a long one cut where its tokenizer never merges across (`Pieces`)
    and its pieces tokenized in one batch, in parallel."""
    bt = tokenizer.backend_tokenizer
    parts = pieces.split(text)
    if len(parts) == 1:
        e = bt.encode(text, add_special_tokens=False)
        return list(e.ids), list(e.offsets)
    ids, offs, a = [], [], 0
    for p, e in zip(parts, bt.encode_batch(parts, add_special_tokens=False)):
        ids += e.ids
        offs += [(x + a, y + a) for x, y in e.offsets]
        a += len(p)
    return ids, offs


def encode_offsets(tokenizer, pieces: Pieces, cache: TokenCache, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Token ids and character offsets of a whole prompt, tokenizing only what follows the part it shares with a
    recent prompt in `cache`, a long remainder in pieces run in parallel (`tokenize_offsets`)."""
    hit = cache.lookup(text)
    if hit is None:
        ids, offs = tokenize_offsets(tokenizer, pieces, text)
        cache.misses += 1
    else:
        head_ids, head_offs, b = hit
        tail_ids, tail_offs = tokenize_offsets(tokenizer, pieces, text[b:])
        ids = head_ids + tail_ids
        offs = head_offs + [(x + b, y + b) for x, y in tail_offs]
        cache.hits += 1
        cache.saved_chars += b
    cache.add(text, ids, offs)
    return ids, offs

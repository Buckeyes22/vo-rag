"""Fail-closed content-corruption detector for vo-rag book ingestion.

Detects CID-font glyph-soup (e.g. `char+3` caesar-shift mojibake), control-character
contamination, and non-text garbage that page-level OCR-confidence gating
misses. Pure function — no I/O, no third-party deps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

_COMMON_WORDS: Final[frozenset[str]] = frozenset(
    [
        "the",
        "of",
        "and",
        "to",
        "a",
        "in",
        "is",
        "that",
        "for",
        "it",
        "with",
        "as",
        "was",
        "on",
        "are",
        "be",
        "this",
        "by",
        "an",
        "at",
        "or",
        "from",
        "but",
        "not",
        "which",
        "you",
        "all",
        "were",
        "when",
        "we",
        "there",
        "can",
        "has",
        "had",
        "more",
        "will",
        "would",
        "force",
        "vehicle",
        "tire",
        "model",
        "system",
        "data",
        "figure",
        "equation",
        "chapter",
    ]
)
_CONTROL_RE: Final = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_WORD_RE: Final = re.compile(r"[A-Za-z]{2,}")

_MAX_CONTROL_RATIO: Final[float] = 0.02
_MIN_SPACE_RATIO: Final[float] = 0.08
_MIN_DICT_HIT_RATIO: Final[float] = 0.04
_MAX_DIGIT_RATIO: Final[float] = 0.20
_MAX_NON_LATIN_RATIO: Final[float] = 0.10


@dataclass(frozen=True)
class CorruptionVerdict:
    is_corrupt: bool
    reason: str
    control_ratio: float
    space_ratio: float
    dict_hit_ratio: float


def _dict_hit_ratio(text: str) -> float:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if not words:
        return 0.0
    return sum(1 for w in words if w in _COMMON_WORDS) / len(words)


def _caesar_shift_recovers(text: str, shift: int = -3) -> bool:
    """CID signature: shifting letters by -3 recovers real words."""

    def sh(c: str) -> str:
        if "A" <= c <= "Z":
            return chr((ord(c) - ord("A") + shift) % 26 + ord("A"))
        if "a" <= c <= "z":
            return chr((ord(c) - ord("a") + shift) % 26 + ord("a"))
        return c

    shifted = "".join(sh(c) for c in text)
    return _dict_hit_ratio(shifted) > _dict_hit_ratio(text) + 0.10


def assess_text_corruption(text: str) -> CorruptionVerdict:
    if not text or not text.strip():
        return CorruptionVerdict(False, "empty", 0.0, 0.0, 0.0)
    n = len(text)
    control_ratio = len(_CONTROL_RE.findall(text)) / n
    space_ratio = sum(1 for c in text if c == " ") / n
    dict_ratio = _dict_hit_ratio(text)
    digit_ratio = sum(1 for c in text if c.isdigit()) / n
    non_latin_ratio = (
        sum(1 for c in text if c.isalpha() and not ("a" <= c <= "z" or "A" <= c <= "Z")) / n
    )

    if control_ratio > _MAX_CONTROL_RATIO:
        return CorruptionVerdict(True, "control_char", control_ratio, space_ratio, dict_ratio)
    if _caesar_shift_recovers(text):
        return CorruptionVerdict(True, "cid_caesar_shift", control_ratio, space_ratio, dict_ratio)
    if (
        n > 200
        and dict_ratio < _MIN_DICT_HIT_RATIO
        and space_ratio < _MIN_SPACE_RATIO
        and digit_ratio < _MAX_DIGIT_RATIO
        and non_latin_ratio < _MAX_NON_LATIN_RATIO
    ):
        return CorruptionVerdict(True, "non_dictionary", control_ratio, space_ratio, dict_ratio)
    return CorruptionVerdict(False, "ok", control_ratio, space_ratio, dict_ratio)

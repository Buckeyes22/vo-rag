"""Split prose at semantic-drift boundaries between adjacent sentences.

Sentence segmentation intentionally reuses the English-centric ``.?!`` punctuation
regex from :mod:`vo_rag.core.chunking.token_ceiling`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence

from vo_rag.core.chunking.counters import TokenCounterFn

__all__ = ["EmbedFn", "SemanticDriftSplitter"]

EmbedFn = Callable[[list[str]], list[Sequence[float]]]

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


class SemanticDriftSplitter:
    """Split text where adjacent sentence embeddings drift beyond a percentile."""

    def __init__(
        self,
        *,
        embed_fn: EmbedFn,
        counter: TokenCounterFn,
        max_tokens: int,
        breakpoint_percentile: float = 95.0,
        buffer_size: int = 1,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if not 0.0 <= breakpoint_percentile <= 100.0:
            raise ValueError("breakpoint_percentile must be between 0 and 100")
        if buffer_size < 0:
            raise ValueError("buffer_size must be non-negative")

        self.embed_fn = embed_fn
        self.counter = counter
        self.max_tokens = max_tokens
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size

    def split_text(self, content: str, section_path: str) -> list[str]:
        """Return prose pieces split by semantic drift and bounded by ``max_tokens``."""
        if not content.strip():
            return []

        sentences = self._sentences(content)
        if len(sentences) <= 1:
            return [content]

        embeddings = self.embed_fn(self._buffered_sentences(sentences))
        if len(embeddings) != len(sentences):
            raise ValueError("embed_fn must return one embedding per input text")

        distances = [
            self._cosine_distance(left, right)
            for left, right in zip(embeddings, embeddings[1:], strict=False)
        ]
        threshold = self._percentile(distances, self.breakpoint_percentile)
        boundaries = [index for index, distance in enumerate(distances) if distance > threshold]

        pieces: list[str] = []
        for segment in self._segments(sentences, boundaries):
            pieces.extend(self._split_at_ceiling(segment))
        return pieces

    @staticmethod
    def _sentences(content: str) -> list[str]:
        return [sentence.strip() for sentence in _SENTENCE_BOUNDARY_RE.split(content.strip())]

    def _buffered_sentences(self, sentences: list[str]) -> list[str]:
        buffered: list[str] = []
        for index in range(len(sentences)):
            start = max(0, index - self.buffer_size)
            end = min(len(sentences), index + self.buffer_size + 1)
            buffered.append(" ".join(sentences[start:end]))
        return buffered

    @staticmethod
    def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
        dot_product = 0.0
        left_norm_squared = 0.0
        right_norm_squared = 0.0
        for left_value, right_value in zip(left, right, strict=True):
            left_float = float(left_value)
            right_float = float(right_value)
            dot_product += left_float * right_float
            left_norm_squared += left_float * left_float
            right_norm_squared += right_float * right_float

        if left_norm_squared == 0.0 or right_norm_squared == 0.0:
            return 1.0

        similarity = dot_product / math.sqrt(left_norm_squared * right_norm_squared)
        similarity = max(-1.0, min(1.0, similarity))
        return 1.0 - similarity

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]

        rank = (percentile / 100.0) * (len(ordered) - 1)
        lower_index = math.floor(rank)
        upper_index = math.ceil(rank)
        if lower_index == upper_index:
            return ordered[lower_index]

        upper_weight = rank - lower_index
        lower_weight = 1.0 - upper_weight
        return (ordered[lower_index] * lower_weight) + (ordered[upper_index] * upper_weight)

    @staticmethod
    def _segments(sentences: list[str], boundaries: list[int]) -> list[list[str]]:
        segments: list[list[str]] = []
        start = 0
        for boundary in boundaries:
            segments.append(sentences[start : boundary + 1])
            start = boundary + 1
        segments.append(sentences[start:])
        return segments

    def _split_at_ceiling(self, sentences: list[str]) -> list[str]:
        segment_text = " ".join(sentences)
        if self.counter(segment_text) <= self.max_tokens:
            return [segment_text]

        pieces: list[str] = []
        current: list[str] = []
        for sentence in sentences:
            if not current:
                current.append(sentence)
                continue

            candidate = " ".join([*current, sentence])
            if self.counter(candidate) <= self.max_tokens:
                current.append(sentence)
                continue

            pieces.append(" ".join(current))
            current = [sentence]

        if current:
            pieces.append(" ".join(current))
        return pieces

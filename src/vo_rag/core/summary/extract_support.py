"""Extract-Support evidence selection (Ou & Lapata discipline).

BM25 tokenization strategy and ``EvidenceSpan`` payload shape are
stable — they are the recursion-invariant leg of the
Extract-Support contract (raw snippet copies from ``chunk.content``,
never paraphrases, never summary text).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal, TypeGuard

import structlog

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.retrieval.bm25_tokenizer import tokenize
from vo_rag.core.summary.prompts import SYSTEM_PROMPTS, USER_TEMPLATES
from vo_rag.core.summary.schemas import EvidenceSpan, NodeLevel

log = structlog.get_logger(__name__)

EvidenceMethod = Literal["bm25", "centroid_sim", "hybrid"]

_MIN_SNIPPET_TOKENS = 12
_MAX_SNIPPET_TOKENS = 80
_BM25_K1 = 1.5
_BM25_B = 0.75
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])|(?:\n{2,})")
_WHITESPACE_RE = re.compile(r"\s+")
_DENSE_VECTOR_METADATA_KEYS = (
    "dense",
    "dense_vector",
    "embedding",
    "embedding_dense",
    "bge_m3_dense",
)


@dataclass(frozen=True)
class _CandidateSnippet:
    chunk: RetrievalChunk
    text: str
    source_order: int
    token_counts: Counter[str]
    token_length: int
    bm25_score: float = 0.0
    centroid_score: float = 0.0
    final_score: float = 0.0


def select_evidence_snippets(
    chunks: list[RetrievalChunk],
    *,
    topic_or_query: str,
    k: int,
    method: EvidenceMethod = "hybrid",
) -> list[EvidenceSpan]:
    """Return top-k extractive snippets from raw chunks for Extract-Support.

    The function only accepts raw RetrievalChunk objects, so it cannot pull
    evidence from recursive summaries. Snippets are copied spans from
    chunk.content and are never paraphrased.
    """

    if k <= 0 or not chunks:
        return []

    candidates = _candidate_snippets(chunks)
    if not candidates:
        return []

    query_tokens = tokenize(topic_or_query)
    bm25_scores = _bm25_scores(candidates, query_tokens)
    centroid_scores = _centroid_scores(chunks)
    has_centroid_scores = any(score > 0.0 for score in centroid_scores.values())

    scored_candidates = []
    for candidate, bm25_score in zip(candidates, bm25_scores, strict=True):
        centroid_score = centroid_scores.get(candidate.chunk.chunk_id, 0.0)
        final_score = _combined_score(
            bm25_score=bm25_score,
            centroid_score=centroid_score,
            method=method,
            has_centroid_scores=has_centroid_scores,
        )
        scored_candidates.append(
            _CandidateSnippet(
                chunk=candidate.chunk,
                text=candidate.text,
                source_order=candidate.source_order,
                token_counts=candidate.token_counts,
                token_length=candidate.token_length,
                bm25_score=bm25_score,
                centroid_score=centroid_score,
                final_score=final_score,
            )
        )

    actual_method = _actual_method(method, has_centroid_scores)
    selected = sorted(scored_candidates, key=lambda item: (-item.final_score, item.source_order))[
        :k
    ]
    return [
        EvidenceSpan(
            chunk_id=item.chunk.chunk_id,
            section_path=item.chunk.section_path,
            text=item.text,
            score=item.final_score,
            selection_method=actual_method,
        )
        for item in selected
    ]


def build_extract_support_prompt(
    *,
    level: NodeLevel,
    primary_input: str,
    supporting_evidence: list[EvidenceSpan],
    display_path: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) ready for the LLM call."""

    system_prompt = SYSTEM_PROMPTS[level]
    user_template = USER_TEMPLATES[level]
    evidence_json = json.dumps(
        [span.model_dump(mode="json") for span in supporting_evidence],
        indent=2,
        sort_keys=True,
    )
    user_prompt = user_template.format(
        primary_input=primary_input,
        supporting_evidence=evidence_json,
        section_breadcrumb=display_path,
    )
    return system_prompt, user_prompt


def _candidate_snippets(chunks: list[RetrievalChunk]) -> list[_CandidateSnippet]:
    candidates: list[_CandidateSnippet] = []
    seen: set[str] = set()
    source_order = 0

    for chunk in chunks:
        for sentence in _sentences_or_spans(chunk.content):
            normalized = _normalize_snippet(sentence)
            if not normalized or normalized in seen:
                continue
            tokens = tokenize(normalized)
            if len(tokens) < _MIN_SNIPPET_TOKENS:
                continue
            seen.add(normalized)
            candidates.append(
                _CandidateSnippet(
                    chunk=chunk,
                    text=normalized,
                    source_order=source_order,
                    token_counts=Counter(tokens),
                    token_length=len(tokens),
                )
            )
            source_order += 1

    if candidates:
        return candidates

    return _fallback_spans(chunks)


def _sentences_or_spans(text: str) -> list[str]:
    normalized_text = text.strip()
    if not normalized_text:
        return []

    spans: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(normalized_text):
        sentence = sentence.strip()
        if not sentence:
            continue
        token_count = len(tokenize(sentence))
        if token_count <= _MAX_SNIPPET_TOKENS:
            spans.append(sentence)
            continue
        spans.extend(_windowed_spans(sentence, _MAX_SNIPPET_TOKENS))
    return spans


def _windowed_spans(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    if len(words) <= max_tokens:
        return [text]
    return [
        " ".join(words[index : index + max_tokens]) for index in range(0, len(words), max_tokens)
    ]


def _fallback_spans(chunks: list[RetrievalChunk]) -> list[_CandidateSnippet]:
    candidates: list[_CandidateSnippet] = []
    for source_order, chunk in enumerate(chunks):
        text = _normalize_snippet(" ".join(chunk.content.split()[:_MAX_SNIPPET_TOKENS]))
        if not text:
            continue
        tokens = tokenize(text)
        candidates.append(
            _CandidateSnippet(
                chunk=chunk,
                text=text,
                source_order=source_order,
                token_counts=Counter(tokens),
                token_length=max(len(tokens), 1),
            )
        )
    return candidates


def _normalize_snippet(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _bm25_scores(candidates: list[_CandidateSnippet], query_tokens: list[str]) -> list[float]:
    if not candidates or not query_tokens:
        return [0.0 for _ in candidates]

    query_counts = Counter(query_tokens)
    doc_freqs: Counter[str] = Counter()
    for candidate in candidates:
        doc_freqs.update(candidate.token_counts.keys())

    doc_count = len(candidates)
    avg_doc_length = sum(candidate.token_length for candidate in candidates) / doc_count
    scores: list[float] = []

    for candidate in candidates:
        score = 0.0
        doc_length = candidate.token_length
        length_norm = _BM25_K1 * (1.0 - _BM25_B + _BM25_B * doc_length / avg_doc_length)
        for token, query_count in query_counts.items():
            term_freq = candidate.token_counts.get(token, 0)
            if term_freq == 0:
                continue
            doc_freq = doc_freqs[token]
            idf = math.log(1.0 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
            term_score = idf * (term_freq * (_BM25_K1 + 1.0)) / (term_freq + length_norm)
            score += term_score * query_count
        scores.append(score)

    return _normalize_scores(scores)


def _centroid_scores(chunks: list[RetrievalChunk]) -> dict[str, float]:
    vectors: dict[str, list[float]] = {}
    for chunk in chunks:
        vector = _dense_vector_from_metadata(chunk.metadata)
        if vector is not None:
            vectors[chunk.chunk_id] = vector

    if not vectors:
        return {}

    vector_lengths = {len(vector) for vector in vectors.values()}
    if len(vector_lengths) != 1:
        return {}

    dim = vector_lengths.pop()
    centroid = [0.0 for _ in range(dim)]
    for vector in vectors.values():
        for index, value in enumerate(vector):
            centroid[index] += value
    centroid = [value / len(vectors) for value in centroid]

    raw_scores = {
        chunk_id: _cosine_similarity(vector, centroid) for chunk_id, vector in vectors.items()
    }
    normalized_values = _normalize_scores(list(raw_scores.values()))
    return dict(zip(raw_scores.keys(), normalized_values, strict=True))


def _dense_vector_from_metadata(metadata: dict[Any, Any]) -> list[float] | None:
    for key in _DENSE_VECTOR_METADATA_KEYS:
        value = metadata.get(key)
        if _is_float_vector(value):
            return [float(item) for item in value]
    return None


def _is_float_vector(value: Any) -> TypeGuard[list[int | float]]:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, int | float) for item in value)
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(lval * rval for lval, rval in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _combined_score(
    *,
    bm25_score: float,
    centroid_score: float,
    method: EvidenceMethod,
    has_centroid_scores: bool,
) -> float:
    if method == "bm25" or not has_centroid_scores:
        return bm25_score
    if method == "centroid_sim":
        return centroid_score
    return (0.75 * bm25_score) + (0.25 * centroid_score)


def _actual_method(method: EvidenceMethod, has_centroid_scores: bool) -> str:
    if method in {"centroid_sim", "hybrid"} and not has_centroid_scores:
        return "bm25"
    return method


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if math.isclose(min_score, max_score):
        return [1.0 if max_score > 0.0 else 0.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]

"""book_rag retrieval eval wiring for the regression gate (VOR-7 §5).

The day-one eval gate (`scripts/run_eval_gate.py`, run by `ci.sh`) auto-discovers
ACTIVE pipelines and, for each, requires a fixture set + a registered evaluator
``(RetrievalVariant, SearchFn)``. This module supplies book_rag's evaluator.

Two explicit evaluator modes:

- ``replay_smoke`` — the **CI-safe, offline** path. It returns, for each query,
  the qrels-relevant chunk_ids ranked by grade. It needs no Qdrant / GPU and
  proves fixture/qrels/metric plumbing only. It does **not** measure live
  retrieval quality.

- ``retrieval_gate`` — the real retrieval-quality gate. It searches a corpus
  snapshot whose chunk IDs match ``qrels.tsv`` and whose chunk TEXT comes from
  the real corpus (see the anti-fabrication contract in
  ``scripts/rebuild_book_rag_eval_corpus.py`` — a snapshot synthesized from the
  queries/qrels themselves makes this gate a tautology and is forbidden).
  Until that snapshot is re-grounded from the live corpus, this mode reports
  ``blocked_missing_corpus_snapshot`` (or ``blocked_missing_retriever``) and
  never falls back to replay.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Self

from vo_rag.core.eval.harness import RetrievalVariant, SearchFn
from vo_rag.core.eval.passages import RelevanceMode, RetrievedDocument
from vo_rag.core.eval.qrels import load_qrels_tsv

BOOK_RAG_EVAL_COLLECTION = "vo_rag_books_sterile_v1"
BOOK_RAG_EVAL_CORPUS_SNAPSHOT = "corpus_snapshot.jsonl"
BOOK_RAG_BLOCKED_MISSING_CORPUS_SNAPSHOT: Final = "blocked_missing_corpus_snapshot"
BOOK_RAG_BLOCKED_MISSING_RETRIEVER: Final = "blocked_missing_retriever"
BOOK_RAG_BLOCKED_MODES: Final = frozenset(
    {
        BOOK_RAG_BLOCKED_MISSING_CORPUS_SNAPSHOT,
        BOOK_RAG_BLOCKED_MISSING_RETRIEVER,
    }
)
BookRagEvalMode = Literal["replay_smoke", "retrieval_gate"]
BookRagReportedMode = Literal[
    "replay_smoke",
    "retrieval_gate",
    "blocked_missing_corpus_snapshot",
    "blocked_missing_retriever",
]
_TOKEN_RE: Final = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class _SnapshotChunk:
    chunk_id: str
    source_id: str
    text: str
    terms: frozenset[str]


@dataclass(frozen=True, slots=True)
class _SnapshotSearchHit:
    id: str
    score: float
    payload: dict[str, Any]


def _terms(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(text.lower()))


class SnapshotBookRagRetriever:
    """Offline retriever over the frozen book_rag eval corpus snapshot."""

    def __init__(self, snapshot_path: Path | str) -> None:
        self._chunks = _load_snapshot_chunks(Path(snapshot_path))

    async def hybrid_search(
        self,
        query: str,
        *,
        collection: str,
        top_k: int = 20,
        prefetch_depth: int = 100,
        **_: Any,
    ) -> list[_SnapshotSearchHit]:
        query_terms = _terms(query)
        scored: list[tuple[float, _SnapshotChunk]] = []
        for chunk in self._chunks:
            # Plain term-overlap lexical scoring, deliberately: no verbatim
            # query-substring bonus. Such a bonus only changes rankings when
            # chunks CONTAIN the query text — i.e. when the snapshot was
            # fabricated from the queries — and its absence keeps this
            # retriever honest over a real corpus snapshot.
            overlap = len(query_terms & chunk.terms)
            score = overlap / max(len(query_terms), 1)
            if score > 0:
                scored.append((score, chunk))

        ranked = sorted(scored, key=lambda item: (-item[0], item[1].chunk_id))[:top_k]
        return [
            _SnapshotSearchHit(
                id=chunk.chunk_id,
                score=score,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "text": chunk.text,
                    "collection": collection,
                    "prefetch_depth": prefetch_depth,
                },
            )
            for score, chunk in ranked
        ]


class BookRagEvaluatorEntry(tuple[RetrievalVariant, SearchFn]):
    """Tuple-compatible evaluator entry with an explicit operational mode."""

    mode: BookRagReportedMode
    relevance_mode: RelevanceMode

    def __new__(
        cls,
        variant: RetrievalVariant,
        search_fn: SearchFn,
        mode: BookRagReportedMode,
        relevance_mode: RelevanceMode = "chunk_id",
    ) -> Self:
        entry = super().__new__(cls, (variant, search_fn))
        entry.mode = mode
        entry.relevance_mode = relevance_mode
        return entry

    @property
    def variant(self) -> RetrievalVariant:
        return self[0]

    @property
    def search_fn(self) -> SearchFn:
        return self[1]


def book_rag_variant() -> RetrievalVariant:
    """The book_rag retrieval variant: top_k=10, prefetch_k=100, reranker on."""
    return RetrievalVariant(
        name="book_rag",
        top_k=10,
        prefetch_k=100,
        reranker_enabled=True,
    )


def _query_text_to_id(queries_path: Path) -> dict[str, str]:
    """Map each query's text -> query_id (accepts ``text`` or ``query_text``)."""
    out: dict[str, str] = {}
    for raw in queries_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        text = record.get("text") or record.get("query_text", "")
        out[str(text)] = str(record["query_id"])
    return out


def _load_snapshot_chunks(snapshot_path: Path) -> list[_SnapshotChunk]:
    chunks: list[_SnapshotChunk] = []
    for raw in snapshot_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        text = str(record["text"])
        chunks.append(
            _SnapshotChunk(
                chunk_id=str(record["chunk_id"]),
                source_id=str(record["source_id"]),
                text=text,
                terms=_terms(text),
            )
        )
    return chunks


def build_book_rag_snapshot_retriever(snapshot_path: Path | str) -> SnapshotBookRagRetriever:
    """Build the offline retrieval-gate retriever from ``corpus_snapshot.jsonl``."""
    return SnapshotBookRagRetriever(snapshot_path)


def build_replay_search_fn(queries_path: Path | str, qrels_path: Path | str) -> SearchFn:
    """Offline, deterministic search_fn keyed by query text.

    For a query text it returns the qrels-relevant chunk_ids ordered by grade
    (desc), then by id for stability. No external services. See module docstring
    for the CI-vs-integration gate split.
    """
    qrels = load_qrels_tsv(qrels_path, pipeline="book_rag", collection="eval")
    by_query = qrels.by_query()  # {query_id: {doc_id: grade}}
    text_to_id = _query_text_to_id(Path(queries_path))

    ranked_by_text: dict[str, list[str]] = {}
    for text, query_id in text_to_id.items():
        graded = by_query.get(query_id, {})
        ranked_by_text[text] = [
            doc_id for doc_id, _ in sorted(graded.items(), key=lambda kv: (-kv[1], kv[0]))
        ]

    def _search(query: str, top_k: int) -> list[str]:
        return ranked_by_text.get(query, [])[:top_k]

    return _search


def build_live_search_fn(
    retriever: Any,
    *,
    collection: str = BOOK_RAG_EVAL_COLLECTION,
    prefetch_depth: int = 100,
    relevance_mode: RelevanceMode = "chunk_id",
) -> SearchFn:
    """Wrap a ``HybridRetriever``-shaped object into a harness ``SearchFn``.

    Returns chunk_ids (from the result payload, falling back to the point id).
    Used by both the snapshot-backed retrieval gate and live Qdrant retrievers.
    """

    async def _search(query: str, top_k: int) -> list[str | RetrievedDocument]:
        results = await retriever.hybrid_search(
            query,
            collection=collection,
            top_k=top_k,
            prefetch_depth=prefetch_depth,
        )
        out: list[str | RetrievedDocument] = []
        for r in results:
            payload = getattr(r, "payload", None) or {}
            payload_dict = _payload_dict(payload)
            chunk_id = payload_dict.get("chunk_id") or getattr(r, "id", None)
            if chunk_id is not None:
                if relevance_mode == "passage":
                    out.append(RetrievedDocument(doc_id=str(chunk_id), payload=payload_dict))
                else:
                    out.append(str(chunk_id))
        return out

    return _search


def _blocked_missing_corpus_snapshot_search_fn(_query: str, _top_k: int) -> list[str]:
    return []


def book_rag_evaluator(
    fixture_dir: Path | str,
    *,
    retriever: Any | None = None,
    mode: str = "replay_smoke",
    relevance_mode: RelevanceMode = "chunk_id",
    collection: str | None = None,
) -> BookRagEvaluatorEntry:
    """Build the evaluator entry for the selected book_rag eval mode.

    ``replay_smoke`` proves offline plumbing. ``retrieval_gate`` is fail-closed:
    without the re-grounded corpus snapshot, it returns a blocked entry instead
    of replaying qrels.
    """
    fixture_dir = Path(fixture_dir)
    variant = book_rag_variant()
    qrels_filename = "qrels_passage.tsv" if relevance_mode == "passage" else "qrels.tsv"
    search_collection = collection or BOOK_RAG_EVAL_COLLECTION
    if mode == "replay_smoke":
        return BookRagEvaluatorEntry(
            variant,
            build_replay_search_fn(fixture_dir / "queries.jsonl", fixture_dir / qrels_filename),
            "replay_smoke",
            relevance_mode,
        )
    if mode == "retrieval_gate":
        snapshot_path = fixture_dir / BOOK_RAG_EVAL_CORPUS_SNAPSHOT
        if not snapshot_path.is_file():
            return BookRagEvaluatorEntry(
                variant,
                _blocked_missing_corpus_snapshot_search_fn,
                BOOK_RAG_BLOCKED_MISSING_CORPUS_SNAPSHOT,
                relevance_mode,
            )
        if retriever is None:
            return BookRagEvaluatorEntry(
                variant,
                _blocked_missing_corpus_snapshot_search_fn,
                BOOK_RAG_BLOCKED_MISSING_RETRIEVER,
                relevance_mode,
            )
        return BookRagEvaluatorEntry(
            variant,
            build_live_search_fn(
                retriever,
                collection=search_collection,
                relevance_mode=relevance_mode,
            ),
            "retrieval_gate",
            relevance_mode,
        )
    raise ValueError(f"unknown book_rag eval mode: {mode}")


def _payload_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


__all__ = [
    "BOOK_RAG_EVAL_COLLECTION",
    "BOOK_RAG_EVAL_CORPUS_SNAPSHOT",
    "BOOK_RAG_BLOCKED_MISSING_CORPUS_SNAPSHOT",
    "BOOK_RAG_BLOCKED_MISSING_RETRIEVER",
    "BOOK_RAG_BLOCKED_MODES",
    "BookRagEvalMode",
    "BookRagEvaluatorEntry",
    "BookRagReportedMode",
    "SnapshotBookRagRetriever",
    "book_rag_evaluator",
    "book_rag_variant",
    "build_book_rag_snapshot_retriever",
    "build_live_search_fn",
    "build_replay_search_fn",
]

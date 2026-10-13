"""Per-source self-retrieval smoke gate — measures the ingestion *ceiling*.

The structural audit catches gross parse/chunk failures; this catches the subtle
ones: a book can parse cleanly yet produce chunks that are mis-embedded or
semantically incoherent, and you only discover it at query time. This gate probes
that at ingest, per source, isolated from the (shared, downstream) architecture
eval.

Core idea — *self-retrieval*: generate a few synthetic queries **from a book's own
chunks**, then retrieve **restricted to that same book** and check the generating
chunk comes back in top-k. If a query generated *from* chunk C can't retrieve C
*from its own book*, that is an ingestion/embedding/chunking defect, not an
architecture problem.

This module is pure: the LLM (query generation) and the retriever are both
**injected**. The score math and gate policy here are unit-testable with fakes; the
real ``LLMClient`` and Qdrant ``HybridRetriever`` are wired in by the backfill CLI
and the workflow step.

``self_retrieval_recall@k = (# queries whose expected supporting chunk ∈ top-k) / N``
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Sequence
from enum import StrEnum
from typing import Any, Protocol

import structlog
from pydantic import Field

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.models.common import VoBaseModel
from vo_rag.pipelines.book_rag.synthetic_queries import (
    SyntheticQueryConfig,
    generate_query_batch,
)

log = structlog.get_logger(__name__)


# A retriever is injected as ``(query_text, top_k) -> [chunk_id, ...]`` in rank
# order. The returned identifiers MUST be in the same namespace as a query's
# ``expected_supporting_chunk_ids`` (i.e. RetrievalChunk.chunk_id). Mapping a live
# ``HybridRetriever`` result onto chunk_ids is the adapter's job, not this
# module's.
RetrieveFn = Callable[[str, int], Awaitable[list[str]]]

# An LLM call matching the synthetic-query generator's injected signature:
# ``(system, user, max_tokens, temperature) -> LLMResponse``.
LLMCall = Callable[[str, str, int, float], Awaitable[LLMResponse]]


class _ScorableQuery(Protocol):
    """The minimal surface the scorer reads off a query.

    PilotQuery satisfies this; tests use a lighter stub. Keeping it structural
    means the scorer never depends on PilotQuery's validators.
    """

    query_id: str
    query_text: str
    expected_supporting_chunk_ids: list[str]


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


class SmokeQueryResult(VoBaseModel):
    """Per-query outcome of the self-retrieval probe."""

    query_id: str
    query_text: str
    expected_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    hit_rank: int | None  # 1-based rank of the first expected chunk found, else None
    hit: bool


class SmokeReport(VoBaseModel):
    """Aggregate self-retrieval result for one source."""

    source_id: str
    n_requested: int  # queries handed to the scorer
    n_scored: int  # queries actually scorable (>=1 expected chunk id)
    top_k: int
    score: float  # self_retrieval_recall@k = hits / n_scored (0.0 when n_scored == 0)
    per_query: list[SmokeQueryResult] = Field(default_factory=list)


class GateDecision(StrEnum):
    PASS = "pass"
    QUARANTINE = "quarantine"
    FAIL = "fail"


class GateResult(VoBaseModel):
    """Decision + the numbers behind it, for logging and provenance."""

    decision: GateDecision
    reason: str
    score: float
    n_scored: int


# ---------------------------------------------------------------------------
# Query generation (wraps the benchmark generator)
# ---------------------------------------------------------------------------


async def generate_self_queries(
    *,
    all_chunks: list[RetrievalChunk],
    n: int,
    llm_call: LLMCall,
    seed: int,
    config: SyntheticQueryConfig | None = None,
    max_concurrent: int = 4,
) -> list[Any]:
    """Generate up to *n* synthetic queries from a source's own chunks.

    Wraps ``synthetic_queries.generate_query_batch`` and keeps only **scorable**
    queries: generation must have succeeded (``record.query is not None``) and the
    query must carry at least one ``expected_supporting_chunk_ids`` (otherwise there
    is nothing to self-retrieve). Dropped records are logged, not raised.

    ``llm_call`` is injected by the caller. The caller is responsible for disabling
    reasoning output (``enable_thinking=False`` / reasoning-strip) in that callable —
    this module never makes a live LLM call.

    Returns the surviving ``PilotQuery`` objects (typed ``Any`` so callers/tests are
    not forced through PilotQuery's validators).
    """
    cfg = config or SyntheticQueryConfig()
    records = await generate_query_batch(
        all_chunks=all_chunks,
        n_queries=n,
        config=cfg,
        seed=seed,
        llm_call=llm_call,
        max_concurrent=max_concurrent,
    )

    queries: list[Any] = []
    n_failed = 0
    n_unscorable = 0
    for record in records:
        if record.query is None:
            n_failed += 1
            continue
        if not record.query.expected_supporting_chunk_ids:
            n_unscorable += 1
            continue
        queries.append(record.query)

    log.info(
        "book_rag.retrieval_smoke.generate",
        requested=n,
        kept=len(queries),
        failed=n_failed,
        unscorable=n_unscorable,
    )
    return queries


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _first_expected_rank(retrieved: Sequence[str], expected: set[str]) -> int | None:
    """1-based rank of the first retrieved id that is an expected chunk, else None."""
    for idx, cid in enumerate(retrieved):
        if cid in expected:
            return idx + 1
    return None


async def score_self_retrieval(
    *,
    queries: Sequence[_ScorableQuery],
    retrieve_fn: RetrieveFn,
    top_k: int,
    source_id: str,
) -> SmokeReport:
    """Run each query through *retrieve_fn* and score self-retrieval recall@k.

    A query *hits* when **any** of its ``expected_supporting_chunk_ids`` appears in
    the top-*k* retrieved ids. Queries with no expected ids are skipped (they cannot
    be scored) and excluded from ``n_scored`` — this keeps a generation gap from
    masquerading as an unretrievable-chunk failure.

    ``score = hits / n_scored`` (``0.0`` when ``n_scored == 0``).
    """
    per_query: list[SmokeQueryResult] = []
    hits = 0
    n_scored = 0

    for query in queries:
        expected_ids = list(query.expected_supporting_chunk_ids)
        if not expected_ids:
            log.debug("book_rag.retrieval_smoke.skip_unscorable", query_id=query.query_id)
            continue

        n_scored += 1
        retrieved = await retrieve_fn(query.query_text, top_k)
        rank = _first_expected_rank(retrieved, set(expected_ids))
        hit = rank is not None
        if hit:
            hits += 1

        per_query.append(
            SmokeQueryResult(
                query_id=query.query_id,
                query_text=query.query_text,
                expected_chunk_ids=expected_ids,
                retrieved_chunk_ids=list(retrieved),
                hit_rank=rank,
                hit=hit,
            )
        )

    score = hits / n_scored if n_scored else 0.0

    report = SmokeReport(
        source_id=source_id,
        n_requested=len(queries),
        n_scored=n_scored,
        top_k=top_k,
        score=score,
        per_query=per_query,
    )
    log.info(
        "book_rag.retrieval_smoke.score",
        source_id=source_id,
        n_requested=report.n_requested,
        n_scored=report.n_scored,
        top_k=top_k,
        score=round(score, 4),
        hits=hits,
    )
    return report


# ---------------------------------------------------------------------------
# Gate policy
# ---------------------------------------------------------------------------


def gate_decision(
    *,
    report: SmokeReport,
    soft_threshold: float = 0.6,
    hard_fail_on_zero: bool = True,
) -> GateResult:
    """Map a SmokeReport to a gate decision.

    Policy (deliberately conservative on the hard path):
      * ``n_scored == 0`` → **QUARANTINE**. We could not probe (no scorable queries
        were generated); zero here is "no evidence", not "proven unretrievable", so
        it must never be a hard FAIL.
      * ``score == 0`` with ``n_scored > 0`` and ``hard_fail_on_zero`` → **FAIL**.
        Every self-query failed to retrieve its own chunk → ingestion is broken.
      * ``score < soft_threshold`` → **QUARANTINE**. Surface for review; do not block.
      * otherwise → **PASS**.

    The soft threshold is inclusive (``score >= soft_threshold`` passes).
    """
    score = report.score
    n_scored = report.n_scored

    if n_scored == 0:
        decision = GateDecision.QUARANTINE
        reason = "no scorable queries generated — cannot probe retrieval"
    elif hard_fail_on_zero and score == 0.0:
        decision = GateDecision.FAIL
        reason = (
            f"self_retrieval_recall@{report.top_k} == 0 over {n_scored} queries: "
            "no self-query retrieved its own chunk — ingestion broken"
        )
    elif score < soft_threshold:
        decision = GateDecision.QUARANTINE
        reason = (
            f"self_retrieval_recall@{report.top_k} {score:.3f} "
            f"< soft_threshold {soft_threshold:.3f}"
        )
    else:
        decision = GateDecision.PASS
        reason = (
            f"self_retrieval_recall@{report.top_k} {score:.3f} "
            f">= soft_threshold {soft_threshold:.3f}"
        )

    log.info(
        "book_rag.retrieval_smoke.gate",
        source_id=report.source_id,
        decision=decision.value,
        score=round(score, 4),
        n_scored=n_scored,
        reason=reason,
    )
    return GateResult(decision=decision, reason=reason, score=score, n_scored=n_scored)


# ---------------------------------------------------------------------------
# Retrieval-hit → chunk_id (namespace match)
# ---------------------------------------------------------------------------


def chunk_id_from_result(point_id: str, payload: dict[str, Any] | None) -> str:
    """Map a retrieval hit to the chunk_id namespace of ``expected_supporting_chunk_ids``.

    Mirrors ``steps._retrieval_chunk_from_payload`` exactly — the stored ``chunk_id``
    if present, else the Qdrant point id. The probe's expected ids are read off those
    same chunks, so any divergence here would turn every real hit into a false miss
    and silently tank the score. Keep this the single source of that rule.
    """
    payload = payload or {}
    return str(payload.get("chunk_id") or payload.get("parent_chunk_id") or point_id)


# ---------------------------------------------------------------------------
# Idempotency key
# ---------------------------------------------------------------------------


def smoke_record_key(source_id: str, prompt_hash: str) -> str:
    """Deterministic id for a smoke run over ``(source, prompt)``.

    Persistence skips re-running the gate when a record already exists for
    this key, so a re-queued book is not re-probed unless the book's chunks were
    reprocessed (delete-by-source then re-ingest) or the generator prompt changed.
    """
    payload = f"{source_id}\x00{prompt_hash}".encode()
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "GateDecision",
    "GateResult",
    "LLMCall",
    "RetrieveFn",
    "SmokeQueryResult",
    "SmokeReport",
    "chunk_id_from_result",
    "gate_decision",
    "generate_self_queries",
    "score_self_retrieval",
    "smoke_record_key",
]

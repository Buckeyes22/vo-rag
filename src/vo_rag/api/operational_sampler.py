"""Low-cardinality background metrics for readiness and useful work."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime
from typing import Any

from vo_rag.api.singletons import SurfaceServices
from vo_rag.core.observability.metrics import (
    dependency_up,
    dispatcher_heartbeat_age_seconds,
    intake_envelopes,
    last_completion_age_seconds,
    oldest_claimable_age_seconds,
    operational_sampler_errors_total,
    sample_ingest_health,
    source_publication_dependency_mismatches,
    source_publication_jobs,
    source_publication_pending_age_seconds,
    source_publication_pointer_mismatches,
    surface_readiness,
)
from vo_rag.pipelines.book_rag.dispatcher_health import dispatcher_health

log = logging.getLogger("vo_rag.api.operational_sampler")

DEFAULT_SAMPLE_INTERVAL_S = 20.0
_QUEUE_SQL = """
SELECT
    COUNT(*) FILTER (WHERE status = 'received') AS received_count,
    COUNT(*) FILTER (WHERE status = 'processing') AS processing_count,
    EXTRACT(EPOCH FROM (NOW() - MIN(created_at) FILTER (
        WHERE source_type = 'file_drop'
          AND (target_pipeline IS NULL OR target_pipeline = 'book_rag')
          AND raw_content #>> '{object_verification,status}' = 'verified'
          AND (
              status = 'received'
              OR (status = 'processing' AND updated_at < NOW() - INTERVAL '6 hours')
          )
    ))) AS oldest_claimable_age_seconds,
    EXTRACT(EPOCH FROM (NOW() - MAX(updated_at) FILTER (
        WHERE status = 'completed'
    ))) AS last_completion_age_seconds
FROM intake_envelopes
"""


def configured_sample_interval_s() -> float:
    """Return a bounded 15–30 second interval; malformed input is ignored."""
    raw = os.environ.get("VORAG_OPERATIONAL_SAMPLE_INTERVAL_S", "").strip()
    try:
        value = float(raw) if raw else DEFAULT_SAMPLE_INTERVAL_S
    except ValueError:
        return DEFAULT_SAMPLE_INTERVAL_S
    return min(30.0, max(15.0, value))


def _number(row: Any, key: str, default: float = -1.0) -> float:
    value = row.get(key) if hasattr(row, "get") else row[key]
    return default if value is None else float(value)


async def sample_operational_metrics(
    services: SurfaceServices,
    settings: Any,
    *,
    now: datetime | None = None,
) -> dict[str, bool]:
    """Perform one active dependency/queue sample and publish its gauges."""
    # Local import keeps the route module independent of lifespan wiring while
    # sharing the exact same active dependency checks as HTTP readiness.
    from vo_rag.api.routes.system import (
        _db_pool_health,
        _embedding_health,
        _qdrant_health,
        _reranker_health,
    )

    db, qdrant, embedding, reranker = await asyncio.gather(
        _db_pool_health(services),
        _qdrant_health(services),
        _embedding_health(settings),
        _reranker_health(services, settings),
    )
    dispatcher = dispatcher_health(now=now or datetime.now(UTC))
    checks = {
        "db_pool": bool(db.get("ok")),
        "qdrant": bool(qdrant.get("ok")),
        "embedding": bool(embedding.get("ok")),
        "reranker": bool(reranker.get("ok")),
        "dispatcher": bool(dispatcher.get("ok")),
    }
    for name, ok in checks.items():
        dependency_up.labels(dependency=name).set(float(ok))

    raw_heartbeat_age = dispatcher.get("age_seconds", -1.0)
    heartbeat_age = float(raw_heartbeat_age) if isinstance(raw_heartbeat_age, int | float) else -1.0
    dispatcher_heartbeat_age_seconds.set(heartbeat_age)

    pool = services.db_pool
    if pool is None:
        intake_envelopes.labels(status="received").set(-1)
        intake_envelopes.labels(status="processing").set(-1)
        oldest_claimable_age_seconds.set(-1)
        last_completion_age_seconds.set(-1)
    else:
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(_QUEUE_SQL)
            if row is None:
                raise RuntimeError("queue aggregate returned no row")
        except Exception:  # noqa: BLE001
            log.exception("operational_queue_sample_failed")
            operational_sampler_errors_total.labels(component="postgres_queue").inc()
            dependency_up.labels(dependency="db_pool").set(0)
            checks["db_pool"] = False
            intake_envelopes.labels(status="received").set(-1)
            intake_envelopes.labels(status="processing").set(-1)
            oldest_claimable_age_seconds.set(-1)
            last_completion_age_seconds.set(-1)
        else:
            intake_envelopes.labels(status="received").set(_number(row, "received_count", 0))
            intake_envelopes.labels(status="processing").set(_number(row, "processing_count", 0))
            oldest_claimable_age_seconds.set(_number(row, "oldest_claimable_age_seconds"))
            last_completion_age_seconds.set(_number(row, "last_completion_age_seconds"))

    # VOR-1434: stuck/overloaded-ingest gauges, derived from Postgres
    # (fails closed to -1 on its own; never affects readiness).
    await sample_ingest_health(pool)
    await sample_source_publications(pool)

    query_ok = all(checks[name] for name in ("db_pool", "qdrant", "embedding", "reranker"))
    intake_ok = all(checks[name] for name in ("db_pool", "dispatcher"))
    surface_readiness.labels(surface="query").set(float(query_ok))
    surface_readiness.labels(surface="intake").set(float(intake_ok))
    return {"query": query_ok, "intake": intake_ok}


_PUBLICATION_STATES = (
    "preparing",
    "staged",
    "verified",
    "published",
    "partial_preserved",
    "preserved_no_write",
    "blocked_missing_evidence",
    "failed_retryable",
    "superseded",
)
_PUBLICATION_JOB_SQL = """
SELECT corpus, state, COUNT(*) AS job_count,
       COALESCE(MAX(EXTRACT(EPOCH FROM NOW() - created_at)), 0) AS oldest_age
FROM source_publication_jobs WHERE corpus = ANY($1::text[])
GROUP BY corpus, state
"""
_PUBLICATION_POINTER_SQL = """
SELECT p.corpus, COUNT(*) AS mismatches
FROM source_publications p
LEFT JOIN source_publication_jobs j ON j.job_id = p.active_job_id
WHERE p.corpus = ANY($1::text[]) AND p.active_revision IS NOT NULL
  AND (j.job_id IS NULL OR j.state <> 'published'
       OR j.corpus IS DISTINCT FROM p.corpus
       OR j.collection IS DISTINCT FROM p.collection
       OR j.source_id IS DISTINCT FROM p.source_id
       OR j.revision IS DISTINCT FROM p.active_revision)
GROUP BY p.corpus
"""
_PUBLICATION_DEPENDENCY_SQL = """
SELECT p.corpus, COUNT(DISTINCT (p.collection, p.source_id)) AS mismatches
FROM source_publications p
JOIN source_publication_jobs j ON j.job_id = p.active_job_id
CROSS JOIN LATERAL jsonb_each(CASE
  WHEN jsonb_typeof(j.recipe->'expected_dependencies') = 'object'
  THEN j.recipe->'expected_dependencies' ELSE '{}'::jsonb END) dep
LEFT JOIN source_publications raw
  ON raw.corpus = 'books' AND raw.collection = 'vo_rag_books_raw_v1'
 AND raw.source_id = dep.key
WHERE p.corpus = ANY($1::text[]) AND p.active_revision IS NOT NULL
  AND (dep.value #>> '{}') IS DISTINCT FROM raw.active_revision::text
GROUP BY p.corpus
"""


async def sample_source_publications(pool: Any) -> None:
    """Expose durable quarantine and pointer health without per-source labels.

    These signals do not gate unrelated intake or infer semantic repair success.
    Errors export unknown values and never leave stale healthy metrics behind.
    """
    corpora = ("books", "reddit")
    try:
        if pool is None:
            raise RuntimeError("publication sampler has no pool")
        async with pool.acquire() as connection, connection.transaction():
            await connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            rows = await connection.fetch(_PUBLICATION_JOB_SQL, list(corpora))
            pointer_rows = await connection.fetch(_PUBLICATION_POINTER_SQL, list(corpora))
            dependency_rows = await connection.fetch(_PUBLICATION_DEPENDENCY_SQL, list(corpora))
    except Exception:  # noqa: BLE001
        operational_sampler_errors_total.labels(component="source_publication").inc()
        for corpus in corpora:
            for state in _PUBLICATION_STATES:
                source_publication_jobs.labels(corpus=corpus, state=state).set(-1)
            source_publication_pending_age_seconds.labels(corpus=corpus).set(-1)
            source_publication_pointer_mismatches.labels(corpus=corpus).set(-1)
            source_publication_dependency_mismatches.labels(corpus=corpus).set(-1)
        return
    pending = {
        "preparing",
        "staged",
        "verified",
        "partial_preserved",
        "blocked_missing_evidence",
        "failed_retryable",
    }
    for corpus in corpora:
        ages: list[float] = []
        for state in _PUBLICATION_STATES:
            matches = [row for row in rows if row["corpus"] == corpus and row["state"] == state]
            source_publication_jobs.labels(corpus=corpus, state=state).set(
                sum(int(row["job_count"]) for row in matches)
            )
            if state in pending:
                ages.extend(float(row["oldest_age"]) for row in matches)
        source_publication_pending_age_seconds.labels(corpus=corpus).set(max(ages, default=0))
        source_publication_pointer_mismatches.labels(corpus=corpus).set(
            sum(int(row["mismatches"]) for row in pointer_rows if row["corpus"] == corpus)
        )
        source_publication_dependency_mismatches.labels(corpus=corpus).set(
            sum(int(row["mismatches"]) for row in dependency_rows if row["corpus"] == corpus)
        )


async def run_operational_sampler(
    services: SurfaceServices,
    settings: Any,
    stop_event: asyncio.Event,
    *,
    interval_s: float | None = None,
) -> None:
    """Sample immediately and then at a bounded cadence until shutdown."""
    interval = configured_sample_interval_s() if interval_s is None else interval_s
    while not stop_event.is_set():
        try:
            await sample_operational_metrics(services, settings)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("operational_sample_failed")
            operational_sampler_errors_total.labels(component="sampler").inc()
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except TimeoutError:
            continue


__all__ = [
    "DEFAULT_SAMPLE_INTERVAL_S",
    "configured_sample_interval_s",
    "run_operational_sampler",
    "sample_operational_metrics",
]

"""PostgreSQL state transitions for source ingest generations."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from vo_rag.core.models.ingest_generation import (
    IngestGenerationState,
    SourceIngestGeneration,
)


def _to_generation(row: Any) -> SourceIngestGeneration:
    values = dict(row)
    if isinstance(values.get("validation_decision"), str):
        values["validation_decision"] = json.loads(values["validation_decision"])
    return SourceIngestGeneration.model_validate(values)


async def create_generation(
    pool: Any,
    generation: SourceIngestGeneration,
) -> bool:
    result = await pool.execute(
        """
        INSERT INTO source_ingest_generations (
            generation_id, source_id, raw_collection, envelope_id, run_id, pipeline,
            schema_version, chunking_version, sterilize_enabled, state,
            owner_token, claim_expires_at,
            expected_raw_points, actual_raw_points,
            expected_sterile_points, actual_sterile_points,
            expected_artifacts, actual_artifacts,
            validation_decision, smoke_decision, failure_reason
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
            $13, $14, $15, $16, $17, $18, $19, $20, $21
        )
        ON CONFLICT (generation_id) DO UPDATE
        SET owner_token = EXCLUDED.owner_token,
            claim_expires_at = EXCLUDED.claim_expires_at,
            updated_at = NOW()
        WHERE source_ingest_generations.state IN ('staging', 'validated')
          AND (
              source_ingest_generations.owner_token = EXCLUDED.owner_token
              OR source_ingest_generations.claim_expires_at IS NULL
              OR source_ingest_generations.claim_expires_at <= NOW()
          )
        """,
        generation.generation_id,
        generation.source_id,
        generation.raw_collection,
        generation.envelope_id,
        generation.run_id,
        generation.pipeline,
        generation.schema_version,
        generation.chunking_version,
        generation.sterilize_enabled,
        generation.state,
        generation.owner_token,
        generation.claim_expires_at,
        generation.expected_raw_points,
        generation.actual_raw_points,
        generation.expected_sterile_points,
        generation.actual_sterile_points,
        generation.expected_artifacts,
        generation.actual_artifacts,
        generation.validation_decision,
        generation.smoke_decision,
        generation.failure_reason,
    )
    return str(result).endswith(" 1")


async def release_generation_claim(
    pool: Any,
    generation_id: str,
    owner_token: str,
) -> bool:
    """Release only the caller's non-terminal generation lease."""
    result = await pool.execute(
        """
        UPDATE source_ingest_generations
        SET owner_token = NULL, claim_expires_at = NOW()
        WHERE generation_id = $1 AND owner_token = $2
          AND state IN ('staging', 'validated')
        """,
        generation_id,
        owner_token,
    )
    return str(result).endswith(" 1")


async def refresh_generation_claim(
    pool: Any,
    generation_id: str,
    owner_token: str,
    claim_expires_at: datetime,
) -> bool:
    """Extend only the current owner's live staging/validated lease."""
    result = await pool.execute(
        """
        UPDATE source_ingest_generations
        SET claim_expires_at = $3
        WHERE generation_id = $1 AND owner_token = $2
          AND state IN ('staging', 'validated')
          AND claim_expires_at > NOW()
        """,
        generation_id,
        owner_token,
        claim_expires_at,
    )
    return str(result).endswith(" 1")


async def generation_claim_is_current(
    pool: Any,
    generation_id: str,
    owner_token: str,
) -> bool:
    """Return whether the caller still owns an unexpired mutable generation."""
    row = await pool.fetchrow(
        """
        SELECT EXISTS (
            SELECT 1
            FROM source_ingest_generations
            WHERE generation_id = $1
              AND owner_token = $2
              AND claim_expires_at > NOW()
              AND state IN ('staging', 'validated')
        ) AS current
        """,
        generation_id,
        owner_token,
    )
    return bool(row["current"]) if row is not None else False


async def get_generation(pool: Any, generation_id: str) -> SourceIngestGeneration | None:
    row = await pool.fetchrow(
        "SELECT * FROM source_ingest_generations WHERE generation_id = $1",
        generation_id,
    )
    return _to_generation(row) if row is not None else None


async def get_active_generation(
    pool: Any,
    source_id: str,
    raw_collection: str,
) -> SourceIngestGeneration | None:
    row = await pool.fetchrow(
        """
        SELECT * FROM source_ingest_generations
        WHERE source_id = $1 AND raw_collection = $2 AND state = 'active'
        ORDER BY activated_at DESC NULLS LAST, created_at DESC
        LIMIT 1
        """,
        source_id,
        raw_collection,
    )
    return _to_generation(row) if row is not None else None


async def get_active_generation_id(
    pool: Any,
    source_id: str,
    raw_collection: str,
) -> str | None:
    """Read the durable active id using a fresh pool operation."""
    row = await pool.fetchrow(
        """
        SELECT generation_id FROM source_ingest_generations
        WHERE source_id = $1 AND raw_collection = $2 AND state = 'active'
        LIMIT 1
        """,
        source_id,
        raw_collection,
    )
    return str(row["generation_id"]) if row is not None else None


async def get_latest_generation_for_envelope(
    pool: Any,
    envelope_id: str,
) -> SourceIngestGeneration | None:
    row = await pool.fetchrow(
        """
        SELECT * FROM source_ingest_generations
        WHERE envelope_id = $1
        ORDER BY created_at DESC
        LIMIT 1
        """,
        envelope_id,
    )
    return _to_generation(row) if row is not None else None


async def count_active_generation_artifacts(
    pool: Any,
    source_id: str,
    raw_collection: str,
    generation_id: str,
) -> int:
    """Count the active artifacts owned by one collection-scoped generation."""
    row = await pool.fetchrow(
        """
        SELECT COUNT(*)::integer AS count
        FROM knowledge_artifacts
        WHERE content->>'source_id' = $1
          AND COALESCE(content->>'raw_collection', 'vo_rag_books_raw_v1') = $2
          AND content->>'generation_id' = $3
          AND status = 'active'
        """,
        source_id,
        raw_collection,
        generation_id,
    )
    return int(row["count"]) if row is not None else 0


async def get_superseded_generation_ids(
    pool: Any,
    source_id: str,
    raw_collection: str,
) -> list[str]:
    rows = await pool.fetch(
        """
        SELECT generation_id FROM source_ingest_generations
        WHERE source_id = $1 AND raw_collection = $2 AND state = 'superseded'
        ORDER BY completed_at ASC NULLS FIRST, created_at ASC
        """,
        source_id,
        raw_collection,
    )
    return [str(row["generation_id"]) for row in rows]


async def update_generation_counts(
    pool: Any,
    generation_id: str,
    *,
    expected_raw_points: int | None = None,
    actual_raw_points: int | None = None,
    expected_sterile_points: int | None = None,
    actual_sterile_points: int | None = None,
    actual_artifacts: int | None = None,
    owner_token: str | None = None,
) -> bool:
    """Update counts, optionally fenced to the current unexpired owner."""
    result = await pool.execute(
        """
        UPDATE source_ingest_generations
        SET expected_raw_points = COALESCE($2, expected_raw_points),
            actual_raw_points = COALESCE($3, actual_raw_points),
            expected_sterile_points = COALESCE($4, expected_sterile_points),
            actual_sterile_points = COALESCE($5, actual_sterile_points),
            actual_artifacts = COALESCE($6, actual_artifacts)
        WHERE generation_id = $1
          AND (
              $7::text IS NULL
              OR (
                  owner_token = $7
                  AND claim_expires_at > NOW()
                  AND state IN ('staging', 'validated')
              )
          )
        """,
        generation_id,
        expected_raw_points,
        actual_raw_points,
        expected_sterile_points,
        actual_sterile_points,
        actual_artifacts,
        owner_token,
    )
    return str(result).endswith(" 1")


async def record_generation_cleanup(
    pool: Any,
    generation_id: str,
    *,
    required: bool,
    error: str | None = None,
) -> None:
    """Persist whether superseded Qdrant points still need deletion."""
    await pool.execute(
        """
        UPDATE source_ingest_generations
        SET cleanup_required = $2,
            cleanup_error = CASE WHEN $2 THEN $3 ELSE NULL END
        WHERE generation_id = $1 AND state IN ('active', 'superseded')
        """,
        generation_id,
        required,
        error,
    )


async def transition_generation(
    pool: Any,
    generation_id: str,
    *,
    from_states: Iterable[IngestGenerationState | str],
    to_state: IngestGenerationState | str,
    validation_decision: dict[str, object] | None = None,
    smoke_decision: str | None = None,
    failure_reason: str | None = None,
    owner_token: str | None = None,
) -> bool:
    allowed = [str(state) for state in from_states]
    result = await pool.execute(
        """
        UPDATE source_ingest_generations
        SET state = $2,
            validation_decision = COALESCE($3, validation_decision),
            smoke_decision = COALESCE($4, smoke_decision),
            failure_reason = COALESCE($5, failure_reason),
            completed_at = CASE WHEN $2 IN ('active','quarantined','failed')
                                THEN NOW() ELSE completed_at END,
            owner_token = CASE WHEN $2 IN ('active','quarantined','failed','superseded')
                               THEN NULL ELSE owner_token END,
            claim_expires_at = CASE
                WHEN $2 IN ('active','quarantined','failed','superseded')
                THEN NULL ELSE claim_expires_at END
        WHERE generation_id = $1 AND state = ANY($6::text[])
          AND (
              $7::text IS NULL
              OR (owner_token = $7 AND claim_expires_at > NOW())
          )
        """,
        generation_id,
        str(to_state),
        validation_decision,
        smoke_decision,
        failure_reason,
        allowed,
        owner_token,
    )
    return str(result).endswith(" 1")


async def activate_generation_record(
    pool: Any,
    generation_id: str,
    source_id: str,
    raw_collection: str,
    owner_token: str | None = None,
) -> None:
    """Serialize the Postgres half of activation and enforce one active row."""
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended($1 || chr(31) || $2, 0))",
            source_id,
            raw_collection,
        )
        await activate_generation_record_locked(
            conn,
            generation_id,
            source_id,
            raw_collection,
            owner_token,
        )


async def activate_generation_record_locked(
    conn: Any,
    generation_id: str,
    source_id: str,
    raw_collection: str,
    owner_token: str | None = None,
) -> None:
    """Persist activation while the caller holds the source advisory lock."""
    row = await conn.fetchrow(
        """
        SELECT state FROM source_ingest_generations
        WHERE generation_id = $1 AND source_id = $2 AND raw_collection = $3
          AND (
              $4::text IS NULL
              OR (owner_token = $4 AND claim_expires_at > NOW())
          )
        FOR UPDATE
        """,
        generation_id,
        source_id,
        raw_collection,
        owner_token,
    )
    if row is None or str(row["state"]) != IngestGenerationState.VALIDATED:
        raise RuntimeError(f"generation {generation_id} is not validated")
    await conn.execute(
        """
        UPDATE source_ingest_generations
        SET state = 'superseded', completed_at = NOW()
        WHERE source_id = $1 AND raw_collection = $3
          AND state = 'active' AND generation_id <> $2
        """,
        source_id,
        generation_id,
        raw_collection,
    )
    await conn.execute(
        """
        UPDATE knowledge_artifacts
        SET status = 'superseded'
        WHERE content->>'source_id' = $1
          AND COALESCE(content->>'raw_collection', 'vo_rag_books_raw_v1') = $3
          AND status = 'active'
          AND content->>'generation_id' IS DISTINCT FROM $2
        """,
        source_id,
        generation_id,
        raw_collection,
    )
    await conn.execute(
        """
        UPDATE knowledge_artifacts
        SET status = 'active'
        WHERE content->>'source_id' = $1
          AND content->>'generation_id' = $2
          AND COALESCE(content->>'raw_collection', 'vo_rag_books_raw_v1') = $3
        """,
        source_id,
        generation_id,
        raw_collection,
    )
    result = await conn.execute(
        """
        UPDATE source_ingest_generations
        SET state = 'active', activated_at = NOW(), completed_at = NOW(),
            cleanup_required = TRUE,
            cleanup_error = 'post-activation cleanup pending',
            owner_token = NULL,
            claim_expires_at = NULL
        WHERE generation_id = $1 AND source_id = $2 AND raw_collection = $3
          AND state = 'validated'
          AND ($4::text IS NULL OR owner_token = $4)
        """,
        generation_id,
        source_id,
        raw_collection,
        owner_token,
    )
    if not str(result).endswith(" 1"):
        raise RuntimeError(f"generation {generation_id} activation lost its validated state")


__all__ = [
    "activate_generation_record",
    "activate_generation_record_locked",
    "count_active_generation_artifacts",
    "create_generation",
    "generation_claim_is_current",
    "get_active_generation",
    "get_active_generation_id",
    "get_generation",
    "get_latest_generation_for_envelope",
    "get_superseded_generation_ids",
    "record_generation_cleanup",
    "refresh_generation_claim",
    "release_generation_claim",
    "transition_generation",
    "update_generation_counts",
]

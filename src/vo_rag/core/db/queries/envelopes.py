"""Typed query helpers for intake_envelopes."""

from __future__ import annotations

import asyncpg  # type: ignore[import-untyped]

from vo_rag.core.db.cursor import decode_cursor, encode_cursor
from vo_rag.core.models.envelope import (
    EnvelopeStatus,
    IntakeEnvelope,
    IntakeSource,
    Priority,
    TargetPipeline,
    TriageResult,
)


def _record_to_envelope(record: asyncpg.Record) -> IntakeEnvelope:
    return IntakeEnvelope.model_validate({**record})


async def insert_envelope(pool: asyncpg.Pool, envelope: IntakeEnvelope) -> None:
    """Insert intake envelope row."""
    await pool.execute(
        """
        INSERT INTO intake_envelopes (
            id, source_type, source_data, raw_content, metadata,
            triage_result, status, target_pipeline, priority, error,
            sovereignty, created_at, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, $8, $9, $10,
            $11, $12, $13
        )
        """,
        envelope.id,
        envelope.source_type,
        envelope.source_data,
        envelope.raw_content,
        envelope.metadata,
        envelope.triage_result.model_dump(mode="json")
        if envelope.triage_result is not None
        else None,
        envelope.status,
        envelope.target_pipeline,
        envelope.priority,
        envelope.error,
        envelope.sovereignty,
        envelope.created_at,
        envelope.updated_at,
    )


async def get_envelope(pool: asyncpg.Pool, envelope_id: str) -> IntakeEnvelope | None:
    """Get envelope by ID."""
    row = await pool.fetchrow("SELECT * FROM intake_envelopes WHERE id = $1", envelope_id)
    if row is None:
        return None
    return _record_to_envelope(row)


async def update_envelope_status(
    pool: asyncpg.Pool,
    envelope_id: str,
    status: EnvelopeStatus | str,
) -> None:
    """Update envelope workflow status."""
    await pool.execute(
        "UPDATE intake_envelopes SET status = $2 WHERE id = $1",
        envelope_id,
        status,
    )


async def fail_envelope(
    pool: asyncpg.Pool,
    envelope_id: str,
    error: str,
) -> None:
    """Mark an envelope failed with a clear operator-facing error."""
    await pool.execute(
        "UPDATE intake_envelopes SET status = $2, error = $3 WHERE id = $1",
        envelope_id,
        EnvelopeStatus.FAILED,
        error,
    )


async def update_envelope_triage(
    pool: asyncpg.Pool,
    envelope_id: str,
    *,
    triage_result: TriageResult | None,
    target_pipeline: TargetPipeline | str | None,
    priority: Priority | str | None,
) -> None:
    """Persist triage result and route decisions."""
    await pool.execute(
        """
        UPDATE intake_envelopes
        SET triage_result = $2,
            target_pipeline = $3,
            priority = COALESCE($4, priority)
        WHERE id = $1
        """,
        envelope_id,
        triage_result.model_dump(mode="json") if triage_result is not None else None,
        target_pipeline,
        priority,
    )


async def list_envelopes(
    pool: asyncpg.Pool,
    *,
    cursor: str | None = None,
    limit: int = 50,
    status: EnvelopeStatus | str | None = None,
    source_type: IntakeSource | str | None = None,
) -> tuple[list[IntakeEnvelope], str | None]:
    """List envelopes via keyset pagination on (created_at DESC, id DESC)."""
    where_clauses = ["TRUE"]
    args: list[object] = []

    if status is not None:
        args.append(status)
        where_clauses.append(f"status = ${len(args)}")
    if source_type is not None:
        args.append(source_type)
        where_clauses.append(f"source_type = ${len(args)}")
    if cursor is not None:
        cursor_created_at, cursor_id = decode_cursor(cursor)
        args.extend([cursor_created_at, cursor_id])
        where_clauses.append(f"(created_at, id) < (${len(args) - 1}, ${len(args)})")

    args.append(limit + 1)

    query = f"""
        SELECT *
        FROM intake_envelopes
        WHERE {" AND ".join(where_clauses)}
        ORDER BY created_at DESC, id DESC
        LIMIT ${len(args)}
    """
    rows = await pool.fetch(query, *args)

    has_next = len(rows) > limit
    page_rows = rows[:limit] if has_next else rows
    items = [_record_to_envelope(row) for row in page_rows]

    next_cursor: str | None = None
    if has_next and items:
        last_item = items[-1]
        next_cursor = encode_cursor(last_item.created_at, last_item.id)
    return items, next_cursor

"""Reverse lookup from a subject hash to envelope IDs and Qdrant point IDs.

These helpers are not called from anywhere in this tree, and purge is by
source (``vo-rag corpus purge``), so there is no purge-by-data-subject
workflow built on them yet.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg  # type: ignore[import-untyped]


async def register_subject(
    pool: asyncpg.Pool,
    *,
    subject_hash: str,
    envelope_id: str | None = None,
    qdrant_points: dict[str, list[str]] | None = None,
) -> None:
    """Idempotent upsert: creates the row if absent, then appends
    envelope_id / qdrant points to the existing record without dupes."""
    if not subject_hash:
        raise ValueError("subject_hash must be non-empty")

    envelope_array = [envelope_id] if envelope_id else []
    points_json = json.dumps(qdrant_points or {})

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO pii_subjects(subject_hash, envelope_ids, qdrant_point_ids)
            VALUES ($1, $2, $3::jsonb)
            ON CONFLICT (subject_hash) DO UPDATE SET
                envelope_ids = ARRAY(
                    SELECT DISTINCT unnest(
                        pii_subjects.envelope_ids || EXCLUDED.envelope_ids
                    )
                ),
                qdrant_point_ids = pii_subjects.qdrant_point_ids || EXCLUDED.qdrant_point_ids
            """,
            subject_hash,
            envelope_array,
            points_json,
        )


async def lookup_subject(pool: asyncpg.Pool, *, subject_hash: str) -> dict[str, Any] | None:
    if not subject_hash:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT subject_hash, first_seen_at, envelope_ids, qdrant_point_ids
            FROM pii_subjects
            WHERE subject_hash = $1
            """,
            subject_hash,
        )
    return dict(row) if row else None


async def delete_subject(pool: asyncpg.Pool, *, subject_hash: str) -> bool:
    async with pool.acquire() as conn:
        row = await conn.fetchval(
            "DELETE FROM pii_subjects WHERE subject_hash = $1 RETURNING subject_hash",
            subject_hash,
        )
    return row is not None


__all__ = ["delete_subject", "lookup_subject", "register_subject"]

"""asyncpg writer + reader for agent_termination_log.

The record is a ``dict[str, Any]`` rather than a typed model, and reason
validation accepts any non-empty string.
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from typing import Any

import asyncpg  # type: ignore[import-untyped]

from vo_rag.core.models import ulid_new


async def record_termination(
    pool: asyncpg.Pool,
    *,
    envelope_id: str,
    pipeline: str,
    record: dict[str, Any],
) -> str:
    """Persist one termination event. Returns the generated row id."""
    reason = record["reason"]
    if not reason:
        raise ValueError("invalid termination reason: empty")
    if not pipeline:
        raise ValueError("pipeline must be non-empty")
    if not envelope_id:
        raise ValueError("envelope_id must be non-empty")

    row_id = ulid_new()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO agent_termination_log(
                id, envelope_id, pipeline, reason,
                step_count, cost_usd, wall_clock_s, sub_questions,
                detail, extra
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
            """,
            row_id,
            envelope_id,
            pipeline,
            reason,
            record["step_count"],
            Decimal(str(record["cost_usd"])),
            record["wall_clock_s"],
            record["sub_questions"],
            record["detail"],
            json.dumps(record.get("extra") or {}),
        )
    return row_id


async def list_terminations(
    pool: asyncpg.Pool,
    *,
    pipeline: str,
    since: datetime,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return recent termination rows for a pipeline since `since`."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, envelope_id, pipeline, reason, step_count, cost_usd,
                   wall_clock_s, sub_questions, detail, extra, occurred_at
            FROM agent_termination_log
            WHERE pipeline = $1 AND occurred_at >= $2
            ORDER BY occurred_at DESC
            LIMIT $3
            """,
            pipeline,
            since,
            limit,
        )
    return [dict(row) for row in rows]


__all__ = ["list_terminations", "record_termination"]

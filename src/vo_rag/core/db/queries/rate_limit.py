from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import asyncpg  # type: ignore[import-untyped]


async def count_hits(
    pool: asyncpg.Pool,
    *,
    source_id: str,
    caller: str,
    window: str,
) -> int:
    """Count hits for one source/caller in the given interval window."""
    count = await pool.fetchval(
        """
        SELECT count(*)
        FROM query_source_hit
        WHERE source_id = $1
          AND caller = $2
          AND hit_at > now() - ($3::interval)
        """,
        source_id,
        caller,
        window,
    )
    return int(count or 0)


async def insert_hits(pool: asyncpg.Pool, *, source_ids: list[str], caller: str) -> None:
    """Insert one query_source_hit row per source id."""
    hit_at = datetime.now(UTC)
    rows = [(str(uuid4()), source_id, caller, hit_at) for source_id in source_ids]
    await pool.executemany(
        """
        INSERT INTO query_source_hit (id, source_id, caller, hit_at)
        VALUES ($1, $2, $3, $4)
        """,
        rows,
    )


async def prune_hits_older_than_days(pool: asyncpg.Pool, *, days: int) -> int:
    """Delete rows older than N days and return deleted count."""
    deleted = await pool.fetchval(
        """
        WITH deleted AS (
            DELETE FROM query_source_hit
            WHERE hit_at < now() - (($1::text || ' days')::interval)
            RETURNING 1
        )
        SELECT count(*) FROM deleted
        """,
        days,
    )
    return int(deleted or 0)

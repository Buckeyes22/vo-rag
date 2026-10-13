from __future__ import annotations

from datetime import datetime

import asyncpg  # type: ignore[import-untyped]


async def insert_log(
    pool: asyncpg.Pool,
    *,
    id: str,
    backup_type: str,
    target: str,
    status: str,
    size_bytes: int | None,
    garage_key: str | None,
    error_message: str | None,
    started_at: datetime,
    completed_at: datetime | None,
) -> None:
    await pool.execute(
        """
        INSERT INTO backup_log (
            id,
            backup_type,
            target,
            status,
            size_bytes,
            garage_key,
            error_message,
            started_at,
            completed_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
        id,
        backup_type,
        target,
        status,
        size_bytes,
        garage_key,
        error_message,
        started_at,
        completed_at,
    )


async def list_recent(
    pool: asyncpg.Pool,
    *,
    backup_type: str,
    limit: int = 50,
) -> list[asyncpg.Record]:
    rows = await pool.fetch(
        """
        SELECT *
        FROM backup_log
        WHERE backup_type = $1
        ORDER BY started_at DESC
        LIMIT $2
        """,
        backup_type,
        limit,
    )
    return list(rows)

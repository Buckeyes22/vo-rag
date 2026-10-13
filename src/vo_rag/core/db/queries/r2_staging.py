from __future__ import annotations

import asyncpg  # type: ignore[import-untyped]


async def record_cleanup_run(pool: asyncpg.Pool, *, removed: int) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO r2_staging_cleanup_run (removed) VALUES ($1)",
            removed,
        )


async def record_staged(
    pool: asyncpg.Pool,
    *,
    r2_key: str,
    direction: str,
    envelope_id: str | None,
    nbytes: int | None,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO r2_staging_log (r2_key, direction, envelope_id, bytes)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (r2_key) DO NOTHING
            """,
            r2_key,
            direction,
            envelope_id,
            nbytes,
        )

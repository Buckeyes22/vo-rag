from __future__ import annotations

from decimal import Decimal

import asyncpg  # type: ignore[import-untyped]


async def insert_cost(
    pool: asyncpg.Pool,
    *,
    id: str,
    envelope_id: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: Decimal,
) -> None:
    """Insert a Claude triage fallback cost record."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO triage_cost(id, envelope_id, model, input_tokens, output_tokens, cost_usd)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            id,
            envelope_id,
            model,
            input_tokens,
            output_tokens,
            cost_usd,
        )


async def cost_last_n_days(pool: asyncpg.Pool, *, days: int) -> Decimal:
    """Return aggregate triage fallback USD cost over the last N days."""
    value = await pool.fetchval(
        """
        SELECT COALESCE(SUM(cost_usd), 0)
        FROM triage_cost
        WHERE created_at >= NOW() - ($1::INT * INTERVAL '1 day')
        """,
        days,
    )
    return Decimal(value)

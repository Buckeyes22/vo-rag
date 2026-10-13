"""Postgres helpers for training_cost table (migration 012)."""

from __future__ import annotations

from decimal import Decimal

import asyncpg  # type: ignore[import-untyped]


async def record_cost(
    pool: asyncpg.Pool,
    *,
    run_id: str,
    persona: str,
    stage: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO training_cost
                (run_id, persona, stage, model, input_tokens, output_tokens, cost_usd)
            VALUES ($1,$2,$3,$4,$5,$6,$7)
            """,
            run_id,
            persona,
            stage,
            model,
            input_tokens,
            output_tokens,
            cost_usd,
        )


async def sum_run_cost(pool: asyncpg.Pool, *, run_id: str) -> float:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT COALESCE(SUM(cost_usd),0) AS total FROM training_cost WHERE run_id=$1",
            run_id,
        )

    total = row["total"]
    return float(total if not isinstance(total, Decimal) else total)

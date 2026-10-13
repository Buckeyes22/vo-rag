"""vlm_cost query helper.

The pool is always injected (callers own the DSN), so this helper takes
``pool`` as an explicit argument.
"""

from __future__ import annotations

import uuid

import asyncpg  # type: ignore[import-untyped]


async def log_vlm_cost(
    pool: asyncpg.Pool,
    *,
    image_id: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO vlm_cost (id, image_id, model, input_tokens, output_tokens, cost_usd)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            str(uuid.uuid4()),
            image_id,
            model,
            input_tokens,
            output_tokens,
            cost_usd,
        )

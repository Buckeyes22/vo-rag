"""lora_adapters query helper.

``PersonaId`` is typed here as ``str``; there is no persona enum in vo-rag.
"""

from __future__ import annotations

from typing import Any

import asyncpg  # type: ignore[import-untyped]

PersonaId = str


async def insert_adapter(pool: asyncpg.Pool, meta: dict[str, Any]) -> None:
    await pool.execute(
        """
        INSERT INTO lora_adapters (
            id,
            persona,
            adapter_path,
            base_model,
            training_date,
            eval_report,
            memorization_report,
            active
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (id)
        DO UPDATE SET
            persona = EXCLUDED.persona,
            adapter_path = EXCLUDED.adapter_path,
            base_model = EXCLUDED.base_model,
            training_date = EXCLUDED.training_date,
            eval_report = EXCLUDED.eval_report,
            memorization_report = EXCLUDED.memorization_report,
            active = EXCLUDED.active
        """,
        meta["id"],
        meta["persona"],
        meta["adapter_path"],
        meta.get("base_model", "Qwen/Qwen3-8B"),
        meta["training_date"],
        meta.get("eval_report"),
        meta.get("memorization_report"),
        meta.get("active", False),
    )


async def list_adapters_q(pool: asyncpg.Pool) -> list[dict[str, Any]]:
    rows = await pool.fetch(
        """
        SELECT id, persona, adapter_path, base_model, training_date,
               eval_report, memorization_report, active
        FROM lora_adapters
        ORDER BY created_at ASC, adapter_path ASC
        """
    )
    return [dict(row) for row in rows]


async def get_active_q(pool: asyncpg.Pool, *, persona: PersonaId) -> dict[str, Any] | None:
    row = await pool.fetchrow(
        """
        SELECT id, persona, adapter_path, base_model, training_date,
               eval_report, memorization_report, active
        FROM lora_adapters
        WHERE persona = $1 AND active = true
        """,
        persona,
    )
    return dict(row) if row is not None else None


async def clear_active_for_persona(conn: asyncpg.Connection, *, persona: PersonaId) -> None:
    await conn.execute(
        "UPDATE lora_adapters SET active=false WHERE persona=$1 AND active=true",
        persona,
    )


async def set_active_q(conn: asyncpg.Connection, *, persona: PersonaId, adapter_path: str) -> None:
    await conn.execute(
        "UPDATE lora_adapters SET active=true WHERE persona=$1 AND adapter_path=$2",
        persona,
        adapter_path,
    )

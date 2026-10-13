from __future__ import annotations

import asyncpg  # type: ignore[import-untyped]


async def insert_answer_log(
    pool: asyncpg.Pool,
    *,
    id: str,
    query_log_id: str | None,
    query_text: str,
    mode: str,
    profile: str,
    model: str,
    prompt_name: str | None,
    prompt_version: int | None,
    collections: list[str],
    context_scope: str,
    retrieved_count: int,
    source_ids: list[str],
    latency_ms: float,
    cost_usd: float,
    compliance_outcome: str,
    warnings: list[str],
) -> None:
    await pool.execute(
        """
        INSERT INTO answer_log (
            id,
            query_log_id,
            query_text,
            mode,
            profile,
            model,
            prompt_name,
            prompt_version,
            collections,
            context_scope,
            retrieved_count,
            source_ids,
            latency_ms,
            cost_usd,
            compliance_outcome,
            warnings
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
        ON CONFLICT (id) DO UPDATE SET
            query_log_id = EXCLUDED.query_log_id,
            query_text = EXCLUDED.query_text,
            mode = EXCLUDED.mode,
            profile = EXCLUDED.profile,
            model = EXCLUDED.model,
            prompt_name = EXCLUDED.prompt_name,
            prompt_version = EXCLUDED.prompt_version,
            collections = EXCLUDED.collections,
            context_scope = EXCLUDED.context_scope,
            retrieved_count = EXCLUDED.retrieved_count,
            source_ids = EXCLUDED.source_ids,
            latency_ms = EXCLUDED.latency_ms,
            cost_usd = EXCLUDED.cost_usd,
            compliance_outcome = EXCLUDED.compliance_outcome,
            warnings = EXCLUDED.warnings
        """,
        id,
        query_log_id,
        query_text,
        mode,
        profile,
        model,
        prompt_name,
        prompt_version,
        collections,
        context_scope,
        retrieved_count,
        source_ids,
        latency_ms,
        cost_usd,
        compliance_outcome,
        warnings,
    )


async def insert_answer_feedback(
    pool: asyncpg.Pool,
    *,
    id: str,
    answer_id: str,
    rating: str,
    reason: str | None,
    selected_source_ids: list[str],
    caller: str | None,
) -> None:
    await pool.execute(
        """
        INSERT INTO answer_feedback (
            id,
            answer_id,
            rating,
            reason,
            selected_source_ids,
            caller
        ) VALUES ($1,$2,$3,$4,$5,$6)
        """,
        id,
        answer_id,
        rating,
        reason,
        selected_source_ids,
        caller,
    )


__all__ = ["insert_answer_feedback", "insert_answer_log"]

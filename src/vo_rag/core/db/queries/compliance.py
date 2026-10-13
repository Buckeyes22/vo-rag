"""Typed query helpers for compliance_log."""

from __future__ import annotations

import asyncpg  # type: ignore[import-untyped]

from vo_rag.core.models.compliance import ComplianceLog


def _record_to_compliance(record: asyncpg.Record) -> ComplianceLog:
    return ComplianceLog.model_validate({**record})


async def insert_compliance_log(
    pool: asyncpg.Pool,
    log_entry: ComplianceLog,
    *,
    gate_mode: str | None = None,
) -> None:
    """Insert compliance log row."""
    args = (
        log_entry.id,
        log_entry.artifact_id,
        log_entry.query_text,
        log_entry.output_text,
        [score.model_dump(mode="json") for score in log_entry.similarity_scores],
        log_entry.source_contributions,
        {key: value.value for key, value in log_entry.citation_tiers.items()},
        log_entry.passed,
        log_entry.failure_reasons,
        log_entry.mode,
        log_entry.created_at,
    )

    if gate_mode is None:
        await pool.execute(
            """
            INSERT INTO compliance_log (
                id, artifact_id, query_text, output_text,
                similarity_scores, source_contributions, citation_tiers,
                passed, failure_reasons, mode, created_at
            ) VALUES (
                $1, $2, $3, $4,
                $5, $6, $7,
                $8, $9, $10, $11
            )
            """,
            *args,
        )
        return

    try:
        await pool.execute(
            """
            INSERT INTO compliance_log (
                id, artifact_id, query_text, output_text,
                similarity_scores, source_contributions, citation_tiers,
                passed, failure_reasons, mode, gate_mode, created_at
            ) VALUES (
                $1, $2, $3, $4,
                $5, $6, $7,
                $8, $9, $10, $11, $12
            )
            """,
            *args[:-1],
            gate_mode,
            args[-1],
        )
    except asyncpg.UndefinedColumnError:
        await pool.execute(
            """
            INSERT INTO compliance_log (
                id, artifact_id, query_text, output_text,
                similarity_scores, source_contributions, citation_tiers,
                passed, failure_reasons, mode, created_at
            ) VALUES (
                $1, $2, $3, $4,
                $5, $6, $7,
                $8, $9, $10, $11
            )
            """,
            *args,
        )


async def get_compliance_log(pool: asyncpg.Pool, log_id: str) -> ComplianceLog | None:
    """Get compliance log by ID."""
    row = await pool.fetchrow("SELECT * FROM compliance_log WHERE id = $1", log_id)
    if row is None:
        return None
    return _record_to_compliance(row)


async def recent_failures(pool: asyncpg.Pool, limit: int = 50) -> list[ComplianceLog]:
    """Return most recent failed compliance rows."""
    rows = await pool.fetch(
        """
        SELECT *
        FROM compliance_log
        WHERE passed = FALSE
        ORDER BY created_at DESC, id DESC
        LIMIT $1
        """,
        limit,
    )
    return [_record_to_compliance(row) for row in rows]

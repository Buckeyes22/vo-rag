"""Typed query helpers for regurgitation_reports.

There is no typed ``TestReport`` model here: the helpers take and return
``dict[str, Any]`` at the boundary.
"""

from __future__ import annotations

from typing import Any

import asyncpg  # type: ignore[import-untyped]

from vo_rag.core.models import ulid_new


async def insert_regurgitation_report(
    conn: asyncpg.Connection,
    report: dict[str, Any],
    *,
    report_id: str | None = None,
) -> str:
    row_id = report_id or ulid_new()
    await conn.execute(
        """
        INSERT INTO regurgitation_reports (
            id,
            total_tests,
            extraction_tests,
            reconstruction_tests,
            verbatim_detected,
            near_verbatim_detected,
            max_verbatim_length,
            max_similarity_score,
            passed,
            failed_tests,
            threshold_recommendations
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """,
        row_id,
        report["total_tests"],
        report["extraction_tests"],
        report["reconstruction_tests"],
        report["verbatim_detected"],
        report["near_verbatim_detected"],
        report["max_verbatim_length"],
        report["max_similarity_score"],
        report["passed"],
        list(report["failed_tests"]),
        report["threshold_recommendations"],
    )
    return row_id

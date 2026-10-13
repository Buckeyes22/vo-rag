"""Typed query helpers for citation_assignments.

There is no typed ``CitationAssignment`` model here: the SQL and ID hashing
helpers use ``dict[str, Any]`` at the boundary.
"""

from __future__ import annotations

import hashlib
from typing import Any

import asyncpg  # type: ignore[import-untyped]


def _assignment_id(output_id: str, assignment: dict[str, Any]) -> str:
    key = (
        f"{output_id}|{assignment['start_offset']}|{assignment['end_offset']}|"
        f"{assignment['tier']}|{assignment.get('source_id') or ''}|{assignment['text_segment']}"
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


async def insert_citation_assignments(
    pool: asyncpg.Pool,
    output_id: str,
    assignments: list[dict[str, Any]],
) -> None:
    if not assignments:
        return

    await pool.executemany(
        """
        INSERT INTO citation_assignments (
            id,
            output_id,
            text_segment,
            start_offset,
            end_offset,
            tier,
            source_id,
            source_title,
            confidence
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
        ON CONFLICT (id) DO UPDATE SET
            text_segment = EXCLUDED.text_segment,
            start_offset = EXCLUDED.start_offset,
            end_offset = EXCLUDED.end_offset,
            tier = EXCLUDED.tier,
            source_id = EXCLUDED.source_id,
            source_title = EXCLUDED.source_title,
            confidence = EXCLUDED.confidence
        """,
        [
            (
                _assignment_id(output_id, assignment),
                output_id,
                assignment["text_segment"],
                assignment["start_offset"],
                assignment["end_offset"],
                assignment["tier"],
                assignment.get("source_id"),
                assignment.get("source_title"),
                assignment["confidence"],
            )
            for assignment in assignments
        ],
    )


async def load_citation_assignments(pool: asyncpg.Pool, output_id: str) -> list[dict[str, Any]]:
    rows = await pool.fetch(
        """
        SELECT text_segment, start_offset, end_offset, tier, source_id, source_title, confidence
        FROM citation_assignments
        WHERE output_id = $1
        ORDER BY start_offset, end_offset
        """,
        output_id,
    )
    return [dict(row) for row in rows]

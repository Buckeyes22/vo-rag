"""Typed query helpers for provenance_records.

There are no typed ``ProvenanceRecord``/``Source`` models here: the helpers
take and return ``dict[str, Any]`` at the boundary.
"""

from __future__ import annotations

from typing import Any

import asyncpg  # type: ignore[import-untyped]


async def insert_provenance_record(pool: asyncpg.Pool, record: dict[str, Any]) -> None:
    await pool.execute(
        """
        INSERT INTO provenance_records (
            output_id,
            query_text,
            query_mode,
            total_sources,
            collection_searched,
            generation_model,
            sources_used,
            retrieval_timestamp
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (output_id) DO UPDATE SET
            query_text = EXCLUDED.query_text,
            query_mode = EXCLUDED.query_mode,
            total_sources = EXCLUDED.total_sources,
            collection_searched = EXCLUDED.collection_searched,
            generation_model = EXCLUDED.generation_model,
            sources_used = EXCLUDED.sources_used,
            retrieval_timestamp = EXCLUDED.retrieval_timestamp
        """,
        record["output_id"],
        record["query_text"],
        record["query_mode"],
        record["total_sources"],
        record["collection_searched"],
        record["generation_model"],
        list(record["sources_used"]),
        record["retrieval_timestamp"],
    )


async def load_provenance_record(pool: asyncpg.Pool, output_id: str) -> dict[str, Any] | None:
    row = await pool.fetchrow(
        """
        SELECT
            output_id,
            query_text,
            query_mode,
            total_sources,
            collection_searched,
            generation_model,
            sources_used,
            retrieval_timestamp
        FROM provenance_records
        WHERE output_id = $1
        """,
        output_id,
    )
    if row is None:
        return None
    return dict(row)


async def _load_sources(pool: asyncpg.Pool, source_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not source_ids:
        return {}

    rows = await pool.fetch(
        """
        SELECT id, title, COALESCE(author, '') AS author, isbn, license_class
        FROM sources
        WHERE id = ANY($1::text[])
        """,
        source_ids,
    )
    return {str(row["id"]): dict(row) for row in rows}

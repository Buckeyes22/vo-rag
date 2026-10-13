"""Per-parent fan-out reader for compliance_sub_chunks.

The only helper takes an injected pool and returns raw rows as
``dict[str, Any]``; there is no typed ``ComplianceSubChunk`` model, so callers
validate the rows themselves. There are no insert or single-parent fetch
helpers.
"""

from __future__ import annotations

from typing import Any

import asyncpg  # type: ignore[import-untyped]


async def fetch_sub_chunks_for_parent_ids(
    pool: asyncpg.Pool,
    parent_chunk_ids: list[str],
) -> list[dict[str, Any]]:
    if not parent_chunk_ids:
        return []

    rows = await pool.fetch(
        """
        SELECT *
        FROM compliance_sub_chunks
        WHERE parent_chunk_id = ANY($1::text[])
        ORDER BY parent_chunk_id, position_in_parent
        """,
        parent_chunk_ids,
    )
    return [dict(row) for row in rows]

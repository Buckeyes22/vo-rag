from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import asyncpg  # type: ignore[import-untyped]


@dataclass(frozen=True)
class QueryLogRow:
    id: str
    query_text: str
    mode: str
    collections: list[str]
    top_k: int | None
    caller: str | None
    created_at: datetime


def _to_row(record: asyncpg.Record) -> QueryLogRow:
    return QueryLogRow(
        id=str(record["id"]),
        query_text=str(record["query_text"]),
        mode=str(record["mode"]),
        collections=[str(collection) for collection in record["collections"]],
        top_k=int(record["top_k"]) if record["top_k"] is not None else None,
        caller=str(record["caller"]) if record["caller"] is not None else None,
        created_at=record["created_at"],
    )


async def insert_query_log(
    pool: asyncpg.Pool,
    *,
    id: str,
    query_text: str,
    mode: str,
    collections: list[str],
    top_k: int | None,
    caller: str | None,
) -> None:
    await pool.execute(
        """
        INSERT INTO query_log (id, query_text, mode, collections, top_k, caller)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        id,
        query_text,
        mode,
        collections,
        top_k,
        caller,
    )


async def fetch_recent_queries(pool: asyncpg.Pool, *, limit: int = 100) -> list[QueryLogRow]:
    rows = await pool.fetch(
        """
        SELECT id, query_text, mode, collections, top_k, caller, created_at
        FROM query_log
        ORDER BY created_at DESC, id DESC
        LIMIT $1
        """,
        limit,
    )
    return [_to_row(row) for row in rows]

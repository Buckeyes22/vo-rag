"""Canonical source identity helpers for book_rag.

The canonical id prefix is `vo-rag-corpus-sha256:`. The three-pass
lookup (exact / prefix / upper) is load-bearing for the idempotency contract
on re-ingest of the same file hash under different operator-supplied source ids.
"""

from __future__ import annotations

from collections.abc import Awaitable, Mapping
from typing import Any, Literal, Protocol

from pydantic import BaseModel

MatchKind = Literal["exact", "prefix", "upper"]

_SOURCE_LOOKUP_SQL = """
SELECT
    id,
    title,
    author,
    COALESCE(license_class, 'unknown') AS license_class,
    file_hash,
    COALESCE(NULLIF(metadata->>'project_id', ''), 'default') AS project_id
FROM sources
WHERE id = $1::text
"""


class _SourceIdentityConnection(Protocol):
    def fetchrow(self, query: str, key: str) -> Awaitable[Mapping[str, Any] | None]:
        """Return one row for the supplied source key, asyncpg-style."""


class CanonicalSource(BaseModel):
    canonical_id: str
    title: str
    author: str | None
    license_class: str
    project_id: str
    file_hash: str | None
    matched_by: MatchKind


def canonical_source_id(file_hash: str) -> str:
    return f"vo-rag-corpus-sha256:{file_hash}"


async def resolve_canonical_source(
    source_id: str, *, conn: _SourceIdentityConnection
) -> CanonicalSource | None:
    for matched_by, lookup_key in _lookup_candidates(source_id):
        row = await conn.fetchrow(_SOURCE_LOOKUP_SQL, lookup_key)
        if row is not None:
            return _row_to_canonical_source(row, matched_by=matched_by)
    return None


def _lookup_candidates(source_id: str) -> list[tuple[MatchKind, str]]:
    return [
        ("exact", source_id),
        ("prefix", canonical_source_id(source_id)),
        ("upper", source_id.upper()),
    ]


def _row_to_canonical_source(row: Mapping[str, Any], *, matched_by: MatchKind) -> CanonicalSource:
    row_id = str(row["id"])
    file_hash = _optional_string(row.get("file_hash"))
    canonical_id = canonical_source_id(file_hash) if file_hash else row_id
    project_id = _optional_string(row.get("project_id")) or "default"
    license_class = _optional_string(row.get("license_class")) or "unknown"
    return CanonicalSource(
        canonical_id=canonical_id,
        title=str(row["title"]),
        author=_optional_string(row.get("author")),
        license_class=license_class,
        project_id=project_id,
        file_hash=file_hash,
        matched_by=matched_by,
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


__all__ = ["CanonicalSource", "canonical_source_id", "resolve_canonical_source"]

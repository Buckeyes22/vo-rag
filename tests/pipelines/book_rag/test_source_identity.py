"""Tests for canonical book source identity resolution.

Content-hash source ids carry the `vo-rag-corpus-sha256:` prefix. Resolution
tries an exact match, then the prefixed id, then the upper-cased id.
"""

from __future__ import annotations

from typing import Any

import pytest

from vo_rag.pipelines.book_rag.source_identity import (
    CanonicalSource,
    canonical_source_id,
    resolve_canonical_source,
)

# Fake file hash used as a test fixture. The literal looks like a 64-bit
# hex digest so detect-secrets flags it; the pragma keeps the gate green
# while preserving the realistic shape. The same constant is reused
# across the file so the allowlist only has to be declared once.
_FAKE_FILE_HASH = "30d9a4332ee35f21"  # pragma: allowlist secret

# Second fake hash for the idempotency test.
_FAKE_FILE_HASH_2 = "abcdef1234567890"  # pragma: allowlist secret


class _FakeConn:
    def __init__(self, *, key: str, row: dict[str, Any] | None) -> None:
        self.key = key
        self.row = row
        self.calls: list[tuple[str, str]] = []

    async def fetchrow(self, sql: str, key: str) -> dict[str, Any] | None:
        self.calls.append((sql, key))
        if key == self.key:
            return self.row
        return None


def _source_row(
    *,
    source_id: str,
    file_hash: str | None = _FAKE_FILE_HASH,
    project_id: str = "motorsport",
) -> dict[str, Any]:
    return {
        "id": source_id,
        "title": "Practical Engine Tuning",
        "author": "Jordan Avery",
        "license_class": "unlicensed",
        "project_id": project_id,
        "file_hash": file_hash,
    }


def test_canonical_source_id_uses_vo_rag_prefix() -> None:
    assert canonical_source_id(_FAKE_FILE_HASH) == f"vo-rag-corpus-sha256:{_FAKE_FILE_HASH}"


@pytest.mark.asyncio
async def test_resolve_canonical_source_exact_match() -> None:
    sid = f"vo-rag-corpus-sha256:{_FAKE_FILE_HASH}"
    conn = _FakeConn(key=sid, row=_source_row(source_id=sid))

    source = await resolve_canonical_source(sid, conn=conn)

    assert source == CanonicalSource(
        canonical_id=sid,
        title="Practical Engine Tuning",
        author="Jordan Avery",
        license_class="unlicensed",
        project_id="motorsport",
        file_hash=_FAKE_FILE_HASH,
        matched_by="exact",
    )
    assert [key for _, key in conn.calls] == [sid]
    assert "metadata->>'project_id'" in conn.calls[0][0]


@pytest.mark.asyncio
async def test_resolve_canonical_source_prefix_match() -> None:
    bare_sid = _FAKE_FILE_HASH
    prefixed_sid = canonical_source_id(bare_sid)
    conn = _FakeConn(key=prefixed_sid, row=_source_row(source_id=prefixed_sid))

    source = await resolve_canonical_source(bare_sid, conn=conn)

    assert source is not None
    assert source.canonical_id == prefixed_sid
    assert source.matched_by == "prefix"
    assert [key for _, key in conn.calls] == [bare_sid, prefixed_sid]


@pytest.mark.asyncio
async def test_resolve_canonical_source_upper_match() -> None:
    conn = _FakeConn(key="R-408", row=_source_row(source_id="R-408", file_hash=None))

    source = await resolve_canonical_source("r-408", conn=conn)

    assert source is not None
    assert source.canonical_id == "R-408"
    assert source.matched_by == "upper"
    assert [key for _, key in conn.calls] == [
        "r-408",
        "vo-rag-corpus-sha256:r-408",
        "R-408",
    ]


@pytest.mark.asyncio
async def test_resolve_canonical_source_unresolved_returns_none() -> None:
    conn = _FakeConn(key="other", row=_source_row(source_id="other"))

    source = await resolve_canonical_source("missing", conn=conn)

    assert source is None
    assert [key for _, key in conn.calls] == [
        "missing",
        "vo-rag-corpus-sha256:missing",
        "MISSING",
    ]


@pytest.mark.asyncio
async def test_resolve_canonical_source_defaults_project_id() -> None:
    sid = f"vo-rag-corpus-sha256:{_FAKE_FILE_HASH}"
    row = _source_row(source_id=sid, project_id="")
    conn = _FakeConn(key=sid, row=row)

    source = await resolve_canonical_source(sid, conn=conn)

    assert source is not None
    assert source.project_id == "default"


def test_canonical_source_id_is_idempotent() -> None:
    """Re-deriving the id from the same hash is byte-identical (F-024 contract)."""
    assert canonical_source_id(_FAKE_FILE_HASH_2) == canonical_source_id(_FAKE_FILE_HASH_2)

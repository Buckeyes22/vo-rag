"""Out-of-band batches must not leave envelopes the dispatcher will claim (VOR-1406).

``ops/rc_ingest_batch.py`` inserts ``file_drop`` envelopes to drive book_rag into
an arbitrary collection. Inserted as ``received`` they match the dispatcher's
claim predicate exactly:

    status IN (received, stale processing)
    AND source_type = 'file_drop'
    AND (target_pipeline IS NULL OR target_pipeline = 'book_rag')
    AND raw_content.garage_key IS NOT NULL AND raw_content.file_hash IS NOT NULL

Observed 2026-08-14: 4153 Reddit envelopes were claimable, and the dispatcher
had to be stopped for the entire run to keep them out of vo_rag_books_raw_v1.

Owning them as ``processing`` closes that, but leaves the stale-reclaim window:
a batch longer than that would have its not-yet-started work stolen mid-run.
Hence the heartbeat, mirroring the dispatcher's own
``_heartbeat_processing_envelope``.
"""

from __future__ import annotations

from typing import Any

import pytest

from vo_rag.core.ingest.batch_ownership import (
    DEFAULT_OWNERSHIP_HEARTBEAT_S,
    touch_owned_envelopes,
)


class _FakePool:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> str:
        self.calls.append((query, args))
        return "UPDATE 2"


@pytest.mark.asyncio
async def test_touch_refreshes_updated_at_for_the_given_envelopes() -> None:
    pool = _FakePool()

    touched = await touch_owned_envelopes(pool, ["a", "b"])

    assert touched == 2
    query, args = pool.calls[0]
    assert "SET updated_at = NOW()" in query
    assert args[0] == ["a", "b"]


@pytest.mark.asyncio
async def test_touch_only_affects_rows_still_processing() -> None:
    """A completed envelope must never be dragged back into the in-flight set."""
    pool = _FakePool()

    await touch_owned_envelopes(pool, ["a"])

    query, _ = pool.calls[0]
    assert "status = 'processing'" in query


@pytest.mark.asyncio
async def test_empty_input_issues_no_query() -> None:
    pool = _FakePool()

    assert await touch_owned_envelopes(pool, []) == 0
    assert pool.calls == []


@pytest.mark.asyncio
async def test_blank_ids_are_ignored() -> None:
    pool = _FakePool()

    touched = await touch_owned_envelopes(pool, ["", "a", ""])

    assert touched == 1
    assert pool.calls[0][1][0] == ["a"]


def test_heartbeat_interval_is_well_inside_the_dispatcher_stale_window() -> None:
    """The batch heartbeat must comfortably beat the dispatcher reclaim window
    by a wide margin or a long batch loses its backlog."""
    from vo_rag.pipelines.book_rag.dispatcher import DEFAULT_STALE_TIMEOUT

    assert DEFAULT_STALE_TIMEOUT.total_seconds() / 4 > DEFAULT_OWNERSHIP_HEARTBEAT_S

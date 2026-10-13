"""Streaming ingest returns the physical IDs resolved after warning publication."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("llama_index")

from tests.pipelines.book_rag.test_quality_availability import _context, _pipeline
from vo_rag.pipelines.book_rag.workflow_li import BookRagWorkflow, ValidatedEvent


@pytest.mark.asyncio
async def test_streamed_output_and_state_use_final_warning_publication_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline()
    smoke_ctx = _context(pipeline)
    workflow = BookRagWorkflow(
        garage=SimpleNamespace(),
        qdrant_store=SimpleNamespace(),
        embedding_client=SimpleNamespace(),
        llm_client=None,
        retriever=SimpleNamespace(),
        parser_router=SimpleNamespace(),
        config=pipeline.config,
    )
    state = SimpleNamespace(raw_point_ids=[], sterile_point_ids=[], smoke_decision=None)

    @asynccontextmanager
    async def edit_state() -> AsyncGenerator[Any, None]:
        yield state

    context = SimpleNamespace(
        store=SimpleNamespace(
            get=AsyncMock(return_value=smoke_ctx.envelope), edit_state=edit_state
        ),
        write_event_to_stream=lambda event: None,
    )

    async def persist(ctx: Any) -> None:
        ctx.raw_point_ids = ["current-raw"]
        ctx.sterile_point_ids = ["current-sterile"]

    monkeypatch.setattr(
        workflow, "_assert_context_generation_lease", AsyncMock(return_value=smoke_ctx)
    )
    monkeypatch.setattr(workflow, "_record_book_quality", AsyncMock())
    monkeypatch.setattr(
        workflow,
        "_generation_coordinator",
        lambda: SimpleNamespace(_persist_quality_warnings=persist),
    )
    event = ValidatedEvent(
        source_id=smoke_ctx.source_id,
        parse_meta={},
        chunks=[],
        raw_point_ids=["stale-raw"],
        sterile_point_ids=["stale-sterile"],
        validation_report=None,
    )
    result = await workflow.retrieval_smoke(context, event)
    assert result.result.qdrant_point_ids == ["current-raw", "current-sterile"]
    assert state.raw_point_ids == ["current-raw"]
    assert state.sterile_point_ids == ["current-sterile"]

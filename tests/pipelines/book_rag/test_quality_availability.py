"""Usable stored passages survive failed or unavailable optional quality checks."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from vo_rag.core.models import ulid_new
from vo_rag.core.models.envelope import IntakeEnvelope, IntakeSource
from vo_rag.pipelines.book_rag import retrieval_smoke as smoke
from vo_rag.pipelines.book_rag import workflow as module
from vo_rag.pipelines.book_rag.config import BookRagConfig
from vo_rag.pipelines.book_rag.validator import ValidationCheck, ValidationReport
from vo_rag.pipelines.book_rag.workflow import BookRagContext, BookRagPipeline


def _pipeline() -> BookRagPipeline:
    return BookRagPipeline(
        garage=SimpleNamespace(),
        qdrant_store=SimpleNamespace(),
        embedding_client=SimpleNamespace(),
        llm_client=SimpleNamespace(),
        retriever=SimpleNamespace(),
        parser_router=SimpleNamespace(),
        config=BookRagConfig(retrieval_smoke_enabled=True),
    )


def _context(pipeline: BookRagPipeline) -> BookRagContext:
    return BookRagContext(
        envelope=IntakeEnvelope(
            id=ulid_new(),
            source_type=IntakeSource.FILE_DROP,
            raw_content={},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        ),
        config=pipeline.config,
        source_id="verified-source",
        raw_point_ids=["acknowledged-point"],
    )


@pytest.mark.asyncio
async def test_failed_validator_preserves_report_and_visible_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Validator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def validate(self, **kwargs: Any) -> ValidationReport:
            return ValidationReport(
                document_id="doc",
                source_id="verified-source",
                passed=False,
                checks=[
                    ValidationCheck(name="sample_query_returns", passed=False, detail="0 hits")
                ],
            )

    monkeypatch.setattr(module, "IngestionValidator", Validator)
    pipeline = _pipeline()
    ctx = _context(pipeline)
    await pipeline._validate(ctx)
    await pipeline._persist_quality_warnings(ctx)
    assert ctx.validation_report is not None and not ctx.validation_report.passed
    assert ctx.raw_point_ids == ["acknowledged-point"]
    assert ctx.envelope.metadata["quality_warnings"][0]["code"] == "sample_query_returns"


@pytest.mark.asyncio
async def test_stalled_smoke_is_bounded_and_does_not_hold_next_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def queries(**kwargs: Any) -> list[Any]:
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()
        return []

    monkeypatch.setattr(smoke, "generate_self_queries", queries)
    monkeypatch.setattr(module, "QUALITY_AUDIT_TIMEOUT_SECONDS", 0.01)
    pipeline = _pipeline()
    for _ in range(2):
        ctx = _context(pipeline)
        await asyncio.wait_for(pipeline._retrieval_smoke(ctx), timeout=0.5)
        assert ctx.smoke_result is None
        assert ctx.quality_warnings[0]["details"] == "TimeoutError"
        assert ctx.raw_point_ids == ["acknowledged-point"]
    assert calls == 2


@pytest.mark.asyncio
async def test_warning_storage_failure_keeps_warnings_and_passages() -> None:
    async def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("offline")

    pipeline = _pipeline()
    pipeline.qdrant_store = SimpleNamespace(set_source_quality_warnings=fail)
    ctx = _context(pipeline)
    ctx.quality_warnings = [{"stage": "validation", "code": "audit_unavailable"}]
    await pipeline._persist_quality_warnings(ctx)
    assert ctx.raw_point_ids == ["acknowledged-point"]
    assert ctx.envelope.metadata["quality_warnings"][0]["code"] == "audit_unavailable"
    assert ctx.envelope.metadata["quality_warnings"][-1]["code"] == "warning_persistence_pending"


@pytest.mark.asyncio
async def test_envelope_warning_failure_still_delivers_passage_warning() -> None:
    from unittest.mock import AsyncMock

    pipeline = _pipeline()
    pipeline.pg_pool = SimpleNamespace(execute=AsyncMock(side_effect=OSError("offline")))
    writer = AsyncMock()
    pipeline.qdrant_store = SimpleNamespace(set_source_quality_warnings=writer)
    ctx = _context(pipeline)
    ctx.quality_warnings = [{"stage": "parse", "code": "low_page_coverage"}]
    await pipeline._persist_quality_warnings(ctx)
    writer.assert_awaited_once()
    assert writer.await_args is not None
    assert writer.await_args.args[2][0]["code"] == "low_page_coverage"
    assert writer.await_args.args[2][-1]["details"] == "envelope:OSError"

"""Manual OTel span coverage for coarse query and book_rag pipeline seams."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

from vo_rag.core.models.envelope import EnvelopeStatus, IntakeEnvelope, IntakeSource
from vo_rag.core.retrieval.execution import execute_retrieval
from vo_rag.core.retrieval.query_mode import QueryMode
from vo_rag.core.retrieval.search import HybridRetriever, SearchResult
from vo_rag.pipelines.book_rag import workflow as workflow_mod
from vo_rag.pipelines.book_rag.source_identity import canonical_source_id
from vo_rag.pipelines.book_rag.workflow import BookRagContext, BookRagPipeline

_TS = datetime(2026, 1, 1, tzinfo=UTC)
_HASH = "a" * 64
_DOC_TEXT = "SECRET_DOCUMENT_BODY_SHOULD_NOT_BE_AN_ATTRIBUTE"


@dataclasses.dataclass(frozen=True)
class _SpanHarness:
    exporter: InMemorySpanExporter
    tracer: Tracer


@pytest.fixture
def span_harness(monkeypatch: pytest.MonkeyPatch) -> Iterator[_SpanHarness]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    def get_test_tracer(name: str) -> Tracer:
        return provider.get_tracer(name)

    monkeypatch.setattr(workflow_mod, "get_tracer", get_test_tracer, raising=False)
    yield _SpanHarness(exporter=exporter, tracer=provider.get_tracer("vo_rag.query"))
    provider.shutdown()


class _FakeRetriever:
    async def cross_collection_search(
        self,
        *,
        query: str,
        collections: list[str],
        top_k: int,
        filters: dict[str, Any] | None,
        mode: QueryMode,
        visual_mode: str | None,
    ) -> list[SearchResult]:
        return [
            SearchResult(
                id="point-1",
                score=0.9,
                collection=collections[0],
                payload={"chunk_id": "chunk-1", "content": _DOC_TEXT},
            )
        ]


@pytest.mark.asyncio
async def test_query_retrieval_and_rerank_emit_coarse_spans(
    span_harness: _SpanHarness,
) -> None:
    def rerank(query: str, hits: list[SearchResult]) -> list[SearchResult]:
        return hits

    results = await execute_retrieval(
        retriever=cast(HybridRetriever, _FakeRetriever()),
        query="Where should trail braking start?",
        mode=QueryMode.INTERNAL,
        requested_collections=["vo_rag_books_raw_v1"],
        top_k=1,
        rerank_fn=rerank,
        phase_context=span_harness.tracer.start_as_current_span,
    )

    assert results[0].payload["content"] == _DOC_TEXT
    spans = span_harness.exporter.get_finished_spans()
    assert {span.name for span in spans} >= {"retrieval", "rerank"}
    assert all(_DOC_TEXT not in str(span.attributes) for span in spans)
    assert all("Where should trail braking start?" not in str(span.attributes) for span in spans)


def _envelope() -> IntakeEnvelope:
    return IntakeEnvelope(
        id="01KT6XE8M1MW42M0Y0JJKRZSVP",  # pragma: allowlist secret
        source_type=IntakeSource.FILE_DROP,
        raw_content={"garage_key": "drops/book.pdf", "file_hash": _HASH},
        status=EnvelopeStatus.RECEIVED,
        created_at=_TS,
        updated_at=_TS,
        metadata={"title": _DOC_TEXT},
    )


class _StubBookRagPipeline(BookRagPipeline):
    async def _register_source(self, ctx: BookRagContext) -> None:
        ctx.source_id = canonical_source_id(_HASH)

    async def _parse(self, ctx: BookRagContext) -> None:
        return None

    async def _visual(self, ctx: BookRagContext) -> None:
        return None

    async def _chunk_embed(self, ctx: BookRagContext) -> None:
        ctx.raw_point_ids = ["raw-point-1"]

    async def _sterilize(self, ctx: BookRagContext) -> None:
        return None

    async def _advanced_understanding(self, ctx: BookRagContext) -> None:
        return None

    async def _register_artifact(self, ctx: BookRagContext) -> None:
        return None

    async def _validate(self, ctx: BookRagContext) -> None:
        return None

    async def _retrieval_smoke(self, ctx: BookRagContext) -> None:
        return None


async def _no_existing_points(collection: str, source_id: str) -> int:
    return 0


def _pipeline() -> BookRagPipeline:
    return _StubBookRagPipeline(
        garage=SimpleNamespace(),
        # count_source is the VOR-1403 duplicate gate ingest() consults before
        # any step runs; 0 keeps this test on the "new source, run it" path.
        qdrant_store=SimpleNamespace(count_source=_no_existing_points),
        embedding_client=SimpleNamespace(),
        llm_client=SimpleNamespace(),
        retriever=SimpleNamespace(),
        parser_router=SimpleNamespace(),
        figures=SimpleNamespace(),
    )


@pytest.mark.asyncio
async def test_ingest_steps_emit_source_scoped_spans(
    span_harness: _SpanHarness,
) -> None:
    pipeline = _pipeline()

    output = await pipeline.ingest(_envelope())

    expected_source_id = canonical_source_id(_HASH)
    assert output.source_id == expected_source_id
    spans = span_harness.exporter.get_finished_spans()
    span_by_name = {span.name: span for span in spans}
    assert "ingest.register_source" in span_by_name
    assert "ingest.chunk_embed" in span_by_name
    register_attrs = span_by_name["ingest.register_source"].attributes
    chunk_attrs = span_by_name["ingest.chunk_embed"].attributes
    assert register_attrs is not None
    assert chunk_attrs is not None
    assert register_attrs["vo_rag.source_id"] == expected_source_id
    assert chunk_attrs["vo_rag.source_id"] == expected_source_id
    assert all(_DOC_TEXT not in str(span.attributes) for span in spans)

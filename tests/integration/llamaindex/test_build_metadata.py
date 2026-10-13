"""VOR-208: recover book metadata post-ingest for AdvancedBuildWorkflow.load_corpus.

These tests exercise the helper seams that lift CLI build.py logic into the
workflow: scrolling the raw collection with a version_hash scope, pulling
(title, author, category_ids) off the first chunk's stamped metadata, and the
settings-aware BuildConfig constructors. They also cover the ingest DAG seam:
_advanced_understanding is a no-op when disabled and raises a pointer error
(rather than a bare NotImplementedError) when an advanced layer is enabled.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import FieldCondition, MatchValue

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.dsi.builder import DsiBuildStats
from vo_rag.core.llm.client import LLMClient
from vo_rag.core.models.envelope import IntakeEnvelope, IntakeSource
from vo_rag.core.storage.qdrant import QdrantStore
from vo_rag.integrations.llamaindex.build_workflow import (
    AdvancedBuildWorkflow,
    BuildState,
)
from vo_rag.pipelines.book_rag.config import BookRagConfig
from vo_rag.pipelines.book_rag.workflow import BookRagContext, BookRagPipeline


class _FakeStore:
    def __init__(self) -> None:
        self._client = object()


class _FakePoint:
    def __init__(self, point_id: str, payload: dict[str, Any]) -> None:
        self.id = point_id
        self.payload = payload


class _FakeScrollClient:
    def __init__(self, pages: list[tuple[list[_FakePoint], str | None]]) -> None:
        self.pages = pages
        self.calls: list[dict[str, Any]] = []

    async def scroll(self, **kwargs: Any) -> tuple[list[_FakePoint], str | None]:
        self.calls.append(kwargs)
        return self.pages.pop(0)


def _field_conditions(scroll_filter: Any) -> list[FieldCondition]:
    conditions = list(scroll_filter.must)
    assert all(isinstance(condition, FieldCondition) for condition in conditions)
    return cast(list[FieldCondition], conditions)


def _condition_pairs(conditions: list[FieldCondition]) -> list[tuple[str, Any]]:
    pairs: list[tuple[str, Any]] = []
    for condition in conditions:
        assert isinstance(condition.match, MatchValue)
        pairs.append((condition.key, condition.match.value))
    return pairs


@pytest.fixture
def workflow(monkeypatch: pytest.MonkeyPatch) -> AdvancedBuildWorkflow:
    store = _FakeStore()
    embed = AsyncMock()
    llm = AsyncMock(spec=LLMClient)
    llm_clients = cast(dict[str, LLMClient], {"text": llm, "dsi": llm})

    serverless = MagicMock()
    serverless.text_model_name = "qwen3-6-27b"
    serverless.dsi_model_name = "qwen3-6-35b-a3b"
    settings = MagicMock()
    settings.serverless = serverless

    wf = AdvancedBuildWorkflow(
        store=cast(QdrantStore, store),
        embedding_client=embed,
        llm_clients=llm_clients,
        settings=settings,
    )
    # Prevent accidental Qdrant or LLM calls in unit tests
    monkeypatch.setattr(wf, "_persist_graph", AsyncMock(return_value=(0, 0)))
    return wf


# ---------------------------------------------------------------------------
# _book_meta
# ---------------------------------------------------------------------------


def test_book_meta_recovers_title_author_category_ids(workflow: AdvancedBuildWorkflow) -> None:
    chunks = [
        RetrievalChunk(
            chunk_id="c-1",
            chunk_type="text",
            document_id="doc-1",
            source_id="src-1",
            section_path="Ch1",
            section_title="Chapter 1",
            chapter_path="Ch1",
            content="body",
            embedding_text="body",
            token_count=2,
            metadata={
                "title": "Test Book",
                "author": "Test Author",
                "domain_tags": ["tag-a", "tag-b"],
            },
        ),
        RetrievalChunk(
            chunk_id="c-2",
            chunk_type="text",
            document_id="doc-1",
            source_id="src-1",
            section_path="Ch2",
            section_title="Chapter 2",
            chapter_path="Ch2",
            content="more body",
            embedding_text="more body",
            token_count=3,
            metadata={},
        ),
    ]

    title, author, category_ids = workflow._book_meta(chunks)

    assert title == "Test Book"
    assert author == "Test Author"
    assert category_ids == ["tag-a", "tag-b"]


def test_book_meta_falls_back_to_source_book(workflow: AdvancedBuildWorkflow) -> None:
    chunk = RetrievalChunk(
        chunk_id="c-1",
        chunk_type="text",
        document_id="doc-1",
        source_id="src-1",
        section_path="Ch1",
        section_title="Chapter 1",
        chapter_path="Ch1",
        content="body",
        embedding_text="body",
        token_count=2,
        metadata={"source_book": "Legacy Title", "author": "Legacy Author"},
    )

    title, author, category_ids = workflow._book_meta([chunk])

    assert title == "Legacy Title"
    assert author == "Legacy Author"
    assert category_ids == []


def test_book_meta_empty_for_no_chunks(workflow: AdvancedBuildWorkflow) -> None:
    assert workflow._book_meta([]) == ("", "", [])


# ---------------------------------------------------------------------------
# _load_chunks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_chunks_adds_version_hash_to_scroll_filter(
    workflow: AdvancedBuildWorkflow,
) -> None:
    client = _FakeScrollClient(
        [
            (
                [
                    _FakePoint(
                        "point-1",
                        {
                            "chunk_id": "chunk-1",
                            "source_id": "src-1",
                            "document_id": "doc-1",
                            "content": "body",
                            "version_hash": "vh-1",
                        },
                    )
                ],
                None,
            )
        ]
    )
    cast(Any, workflow.store)._client = client

    chunks = await workflow._load_chunks("src-1", version_hash="vh-1")

    assert [chunk.chunk_id for chunk in chunks] == ["chunk-1"]
    must = _field_conditions(client.calls[0]["scroll_filter"])
    assert _condition_pairs(must) == [
        ("source_id", "src-1"),
        ("version_hash", "vh-1"),
    ]


@pytest.mark.asyncio
async def test_load_chunks_accumulates_paginated_scroll_results(
    workflow: AdvancedBuildWorkflow,
) -> None:
    client = _FakeScrollClient(
        [
            (
                [
                    _FakePoint(
                        "point-1",
                        {
                            "chunk_id": "chunk-1",
                            "source_id": "src-1",
                            "document_id": "doc-1",
                            "content": "first",
                            "version_hash": "vh-1",
                        },
                    )
                ],
                "offset-2",
            ),
            (
                [
                    _FakePoint(
                        "point-2",
                        {
                            "chunk_id": "chunk-2",
                            "source_id": "src-1",
                            "document_id": "doc-1",
                            "content": "second",
                            "version_hash": "vh-1",
                        },
                    )
                ],
                None,
            ),
        ]
    )
    cast(Any, workflow.store)._client = client

    chunks = await workflow._load_chunks("src-1", version_hash="vh-1")

    assert [chunk.chunk_id for chunk in chunks] == ["chunk-1", "chunk-2"]
    assert len(client.calls) == 2
    assert [call["offset"] for call in client.calls] == [None, "offset-2"]
    for call in client.calls:
        must = _field_conditions(call["scroll_filter"])
        assert _condition_pairs(must) == [
            ("source_id", "src-1"),
            ("version_hash", "vh-1"),
        ]


@pytest.mark.asyncio
async def test_load_chunks_raises_value_error_for_empty_scroll_result(
    workflow: AdvancedBuildWorkflow,
) -> None:
    client = _FakeScrollClient([([], None)])
    cast(Any, workflow.store)._client = client

    with pytest.raises(ValueError, match="no chunks for source_id=src-1"):
        await workflow._load_chunks("src-1")


@pytest.mark.asyncio
async def test_dsi_build_config_uses_settings_model_name(
    workflow: AdvancedBuildWorkflow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DsiBuildConfig.summary_model is resolved from settings, not hard-coded."""
    captured: list[Any] = []

    from vo_rag.integrations.llamaindex import build_workflow as bw_mod

    async def _fake_build_dsi_for_book(*, config: Any, **_: Any) -> tuple[None, dict[str, Any]]:
        captured.append(config)
        return None, {}

    monkeypatch.setattr(bw_mod, "build_dsi_for_book", _fake_build_dsi_for_book)
    monkeypatch.setattr(bw_mod, "ensure_dsi_collection", AsyncMock())
    monkeypatch.setattr(bw_mod, "persist_dsi_records", AsyncMock(return_value=0))

    st = BuildState(
        source_id="src-1",
        title="T",
        author="A",
        category_ids=["c"],
        chunks=[
            RetrievalChunk(
                chunk_id="c-1",
                chunk_type="text",
                document_id="d1",
                source_id="src-1",
                section_path="",
                section_title="",
                chapter_path="",
                content="x",
                embedding_text="x",
                token_count=1,
            )
        ],
    )
    workflow.llm["dsi"] = AsyncMock(spec=LLMClient)
    workflow.embed = AsyncMock()
    workflow.embed.embed = AsyncMock(return_value={"dense": [[]], "sparse": [[]], "colbert": [[]]})

    await workflow._build_dsi(st)

    assert len(captured) == 1
    config = captured[0]
    assert config.extraction_config.summary_model == "qwen3-6-35b-a3b"


@pytest.mark.asyncio
async def test_dsi_build_summary_is_json_serializable(
    workflow: AdvancedBuildWorkflow, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from vo_rag.integrations.llamaindex import build_workflow as bw_mod

    stats = DsiBuildStats(
        book_id="src-1",
        extraction_attempts=1,
        extraction_succeeded=True,
        validation_passed=True,
        validation_violations=0,
        final_action="accept",
        input_tokens=10,
        output_tokens=5,
        wall_time_ms=12,
    )

    async def _fake_build_dsi_for_book(*, config: Any, **_: Any) -> tuple[None, DsiBuildStats]:
        return None, stats

    monkeypatch.setattr(bw_mod, "build_dsi_for_book", _fake_build_dsi_for_book)

    st = BuildState(
        source_id="src-1",
        title="T",
        author="A",
        category_ids=["c"],
        chunks=[
            RetrievalChunk(
                chunk_id="c-1",
                chunk_type="text",
                document_id="d1",
                source_id="src-1",
                section_path="",
                section_title="",
                chapter_path="",
                content="x",
                embedding_text="x",
                token_count=1,
            )
        ],
    )

    result = await workflow._build_dsi(st)

    assert isinstance(result["stats"], dict)
    assert result["stats"]["book_id"] == "src-1"
    json.dumps(result)


# ---------------------------------------------------------------------------
# Ingest DAG seam: _advanced_understanding
# ---------------------------------------------------------------------------


def _envelope() -> IntakeEnvelope:
    return IntakeEnvelope(
        id="01KT6XE8M1MW42M0Y0JJKRZSVP",
        source_type=IntakeSource.FILE_DROP,
        raw_content={"file_hash": "a" * 64, "garage_key": "drops/book.pdf"},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _pipeline(**overrides: Any) -> BookRagPipeline:
    base = {
        "garage": AsyncMock(),
        "qdrant_store": AsyncMock(),
        "embedding_client": AsyncMock(),
        "llm_client": AsyncMock(),
        "retriever": AsyncMock(),
        "parser_router": AsyncMock(),
    }
    base.update(overrides)
    return BookRagPipeline(**base)


@pytest.mark.asyncio
async def test_advanced_understanding_noop_when_all_disabled() -> None:
    pipeline = _pipeline()
    ctx = BookRagContext(envelope=_envelope(), config=BookRagConfig())
    await pipeline._advanced_understanding(ctx)


@pytest.mark.asyncio
async def test_advanced_understanding_raises_pointer_error_when_enabled() -> None:
    pipeline = _pipeline(config=BookRagConfig(enable_dsi=True))
    ctx = BookRagContext(envelope=_envelope(), config=pipeline.config)
    with pytest.raises(RuntimeError, match="AdvancedBuildWorkflow") as exc_info:
        await pipeline._advanced_understanding(ctx)
    assert "dsi" in str(exc_info.value)
    assert "NotImplementedError" not in str(type(exc_info.value).__name__)

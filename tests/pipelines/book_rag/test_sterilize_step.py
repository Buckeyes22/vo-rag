"""Unit tests for the book_rag sterilize step.

Verifies do_sterilize and helper functions using fake LLM client,
fake QdrantStore, and mocked sterilize_chunks.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.llm.client import LLMClient
from vo_rag.core.llm.output_validator import ValidationResult
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.storage.points import VoPoint
from vo_rag.core.storage.qdrant import QdrantStore
from vo_rag.pipelines.book_rag.config import BookRagConfig
from vo_rag.pipelines.book_rag.steps.sterilize import do_sterilize

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


class _FakeProvider:
    name = "fake"
    model = "fake-model"

    async def complete_raw(self, **kwargs: Any) -> LLMResponse:
        return LLMResponse(
            text='{"sterile_content": "ok"}',
            model=self.model,
            input_tokens=10,
            output_tokens=5,
            provider=self.name,
        )

    async def aclose(self) -> None:
        return None


def _fake_llm_client() -> LLMClient:
    return LLMClient(provider=_FakeProvider())


def _chunk(**overrides: Any) -> RetrievalChunk:
    defaults: dict[str, Any] = {
        "chunk_id": "chunk-1",
        "chunk_type": "text",
        "document_id": "doc-1",
        "source_id": "src-1",
        "page_number": 1,
        "page_range": None,
        "section_path": "1.2 Braking",
        "section_title": "Braking Technique",
        "chapter_path": "Ch1",
        "content": "Trail braking transfers weight to the front axle.",
        "embedding_text": "Trail braking transfers weight to the front axle.",
        "token_count": 10,
        "has_image": False,
        "image_id": None,
        "image_ref": None,
        "vlm_description": None,
        "metadata": {},
    }
    defaults.update(overrides)
    return RetrievalChunk(**defaults)


@pytest.fixture
def fake_store() -> AsyncMock:
    store = AsyncMock(spec=QdrantStore)
    store.upsert_vectors = AsyncMock(return_value=None)
    return store


@pytest.fixture
def fake_embed() -> AsyncMock:
    async def _embed(texts: list[str]) -> dict[str, Any]:
        return {
            "dense": [[0.1] * 1024 for _ in texts],
            "sparse": [{"indices": [1, 2], "values": [0.5, 0.5]} for _ in texts],
            "colbert": [[[0.1] * 1024] for _ in texts],
        }

    return AsyncMock(side_effect=_embed)


@pytest.fixture
def config_enabled() -> BookRagConfig:
    return BookRagConfig(sterilize_enabled=True, min_visual_sterile_yield=0.70)


@pytest.fixture
def config_disabled() -> BookRagConfig:
    return BookRagConfig(sterilize_enabled=False, min_visual_sterile_yield=0.70)


# ---------------------------------------------------------------------------
# Invariant: disabled short-circuits with logged skip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sterilize_disabled_returns_raw_chunks(
    fake_store: AsyncMock,
    fake_embed: AsyncMock,
    config_disabled: BookRagConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    raw = [_chunk()]
    result = await do_sterilize(
        raw_chunks=raw,
        llm_client=_fake_llm_client(),
        config=config_disabled,
        store=fake_store,
        embed=fake_embed,
    )
    assert result == raw
    fake_store.upsert_vectors.assert_not_called()
    fake_embed.assert_not_called()
    assert "sterilize skipped" in caplog.text


# ---------------------------------------------------------------------------
# Invariant: enabled path calls sterilize_chunks + builds VoPoint + upserts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sterilize_enabled_calls_sterilize_chunks(
    fake_store: AsyncMock,
    fake_embed: AsyncMock,
    config_enabled: BookRagConfig,
) -> None:
    raw = [_chunk()]
    sterile = [_chunk(content="Sterilized.", embedding_text="Sterilized.")]

    client = _fake_llm_client()
    with patch(
        "vo_rag.pipelines.book_rag.steps.sterilize.sterilize_chunks",
        new_callable=AsyncMock,
        return_value=sterile,
    ) as mock_sterilize:
        result = await do_sterilize(
            raw_chunks=raw,
            llm_client=client,
            config=config_enabled,
            store=fake_store,
            embed=fake_embed,
        )

    mock_sterilize.assert_awaited_once_with(raw, client)
    assert result == sterile


@pytest.mark.asyncio
async def test_builds_vopoint_records_with_stage_sterile(
    fake_store: AsyncMock,
    fake_embed: AsyncMock,
    config_enabled: BookRagConfig,
) -> None:
    raw = [_chunk(source_id="src-1")]
    sterile = [_chunk(source_id="src-1", content="Sterilized.", embedding_text="Sterilized.")]

    with patch(
        "vo_rag.pipelines.book_rag.steps.sterilize.sterilize_chunks",
        new_callable=AsyncMock,
        return_value=sterile,
    ):
        await do_sterilize(
            raw_chunks=raw,
            llm_client=_fake_llm_client(),
            config=config_enabled,
            store=fake_store,
            embed=fake_embed,
        )

    fake_store.upsert_vectors.assert_awaited_once()
    call = fake_store.upsert_vectors.await_args
    assert call is not None
    collection, points = call.args
    assert collection == "vo_rag_books_sterile_v1"
    assert len(points) == 1
    point = points[0]
    assert isinstance(point, VoPoint)
    assert point.source_id == "src-1"
    assert point.stage == "sterile"
    assert point.chunk_index == 0
    assert len(point.dense) == 1024
    assert point.sparse == {1: 0.5, 2: 0.5}
    assert point.payload.get("chunk_type") == "text"


@pytest.mark.asyncio
async def test_sterile_payload_denormalizes_book_identity_fields(
    fake_store: AsyncMock,
    fake_embed: AsyncMock,
    config_enabled: BookRagConfig,
) -> None:
    metadata = {
        "source_book": "PG Book (PG Author)",
        "author": "PG Author",
        "license_class": "licensed_internal",
        "canonical_source_id": "vo-rag-corpus-sha256:abc",
        "domain_tags": ["vehicle-dynamics"],
        "project_id": "motorsport",
    }
    raw = [_chunk(source_id="src-1", metadata=metadata)]
    sterile = [
        _chunk(
            source_id="src-1",
            content="Sterilized.",
            embedding_text="Sterilized.",
            metadata=metadata,
        )
    ]

    with patch(
        "vo_rag.pipelines.book_rag.steps.sterilize.sterilize_chunks",
        new_callable=AsyncMock,
        return_value=sterile,
    ):
        await do_sterilize(
            raw_chunks=raw,
            llm_client=_fake_llm_client(),
            config=config_enabled,
            store=fake_store,
            embed=fake_embed,
        )

    fake_store.upsert_vectors.assert_awaited_once()
    _collection, points = fake_store.upsert_vectors.await_args.args
    payload = points[0].payload
    assert payload["source_book"] == "PG Book (PG Author)"
    assert payload["author"] == "PG Author"
    assert payload["license_class"] == "licensed_internal"
    assert payload["canonical_source_id"] == "vo-rag-corpus-sha256:abc"
    assert payload["domain_tags"] == ["vehicle-dynamics"]
    assert payload["project_id"] == "motorsport"


@pytest.mark.asyncio
async def test_upserts_multiple_chunks(
    fake_store: AsyncMock,
    fake_embed: AsyncMock,
    config_enabled: BookRagConfig,
) -> None:
    raw = [
        _chunk(source_id="src-1", chunk_id="c1"),
        _chunk(source_id="src-1", chunk_id="c2"),
    ]
    sterile = [
        _chunk(source_id="src-1", chunk_id="c1-s", content="One.", embedding_text="One."),
        _chunk(source_id="src-1", chunk_id="c2-s", content="Two.", embedding_text="Two."),
    ]

    with patch(
        "vo_rag.pipelines.book_rag.steps.sterilize.sterilize_chunks",
        new_callable=AsyncMock,
        return_value=sterile,
    ):
        await do_sterilize(
            raw_chunks=raw,
            llm_client=_fake_llm_client(),
            config=config_enabled,
            store=fake_store,
            embed=fake_embed,
        )

    fake_store.upsert_vectors.assert_awaited_once()
    _collection, points = fake_store.upsert_vectors.await_args.args
    assert len(points) == 2
    assert points[0].chunk_index == 0
    assert points[1].chunk_index == 1


# ---------------------------------------------------------------------------
# Invariant: every LLM call uses complete_json_protected with allowed_domains=[]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_json_protected_called_with_allowed_domains_empty_list(
    fake_store: AsyncMock,
    fake_embed: AsyncMock,
    config_enabled: BookRagConfig,
) -> None:
    raw = [_chunk()]
    calls: list[dict[str, Any]] = []

    async def _recording_fake(
        client: LLMClient,
        *,
        schema: type[Any],
        untrusted_content: str,
        pipeline: str,
        allowed_domains: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, LLMResponse, ValidationResult, bool]:
        calls.append({"allowed_domains": allowed_domains, "pipeline": pipeline})
        parsed = schema.model_validate(
            {"sterile_content": "ok", "facts": [], "equations": [], "concepts": []}
        )
        resp = LLMResponse(
            text='{"sterile_content": "ok"}',
            model="fake-model",
            input_tokens=10,
            output_tokens=5,
            provider="fake",
        )
        return parsed, resp, ValidationResult(ok=True, signals=[]), False

    with patch("vo_rag.core.compliance.sterilizer.complete_json_protected", _recording_fake):
        await do_sterilize(
            raw_chunks=raw,
            llm_client=_fake_llm_client(),
            config=config_enabled,
            store=fake_store,
            embed=fake_embed,
        )

    assert len(calls) == 1
    assert calls[0]["allowed_domains"] == []
    assert calls[0]["pipeline"] == "book_rag"


# ---------------------------------------------------------------------------
# Visual sidecar path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_visual_yield_above_floor_no_warning(
    fake_store: AsyncMock,
    fake_embed: AsyncMock,
    config_enabled: BookRagConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    raw = [
        _chunk(chunk_type="text"),
        _chunk(chunk_type="figure_composite", chunk_id="fc1"),
        _chunk(chunk_type="figure_composite", chunk_id="fc2"),
    ]
    sterile = [
        _chunk(chunk_type="text", content="Sterilized text."),
        _chunk(chunk_type="figure_composite", chunk_id="fc1-s", content="Sterilized fig 1."),
        _chunk(chunk_type="figure_composite", chunk_id="fc2-s", content="Sterilized fig 2."),
    ]

    with patch(
        "vo_rag.pipelines.book_rag.steps.sterilize.sterilize_chunks",
        new_callable=AsyncMock,
        return_value=sterile,
    ):
        await do_sterilize(
            raw_chunks=raw,
            llm_client=_fake_llm_client(),
            config=config_enabled,
            store=fake_store,
            embed=fake_embed,
        )

    assert "visual_yield_below_floor" not in caplog.text


@pytest.mark.asyncio
async def test_visual_yield_below_floor_logs_warning(
    fake_store: AsyncMock,
    fake_embed: AsyncMock,
    config_enabled: BookRagConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    raw = [
        _chunk(chunk_type="text"),
        _chunk(chunk_type="figure_composite", chunk_id="fc1"),
        _chunk(chunk_type="figure_composite", chunk_id="fc2"),
    ]
    # Only 1 of 2 figure_composite chunks survives → yield = 0.5 < 0.70
    sterile = [
        _chunk(chunk_type="text", content="Sterilized text."),
        _chunk(chunk_type="figure_composite", chunk_id="fc1-s", content="Sterilized fig 1."),
    ]

    with patch(
        "vo_rag.pipelines.book_rag.steps.sterilize.sterilize_chunks",
        new_callable=AsyncMock,
        return_value=sterile,
    ):
        await do_sterilize(
            raw_chunks=raw,
            llm_client=_fake_llm_client(),
            config=config_enabled,
            store=fake_store,
            embed=fake_embed,
        )

    assert "visual_yield_below_floor" in caplog.text


@pytest.mark.asyncio
async def test_no_visual_chunks_skips_yield_check(
    fake_store: AsyncMock,
    fake_embed: AsyncMock,
    config_enabled: BookRagConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    raw = [_chunk(chunk_type="text")]
    sterile = [_chunk(chunk_type="text", content="Sterilized.")]

    with patch(
        "vo_rag.pipelines.book_rag.steps.sterilize.sterilize_chunks",
        new_callable=AsyncMock,
        return_value=sterile,
    ):
        await do_sterilize(
            raw_chunks=raw,
            llm_client=_fake_llm_client(),
            config=config_enabled,
            store=fake_store,
            embed=fake_embed,
        )

    assert "visual_yield_below_floor" not in caplog.text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_chunks_returns_empty(
    fake_store: AsyncMock,
    fake_embed: AsyncMock,
    config_enabled: BookRagConfig,
) -> None:
    result = await do_sterilize(
        raw_chunks=[],
        llm_client=_fake_llm_client(),
        config=config_enabled,
        store=fake_store,
        embed=fake_embed,
    )
    assert result == []
    fake_store.upsert_vectors.assert_not_called()
    fake_embed.assert_not_called()

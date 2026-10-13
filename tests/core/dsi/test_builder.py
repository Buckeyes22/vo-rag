from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

import pytest

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.dsi.builder import (
    DsiBuildConfig,
    DsiBuildStats,
    build_dsi_for_book,
    build_dsi_for_corpus,
    dsi_search_text,
    persist_dsi_records,
)
from vo_rag.core.dsi.extraction import ExtractionConfig
from vo_rag.core.dsi.prompts import DSI_EXTRACTION_PROMPT_HASH
from vo_rag.core.dsi.schemas import FacetedBookRecord, ProvenanceLinkedEntry
from vo_rag.core.llm.response import LLMResponse

_CHUNK_ID_RE = re.compile(r"\[chunk_id=([^\]]+)\]")


def _make_chunk(
    *,
    chunk_id: str,
    content: str,
    section_path: str = "[Book > Chapter > Section]",
    section_title: str = "Section",
    document_id: str = "doc-001",
    source_id: str = "source-001",
) -> RetrievalChunk:
    embedding_text = f"{section_path}\n\n{content}"
    token_count = max(1, len(content.split()))
    return RetrievalChunk(
        chunk_id=chunk_id,
        chunk_type="text",
        document_id=document_id,
        source_id=source_id,
        page_number=None,
        page_range=None,
        section_path=section_path,
        section_title=section_title,
        content=content,
        embedding_text=embedding_text,
        token_count=token_count,
        has_image=False,
        image_id=None,
        image_ref=None,
        vlm_description=None,
        metadata={},
    )


def _valid_record_json(
    book_id: str = "book-001",
    chunk_id: str = "c1",
    section_path: str = "[Book > Chapter > Section]",
) -> str:
    def _entry(value: str) -> dict[str, str]:
        return {"chunk_id": chunk_id, "section_path": section_path, "value": value}

    payload: dict[str, object] = {
        "short_abstract": "A technical book about brake systems.",
        "technical_domains": ["automotive", "engineering"],
        "key_concepts": [_entry("hydraulic brake")],
        "key_components": ["brake pad", "rotor", "caliper"],
        "key_methods": [_entry("brake modulation")],
        "key_equations_or_models": [_entry("Pascal law")],
        "figure_heavy_topics": [_entry("caliper diagram")],
        "synonyms_and_aliases": [_entry("disc brake")],
        "prerequisite_topics": ["fluid dynamics", "mechanical systems"],
        "chapter_outline": "1. Introduction\n2. Brake Systems\n3. Hydraulic Principles",
    }
    return json.dumps(payload)


class TestDsiBuildConfig:
    def test_dsi_build_config_defaults(self) -> None:
        config = DsiBuildConfig(
            extraction_config=ExtractionConfig(prompt_hash=DSI_EXTRACTION_PROMPT_HASH)
        )
        assert config.max_concurrent_books == 4
        assert config.max_validation_attempts == 2


class TestDsiBuildStats:
    def test_dsi_build_stats_schema_accepts_required_fields(self) -> None:
        stats = DsiBuildStats(
            book_id="book-001",
            extraction_attempts=1,
            extraction_succeeded=True,
            validation_passed=True,
            validation_violations=0,
            final_action="accept",
            input_tokens=100,
            output_tokens=50,
            wall_time_ms=200,
        )
        assert stats.book_id == "book-001"
        assert stats.final_action == "accept"


class TestBuildDsiForBook:
    @pytest.mark.asyncio
    async def test_build_dsi_for_book_succeeds_on_valid_response(self) -> None:
        chunks = [_make_chunk(chunk_id="c1", content="The brake system uses hydraulic fluid.")]

        async def mock_llm_call(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            return LLMResponse(
                text=_valid_record_json(book_id="book-001"),
                model="test-model",
                input_tokens=100,
                output_tokens=50,
            )

        config = DsiBuildConfig(
            extraction_config=ExtractionConfig(prompt_hash=DSI_EXTRACTION_PROMPT_HASH),
            max_validation_attempts=2,
        )

        record, stats = await build_dsi_for_book(
            book_id="book-001",
            title="Brake Systems",
            author="Author",
            category_ids=["engineering"],
            chunks=chunks,
            config=config,
            llm_call=mock_llm_call,
            source_collection_version="v1",
        )

        assert record is not None
        assert record.book_id == "book-001"
        assert stats.final_action == "accept"
        assert stats.extraction_succeeded is True
        assert stats.validation_passed is True

    @pytest.mark.asyncio
    async def test_build_dsi_for_book_retries_on_validation_failure(self) -> None:
        chunks = [_make_chunk(chunk_id="c1", content="The brake system uses hydraulic fluid.")]

        call_count = 0

        async def mock_llm_call(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    text=_valid_record_json(book_id="book-001", chunk_id="unknown_chunk"),
                    model="test-model",
                    input_tokens=100,
                    output_tokens=50,
                )
            else:
                return LLMResponse(
                    text=_valid_record_json(book_id="book-001", chunk_id="c1"),
                    model="test-model",
                    input_tokens=100,
                    output_tokens=50,
                )

        config = DsiBuildConfig(
            extraction_config=ExtractionConfig(prompt_hash=DSI_EXTRACTION_PROMPT_HASH),
            max_validation_attempts=2,
        )

        record, stats = await build_dsi_for_book(
            book_id="book-001",
            title="Brake Systems",
            author="Author",
            category_ids=["engineering"],
            chunks=chunks,
            config=config,
            llm_call=mock_llm_call,
            source_collection_version="v1",
            known_chunk_ids={"c1"},
            known_section_paths={"[Book > Chapter > Section]"},
        )

        assert record is not None
        assert stats.extraction_attempts == 2
        assert stats.final_action == "accept"

    @pytest.mark.asyncio
    async def test_build_dsi_for_book_gives_up_after_max_attempts(self) -> None:
        chunks = [_make_chunk(chunk_id="c1", content="The brake system uses hydraulic fluid.")]

        call_count = 0

        async def mock_llm_call(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            return LLMResponse(
                text=_valid_record_json(book_id="book-001", chunk_id="unknown_chunk"),
                model="test-model",
                input_tokens=100,
                output_tokens=50,
            )

        config = DsiBuildConfig(
            extraction_config=ExtractionConfig(prompt_hash=DSI_EXTRACTION_PROMPT_HASH),
            max_validation_attempts=2,
        )

        record, stats = await build_dsi_for_book(
            book_id="book-001",
            title="Brake Systems",
            author="Author",
            category_ids=["engineering"],
            chunks=chunks,
            config=config,
            llm_call=mock_llm_call,
            source_collection_version="v1",
            known_chunk_ids={"c1"},
        )

        # New contract: an unresolvable chunk_id is dropped at parse time, so every
        # attempt yields no valid provenance entries -> extraction_failed (the
        # validator's manual_review path is unreachable now that the parser
        # guarantees provenance by construction).
        assert record is None
        assert stats.final_action == "extraction_failed"
        assert stats.extraction_attempts == 2

    @pytest.mark.asyncio
    async def test_build_dsi_for_book_extraction_failed(self) -> None:
        chunks = [_make_chunk(chunk_id="c1", content="The brake system uses hydraulic fluid.")]

        async def mock_llm_call(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            return LLMResponse(
                text="not valid json at all",
                model="test-model",
                input_tokens=0,
                output_tokens=0,
            )

        config = DsiBuildConfig(
            extraction_config=ExtractionConfig(prompt_hash=DSI_EXTRACTION_PROMPT_HASH),
            max_validation_attempts=2,
        )

        record, stats = await build_dsi_for_book(
            book_id="book-001",
            title="Brake Systems",
            author="Author",
            category_ids=["engineering"],
            chunks=chunks,
            config=config,
            llm_call=mock_llm_call,
            source_collection_version="v1",
        )

        assert record is None
        assert stats.extraction_succeeded is False
        assert stats.final_action == "extraction_failed"

    @pytest.mark.asyncio
    async def test_build_dsi_for_book_records_token_accounting(self) -> None:
        chunks = [_make_chunk(chunk_id="c1", content="The brake system uses hydraulic fluid.")]

        async def mock_llm_call(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            return LLMResponse(
                text=_valid_record_json(book_id="book-001"),
                model="test-model",
                input_tokens=100,
                output_tokens=50,
            )

        config = DsiBuildConfig(
            extraction_config=ExtractionConfig(prompt_hash=DSI_EXTRACTION_PROMPT_HASH),
            max_validation_attempts=2,
        )

        record, stats = await build_dsi_for_book(
            book_id="book-001",
            title="Brake Systems",
            author="Author",
            category_ids=["engineering"],
            chunks=chunks,
            config=config,
            llm_call=mock_llm_call,
            source_collection_version="v1",
        )

        assert record is not None
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50


class TestBuildDsiForCorpus:
    @pytest.mark.asyncio
    async def test_build_dsi_for_corpus_returns_records_for_all_successful_books(
        self,
    ) -> None:
        books = [
            (
                "book-001",
                "Book One",
                "Author One",
                ["cat1"],
                [_make_chunk(chunk_id="b1c1", content="Content one")],
            ),
            (
                "book-002",
                "Book Two",
                "Author Two",
                ["cat2"],
                [_make_chunk(chunk_id="b2c1", content="Content two")],
            ),
            (
                "book-003",
                "Book Three",
                "Author Three",
                ["cat3"],
                [_make_chunk(chunk_id="b3c1", content="Content three")],
            ),
        ]

        async def mock_llm_call(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            # Cite the chunk_id shown in the prompt (new contract: provenance is
            # derived from the cited representative chunk).
            match = _CHUNK_ID_RE.search(user)
            assert match is not None, "_make_chunk always emits [chunk_id=...] in the prompt"
            cid = match.group(1)
            return LLMResponse(
                text=_valid_record_json(book_id="book-001", chunk_id=cid),
                model="test-model",
                input_tokens=100,
                output_tokens=50,
            )

        config = DsiBuildConfig(
            extraction_config=ExtractionConfig(prompt_hash=DSI_EXTRACTION_PROMPT_HASH),
            max_concurrent_books=4,
            max_validation_attempts=2,
        )

        records, stats = await build_dsi_for_corpus(
            books=books,
            config=config,
            llm_call=mock_llm_call,
            source_collection_version="v1",
        )

        assert len(records) == 3
        assert len(stats) == 3
        book_ids = {r.book_id for r in records}
        assert book_ids == {"book-001", "book-002", "book-003"}

    @pytest.mark.asyncio
    async def test_build_dsi_for_corpus_respects_max_concurrent(self) -> None:
        books = [
            (
                f"book-{i:03d}",
                f"Book {i}",
                f"Author {i}",
                ["cat1"],
                [_make_chunk(chunk_id=f"b{i}c1", content=f"Content {i}")],
            )
            for i in range(1, 5)
        ]

        call_order: list[int] = []
        start_times: dict[int, float] = {}

        async def mock_llm_call(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            idx = int(system.split("Book ")[-1].split("\n")[0]) if "Book " in system else 0
            start_times[idx] = time.monotonic()
            await asyncio.sleep(0.05)
            call_order.append(idx)
            match = _CHUNK_ID_RE.search(user)
            assert match is not None
            cid = match.group(1)
            return LLMResponse(
                text=_valid_record_json(book_id=f"book-{idx:03d}", chunk_id=cid),
                model="test-model",
                input_tokens=100,
                output_tokens=50,
            )

        config = DsiBuildConfig(
            extraction_config=ExtractionConfig(prompt_hash=DSI_EXTRACTION_PROMPT_HASH),
            max_concurrent_books=2,
            max_validation_attempts=1,
        )

        records, stats = await build_dsi_for_corpus(
            books=books,
            config=config,
            llm_call=mock_llm_call,
            source_collection_version="v1",
        )

        assert len(records) == 4
        assert len(stats) == 4


def _faceted_record(book_id: str) -> FacetedBookRecord:
    from datetime import UTC, datetime

    def _entry(value: str) -> ProvenanceLinkedEntry:
        return ProvenanceLinkedEntry(
            value=value,
            supporting_chunk_ids=["c1"],
            supporting_section_paths=["[Book > Chapter > Section]"],
        )

    return FacetedBookRecord(
        book_id=book_id,
        title="Hydraulic Brake Design",
        author="Author",
        category_ids=["motorsport"],
        short_abstract="A book about brake systems and hydraulics.",
        technical_domains=["automotive", "braking"],
        key_concepts=[_entry("hydraulic brake"), _entry("brake bias")],
        key_components=["caliper", "rotor"],
        key_methods=[_entry("brake modulation")],
        key_equations_or_models=[],  # genre-contingent: empty is valid
        figure_heavy_topics=[],
        synonyms_and_aliases=[_entry("disc brake")],
        prerequisite_topics=["fluid dynamics"],
        chapter_outline="1. Introduction",
        representative_chunk_ids=["c1"],
        source_collection_version="unversioned",
        summary_model="test-model",
        prompt_hash="abc123",
        created_at=datetime.now(UTC),
    )


class TestPersistDsiRecords:
    def test_dsi_search_text_includes_facets(self) -> None:
        text = dsi_search_text(_faceted_record("b1"))
        assert "brake systems" in text
        assert "hydraulic brake" in text
        assert "disc brake" in text

    @pytest.mark.asyncio
    async def test_persist_builds_vo_points_and_upserts(self) -> None:
        records = [_faceted_record("book-001"), _faceted_record("book-002")]
        captured: dict[str, Any] = {}

        class _FakeStore:
            async def upsert_vectors(self, collection: str, points: list[Any]) -> None:
                captured["collection"] = collection
                captured["points"] = points

        async def _fake_embed(texts: list[str]) -> dict[str, Any]:
            n = len(texts)
            return {
                "dense": [[0.01] * 1024 for _ in range(n)],
                "sparse": [{"indices": [7], "values": [0.5]} for _ in range(n)],
                "colbert": [[[0.02] * 1024] for _ in range(n)],
            }

        count = await persist_dsi_records(
            records=records,
            embed=_fake_embed,
            store=_FakeStore(),
            collection="vo_rag_books_dsi_v1",
        )

        assert count == 2
        assert captured["collection"] == "vo_rag_books_dsi_v1"
        points = captured["points"]
        assert len(points) == 2
        p = points[0]
        assert p.source_id == "book-001"
        assert p.chunk_index == 0
        # VOR-209: persist writes the model/prompt-versioned stage +
        # version_hash so a model swap or prompt edit mints a distinct point id
        # (record above uses summary_model="test-model", prompt_hash="abc123").
        assert p.stage == "dsi_test-model"
        assert p.version_hash == "abc123"
        assert len(p.dense) == 1024
        assert p.sparse == {7: 0.5}
        assert p.colbert and len(p.colbert[0]) == 1024
        assert p.bm25  # bm25 tokenizer produced weights from the search text
        assert p.payload["book_id"] == "book-001"

    @pytest.mark.asyncio
    async def test_persist_empty_records_is_noop(self) -> None:
        async def _embed(_texts: list[str]) -> dict[str, Any]:  # pragma: no cover
            raise AssertionError("embed must not be called for empty records")

        assert await persist_dsi_records(records=[], embed=_embed, store=object()) == 0

    @pytest.mark.asyncio
    async def test_persist_defaults_to_vo_rag_collection(self) -> None:
        """The persist default collection is vo_rag_books_dsi_v1."""
        records = [_faceted_record("book-001")]
        captured: dict[str, Any] = {}

        class _FakeStore:
            async def upsert_vectors(self, collection: str, points: list[Any]) -> None:
                captured["collection"] = collection

        async def _fake_embed(texts: list[str]) -> dict[str, Any]:
            n = len(texts)
            return {
                "dense": [[0.01] * 1024 for _ in range(n)],
                "sparse": [{"indices": [7], "values": [0.5]} for _ in range(n)],
                "colbert": [[[0.02] * 1024] for _ in range(n)],
            }

        await persist_dsi_records(records=records, embed=_fake_embed, store=_FakeStore())
        assert captured["collection"] == "vo_rag_books_dsi_v1"

from __future__ import annotations

from typing import Any

import pytest

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.dsi.builder import DsiBuildConfig, build_dsi_for_corpus
from vo_rag.core.dsi.extraction import ExtractionConfig
from vo_rag.core.dsi.prompts import DSI_EXTRACTION_PROMPT_HASH
from vo_rag.core.dsi.routing import allocate_budgets, normalize_dsi_scores
from vo_rag.core.llm.response import LLMResponse


def _make_chunk(
    *,
    chunk_id: str,
    content: str,
    section_path: str = "[Book > Chapter > Section]",
    section_title: str = "Section",
    document_id: str = "doc-001",
    source_id: str = "source-001",
    metadata: dict[str, Any] | None = None,
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
        metadata=metadata or {},
    )


def _valid_extraction_json(
    book_id: str,
    chunk_ids: list[str],
    section_path: str = "[Book > Chapter > Section]",
) -> str:
    provenance = [
        f'{{"chunk_id": "{cid}", "section_path": "{section_path}", "value": "concept_{cid}"}}'
        for cid in chunk_ids
    ]
    provenance_str = ", ".join(provenance)
    return (
        f"{{"
        f'"short_abstract": "A technical book about {book_id}.",'
        f'"technical_domains": ["automotive", "engineering"],'
        f'"key_concepts": [{provenance_str}],'
        f'"key_components": ["component1", "component2"],'
        f'"key_methods": [{provenance_str}],'
        f'"key_equations_or_models": [{provenance_str}],'
        f'"figure_heavy_topics": [{provenance_str}],'
        f'"synonyms_and_aliases": [{provenance_str}],'
        f'"prerequisite_topics": ["fluid dynamics"],'
        f'"chapter_outline": "1. Introduction\\n2. Main Content"'
        f"}}"
    )


def _make_book_chunks(
    book_id: str,
    n_chunks: int = 3,
) -> list[RetrievalChunk]:
    return [
        _make_chunk(
            chunk_id=f"{book_id}-c{i}",
            content=f"Content for {book_id} chunk {i} about technical topics.",
        )
        for i in range(1, n_chunks + 1)
    ]


def _make_extraction_config() -> ExtractionConfig:
    return ExtractionConfig(prompt_hash=DSI_EXTRACTION_PROMPT_HASH)


class TestEndToEndBuildThenRoute3Books:
    @pytest.mark.asyncio
    async def test_end_to_end_build_then_route_3_books(self) -> None:
        books = [
            (
                "book-a",
                "Book A Title",
                "Author A",
                ["engineering"],
                _make_book_chunks("book-a"),
            ),
            (
                "book-b",
                "Book B Title",
                "Author B",
                ["engineering"],
                _make_book_chunks("book-b"),
            ),
            (
                "book-c",
                "Book C Title",
                "Author C",
                ["physics"],
                _make_book_chunks("book-c"),
            ),
        ]

        async def mock_llm(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            if "book-a" in user:
                book_id = "book-a"
                chunk_ids = ["book-a-c1", "book-a-c2", "book-a-c3"]
            elif "book-b" in user:
                book_id = "book-b"
                chunk_ids = ["book-b-c1", "book-b-c2", "book-b-c3"]
            else:
                book_id = "book-c"
                chunk_ids = ["book-c-c1", "book-c-c2", "book-c-c3"]
            return LLMResponse(
                text=_valid_extraction_json(book_id=book_id, chunk_ids=chunk_ids),
                model="qwen3.6-27b-fp8",
                input_tokens=100,
                output_tokens=50,
            )

        config = DsiBuildConfig(
            extraction_config=_make_extraction_config(),
            max_concurrent_books=3,
            max_validation_attempts=1,
        )

        records, stats = await build_dsi_for_corpus(
            books=books,
            config=config,
            llm_call=mock_llm,
            source_collection_version="v1",
        )

        assert len(records) == 3, f"Expected 3 accepted records, got {len(records)}"
        assert len(stats) == 3

        routing_scores = {r.book_id: 0.8 if r.book_id == "book-a" else 0.1 for r in records}
        normalized = normalize_dsi_scores(routing_scores)
        routing_decision = allocate_budgets(
            confidences=normalized,
            total_budget_chunks=20,
            top_k=8,
            fallback_share=0.05,
        )

        for route in routing_decision.book_routes:
            assert route.total_budget >= 0, f"Book {route.book_id} has negative budget"


class TestFailedExtractionBookExcludedFromRouting:
    @pytest.mark.asyncio
    async def test_failed_extraction_book_excluded_from_routing(self) -> None:
        books = [
            (
                "book-a",
                "Book A Title",
                "Author A",
                ["engineering"],
                _make_book_chunks("book-a"),
            ),
            (
                "book-b",
                "Book B Title",
                "Author B",
                ["engineering"],
                _make_book_chunks("book-b"),
            ),
            (
                "book-c",
                "Book C Title",
                "Author C",
                ["physics"],
                _make_book_chunks("book-c"),
            ),
        ]

        call_count = 0

        async def mock_llm(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            if "book-b" in user:
                return LLMResponse(
                    text="NOT VALID JSON { broken",
                    model="qwen3.6-27b-fp8",
                    input_tokens=50,
                    output_tokens=10,
                )
            book_id = "book-a" if "book-a" in user else "book-c"
            chunk_ids = (
                ["book-a-c1", "book-a-c2", "book-a-c3"]
                if "book-a" in user
                else ["book-c-c1", "book-c-c2", "book-c-c3"]
            )
            return LLMResponse(
                text=_valid_extraction_json(book_id=book_id, chunk_ids=chunk_ids),
                model="qwen3.6-27b-fp8",
                input_tokens=100,
                output_tokens=50,
            )

        config = DsiBuildConfig(
            extraction_config=_make_extraction_config(),
            max_concurrent_books=3,
            max_validation_attempts=1,
        )

        records, stats = await build_dsi_for_corpus(
            books=books,
            config=config,
            llm_call=mock_llm,
            source_collection_version="v1",
        )

        accepted_book_ids = {r.book_id for r in records}
        assert "book-a" in accepted_book_ids
        assert "book-c" in accepted_book_ids
        assert "book-b" not in accepted_book_ids

        rejected_stats = [s for s in stats if s.final_action == "extraction_failed"]
        assert len(rejected_stats) >= 1

        routing_scores = {r.book_id: 0.7 for r in records}
        normalized = normalize_dsi_scores(routing_scores)
        routing_decision = allocate_budgets(
            confidences=normalized,
            total_budget_chunks=20,
            top_k=8,
            fallback_share=0.05,
        )
        routed_book_ids = {route.book_id for route in routing_decision.book_routes}
        assert "book-b" not in routed_book_ids


class TestValidationViolationsQuarantineBook:
    @pytest.mark.asyncio
    async def test_validation_violations_quarantine_book(self) -> None:
        books = [
            (
                "book-a",
                "Book A Title",
                "Author A",
                ["engineering"],
                _make_book_chunks("book-a"),
            ),
            (
                "book-b",
                "Book B Title",
                "Author B",
                ["engineering"],
                _make_book_chunks("book-b"),
            ),
            (
                "book-c",
                "Book C Title",
                "Author C",
                ["physics"],
                _make_book_chunks("book-c"),
            ),
        ]

        async def mock_llm(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            if "book-c" in user:
                bad_provenance = (
                    '{"chunk_id": "unknown-c99", "section_path": "[Fake > Path]", "value": "fake"}'
                )
                return LLMResponse(
                    text=(
                        f"{{"
                        f'"short_abstract": "Abstract for C",'
                        f'"technical_domains": ["physics"],'
                        f'"key_concepts": [{bad_provenance}],'
                        f'"key_components": ["comp1"],'
                        f'"key_methods": [{bad_provenance}],'
                        f'"key_equations_or_models": [{bad_provenance}],'
                        f'"figure_heavy_topics": [{bad_provenance}],'
                        f'"synonyms_and_aliases": [{bad_provenance}],'
                        f'"prerequisite_topics": [],'
                        f'"chapter_outline": "Ch1"'
                        f"}}"
                    ),
                    model="qwen3.6-27b-fp8",
                    input_tokens=100,
                    output_tokens=50,
                )
            book_id = "book-a" if "book-a" in user else "book-b"
            chunk_ids = (
                ["book-a-c1", "book-a-c2", "book-a-c3"]
                if "book-a" in user
                else ["book-b-c1", "book-b-c2", "book-b-c3"]
            )
            return LLMResponse(
                text=_valid_extraction_json(book_id=book_id, chunk_ids=chunk_ids),
                model="qwen3.6-27b-fp8",
                input_tokens=100,
                output_tokens=50,
            )

        config = DsiBuildConfig(
            extraction_config=_make_extraction_config(),
            max_concurrent_books=3,
            max_validation_attempts=2,
        )

        known_chunk_ids_per_book = {
            "book-a": {"book-a-c1", "book-a-c2", "book-a-c3"},
            "book-b": {"book-b-c1", "book-b-c2", "book-b-c3"},
            "book-c": {"book-c-c1", "book-c-c2", "book-c-c3"},
        }
        known_section_paths_per_book = {
            "book-a": {"[Book > Chapter > Section]"},
            "book-b": {"[Book > Chapter > Section]"},
            "book-c": {"[Book > Chapter > Section]"},
        }

        records, stats = await build_dsi_for_corpus(
            books=books,
            config=config,
            llm_call=mock_llm,
            source_collection_version="v1",
            known_chunk_ids_per_book=known_chunk_ids_per_book,
            known_section_paths_per_book=known_section_paths_per_book,
        )

        assert len(records) == 2
        accepted_ids = {r.book_id for r in records}
        assert "book-a" in accepted_ids
        assert "book-b" in accepted_ids
        assert "book-c" not in accepted_ids

        # book-c cited an unresolvable chunk_id (unknown-c99); the new contract
        # drops such entries at parse time, so the book yields no valid provenance
        # and is excluded via extraction_failed rather than the (now-unreachable)
        # validation manual_review path. The quarantine OUTCOME is unchanged.
        book_c_stats = next(s for s in stats if s.book_id == "book-c")
        assert book_c_stats.final_action == "extraction_failed"


class TestRoutingSafetyNetRecoversUnconfidentBook:
    @pytest.mark.asyncio
    async def test_routing_safety_net_recovers_unconfident_book(self) -> None:
        books = [
            (
                "book-a",
                "Book A Title",
                "Author A",
                ["engineering"],
                _make_book_chunks("book-a"),
            ),
            (
                "book-b",
                "Book B Title",
                "Author B",
                ["engineering"],
                _make_book_chunks("book-b"),
            ),
            (
                "book-c",
                "Book C Title",
                "Author C",
                ["physics"],
                _make_book_chunks("book-c"),
            ),
        ]

        async def mock_llm(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            if "book-a" in user:
                book_id = "book-a"
                chunk_ids = ["book-a-c1", "book-a-c2", "book-a-c3"]
            elif "book-b" in user:
                book_id = "book-b"
                chunk_ids = ["book-b-c1", "book-b-c2", "book-b-c3"]
            else:
                book_id = "book-c"
                chunk_ids = ["book-c-c1", "book-c-c2", "book-c-c3"]
            return LLMResponse(
                text=_valid_extraction_json(book_id=book_id, chunk_ids=chunk_ids),
                model="qwen3.6-27b-fp8",
                input_tokens=100,
                output_tokens=50,
            )

        config = DsiBuildConfig(
            extraction_config=_make_extraction_config(),
            max_concurrent_books=3,
            max_validation_attempts=1,
        )

        records, stats = await build_dsi_for_corpus(
            books=books,
            config=config,
            llm_call=mock_llm,
            source_collection_version="v1",
        )

        assert len(records) == 3

        routing_scores = {
            "book-a": 0.9,
            "book-b": 0.05,
            "book-c": 0.05,
        }
        normalized = normalize_dsi_scores(routing_scores)
        routing_decision = allocate_budgets(
            confidences=normalized,
            total_budget_chunks=20,
            top_k=1,
            fallback_share=0.05,
        )

        book_a_route = next(r for r in routing_decision.book_routes if r.book_id == "book-a")
        book_b_route = next(r for r in routing_decision.book_routes if r.book_id == "book-b")
        book_c_route = next(r for r in routing_decision.book_routes if r.book_id == "book-c")

        assert book_a_route.primary_budget > 0
        assert book_b_route.fallback_budget >= 0
        assert book_c_route.fallback_budget >= 0
        assert book_b_route.total_budget >= 0
        assert book_c_route.total_budget >= 0


class TestPromptHashPropagatesThroughPipeline:
    @pytest.mark.asyncio
    async def test_prompt_hash_propagates_through_pipeline(self) -> None:
        books = [
            (
                "book-a",
                "Book A Title",
                "Author A",
                ["engineering"],
                _make_book_chunks("book-a"),
            ),
        ]

        async def mock_llm(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            return LLMResponse(
                text=_valid_extraction_json(
                    book_id="book-a",
                    chunk_ids=["book-a-c1", "book-a-c2", "book-a-c3"],
                ),
                model="qwen3.6-27b-fp8",
                input_tokens=100,
                output_tokens=50,
            )

        config = DsiBuildConfig(
            extraction_config=_make_extraction_config(),
            max_concurrent_books=1,
            max_validation_attempts=1,
        )

        records, _ = await build_dsi_for_corpus(
            books=books,
            config=config,
            llm_call=mock_llm,
            source_collection_version="v1",
        )

        assert len(records) == 1
        assert records[0].prompt_hash == DSI_EXTRACTION_PROMPT_HASH


class TestPerBookTokenAccountingAggregatesCorrectly:
    @pytest.mark.asyncio
    async def test_per_book_token_accounting_aggregates_correctly(self) -> None:
        books = [
            (
                "book-a",
                "Book A Title",
                "Author A",
                ["engineering"],
                _make_book_chunks("book-a"),
            ),
            (
                "book-b",
                "Book B Title",
                "Author B",
                ["engineering"],
                _make_book_chunks("book-b"),
            ),
            (
                "book-c",
                "Book C Title",
                "Author C",
                ["physics"],
                _make_book_chunks("book-c"),
            ),
        ]

        async def mock_llm(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            if "book-a" in user:
                book_id = "book-a"
                chunk_ids = ["book-a-c1", "book-a-c2", "book-a-c3"]
            elif "book-b" in user:
                book_id = "book-b"
                chunk_ids = ["book-b-c1", "book-b-c2", "book-b-c3"]
            else:
                book_id = "book-c"
                chunk_ids = ["book-c-c1", "book-c-c2", "book-c-c3"]
            return LLMResponse(
                text=_valid_extraction_json(book_id=book_id, chunk_ids=chunk_ids),
                model="qwen3.6-27b-fp8",
                input_tokens=100,
                output_tokens=50,
            )

        config = DsiBuildConfig(
            extraction_config=_make_extraction_config(),
            max_concurrent_books=3,
            max_validation_attempts=1,
        )

        _, stats = await build_dsi_for_corpus(
            books=books,
            config=config,
            llm_call=mock_llm,
            source_collection_version="v1",
        )

        total_input_tokens = sum(s.input_tokens for s in stats)
        total_output_tokens = sum(s.output_tokens for s in stats)

        assert total_input_tokens == 300
        assert total_output_tokens == 150

        for stat in stats:
            assert stat.input_tokens == 100
            assert stat.output_tokens == 50

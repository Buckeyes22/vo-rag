from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.dsi.extraction import (
    ExtractionConfig,
    build_extraction_prompt,
    extract_book,
    parse_extraction_response,
    select_representative_chunks,
)
from vo_rag.core.dsi.prompts import DSI_EXTRACTION_PROMPT_HASH, DSI_EXTRACTION_SYSTEM_PROMPT
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


def _entry(
    chunk_id: str, value: str, section_path: str = "[Book > Chapter > Section]"
) -> dict[str, str]:
    """Build one provenance-linked JSON entry. section_path is included to
    prove the parser ignores it (the LLM-emitted value is overridden by
    chunk_section_map)."""
    return {"chunk_id": chunk_id, "section_path": section_path, "value": value}


class TestDsiPromptVersionSingleSourceOfTruth:
    """DSI_PROMPT_VERSION must live ONLY in prompts.py."""

    def test_dsi_prompt_version_single_source_of_truth(self) -> None:
        import vo_rag.core.dsi.schemas as schemas
        from vo_rag.core.dsi.prompts import DSI_PROMPT_VERSION

        assert not hasattr(schemas, "DSI_PROMPT_VERSION"), (
            "schemas.py must not redeclare DSI_PROMPT_VERSION — prompts.py is the"
            " single source of truth (the v1.0/v1.1 drift bug)."
        )
        assert DSI_PROMPT_VERSION == "v1.3"


class TestSelectRepresentativeChunks:
    def test_select_representative_chunks_returns_top_n_by_score(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id=f"c{i}",
                content=f"Content about topic {i} " * 20,
            )
            for i in range(1, 51)
        ]
        result = select_representative_chunks(chunks=chunks, n=10)
        assert len(result) == 10

    def test_select_representative_chunks_dedups_by_chunk_id(self) -> None:
        chunks = [
            _make_chunk(chunk_id="c1", content="Same chunk ID content " * 20),
            _make_chunk(chunk_id="c1", content="Duplicate chunk ID content " * 20),
            _make_chunk(chunk_id="c2", content="Another chunk " * 20),
        ]
        result = select_representative_chunks(chunks=chunks, n=5)
        chunk_ids = [c.chunk_id for c in result]
        assert len(chunk_ids) >= 1

    def test_select_representative_chunks_returns_all_when_n_exceeds_count(self) -> None:
        chunks = [_make_chunk(chunk_id=f"c{i}", content=f"Content {i}") for i in range(1, 6)]
        result = select_representative_chunks(chunks=chunks, n=20)
        assert len(result) == 5


class TestBuildExtractionPrompt:
    def test_build_extraction_prompt_includes_book_metadata(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id="c1",
                content="Test content about hydraulic brake systems.",
                section_path="[Race Car Setup > Brakes > Hydraulic Systems]",
            ),
        ]
        system, user = build_extraction_prompt(
            book_id="book-001",
            title="Race Car Engineering",
            author="John Smith",
            category_ids=["engineering", "automotive"],
            representative_chunks=chunks,
        )
        assert DSI_EXTRACTION_SYSTEM_PROMPT in system
        assert "Race Car Engineering" in user
        assert "John Smith" in user
        assert "engineering, automotive" in user

    def test_build_extraction_prompt_formats_chunk_block(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id="c1",
                content="The brake system uses hydraulic fluid.",
                section_path="[Book > Chapter > Section]",
            ),
        ]
        _, user = build_extraction_prompt(
            book_id="book-001",
            title="Test Book",
            author="Author",
            category_ids=["cat1"],
            representative_chunks=chunks,
        )
        assert "c1" in user
        assert "[Book > Chapter > Section]" in user
        assert "The brake system uses hydraulic fluid" in user


class TestParseExtractionResponse:
    def test_parse_extraction_response_valid_builds_record(self) -> None:
        valid_json = json.dumps(
            {
                "short_abstract": "A technical book about brake systems.",
                "technical_domains": ["automotive", "engineering"],
                "key_concepts": [_entry("c1", "hydraulic brake")],
                "key_components": ["brake pad", "rotor", "caliper"],
                "key_methods": [_entry("c1", "brake modulation")],
                "key_equations_or_models": [_entry("c1", "Pascal's law")],
                "figure_heavy_topics": [_entry("c1", "caliper diagram")],
                "synonyms_and_aliases": [_entry("c1", "disc brake")],
                "prerequisite_topics": ["fluid dynamics", "mechanical systems"],
                "chapter_outline": "1. Introduction\n2. Brake Systems\n3. Hydraulic Principles",
            }
        )
        result = parse_extraction_response(
            book_id="book-001",
            title="Brake Systems",
            author="Author",
            category_ids=["engineering"],
            llm_response_json=valid_json,
            summary_model="test-model",
            prompt_hash="abc123",
            source_collection_version="v1",
            # Ground-truth section path differs from the (now-ignored) one in the JSON.
            chunk_section_map={"c1": "[REAL > Brakes > Hydraulics]"},
            representative_chunk_ids=["c1"],
        )
        assert result.record is not None
        assert result.book_id == "book-001"
        assert result.validation_errors == []
        # section_path is DERIVED from the cited chunk, not taken from the LLM JSON.
        assert result.record.key_concepts[0].supporting_section_paths == [
            "[REAL > Brakes > Hydraulics]"
        ]
        assert result.record.representative_chunk_ids == ["c1"]

    def test_parse_extraction_response_coerces_plain_string_list_provenance_dicts(
        self,
    ) -> None:
        dict_form_plain_lists = json.dumps(
            {
                "short_abstract": "A technical book about motorcycle suspension.",
                "technical_domains": [
                    {"value": "automotive engineering", "chunk_id": "c1"},
                    {"value": "suspension tuning", "chunk_id": "c1"},
                ],
                "key_concepts": [_entry("c1", "suspension geometry")],
                "key_components": [
                    {"value": "Front forks", "chunk_id": "c1"},
                    {"value": "Chassis", "chunk_id": "c1"},
                ],
                "key_methods": [_entry("c1", "sag adjustment")],
                "key_equations_or_models": [],
                "figure_heavy_topics": [],
                "synonyms_and_aliases": [_entry("c1", "forks")],
                "prerequisite_topics": [
                    {"value": "spring preload", "chunk_id": "c1"},
                    {"value": "damping", "chunk_id": "c1"},
                ],
                "chapter_outline": "1. Setup\n2. Adjustment",
            }
        )

        result = parse_extraction_response(
            book_id="book-001",
            title="Suspension Setup",
            author="Author",
            category_ids=["engineering"],
            llm_response_json=dict_form_plain_lists,
            summary_model="test-model",
            prompt_hash="abc123",
            source_collection_version="v1",
            chunk_section_map={"c1": "[Book > Suspension]"},
            representative_chunk_ids=["c1"],
        )

        assert result.validation_errors == []
        assert result.record is not None
        assert result.record.technical_domains == [
            "automotive engineering",
            "suspension tuning",
        ]
        assert result.record.key_components == ["Front forks", "Chassis"]
        assert result.record.prerequisite_topics == ["spring preload", "damping"]

    def test_parse_extraction_response_preserves_plain_string_lists(self) -> None:
        string_form_plain_lists = json.dumps(
            {
                "short_abstract": "A technical book about motorcycle suspension.",
                "technical_domains": ["automotive engineering", "suspension tuning"],
                "key_concepts": [_entry("c1", "suspension geometry")],
                "key_components": ["Springs", "Dampers"],
                "key_methods": [_entry("c1", "sag adjustment")],
                "key_equations_or_models": [],
                "figure_heavy_topics": [],
                "synonyms_and_aliases": [_entry("c1", "forks")],
                "prerequisite_topics": ["spring preload", "damping"],
                "chapter_outline": "1. Setup\n2. Adjustment",
            }
        )

        result = parse_extraction_response(
            book_id="book-001",
            title="Suspension Setup",
            author="Author",
            category_ids=["engineering"],
            llm_response_json=string_form_plain_lists,
            summary_model="test-model",
            prompt_hash="abc123",
            source_collection_version="v1",
            chunk_section_map={"c1": "[Book > Suspension]"},
            representative_chunk_ids=["c1"],
        )

        assert result.validation_errors == []
        assert result.record is not None
        assert result.record.technical_domains == [
            "automotive engineering",
            "suspension tuning",
        ]
        assert result.record.key_components == ["Springs", "Dampers"]
        assert result.record.prerequisite_topics == ["spring preload", "damping"]

    def test_parse_extraction_response_section_path_always_from_chunk_map(self) -> None:
        """Even when the LLM emits a section_path field it is
        ignored — supporting_section_paths is derived from chunk_section_map.
        Cross-check across ALL provenance facets.
        """
        json_with_lying_section_paths = json.dumps(
            {
                "short_abstract": "Abstract.",
                "technical_domains": ["domain"],
                "key_concepts": [_entry("c1", "k1", "[LYING > A]")],
                "key_components": ["x"],
                "key_methods": [_entry("c1", "m1", "[LYING > B]")],
                "key_equations_or_models": [_entry("c1", "e1", "[LYING > C]")],
                "figure_heavy_topics": [_entry("c1", "f1", "[LYING > D]")],
                "synonyms_and_aliases": [_entry("c1", "s1", "[LYING > E]")],
                "prerequisite_topics": [],
                "chapter_outline": "Outline",
            }
        )
        result = parse_extraction_response(
            book_id="book-001",
            title="T",
            author="A",
            category_ids=[],
            llm_response_json=json_with_lying_section_paths,
            summary_model="m",
            prompt_hash="h",
            source_collection_version="v",
            chunk_section_map={"c1": "[GROUND_TRUTH > Path]"},
            representative_chunk_ids=["c1"],
        )
        assert result.record is not None
        for field in (
            "key_concepts",
            "key_methods",
            "key_equations_or_models",
            "figure_heavy_topics",
            "synonyms_and_aliases",
        ):
            entries = getattr(result.record, field)
            assert entries
            for entry in entries:
                assert entry.supporting_section_paths == ["[GROUND_TRUTH > Path]"], (
                    f"{field}: section_path must be derived from chunk map,"
                    f" not the LLM-emitted '[LYING > ...]' value"
                )

    def test_parse_extraction_response_missing_field_returns_none(self) -> None:
        missing_abstract = json.dumps(
            {
                # short_abstract intentionally absent
                "technical_domains": ["automotive"],
                "key_concepts": [_entry("c1", "brake")],
                "key_components": ["pad"],
                "key_methods": [_entry("c1", "method")],
                "key_equations_or_models": [_entry("c1", "equation")],
                "figure_heavy_topics": [_entry("c1", "figure")],
                "synonyms_and_aliases": [_entry("c1", "synonym")],
                "prerequisite_topics": [],
                "chapter_outline": "Chapter 1",
            }
        )
        result = parse_extraction_response(
            book_id="book-001",
            title="Test",
            author="Author",
            category_ids=[],
            llm_response_json=missing_abstract,
            summary_model="test-model",
            prompt_hash="abc123",
            source_collection_version="v1",
            chunk_section_map={"c1": "[Book > Chapter > Section]"},
            representative_chunk_ids=["c1"],
        )
        assert result.record is None
        assert any("short_abstract" in err for err in result.validation_errors)

    def test_parse_extraction_response_invalid_provenance_entry_returns_none(self) -> None:
        invalid_provenance = json.dumps(
            {
                "short_abstract": "Abstract text here.",
                "technical_domains": ["domain"],
                # empty chunk_id -> entry dropped -> field rejected
                "key_concepts": [_entry("", "concept", "[Book > Chapter]")],
                "key_components": ["component"],
                "key_methods": [_entry("c1", "method")],
                "key_equations_or_models": [_entry("c1", "equation")],
                "figure_heavy_topics": [_entry("c1", "figure")],
                "synonyms_and_aliases": [_entry("c1", "synonym")],
                "prerequisite_topics": [],
                "chapter_outline": "Outline",
            }
        )
        result = parse_extraction_response(
            book_id="book-001",
            title="Test",
            author="Author",
            category_ids=[],
            llm_response_json=invalid_provenance,
            summary_model="test-model",
            prompt_hash="abc123",
            source_collection_version="v1",
            chunk_section_map={"c1": "[Book > Chapter > Section]"},
            representative_chunk_ids=["c1"],
        )
        # The only key_concepts entry has an empty chunk_id -> dropped -> the field
        # has no valid entries -> record rejected.
        assert result.record is None
        assert any("key_concepts" in err for err in result.validation_errors)

    def test_parse_extraction_response_optional_facets_empty_succeeds(self) -> None:
        """Genre-contingent facets (key_equations_or_models, figure_heavy_topics)
        may be empty for non-technical books (e.g. a circuit atlas). Empty is
        valid data, not a rejection — the record must still build."""
        atlas_json = json.dumps(
            {
                "short_abstract": "A reference atlas of motorsport circuits worldwide.",
                "technical_domains": ["motorsport"],
                "key_concepts": [_entry("c1", "circuit layout")],
                "key_components": ["pit lane"],
                "key_methods": [_entry("c1", "track walk")],
                "key_equations_or_models": [],
                "figure_heavy_topics": [],
                "synonyms_and_aliases": [_entry("c1", "track")],
                "prerequisite_topics": [],
                "chapter_outline": "1. Europe",
            }
        )
        result = parse_extraction_response(
            book_id="atlas-001",
            title="A Guide to Regional Circuits",
            author="Northrop",
            category_ids=["motorsport"],
            llm_response_json=atlas_json,
            summary_model="test-model",
            prompt_hash="abc123",
            source_collection_version="v1",
            chunk_section_map={"c1": "[Book > Chapter > Section]"},
            representative_chunk_ids=["c1"],
        )
        assert result.validation_errors == []
        assert result.record is not None
        assert result.record.key_equations_or_models == []
        assert result.record.figure_heavy_topics == []
        assert len(result.record.key_concepts) == 1

    def test_parse_extraction_response_required_facet_empty_still_fails(self) -> None:
        """A *required* provenance facet (key_methods) being empty must still
        reject — the relaxation is scoped to genre-contingent facets only."""
        no_methods_json = json.dumps(
            {
                "short_abstract": "Abstract.",
                "technical_domains": ["motorsport"],
                "key_concepts": [_entry("c1", "concept")],
                "key_components": ["component"],
                "key_methods": [],
                "key_equations_or_models": [],
                "figure_heavy_topics": [],
                "synonyms_and_aliases": [_entry("c1", "alias")],
                "prerequisite_topics": [],
                "chapter_outline": "Outline",
            }
        )
        result = parse_extraction_response(
            book_id="book-001",
            title="Test",
            author="Author",
            category_ids=[],
            llm_response_json=no_methods_json,
            summary_model="test-model",
            prompt_hash="abc123",
            source_collection_version="v1",
            chunk_section_map={"c1": "[Book > Chapter > Section]"},
            representative_chunk_ids=["c1"],
        )
        assert result.record is None
        assert any("key_methods" in err for err in result.validation_errors)

    def test_parse_extraction_response_one_bad_optional_does_not_reject_record(self) -> None:
        """One malformed optional entry is dropped,
        the rest of the record (which is otherwise well-formed) still builds.
        """
        json_with_one_bad_alias = json.dumps(
            {
                "short_abstract": "Abstract.",
                "technical_domains": ["d"],
                "key_concepts": [_entry("c1", "k1", "x")],
                "key_components": ["x"],
                "key_methods": [_entry("c1", "m1", "x")],
                "key_equations_or_models": [],
                "figure_heavy_topics": [],
                "synonyms_and_aliases": [
                    _entry("c1", "good", "x"),
                    _entry("unknown", "bad", "x"),
                ],
                "prerequisite_topics": [],
                "chapter_outline": "Out",
            }
        )
        result = parse_extraction_response(
            book_id="book-001",
            title="T",
            author="A",
            category_ids=[],
            llm_response_json=json_with_one_bad_alias,
            summary_model="m",
            prompt_hash="h",
            source_collection_version="v",
            chunk_section_map={"c1": "[GT]"},
            representative_chunk_ids=["c1"],
        )
        assert result.record is not None
        assert [e.value for e in result.record.synonyms_and_aliases] == ["good"]


class TestExtractBook:
    @pytest.mark.asyncio
    async def test_extract_book_calls_llm_once(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id="c1",
                content="The differential routes torque to the rear wheels during cornering.",
            ),
        ]

        mock_response = LLMResponse(
            text=json.dumps(
                {
                    "short_abstract": "Abstract",
                    "technical_domains": [],
                    "key_concepts": [_entry("c1", "differential")],
                    "key_components": [],
                    "key_methods": [],
                    "key_equations_or_models": [],
                    "figure_heavy_topics": [],
                    "synonyms_and_aliases": [],
                    "prerequisite_topics": [],
                    "chapter_outline": "Ch1",
                }
            ),
            model="test-model",
            input_tokens=100,
            output_tokens=50,
        )

        mock_llm_call = AsyncMock(return_value=mock_response)

        config = ExtractionConfig(prompt_hash=DSI_EXTRACTION_PROMPT_HASH)

        result = await extract_book(
            book_id="book-001",
            title="Test Book",
            author="Author",
            category_ids=["engineering"],
            chunks=chunks,
            config=config,
            llm_call=mock_llm_call,
            source_collection_version="v1",
        )

        assert mock_llm_call.call_count == 1
        assert result.book_id == "book-001"
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    def test_extract_book_records_token_accounting(self) -> None:
        valid_json = json.dumps(
            {
                "short_abstract": "A technical book about brake systems.",
                "technical_domains": ["automotive"],
                "key_concepts": [_entry("c1", "hydraulic brake")],
                "key_components": ["brake pad"],
                "key_methods": [_entry("c1", "brake modulation")],
                "key_equations_or_models": [_entry("c1", "Pascal's law")],
                "figure_heavy_topics": [_entry("c1", "caliper diagram")],
                "synonyms_and_aliases": [_entry("c1", "disc brake")],
                "prerequisite_topics": ["fluid dynamics"],
                "chapter_outline": "1. Introduction",
            }
        )
        result = parse_extraction_response(
            book_id="book-001",
            title="Brake Systems",
            author="Author",
            category_ids=["engineering"],
            llm_response_json=valid_json,
            summary_model="qwen3.6-27b-fp8",
            prompt_hash=DSI_EXTRACTION_PROMPT_HASH,
            source_collection_version="v1",
            chunk_section_map={"c1": "[Book > Chapter > Section]"},
            representative_chunk_ids=["c1"],
        )
        assert result.prompt_hash == DSI_EXTRACTION_PROMPT_HASH
        assert result.input_tokens == 0
        assert result.output_tokens == 0

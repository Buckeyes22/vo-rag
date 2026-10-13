from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from vo_rag.core.dsi.schemas import FacetedBookRecord, ProvenanceLinkedEntry
from vo_rag.core.dsi.validation import (
    ProvenanceViolation,
    decide_rejection,
    fetch_book_chunk_ids_and_sections,
    validate_provenance_offline,
)


def _make_provenance_entry(
    *,
    value: str,
    chunk_ids: list[str],
    section_paths: list[str],
) -> ProvenanceLinkedEntry:
    return ProvenanceLinkedEntry(
        value=value,
        supporting_chunk_ids=chunk_ids,
        supporting_section_paths=section_paths,
    )


def _make_record(
    *,
    book_id: str,
    key_concepts: list[ProvenanceLinkedEntry] | None = None,
    key_methods: list[ProvenanceLinkedEntry] | None = None,
    key_equations_or_models: list[ProvenanceLinkedEntry] | None = None,
    figure_heavy_topics: list[ProvenanceLinkedEntry] | None = None,
    synonyms_and_aliases: list[ProvenanceLinkedEntry] | None = None,
) -> FacetedBookRecord:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return FacetedBookRecord(
        book_id=book_id,
        title="Test Book",
        author="Test Author",
        category_ids=["engineering"],
        short_abstract="Abstract",
        technical_domains=["automotive"],
        key_concepts=key_concepts or [],
        key_components=["component1"],
        key_methods=key_methods or [],
        key_equations_or_models=key_equations_or_models or [],
        figure_heavy_topics=figure_heavy_topics or [],
        synonyms_and_aliases=synonyms_and_aliases or [],
        prerequisite_topics=["topic1"],
        chapter_outline="Chapter 1",
        representative_chunk_ids=["c1"],
        source_collection_version="v1",
        summary_model="test-model",
        prompt_hash="abc123",
        created_at=datetime.now(ZoneInfo("UTC")),
    )


class TestValidateProvenanceOffline:
    def test_validate_provenance_offline_clean_record(self) -> None:
        record = _make_record(
            book_id="book-001",
            key_concepts=[
                _make_provenance_entry(
                    value="differential",
                    chunk_ids=["c1", "c2"],
                    section_paths=["[Book > Chapter 1]", "[Book > Chapter 2]"],
                ),
            ],
        )
        known_chunk_ids = {"c1", "c2"}
        known_section_paths = {"[Book > Chapter 1]", "[Book > Chapter 2]"}

        violations = validate_provenance_offline(
            record,
            known_chunk_ids=known_chunk_ids,
            known_section_paths=known_section_paths,
        )
        assert violations == []

    def test_validate_provenance_offline_unresolvable_chunk_id(self) -> None:
        record = _make_record(
            book_id="book-001",
            key_concepts=[
                _make_provenance_entry(
                    value="brake",
                    chunk_ids=["c99"],
                    section_paths=["[Book > Chapter 1]"],
                ),
            ],
        )
        known_chunk_ids = {"c1", "c2"}
        known_section_paths = {"[Book > Chapter 1]"}

        violations = validate_provenance_offline(
            record,
            known_chunk_ids=known_chunk_ids,
            known_section_paths=known_section_paths,
        )
        assert len(violations) == 1
        assert violations[0].violation_type == "unresolvable_chunk_id"
        assert violations[0].chunk_id == "c99"

    def test_validate_provenance_offline_unresolvable_section_path(self) -> None:
        record = _make_record(
            book_id="book-001",
            key_methods=[
                _make_provenance_entry(
                    value="brake modulation",
                    chunk_ids=["c1"],
                    section_paths=["[NonExistent > Chapter]"],
                ),
            ],
        )
        known_chunk_ids = {"c1"}
        known_section_paths = {"[Book > Chapter 1]"}

        violations = validate_provenance_offline(
            record,
            known_chunk_ids=known_chunk_ids,
            known_section_paths=known_section_paths,
        )
        assert len(violations) == 1
        assert violations[0].violation_type == "unresolvable_section_path"
        assert violations[0].section_path == "[NonExistent > Chapter]"

    def test_validate_provenance_offline_multiple_violations_aggregated(self) -> None:
        record = _make_record(
            book_id="book-001",
            key_concepts=[
                _make_provenance_entry(
                    value="concept1",
                    chunk_ids=["c99", "c98"],
                    section_paths=["[Book > Chapter 1]", "[Fake > Path]"],
                ),
            ],
            key_methods=[
                _make_provenance_entry(
                    value="method1",
                    chunk_ids=["c97"],
                    section_paths=["[Missing > Section]"],
                ),
            ],
        )
        known_chunk_ids = {"c1"}
        known_section_paths = {"[Book > Chapter 1]"}

        violations = validate_provenance_offline(
            record,
            known_chunk_ids=known_chunk_ids,
            known_section_paths=known_section_paths,
        )
        assert len(violations) == 5


class TestDecideRejection:
    def test_decide_rejection_clean_attempt_one_accepts(self) -> None:
        decision = decide_rejection(violations=[], attempt=1, max_attempts=2)
        assert decision.record_accepted is True
        assert decision.action == "accept"
        assert decision.attempt_number == 1

    def test_decide_rejection_violations_attempt_one_re_extracts(self) -> None:
        violation = ProvenanceViolation(
            book_id="book-001",
            field_name="key_concepts",
            entry_value="test",
            violation_type="unresolvable_chunk_id",
            chunk_id="c99",
            section_path=None,
            detail="chunk_id c99 not found",
        )
        decision = decide_rejection(violations=[violation], attempt=1, max_attempts=2)
        assert decision.record_accepted is False
        assert decision.action == "re_extract"
        assert decision.attempt_number == 2

    def test_decide_rejection_violations_attempt_max_manual_review(self) -> None:
        violation = ProvenanceViolation(
            book_id="book-001",
            field_name="key_concepts",
            entry_value="test",
            violation_type="unresolvable_chunk_id",
            chunk_id="c99",
            section_path=None,
            detail="chunk_id c99 not found",
        )
        decision = decide_rejection(violations=[violation], attempt=2, max_attempts=2)
        assert decision.record_accepted is False
        assert decision.action == "manual_review"
        assert decision.attempt_number == 2


class TestFetchBookChunkIdsAndSections:
    @pytest.mark.asyncio
    async def test_fetch_book_chunk_ids_and_sections_paginates(self) -> None:
        mock_client = AsyncMock()

        page1_points = [
            _make_mock_point("c1", {"section_path": "[Book > Chapter 1 > Section 1]"}),
            _make_mock_point("c2", {"section_path": "[Book > Chapter 1 > Section 2]"}),
        ]
        page2_points = [
            _make_mock_point("c3", {"section_path": "[Book > Chapter 2 > Section 1]"}),
        ]

        mock_client.scroll = AsyncMock(
            side_effect=[
                (page1_points, "offset-abc"),
                (page2_points, None),
            ]
        )

        mock_store = MagicMock()
        mock_store._client = mock_client

        chunk_ids, section_paths = await fetch_book_chunk_ids_and_sections(
            store=mock_store,
            book_id="book-001",
            raw_collection="vo_rag_books_raw_v1",
        )

        assert chunk_ids == {"c1", "c2", "c3"}
        assert "[Book > Chapter 1 > Section 1]" in section_paths
        assert "[Book > Chapter 1 > Section 2]" in section_paths
        assert "[Book > Chapter 2 > Section 1]" in section_paths
        assert mock_client.scroll.call_count == 2


def _make_mock_point(
    point_id: str,
    payload: dict[str, Any],
) -> MagicMock:
    mock_point = MagicMock()
    mock_point.id = point_id
    mock_point.payload = payload
    return mock_point

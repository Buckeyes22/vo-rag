"""Tests for vo_rag.core.summary.extract_support.

``select_evidence_snippets`` returns RAW
copied spans from ``chunk.content`` (not summaries, not paraphrases) and
that ``EvidenceSpan.section_path`` is copied from the source
``RetrievalChunk.section_path`` — never inferred by the LLM.
"""

from __future__ import annotations

from typing import Any

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.summary.extract_support import (
    build_extract_support_prompt,
    select_evidence_snippets,
)
from vo_rag.core.summary.prompts import SYSTEM_PROMPTS
from vo_rag.core.summary.schemas import EvidenceSpan, NodeLevel


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


class TestSelectEvidenceSnippetsReturnType:
    def test_returns_list_evidence_span(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id="c1",
                content=(
                    "The brake system uses hydraulic fluid to transfer force "
                    "from the pedal to the calipers."
                ),
            ),
        ]
        result = select_evidence_snippets(chunks, topic_or_query="brake system", k=5)
        assert isinstance(result, list)
        assert all(isinstance(span, EvidenceSpan) for span in result)

    def test_span_fields_match_schema(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id="c1",
                content="The differential routes torque to the rear wheels during cornering.",
            ),
        ]
        result = select_evidence_snippets(chunks, topic_or_query="differential", k=5)
        assert len(result) == 1
        span = result[0]
        assert hasattr(span, "chunk_id")
        assert hasattr(span, "section_path")
        assert hasattr(span, "text")
        assert hasattr(span, "score")
        assert hasattr(span, "selection_method")


class TestSelectEvidenceSnippetsChunkIdInvariant:
    def test_chunk_id_from_input_chunks(self) -> None:
        chunk_ids = ["c1", "c2", "c3"]
        chunks = [
            _make_chunk(
                chunk_id=cid, content=f"Sentence number {i} about tire pressure dynamics." * 3
            )
            for i, cid in enumerate(chunk_ids)
        ]
        result = select_evidence_snippets(chunks, topic_or_query="tire pressure", k=5)
        for span in result:
            assert span.chunk_id in chunk_ids


class TestSelectEvidenceSnippetsSectionPathInvariant:
    def test_section_path_matches_source_chunk(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id="c1",
                content="The diffuser creates low pressure under the car to generate downforce.",
                section_path="[Book One > Chapter 3 > Diffuser Design]",
            ),
        ]
        result = select_evidence_snippets(chunks, topic_or_query="diffuser downforce", k=5)
        assert len(result) == 1
        assert result[0].section_path == "[Book One > Chapter 3 > Diffuser Design]"


class TestSelectEvidenceSnippetsTopK:
    def test_k5_respects_limit(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id=f"c{i}",
                content=f"The engine produces {i * 100} horsepower during peak rev range. " * 5,
            )
            for i in range(1, 101)
        ]
        result = select_evidence_snippets(chunks, topic_or_query="engine horsepower", k=5)
        assert len(result) <= 5

    def test_k0_returns_empty(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id="c1",
                content="A dampers role is to control unwanted suspension movement.",
            ),
        ]
        result = select_evidence_snippets(chunks, topic_or_query="damper", k=0)
        assert result == []

    def test_empty_chunks_returns_empty(self) -> None:
        result = select_evidence_snippets([], topic_or_query="any query", k=5)
        assert result == []


class TestSelectEvidenceSnippetsBm25Determinism:
    def test_same_inputs_produce_identical_ordering(self) -> None:
        content = (
            "The aerodynamic wing generates downforce at high speeds. "
            "Front wings direct airflow around the tires. "
            "Rear wings provide stability through balanced angle of attack. "
            "Undertray diffusers accelerate airflow to create low pressure."
        )
        chunks = [_make_chunk(chunk_id=f"c{i}", content=content) for i in range(1, 6)]
        query = "aerodynamic wing downforce"

        result_a = select_evidence_snippets(chunks, topic_or_query=query, k=5)
        result_b = select_evidence_snippets(chunks, topic_or_query=query, k=5)

        assert [span.chunk_id for span in result_a] == [span.chunk_id for span in result_b]


class TestSelectEvidenceSnippetsMethodLiteral:
    def test_method_bm25_sets_selection_method(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id="c1",
                content=(
                    "The turbocharger forces additional air into the "
                    "combustion chamber for more power."
                ),
            ),
        ]
        result = select_evidence_snippets(chunks, topic_or_query="turbocharger", k=5, method="bm25")
        for span in result:
            assert span.selection_method == "bm25"

    def test_method_centroid_sim_sets_selection_method(self) -> None:
        dense_vector = [0.1] * 1024
        chunks = [
            _make_chunk(
                chunk_id="c1",
                content="Weight transfer occurs during braking and acceleration phases of driving.",
                metadata={"dense": dense_vector},
            ),
        ]
        result = select_evidence_snippets(
            chunks, topic_or_query="weight transfer", k=5, method="centroid_sim"
        )
        for span in result:
            assert span.selection_method == "centroid_sim"


class TestSelectEvidenceSnippetsHybridFallback:
    def test_hybrid_falls_back_to_bm25_when_no_centroid_scores(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id="c1",
                content="The anti-roll bar reduces body roll during cornering maneuvers on track.",
            ),
        ]
        result = select_evidence_snippets(
            chunks, topic_or_query="anti-roll bar", k=5, method="hybrid"
        )
        for span in result:
            assert span.selection_method == "bm25"

    def test_hybrid_uses_bm25_fallback_with_empty_metadata(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id="c1",
                content=(
                    "Brake pad material affects stopping distance and rotor "
                    "wear rates significantly."
                ),
                metadata={},
            ),
        ]
        result = select_evidence_snippets(
            chunks, topic_or_query="brake pad material", k=5, method="hybrid"
        )
        for span in result:
            assert span.selection_method == "bm25"


class TestBuildExtractSupportPromptReturnType:
    def test_returns_tuple_of_two_strings(self) -> None:
        evidence = [
            EvidenceSpan(
                chunk_id="c1",
                section_path="[Book > Chapter > Section]",
                text="Sample extracted text from the chunk.",
                score=0.95,
                selection_method="bm25",
            ),
        ]
        system, user = build_extract_support_prompt(
            level=NodeLevel.SECTION,
            primary_input="Primary input text for the summary.",
            supporting_evidence=evidence,
            display_path="[Book > Chapter > Section]",
        )
        assert isinstance(system, str)
        assert isinstance(user, str)
        assert len(system) > 0
        assert len(user) > 0


class TestBuildExtractSupportPromptSystemPrompt:
    def test_section_system_prompt_exact_match(self) -> None:
        evidence = [
            EvidenceSpan(
                chunk_id="c1",
                section_path="[Book > Chapter > Section]",
                text="Sample text.",
                score=0.5,
                selection_method="bm25",
            ),
        ]
        system, _ = build_extract_support_prompt(
            level=NodeLevel.SECTION,
            primary_input="Primary input.",
            supporting_evidence=evidence,
            display_path="[Book > Chapter > Section]",
        )
        assert system == SYSTEM_PROMPTS[NodeLevel.SECTION]


class TestBuildExtractSupportPromptUserTemplate:
    def test_primary_input_in_user_prompt(self) -> None:
        evidence = [
            EvidenceSpan(
                chunk_id="c1",
                section_path="[Book > Chapter > Section]",
                text="Extracted snippet text.",
                score=0.8,
                selection_method="bm25",
            ),
        ]
        primary = "This is the primary section input text about cornering forces."
        _, user = build_extract_support_prompt(
            level=NodeLevel.SECTION,
            primary_input=primary,
            supporting_evidence=evidence,
            display_path="[Book > Chapter > Section]",
        )
        assert primary in user

    def test_supporting_evidence_json_in_user_prompt(self) -> None:
        evidence = [
            EvidenceSpan(
                chunk_id="c1",
                section_path="[Book > Chapter > Section]",
                text="Supporting text one.",
                score=0.9,
                selection_method="bm25",
            ),
            EvidenceSpan(
                chunk_id="c2",
                section_path="[Book > Chapter > Section]",
                text="Supporting text two.",
                score=0.7,
                selection_method="bm25",
            ),
        ]
        _, user = build_extract_support_prompt(
            level=NodeLevel.SECTION,
            primary_input="Primary input text.",
            supporting_evidence=evidence,
            display_path="[Book > Chapter > Section]",
        )
        assert "c1" in user
        assert "c2" in user
        assert "Supporting text one" in user
        assert "Supporting text two" in user

    def test_section_breadcrumb_in_user_prompt(self) -> None:
        evidence: list[EvidenceSpan] = []
        breadcrumb = "[Race Car Setup > Suspension Geometry > Camber Angle]"
        _, user = build_extract_support_prompt(
            level=NodeLevel.SECTION,
            primary_input="Primary input.",
            supporting_evidence=evidence,
            display_path=breadcrumb,
        )
        assert breadcrumb in user


# ---------------------------------------------------------------------------
# Evidence snippets are raw copies
# ---------------------------------------------------------------------------


def test_select_evidence_snippets_returns_raw_copied_snippets_not_summaries() -> None:
    """Evidence must be a verbatim copy of chunk.content,
    never a paraphrase or summary. Every emitted span's text must appear
    as a contiguous substring of its source chunk's content."""

    source_text = (
        "Brake cooling ducts feed air to the front rotors to manage heat "
        "buildup during long stints. Duct geometry is tuned per circuit."
    )
    chunks = [
        _make_chunk(chunk_id="raw-1", content=source_text),
    ]
    result = select_evidence_snippets(chunks, topic_or_query="brake cooling ducts", k=3)

    assert len(result) >= 1
    for span in result:
        assert span.chunk_id == "raw-1"
        assert span.text in source_text, (
            "EvidenceSpan.text must be a contiguous substring of the source "
            "chunk content — not a paraphrase or summary."
        )


def test_evidence_span_section_path_is_copied_from_source_chunk() -> None:
    """EvidenceSpan.section_path is a copy of the source
    RetrievalChunk.section_path. The LLM never gets to pick this value —
    it is taken straight off the chunk. Distinct section paths per chunk
    must round-trip into distinct evidence spans."""

    chunks = [
        _make_chunk(
            chunk_id="c-a",
            content="The front wing endplate design shapes vortex structures.",
            section_path="[Book > Chapter 1 > Section A]",
        ),
        _make_chunk(
            chunk_id="c-b",
            content="The rear wing main plane angle balances drag and downforce.",
            section_path="[Book > Chapter 2 > Section B]",
        ),
    ]
    result = select_evidence_snippets(chunks, topic_or_query="wing design", k=4)

    by_chunk = {span.chunk_id: span.section_path for span in result}
    if "c-a" in by_chunk:
        assert by_chunk["c-a"] == "[Book > Chapter 1 > Section A]"
    if "c-b" in by_chunk:
        assert by_chunk["c-b"] == "[Book > Chapter 2 > Section B]"

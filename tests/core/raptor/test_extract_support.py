from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.raptor.extract_support import (
    assemble_recursive_evidence,
    build_level_extract_support_prompt,
    select_raw_evidence_for_cluster,
    verify_raw_evidence_invariant,
)
from vo_rag.core.raptor.schemas import ClusterLevel
from vo_rag.core.summary.schemas import EvidenceSpan


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


class TestSelectRawEvidenceForCluster:
    def test_select_raw_evidence_for_cluster_returns_top_m_by_score(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id=f"c{i}",
                content=f"The brake system uses hydraulic fluid to transfer force for tire {i}.",
            )
            for i in range(10)
        ]
        result = select_raw_evidence_for_cluster(
            cluster_chunks=chunks,
            topic_or_query="brake system hydraulic fluid",
            k=3,
        )
        assert len(result) <= 3
        for span in result:
            assert "chunk_id" in span
            assert "section_path" in span
            assert "text" in span
            assert "score" in span
            assert "selection_method" in span

    def test_select_raw_evidence_for_cluster_only_returns_raw_chunks(self) -> None:
        chunk_ids = {"c1", "c2", "c3", "c4", "c5"}
        chunks = [
            _make_chunk(
                chunk_id=cid,
                content=f"Content for {cid} about tire pressure dynamics.",
            )
            for cid in sorted(chunk_ids)
        ]
        result = select_raw_evidence_for_cluster(
            cluster_chunks=chunks,
            topic_or_query="tire pressure dynamics",
            k=5,
        )
        result_ids = {span["chunk_id"] for span in result}
        assert result_ids.issubset(chunk_ids)

    def test_select_raw_evidence_for_cluster_rejects_non_raw_evidence(self) -> None:
        """select_raw_evidence_for_cluster rejects non-raw evidence
        via verify_raw_evidence_invariant."""
        chunks = [
            _make_chunk(chunk_id="c1", content="Raw content one."),
            _make_chunk(chunk_id="c2", content="Raw content two."),
        ]
        bad_evidence = [
            EvidenceSpan(
                chunk_id="summary-node-abc",
                section_path="path",
                text="summary text",
                score=0.9,
                selection_method="bm25",
            )
        ]
        with (
            patch(
                "vo_rag.core.raptor.extract_support.select_evidence_snippets",
                return_value=bad_evidence,
            ),
            pytest.raises(
                ValueError,
                match="selected RAPTOR evidence must reference only raw cluster chunk IDs",
            ),
        ):
            select_raw_evidence_for_cluster(
                cluster_chunks=chunks,
                topic_or_query="test",
                k=5,
            )


class TestAssembleRecursiveEvidence:
    def test_assemble_recursive_evidence_at_level_0_returns_raw_chunks(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id=f"c{i}",
                content=f"Raw content chunk {i} about differential gear.",
            )
            for i in range(5)
        ]
        _, evidence = assemble_recursive_evidence(
            level=ClusterLevel.LEVEL_0,
            child_primary_text="Primary input text for level 0.",
            all_underlying_raw_chunks=chunks,
            k_evidence=3,
        )
        raw_ids = {c.chunk_id for c in chunks}
        for span in evidence:
            assert span["chunk_id"] in raw_ids

    def test_assemble_recursive_evidence_at_level_1_returns_raw_chunks(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id=f"c{i}",
                content=f"Raw content chunk {i} about aerodynamic downforce.",
            )
            for i in range(5)
        ]
        _, evidence = assemble_recursive_evidence(
            level=ClusterLevel.LEVEL_1,
            child_primary_text="Summary text from LEVEL_0 clusters — not a raw chunk.",
            all_underlying_raw_chunks=chunks,
            k_evidence=3,
        )
        raw_ids = {c.chunk_id for c in chunks}
        for span in evidence:
            assert span["chunk_id"] in raw_ids

    def test_assemble_recursive_evidence_at_level_2_returns_raw_chunks(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id=f"c{i}",
                content=f"Raw content chunk {i} about turbocharger boost.",
            )
            for i in range(5)
        ]
        _, evidence = assemble_recursive_evidence(
            level=ClusterLevel.LEVEL_2,
            child_primary_text="Top-level summary of LEVEL_1 clusters — not a raw chunk.",
            all_underlying_raw_chunks=chunks,
            k_evidence=3,
        )
        raw_ids = {c.chunk_id for c in chunks}
        for span in evidence:
            assert span["chunk_id"] in raw_ids


class TestVerifyRawEvidenceInvariant:
    def test_verify_raw_evidence_invariant_accepts_only_raw_chunks(self) -> None:
        raw_chunk_ids = {"c1", "c2", "c3"}
        evidence: list[dict[str, str | float]] = [
            {
                "chunk_id": "c1",
                "section_path": "path",
                "text": "text",
                "score": 0.9,
                "selection_method": "bm25",
            },
            {
                "chunk_id": "c2",
                "section_path": "path",
                "text": "text",
                "score": 0.8,
                "selection_method": "bm25",
            },
        ]
        result = verify_raw_evidence_invariant(evidence, raw_chunk_ids)
        assert result is True

    def test_verify_raw_evidence_invariant_rejects_summary_chunks(self) -> None:
        raw_chunk_ids = {"c1", "c2"}
        evidence: list[dict[str, str | float]] = [
            {
                "chunk_id": "c1",
                "section_path": "path",
                "text": "text",
                "score": 0.9,
                "selection_method": "bm25",
            },
            {
                "chunk_id": "summary-node-abc",
                "section_path": "path",
                "text": "text",
                "score": 0.7,
                "selection_method": "bm25",
            },
        ]
        result = verify_raw_evidence_invariant(evidence, raw_chunk_ids)
        assert result is False


class TestBuildLevelExtractSupportPrompt:
    def test_build_level_extract_support_prompt_contains_evidence_block(self) -> None:
        evidence: list[dict[str, str | float]] = [
            {
                "chunk_id": "c1",
                "section_path": "[Book > Chapter]",
                "text": "Extracted evidence text.",
                "score": 0.95,
                "selection_method": "bm25",
            },
        ]
        system_prompt, user_prompt = build_level_extract_support_prompt(
            level=ClusterLevel.LEVEL_0,
            primary_input="Primary input text for the summary.",
            supporting_evidence=evidence,
            book_id="book-001",
        )
        assert isinstance(system_prompt, str)
        assert isinstance(user_prompt, str)
        assert len(system_prompt) > 0
        assert len(user_prompt) > 0
        assert "c1" in user_prompt
        assert "Extracted evidence text" in user_prompt

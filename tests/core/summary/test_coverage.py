"""Tests for vo_rag.core.summary.coverage.

The judge is injected as a callable ``(system, user, max_tokens, temperature) ->
LLMResponse`` so tests never hit a live LLM.
"""

from __future__ import annotations

from typing import Any

import pytest

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.summary.coverage import (
    COVERAGE_PROMPT_HASH,
    CoverageJudgeConfig,
    CoverageJudgment,
    CoverageResult,
    FaithfulnessClaim,
    FaithfulnessResult,
    KeyPoint,
    compute_summary_quality_score,
    extract_claims_from_summary,
    judge_coverage,
    judge_faithfulness,
)


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


class TestExtractClaimsFromSummary:
    def test_extract_claims_from_summary_returns_one_per_sentence(self) -> None:
        summary = (
            "The brake system uses hydraulic fluid. "
            "The differential routes torque to rear wheels. "
            "Weight transfer occurs during braking"
        )
        claims = extract_claims_from_summary(summary)
        assert len(claims) == 3
        assert "brake system uses hydraulic fluid" in claims[0]
        assert "differential routes torque" in claims[1]
        assert "Weight transfer occurs" in claims[2]

    def test_extract_claims_from_summary_empty_string_returns_empty(self) -> None:
        claims = extract_claims_from_summary("")
        assert claims == []


class TestJudgeCoverage:
    @pytest.mark.asyncio
    async def test_judge_coverage_aggregates_yes_no_into_rate(self) -> None:
        key_points = [
            KeyPoint(
                key_point_id="kp-alpha",
                text="hydraulic brake system description",
                source_chunk_ids=["c1"],
            ),
            KeyPoint(
                key_point_id="kp-beta",
                text="torque routing through differential",
                source_chunk_ids=["c2"],
            ),
            KeyPoint(
                key_point_id="kp-gamma",
                text="weight transfer dynamics",
                source_chunk_ids=["c3"],
            ),
        ]
        summary_text = (
            "The brake system relies on hydraulic fluid. "
            "Torque is routed through the differential. "
            "The car is painted red."
        )

        async def fake_llm_call(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            if "hydraulic brake system" in user:
                text = (
                    '{"covered": true, "confidence": 0.9, '
                    '"evidence_in_summary": "brake system hydraulic"}'
                )
            elif "torque routing" in user:
                text = (
                    '{"covered": true, "confidence": 0.8, '
                    '"evidence_in_summary": "torque routed through differential"}'
                )
            else:
                text = '{"covered": false, "confidence": 0.7, "evidence_in_summary": null}'
            return LLMResponse(text=text, model="fake", latency_ms=10)

        config = CoverageJudgeConfig()
        result = await judge_coverage(
            key_points=key_points,
            summary_text=summary_text,
            summary_node_id="sn-001",
            config=config,
            llm_call=fake_llm_call,
        )
        assert result.n_key_points_total == 3
        assert result.n_key_points_covered == 2
        assert result.coverage_rate == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_judge_coverage_records_per_key_point_judgments(self) -> None:
        key_points = [
            KeyPoint(key_point_id="kp1", text="first key point text", source_chunk_ids=["c1"]),
        ]
        summary_text = "This summary covers the first key point."

        async def fake_llm_call(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            return LLMResponse(
                text=(
                    '{"covered": true, "confidence": 0.95, '
                    '"evidence_in_summary": "covers the first key point"}'
                ),
                model="fake",
                latency_ms=10,
            )

        config = CoverageJudgeConfig()
        result = await judge_coverage(
            key_points=key_points,
            summary_text=summary_text,
            summary_node_id="sn-002",
            config=config,
            llm_call=fake_llm_call,
        )
        assert len(result.per_key_point_judgments) == 1
        judgment = result.per_key_point_judgments[0]
        assert judgment.key_point_id == "kp1"
        assert judgment.covered is True
        assert judgment.confidence == pytest.approx(0.95)
        assert judgment.evidence_in_summary == "covers the first key point"

    @pytest.mark.asyncio
    async def test_judge_coverage_propagates_prompt_hash(self) -> None:
        key_points = [
            KeyPoint(key_point_id="kp1", text="text", source_chunk_ids=["c1"]),
        ]

        async def fake_llm_call(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            return LLMResponse(
                text='{"covered": false, "confidence": 0.5, "evidence_in_summary": null}',
                model="fake",
                latency_ms=10,
            )

        config = CoverageJudgeConfig()
        result = await judge_coverage(
            key_points=key_points,
            summary_text="summary",
            summary_node_id="sn-003",
            config=config,
            llm_call=fake_llm_call,
        )
        assert result.judge_prompt_hash == COVERAGE_PROMPT_HASH


class TestJudgeFaithfulness:
    @pytest.mark.asyncio
    async def test_judge_faithfulness_aggregates_grounded_into_rate(self) -> None:
        chunks = [
            _make_chunk(chunk_id="c1", content="The brake system uses hydraulic fluid."),
            _make_chunk(chunk_id="c2", content="The differential routes torque to rear wheels."),
        ]
        summary_text = "The brake system uses hydraulic fluid. The engine runs on fuel."

        async def fake_llm_call(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            if "brake system" in user and "engine" not in user:
                text = '{"grounded": true, "confidence": 0.9, "supporting_chunk_ids": ["c1"]}'
            else:
                text = '{"grounded": false, "confidence": 0.6, "supporting_chunk_ids": []}'
            return LLMResponse(text=text, model="fake", latency_ms=10)

        config = CoverageJudgeConfig()
        result = await judge_faithfulness(
            summary_text=summary_text,
            summary_node_id="sn-004",
            source_chunks=chunks,
            config=config,
            llm_call=fake_llm_call,
        )
        assert result.n_claims_extracted == 2
        assert result.n_claims_grounded == 1
        assert result.faithfulness_rate == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_judge_faithfulness_records_supporting_chunk_ids(self) -> None:
        chunks = [
            _make_chunk(chunk_id="c1", content="The brake system uses hydraulic fluid."),
            _make_chunk(chunk_id="c2", content="The differential routes torque."),
        ]
        summary_text = (
            "The brake system uses hydraulic fluid. The differential routes torque to the wheels."
        )

        async def fake_llm_call(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            if "differential routes torque" in user:
                text = '{"grounded": true, "confidence": 0.88, "supporting_chunk_ids": ["c2"]}'
            else:
                text = '{"grounded": true, "confidence": 0.95, "supporting_chunk_ids": ["c1"]}'
            return LLMResponse(text=text, model="fake", latency_ms=10)

        config = CoverageJudgeConfig()
        result = await judge_faithfulness(
            summary_text=summary_text,
            summary_node_id="sn-005",
            source_chunks=chunks,
            config=config,
            llm_call=fake_llm_call,
        )
        assert len(result.per_claim_judgments) == 2
        grounded_claims = [j for j in result.per_claim_judgments if j.grounded]
        assert len(grounded_claims) == 2
        for claim in grounded_claims:
            assert len(claim.supporting_chunk_ids) > 0


class TestComputeSummaryQualityScore:
    def test_compute_summary_quality_score_composite_is_average(self) -> None:
        coverage_result = CoverageResult(
            summary_node_id="sn-006",
            n_key_points_total=2,
            n_key_points_covered=2,
            coverage_rate=1.0,
            per_key_point_judgments=[
                CoverageJudgment(
                    key_point_id="kp1", covered=True, confidence=0.9, evidence_in_summary="ev1"
                ),
                CoverageJudgment(
                    key_point_id="kp2", covered=True, confidence=0.9, evidence_in_summary="ev2"
                ),
            ],
            judge_prompt_hash="hash_cov",
        )
        faithfulness_result = FaithfulnessResult(
            summary_node_id="sn-006",
            n_claims_extracted=2,
            n_claims_grounded=2,
            faithfulness_rate=1.0,
            per_claim_judgments=[
                FaithfulnessClaim(
                    claim_id="cl1",
                    claim_text="c1",
                    grounded=True,
                    confidence=0.9,
                    supporting_chunk_ids=["c1"],
                ),
                FaithfulnessClaim(
                    claim_id="cl2",
                    claim_text="c2",
                    grounded=True,
                    confidence=0.9,
                    supporting_chunk_ids=["c2"],
                ),
            ],
            judge_prompt_hash="hash_faith",
        )
        score = compute_summary_quality_score(
            coverage=coverage_result, faithfulness=faithfulness_result
        )
        assert score.composite_score == pytest.approx(1.0)
        assert score.summary_node_id == "sn-006"

    def test_compute_summary_quality_score_ci_low_lte_composite(self) -> None:
        coverage_result = CoverageResult(
            summary_node_id="sn-007",
            n_key_points_total=2,
            n_key_points_covered=1,
            coverage_rate=0.5,
            per_key_point_judgments=[
                CoverageJudgment(
                    key_point_id="kp1", covered=True, confidence=0.9, evidence_in_summary="ev1"
                ),
                CoverageJudgment(
                    key_point_id="kp2", covered=False, confidence=0.7, evidence_in_summary=None
                ),
            ],
            judge_prompt_hash="hash_cov",
        )
        faithfulness_result = FaithfulnessResult(
            summary_node_id="sn-007",
            n_claims_extracted=2,
            n_claims_grounded=1,
            faithfulness_rate=0.5,
            per_claim_judgments=[
                FaithfulnessClaim(
                    claim_id="cl1",
                    claim_text="c1",
                    grounded=True,
                    confidence=0.8,
                    supporting_chunk_ids=["c1"],
                ),
                FaithfulnessClaim(
                    claim_id="cl2",
                    claim_text="c2",
                    grounded=False,
                    confidence=0.6,
                    supporting_chunk_ids=[],
                ),
            ],
            judge_prompt_hash="hash_faith",
        )
        score = compute_summary_quality_score(
            coverage=coverage_result, faithfulness=faithfulness_result
        )
        assert score.composite_score == pytest.approx(0.5)
        assert score.composite_ci_low <= score.composite_score


class TestCoverageResultValidation:
    def test_coverage_result_validates_with_empty_judgments(self) -> None:
        result = CoverageResult(
            summary_node_id="sn-empty",
            n_key_points_total=0,
            n_key_points_covered=0,
            coverage_rate=0.0,
            per_key_point_judgments=[],
            judge_prompt_hash="hash",
        )
        assert result.n_key_points_total == 0
        assert result.n_key_points_covered == 0
        assert result.coverage_rate == 0.0


class TestFaithfulnessResultValidation:
    def test_faithfulness_result_validates_with_empty_judgments(self) -> None:
        result = FaithfulnessResult(
            summary_node_id="sn-empty-faith",
            n_claims_extracted=0,
            n_claims_grounded=0,
            faithfulness_rate=0.0,
            per_claim_judgments=[],
            judge_prompt_hash="hash",
        )
        assert result.n_claims_extracted == 0
        assert result.n_claims_grounded == 0
        assert result.faithfulness_rate == 0.0

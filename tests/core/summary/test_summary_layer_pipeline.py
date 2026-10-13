"""End-to-end integration smoke for the Layer 1A summary pipeline (VOR-4).

Combines:

- Summary builder (mocked LLM + mocked embed + in-memory Qdrant store)
- Coverage + faithfulness scoring (mocked judge LLM)
- Evaluation gate (SummaryMechanicalResult + SummaryPerQueryScore)
- Summary-layer aggregator (decide_phase2 + emit_phase2_report)

Mirrors the patterns in ``test_builder.py`` and the dsi integration test.
No live LLMs, no live Qdrant.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.embedding.schemas import SparseVec
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.storage.qdrant import QdrantStore
from vo_rag.core.summary.builder import BuildConfig, SummaryTreeBuilder
from vo_rag.core.summary.coverage import (
    COVERAGE_PROMPT_HASH,
    CoverageJudgeConfig,
    KeyPoint,
    compute_summary_quality_score,
    judge_coverage,
    judge_faithfulness,
)
from vo_rag.core.summary.evaluation import (
    SummaryMechanicalResult,
    SummaryPerQueryScore,
    emit_summary_eval_report,
    evaluate_summary_layer,
)
from vo_rag.core.summary.phase2_aggregator import (
    aggregate_corpus_quality,
    decide_phase2,
    emit_phase2_report,
)

# ---------------------------------------------------------------------------
# Shared test doubles (mirrored from tests/core/summary/test_builder.py)
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    chunk_id: str,
    content: str,
    section_path: str,
    page_number: int | None = None,
) -> RetrievalChunk:
    """Build a minimal RetrievalChunk."""
    return RetrievalChunk(
        chunk_id=chunk_id,
        chunk_type="text",
        document_id="book-int-001",
        source_id="source-int-001",
        page_number=page_number,
        page_range=None,
        section_path=section_path,
        section_title="Test Section",
        content=content,
        embedding_text=f"{section_path}\n\n{content}",
        token_count=max(1, len(content.split())),
        has_image=False,
        image_id=None,
        image_ref=None,
        vlm_description=None,
        metadata={},
    )


class _InMemoryQdrantClient:
    """Dict-backed AsyncQdrantClient test double — no unittest.mock."""

    def __init__(self) -> None:
        self._points: dict[str, dict[str, Any]] = {}

    async def upsert(self, collection_name: str, points: list[Any], wait: bool = True) -> None:
        for point in points:
            self._points[point.id] = {
                "id": point.id,
                "vector": point.vector,
                "payload": point.payload,
            }

    async def retrieve(
        self, collection_name: str, ids: list[str], with_payload: bool = True
    ) -> list[Any]:
        class _Point:
            def __init__(self, data: dict[str, Any]) -> None:
                self.id = data["id"]
                self.payload = data["payload"]
                self.vector = data.get("vector", {})

        return [_Point(self._points[oid]) for oid in ids if oid in self._points]

    async def scroll(
        self,
        collection_name: str,
        scroll_filter: Any = None,
        limit: int = 100,
        with_payload: bool = True,
    ) -> tuple[list[Any], Any]:
        class _Point:
            def __init__(self, data: dict[str, Any]) -> None:
                self.id = data["id"]
                self.payload = data["payload"]

        return [_Point(p) for p in list(self._points.values())[:limit]], None

    async def get_collections(self) -> Any:
        class _Collections:
            collections: list[Any] = []

        return _Collections()

    async def create_collection(
        self,
        collection_name: str,
        vectors_config: Any = None,
        sparse_vectors_config: Any = None,
    ) -> None:
        pass


class _InMemoryStore(QdrantStore):
    """QdrantStore subclass backed by _InMemoryQdrantClient."""

    def __init__(self) -> None:
        # Bypass QdrantStore.__init__ (it would try to construct a real
        # AsyncQdrantClient); we just need an object that exposes the
        # in-memory store via the same attribute name the parent class
        # uses for persistence and selective-expansion seams.
        self._client = _InMemoryQdrantClient()


def _mock_embed_client() -> Any:
    class _FakeEmbedClient:
        async def embed(
            self,
            texts: list[str],
            *,
            return_dense: bool = True,
            return_sparse: bool = False,
            return_colbert: bool = False,
        ) -> dict[str, Any]:
            dense = [0.05] * 1024
            return {
                "dense": [dense] * len(texts),
                "sparse": [SparseVec(indices=[0, 1, 2], values=[0.1, 0.2, 0.3])] * len(texts),
                "colbert": [[[0.05] * 1024]] * len(texts) if return_colbert else [],
            }

    return _FakeEmbedClient()


def _mock_llm_client(response_text: str = "Summary sentence one. Summary sentence two.") -> Any:
    class _FakeLLMClient:
        def __init__(self, text: str) -> None:
            self._text = text

        async def complete(
            self,
            *,
            system: str,
            messages: list[dict[str, Any]],
            max_tokens: int,
            temperature: float,
            extra: dict[str, Any] | None = None,
        ) -> LLMResponse:
            return LLMResponse(
                text=self._text, model="test-model", input_tokens=50, output_tokens=20
            )

    return _FakeLLMClient(response_text)


def _default_build_config() -> BuildConfig:
    return BuildConfig(
        summary_model="test-model",
        embedding_model="BGE-M3",
        embedding_version="test-v1",
        source_collection_version="test-v1",
    )


def _two_section_chunks() -> list[RetrievalChunk]:
    """Two chunks in one chapter → 2 section + 1 chapter + 1 book = 4 LLM calls."""
    return [
        _make_chunk(
            chunk_id="c1",
            content="Aerodynamics of the front wing generates downforce.",
            section_path="[Book > Ch1 > S1]",
            page_number=1,
        ),
        _make_chunk(
            chunk_id="c2",
            content="Rear wing stall reduces drag on straight sections.",
            section_path="[Book > Ch1 > S2]",
            page_number=2,
        ),
    ]


def _passing_mechanical() -> SummaryMechanicalResult:
    return SummaryMechanicalResult(
        n_books_processed=10,
        n_books_extraction_failed=0,
        failure_rate=0.0,
        n_evidence_chunk_id_violations=0,
        prompt_hash_invariance_passed=True,
    )


def _strong_per_query_scores(n: int = 20) -> list[SummaryPerQueryScore]:
    """Scores that produce recall non-inferiority + nDCG superiority."""
    return [
        SummaryPerQueryScore(
            query_id=f"q{i}",
            with_summary_recall_at_10=0.85,
            without_summary_recall_at_10=0.80,
            with_summary_ndcg_at_10=0.88,
            without_summary_ndcg_at_10=0.72,
        )
        for i in range(n)
    ]


def _weak_per_query_scores(n: int = 20) -> list[SummaryPerQueryScore]:
    """Scores that are effectively tied — recall non-inferior but no superior metric."""
    return [
        SummaryPerQueryScore(
            query_id=f"q{i}",
            with_summary_recall_at_10=0.75,
            without_summary_recall_at_10=0.74,
            with_summary_ndcg_at_10=0.74,
            without_summary_ndcg_at_10=0.73,
        )
        for i in range(n)
    ]


def _judge_always_yes() -> Any:
    """Injectable llm_call that returns covered=true and grounded=true for every call."""

    async def _llm_call(system: str, user: str, max_tokens: int, temperature: float) -> LLMResponse:
        if "covered" in system or "key point" in system.lower():
            payload = json.dumps(
                {"covered": True, "confidence": 0.95, "evidence_in_summary": "matching text"}
            )
        else:
            payload = json.dumps(
                {"grounded": True, "confidence": 0.95, "supporting_chunk_ids": ["c1"]}
            )
        return LLMResponse(text=payload, model="judge-model", input_tokens=30, output_tokens=15)

    return _llm_call


def _judge_mostly_no() -> Any:
    """Injectable llm_call that returns covered=false and grounded=false for every call."""

    async def _llm_call(system: str, user: str, max_tokens: int, temperature: float) -> LLMResponse:
        if "covered" in system or "key point" in system.lower():
            payload = json.dumps(
                {"covered": False, "confidence": 0.85, "evidence_in_summary": None}
            )
        else:
            payload = json.dumps(
                {"grounded": False, "confidence": 0.85, "supporting_chunk_ids": []}
            )
        return LLMResponse(text=payload, model="judge-model", input_tokens=30, output_tokens=15)

    return _llm_call


def _make_key_points(node_id: str, chunk_ids: list[str]) -> list[KeyPoint]:
    return [
        KeyPoint(
            key_point_id=f"{node_id}-kp{i}",
            text=f"Key point {i} about aerodynamics and racing strategy",
            source_chunk_ids=[cid],
        )
        for i, cid in enumerate(chunk_ids)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEndToEndBuildScoreDecideEnable:
    @pytest.mark.asyncio
    async def test_end_to_end_build_score_decide_enable(self) -> None:
        """Build tree → score with all-YES judge → eval accepts → aggregator enables."""
        chunks = _two_section_chunks()
        builder = SummaryTreeBuilder(
            llm_client=_mock_llm_client(),
            embed_client=_mock_embed_client(),
            store=_InMemoryStore(),
            config=_default_build_config(),
        )
        nodes = await builder.build_book_tree(book_id="book-int-001", chunks=chunks)
        # Two same-chapter chunks coalesce into one leaf → leaf + book (balanced
        # tree adds no chapter tier below the fan-out cap).
        assert len(nodes) >= 2

        config = CoverageJudgeConfig()
        judge_call = _judge_always_yes()

        quality_scores = []
        for node in nodes:
            key_points = _make_key_points(node.node_id, node.source_chunk_ids[:2])
            coverage = await judge_coverage(
                key_points=key_points,
                summary_text=node.summary_text,
                summary_node_id=node.node_id,
                config=config,
                llm_call=judge_call,
            )
            source_chunks = [c for c in chunks if c.chunk_id in node.source_chunk_ids]
            if not source_chunks:
                source_chunks = chunks
            faithfulness = await judge_faithfulness(
                summary_text=node.summary_text,
                summary_node_id=node.node_id,
                source_chunks=source_chunks,
                config=config,
                llm_call=judge_call,
            )
            quality_scores.append(
                compute_summary_quality_score(coverage=coverage, faithfulness=faithfulness)
            )

        corpus_quality = aggregate_corpus_quality(quality_scores, min_required_composite=0.70)
        assert corpus_quality.pass_gate is True

        eval_verdict = evaluate_summary_layer(
            mechanical=_passing_mechanical(),
            per_query_scores=_strong_per_query_scores(),
        )
        assert eval_verdict.summary_layer_accepted is True

        final = decide_phase2(evaluation_verdict=eval_verdict, corpus_quality=corpus_quality)
        assert final.action == "enable_summary_layer"
        assert final.keep_summary_layer is True
        assert final.failure_reasons == []


class TestEndToEndBuildScoreDecideQualityFail:
    @pytest.mark.asyncio
    async def test_end_to_end_build_score_decide_quality_fail(self) -> None:
        """Mocked judge returns all NO → corpus composite < 0.70 → aggregator disables."""
        chunks = _two_section_chunks()
        builder = SummaryTreeBuilder(
            llm_client=_mock_llm_client(),
            embed_client=_mock_embed_client(),
            store=_InMemoryStore(),
            config=_default_build_config(),
        )
        nodes = await builder.build_book_tree(book_id="book-int-002", chunks=chunks)

        config = CoverageJudgeConfig()
        judge_call = _judge_mostly_no()

        quality_scores = []
        for node in nodes:
            key_points = _make_key_points(node.node_id, node.source_chunk_ids[:2])
            coverage = await judge_coverage(
                key_points=key_points,
                summary_text=node.summary_text,
                summary_node_id=node.node_id,
                config=config,
                llm_call=judge_call,
            )
            source_chunks = [c for c in chunks if c.chunk_id in node.source_chunk_ids]
            if not source_chunks:
                source_chunks = chunks
            faithfulness = await judge_faithfulness(
                summary_text=node.summary_text,
                summary_node_id=node.node_id,
                source_chunks=source_chunks,
                config=config,
                llm_call=judge_call,
            )
            quality_scores.append(
                compute_summary_quality_score(coverage=coverage, faithfulness=faithfulness)
            )

        corpus_quality = aggregate_corpus_quality(quality_scores, min_required_composite=0.70)
        assert corpus_quality.pass_gate is False
        assert corpus_quality.mean_composite_score < 0.70

        eval_verdict = evaluate_summary_layer(
            mechanical=_passing_mechanical(),
            per_query_scores=_strong_per_query_scores(),
        )

        final = decide_phase2(evaluation_verdict=eval_verdict, corpus_quality=corpus_quality)
        assert final.action == "disable_summary_layer"
        assert final.keep_summary_layer is False
        assert len(final.failure_reasons) > 0


class TestEndToEndMechViolationsYieldRerun:
    @pytest.mark.asyncio
    async def test_end_to_end_mech_violations_yield_rerun(self) -> None:
        """Mechanical gate has evidence_chunk_id violations → eval=rerun → aggregator=rerun."""
        mechanical_with_violations = SummaryMechanicalResult(
            n_books_processed=10,
            n_books_extraction_failed=0,
            failure_rate=0.0,
            n_evidence_chunk_id_violations=3,
            prompt_hash_invariance_passed=True,
        )
        assert mechanical_with_violations.pass_gate is False

        eval_verdict = evaluate_summary_layer(
            mechanical=mechanical_with_violations,
            per_query_scores=_strong_per_query_scores(),
        )
        assert eval_verdict.action == "rerun_after_fixes"
        assert eval_verdict.summary_layer_accepted is False
        assert any(
            "provenance" in r.lower() or "chunk-id" in r.lower() or "chunk_id" in r.lower()
            for r in eval_verdict.failure_reasons
        )

        corpus_quality = aggregate_corpus_quality(
            [
                _build_quality_score(f"node-{i}", coverage_rate=0.85, faithfulness_rate=0.85)
                for i in range(3)
            ]
        )

        final = decide_phase2(evaluation_verdict=eval_verdict, corpus_quality=corpus_quality)
        assert final.action == "rerun_after_fixes"
        assert final.keep_summary_layer is False


class TestPerSummaryNodeScoringRecordsPromptHash:
    @pytest.mark.asyncio
    async def test_per_summary_node_scoring_records_prompt_hash(self) -> None:
        """prompt_hash propagates from CoverageJudgeConfig through to CoverageResult."""
        chunks = [
            _make_chunk(
                chunk_id="c-hash-1",
                content="Downforce generated by front wing at high speed.",
                section_path="[Book > Ch1 > S1]",
                page_number=1,
            ),
        ]
        builder = SummaryTreeBuilder(
            llm_client=_mock_llm_client(),
            embed_client=_mock_embed_client(),
            store=_InMemoryStore(),
            config=_default_build_config(),
        )
        nodes = await builder.build_book_tree(book_id="book-hash-int", chunks=chunks)
        assert len(nodes) >= 1

        config = CoverageJudgeConfig()
        assert config.prompt_hash == COVERAGE_PROMPT_HASH

        node = nodes[0]
        key_points = [
            KeyPoint(
                key_point_id="kp-hash-1",
                text="Downforce at high speed",
                source_chunk_ids=["c-hash-1"],
            )
        ]

        async def _judge_call(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            payload = json.dumps(
                {"covered": True, "confidence": 0.9, "evidence_in_summary": "downforce"}
            )
            return LLMResponse(text=payload, model="judge", input_tokens=20, output_tokens=10)

        coverage = await judge_coverage(
            key_points=key_points,
            summary_text=node.summary_text,
            summary_node_id=node.node_id,
            config=config,
            llm_call=_judge_call,
        )

        assert coverage.judge_prompt_hash == COVERAGE_PROMPT_HASH
        assert len(coverage.judge_prompt_hash) == 64
        assert all(c in "0123456789abcdef" for c in coverage.judge_prompt_hash)


class TestAggregatorVerdictFailureReasonsPinpointGate:
    @pytest.mark.asyncio
    async def test_aggregator_verdict_failure_reasons_pinpoint_gate(self) -> None:
        """Each failure mode produces a failure_reasons entry that names the failing gate."""
        # Gate 1: mechanical failure rate exceeded
        mech_rate_fail = SummaryMechanicalResult(
            n_books_processed=10,
            n_books_extraction_failed=3,
            failure_rate=3 / 13,
            max_acceptable_failure_rate=0.05,
            n_evidence_chunk_id_violations=0,
            prompt_hash_invariance_passed=True,
        )
        assert mech_rate_fail.pass_gate is False
        verdict_rate = evaluate_summary_layer(
            mechanical=mech_rate_fail,
            per_query_scores=_strong_per_query_scores(),
        )
        assert any("failure rate" in r.lower() for r in verdict_rate.failure_reasons)

        # Gate 2: evidence chunk-id violations
        mech_violations = SummaryMechanicalResult(
            n_books_processed=10,
            n_books_extraction_failed=0,
            failure_rate=0.0,
            n_evidence_chunk_id_violations=5,
            prompt_hash_invariance_passed=True,
        )
        assert mech_violations.pass_gate is False
        verdict_viol = evaluate_summary_layer(
            mechanical=mech_violations,
            per_query_scores=_strong_per_query_scores(),
        )
        assert any(
            "provenance" in r.lower() or "chunk-id" in r.lower() or "chunk_id" in r.lower()
            for r in verdict_viol.failure_reasons
        )

        # Gate 3: prompt hash invariance failed
        mech_hash_fail = SummaryMechanicalResult(
            n_books_processed=10,
            n_books_extraction_failed=0,
            failure_rate=0.0,
            n_evidence_chunk_id_violations=0,
            prompt_hash_invariance_passed=False,
        )
        assert mech_hash_fail.pass_gate is False
        verdict_hash = evaluate_summary_layer(
            mechanical=mech_hash_fail,
            per_query_scores=_strong_per_query_scores(),
        )
        assert any(
            "prompt" in r.lower() or "hash" in r.lower() or "invariance" in r.lower()
            for r in verdict_hash.failure_reasons
        )

        # Gate 4: corpus quality fails → aggregator names quality gate
        eval_accept = evaluate_summary_layer(
            mechanical=_passing_mechanical(),
            per_query_scores=_strong_per_query_scores(),
        )
        assert eval_accept.action == "accept_summary_layer"
        low_quality = aggregate_corpus_quality(
            [_build_quality_score(f"n{i}", 0.30, 0.30) for i in range(5)]
        )
        final = decide_phase2(evaluation_verdict=eval_accept, corpus_quality=low_quality)
        assert final.action == "disable_summary_layer"
        assert any(
            "quality" in r.lower() or "composite" in r.lower() for r in final.failure_reasons
        )


class TestAggregatorReportRendersPerGateStatus:
    @pytest.mark.asyncio
    async def test_aggregator_report_renders_per_gate_status(self) -> None:
        """emit_phase2_report output contains lines for each key gate result."""
        eval_verdict = evaluate_summary_layer(
            mechanical=_passing_mechanical(),
            per_query_scores=_strong_per_query_scores(),
        )
        corpus_quality = aggregate_corpus_quality(
            [_build_quality_score(f"node-{i}", 0.85, 0.85) for i in range(5)]
        )
        final = decide_phase2(evaluation_verdict=eval_verdict, corpus_quality=corpus_quality)

        report = emit_phase2_report(final)
        assert isinstance(report, str)

        lines = report.splitlines()
        assert len(lines) > 10

        # Evaluation status line
        assert any("action" in line.lower() or "eval" in line.lower() for line in lines)
        # Corpus quality status lines
        assert any(
            "corpus quality" in line.lower() or "mean_composite" in line.lower() for line in lines
        )
        # Final action line
        assert any(
            "enable_summary_layer" in line
            or "disable_summary_layer" in line
            or "rerun_after_fixes" in line
            for line in lines
        )

        # Also verify the eval-side report via emit_summary_eval_report
        eval_report = emit_summary_eval_report(eval_verdict)
        assert "LAYER 1A SUMMARY EVALUATION REPORT" in eval_report
        eval_report_lines = eval_report.splitlines()
        # Recall@10 section present
        assert any("Recall" in line for line in eval_report_lines)
        # nDCG section present
        assert any("nDCG" in line for line in eval_report_lines)
        # Verdict line present
        assert any(
            "Verdict" in line or "ACCEPT" in line or "REJECT" in line for line in eval_report_lines
        )


# ---------------------------------------------------------------------------
# Private helper (not a test, no pytest prefix) used by multiple tests
# ---------------------------------------------------------------------------


def _build_quality_score(
    node_id: str,
    coverage_rate: float,
    faithfulness_rate: float,
) -> Any:
    """Build a SummaryQualityScore without calling a live judge."""
    from vo_rag.core.summary.coverage import (
        CoverageJudgment,
        CoverageResult,
        FaithfulnessClaim,
        FaithfulnessResult,
        SummaryQualityScore,
    )

    n_kp = 4
    n_covered = round(coverage_rate * n_kp)
    kp_judgments = [
        CoverageJudgment(
            key_point_id=f"{node_id}-kp{i}",
            covered=i < n_covered,
            confidence=0.9,
            evidence_in_summary="evidence" if i < n_covered else None,
        )
        for i in range(n_kp)
    ]
    coverage = CoverageResult(
        summary_node_id=node_id,
        n_key_points_total=n_kp,
        n_key_points_covered=n_covered,
        coverage_rate=n_covered / n_kp,
        per_key_point_judgments=kp_judgments,
        judge_prompt_hash="test_cov_hash",
    )

    n_cl = 4
    n_grounded = round(faithfulness_rate * n_cl)
    cl_judgments = [
        FaithfulnessClaim(
            claim_id=f"{node_id}-cl{i}",
            claim_text=f"claim {i}",
            grounded=i < n_grounded,
            confidence=0.9,
            supporting_chunk_ids=[f"c{i}"] if i < n_grounded else [],
        )
        for i in range(n_cl)
    ]
    faithfulness = FaithfulnessResult(
        summary_node_id=node_id,
        n_claims_extracted=n_cl,
        n_claims_grounded=n_grounded,
        faithfulness_rate=n_grounded / n_cl,
        per_claim_judgments=cl_judgments,
        judge_prompt_hash="test_faith_hash",
    )

    composite = 0.5 * coverage_rate + 0.5 * faithfulness_rate
    return SummaryQualityScore(
        summary_node_id=node_id,
        coverage=coverage,
        faithfulness=faithfulness,
        composite_score=composite,
        composite_ci_low=composite - 0.05,
    )

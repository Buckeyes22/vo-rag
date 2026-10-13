from __future__ import annotations

import pytest

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.graph.phase6b_aggregator import GraphSmokeResult
from vo_rag.core.graph.smoke_runner import (
    PerQuerySmokeScore,
    PilotQueryWithEvidence,
    SmokeRunConfig,
    SmokeRunResult,
    aggregate_smoke_metrics,
    compute_per_query_score,
    run_phase6a_smoke,
    to_phase6b_smoke_result,
)


def _chunk(chunk_id: str) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        chunk_type="text",
        document_id="doc1",
        source_id="src1",
        page_number=1,
        page_range="1",
        section_path="/test",
        section_title="Test",
        content="content",
        embedding_text="content",
        token_count=10,
    )


def _query(
    query_id: str,
    gold_chunk_ids: set[str],
    gold_relevance_scores: dict[str, float],
) -> PilotQueryWithEvidence:
    return PilotQueryWithEvidence(
        query_id=query_id,
        query_text=f"query {query_id}",
        gold_chunk_ids=gold_chunk_ids,
        gold_relevance_scores=gold_relevance_scores,
    )


class TestSmokeRunConfig:
    def test_smoke_run_config_defaults(self) -> None:
        config = SmokeRunConfig()
        assert config.min_node_count_pilot == 100
        assert config.min_edge_count_pilot == 200
        assert config.judge_min_improvement_pp == 1.0
        assert config.metrics_to_test == ["recall_at_10", "ndcg_at_10"]
        assert config.pilot_query_count == 20


class TestPilotQueryWithEvidence:
    def test_pilot_query_with_evidence_schema_accepts_gold_chunks_and_scores(self) -> None:
        query = _query("q1", {"chunk_a", "chunk_b"}, {"chunk_a": 1.0, "chunk_b": 0.5})
        assert query.gold_chunk_ids == {"chunk_a", "chunk_b"}
        assert query.gold_relevance_scores == {"chunk_a": 1.0, "chunk_b": 0.5}


class TestComputePerQueryScore:
    def test_compute_per_query_score_perfect_with_graph(self) -> None:
        query = _query("perfect", {"a", "b", "c"}, {"a": 1.0, "b": 0.8, "c": 0.6})
        score = compute_per_query_score(
            query=query,
            with_graph_chunks=[_chunk("a"), _chunk("b"), _chunk("c")],
            without_graph_chunks=[],
        )
        assert score.with_graph_recall_at_10 == 1.0
        assert score.without_graph_recall_at_10 == 0.0

    def test_compute_per_query_score_better_than_without(self) -> None:
        query = _query("compare", {"a", "b"}, {"a": 1.0, "b": 0.5})
        score = compute_per_query_score(
            query=query,
            with_graph_chunks=[_chunk("a"), _chunk("b")],
            without_graph_chunks=[_chunk("a")],
        )
        assert score.with_graph_recall_at_10 > score.without_graph_recall_at_10
        assert score.with_graph_ndcg_at_10 >= score.without_graph_ndcg_at_10

    def test_compute_per_query_score_handles_empty_gold(self) -> None:
        query = _query("empty_gold", set(), {})
        score = compute_per_query_score(
            query=query,
            with_graph_chunks=[_chunk("x"), _chunk("y")],
            without_graph_chunks=[_chunk("x")],
        )
        assert score.with_graph_recall_at_10 == 0.0
        assert score.without_graph_recall_at_10 == 0.0
        assert score.with_graph_ndcg_at_10 == 0.0
        assert score.without_graph_ndcg_at_10 == 0.0


class TestAggregateSmokeMetrics:
    def test_aggregate_smoke_metrics_means_per_metric_correct(self) -> None:
        scores = [
            PerQuerySmokeScore(
                query_id="q1",
                with_graph_recall_at_10=1.0,
                without_graph_recall_at_10=0.5,
                with_graph_ndcg_at_10=0.9,
                without_graph_ndcg_at_10=0.4,
            ),
            PerQuerySmokeScore(
                query_id="q2",
                with_graph_recall_at_10=0.8,
                without_graph_recall_at_10=0.4,
                with_graph_ndcg_at_10=0.7,
                without_graph_ndcg_at_10=0.3,
            ),
        ]
        config = SmokeRunConfig(metrics_to_test=["recall_at_10", "ndcg_at_10"])
        improvements, min_improved = aggregate_smoke_metrics(per_query_scores=scores, config=config)
        recall_diff = (1.0 - 0.5 + 0.8 - 0.4) / 2 * 100.0
        ndcg_diff = (0.9 - 0.4 + 0.7 - 0.3) / 2 * 100.0
        assert improvements["recall_at_10"] == pytest.approx(recall_diff)
        assert improvements["ndcg_at_10"] == pytest.approx(ndcg_diff)
        assert min_improved is True

    def test_aggregate_smoke_metrics_min_improvement_detected_above_threshold(self) -> None:
        scores = [
            PerQuerySmokeScore(
                query_id="q1",
                with_graph_recall_at_10=0.6,
                without_graph_recall_at_10=0.5,
                with_graph_ndcg_at_10=0.6,
                without_graph_ndcg_at_10=0.5,
            ),
        ]
        config = SmokeRunConfig(metrics_to_test=["recall_at_10"], judge_min_improvement_pp=5.0)
        _, min_improved = aggregate_smoke_metrics(per_query_scores=scores, config=config)
        assert min_improved is True

    def test_aggregate_smoke_metrics_min_improvement_not_detected_below_threshold(self) -> None:
        scores = [
            PerQuerySmokeScore(
                query_id="q1",
                with_graph_recall_at_10=0.51,
                without_graph_recall_at_10=0.50,
                with_graph_ndcg_at_10=0.51,
                without_graph_ndcg_at_10=0.50,
            ),
        ]
        config = SmokeRunConfig(metrics_to_test=["recall_at_10"], judge_min_improvement_pp=5.0)
        _, min_improved = aggregate_smoke_metrics(per_query_scores=scores, config=config)
        assert min_improved is False


@pytest.mark.asyncio
class TestRunPhase6aSmoke:
    async def test_run_phase6a_smoke_end_to_end_mocked(self) -> None:
        queries = [_query("q1", {"a", "b"}, {"a": 1.0, "b": 0.5})]

        async def with_graph(_: str) -> list[RetrievalChunk]:
            return [_chunk("a"), _chunk("b")]

        async def without_graph(_: str) -> list[RetrievalChunk]:
            return [_chunk("a")]

        config = SmokeRunConfig(
            min_node_count_pilot=10, min_edge_count_pilot=20, judge_min_improvement_pp=1.0
        )
        result = await run_phase6a_smoke(
            pilot_queries=queries,
            n_nodes_built=50,
            n_edges_built=100,
            with_graph_retriever=with_graph,
            without_graph_retriever=without_graph,
            config=config,
        )
        assert isinstance(result, SmokeRunResult)
        assert result.n_nodes_built == 50
        assert result.smoke_passed is True
        assert result.per_query_scores[0].with_graph_recall_at_10 == 1.0

    async def test_run_phase6a_smoke_fails_node_count_threshold(self) -> None:
        queries = [_query("q1", {"a"}, {"a": 1.0})]

        async def with_graph(_: str) -> list[RetrievalChunk]:
            return [_chunk("a")]

        async def without_graph(_: str) -> list[RetrievalChunk]:
            return [_chunk("a")]

        config = SmokeRunConfig(
            min_node_count_pilot=1000, min_edge_count_pilot=200, judge_min_improvement_pp=1.0
        )
        result = await run_phase6a_smoke(
            pilot_queries=queries,
            n_nodes_built=50,
            n_edges_built=100,
            with_graph_retriever=with_graph,
            without_graph_retriever=without_graph,
            config=config,
        )
        assert result.smoke_passed is False
        assert result.n_nodes_built == 50


class TestToPhase6bSmokeResult:
    def test_to_phase6b_smoke_result_preserves_keys(self) -> None:
        smoke = SmokeRunResult(
            n_nodes_built=150,
            n_edges_built=300,
            per_query_scores=[],
            mean_improvements={"recall_at_10": 5.0, "ndcg_at_10": 3.0},
            min_one_metric_improved=True,
            smoke_passed=True,
        )
        graph_result = to_phase6b_smoke_result(smoke)
        assert isinstance(graph_result, GraphSmokeResult)
        assert graph_result.passed is True
        assert graph_result.n_nodes_built == 150
        assert graph_result.n_edges_built == 300
        assert graph_result.min_one_metric_improved is True
        assert graph_result.improvement_breakdown == {"recall_at_10": 5.0, "ndcg_at_10": 3.0}

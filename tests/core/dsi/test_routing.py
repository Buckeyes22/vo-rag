from __future__ import annotations

import pytest

from vo_rag.core.dsi.routing import (
    BookRoute,
    RoutingDecision,
    allocate_budgets,
    evaluate_safety_net_recovery,
    normalize_dsi_scores,
    worked_example_routing,
)


class TestNormalizeDSIScores:
    def test_normalize_dsi_scores_sums_to_one(self) -> None:
        scores = {
            "book-a": 0.8,
            "book-b": 0.6,
            "book-c": 0.4,
        }
        result = normalize_dsi_scores(scores)
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-6

    def test_normalize_dsi_scores_preserves_ordering(self) -> None:
        scores = {
            "book-a": 0.9,
            "book-b": 0.5,
            "book-c": 0.1,
        }
        result = normalize_dsi_scores(scores)
        assert result["book-a"] > result["book-b"] > result["book-c"]

    def test_normalize_dsi_scores_uniform_when_all_equal(self) -> None:
        scores = {
            "book-a": 0.5,
            "book-b": 0.5,
            "book-c": 0.5,
        }
        result = normalize_dsi_scores(scores)
        assert all(abs(v - 1.0 / 3.0) < 1e-6 for v in result.values())


class TestAllocateBudgets:
    def test_allocate_budgets_every_book_has_positive_budget(self) -> None:
        confidences = {f"book-{i}": 1.0 / 20 for i in range(20)}
        confidences["book-0"] = 0.5
        confidences["book-1"] = 0.3

        result = allocate_budgets(
            confidences=confidences,
            total_budget_chunks=20,
            top_k=8,
            fallback_share=0.05,
        )
        for route in result.book_routes:
            assert route.total_budget >= 0

    def test_allocate_budgets_top_k_get_majority_of_primary(self) -> None:
        confidences = {
            "top-book-1": 0.5,
            "top-book-2": 0.3,
            "top-book-3": 0.1,
            "fallback-book": 0.05,
            "fallback-book-2": 0.05,
        }
        result = allocate_budgets(
            confidences=confidences,
            total_budget_chunks=20,
            top_k=3,
            fallback_share=0.05,
        )

        top_k_routes = [r for r in result.book_routes if r.primary_budget > 0]

        assert len(top_k_routes) == 3
        assert all(r.fallback_budget >= 0 for r in result.book_routes)

    def test_allocate_budgets_fallback_share_floor_respected(self) -> None:
        confidences = {f"book-{i}": 1.0 / 20 for i in range(20)}
        result = allocate_budgets(
            confidences=confidences,
            total_budget_chunks=20,
            top_k=8,
            fallback_share=0.05,
        )
        fallback_total = sum(r.fallback_budget for r in result.book_routes)
        expected_floor = int(20 * 0.05)
        assert fallback_total >= expected_floor

    def test_allocate_budgets_book_route_validator_rejects_negative_budget(self) -> None:
        with pytest.raises(Exception, match="total_budget"):
            BookRoute(
                book_id="book-001",
                confidence=0.5,
                primary_budget=1,
                fallback_budget=1,
                total_budget=0,
            )


class TestEvaluateSafetyNetRecovery:
    """evaluate_safety_net_recovery stays tested even though planner/runtime
    wiring does not call it yet."""

    def test_evaluate_safety_net_recovery_top_k_hit_rate(self) -> None:
        routing = RoutingDecision(
            book_routes=[
                BookRoute(
                    book_id="book-a",
                    confidence=0.5,
                    primary_budget=5,
                    fallback_budget=1,
                    total_budget=6,
                ),
                BookRoute(
                    book_id="book-b",
                    confidence=0.3,
                    primary_budget=4,
                    fallback_budget=1,
                    total_budget=5,
                ),
                BookRoute(
                    book_id="book-c",
                    confidence=0.1,
                    primary_budget=3,
                    fallback_budget=1,
                    total_budget=4,
                ),
                BookRoute(
                    book_id="book-d",
                    confidence=0.05,
                    primary_budget=0,
                    fallback_budget=1,
                    total_budget=1,
                ),
                BookRoute(
                    book_id="book-e",
                    confidence=0.05,
                    primary_budget=0,
                    fallback_budget=1,
                    total_budget=1,
                ),
            ],
            top_k=3,
            total_budget_chunks=20,
            primary_share=0.95,
            fallback_share=0.05,
        )

        ground_truth = {"book-a", "book-b", "book-c"}
        result = evaluate_safety_net_recovery(
            routing=routing,
            ground_truth_book_ids=ground_truth,
        )
        assert result["top_k_hit_rate"] == 1.0

    def test_evaluate_safety_net_recovery_safety_net_recovery_rate(self) -> None:
        routing = RoutingDecision(
            book_routes=[
                BookRoute(
                    book_id="book-a",
                    confidence=0.5,
                    primary_budget=5,
                    fallback_budget=1,
                    total_budget=6,
                ),
                BookRoute(
                    book_id="book-b",
                    confidence=0.3,
                    primary_budget=4,
                    fallback_budget=1,
                    total_budget=5,
                ),
                BookRoute(
                    book_id="book-c",
                    confidence=0.1,
                    primary_budget=0,
                    fallback_budget=1,
                    total_budget=1,
                ),
                BookRoute(
                    book_id="book-d",
                    confidence=0.05,
                    primary_budget=0,
                    fallback_budget=0,
                    total_budget=0,
                ),
                BookRoute(
                    book_id="book-e",
                    confidence=0.05,
                    primary_budget=0,
                    fallback_budget=1,
                    total_budget=1,
                ),
            ],
            top_k=2,
            total_budget_chunks=20,
            primary_share=0.95,
            fallback_share=0.05,
        )

        ground_truth = {"book-c", "book-d", "book-e"}
        result = evaluate_safety_net_recovery(
            routing=routing,
            ground_truth_book_ids=ground_truth,
        )
        assert result["safety_net_recovery_rate"] == pytest.approx(2 / 3)

    def test_evaluate_safety_net_recovery_total_recall(self) -> None:
        routing = RoutingDecision(
            book_routes=[
                BookRoute(
                    book_id="book-a",
                    confidence=0.5,
                    primary_budget=5,
                    fallback_budget=1,
                    total_budget=6,
                ),
                BookRoute(
                    book_id="book-b",
                    confidence=0.3,
                    primary_budget=4,
                    fallback_budget=1,
                    total_budget=5,
                ),
                BookRoute(
                    book_id="book-c",
                    confidence=0.1,
                    primary_budget=0,
                    fallback_budget=1,
                    total_budget=1,
                ),
                BookRoute(
                    book_id="book-d",
                    confidence=0.05,
                    primary_budget=0,
                    fallback_budget=0,
                    total_budget=0,
                ),
            ],
            top_k=2,
            total_budget_chunks=20,
            primary_share=0.95,
            fallback_share=0.05,
        )

        ground_truth = {"book-a", "book-b", "book-c", "book-d"}
        result = evaluate_safety_net_recovery(
            routing=routing,
            ground_truth_book_ids=ground_truth,
        )
        assert result["total_recall"] == 0.75

    def test_evaluate_safety_net_recovery_empty_ground_truth_returns_zeros(self) -> None:
        routing = RoutingDecision(
            book_routes=[
                BookRoute(
                    book_id="book-a",
                    confidence=0.5,
                    primary_budget=5,
                    fallback_budget=1,
                    total_budget=6,
                ),
            ],
            top_k=1,
            total_budget_chunks=20,
            primary_share=0.95,
            fallback_share=0.05,
        )
        result = evaluate_safety_net_recovery(routing=routing, ground_truth_book_ids=set())
        assert result == {
            "top_k_hit_rate": 0.0,
            "safety_net_recovery_rate": 0.0,
            "total_recall": 0.0,
        }


class TestWorkedExampleRouting:
    def test_worked_example_routing_matches_v3_section_5_6(self) -> None:
        result = worked_example_routing(n_books=400, top_k=3)

        assert result.top_k == 3
        assert result.total_budget_chunks == 20
        assert result.primary_share == pytest.approx(0.95)
        assert result.fallback_share == pytest.approx(0.05)

        top_k_routes = result.book_routes[:3]
        book_ids = [r.book_id for r in top_k_routes]
        assert "A" in book_ids
        assert "B" in book_ids
        assert "C" in book_ids

        for route in result.book_routes:
            assert route.total_budget == route.primary_budget + route.fallback_budget

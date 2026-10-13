"""DSI query-time routing — softmax + safety-net budget allocation.

DSI is an advanced-understanding core layer; planner/runtime wiring
(consumers of ``allocate_budgets`` / ``evaluate_safety_net_recovery``) does
not exist yet. The functions are fully unit-tested so they stay verified until
a planner uses them.
"""

from __future__ import annotations

import math

from pydantic import model_validator

from vo_rag.core.models.common import VoBaseModel


class BookRoute(VoBaseModel):
    book_id: str
    confidence: float
    primary_budget: int
    fallback_budget: int
    total_budget: int

    @model_validator(mode="after")
    def _validate_invariants(self) -> BookRoute:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.total_budget != self.primary_budget + self.fallback_budget:
            raise ValueError(
                f"total_budget ({self.total_budget}) must equal "
                f"primary_budget ({self.primary_budget}) + fallback_budget ({self.fallback_budget})"
            )
        return self


class RoutingDecision(VoBaseModel):
    book_routes: list[BookRoute]
    top_k: int
    total_budget_chunks: int
    primary_share: float
    fallback_share: float


def normalize_dsi_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}

    max_score = max(scores.values())
    exp_scores: dict[str, float] = {}
    for book_id, score in scores.items():
        exp_scores[book_id] = math.exp(score - max_score)

    sum_exp = sum(exp_scores.values())
    if sum_exp == 0:
        n = len(scores)
        return dict.fromkeys(scores, 1.0 / n)

    return {book_id: exp_s / sum_exp for book_id, exp_s in exp_scores.items()}


def allocate_budgets(
    *,
    confidences: dict[str, float],
    total_budget_chunks: int = 20,
    top_k: int = 8,
    fallback_share: float = 0.05,
) -> RoutingDecision:
    n_books = len(confidences)
    if n_books == 0:
        return RoutingDecision(
            book_routes=[],
            top_k=top_k,
            total_budget_chunks=total_budget_chunks,
            primary_share=1.0 - fallback_share,
            fallback_share=fallback_share,
        )

    sorted_books = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
    top_k_books_list = sorted_books[:top_k]
    non_top_k_books_list = sorted_books[top_k:]

    fallback_total = math.floor(total_budget_chunks * fallback_share)
    n_non_top_k = len(non_top_k_books_list)

    if n_non_top_k > 0:
        fallback_per_non_top_k = max(1, math.floor(fallback_total / n_non_top_k))
    else:
        fallback_per_non_top_k = 0

    total_fallback_used = fallback_per_non_top_k * n_non_top_k
    primary_total = total_budget_chunks - total_fallback_used

    top_k_confs = dict(top_k_books_list)
    sum_top_k_confs = sum(top_k_confs.values())

    book_routes: list[BookRoute] = []

    for book_id, conf in sorted_books:
        if book_id in top_k_confs:
            if sum_top_k_confs > 0 and primary_total > 0:
                proportional_primary = math.floor(
                    primary_total * (top_k_confs[book_id] / sum_top_k_confs)
                )
                primary_budget = max(1, proportional_primary)
            else:
                primary_budget = 1
            fallback_budget = fallback_per_non_top_k if n_non_top_k > 0 else 0
        else:
            primary_budget = 0
            fallback_budget = fallback_per_non_top_k

        total_budget = primary_budget + fallback_budget

        book_routes.append(
            BookRoute(
                book_id=book_id,
                confidence=conf,
                primary_budget=primary_budget,
                fallback_budget=fallback_budget,
                total_budget=total_budget,
            )
        )

    return RoutingDecision(
        book_routes=book_routes,
        top_k=top_k,
        total_budget_chunks=total_budget_chunks,
        primary_share=1.0 - fallback_share,
        fallback_share=fallback_share,
    )


def evaluate_safety_net_recovery(
    *, routing: RoutingDecision, ground_truth_book_ids: set[str]
) -> dict[str, float]:
    if not ground_truth_book_ids:
        return {
            "top_k_hit_rate": 0.0,
            "safety_net_recovery_rate": 0.0,
            "total_recall": 0.0,
        }

    top_k_book_ids: set[str] = set()
    for route in routing.book_routes[: routing.top_k]:
        top_k_book_ids.add(route.book_id)

    relevant_ids = set(ground_truth_book_ids)

    top_k_hits = relevant_ids & top_k_book_ids
    top_k_hit_rate = len(top_k_hits) / len(relevant_ids)

    safety_net_relevant = relevant_ids - top_k_book_ids
    safety_net_book_ids_with_fallback: set[str] = set()
    for route in routing.book_routes[routing.top_k :]:
        if route.fallback_budget > 0:
            safety_net_book_ids_with_fallback.add(route.book_id)

    safety_net_recovery = safety_net_relevant & safety_net_book_ids_with_fallback
    if len(safety_net_relevant) > 0:
        safety_net_recovery_rate = len(safety_net_recovery) / len(safety_net_relevant)
    else:
        safety_net_recovery_rate = 0.0

    books_with_any_budget: set[str] = {
        route.book_id for route in routing.book_routes if route.total_budget > 0
    }
    total_recall_set = relevant_ids & books_with_any_budget
    total_recall = len(total_recall_set) / len(relevant_ids)

    return {
        "top_k_hit_rate": top_k_hit_rate,
        "safety_net_recovery_rate": safety_net_recovery_rate,
        "total_recall": total_recall,
    }


def worked_example_routing(n_books: int = 400, top_k: int = 3) -> RoutingDecision:
    if n_books == 400:
        confidences: dict[str, float] = {
            "A": 0.45,
            "B": 0.22,
            "C": 0.13,
        }
        remaining_confidence = 1.0 - (0.45 + 0.22 + 0.13)
        per_book = remaining_confidence / (n_books - 3)
        for i in range(3, n_books):
            confidences[f"book_{i}"] = per_book
    elif n_books >= 3:
        confidences = {"A": 0.45, "B": 0.22, "C": 0.13}
        remaining = 1.0 - 0.8
        per_book = remaining / (n_books - 3)
        for i in range(3, n_books):
            confidences[f"book_{i}"] = per_book
    else:
        raise ValueError("n_books must be >= 3")

    normalized = normalize_dsi_scores(confidences)
    return allocate_budgets(
        confidences=normalized,
        total_budget_chunks=20,
        top_k=top_k,
        fallback_share=0.05,
    )


__all__ = [
    "BookRoute",
    "RoutingDecision",
    "normalize_dsi_scores",
    "allocate_budgets",
    "evaluate_safety_net_recovery",
    "worked_example_routing",
]

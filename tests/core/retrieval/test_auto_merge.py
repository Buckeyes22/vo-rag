"""Tests for small-to-big auto-merge retrieval post-processing."""

from __future__ import annotations

from vo_rag.core.retrieval.auto_merge import auto_merge
from vo_rag.core.retrieval.search import SearchResult


def _child(
    child_id: str,
    *,
    score: float,
    parent_id: str | None = "P",
    parent_child_count: int | None = 4,
    collection: str = "books",
) -> SearchResult:
    payload: dict[str, object] = {
        "chunk_id": child_id,
        "source_id": "source-1",
        "content": f"child {child_id}",
    }
    if parent_id is not None:
        payload["parent_id"] = parent_id
    if parent_child_count is not None:
        payload["parent_child_count"] = parent_child_count
    return SearchResult(id=child_id, score=score, collection=collection, payload=payload)


def _unparented(hit_id: str, *, score: float) -> SearchResult:
    return SearchResult(
        id=hit_id,
        score=score,
        collection="books",
        payload={
            "chunk_id": hit_id,
            "source_id": "source-unparented",
            "content": f"unparented {hit_id}",
        },
    )


def _parent(parent_id: str, *, score: float = 0.0) -> SearchResult:
    return SearchResult(
        id=parent_id,
        score=score,
        collection="books",
        payload={
            "chunk_id": parent_id,
            "source_id": "source-1",
            "content": f"parent {parent_id}",
            "page_number": 12,
            "title": "Auto Merge Source",
        },
    )


def test_auto_merge_collapses_majority_children_at_best_child_slot() -> None:
    hits = [
        _child("c1", score=0.91),
        _unparented("u1", score=0.89),
        _child("c2", score=0.72),
        _child("c3", score=0.83),
        _unparented("u2", score=0.6),
    ]
    calls: list[str] = []

    def fetch_parent(parent_id: str) -> SearchResult | None:
        calls.append(parent_id)
        return _parent(parent_id)

    merged = auto_merge(hits, fetch_parent, threshold=0.5)

    assert [result.id for result in merged] == ["P", "u1", "u2"]
    assert merged[0].score == 0.91
    assert merged[1] is hits[1]
    assert merged[2] is hits[4]
    assert calls == ["P"]


def test_auto_merge_does_not_fetch_parent_below_threshold() -> None:
    hits = [_child("c1", score=0.91)]
    calls: list[str] = []

    def fetch_parent(parent_id: str) -> SearchResult | None:
        calls.append(parent_id)
        return _parent(parent_id)

    merged = auto_merge(hits, fetch_parent, threshold=0.5)

    assert merged == hits
    assert calls == []


def test_auto_merge_passes_unparented_hits_through_unchanged() -> None:
    hits = [_unparented("u1", score=0.8), _unparented("u2", score=0.7)]

    def fetch_parent(parent_id: str) -> SearchResult | None:
        raise AssertionError(f"unexpected parent fetch for {parent_id}")

    merged = auto_merge(hits, fetch_parent, threshold=0.5)

    assert merged == hits
    assert merged[0] is hits[0]
    assert merged[1] is hits[1]


def test_auto_merge_preserves_parent_payload_synthesis_fields() -> None:
    hits = [_child("c1", score=0.91), _child("c2", score=0.82)]
    parent = _parent("P")

    def fetch_parent(parent_id: str) -> SearchResult | None:
        assert parent_id == "P"
        return parent

    merged = auto_merge(hits, fetch_parent, threshold=0.5)

    assert merged == [
        SearchResult(
            id="P",
            score=0.91,
            collection="books",
            payload=parent.payload,
        )
    ]
    assert merged[0].payload["content"] == "parent P"
    assert merged[0].payload["chunk_id"] == "P"
    assert merged[0].payload["source_id"] == "source-1"
    assert merged[0].payload["page_number"] == 12
    assert merged[0].payload["title"] == "Auto Merge Source"


def test_auto_merge_keeps_children_when_parent_fetch_returns_none() -> None:
    hits = [_child("c1", score=0.91), _child("c2", score=0.82), _child("c3", score=0.7)]
    calls: list[str] = []

    def fetch_parent(parent_id: str) -> SearchResult | None:
        calls.append(parent_id)
        return None

    merged = auto_merge(hits, fetch_parent, threshold=0.5)

    assert merged == hits
    assert calls == ["P"]


def test_auto_merge_merges_at_exact_threshold_boundary() -> None:
    hits = [_child("c1", score=0.91), _child("c2", score=0.82)]
    calls: list[str] = []

    def fetch_parent(parent_id: str) -> SearchResult | None:
        calls.append(parent_id)
        return _parent(parent_id)

    merged = auto_merge(hits, fetch_parent, threshold=0.5)

    assert [result.id for result in merged] == ["P"]
    assert merged[0].score == 0.91
    assert calls == ["P"]

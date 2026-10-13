"""Contract tests for SearchResult -> NodeWithScore identity handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vo_rag.integrations.llamaindex.convert import (
    ChunkIdMissingError,
    node_with_score_to_search_result,
    search_result_to_node_with_score,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from vo_rag.core.retrieval.search import SearchResult


def test_search_result_to_node_uses_payload_chunk_id_and_carries_point_id(
    make_search_result: Callable[..., SearchResult],
) -> None:
    result = make_search_result(id="qdrant-point-123")

    node_with_score = search_result_to_node_with_score(result)

    assert node_with_score.node.id_ == result.payload["chunk_id"]
    assert node_with_score.node.metadata["qdrant_point_id"] == result.id


def test_search_result_to_node_raises_when_chunk_id_missing(
    make_search_result: Callable[..., SearchResult],
) -> None:
    result = make_search_result(
        id="qdrant-point-should-not-be-node-id",
        payload={"content": "body text", "section_path": "A > B"},
    )

    with pytest.raises(ChunkIdMissingError):
        search_result_to_node_with_score(result)


def test_search_result_round_trip_preserves_license_and_trust_annotations(
    make_search_result: Callable[..., SearchResult],
) -> None:
    result = make_search_result(
        extra_payload={
            "license_class": "pending",
            "_vo_rag_license_warning": "source_license_class=pending",
            "trust_level": "quarantined",
        }
    )

    round_tripped = node_with_score_to_search_result(search_result_to_node_with_score(result))

    assert round_tripped.payload["license_class"] == "pending"
    assert round_tripped.payload["_vo_rag_license_warning"] == "source_license_class=pending"
    assert round_tripped.payload["trust_level"] == "quarantined"

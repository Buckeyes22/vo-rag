"""Exercise real Qdrant filters before ranking for legacy, staged and published sources."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from vo_rag.core.storage.publication_filter import publication_corpus, publication_filter


@pytest.mark.parametrize("revision,expected", [(None, [1, 4]), (2, [3, 4])])
def test_pointer_predicate_hides_incomplete_replacements_before_top_k(
    revision: int | None, expected: list[int]
) -> None:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        "test", vectors_config={"dense": VectorParams(size=2, distance=Distance.COSINE)}
    )
    client.upsert(
        "test",
        points=[
            PointStruct(id=1, vector={"dense": [0.7, 0.3]}, payload={"source_id": "book"}),
            PointStruct(
                id=2,
                vector={"dense": [1.0, 0.0]},
                payload={"source_id": "book", "source_revision": 1},
            ),
            PointStruct(
                id=3,
                vector={"dense": [0.9, 0.1]},
                payload={"source_id": "book", "source_revision": 2},
            ),
            PointStruct(id=4, vector={"dense": [0.6, 0.4]}, payload={"source_id": "other"}),
            PointStruct(
                id=5,
                vector={"dense": [1.0, 0.0]},
                payload={"source_id": "unregistered", "source_revision": 9},
            ),
        ],
    )
    result = client.query_points(
        "test",
        query=[1.0, 0.0],
        using="dense",
        limit=2,
        query_filter=publication_filter({"book": revision}),
    )
    assert [point.id for point in result.points] == expected
    client.close()


def test_caller_source_scope_remains_a_conjunction() -> None:
    caller = Filter(must=[FieldCondition(key="source_id", match=MatchValue(value="other"))])
    actual = publication_filter({"book": 1}, caller)
    assert isinstance(actual.must, list) and actual.must[0] == caller


@pytest.mark.parametrize(
    "name,corpus",
    [
        ("vo_rag_books_raw_v1", "books"),
        ("vo_rag_books_graph_edges_v1", "books"),
        ("vo_rag_reddit_v1", "reddit"),
        ("vo_rag_brain_v1", None),
        ("vo_rag_urls_v1", None),
    ],
)
def test_corpus_namespaces_preserve_brain_and_other_corpora(name: str, corpus: str | None) -> None:
    assert publication_corpus(name) == corpus

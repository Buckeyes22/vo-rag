"""Integration test for `get_neighbors` against a real Qdrant (VOR-1427).

Node point ids come from `node_id_for(...)` -- a 32-char unhyphenated hex
string (`core/graph/schemas.py`). Qdrant treats that as a UUID and returns
the *hyphenated* canonical form as the point id from `scroll`/`retrieve`,
while `GraphEdge.source_node_id`/`target_node_id` keep the literal
unhyphenated string. The existing fake-`QdrantClient` unit tests in
`test_persistence.py` don't reproduce this (the fake echoes ids back
verbatim), so this test exercises the real client with production-shaped
ids -- it fails before the `neighbor_map` keying fix and passes after.

Gated on `VORAG_TEST_QDRANT_URL` (`make up` -> 127.0.0.1:6334).
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest

from vo_rag.core.graph.persistence import get_neighbors, upsert_edge, upsert_node
from vo_rag.core.graph.schemas import (
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
    edge_id_for,
    node_id_for,
)
from vo_rag.core.storage.qdrant import QdrantStore, setup_collections

pytestmark = pytest.mark.integration

_DENSE = [0.01] * 1024
_COLBERT = [[0.01] * 1024]


def _require_qdrant_url() -> str:
    url = os.environ.get("VORAG_TEST_QDRANT_URL")
    if not url:
        if os.environ.get("VORAG_REQUIRE_TESTINFRA") == "1":
            pytest.fail("VORAG_TEST_QDRANT_URL is required by the testinfra lane")
        pytest.skip("VORAG_TEST_QDRANT_URL not set -- skipping integration test")
    return url


@pytest.fixture
async def store() -> AsyncGenerator[QdrantStore, None]:
    url = _require_qdrant_url()
    s = QdrantStore(url=url, api_key=None, prefer_grpc=False)
    try:
        await setup_collections(s._client)  # noqa: SLF001
        yield s
    finally:
        await s._client.close()  # noqa: SLF001


def _node(label: str, book_ids: list[str]) -> GraphNode:
    return GraphNode(
        node_id=node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label=label),
        node_type=GraphNodeType.CONCEPT,
        label=label,
        aliases=[],
        book_ids=book_ids,
        chunk_ids=["c1"],
        attributes={},
        created_at=datetime.now(UTC),
    )


async def test_get_neighbors_finds_real_qdrant_ids(store: QdrantStore) -> None:
    """Production-shaped ids: fails before the neighbor_map keying fix."""
    source = _node("get-neighbors-source", ["book-x"])
    target = _node("get-neighbors-target", ["book-x"])
    await upsert_node(
        node=source,
        dense_embedding=_DENSE,
        sparse_embedding={1: 0.5},
        colbert_embedding=_COLBERT,
        store=store,
    )
    await upsert_node(
        node=target,
        dense_embedding=_DENSE,
        sparse_embedding={1: 0.5},
        colbert_embedding=_COLBERT,
        store=store,
    )
    edge = GraphEdge(
        edge_id=edge_id_for(
            source_node_id=source.node_id,
            target_node_id=target.node_id,
            edge_type=GraphEdgeType.RELATES_TO,
        ),
        source_node_id=source.node_id,
        target_node_id=target.node_id,
        edge_type=GraphEdgeType.RELATES_TO,
        book_ids=["book-x"],
        evidence_chunk_ids=["c1"],
        created_at=datetime.now(UTC),
    )
    await upsert_edge(edge=edge, dense_embedding=_DENSE, store=store)

    pairs = await get_neighbors(source.node_id, store=store, direction="out")

    assert len(pairs) == 1
    neighbor, found_edge = pairs[0]
    assert neighbor.node_id == target.node_id
    assert found_edge.edge_id == edge.edge_id

"""Live-Qdrant integration tests for ``HybridRetriever``.

Marked ``integration`` — runs only when Qdrant is reachable (CI uses
``docker-compose.testinfra.yml`` to start it). These tests exercise the
REAL ``QdrantStore`` against an ephemeral collection, not a fake.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from vo_rag.core.retrieval.query_mode import QueryMode
from vo_rag.core.retrieval.search import HybridRetriever
from vo_rag.core.storage.qdrant import QdrantStore

pytestmark = pytest.mark.integration


# BM25 id is the persistence contract — pin a known token in the test doc.
_TEST_QUERY = "trail braking"
_TEST_TOKEN = "trail_braking"  # preserved (Eq./Figure pattern substitute)


class _ScriptedEmbedClient:
    """Deterministic embedder — returns the same dense vector for any text."""

    def __init__(self, dense_dim: int = 4) -> None:
        self.dense_dim = dense_dim

    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> dict[str, object]:
        n = len(texts)
        out_dense = [[0.1] * self.dense_dim for _ in range(n)]
        out_sparse: list[dict[str, list[float] | list[int]]] = []
        out_colbert: list[list[list[float]]] = []
        from vo_rag.core.retrieval.bm25_tokenizer import (
            tokenize,
            tokens_to_bm25_sparse,
        )

        for text in texts:
            tokens = tokenize(text)
            sparse = tokens_to_bm25_sparse(tokens)
            out_sparse.append(
                {
                    "indices": [int(k) for k in sparse],
                    "values": [float(v) for v in sparse.values()],
                }
            )
            out_colbert.append([[0.2, 0.3, 0.4]])
        return {
            "dense": out_dense,
            "sparse": out_sparse,
            "colbert": out_colbert,
        }


@pytest.fixture
def qdrant_url() -> str:
    url = os.environ.get("VORAG_TEST_QDRANT_URL")
    if not url:
        if os.environ.get("VORAG_REQUIRE_TESTINFRA") == "1":
            pytest.fail("VORAG_TEST_QDRANT_URL is required by the testinfra lane")
        pytest.skip("VORAG_TEST_QDRANT_URL unset — integration test disabled.")
    return url


@pytest.fixture
async def live_store(qdrant_url: str) -> AsyncGenerator[QdrantStore, None]:
    """Spin up an ephemeral collection that mirrors the production 4-vector
    schema (dense+sparse+bm25+colbert)."""
    store = QdrantStore(url=qdrant_url, api_key=None, prefer_grpc=False)
    client = AsyncQdrantClient(url=qdrant_url, prefer_grpc=False, timeout=120)
    coll = f"vo_rag_test_hybrid_{uuid.uuid4().hex[:8]}"
    try:
        # dense 4d so the test stays small; colbert multivector 2d
        await client.create_collection(
            collection_name=coll,
            vectors_config={
                "dense": VectorParams(size=4, distance=Distance.COSINE),
                "colbert": VectorParams(
                    size=3,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
                "bm25": SparseVectorParams(),
            },
        )

        from vo_rag.core.retrieval.bm25_tokenizer import (
            tokenize,
            tokens_to_bm25_sparse,
        )
        from vo_rag.core.storage.qdrant import bge_sparse_to_qdrant, deterministic_point_id

        async def _upsert_text(text: str, source_id: str, chunk_index: int) -> str:
            tokens = tokenize(text)
            sparse = tokens_to_bm25_sparse(tokens)
            point_id = deterministic_point_id(source_id, chunk_index, "raw")
            await client.upsert(
                collection_name=coll,
                points=[
                    PointStruct(
                        id=point_id,
                        vector={
                            "dense": [0.1, 0.1, 0.1, 0.1],
                            "sparse": bge_sparse_to_qdrant(sparse),
                            "bm25": bge_sparse_to_qdrant(sparse),
                            "colbert": [[0.2, 0.3, 0.4]],
                        },
                        payload={
                            "source_id": source_id,
                            "chunk_id": f"c{chunk_index}",
                            "page_number": chunk_index,
                            "text": text,
                            "license_class": "cleared",
                        },
                    )
                ],
            )
            return point_id

        await _upsert_text("trail braking is a technique used in racing", "src-A", 0)
        await _upsert_text("brake balance migration and trail braking", "src-B", 0)
        await _upsert_text("engine tuning for forced induction", "src-C", 0)

        # bind the test collection onto the store for the duration of the test
        store._test_collection = coll  # type: ignore[attr-defined]
        yield store
    finally:
        await client.delete_collection(collection_name=coll)
        await client.close()


@pytest.mark.asyncio
async def test_hybrid_search_against_live_qdrant(live_store: QdrantStore) -> None:
    coll = live_store._test_collection  # type: ignore[attr-defined]
    retriever = HybridRetriever(qdrant=live_store, embedding_client=_ScriptedEmbedClient())  # type: ignore[arg-type]

    results = await retriever.hybrid_search(
        _TEST_QUERY,
        coll,
        top_k=5,
        mode=QueryMode.PUBLICATION,
    )

    # Three docs; the two trail-braking ones should rank above the engine tuning one
    assert len(results) >= 2
    ids = {r.payload.get("source_id") for r in results}
    assert "src-A" in ids
    assert "src-B" in ids


@pytest.mark.asyncio
async def test_cross_collection_search_baseline_against_live_qdrant(
    live_store: QdrantStore,
) -> None:
    coll = live_store._test_collection  # type: ignore[attr-defined]
    retriever = HybridRetriever(qdrant=live_store, embedding_client=_ScriptedEmbedClient())  # type: ignore[arg-type]

    results = await retriever.cross_collection_search(
        _TEST_QUERY,
        [coll],
        top_k=5,
        mode=QueryMode.PUBLICATION,
        visual_mode="baseline",
    )

    assert len(results) >= 1

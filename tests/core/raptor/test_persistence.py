from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import pytest
from qdrant_client import AsyncQdrantClient

from vo_rag.core.raptor.persistence import (
    RAPTOR_COLLECTION,
    default_m_for,
    ensure_raptor_collection,
    expand_cluster_node,
    get_children,
    get_cluster_node,
    upsert_cluster_node,
)
from vo_rag.core.raptor.schemas import ClusterLevel, ClusterNode, cluster_node_id_for
from vo_rag.core.storage.qdrant import QdrantStore


class InMemoryQdrantStore:
    def __init__(self) -> None:
        self._points: dict[str, dict[str, Any]] = {}
        self._collections: set[str] = set()
        self._payload_indexes: dict[str, set[str]] = {}
        self._create_collection_calls: list[str] = []

    async def upsert(self, collection_name: str, points: list[Any], wait: bool = True) -> None:
        for point in points:
            self._points[str(point.id)] = {
                "id": point.id,
                "vector": point.vector,
                "payload": point.payload,
            }

    async def retrieve(
        self, collection_name: str, ids: list[Any], with_payload: bool = True
    ) -> list[Any]:
        from qdrant_client.models import ScoredPoint

        results: list[Any] = []
        for pid in ids:
            key = str(pid)
            if key in self._points:
                pt = self._points[key]
                results.append(
                    ScoredPoint(
                        id=pt["id"],
                        version=0,
                        score=1.0,
                        vector=pt["vector"],
                        payload=pt["payload"],
                    )
                )
        return results

    async def scroll(
        self,
        collection_name: str,
        scroll_filter: Any = None,
        limit: int = 1000,
        with_payload: bool = True,
    ) -> tuple[list[Any], None]:
        from qdrant_client.models import ScoredPoint

        matched: list[Any] = []
        for _key, pt in self._points.items():
            if scroll_filter is None:
                matched.append(
                    ScoredPoint(
                        id=pt["id"],
                        version=0,
                        score=1.0,
                        vector=pt["vector"],
                        payload=pt["payload"],
                    )
                )
            else:
                payload = pt["payload"]
                must_conditions = scroll_filter.must
                match = True
                for cond in must_conditions:
                    field_name = cond.key
                    match_val = cond.match.value
                    if payload.get(field_name) != match_val:
                        match = False
                        break
                if match:
                    matched.append(
                        ScoredPoint(
                            id=pt["id"],
                            version=0,
                            score=1.0,
                            vector=pt["vector"],
                            payload=pt["payload"],
                        )
                    )
        return (matched[:limit], None)

    async def get_collections(self) -> Any:
        class CollectionsResponse:
            def __init__(self, collections: list[Any]) -> None:
                self.collections = collections

        class CollectionInfo:
            def __init__(self, name: str) -> None:
                self.name = name

        return CollectionsResponse([CollectionInfo(name) for name in self._collections])

    async def create_collection(
        self,
        collection_name: str,
        vectors_config: Any,
        sparse_vectors_config: Any = None,
    ) -> None:
        self._create_collection_calls.append(collection_name)
        self._collections.add(collection_name)

    async def create_payload_index(
        self, collection_name: str, field_name: str, field_schema: Any
    ) -> None:
        if collection_name not in self._payload_indexes:
            self._payload_indexes[collection_name] = set()
        self._payload_indexes[collection_name].add(field_name)


class InMemoryEmbeddingClient:
    def __init__(self) -> None:
        self.call_count = 0

    async def embed(
        self,
        texts: list[str],
        return_dense: bool = False,
        return_sparse: bool = False,
        return_colbert: bool = False,
    ) -> dict[str, Any]:
        self.call_count += 1
        return {
            "dense": [self._fake_dense()],
            "sparse": [self._fake_sparse()],
            "colbert": [self._fake_colbert()],
        }

    async def aclose(self) -> None:
        pass

    def _fake_dense(self) -> list[float]:
        import random

        random.seed(42)
        return [random.random() for _ in range(1024)]

    def _fake_sparse(self) -> dict[str, list[int] | list[float]]:
        # Match EmbeddingClient's actual JSON response, not an attribute object.
        return {"indices": list(range(0, 50)), "values": [0.1] * 50}

    def _fake_colbert(self) -> list[list[float]]:
        import random

        random.seed(42)
        return [[random.random() for _ in range(1024)] for _ in range(8)]


class InMemoryQdrantStoreForSearch:
    def __init__(self, points_by_chunk_id: dict[str, dict[str, Any]]) -> None:
        self._points_by_chunk_id = points_by_chunk_id

    async def search_hybrid(
        self,
        collection: str,
        query_dense: list[float],
        query_sparse: dict[int, float],
        query_bm25: dict[int, float],
        query_colbert: list[list[float]],
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        prefetch_depth: int = 100,
    ) -> list[Any]:
        from qdrant_client.models import ScoredPoint

        chunk_ids_list = filters.get("chunk_id", []) if filters else []
        results = []
        for chunk_id in chunk_ids_list[:top_k]:
            if chunk_id in self._points_by_chunk_id:
                pt_data = self._points_by_chunk_id[chunk_id]
                results.append(
                    ScoredPoint(
                        id=chunk_id,
                        version=0,
                        score=0.9,
                        vector={},
                        payload=pt_data,
                    )
                )
        return results


class TestDefaultMFor:
    def test_default_m_for_level_0_returns_8(self) -> None:
        assert default_m_for(ClusterLevel.LEVEL_0) == 8

    def test_default_m_for_level_1_returns_15(self) -> None:
        assert default_m_for(ClusterLevel.LEVEL_1) == 15

    def test_default_m_for_level_2_returns_25(self) -> None:
        assert default_m_for(ClusterLevel.LEVEL_2) == 25


class TestClusterNodeIdFor:
    def test_cluster_node_id_for_is_deterministic(self) -> None:
        chunk_ids_a = ["c1", "c2", "c3"]
        chunk_ids_b = ["c3", "c1", "c2"]
        result_a = cluster_node_id_for(
            book_id="book-001",
            level=ClusterLevel.LEVEL_0,
            member_chunk_ids=chunk_ids_a,
            prompt_hash="hash-v1",
        )
        result_b = cluster_node_id_for(
            book_id="book-001",
            level=ClusterLevel.LEVEL_0,
            member_chunk_ids=chunk_ids_b,
            prompt_hash="hash-v1",
        )
        assert result_a == result_b

    def test_cluster_node_id_for_different_orderings_produce_same_id(self) -> None:
        ids_1 = ["chunk-a", "chunk-b"]
        ids_2 = ["chunk-b", "chunk-a"]
        id_1 = cluster_node_id_for(
            book_id="book-x",
            level=ClusterLevel.LEVEL_1,
            member_chunk_ids=ids_1,
            prompt_hash="v1",
        )
        id_2 = cluster_node_id_for(
            book_id="book-x",
            level=ClusterLevel.LEVEL_1,
            member_chunk_ids=ids_2,
            prompt_hash="v1",
        )
        assert id_1 == id_2

    def test_cluster_node_id_for_different_levels_produce_different_ids(self) -> None:
        ids = ["c1", "c2", "c3"]
        id_l0 = cluster_node_id_for(
            book_id="book-001",
            level=ClusterLevel.LEVEL_0,
            member_chunk_ids=ids,
            prompt_hash="hash-v1",
        )
        id_l1 = cluster_node_id_for(
            book_id="book-001",
            level=ClusterLevel.LEVEL_1,
            member_chunk_ids=ids,
            prompt_hash="hash-v1",
        )
        assert id_l0 != id_l1

    def test_cluster_node_id_for_deterministic_over_sorted_ids(self) -> None:
        """cluster_node_id_for is deterministic over sorted source chunk IDs."""
        ids_shuffled = ["z", "a", "m", "b", "c"]
        ids_sorted = sorted(ids_shuffled)
        id_shuffled = cluster_node_id_for(
            book_id="book-001",
            level=ClusterLevel.LEVEL_0,
            member_chunk_ids=ids_shuffled,
            prompt_hash="hash-v1",
        )
        id_sorted = cluster_node_id_for(
            book_id="book-001",
            level=ClusterLevel.LEVEL_0,
            member_chunk_ids=ids_sorted,
            prompt_hash="hash-v1",
        )
        assert id_shuffled == id_sorted


class TestEnsureRaptorCollection:
    async def test_ensure_raptor_collection_creates_with_bge_m3_config(self) -> None:
        store = InMemoryQdrantStore()
        created = await ensure_raptor_collection(cast(AsyncQdrantClient, store))
        assert created is True
        assert RAPTOR_COLLECTION in store._collections

    async def test_ensure_raptor_collection_creates_explicit_suffixed_collection(self) -> None:
        store = InMemoryQdrantStore()
        created = await ensure_raptor_collection(
            cast(AsyncQdrantClient, store), collection="vo_rag_books_raptor_v1__ab14b"
        )
        assert created is True
        assert store._create_collection_calls == ["vo_rag_books_raptor_v1__ab14b"]

    async def test_ensure_raptor_collection_defaults_to_canonical_collection(self) -> None:
        store = InMemoryQdrantStore()
        created = await ensure_raptor_collection(cast(AsyncQdrantClient, store))
        assert created is True
        assert store._create_collection_calls == [RAPTOR_COLLECTION]

    async def test_ensure_raptor_collection_idempotent_when_exists(self) -> None:
        store = InMemoryQdrantStore()
        store._collections.add(RAPTOR_COLLECTION)
        created = await ensure_raptor_collection(cast(AsyncQdrantClient, store))
        assert created is False


class TestUpsertClusterNode:
    async def test_upsert_cluster_node_uses_node_id_as_point_id(self) -> None:
        store = InMemoryQdrantStore()
        node = ClusterNode(
            node_id="node-test-123",
            book_id="book-001",
            level=ClusterLevel.LEVEL_0,
            cluster_index_within_level=0,
            parent_node_id=None,
            child_node_ids=[],
            source_chunk_ids=["c1", "c2"],
            direct_child_chunk_ids=["c1", "c2"],
            evidence_spans=[],
            cluster_centroid_umap=[],
            gmm_cluster_probability=1.0,
            summary_text="Test summary",
            summary_model="test-model",
            prompt_hash="hash123",
            prompt_version="v1",
            embedding_model="bge-m3",
            embedding_version="1.0",
            umap_random_state=42,
            gmm_random_state=42,
            gmm_cluster_selection_criterion="bic",
            created_at=datetime.now(),
            source_collection_version="v1",
        )

        class FakeQdrantStoreWrapper:
            def __init__(self, client: InMemoryQdrantStore) -> None:
                self._client = client

        wrapper = FakeQdrantStoreWrapper(store)

        await upsert_cluster_node(
            node,
            dense_embedding=[0.1] * 1024,
            sparse_embedding={0: 0.1, 1: 0.2},
            colbert_embedding=[[0.1] * 1024],
            store=cast(QdrantStore, wrapper),
        )

        assert "node-test-123" in store._points


class TestGetClusterNode:
    async def test_get_cluster_node_returns_none_when_absent(self) -> None:
        store = InMemoryQdrantStore()

        class FakeWrapper:
            _client: InMemoryQdrantStore

        wrapper = FakeWrapper()
        wrapper._client = store
        result = await get_cluster_node(
            "nonexistent-id",
            store=cast(QdrantStore, wrapper),
        )
        assert result is None

    async def test_get_cluster_node_returns_node_when_present(self) -> None:
        store = InMemoryQdrantStore()
        node_payload = {
            "node_id": "node-present-456",
            "book_id": "book-002",
            "level": "level_0",
            "cluster_index_within_level": 1,
            "parent_node_id": None,
            "child_node_ids": [],
            "source_chunk_ids": ["c1"],
            "direct_child_chunk_ids": ["c1"],
            "evidence_spans": [],
            "cluster_centroid_umap": [],
            "gmm_cluster_probability": 1.0,
            "summary_text": "A summary",
            "summary_model": "model",
            "prompt_hash": "h",
            "prompt_version": "v1",
            "embedding_model": "bge-m3",
            "embedding_version": "1.0",
            "umap_random_state": 42,
            "gmm_random_state": 42,
            "gmm_cluster_selection_criterion": "bic",
            "created_at": datetime.now().isoformat(),
            "source_collection_version": "v1",
        }
        store._points["node-present-456"] = {
            "id": "node-present-456",
            "vector": {},
            "payload": node_payload,
        }

        class FakeWrapper:
            _client: InMemoryQdrantStore

        wrapper = FakeWrapper()
        wrapper._client = store
        result = await get_cluster_node("node-present-456", store=cast(QdrantStore, wrapper))
        assert result is not None
        assert result.node_id == "node-present-456"


class TestGetChildren:
    async def test_get_children_filters_by_parent_node_id(self) -> None:
        store = InMemoryQdrantStore()

        for i in range(3):
            store._points[f"child-{i}"] = {
                "id": f"child-{i}",
                "vector": {},
                "payload": {
                    "node_id": f"child-{i}",
                    "book_id": "book-001",
                    "level": "level_0",
                    "cluster_index_within_level": i,
                    "parent_node_id": "parent-node-abc",
                    "child_node_ids": [],
                    "source_chunk_ids": [],
                    "direct_child_chunk_ids": [],
                    "evidence_spans": [],
                    "cluster_centroid_umap": [],
                    "gmm_cluster_probability": 1.0,
                    "summary_text": "",
                    "summary_model": "",
                    "prompt_hash": "",
                    "prompt_version": "v1",
                    "embedding_model": "",
                    "embedding_version": "",
                    "umap_random_state": 0,
                    "gmm_random_state": 0,
                    "gmm_cluster_selection_criterion": "bic",
                    "created_at": datetime.now().isoformat(),
                    "source_collection_version": "v1",
                },
            }

        store._points["other-parent-child"] = {
            "id": "other-parent-child",
            "vector": {},
            "payload": {
                "node_id": "other-parent-child",
                "book_id": "book-001",
                "level": "level_0",
                "cluster_index_within_level": 99,
                "parent_node_id": "other-parent",
                "child_node_ids": [],
                "source_chunk_ids": [],
                "direct_child_chunk_ids": [],
                "evidence_spans": [],
                "cluster_centroid_umap": [],
                "gmm_cluster_probability": 1.0,
                "summary_text": "",
                "summary_model": "",
                "prompt_hash": "",
                "prompt_version": "v1",
                "embedding_model": "",
                "embedding_version": "",
                "umap_random_state": 0,
                "gmm_random_state": 0,
                "gmm_cluster_selection_criterion": "bic",
                "created_at": datetime.now().isoformat(),
                "source_collection_version": "v1",
            },
        }

        class FakeWrapper:
            _client: InMemoryQdrantStore

        wrapper = FakeWrapper()
        wrapper._client = store
        children = await get_children("parent-node-abc", store=cast(QdrantStore, wrapper))
        assert len(children) == 3
        for child in children:
            assert child.parent_node_id == "parent-node-abc"


class TestExpandClusterNode:
    async def test_expand_cluster_node_returns_only_source_chunk_ids(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chunk_data = {
            "c1": {
                "chunk_id": "c1",
                "chunk_type": "text",
                "document_id": "doc-001",
                "source_id": "src-001",
                "page_number": 1,
                "page_range": "1",
                "section_path": "Ch1",
                "section_title": "Chapter 1",
                "content": "Content about brakes",
                "embedding_text": "Ch1\n\nContent about brakes",
                "token_count": 10,
                "has_image": False,
            },
            "c2": {
                "chunk_id": "c2",
                "chunk_type": "text",
                "document_id": "doc-001",
                "source_id": "src-001",
                "page_number": 2,
                "page_range": "2",
                "section_path": "Ch1",
                "section_title": "Chapter 1",
                "content": "Content about tires",
                "embedding_text": "Ch1\n\nContent about tires",
                "token_count": 10,
                "has_image": False,
            },
            "c3": {
                "chunk_id": "c3",
                "chunk_type": "text",
                "document_id": "doc-001",
                "source_id": "src-001",
                "page_number": 3,
                "page_range": "3",
                "section_path": "Ch1",
                "section_title": "Chapter 1",
                "content": "Content about aerodynamics",
                "embedding_text": "Ch1\n\nContent about aerodynamics",
                "token_count": 10,
                "has_image": False,
            },
        }

        node = ClusterNode(
            node_id="node-expand-test",
            book_id="book-001",
            level=ClusterLevel.LEVEL_0,
            cluster_index_within_level=0,
            parent_node_id=None,
            child_node_ids=[],
            source_chunk_ids=["c1", "c2"],
            direct_child_chunk_ids=["c1", "c2"],
            evidence_spans=[],
            cluster_centroid_umap=[],
            gmm_cluster_probability=1.0,
            summary_text="Test",
            summary_model="test",
            prompt_hash="h",
            prompt_version="v1",
            embedding_model="bge-m3",
            embedding_version="1.0",
            umap_random_state=42,
            gmm_random_state=42,
            gmm_cluster_selection_criterion="bic",
            created_at=datetime.now(),
            source_collection_version="v1",
        )

        search_store = InMemoryQdrantStoreForSearch(chunk_data)
        embed_client = InMemoryEmbeddingClient()

        class FakeWrapper:
            _client: InMemoryQdrantStoreForSearch

            async def search_hybrid(self, **kwargs):
                return await search_store.search_hybrid(**kwargs)

        class FakeSettings:
            class embedding:
                query_url = "http://fake"
                timeout_s = 10

        import vo_rag.core.config.settings as config_module
        import vo_rag.core.embedding.client as embed_module

        monkeypatch.setattr(embed_module, "EmbeddingClient", lambda *a, **kw: embed_client)
        monkeypatch.setattr(config_module, "get_settings", lambda: FakeSettings())

        results = await expand_cluster_node(
            node=node,
            query="brakes",
            m=5,
            qdrant=cast(QdrantStore, FakeWrapper()),
            raw_collection="vo_rag_books_raw_v1",
        )
        result_ids = {r.chunk_id for r in results}
        assert result_ids.issubset({"c1", "c2"})

    async def test_expand_cluster_node_filters_extras_post_hoc(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        node_source_ids = {"x1", "x2"}
        all_chunk_data = {
            "x1": {
                "chunk_id": "x1",
                "chunk_type": "text",
                "document_id": "doc-x",
                "source_id": "src-x",
                "page_number": 1,
                "page_range": "1",
                "section_path": "X",
                "section_title": "Section X",
                "content": "Content x1",
                "embedding_text": "X\n\nContent x1",
                "token_count": 5,
                "has_image": False,
            },
            "x2": {
                "chunk_id": "x2",
                "chunk_type": "text",
                "document_id": "doc-x",
                "source_id": "src-x",
                "page_number": 2,
                "page_range": "2",
                "section_path": "X",
                "section_title": "Section X",
                "content": "Content x2",
                "embedding_text": "X\n\nContent x2",
                "token_count": 5,
                "has_image": False,
            },
            "outside-id": {
                "chunk_id": "outside-id",
                "chunk_type": "text",
                "document_id": "doc-y",
                "source_id": "src-y",
                "page_number": 1,
                "page_range": "1",
                "section_path": "Y",
                "section_title": "Section Y",
                "content": "Content outside",
                "embedding_text": "Y\n\nContent outside",
                "token_count": 5,
                "has_image": False,
            },
        }

        node = ClusterNode(
            node_id="node-filter-test",
            book_id="book-x",
            level=ClusterLevel.LEVEL_1,
            cluster_index_within_level=0,
            parent_node_id=None,
            child_node_ids=[],
            source_chunk_ids=sorted(node_source_ids),
            direct_child_chunk_ids=sorted(node_source_ids),
            evidence_spans=[],
            cluster_centroid_umap=[],
            gmm_cluster_probability=1.0,
            summary_text="Test",
            summary_model="test",
            prompt_hash="h",
            prompt_version="v1",
            embedding_model="bge-m3",
            embedding_version="1.0",
            umap_random_state=42,
            gmm_random_state=42,
            gmm_cluster_selection_criterion="bic",
            created_at=datetime.now(),
            source_collection_version="v1",
        )

        search_store = InMemoryQdrantStoreForSearch(all_chunk_data)
        embed_client = InMemoryEmbeddingClient()

        class FakeWrapper:
            _client: InMemoryQdrantStoreForSearch

            async def search_hybrid(self, **kwargs):
                return await search_store.search_hybrid(**kwargs)

        class FakeSettings:
            class embedding:
                query_url = "http://fake"
                timeout_s = 10

        import vo_rag.core.config.settings as config_module
        import vo_rag.core.embedding.client as embed_module

        monkeypatch.setattr(embed_module, "EmbeddingClient", lambda *a, **kw: embed_client)
        monkeypatch.setattr(config_module, "get_settings", lambda: FakeSettings())

        results = await expand_cluster_node(
            node=node,
            query="test query",
            m=10,
            qdrant=cast(QdrantStore, FakeWrapper()),
            raw_collection="vo_rag_books_raw_v1",
        )
        result_ids = {r.chunk_id for r in results}
        assert "outside-id" not in result_ids
        assert result_ids.issubset(node_source_ids)

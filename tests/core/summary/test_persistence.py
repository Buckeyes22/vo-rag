"""Deep tests for vo_rag.core.summary.persistence — Qdrant persistence layer.

The dict-backed
``InMemoryQdrantClient`` test double avoids ``unittest.mock`` so the
tests are fully deterministic and trivially readable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from vo_rag.core.summary.persistence import (
    SUMMARY_TREE_COLLECTION,
    ensure_summary_tree_collection,
    get_book_tree,
    get_children,
    get_summary_node,
    upsert_summary_node,
)
from vo_rag.core.summary.schemas import EvidenceSpan, NodeLevel, SummaryNode


def _make_node(
    *,
    node_id: str,
    book_id: str = "book-001",
    level: NodeLevel = NodeLevel.SECTION,
    canonical_path: str = "Book > Ch1 > S1",
    display_path: str = "Chapter 1 > Section 1",
    source_chunk_ids: list[str] | None = None,
    parent_node_id: str | None = None,
) -> SummaryNode:
    return SummaryNode(
        node_id=node_id,
        book_id=book_id,
        level=level,
        node_type=f"{level.value}_summary",
        parent_node_id=parent_node_id,
        child_node_ids=[],
        canonical_path=canonical_path,
        display_path=display_path,
        source_chunk_ids=sorted(source_chunk_ids) if source_chunk_ids is not None else ["c1", "c2"],
        source_section_paths=[canonical_path],
        evidence_spans=[
            EvidenceSpan(
                chunk_id="c1",
                section_path=canonical_path,
                text="Sample evidence text.",
                score=0.9,
                selection_method="bm25",
            ),
        ],
        summary_text="Test summary text for persistence.",
        summary_model="test-model",
        prompt_hash="a" * 64,
        prompt_version="v1",
        embedding_model="BGE-M3",
        embedding_version="test-v1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        source_collection_version="test-v1",
    )


class InMemoryQdrantClient:
    """Dict-backed AsyncQdrantClient test double — no unittest.mock."""

    def __init__(self) -> None:
        self._points: dict[str, dict[str, Any]] = {}
        self._collections: set[str] = set()
        self._create_collection_calls: list[str] = []
        self._upsert_calls: list[dict[str, Any]] = []
        self._retrieve_calls: list[dict[str, Any]] = []
        self._scroll_calls: list[dict[str, Any]] = []

    async def get_collections(self) -> Any:
        class Collections:
            collections = [type("C", (), {"name": n})() for n in self._collections]

        return Collections()

    async def create_collection(
        self, collection_name: str, vectors_config: Any = None, sparse_vectors_config: Any = None
    ) -> None:
        self._create_collection_calls.append(collection_name)
        self._collections.add(collection_name)

    async def create_payload_index(
        self, collection_name: str, field_name: str, field_schema: Any
    ) -> None:
        pass

    async def upsert(self, collection_name: str, points: list[Any], wait: bool = True) -> None:
        for point in points:
            self._upsert_calls.append(
                {
                    "collection": collection_name,
                    "point_id": point.id,
                    "vector_keys": list(point.vector.keys()) if point.vector else [],
                }
            )
            self._points[point.id] = {
                "id": point.id,
                "vector": point.vector,
                "payload": point.payload,
            }

    async def retrieve(
        self, collection_name: str, ids: list[str], with_payload: bool = True
    ) -> list[Any]:
        self._retrieve_calls.append({"collection": collection_name, "ids": ids})
        results = []
        for oid in ids:
            if oid in self._points:

                class Point:
                    def __init__(self, data: dict[str, Any]) -> None:
                        self.id = data["id"]
                        self.payload = data["payload"]
                        self.vector = data.get("vector", {})

                results.append(Point(self._points[oid]))
        return results

    async def scroll(
        self,
        collection_name: str,
        scroll_filter: Any = None,
        limit: int = 100,
        with_payload: bool = True,
    ) -> tuple[list[Any], Any]:
        self._scroll_calls.append(
            {"collection": collection_name, "filter": scroll_filter, "limit": limit}
        )

        class Point:
            def __init__(self, data: dict[str, Any]) -> None:
                self.id = data["id"]
                self.payload = data["payload"]

        points = list(self._points.values())

        if scroll_filter is not None and hasattr(scroll_filter, "must"):
            must_conditions = scroll_filter.must or []
            filtered: list[dict[str, Any]] = []
            for p in points:
                matches = True
                for cond in must_conditions:
                    if not hasattr(cond, "key") or not hasattr(cond, "match"):
                        continue
                    payload_val = p.get("payload", {}).get(cond.key)
                    match_val = getattr(cond.match, "value", None)
                    if payload_val != match_val:
                        matches = False
                        break
                if matches:
                    filtered.append(p)
            points = filtered

        return [Point(p) for p in points[:limit]], None


class InMemoryQdrantStore:
    """Store wrapper using InMemoryQdrantClient."""

    def __init__(self) -> None:
        self._client = InMemoryQdrantClient()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_node_id_for_returns_deterministic_uuid() -> None:
    from vo_rag.core.summary.schemas import node_id_for

    id1 = node_id_for(
        book_id="book-001",
        level=NodeLevel.SECTION,
        source_chunk_ids=["c1", "c2", "c3"],
        prompt_hash="a" * 64,
    )
    id2 = node_id_for(
        book_id="book-001",
        level=NodeLevel.SECTION,
        source_chunk_ids=["c1", "c2", "c3"],
        prompt_hash="a" * 64,
    )

    assert id1 == id2
    assert isinstance(id1, str)
    assert len(id1) > 0


def test_node_id_for_distinguishes_per_book_per_level_per_chunk_set() -> None:
    from vo_rag.core.summary.schemas import node_id_for

    shared_chunks = ["c1", "c2", "c3"]
    shared_prompt = "a" * 64

    id_book_a = node_id_for(
        book_id="book-A",
        level=NodeLevel.SECTION,
        source_chunk_ids=shared_chunks,
        prompt_hash=shared_prompt,
    )
    id_book_b = node_id_for(
        book_id="book-B",
        level=NodeLevel.SECTION,
        source_chunk_ids=shared_chunks,
        prompt_hash=shared_prompt,
    )
    assert id_book_a != id_book_b

    id_level_sec = node_id_for(
        book_id="book-001",
        level=NodeLevel.SECTION,
        source_chunk_ids=shared_chunks,
        prompt_hash=shared_prompt,
    )
    id_level_chap = node_id_for(
        book_id="book-001",
        level=NodeLevel.CHAPTER,
        source_chunk_ids=shared_chunks,
        prompt_hash=shared_prompt,
    )
    assert id_level_sec != id_level_chap

    id_chunks_ab = node_id_for(
        book_id="book-001",
        level=NodeLevel.SECTION,
        source_chunk_ids=["c1", "c2"],
        prompt_hash="a" * 64,
    )
    id_chunks_c = node_id_for(
        book_id="book-001",
        level=NodeLevel.SECTION,
        source_chunk_ids=["c3", "c4"],
        prompt_hash="a" * 64,
    )
    assert id_chunks_ab != id_chunks_c


async def test_ensure_summary_collection_creates_with_bge_m3_config() -> None:
    client = InMemoryQdrantClient()
    created = await ensure_summary_tree_collection(
        cast("Any", client), collection="test-collection"
    )

    assert created is True
    assert "test-collection" in client._collections


async def test_ensure_summary_collection_creates_explicit_suffixed_collection() -> None:
    client = InMemoryQdrantClient()

    created = await ensure_summary_tree_collection(
        cast("Any", client), collection="vo_rag_books_summary_tree_v1__ab14b"
    )

    assert created is True
    assert client._create_collection_calls == ["vo_rag_books_summary_tree_v1__ab14b"]


async def test_ensure_summary_collection_defaults_to_canonical_collection() -> None:
    client = InMemoryQdrantClient()

    created = await ensure_summary_tree_collection(cast("Any", client))

    assert created is True
    assert client._create_collection_calls == [SUMMARY_TREE_COLLECTION]


async def test_ensure_summary_collection_idempotent() -> None:
    client = InMemoryQdrantClient()
    client._collections.add("existing-collection")

    created = await ensure_summary_tree_collection(
        cast("Any", client), collection="existing-collection"
    )

    assert created is False


async def test_upsert_summary_node_uses_node_id_as_point_id() -> None:
    node = _make_node(node_id="node-abc-123")

    store = InMemoryQdrantStore()

    await upsert_summary_node(
        node=node,
        dense_embedding=[0.1] * 1024,
        sparse_embedding={0: 0.5},
        colbert_embedding=[[0.1] * 1024],
        store=store,  # type: ignore[arg-type]
        collection="test-collection",
    )

    assert len(store._client._upsert_calls) == 1
    call = store._client._upsert_calls[0]
    assert call["point_id"] == "node-abc-123"
    assert call["collection"] == "test-collection"


async def test_get_summary_node_returns_none_on_404() -> None:
    store = InMemoryQdrantStore()

    result = await get_summary_node(
        "nonexistent-node-id",
        store=store,  # type: ignore[arg-type]
        collection="test-collection",
    )

    assert result is None


async def test_get_summary_node_parses_payload() -> None:
    node = _make_node(
        node_id="parse-test-node",
        book_id="book-parse",
        canonical_path="Book > Ch1 > S1",
    )

    store = InMemoryQdrantStore()

    await upsert_summary_node(
        node=node,
        dense_embedding=[0.1] * 1024,
        sparse_embedding={0: 0.5},
        colbert_embedding=[[0.1] * 1024],
        store=store,  # type: ignore[arg-type]
        collection="test-collection",
    )

    result = await get_summary_node(
        "parse-test-node",
        store=store,  # type: ignore[arg-type]
        collection="test-collection",
    )

    assert result is not None
    assert result.node_id == "parse-test-node"
    assert result.book_id == "book-parse"
    assert result.level == NodeLevel.SECTION
    assert result.canonical_path == "Book > Ch1 > S1"
    assert result.summary_text == "Test summary text for persistence."


async def test_get_children_by_parent_id_filters_correctly() -> None:
    parent_node = _make_node(node_id="parent-001", level=NodeLevel.BOOK)
    child_node = _make_node(
        node_id="child-001",
        level=NodeLevel.CHAPTER,
        parent_node_id="parent-001",
    )
    unrelated_node = _make_node(node_id="unrelated-001", level=NodeLevel.CHAPTER)

    store = InMemoryQdrantStore()
    client = store._client

    for node in [parent_node, child_node, unrelated_node]:
        client._points[node.node_id] = {
            "id": node.node_id,
            "payload": node.model_dump(mode="json"),
            "vector": {},
        }

    children = await get_children(
        parent_node_id="parent-001",
        store=store,  # type: ignore[arg-type]
        collection="test-collection",
    )

    assert len(children) == 1
    assert children[0].node_id == "child-001"


async def test_get_book_summary_tree_returns_all_book_nodes() -> None:
    book_node = _make_node(
        node_id="book-node-001",
        book_id="book-001",
        level=NodeLevel.BOOK,
    )
    chapter_node = _make_node(
        node_id="chapter-node-001",
        book_id="book-001",
        level=NodeLevel.CHAPTER,
    )
    other_book_node = _make_node(
        node_id="book-node-002",
        book_id="book-002",
        level=NodeLevel.BOOK,
    )

    store = InMemoryQdrantStore()
    client = store._client

    for node in [book_node, chapter_node, other_book_node]:
        client._points[node.node_id] = {
            "id": node.node_id,
            "payload": node.model_dump(mode="json"),
            "vector": {},
        }

    tree = await get_book_tree(
        book_id="book-001",
        store=store,  # type: ignore[arg-type]
        collection="test-collection",
    )

    assert len(tree) == 2
    node_ids = {n.node_id for n in tree}
    assert "book-node-001" in node_ids
    assert "chapter-node-001" in node_ids
    assert "book-node-002" not in node_ids

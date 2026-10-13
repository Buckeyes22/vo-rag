"""Regression cases for writer-independent passage and retrieval contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
from qdrant_client.models import FusionQuery, ScoredPoint

from vo_rag.core.reranking.client import RerankerClient
from vo_rag.core.retrieval.execution import execute_retrieval
from vo_rag.core.retrieval.fusion import rrf_merge_and_dedupe
from vo_rag.core.retrieval.search import HybridRetriever, SearchResult
from vo_rag.core.storage.passage_contract import passage_payload
from vo_rag.core.storage.qdrant import QdrantStore
from vo_rag.ops.passage_migration import migrate


def hit(identity: str, source: str = "long") -> SearchResult:
    return SearchResult(
        id=identity,
        collection="c",
        score=1.0,
        payload={"source_id": source, "chunk_id": identity, "content": identity},
    )


def test_legacy_sections_on_one_page_survive_but_identical_points_collapse() -> None:
    results = [
        SearchResult(
            id=str(i), score=1, collection="c", payload={"source_id": "page", "page_number": 0}
        )
        for i in range(14)
    ]
    merged = rrf_merge_and_dedupe([results, results], top_k=30)
    assert len(merged) == 14


def test_writer_preserves_parent_identity_and_declares_substitute() -> None:
    payload = passage_payload(
        {"parent_chunk_id": "raw", "chunk_id": "sterile"},
        point_id="point",
        source_id="book",
        dense=[1],
        colbert=[[1]],
    )
    assert payload["chunk_id"] == "sterile" and payload["parent_chunk_id"] == "raw"
    assert payload["colbert_representation"] == "dense_substitute"


async def test_candidate_overfetch_selects_diverse_pool_before_rerank() -> None:
    requested: list[int] = []
    selected: list[SearchResult] = []

    class Retriever:
        async def cross_collection_search(self, **kwargs: Any) -> list[SearchResult]:
            requested.append(kwargs["top_k"])
            return [hit(str(i)) for i in range(80)] + [hit("answer", "other")]

    def rerank(query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        selected.extend(candidates)
        return sorted(candidates, key=lambda candidate: candidate.id != "answer")

    result = await execute_retrieval(
        retriever=cast(Any, Retriever()),
        requested_collections=["c"],
        query="ordinary question",
        top_k=8,
        rerank_fn=rerank,
    )
    assert requested == [100] and len(selected) == 40
    assert result[0].id == "answer"
    assert len({candidate.payload["source_id"] for candidate in selected}) == 2


async def test_candidate_bound_does_not_reduce_large_requested_output() -> None:
    class Retriever:
        async def cross_collection_search(self, **kwargs: Any) -> list[SearchResult]:
            return [hit(str(i), str(i)) for i in range(kwargs["top_k"])]

    result = await execute_retrieval(
        retriever=cast(Any, Retriever()),
        requested_collections=["c"],
        query="q",
        top_k=200,
        rerank_fn=lambda query, hits: hits,
    )
    assert len(result) == 200


async def test_modality_preferences_backfill_text_and_visual_shortfalls() -> None:
    class Embed:
        async def embed(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"dense": [[1]], "sparse": [{"indices": [], "values": []}], "colbert": [[[1]]]}

    class Store:
        async def search_hybrid(self, **kwargs: Any) -> list[ScoredPoint]:
            filt = kwargs["filters"].model_dump_json()
            visual = "figure" in filt
            rows = (
                []
                if visual
                else [
                    ScoredPoint(
                        id=i, score=1, version=0, payload={"chunk_id": str(i), "source_id": "s"}
                    )
                    for i in range(8)
                ]
            )
            return rows

    result = await HybridRetriever(cast(Any, Store()), Embed()).cross_collection_search(
        "diagram explanation", ["vo_rag_brain_v1"], top_k=8
    )
    assert len(result) == 8


async def test_store_fuses_compatible_vectors_instead_of_dense_substitute_colbert() -> None:
    captured: dict[str, Any] = {}

    class Client:
        async def query_points(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(points=[])

    store = object.__new__(QdrantStore)
    store._client = Client()
    await store.search_hybrid("c", [1], {2: 0.5}, {3: 1.0}, [[9]])
    assert isinstance(captured["query"], FusionQuery)
    assert "using" not in captured
    assert [item.using for item in captured["prefetch"]] == ["dense", "sparse", "bm25"]


async def test_rerank_batches_bytes_recovers_413_and_preserves_payloads() -> None:
    sizes: list[int] = []
    rows = [hit(str(i), str(i)) for i in range(45)]
    for row in rows:
        row.payload["content"] = "Unicode Ω " * 1000
        row.payload["private_metadata"] = "never sent"

    def handler(request: httpx.Request) -> httpx.Response:
        sizes.append(len(request.content))
        body = json.loads(request.content)
        assert all(not row["payload"] for row in body["candidates"])
        if len(body["candidates"]) > 2:
            return httpx.Response(413)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"id": row["id"], "score": float(row["id"])} for row in body["candidates"]
                ]
            },
        )

    client = RerankerClient("http://fixture")
    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x")
    try:
        results = await client.rerank("q", rows, top_k=8)
    finally:
        await client.aclose()
    assert max(sizes) <= 48_000
    assert [row.id for row in results] == [str(i) for i in range(44, 36, -1)]
    assert results[0].payload == rows[-1].payload


async def test_owner_migration_is_idempotent_and_checkpoints_after_ack(tmp_path: Path) -> None:
    payloads: dict[str, dict[str, Any]] = {"a": {"summary_text": "useful passage", "book_id": "b"}}
    calls: list[Any] = []

    class Client:
        async def scroll(self, **kwargs: Any) -> Any:
            return [SimpleNamespace(id="a", payload=payloads["a"])], None

        async def batch_update_points(self, **kwargs: Any) -> None:
            calls.extend(kwargs["update_operations"])
            payloads["a"].update(kwargs["update_operations"][-1].set_payload.payload)

    store = object.__new__(QdrantStore)
    store._client = Client()
    state = tmp_path / "migration.json"
    result = await migrate(store, ["vo_rag_books_summary_tree_v1"], state, apply=True)
    assert result["collections"]["vo_rag_books_summary_tree_v1"]["changed"] == 1
    assert len(calls) == 2 and payloads["a"]["chunk_id"] == "a"
    await migrate(store, ["vo_rag_books_summary_tree_v1"], state, apply=True)
    assert len(calls) == 2


async def test_rerank_utf8_budget_reserves_allowed_query_bytes() -> None:
    candidate = hit("unicode")
    candidate.payload["content"] = "😀" * 8192
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sizes.append(len(request.content))
        return httpx.Response(200, json={"results": [{"id": "unicode", "score": -3.0}]})

    client = RerankerClient("http://x")
    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x")
    try:
        result = await client.rerank("😀" * 4000, [candidate])
        assert result[0].score == -3.0 and max(sizes) <= 48000
        assert result[0].payload["content"] == "😀" * 8192
    finally:
        await client.aclose()


async def test_rerank_rejects_incomplete_duplicate_foreign_and_nonfinite_responses() -> None:
    import pytest

    for response_rows in [
        [],
        [{"id": "foreign", "score": 1}],
        [{"id": "a", "score": None}],
        [{"id": "a", "score": "NaN"}],
        [{"id": "a", "score": "Infinity"}],
        [{"id": "a", "score": 1}, {"id": "a", "score": 1}],
    ]:

        def handler(request: httpx.Request, rows: Any = response_rows) -> httpx.Response:
            return httpx.Response(200, json={"results": rows})

        client = RerankerClient("http://x")
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://x"
        )
        try:
            with pytest.raises(ValueError, match="reranker returned"):
                await client.rerank("q", [hit("a")])
        finally:
            await client.aclose()


async def test_ranked_core_preserves_second_strong_passage_with_many_sources() -> None:
    selected: list[SearchResult] = []

    class Retriever:
        async def cross_collection_search(self, **kwargs: Any) -> list[SearchResult]:
            return [hit("intro", "target"), hit("answer", "target")] + [
                hit(str(i), str(i)) for i in range(80)
            ]

    def rerank(query: str, rows: list[SearchResult]) -> list[SearchResult]:
        selected.extend(rows)
        return rows

    await execute_retrieval(
        retriever=cast(Any, Retriever()),
        requested_collections=["c"],
        query="question",
        top_k=8,
        rerank_fn=rerank,
    )
    assert len(selected) == 40 and selected[1].id == "answer"


async def test_derived_migration_partial_vector_write_replays_before_contract_stamp(
    tmp_path: Path,
) -> None:
    import pytest

    payload: dict[str, Any] = {"book_id": "b", "summary_text": "actual source passage"}
    vectors: dict[str, Any] = {"dense": [1], "sparse": {999: 1}, "bm25": {999: 1}}

    class Client:
        def __init__(self) -> None:
            self.interrupt = True

        async def scroll(self, **kwargs: Any) -> Any:
            return [SimpleNamespace(id="a", payload=payload)], None

        async def batch_update_points(self, **kwargs: Any) -> None:
            operations = kwargs["update_operations"]
            vectors.update(operations[0].update_vectors.points[0].vector)
            if self.interrupt:
                self.interrupt = False
                raise OSError("lost response after first operation")
            payload.update(operations[1].set_payload.payload)

    store = object.__new__(QdrantStore)
    store._client = Client()
    state = tmp_path / "migration.json"
    with pytest.raises(OSError):
        await migrate(store, ["vo_rag_books_summary_tree_v1"], state, apply=True)
    assert (
        json.loads(state.read_text())["collections"]["vo_rag_books_summary_tree_v1"]["scanned"] == 0
    )
    assert "writer_contract_version" not in payload
    assert vectors["dense"] == [1] and vectors["sparse"] == {999: 1}
    assert vectors["bm25"] != {999: 1}
    await migrate(store, ["vo_rag_books_summary_tree_v1"], state, apply=True)
    assert payload["writer_contract_version"] == 1
    assert payload["bm25_representation"] == "hashed_tokens_v1"
    assert json.loads(state.read_text())["collections"]["vo_rag_books_summary_tree_v1"]["complete"]


async def test_store_cross_collection_preserves_sections_and_intentional_parent_collapse() -> None:
    class Store(QdrantStore):
        def __init__(self) -> None:
            pass

        async def search_hybrid(
            self, collection: str, *args: Any, **kwargs: Any
        ) -> list[ScoredPoint]:
            return [
                ScoredPoint(
                    id=i,
                    score=1,
                    version=0,
                    payload={
                        "source_id": "source",
                        "page_number": 1,
                        "chunk_id": "section-" + str(i),
                        **({"parent_chunk_id": "shared-parent"} if i > 1 else {}),
                    },
                )
                for i in range(4)
            ]

    result = await Store().search_cross_collection(["c1", "c2"], [1], {}, {}, [[1]], top_k=8)
    assert len(result) == 3
    assert {point.id for point in result} == {0, 1, 2}


async def test_brain_metadata_filter_indexes_are_reconciled_for_existing_collection() -> None:
    from vo_rag.core.storage.qdrant import _ensure_payload_indexes

    created: dict[str, Any] = {}

    class Client:
        async def get_collection(self, name: str) -> Any:
            return SimpleNamespace(payload_schema={"source_id": object()})

        async def create_payload_index(self, **kwargs: Any) -> None:
            created[kwargs["field_name"]] = kwargs["field_schema"]

    await _ensure_payload_indexes(cast(Any, Client()), "vo_rag_brain_v1")
    assert {"tags", "subject_type", "page_area", "writer_version", "version_hash"} <= created.keys()


async def test_standard_rerank_hook_gets_explicit_pool_output_bound() -> None:
    requested: list[int] = []

    class Retriever:
        async def cross_collection_search(self, **kwargs: Any) -> list[SearchResult]:
            return [hit(str(i), str(i)) for i in range(kwargs["top_k"])]

    def standard_hook(query: str, rows: list[SearchResult], top_k: int = 20) -> list[SearchResult]:
        requested.append(top_k)
        return rows[:top_k]

    result = await execute_retrieval(
        retriever=cast(Any, Retriever()),
        requested_collections=["c"],
        query="q",
        top_k=200,
        rerank_fn=standard_hook,
    )
    assert len(result) == 200 and requested == [200]


async def test_owner_migration_empty_tokens_real_http_preserves_vectors_and_replays(
    tmp_path: Path,
) -> None:
    import pytest
    from qdrant_client.http.exceptions import UnexpectedResponse

    point_id = "060658a0-1490-49ee-9c64-b76ab05524f0"
    payload: dict[str, Any] = {"book_id": "b", "label": "the and"}
    vectors: dict[str, Any] = {"dense": [1.0], "sparse": {123: 0.7}, "bm25": {123: 0.7}}
    requests: list[dict[str, Any]] = []
    interrupted = True

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal interrupted
        body = json.loads(request.content)
        if request.url.path.endswith("/scroll"):
            return httpx.Response(
                200,
                json={
                    "result": {
                        "points": [{"id": point_id, "payload": payload}],
                        "next_page_offset": None,
                    },
                    "status": "ok",
                    "time": 0,
                },
            )
        assert request.url.path.endswith("/batch")
        requests.append(body)
        operations = body["operations"]
        assert operations[0] == {"delete_vectors": {"points": [point_id], "vector": ["bm25"]}}
        assert not any("update_vectors" in operation for operation in operations)
        vectors.pop("bm25", None)
        if interrupted:
            interrupted = False
            return httpx.Response(500, json={"status": {"error": "lost reply after deletion"}})
        payload.update(operations[1]["set_payload"]["payload"])
        return httpx.Response(
            200,
            json={
                "result": [{"operation_id": 1, "status": "completed"} for _ in operations],
                "status": "ok",
                "time": 0,
            },
        )

    store = QdrantStore("http://q", prefer_grpc=False)
    transport = store._client._client.openapi_client.client
    await transport._async_client.aclose()
    transport._async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    state = tmp_path / "migration.json"
    try:
        with pytest.raises(UnexpectedResponse):
            await migrate(store, ["vo_rag_books_graph_nodes_v1"], state, apply=True)
        assert (
            json.loads(state.read_text())["collections"]["vo_rag_books_graph_nodes_v1"]["scanned"]
            == 0
        )
        assert "writer_contract_version" not in payload
        assert vectors == {"dense": [1.0], "sparse": {123: 0.7}}
        await migrate(store, ["vo_rag_books_graph_nodes_v1"], state, apply=True)
        assert payload["bm25_representation"] == "absent_no_tokens"
        assert payload["chunk_id"] == point_id and payload["source_id"] == "b"
        assert json.loads(state.read_text())["collections"]["vo_rag_books_graph_nodes_v1"][
            "complete"
        ]
        assert requests[0] == requests[1]
    finally:
        await store._client.close()


async def test_empty_sparse_queries_send_only_dense_prefetch_real_http() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"result": {"points": []}, "status": "ok", "time": 0})

    store = QdrantStore("http://q", prefer_grpc=False)
    transport = store._client._client.openapi_client.client
    await transport._async_client.aclose()
    transport._async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await store.search_hybrid("vo_rag_books_graph_nodes_v1", [1], {}, {}, [])
        assert [row["using"] for row in captured[0]["prefetch"]] == ["dense"]
        assert captured[0]["prefetch"][0]["query"] == [1.0]
        await store.search_hybrid("c", [], {}, {123: 1.0}, [])
        assert [row["using"] for row in captured[1]["prefetch"]] == ["bm25"]
        assert captured[1]["prefetch"][0]["query"]["indices"] == [123]
        assert await store.search_hybrid("c", [], {}, {}, []) == []
        assert len(captured) == 2
    finally:
        await store._client.close()


async def test_passage_writers_preserve_their_empty_sparse_contract_real_http() -> None:
    from vo_rag.core.graph.persistence import upsert_node
    from vo_rag.core.raptor.persistence import upsert_cluster_node
    from vo_rag.core.storage.points import VoPoint
    from vo_rag.core.summary.persistence import upsert_summary_node

    written: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        written.extend(json.loads(request.content)["points"])
        return httpx.Response(
            200,
            json={"result": {"operation_id": 1, "status": "completed"}, "status": "ok", "time": 0},
        )

    store = QdrantStore("http://q", prefer_grpc=False)
    transport = store._client._client.openapi_client.client
    await transport._async_client.aclose()
    transport._async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    point_id = "060658a0-1490-49ee-9c64-b76ab05524f0"
    node = SimpleNamespace(
        node_id=point_id,
        book_id="book",
        label="the and",
        summary_text="the and",
        model_dump=lambda **kwargs: {
            "node_id": point_id,
            "label": "the and",
            "summary_text": "the and",
            "book_id": "book",
        },
    )
    dense = [0.1] * 1024
    try:
        for writer in (upsert_node, upsert_cluster_node, upsert_summary_node):
            await writer(
                node=cast(Any, node),
                dense_embedding=dense,
                sparse_embedding={},
                colbert_embedding=[],
                store=cast(Any, store),
            )
        await store.upsert_vectors(
            "vo_rag_brain_v1",
            [
                VoPoint(
                    source_id="book",
                    chunk_index=0,
                    stage="raw",
                    dense=dense,
                    sparse={},
                    bm25={},
                    colbert=[],
                    payload={"content": "the and", "chunk_id": point_id},
                )
            ],
        )
        assert len(written) == 4
        for point in written[:3]:
            assert set(point["vector"]) == {"dense"}
        assert set(written[3]["vector"]) == {"dense", "sparse", "bm25"}
        assert written[3]["vector"]["sparse"] == {"indices": [], "values": []}
        assert written[3]["vector"]["bm25"] == {"indices": [], "values": []}
        for point in written:
            assert point["vector"]["dense"] == dense
            assert point["payload"]["bm25_representation"] == "absent_no_tokens"
            assert point["payload"]["colbert_representation"] == "absent"
            assert point["payload"]["chunk_id"] == point_id
        assert all(point["id"] == point_id for point in written[:3])
    finally:
        await store._client.close()


async def test_owner_modifier_reconciliation_real_http_is_readonly_then_idempotent(
    tmp_path: Path,
) -> None:
    from qdrant_client.models import Modifier, SparseVectorParams

    from vo_rag.ops.passage_migration import reconcile_sparse_modifiers

    modifiers = {"sparse": SparseVectorParams(), "bm25": SparseVectorParams()}
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        body = json.loads(request.content)
        requests.append(body)
        assert body == {
            "sparse_vectors": {"sparse": {"modifier": "idf"}, "bm25": {"modifier": "idf"}}
        }
        for name, config in body["sparse_vectors"].items():
            modifiers[name] = SparseVectorParams(**config)
        return httpx.Response(200, json={"result": True, "status": "ok", "time": 0})

    store = QdrantStore("http://q", prefer_grpc=False)
    transport = store._client._client.openapi_client.client
    await transport._async_client.aclose()
    transport._async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def info(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(sparse_vectors=modifiers))
        )

    store._client.get_collection = info
    try:
        before = await reconcile_sparse_modifiers(store, ["c"])
        assert before["c"]["requires_idf"] == ["sparse", "bm25"] and not requests
        # A completed point checkpoint must not suppress configuration repair.
        state = tmp_path / "complete.json"
        state.write_text(json.dumps({"collections": {"c": {"complete": True}}}))
        await reconcile_sparse_modifiers(store, ["c"], apply=True)
        await migrate(store, ["c"], state, apply=True)
        assert all(params.modifier == Modifier.IDF for params in modifiers.values())
        after = await reconcile_sparse_modifiers(store, ["c"], apply=True)
        assert after["c"]["requires_idf"] == [] and len(requests) == 1
    finally:
        await store._client.close()


def test_all_collection_creators_share_idf_scoring_contract() -> None:
    from qdrant_client.models import Modifier

    from vo_rag.core.graph.persistence import _node_sparse_vector_config
    from vo_rag.core.raptor.persistence import _sparse_vector_config as raptor
    from vo_rag.core.storage.qdrant import _sparse_vector_config as common
    from vo_rag.core.summary.persistence import _sparse_vector_config as summary

    for builder in (_node_sparse_vector_config, raptor, summary, common):
        assert {name: config.modifier for name, config in builder().items()} == {
            "sparse": Modifier.IDF,
            "bm25": Modifier.IDF,
        }


async def test_raptor_evidence_query_uses_hashed_lexical_ids(monkeypatch: Any) -> None:
    from vo_rag.core.raptor.persistence import expand_cluster_node
    from vo_rag.core.retrieval.bm25_tokenizer import tokenize, tokens_to_bm25_sparse

    captured: dict[str, Any] = {}

    async_client = httpx.AsyncClient

    def embed_wire(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["texts"] == ["running mechanics"]
        return httpx.Response(
            200,
            json={
                "dense": [[1.0]],
                "sparse": [{"indices": [999], "values": [0.5]}],
                "colbert": [[]],
            },
        )

    def embedding_http(*args: Any, **kwargs: Any) -> Any:
        return async_client(*args, transport=httpx.MockTransport(embed_wire), **kwargs)

    class Store:
        async def search_hybrid(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return []

    monkeypatch.setattr(httpx, "AsyncClient", embedding_http)
    await expand_cluster_node(
        node=cast(Any, SimpleNamespace(source_chunk_ids=["source"])),
        query="running mechanics",
        m=1,
        qdrant=cast(Any, Store()),
    )
    assert captured["query_sparse"] == {999: 0.5}
    assert captured["query_bm25"] == tokens_to_bm25_sparse(tokenize("running mechanics"))
    assert 999 not in captured["query_bm25"]


def test_exact_copies_preserve_provenance_context_and_visual_evidence() -> None:
    from vo_rag.core.retrieval.fusion import dedupe_exact_evidence

    rows = [
        hit("a", "copy-a"),
        hit("b", "copy-b"),
        hit("distinct", "copy-a"),
        hit("different-heading", "other"),
        hit("image", "image-source"),
        hit("other-model", "model"),
    ]
    for row in (rows[0], rows[1], rows[3], rows[4], rows[5]):
        row.payload.update(content="Identical evidence paragraph.", section_title="Shared heading")
    rows[1].payload.update(
        path="wiki/reference.md",
        commit="abc123",
        url="https://example.test/reference",
        fetched_at="2026-10-02",
    )
    rows[3].payload["section_title"] = "Different context"
    rows[4].payload["has_image"] = True
    rows[5].payload["title"] = "Different entity"
    selected = dedupe_exact_evidence(rows)
    assert [row.id for row in selected] == [
        "a",
        "distinct",
        "different-heading",
        "image",
        "other-model",
    ]
    assert {item["id"] for item in selected[0].payload["duplicate_provenance"]} == {"a", "b"}
    alternate = next(
        item for item in selected[0].payload["duplicate_provenance"] if item["id"] == "b"
    )
    assert alternate["path"] == "wiki/reference.md" and alternate["commit"] == "abc123"
    assert (
        alternate["url"] == "https://example.test/reference"
        and alternate["fetched_at"] == "2026-10-02"
    )
    assert "duplicate_provenance" not in rows[0].payload
    assert dedupe_exact_evidence(selected) == selected


async def test_exact_copies_do_not_consume_rerank_pool_or_final_slots() -> None:
    selected: list[SearchResult] = []

    class Retriever:
        async def cross_collection_search(self, **kwargs: Any) -> list[SearchResult]:
            rows = []
            for copy in range(3):
                for passage in range(20):
                    row = hit(f"{copy}-{passage}", f"copy-{copy}")
                    row.payload["content"] = f"Exact passage number {passage}"
                    rows.append(row)
            return rows

    def rerank(query: str, rows: list[SearchResult]) -> list[SearchResult]:
        selected.extend(rows)
        return rows

    result = await execute_retrieval(
        retriever=cast(Any, Retriever()),
        requested_collections=["c"],
        query="ordinary question",
        top_k=8,
        rerank_fn=rerank,
    )
    assert len(selected) == 20 and len(result) == 8
    assert len({row.payload["content"] for row in result}) == 8
    assert all(len(row.payload["duplicate_provenance"]) == 3 for row in result)


async def test_migration_groups_identical_patches_real_http_and_replays(tmp_path: Path) -> None:
    import pytest
    from qdrant_client.http.exceptions import UnexpectedResponse

    payloads = {str(index): {"source_id": "source", "chunk_id": str(index)} for index in range(3)}
    payloads["3"] = {"source_id": "source"}  # unique identity repair stays separate
    batches: list[dict[str, Any]] = []
    interrupted = True

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal interrupted
        if request.url.path.endswith("/scroll"):
            return httpx.Response(
                200,
                json={
                    "result": {
                        "points": [
                            {"id": int(identity), "payload": payload}
                            for identity, payload in payloads.items()
                        ],
                        "next_page_offset": None,
                    },
                    "status": "ok",
                    "time": 0,
                },
            )
        body = json.loads(request.content)
        batches.append(body)
        if interrupted:
            assert len(body["operations"]) == 2
            assert body["operations"][0]["set_payload"]["points"] == [0, 1, 2]
            assert body["operations"][1]["set_payload"]["points"] == [3]
            # The first grouped patch took effect before the reply was lost.
            first = body["operations"][0]["set_payload"]
            for identity in first["points"]:
                payloads[str(identity)].update(first["payload"])
            interrupted = False
            return httpx.Response(
                500, json={"status": {"error": "uncertain reply before acknowledgement"}}
            )
        for operation in body["operations"]:
            for identity in operation["set_payload"]["points"]:
                payloads[str(identity)].update(operation["set_payload"]["payload"])
        return httpx.Response(
            200,
            json={
                "result": [{"operation_id": 1, "status": "completed"} for _ in body["operations"]],
                "status": "ok",
                "time": 0,
            },
        )

    store = QdrantStore("http://q", prefer_grpc=False)
    transport = store._client._client.openapi_client.client
    await transport._async_client.aclose()
    transport._async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    state = tmp_path / "grouped.json"
    try:
        with pytest.raises(UnexpectedResponse):
            await migrate(store, ["vo_rag_books_raw_v1"], state, apply=True)
        pending = json.loads(state.read_text())["collections"]["vo_rag_books_raw_v1"]
        assert pending["scanned"] == pending["changed"] == 0
        assert pending["offset"] is None and pending["pending"]["changed"] == 4
        result = await migrate(store, ["vo_rag_books_raw_v1"], state, apply=True)
        assert len(batches[1]["operations"]) == 1
        assert batches[1]["operations"][0]["set_payload"]["points"] == [3]
        assert result["collections"]["vo_rag_books_raw_v1"] == {
            "offset": None,
            "scanned": 4,
            "changed": 4,
            "complete": True,
        }
        assert payloads["3"]["chunk_id"] == "3"
    finally:
        await store._client.close()


def test_observed_physical_book_copy_shape_preserves_three_citations() -> None:
    from vo_rag.core.storage.passage_contract import collapse_evidence_payloads

    # Captured physical-copy identity/context shape; prose replaced with a
    # fixture sentence instead of committing private corpus book excerpts.
    # Source SHA256 content digests below are identifiers, never credentials.
    observed: list[dict[str, Any]] = [
        {
            "id": "1fc073d8-c51c-1bf5-2bac-5cdb4c8405b1",
            "collection": "vo_rag_books_raw_v1",
            "payload": {
                "chunk_id": "6b0b9d50-3346-8ec9-c2ff-65064df8a59b",
                "source_id": (
                    "vo-rag-corpus-sha256:8028cf68b0eb615e645af365be982c20c3a58c60ea2a"
                    "8bfac9fbd5e920b7112e"  # pragma: allowlist secret
                ),
                "section_path": (
                    "[The Craft of Fast Driving (Corner Craft) > CHAPT"
                    "ER 10 DRIVING AT THE LIMIT > 10 DRIVING AT THE LIMIT > TRAIL BRAK"
                    "ING]"
                ),
                "section_title": "TRAIL BRAKING",
                "chunk_type": "text",
                "has_image": False,
                "source_book": "The Craft of Fast Dri___28727703",
                "metadata": {
                    "title": "The Craft of Fast Dri___28727703",
                    "source_book": "The Craft of Fast Dri___28727703",
                },
                "content": "Same observed answer text.",
            },
        },
        {
            "id": "c8a63b84-f058-5de7-3c70-010574038e36",
            "collection": "vo_rag_books_raw_v1",
            "payload": {
                "chunk_id": "3adca79a-44f8-5761-149f-31eea7afa42e",
                "source_id": (
                    "vo-rag-corpus-sha256:28f462d3e713a9cccc48b570c06da7ffa86009fcd083"
                    "6f4d0c6d0882f17abaf6"  # pragma: allowlist secret
                ),
                "section_path": (
                    "[The Craft of Fast Driving > CHAPTER 10 DRIVING AT"
                    " THE LIMIT > 10 DRIVING AT THE LIMIT > TRAIL BRAKING]"
                ),
                "section_title": "TRAIL BRAKING",
                "chunk_type": "text",
                "has_image": False,
                "source_book": "The Craft of Fast Dri___18248947",
                "metadata": {
                    "title": "The Craft of Fast Dri___18248947",
                    "source_book": "The Craft of Fast Dri___18248947",
                },
                "content": "Same observed answer text.",
            },
        },
        {
            "id": "fdbfe772-70f1-5257-21f9-2ad27b805a36",
            "collection": "vo_rag_books_raw_v1",
            "payload": {
                "chunk_id": "1b3c330b-a94e-672a-6e7e-71cd148f823c",
                "source_id": (
                    "vo-rag-corpus-sha256:4b9705883be079c63f1d7cf8842edba0d51c1d665e9c"
                    "72b9dd9a909eafe80a9e"  # pragma: allowlist secret
                ),
                "section_path": (
                    "[The Craft of Fast Driving > CHAPTER 10 DRIVING AT"
                    " THE LIMIT > 10 DRIVING AT THE LIMIT > TRAIL BRAKING]"
                ),
                "section_title": "TRAIL BRAKING",
                "chunk_type": "text",
                "has_image": False,
                "source_book": "The Craft of Fast Dri_",
                "metadata": {
                    "title": "The Craft of Fast Dri_",
                    "source_book": "The Craft of Fast Dri_",
                },
                "content": "Same observed answer text.",
            },
        },
    ]
    for row in observed:
        row["payload"]["document_id"] = row["payload"]["source_id"].partition(":")[2]
    rows = [(row["payload"], row["id"], row["collection"]) for row in observed]
    collapsed = collapse_evidence_payloads(rows)
    assert len(collapsed) == 1 and collapsed[0][0] == 0
    assert {item["source_id"] for item in collapsed[0][1]["duplicate_provenance"]} == {
        row[0]["source_id"] for row in rows
    }
    assert {item["section_path"] for item in collapsed[0][1]["duplicate_provenance"]} == {
        row[0]["section_path"] for row in rows
    }
    assert all("title" not in row[0] for row in rows)
    assert len({row[0]["metadata"]["title"] for row in rows}) == 3
    assert all("content_hash" not in row[0] and "logical_book_key" not in row[0] for row in rows)


def test_repeated_same_point_is_collapsed_without_fabricating_alternate_citation() -> None:
    from vo_rag.core.storage.passage_contract import collapse_evidence_payloads

    payload = {"content": "One text answer", "section_title": "Context"}
    rows = [(payload, "same-point", "raw"), (payload, "same-point", "raw")]
    assert collapse_evidence_payloads(rows) == [(0, payload)]
    assert "duplicate_provenance" not in payload


async def test_supported_store_same_track_parent_votes_and_citations_match_fusion() -> None:
    unique = ScoredPoint(
        id="unique", version=0, score=1, payload={"source_id": "other", "chunk_id": "unique"}
    )
    raw = ScoredPoint(
        id="raw",
        version=0,
        score=1,
        payload={"source_id": "book", "chunk_id": "raw", "parent_chunk_id": "parent"},
    )
    sterile = ScoredPoint(
        id="sterile",
        version=0,
        score=1,
        payload={"source_id": "book", "chunk_id": "sterile", "parent_chunk_id": "parent"},
    )

    class Store(QdrantStore):
        def __init__(self) -> None:
            pass

        async def search_hybrid(
            self, collection: str, *args: Any, **kwargs: Any
        ) -> list[ScoredPoint]:
            return [unique, raw, sterile] if collection == "first" else [sterile]

    same = await Store().search_cross_collection(["first"], [1], {}, {}, [], top_k=3)
    assert [row.id for row in same] == ["unique", "raw"]
    assert same[0].score == 1 / 61 and same[1].score == 1 / 62
    assert {item["id"] for item in (same[1].payload or {})["duplicate_provenance"]} == {
        "raw",
        "sterile",
    }
    independent = await Store().search_cross_collection(
        ["first", "second"], [1], {}, {}, [], top_k=3
    )
    assert [row.id for row in independent] == ["raw", "unique"]
    assert independent[0].score == 1 / 62 + 1 / 61
    assert len((independent[0].payload or {})["duplicate_provenance"]) == 3
    assert "duplicate_provenance" not in (raw.payload or {})


def test_verified_visual_copies_preserve_actual_reference_shape_and_context() -> None:
    """Observed physical-copy references share bytes, unlike the adjacent image."""
    from copy import deepcopy

    from vo_rag.core.retrieval.fusion import dedupe_exact_evidence

    # Actual parser SHA-256 identities and physical-copy references, not credentials.
    # pragma: allowlist secret
    image_id = "56b63d8dbccd62cfe985280a61b27c55a646ca6d67a9122e824c8cf061275719"
    # pragma: allowlist secret
    other_id = "6779a6d4fdf0de352455e69fecc21b7ac99352dcca6442e7c6634206086096a5"
    rows = [hit(str(index), f"physical-copy-{index}") for index in range(9)]
    for index, row in enumerate(rows):
        row.payload.update(
            content="A shared figure caption and its surrounding evidence.",
            chunk_type="figure_composite",
            has_image=True,
            image_id=image_id,
            image_ref=f"copy-{index}/p013_img045.jpg",
            section_title="TRAIL BRAKING",
            section_path="[The Craft of Fast Driving > TRAIL BRAKING]",
            metadata={
                "title": f"The Craft of Fast Dri___{index}",
                "source_book": f"The Craft of Fast Dri___{index}",
            },
        )
    references = [
        # pragma: allowlist secret
        "8028cf68b0eb615e645af365be982c20c3a58c60ea2a8bfac9fbd5e920b7112e/p013_img046.jpeg",
        # pragma: allowlist secret
        "28f462d3e713a9cccc48b570c06da7ffa86009fcd0836f4d0c6d0882f17abaf6/p013_img045.jpg",
        # pragma: allowlist secret
        "4b9705883be079c63f1d7cf8842edba0d51c1d665e9c72b9dd9a909eafe80a9e/p013_img045.jpg",
    ]
    for row, reference in zip(rows, references, strict=False):
        row.payload["image_ref"] = reference
    rows[0].payload["image_id"] = other_id
    rows[3].payload["section_title"] = "Different heading"
    rows[4].payload["title"] = "Different entity"
    rows[5].payload["content"] = "Different annotation of the same image."
    rows[6].payload.pop("image_id")
    rows[7].payload["image_id"] = "not-a-verified-byte-hash"
    rows[8].payload["image_id"] = ""
    originals = deepcopy(rows)
    selected = dedupe_exact_evidence(rows)
    assert [row.id for row in selected] == ["0", "1", "3", "4", "5", "6", "7", "8"]
    aliases = selected[1].payload["duplicate_provenance"]
    assert [alias["id"] for alias in aliases] == ["1", "2"]
    assert [alias["image_ref"] for alias in aliases] == references[1:]
    assert all(alias["image_id"] == image_id for alias in aliases)
    assert rows == originals
    assert dedupe_exact_evidence(selected) == selected


async def test_verified_visual_copies_leave_slots_for_distinct_images() -> None:
    captured: list[SearchResult] = []

    class Retriever:
        async def cross_collection_search(self, **kwargs: Any) -> list[SearchResult]:
            rows = [hit(str(index), f"copy-{index}") for index in range(12)]
            for index, row in enumerate(rows):
                row.payload.update(
                    content="Shared caption",
                    has_image=True,
                    image_ref=f"copy-{index}/image.png",
                    image_id=f"{index // 3:064x}",
                    section_title="Shared context",
                )
            return rows

    def rerank(query: str, rows: list[SearchResult]) -> list[SearchResult]:
        captured.extend(rows)
        return rows

    result = await execute_retrieval(
        retriever=cast(Any, Retriever()),
        requested_collections=["c"],
        query="visual question",
        top_k=4,
        rerank_fn=rerank,
    )
    assert len(captured) == len(result) == 4
    assert len({row.payload["image_id"] for row in result}) == 4
    assert all(len(row.payload["duplicate_provenance"]) == 3 for row in result)


def test_verified_visual_context_binds_chapters_and_missing_path_entities() -> None:
    from vo_rag.core.retrieval.fusion import dedupe_exact_evidence

    rows = [hit(str(index), f"source-{index}") for index in range(5)]
    for row in rows:
        row.payload.update(
            content="Same caption",
            chunk_type="figure_composite",
            image_id="a" * 64,
            section_title="Shared leaf heading",
            section_path="[Book A > Chapter 1 > Shared leaf]",
            metadata={"title": "Book A", "source_book": "Book A"},
        )
    rows[1].payload["section_path"] = "[Book A > Chapter 2 > Shared leaf]"
    rows[2].payload["section_path"] = "[Book B > Chapter 1 > Shared leaf]"
    for row in rows[3:]:
        row.payload.pop("section_path")
    rows[4].payload["metadata"] = {"title": "Book B", "source_book": "Book B"}
    assert dedupe_exact_evidence(rows) == rows

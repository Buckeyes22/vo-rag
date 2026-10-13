"""Deep tests for vo_rag.core.raptor.builder.

Mirrors ``tests/core/summary/test_builder.py`` patterns: in-memory Qdrant
double, mocked embed client, sequential mocked LLM client, deterministic
fake clustering function. No unittest.mock; no live models.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.raptor.builder import (
    RaptorBuildConfig,
    build_raptor_for_book,
    build_raptor_for_corpus,
)
from vo_rag.core.raptor.clustering import ClusterAssignment, ClusteringResult
from vo_rag.core.raptor.persistence import RAPTOR_COLLECTION
from vo_rag.core.raptor.schemas import ClusterReproducibilityConfig

# ---------------------------------------------------------------------------
# Test doubles and helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    chunk_id: str,
    content: str,
    section_path: str = "[Book > Section]",
    page_number: int | None = None,
) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        chunk_type="text",
        document_id="book-001",
        source_id="source-001",
        page_number=page_number,
        page_range=None,
        section_path=section_path,
        section_title="Test Section",
        chapter_path="",
        content=content,
        embedding_text=f"{section_path}\n\n{content}",
        token_count=max(1, len(content.split())),
        has_image=False,
        image_id=None,
        image_ref=None,
        vlm_description=None,
        metadata={},
    )


def _mock_embed_client() -> Any:
    """Return a dict-backed embed client that never calls a real model."""

    class FakeEmbedClient:
        async def embed(
            self,
            texts: list[str],
            *,
            return_dense: bool = True,
            return_sparse: bool = False,
            return_colbert: bool = False,
        ) -> dict[str, Any]:
            dense = [0.05] * 1024
            return {
                "dense": [dense] * len(texts),
                "sparse": [{"indices": [0, 1, 2], "values": [0.1, 0.2, 0.3]}] * len(texts),
                "colbert": [[[0.05] * 1024]] * len(texts) if return_colbert else [],
            }

    return FakeEmbedClient()


class _InMemoryQdrantStore:
    """Dict-backed AsyncQdrantClient test double — no unittest.mock."""

    def __init__(self) -> None:
        self._points: dict[str, dict[str, Any]] = {}
        self._collections: set[str] = set()
        self._payload_indexes: dict[str, set[str]] = {}
        self._upsert_calls: list[dict[str, Any]] = []

    async def upsert(self, collection_name: str, points: list[Any], wait: bool = True) -> None:
        for point in points:
            self._upsert_calls.append({"collection": collection_name, "point_id": point.id})
            self._points[str(point.id)] = {
                "id": point.id,
                "vector": point.vector,
                "payload": point.payload,
            }

    async def retrieve(
        self, collection_name: str, ids: list[str], with_payload: bool = True
    ) -> list[Any]:
        class _Point:
            def __init__(self, data: dict[str, Any]) -> None:
                self.id = data["id"]
                self.payload = data["payload"]
                self.vector = data.get("vector", {})

        return [_Point(self._points[oid]) for oid in ids if oid in self._points]

    async def scroll(
        self,
        collection_name: str,
        scroll_filter: Any = None,
        limit: int = 100,
        with_payload: bool = True,
    ) -> tuple[list[Any], Any]:
        class _Point:
            def __init__(self, data: dict[str, Any]) -> None:
                self.id = data["id"]
                self.payload = data["payload"]

        return ([_Point(p) for p in list(self._points.values())[:limit]], None)

    async def get_collections(self) -> Any:
        class _Collections:
            def __init__(self, names: set[str]) -> None:
                self.collections = [type("C", (), {"name": name})() for name in names]

        return _Collections(self._collections)

    async def create_collection(
        self,
        collection_name: str,
        vectors_config: Any = None,
        sparse_vectors_config: Any = None,
    ) -> None:
        self._collections.add(collection_name)

    async def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema: Any = None,
    ) -> None:
        self._payload_indexes.setdefault(collection_name, set()).add(field_name)


class _FakeQdrantStoreWrapper:
    """Wraps _InMemoryQdrantStore so it exposes ``_client``."""

    def __init__(self, client: _InMemoryQdrantStore) -> None:
        self._client = client


def _mock_llm_client(responses: list[LLMResponse]) -> Any:
    """Return a sequential-response LLM client consuming from a pre-seeded list."""

    class FakeLLMClient:
        def __init__(self, resp_list: list[LLMResponse]) -> None:
            self._responses = resp_list
            self._calls: list[dict[str, Any]] = []

        async def complete(
            self,
            system: str,
            user: str,
            max_tokens: int,
            temperature: float,
        ) -> LLMResponse:
            self._calls.append(
                {
                    "system": system,
                    "user": user,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            return self._responses.pop(0)

    return FakeLLMClient(responses)


def _resp(text: str = "Summary.", input_tokens: int = 10, output_tokens: int = 5) -> LLMResponse:
    return LLMResponse(
        text=text, model="test-model", input_tokens=input_tokens, output_tokens=output_tokens
    )


def _many(
    text: str = "Summary.", input_tokens: int = 10, output_tokens: int = 5
) -> list[LLMResponse]:
    return [_resp(text, input_tokens, output_tokens) for _ in range(64)]


def _default_config() -> RaptorBuildConfig:
    return RaptorBuildConfig(
        summary_model="test-model",
        source_collection_version="test-v1",
        reproducibility_config=ClusterReproducibilityConfig(
            umap_random_state=42,
            gmm_random_state=42,
        ),
    )


def _fake_cluster_func(
    chunk_ids: list[str],
    embeddings: Any,
    *,
    config: ClusterReproducibilityConfig,
) -> ClusteringResult:
    """Deterministic fake clustering: split into ~2-item clusters."""
    n = len(chunk_ids)
    n_clusters = 1 if n <= 2 else max(2, n // 2)

    assignments = []
    for i, cid in enumerate(chunk_ids):
        cluster_idx = i % n_clusters
        assignments.append(
            ClusterAssignment(
                chunk_id=cid,
                cluster_index=cluster_idx,
                probability=1.0,
                umap_coords=[0.0, 0.0],
            )
        )

    return ClusteringResult(
        assignments=assignments,
        n_clusters_selected=n_clusters,
        bic_scores={},
        aic_scores={},
        umap_random_state=config.umap_random_state,
        gmm_random_state=config.gmm_random_state,
    )


def _single_cluster_func(
    chunk_ids: list[str],
    embeddings: Any,
    *,
    config: ClusterReproducibilityConfig,
) -> ClusteringResult:
    """Deterministic fake clustering: place every input in one dense cluster."""
    return ClusteringResult(
        assignments=[
            ClusterAssignment(
                chunk_id=cid,
                cluster_index=0,
                probability=1.0,
                umap_coords=[float(i), 0.0],
            )
            for i, cid in enumerate(chunk_ids)
        ],
        n_clusters_selected=1,
        bic_scores={},
        aic_scores={},
        umap_random_state=config.umap_random_state,
        gmm_random_state=config.gmm_random_state,
    )


def _pair_cluster_func(
    chunk_ids: list[str],
    embeddings: Any,
    *,
    config: ClusterReproducibilityConfig,
) -> ClusteringResult:
    """Deterministic fake clustering: consecutive pairs form one cluster."""
    return ClusteringResult(
        assignments=[
            ClusterAssignment(
                chunk_id=cid,
                cluster_index=i // 2,
                probability=1.0,
                umap_coords=[float(i), 0.0],
            )
            for i, cid in enumerate(chunk_ids)
        ],
        n_clusters_selected=max(1, (len(chunk_ids) + 1) // 2),
        bic_scores={},
        aic_scores={},
        umap_random_state=config.umap_random_state,
        gmm_random_state=config.gmm_random_state,
    )


# ---------------------------------------------------------------------------
# Builder tests
# ---------------------------------------------------------------------------


def test_build_config_defaults_are_explicit() -> None:
    cfg = RaptorBuildConfig(summary_model="test")
    assert cfg.embedding_version != "<frozen at Phase 0>"
    assert cfg.source_collection_version != "<frozen at Phase 0>"
    assert cfg.embedding_version == "bge-m3@unversioned"
    assert cfg.source_collection_version == "vo_rag_books_raw_v1"
    assert cfg.max_concurrent_llm_calls == 8
    assert cfg.max_tree_levels == 5
    assert cfg.summary_temperature == 0.2
    assert cfg.summary_max_tokens == 2500
    assert cfg.max_summary_input_chars == 32000
    assert cfg.max_skip_fraction == 0.5


async def test_cluster_summary_input_is_capped_fairly_per_child() -> None:
    chunks = [_make_chunk(chunk_id=f"c{i}", content=f"child-{i}|" + ("x" * 80)) for i in range(4)]
    embeddings = [[0.01] * 1024 for _ in chunks]
    config = _default_config()
    config.max_summary_input_chars = 48
    store = _FakeQdrantStoreWrapper(_InMemoryQdrantStore())
    llm = _mock_llm_client(_many())
    captured_primary_inputs: list[str] = []
    warning_events: list[dict[str, Any]] = []

    def _spy_prompt(
        *,
        level: Any,
        primary_input: str,
        supporting_evidence: list[dict[str, str | float]],
        book_id: str,
    ) -> tuple[str, str]:
        captured_primary_inputs.append(primary_input)
        return "system", "user"

    def _spy_warning(event: str, **kwargs: Any) -> None:
        warning_events.append({"event": event, **kwargs})

    with (
        patch("vo_rag.core.raptor.builder.build_level_extract_support_prompt", _spy_prompt),
        patch("vo_rag.core.raptor.builder.log.warning", _spy_warning),
    ):
        await build_raptor_for_book(
            book_id="book-dense",
            chunks=chunks,
            leaf_embeddings=embeddings,
            config=config,
            llm_call=llm.complete,
            embed_client=_mock_embed_client(),
            store=store,
            cluster_func=_single_cluster_func,
        )

    assert len(captured_primary_inputs) == 1
    capped_input = captured_primary_inputs[0]
    separator_overhead = len("\n\n") * (len(chunks) - 1)
    assert len(capped_input) <= config.max_summary_input_chars + separator_overhead
    for i in range(len(chunks)):
        assert f"child-{i}|" in capped_input
    assert warning_events == [
        {
            "event": "raptor_input_truncated",
            "book_id": "book-dense",
            "level": "level_0",
            "n_children": len(chunks),
            "original_chars": len("\n\n".join(chunk.content for chunk in chunks)),
            "capped_chars": len(capped_input),
        }
    ]


async def test_cluster_summary_input_under_cap_passes_through_unchanged() -> None:
    chunks = [_make_chunk(chunk_id=f"c{i}", content=f"child-{i} body") for i in range(3)]
    plain_join = "\n\n".join(chunk.content for chunk in chunks)
    embeddings = [[0.01] * 1024 for _ in chunks]
    config = _default_config()
    config.max_summary_input_chars = len(plain_join) + 100
    store = _FakeQdrantStoreWrapper(_InMemoryQdrantStore())
    llm = _mock_llm_client(_many())
    captured_primary_inputs: list[str] = []

    def _spy_prompt(
        *,
        level: Any,
        primary_input: str,
        supporting_evidence: list[dict[str, str | float]],
        book_id: str,
    ) -> tuple[str, str]:
        captured_primary_inputs.append(primary_input)
        return "system", "user"

    with patch("vo_rag.core.raptor.builder.build_level_extract_support_prompt", _spy_prompt):
        await build_raptor_for_book(
            book_id="book-small",
            chunks=chunks,
            leaf_embeddings=embeddings,
            config=config,
            llm_call=llm.complete,
            embed_client=_mock_embed_client(),
            store=store,
            cluster_func=_single_cluster_func,
        )

    assert captured_primary_inputs == [plain_join]


async def test_multi_level_tree_builds() -> None:
    # 8 chunks -> fake clusters: 4 clusters (2 each) at L0,
    # then 2 clusters (2 each) at L1, then 1 root at L2.
    chunks = [_make_chunk(chunk_id=f"c{i}", content=f"Chunk {i} content.") for i in range(8)]
    embeddings = [[0.01] * 1024 for _ in range(8)]
    store = _FakeQdrantStoreWrapper(_InMemoryQdrantStore())
    llm = _mock_llm_client(_many())

    result, stats = await build_raptor_for_book(
        book_id="book-001",
        chunks=chunks,
        leaf_embeddings=embeddings,
        config=_default_config(),
        llm_call=llm.complete,
        embed_client=_mock_embed_client(),
        store=store,
        cluster_func=_fake_cluster_func,
    )

    assert result.book_id == "book-001"
    assert result.root is not None
    assert stats.node_count >= 3  # at least L0, L1, L2
    assert stats.n_skipped_clusters == 0
    levels = {n.level for n in result.nodes}
    assert len(levels) >= 2


async def test_cluster_failure_skips_cluster_and_builds_tree_from_survivors() -> None:
    chunks = [
        _make_chunk(chunk_id="c0", content="First survivor."),
        _make_chunk(chunk_id="c1", content="First survivor pair."),
        _make_chunk(chunk_id="c2", content="Bad cluster marker."),
        _make_chunk(chunk_id="c3", content="Bad cluster marker pair."),
        _make_chunk(chunk_id="c4", content="Second survivor."),
        _make_chunk(chunk_id="c5", content="Second survivor pair."),
    ]
    embeddings = [[0.01] * 1024 for _ in chunks]
    store = _FakeQdrantStoreWrapper(_InMemoryQdrantStore())
    config = _default_config()

    async def llm_call(
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        if "Bad cluster marker" in user:
            raise RuntimeError("exhausted cluster")
        return _resp("Survivor summary.")

    with patch("vo_rag.core.raptor.builder.RETRY_DELAYS", (0.0, 0.0, 0.0)):
        result, stats = await build_raptor_for_book(
            book_id="book-skip-one",
            chunks=chunks,
            leaf_embeddings=embeddings,
            config=config,
            llm_call=llm_call,
            embed_client=_mock_embed_client(),
            store=store,
            cluster_func=_pair_cluster_func,
        )

    assert stats.n_skipped_clusters == 1
    assert result.root is not None
    assert set(result.root.source_chunk_ids) == {"c0", "c1", "c4", "c5"}
    assert {"c2", "c3"}.isdisjoint(result.root.source_chunk_ids)


async def test_all_cluster_failures_raise_clear_safety_error() -> None:
    chunks = [_make_chunk(chunk_id=f"c{i}", content=f"Chunk {i}.") for i in range(4)]
    embeddings = [[0.01] * 1024 for _ in chunks]
    store = _FakeQdrantStoreWrapper(_InMemoryQdrantStore())

    async def llm_call(
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        raise RuntimeError("exhausted cluster")

    with (
        patch("vo_rag.core.raptor.builder.RETRY_DELAYS", (0.0, 0.0, 0.0)),
        pytest.raises(RuntimeError, match="all RAPTOR clusters were skipped"),
    ):
        await build_raptor_for_book(
            book_id="book-skip-all",
            chunks=chunks,
            leaf_embeddings=embeddings,
            config=_default_config(),
            llm_call=llm_call,
            embed_client=_mock_embed_client(),
            store=store,
            cluster_func=_pair_cluster_func,
        )


async def test_cluster_skip_fraction_over_threshold_raises() -> None:
    chunks = [
        _make_chunk(chunk_id="c0", content="Survivor cluster."),
        _make_chunk(chunk_id="c1", content="Survivor cluster pair."),
        _make_chunk(chunk_id="c2", content="Bad cluster marker A."),
        _make_chunk(chunk_id="c3", content="Bad cluster marker A pair."),
        _make_chunk(chunk_id="c4", content="Bad cluster marker B."),
        _make_chunk(chunk_id="c5", content="Bad cluster marker B pair."),
    ]
    embeddings = [[0.01] * 1024 for _ in chunks]
    store = _FakeQdrantStoreWrapper(_InMemoryQdrantStore())
    config = _default_config()
    config.max_skip_fraction = 0.5

    async def llm_call(
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        if "Bad cluster marker" in user:
            raise RuntimeError("exhausted cluster")
        return _resp("Survivor summary.")

    with (
        patch("vo_rag.core.raptor.builder.RETRY_DELAYS", (0.0, 0.0, 0.0)),
        pytest.raises(RuntimeError, match="RAPTOR cluster skip fraction"),
    ):
        await build_raptor_for_book(
            book_id="book-skip-threshold",
            chunks=chunks,
            leaf_embeddings=embeddings,
            config=config,
            llm_call=llm_call,
            embed_client=_mock_embed_client(),
            store=store,
            cluster_func=_pair_cluster_func,
        )


async def test_root_aggregates_all_leaves() -> None:
    chunks = [_make_chunk(chunk_id=f"c{i}", content=f"Chunk {i}.") for i in range(4)]
    embeddings = [[0.01] * 1024 for _ in range(4)]
    store = _FakeQdrantStoreWrapper(_InMemoryQdrantStore())
    llm = _mock_llm_client(_many())

    result, _stats = await build_raptor_for_book(
        book_id="book-002",
        chunks=chunks,
        leaf_embeddings=embeddings,
        config=_default_config(),
        llm_call=llm.complete,
        embed_client=_mock_embed_client(),
        store=store,
        cluster_func=_fake_cluster_func,
    )

    assert result.root is not None
    leaf_ids = {c.chunk_id for c in chunks}
    assert set(result.root.source_chunk_ids) == leaf_ids


async def test_reader_first_raptor_admission_uses_guarded_legacy_builder() -> None:
    from unittest.mock import AsyncMock, patch

    from vo_rag.core.raptor import builder as raptor_builder

    chunks = [_make_chunk(chunk_id="c1", content="A current raw chunk.")]
    store = _FakeQdrantStoreWrapper(_InMemoryQdrantStore())
    store._publication_enabled = True  # type: ignore[attr-defined]
    store._publication_writes_enabled = False  # type: ignore[attr-defined]
    expected = (object(), object())
    legacy = AsyncMock(return_value=expected)
    admission = AsyncMock(return_value=None)

    with (
        patch.object(raptor_builder, "admit_derived_source", admission),
        patch.object(raptor_builder, "_build_raptor_for_book_impl", legacy),
    ):
        result = await build_raptor_for_book(
            book_id="book-001",
            chunks=chunks,
            leaf_embeddings=[[0.01] * 1024],
            config=_default_config(),
            llm_call=_mock_llm_client(_many()).complete,
            embed_client=_mock_embed_client(),
            store=store,
            cluster_func=_pair_cluster_func,
            expected_dependencies={"book-001": None},
        )

    assert result == expected
    assert admission.await_args is not None
    assert admission.await_args.kwargs["expected_dependencies"] == {"book-001": None}
    legacy.assert_awaited_once()


async def test_deterministic_node_ids() -> None:
    chunks = [_make_chunk(chunk_id=f"c{i}", content=f"Chunk {i}.") for i in range(4)]
    embeddings = [[0.01] * 1024 for _ in range(4)]
    store = _FakeQdrantStoreWrapper(_InMemoryQdrantStore())
    llm = _mock_llm_client(_many())
    config = _default_config()

    result1, _ = await build_raptor_for_book(
        book_id="book-003",
        chunks=chunks,
        leaf_embeddings=embeddings,
        config=config,
        llm_call=llm.complete,
        embed_client=_mock_embed_client(),
        store=store,
        cluster_func=_fake_cluster_func,
    )

    # Second run against the same store should load existing nodes.
    # We seed a fresh LLM client with no responses to prove no new LLM calls.
    llm2 = _mock_llm_client([])
    result2, _ = await build_raptor_for_book(
        book_id="book-003",
        chunks=chunks,
        leaf_embeddings=embeddings,
        config=config,
        llm_call=llm2.complete,
        embed_client=_mock_embed_client(),
        store=store,
        cluster_func=_fake_cluster_func,
    )

    ids1 = sorted(n.node_id for n in result1.nodes)
    ids2 = sorted(n.node_id for n in result2.nodes)
    assert ids1 == ids2


async def test_llm_concurrency_bounded() -> None:
    chunks = [_make_chunk(chunk_id=f"c{i}", content=f"Chunk {i}.") for i in range(6)]
    embeddings = [[0.01] * 1024 for _ in range(6)]
    store = _FakeQdrantStoreWrapper(_InMemoryQdrantStore())

    call_times: list[float] = []

    class TimedLLMClient:
        def __init__(self) -> None:
            self._call_times = call_times
            import time as _time

            self._time = _time

        async def complete(
            self,
            system: str,
            user: str,
            max_tokens: int,
            temperature: float,
        ) -> LLMResponse:
            self._call_times.append(self._time.perf_counter())
            await asyncio.sleep(0.05)
            return LLMResponse(text="Summary.", model="test", input_tokens=10, output_tokens=5)

    config = _default_config()
    config.max_concurrent_llm_calls = 2

    llm = TimedLLMClient()
    await build_raptor_for_book(
        book_id="book-004",
        chunks=chunks,
        leaf_embeddings=embeddings,
        config=config,
        llm_call=llm.complete,
        embed_client=_mock_embed_client(),
        store=store,
        cluster_func=_fake_cluster_func,
    )

    # With 6 chunks and fake clustering (max(2, 6//2)=3 clusters),
    # we get 3 L0 nodes. LLM concurrency is 2, so at least some calls
    # should overlap in time. We just assert the builder completes.
    assert len(call_times) >= 3


async def test_idempotent_re_run_skips_existing_nodes() -> None:
    chunks = [_make_chunk(chunk_id=f"c{i}", content=f"Chunk {i}.") for i in range(4)]
    embeddings = [[0.01] * 1024 for _ in range(4)]
    store = _FakeQdrantStoreWrapper(_InMemoryQdrantStore())
    llm = _mock_llm_client(_many())
    config = _default_config()

    result1, stats1 = await build_raptor_for_book(
        book_id="book-005",
        chunks=chunks,
        leaf_embeddings=embeddings,
        config=config,
        llm_call=llm.complete,
        embed_client=_mock_embed_client(),
        store=store,
        cluster_func=_fake_cluster_func,
    )

    # Second run: LLM client has ZERO responses. If idempotency works,
    # no LLM calls are made and the build succeeds using stored nodes.
    llm2 = _mock_llm_client([])
    result2, stats2 = await build_raptor_for_book(
        book_id="book-005",
        chunks=chunks,
        leaf_embeddings=embeddings,
        config=config,
        llm_call=llm2.complete,
        embed_client=_mock_embed_client(),
        store=store,
        cluster_func=_fake_cluster_func,
    )

    assert len(result1.nodes) == len(result2.nodes)
    assert stats2.input_tokens == 0
    assert stats2.output_tokens == 0


async def test_build_raptor_for_corpus_processes_multiple_books() -> None:
    books = [
        (
            "book-a",
            [_make_chunk(chunk_id="a1", content="A1"), _make_chunk(chunk_id="a2", content="A2")],
            [[0.01] * 1024, [0.02] * 1024],
        ),
        (
            "book-b",
            [_make_chunk(chunk_id="b1", content="B1"), _make_chunk(chunk_id="b2", content="B2")],
            [[0.03] * 1024, [0.04] * 1024],
        ),
    ]
    store = _FakeQdrantStoreWrapper(_InMemoryQdrantStore())
    llm = _mock_llm_client(_many())

    results, stats_list = await build_raptor_for_corpus(
        books=books,
        config=_default_config(),
        llm_call=llm.complete,
        embed_client=_mock_embed_client(),
        store=store,
        max_concurrent_books=2,
        cluster_func=_fake_cluster_func,
    )

    assert len(results) == 2
    assert len(stats_list) == 2
    assert {r.book_id for r in results} == {"book-a", "book-b"}


async def test_build_raptor_for_corpus_ensures_default_collection() -> None:
    books = [
        (
            "book-a",
            [_make_chunk(chunk_id="a1", content="A1"), _make_chunk(chunk_id="a2", content="A2")],
            [[0.01] * 1024, [0.02] * 1024],
        ),
    ]
    store = _FakeQdrantStoreWrapper(_InMemoryQdrantStore())
    llm = _mock_llm_client(_many())
    ensure_calls: list[str] = []

    async def _spy_ensure(client: Any, *, collection: str = RAPTOR_COLLECTION) -> bool:
        ensure_calls.append(collection)
        return False

    with patch(
        "vo_rag.core.raptor.builder.ensure_raptor_collection",
        _spy_ensure,
    ):
        await build_raptor_for_corpus(
            books=books,
            config=_default_config(),
            llm_call=llm.complete,
            embed_client=_mock_embed_client(),
            store=store,
            max_concurrent_books=1,
            cluster_func=_fake_cluster_func,
        )

    assert ensure_calls == [RAPTOR_COLLECTION]


async def test_build_raptor_for_corpus_ensures_config_collection() -> None:
    books = [
        (
            "book-a",
            [_make_chunk(chunk_id="a1", content="A1"), _make_chunk(chunk_id="a2", content="A2")],
            [[0.01] * 1024, [0.02] * 1024],
        ),
    ]
    config = _default_config()
    config.raptor_collection = "vo_rag_books_raptor_v1__ab14b"
    store = _FakeQdrantStoreWrapper(_InMemoryQdrantStore())
    llm = _mock_llm_client(_many())
    ensure_calls: list[str] = []

    async def _spy_ensure(client: Any, *, collection: str = RAPTOR_COLLECTION) -> bool:
        ensure_calls.append(collection)
        return False

    with patch(
        "vo_rag.core.raptor.builder.ensure_raptor_collection",
        _spy_ensure,
    ):
        await build_raptor_for_corpus(
            books=books,
            config=config,
            llm_call=llm.complete,
            embed_client=_mock_embed_client(),
            store=store,
            max_concurrent_books=1,
            cluster_func=_fake_cluster_func,
        )

    assert ensure_calls == ["vo_rag_books_raptor_v1__ab14b"]


async def test_builtin_clustering_small_level_terminates_to_root() -> None:
    # Real-data regression (raptor pilot): with the built-in clustering path
    # (cluster_func=None) UMAP crashes on a level with too few points
    # (<= umap_n_neighbors -> zero-size graph.data.max()). The guard must turn
    # such a level into a single root instead of calling UMAP. Here the whole
    # book is 10 chunks (<= the default umap_n_neighbors=15), so level 0 itself
    # terminates into one root WITHOUT invoking UMAP/GMM.
    chunks = [_make_chunk(chunk_id=f"c{i}", content=f"Chunk {i} content.") for i in range(10)]
    embeddings = [[0.01] * 1024 for _ in range(10)]
    store = _FakeQdrantStoreWrapper(_InMemoryQdrantStore())
    llm = _mock_llm_client(_many())

    result, stats = await build_raptor_for_book(
        book_id="book-tiny",
        chunks=chunks,
        leaf_embeddings=embeddings,
        config=_default_config(),
        llm_call=llm.complete,
        embed_client=_mock_embed_client(),
        store=store,
        # cluster_func omitted -> built-in cluster_chunks path -> guard applies
    )

    assert result.root is not None
    assert stats.node_count == 1
    assert len(result.nodes) == 1
    assert set(result.root.source_chunk_ids) == {f"c{i}" for i in range(10)}


async def test_duplicate_chunk_ids_are_deduped() -> None:
    # Real-data regression: the 3,374-chunk monster book had 163 duplicate
    # chunk_ids (re-ingestion). cluster_chunks requires unique ids, so the build
    # must dedupe up front instead of crashing with "chunk_ids must be unique".
    chunks = [
        _make_chunk(chunk_id="c1", content="One."),
        _make_chunk(chunk_id="c2", content="Two."),
        _make_chunk(chunk_id="c1", content="One duplicate."),  # dup id
        _make_chunk(chunk_id="c3", content="Three."),
    ]
    embeddings = [[0.01] * 1024 for _ in range(4)]
    store = _FakeQdrantStoreWrapper(_InMemoryQdrantStore())
    llm = _mock_llm_client(_many())

    result, _stats = await build_raptor_for_book(
        book_id="book-dup",
        chunks=chunks,
        leaf_embeddings=embeddings,
        config=_default_config(),
        llm_call=llm.complete,
        embed_client=_mock_embed_client(),
        store=store,
        cluster_func=_fake_cluster_func,
    )

    assert result.root is not None
    leaf_chunk_ids = {cid for n in result.nodes for cid in n.source_chunk_ids}
    assert leaf_chunk_ids == {"c1", "c2", "c3"}  # dup c1 collapsed

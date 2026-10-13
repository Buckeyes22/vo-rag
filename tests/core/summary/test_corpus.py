"""Tests for the summary corpus-level driver ``build_summary_for_corpus``."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import patch

import pytest

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.embedding.schemas import SparseVec
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.storage.qdrant import QdrantStore
from vo_rag.core.summary.builder import (
    BuildConfig,
    SummaryCorpusStats,
    SummaryTreeBuilder,
    build_summary_for_corpus,
)
from vo_rag.core.summary.persistence import SUMMARY_TREE_COLLECTION

# ---------------------------------------------------------------------------
# Shared test doubles (mirrored from tests/core/summary/test_builder.py)
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    chunk_id: str,
    content: str,
    section_path: str,
    page_number: int | None = None,
) -> RetrievalChunk:
    """Build a minimal RetrievalChunk."""
    return RetrievalChunk(
        chunk_id=chunk_id,
        chunk_type="text",
        document_id="book-001",
        source_id="source-001",
        page_number=page_number,
        page_range=None,
        section_path=section_path,
        section_title="Test Section",
        content=content,
        embedding_text=f"{section_path}\n\n{content}",
        token_count=max(1, len(content.split())),
        has_image=False,
        image_id=None,
        image_ref=None,
        vlm_description=None,
        metadata={},
    )


class _InMemoryQdrantClient:
    """Dict-backed AsyncQdrantClient test double — no unittest.mock."""

    def __init__(self) -> None:
        self._points: dict[str, dict[str, Any]] = {}
        self._get_collections_calls: list[Any] = []

    async def upsert(self, collection_name: str, points: list[Any], wait: bool = True) -> None:
        for point in points:
            self._points[point.id] = {
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

        return [_Point(p) for p in list(self._points.values())[:limit]], None

    async def get_collections(self) -> Any:
        self._get_collections_calls.append(True)

        class _Collections:
            collections: list[Any] = []

        return _Collections()

    async def create_collection(
        self,
        collection_name: str,
        vectors_config: Any = None,
        sparse_vectors_config: Any = None,
    ) -> None:
        pass

    async def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema: Any = None,
    ) -> None:
        pass


class _InMemoryStore(QdrantStore):
    """QdrantStore subclass backed by _InMemoryQdrantClient."""

    def __init__(self) -> None:
        # Bypass QdrantStore.__init__ (it would try to construct a real
        # AsyncQdrantClient); we just need an object that exposes the
        # in-memory store via the same attribute name the parent class
        # uses for persistence and selective-expansion seams.
        self._client = _InMemoryQdrantClient()


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
                "sparse": [SparseVec(indices=[0, 1, 2], values=[0.1, 0.2, 0.3])] * len(texts),
                "colbert": [[[0.05] * 1024]] * len(texts) if return_colbert else [],
            }

    return FakeEmbedClient()


def _mock_llm_client(responses: list[LLMResponse]) -> Any:
    """Return a sequential-response LLM client consuming from a pre-seeded list."""

    class FakeLLMClient:
        def __init__(self, resp_list: list[LLMResponse]) -> None:
            self._responses = resp_list
            self._calls: list[dict[str, Any]] = []

        async def complete(
            self,
            *,
            system: str,
            messages: list[dict[str, Any]],
            max_tokens: int,
            temperature: float,
            extra: dict[str, Any] | None = None,
        ) -> LLMResponse:
            self._calls.append(
                {
                    "system": system,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            return self._responses.pop(0)

    return FakeLLMClient(responses)


def _resp(text: str = "Summary.", input_tokens: int = 10, output_tokens: int = 5) -> LLMResponse:
    """Shorthand to build a LLMResponse for tests."""
    return LLMResponse(
        text=text, model="test-model", input_tokens=input_tokens, output_tokens=output_tokens
    )


def _many(
    text: str = "Summary.", input_tokens: int = 10, output_tokens: int = 5
) -> list[LLMResponse]:
    """A comfortably-large response pool."""
    return [_resp(text, input_tokens, output_tokens) for _ in range(64)]


def _default_config() -> BuildConfig:
    return BuildConfig(
        summary_model="test-model",
        embedding_model="BGE-M3",
        embedding_version="test-v1",
        source_collection_version="test-v1",
    )


def _builder(llm: Any, config: BuildConfig | None = None) -> SummaryTreeBuilder:
    return SummaryTreeBuilder(
        llm_client=llm,
        embed_client=_mock_embed_client(),
        store=_InMemoryStore(),
        config=config or _default_config(),
    )


# ---------------------------------------------------------------------------
# Corpus driver tests
# ---------------------------------------------------------------------------


class TestBuildSummaryForCorpus:
    @pytest.mark.asyncio
    async def test_builds_and_aggregates_multiple_books(self) -> None:
        """Corpus driver over 2 books builds per-book trees and aggregates nodes."""
        book1_chunks = [
            _make_chunk(chunk_id="b1c1", content="Book 1 content.", section_path="[B1 > Ch1]"),
        ]
        book2_chunks = [
            _make_chunk(chunk_id="b2c1", content="Book 2 content.", section_path="[B2 > Ch1]"),
        ]
        books: list[tuple[str, list[RetrievalChunk]]] = [
            ("book-001", book1_chunks),
            ("book-002", book2_chunks),
        ]

        builder = _builder(_mock_llm_client(_many()))
        all_nodes, all_stats = await build_summary_for_corpus(
            books=books,
            builder=builder,
            max_concurrent_books=2,
        )

        # Each book gets at least a leaf + book root.
        book1_nodes = [n for n in all_nodes if n.book_id == "book-001"]
        book2_nodes = [n for n in all_nodes if n.book_id == "book-002"]
        assert len(book1_nodes) >= 2
        assert len(book2_nodes) >= 2

        assert len(all_stats) == 2
        stat_by_book = {s.book_id: s for s in all_stats}
        assert "book-001" in stat_by_book
        assert "book-002" in stat_by_book
        assert stat_by_book["book-001"].node_count == len(book1_nodes)
        assert stat_by_book["book-002"].node_count == len(book2_nodes)
        assert all(isinstance(s, SummaryCorpusStats) for s in all_stats)
        assert all(s.wall_time_ms >= 0 for s in all_stats)

    @pytest.mark.asyncio
    async def test_bounded_concurrency_respected(self) -> None:
        """max_concurrent_books=1 forces sequential book processing."""
        call_times: list[float] = []

        class TimedLLMClient:
            def __init__(self) -> None:
                self._call_times = call_times

            async def complete(
                self,
                *,
                system: str,
                messages: list[dict[str, Any]],
                max_tokens: int,
                temperature: float,
                extra: dict[str, Any] | None = None,
            ) -> LLMResponse:
                self._call_times.append(time.monotonic())
                await asyncio.sleep(0.05)
                return LLMResponse(text="Summary.", model="test", input_tokens=10, output_tokens=5)

        book1_chunks = [
            _make_chunk(chunk_id="b1c1", content="Book 1 content.", section_path="[B1 > Ch1]"),
        ]
        book2_chunks = [
            _make_chunk(chunk_id="b2c1", content="Book 2 content.", section_path="[B2 > Ch1]"),
        ]
        books: list[tuple[str, list[RetrievalChunk]]] = [
            ("book-001", book1_chunks),
            ("book-002", book2_chunks),
        ]

        builder = _builder(
            TimedLLMClient(),
            BuildConfig(
                summary_model="test-model",
                embedding_model="BGE-M3",
                embedding_version="test-v1",
                source_collection_version="test-v1",
                leaf_min_chunks=1,
                leaf_max_chunks=1,
                max_fanout=10,
            ),
        )
        t0 = time.monotonic()
        await build_summary_for_corpus(
            books=books,
            builder=builder,
            max_concurrent_books=1,
        )
        elapsed = time.monotonic() - t0

        # With max_concurrent_books=1, each book's LLM calls are sequential.
        # Two books × at least 2 LLM calls each × 0.05s sleep = ~0.2s.
        assert elapsed >= 0.15, f"Expected sequential processing, elapsed={elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_ensure_collection_called_once(self) -> None:
        """ensure_summary_tree_collection is called exactly once at start."""
        book1_chunks = [
            _make_chunk(chunk_id="b1c1", content="Book 1 content.", section_path="[B1 > Ch1]"),
        ]
        books: list[tuple[str, list[RetrievalChunk]]] = [("book-001", book1_chunks)]

        builder = _builder(_mock_llm_client(_many()))

        ensure_calls: list[str] = []

        async def _spy_ensure(client: Any, *, collection: str = SUMMARY_TREE_COLLECTION) -> bool:
            ensure_calls.append(collection)
            return False

        with patch(
            "vo_rag.core.summary.builder.ensure_summary_tree_collection",
            _spy_ensure,
        ):
            await build_summary_for_corpus(
                books=books,
                builder=builder,
                max_concurrent_books=1,
            )

        assert len(ensure_calls) == 1
        assert ensure_calls[0] == SUMMARY_TREE_COLLECTION

    @pytest.mark.asyncio
    async def test_ensure_collection_uses_builder_config_collection(self) -> None:
        book1_chunks = [
            _make_chunk(chunk_id="b1c1", content="Book 1 content.", section_path="[B1 > Ch1]"),
        ]
        books: list[tuple[str, list[RetrievalChunk]]] = [("book-001", book1_chunks)]
        config = _default_config()
        config.summary_collection = "vo_rag_books_summary_tree_v1__ab14b"
        builder = _builder(_mock_llm_client(_many()), config=config)
        ensure_calls: list[str] = []

        async def _spy_ensure(client: Any, *, collection: str = SUMMARY_TREE_COLLECTION) -> bool:
            ensure_calls.append(collection)
            return False

        with patch(
            "vo_rag.core.summary.builder.ensure_summary_tree_collection",
            _spy_ensure,
        ):
            await build_summary_for_corpus(
                books=books,
                builder=builder,
                max_concurrent_books=1,
            )

        assert ensure_calls == ["vo_rag_books_summary_tree_v1__ab14b"]

    @pytest.mark.asyncio
    async def test_empty_corpus_returns_empty(self) -> None:
        """Empty books list returns empty nodes and stats."""
        builder = _builder(_mock_llm_client(_many()))
        all_nodes, all_stats = await build_summary_for_corpus(
            books=[],
            builder=builder,
            max_concurrent_books=2,
        )
        assert all_nodes == []
        assert all_stats == []

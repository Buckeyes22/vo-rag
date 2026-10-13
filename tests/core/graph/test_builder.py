from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.graph.builder import (
    GraphBuildConfig,
    GraphBuildStats,
    build_graph_for_book,
    dedupe_edges_across_books,
    dedupe_nodes_across_books,
    persist_graph,
)
from vo_rag.core.graph.extraction import GraphExtractionConfig
from vo_rag.core.graph.prompts import GRAPH_EXTRACTION_PROMPT_HASH
from vo_rag.core.graph.schemas import (
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
    edge_id_for,
    node_id_for,
)
from vo_rag.core.llm.response import LLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime.now(UTC)
_EARLIER = _NOW - timedelta(hours=1)


def _make_node(
    *,
    label: str,
    node_type: GraphNodeType = GraphNodeType.CONCEPT,
    book_ids: list[str] | None = None,
    chunk_ids: list[str] | None = None,
    aliases: list[str] | None = None,
    created_at: datetime | None = None,
) -> GraphNode:
    return GraphNode(
        node_id=node_id_for(node_type=node_type, canonical_label=label),
        node_type=node_type,
        label=label,
        aliases=aliases or [],
        book_ids=book_ids or ["book-001"],
        chunk_ids=chunk_ids or [],
        created_at=created_at or _NOW,
    )


def _make_edge(
    *,
    source_id: str,
    target_id: str,
    edge_type: GraphEdgeType = GraphEdgeType.RELATES_TO,
    evidence_chunk_ids: list[str] | None = None,
    weight: float = 1.0,
) -> GraphEdge:
    return GraphEdge(
        edge_id=edge_id_for(
            source_node_id=source_id,
            target_node_id=target_id,
            edge_type=edge_type,
        ),
        source_node_id=source_id,
        target_node_id=target_id,
        edge_type=edge_type,
        book_ids=["book-1"],
        evidence_chunk_ids=evidence_chunk_ids or [],
        weight=weight,
        created_at=_NOW,
    )


def _make_chunk(chunk_id: str, content: str = "Content.") -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        chunk_type="text",
        document_id="doc-001",
        source_id="src-001",
        page_number=None,
        page_range=None,
        section_path="[Book > Chapter > Section]",
        section_title="Section",
        content=content,
        embedding_text=f"[Book > Chapter > Section]\n\n{content}",
        token_count=max(1, len(content.split())),
        has_image=False,
        metadata={},
    )


def _make_build_config() -> GraphBuildConfig:
    return GraphBuildConfig(
        extraction_config=GraphExtractionConfig(
            prompt_hash=GRAPH_EXTRACTION_PROMPT_HASH,
        ),
        chunks_per_extraction_call=30,
    )


@dataclass
class _FakePoint:
    id: str
    payload: dict[str, Any]
    vector: dict[str, Any] = field(default_factory=dict)


class _FakeQdrantClient:
    def __init__(self) -> None:
        self.points: list[_FakePoint] = []

    async def upsert(
        self,
        *,
        collection_name: str,
        points: list[Any],
        wait: bool = True,
    ) -> None:
        self.points.extend(
            _FakePoint(id=str(point.id), payload=dict(point.payload), vector=dict(point.vector))
            for point in points
        )


class _FakeStore:
    source_publication_enabled = False

    def __init__(self) -> None:
        self._client = _FakeQdrantClient()


class _RecordingEmbeddingClient:
    def __init__(
        self,
        *,
        dense: list[list[float]] | None = None,
        colbert: list[list[list[float]]] | None = None,
    ) -> None:
        self.calls: list[tuple[list[str], bool, bool, bool]] = []
        self._dense = dense
        self._colbert = colbert

    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = False,
    ) -> dict[str, Any]:
        self.calls.append((texts, return_dense, return_sparse, return_colbert))
        result: dict[str, Any] = {}
        if return_dense:
            result["dense"] = self._dense or [[0.1] * 1024 for _ in texts]
        if return_sparse:
            result["sparse"] = [{0: 0.5} for _ in texts]
        if return_colbert:
            result["colbert"] = self._colbert or [[[0.1] * 1024] for _ in texts]
        return result


# ---------------------------------------------------------------------------
# dedupe_nodes_across_books
# ---------------------------------------------------------------------------


class TestDedupeNodesAcrossBooks:
    def test_dedupe_nodes_across_books_unions_book_ids(self) -> None:
        node_a = _make_node(label="Camber", book_ids=["book-001"], chunk_ids=["c1"])
        node_b = _make_node(label="Camber", book_ids=["book-002"], chunk_ids=["c2"])

        result = dedupe_nodes_across_books({"book-001": [node_a], "book-002": [node_b]})

        assert len(result) == 1
        assert set(result[0].book_ids) == {"book-001", "book-002"}

    def test_dedupe_nodes_across_books_unions_chunk_ids(self) -> None:
        node_a = _make_node(label="Slip Angle", book_ids=["book-001"], chunk_ids=["c1", "c2"])
        node_b = _make_node(label="Slip Angle", book_ids=["book-002"], chunk_ids=["c2", "c3"])

        result = dedupe_nodes_across_books({"book-001": [node_a], "book-002": [node_b]})

        assert set(result[0].chunk_ids) == {"c1", "c2", "c3"}

    def test_dedupe_nodes_across_books_unions_aliases(self) -> None:
        node_a = _make_node(label="Oversteer", aliases=["snap oversteer"], chunk_ids=["c1"])
        node_b = _make_node(label="Oversteer", aliases=["power oversteer"], chunk_ids=["c2"])

        result = dedupe_nodes_across_books({"book-001": [node_a], "book-002": [node_b]})

        assert "snap oversteer" in result[0].aliases
        assert "power oversteer" in result[0].aliases

    def test_dedupe_nodes_keeps_earliest_created_at(self) -> None:
        node_early = _make_node(label="Downforce", chunk_ids=["c1"], created_at=_EARLIER)
        node_late = _make_node(label="Downforce", chunk_ids=["c2"], created_at=_NOW)

        result = dedupe_nodes_across_books({"book-001": [node_early], "book-002": [node_late]})

        assert result[0].created_at == _EARLIER

    def test_dedupe_nodes_single_book_no_change(self) -> None:
        node = _make_node(label="Aero", chunk_ids=["c1"])
        result = dedupe_nodes_across_books({"book-001": [node]})
        assert len(result) == 1
        assert result[0].node_id == node.node_id


# ---------------------------------------------------------------------------
# dedupe_edges_across_books
# ---------------------------------------------------------------------------


class TestDedupeEdgesAcrossBooks:
    def _shared_edge_pair(self) -> tuple[GraphEdge, GraphEdge]:
        src = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="Braking")
        tgt = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="ABS")
        edge_a = _make_edge(
            source_id=src,
            target_id=tgt,
            evidence_chunk_ids=["c1"],
            weight=2.0,
        )
        edge_b = _make_edge(
            source_id=src,
            target_id=tgt,
            evidence_chunk_ids=["c2"],
            weight=3.0,
        )
        return edge_a, edge_b

    def test_dedupe_edges_across_books_unions_evidence_chunk_ids(self) -> None:
        edge_a, edge_b = self._shared_edge_pair()
        result = dedupe_edges_across_books({"book-001": [edge_a], "book-002": [edge_b]})
        assert len(result) == 1
        assert set(result[0].evidence_chunk_ids) == {"c1", "c2"}

    def test_dedupe_edges_across_books_sums_weight_with_cap(self) -> None:
        edge_a, edge_b = self._shared_edge_pair()
        result = dedupe_edges_across_books({"book-001": [edge_a], "book-002": [edge_b]})
        assert result[0].weight == pytest.approx(5.0)

    def test_dedupe_edges_weight_cap_at_100(self) -> None:
        src = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="X")
        tgt = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="Y")
        edge_a = _make_edge(source_id=src, target_id=tgt, weight=70.0)
        edge_b = _make_edge(source_id=src, target_id=tgt, weight=70.0)
        result = dedupe_edges_across_books({"b1": [edge_a], "b2": [edge_b]})
        assert result[0].weight == pytest.approx(100.0)

    def test_dedupe_edges_single_book_no_change(self) -> None:
        src = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="Draft")
        tgt = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="Slipstream")
        edge = _make_edge(source_id=src, target_id=tgt, weight=1.5)
        result = dedupe_edges_across_books({"b1": [edge]})
        assert len(result) == 1
        assert result[0].weight == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# build_graph_for_book (async)
# ---------------------------------------------------------------------------


class TestBuildGraphForBook:
    @staticmethod
    def _make_llm_response_json(chunk_ids: list[str]) -> str:
        nodes = [
            {
                "node_type": "concept",
                "label": f"Concept_{cid}",
                "aliases": [],
                "chunk_ids": [cid],
            }
            for cid in chunk_ids
        ]
        return json.dumps({"nodes": nodes, "edges": []})

    @staticmethod
    def _make_single_node_response(label: str) -> str:
        return json.dumps(
            {
                "nodes": [
                    {
                        "node_type": "concept",
                        "label": label,
                        "aliases": [],
                        "chunk_ids": ["c0"],
                    }
                ],
                "edges": [],
            }
        )

    @pytest.mark.asyncio
    async def test_build_graph_for_book_chunks_in_batches(self) -> None:
        """With batch size 2 and 4 chunks, llm_call should be invoked 2 times."""
        chunks = [_make_chunk(f"c{i}", f"Content about topic {i}.") for i in range(4)]
        call_count = 0

        async def mock_llm(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            # Return empty graph — minimal valid response
            return LLMResponse(
                text=json.dumps({"nodes": [], "edges": []}),
                model="qwen3.6-27b-fp8",
                input_tokens=10,
                output_tokens=5,
            )

        config = GraphBuildConfig(
            extraction_config=GraphExtractionConfig(
                prompt_hash=GRAPH_EXTRACTION_PROMPT_HASH,
            ),
            chunks_per_extraction_call=2,
        )

        _result, stats = await build_graph_for_book(
            book_id="book-001",
            title="T",
            author="A",
            chunks=chunks,
            config=config,
            llm_call=mock_llm,
        )

        assert call_count == 2
        assert stats.n_skipped_batches == 0

    @pytest.mark.asyncio
    async def test_build_graph_for_book_returns_combined_stats(self) -> None:
        chunks = [_make_chunk("c1", "Aerodynamic drag acts opposite to the direction of motion.")]
        payload = json.dumps(
            {
                "nodes": [
                    {
                        "node_type": "concept",
                        "label": "Aerodynamic Drag",
                        "aliases": [],
                        # Model references the compact batch-local label (c0), not
                        # the real chunk_id; the parser maps it back to "c1".
                        "chunk_ids": ["c0"],
                    }
                ],
                "edges": [],
            }
        )

        async def mock_llm(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            return LLMResponse(
                text=payload,
                model="qwen3.6-27b-fp8",
                input_tokens=50,
                output_tokens=30,
            )

        result, stats = await build_graph_for_book(
            book_id="book-001",
            title="Race Aerodynamics",
            author="Sam Ellis",
            chunks=chunks,
            config=_make_build_config(),
            llm_call=mock_llm,
        )

        assert isinstance(stats, GraphBuildStats)
        assert stats.book_id == "book-001"
        assert stats.n_nodes_extracted >= 0
        assert stats.n_nodes_after_dedup <= stats.n_nodes_extracted
        assert stats.wall_time_ms >= 0
        assert stats.n_skipped_batches == 0
        assert len(result.nodes) == 1

    @pytest.mark.asyncio
    async def test_build_graph_for_book_skips_failed_batch_and_preserves_order(self) -> None:
        chunks = [
            _make_chunk("c0", "First survivor."),
            _make_chunk("c1", "First survivor pair."),
            _make_chunk("c2", "Bad batch marker."),
            _make_chunk("c3", "Bad batch marker pair."),
            _make_chunk("c4", "Second survivor."),
            _make_chunk("c5", "Second survivor pair."),
        ]
        config = GraphBuildConfig(
            extraction_config=GraphExtractionConfig(prompt_hash=GRAPH_EXTRACTION_PROMPT_HASH),
            chunks_per_extraction_call=2,
        )

        async def mock_llm(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            if "Bad batch marker" in user:
                raise RuntimeError("batch exhausted")
            if "Second survivor" in user:
                label = "Second Survivor Concept"
            else:
                label = "First Survivor Concept"
            return LLMResponse(
                text=self._make_single_node_response(label),
                model="qwen3.6-27b-fp8",
                input_tokens=10,
                output_tokens=5,
            )

        result, stats = await build_graph_for_book(
            book_id="book-skip-one",
            title="T",
            author="A",
            chunks=chunks,
            config=config,
            llm_call=mock_llm,
        )

        assert [node.label for node in result.nodes] == [
            "First Survivor Concept",
            "Second Survivor Concept",
        ]
        assert [node.chunk_ids for node in result.nodes] == [["c0"], ["c4"]]
        assert stats.n_skipped_batches == 1
        assert stats.n_nodes_extracted == 2

    @pytest.mark.asyncio
    async def test_build_graph_for_book_all_failed_batches_raise_clear_safety_error(self) -> None:
        chunks = [_make_chunk(f"c{i}", f"Bad batch marker {i}.") for i in range(4)]
        config = GraphBuildConfig(
            extraction_config=GraphExtractionConfig(prompt_hash=GRAPH_EXTRACTION_PROMPT_HASH),
            chunks_per_extraction_call=2,
        )

        async def mock_llm(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            raise RuntimeError("batch exhausted")

        with pytest.raises(RuntimeError, match="all graph extraction batches were skipped"):
            await build_graph_for_book(
                book_id="book-skip-all",
                title="T",
                author="A",
                chunks=chunks,
                config=config,
                llm_call=mock_llm,
            )

    @pytest.mark.asyncio
    async def test_build_graph_for_book_skip_fraction_over_threshold_raises(self) -> None:
        chunks = [
            _make_chunk("c0", "Survivor batch."),
            _make_chunk("c1", "Survivor batch pair."),
            _make_chunk("c2", "Bad batch marker A."),
            _make_chunk("c3", "Bad batch marker A pair."),
            _make_chunk("c4", "Bad batch marker B."),
            _make_chunk("c5", "Bad batch marker B pair."),
        ]
        config = GraphBuildConfig(
            extraction_config=GraphExtractionConfig(prompt_hash=GRAPH_EXTRACTION_PROMPT_HASH),
            chunks_per_extraction_call=2,
            max_skip_fraction=0.5,
        )

        async def mock_llm(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            if "Bad batch marker" in user:
                raise RuntimeError("batch exhausted")
            return LLMResponse(
                text=self._make_single_node_response("Survivor Concept"),
                model="qwen3.6-27b-fp8",
                input_tokens=10,
                output_tokens=5,
            )

        with pytest.raises(RuntimeError, match="graph batch skip fraction"):
            await build_graph_for_book(
                book_id="book-skip-threshold",
                title="T",
                author="A",
                chunks=chunks,
                config=config,
                llm_call=mock_llm,
            )

    @pytest.mark.asyncio
    async def test_build_graph_for_book_orients_directed_edges_before_returning(self) -> None:
        chunks = [_make_chunk("c1", "The Honda EU2000i Generator is a compact generator.")]
        payload = json.dumps(
            {
                "nodes": [
                    {
                        "node_type": "component",
                        "label": "Generator",
                        "aliases": [],
                        "chunk_ids": ["c0"],
                    },
                    {
                        "node_type": "component",
                        "label": "Honda EU2000i Generator",
                        "aliases": [],
                        "chunk_ids": ["c0"],
                    },
                ],
                "edges": [
                    {
                        "source_label": "Generator",
                        "source_type": "component",
                        "target_label": "Honda EU2000i Generator",
                        "target_type": "component",
                        "edge_type": "is_a",
                        "evidence_chunk_ids": ["c0"],
                    }
                ],
            }
        )

        async def mock_llm(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> LLMResponse:
            return LLMResponse(
                text=payload,
                model="qwen3.6-27b-fp8",
                input_tokens=50,
                output_tokens=30,
            )

        result, stats = await build_graph_for_book(
            book_id="book-001",
            title="Generator Manual",
            author="Honda",
            chunks=chunks,
            config=_make_build_config(),
            llm_call=mock_llm,
        )

        labels = {node.node_id: node.label for node in result.nodes}
        assert len(result.edges) == 1
        assert labels[result.edges[0].source_node_id] == "Honda EU2000i Generator"
        assert labels[result.edges[0].target_node_id] == "Generator"
        assert stats.direction_coverage == 1.0
        assert stats.n_low_confidence_edges == 0


class TestPersistGraph:
    @pytest.mark.asyncio
    async def test_publication_mode_fails_closed_before_embedding(self) -> None:
        node = _make_node(label="Publication guard")
        embedding_client = _RecordingEmbeddingClient()
        store = _FakeStore()
        store.source_publication_enabled = True

        with pytest.raises(ValueError, match="complete shared graph dependency roster"):
            await persist_graph(
                nodes=[node],
                edges=[],
                embedding_client=embedding_client,
                store=store,
            )

        assert embedding_client.calls == []
        assert store._client.points == []

    @pytest.mark.asyncio
    async def test_persist_graph_uses_one_embedding_call_per_node_batch(self) -> None:
        node_a = _make_node(label="Anti Dive", chunk_ids=["c1"])
        node_b = _make_node(label="Brake Bias", chunk_ids=["c2"])
        embedding_client = _RecordingEmbeddingClient()
        store = _FakeStore()

        persisted = await persist_graph(
            nodes=[node_a, node_b],
            edges=[],
            embedding_client=embedding_client,
            store=store,
            embed_batch_size=32,
        )

        assert persisted == (2, 0)
        assert embedding_client.calls == [
            (["Anti Dive", "Brake Bias"], True, True, False),
        ]

    @pytest.mark.asyncio
    async def test_persist_graph_defaults_to_dense_colbert_fallback_when_response_has_no_colbert(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("VORAG_BOOK_RAG_INCLUDE_COLBERT", raising=False)
        dense = [0.25] * 1024
        node = _make_node(label="Yaw Moment", chunk_ids=["c1"])
        embedding_client = _RecordingEmbeddingClient(dense=[dense])
        store = _FakeStore()

        persisted = await persist_graph(
            nodes=[node],
            edges=[],
            embedding_client=embedding_client,
            store=store,
        )

        assert persisted == (1, 0)
        assert embedding_client.calls == [
            (["Yaw Moment"], True, True, False),
        ]
        assert store._client.points[0].vector["colbert"] == [dense]
        assert store._client.points[0].vector["colbert"]

    @pytest.mark.asyncio
    async def test_persist_graph_uses_present_colbert_rows(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VORAG_BOOK_RAG_INCLUDE_COLBERT", "1")
        dense = [0.25] * 1024
        colbert = [[[0.75] * 1024, [0.5] * 1024]]
        node = _make_node(label="Trail Braking", chunk_ids=["c1"])
        embedding_client = _RecordingEmbeddingClient(dense=[dense], colbert=colbert)
        store = _FakeStore()

        persisted = await persist_graph(
            nodes=[node],
            edges=[],
            embedding_client=embedding_client,
            store=store,
        )

        assert persisted == (1, 0)
        assert embedding_client.calls == [
            (["Trail Braking"], True, True, True),
        ]
        assert store._client.points[0].vector["colbert"] == colbert[0]

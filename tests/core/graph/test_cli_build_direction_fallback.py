from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.graph.builder import GraphBuildStats
from vo_rag.core.graph.direction import orient_edges
from vo_rag.core.graph.extraction import GraphExtractionResult
from vo_rag.core.graph.schemas import GraphEdge, GraphEdgeType, GraphNode, GraphNodeType


def _fake_chunk(source_id: str) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id="c1",
        chunk_type="text",
        document_id=source_id,
        source_id=source_id,
        page_number=1,
        page_range="1",
        section_path="[Book > Ch1]",
        section_title="Ch1",
        chapter_path="[Book > Ch1]",
        content="hello world",
        embedding_text="hello world",
        token_count=2,
        metadata={"title": "Test Book", "author": "Test Author", "domain_tags": ["tag-a"]},
    )


def _fake_settings() -> Any:
    return SimpleNamespace(
        qdrant=SimpleNamespace(
            url="http://localhost:6333",
            api_key="",
            prefer_grpc=True,
        ),
        embedding=SimpleNamespace(
            batch_url="http://localhost:8001",
            timeout_s=30,
        ),
    )


class _FakeEmbedder:
    async def aclose(self) -> None:
        pass


class _FakeLocalLLM:
    async def aclose(self) -> None:
        pass


class _FakeStore:
    _client: Any = None


@pytest.mark.asyncio
async def test_build_graph_impl_persists_local_reorientation_without_post_orientation_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.cli.build import _build_graph_impl

    monkeypatch.delenv("VORAG_GRAPH_27B_FALLBACK", raising=False)
    monkeypatch.delenv("VORAG_GRAPH_LOCAL_REORIENT", raising=False)
    monkeypatch.setenv("VORAG_GRAPH_DIRECTION_MIN_COVERAGE", "0.85")

    tool = GraphNode(
        node_id="n-tool",
        node_type=GraphNodeType.COMPONENT,
        label="Pyrometer",
        book_ids=["book-local-reorient"],
        chunk_ids=["c1"],
        aliases=[],
        created_at=datetime.now(UTC),
    )
    process = GraphNode(
        node_id="n-process",
        node_type=GraphNodeType.METHOD,
        label="Tire Temperature Check",
        book_ids=["book-local-reorient"],
        chunk_ids=["c1"],
        aliases=[],
        created_at=datetime.now(UTC),
    )
    low_source = GraphNode(
        node_id="n-low-source",
        node_type=GraphNodeType.CONCEPT,
        label="Baseline",
        book_ids=["book-local-reorient"],
        chunk_ids=["c1"],
        aliases=[],
        created_at=datetime.now(UTC),
    )
    low_target = GraphNode(
        node_id="n-low-target",
        node_type=GraphNodeType.CONCEPT,
        label="Reference",
        book_ids=["book-local-reorient"],
        chunk_ids=["c1"],
        aliases=[],
        created_at=datetime.now(UTC),
    )
    medium_edge = GraphEdge(
        edge_id="edge-medium",
        edge_type=GraphEdgeType.USED_BY,
        source_node_id=tool.node_id,
        target_node_id=process.node_id,
        book_ids=["book-local-reorient"],
        evidence_chunk_ids=["c1"],
        weight=1.0,
        created_at=datetime.now(UTC),
    )
    low_confidence_edge = GraphEdge(
        edge_id="edge-low",
        edge_type=GraphEdgeType.USED_BY,
        source_node_id=low_source.node_id,
        target_node_id=low_target.node_id,
        book_ids=["book-local-reorient"],
        evidence_chunk_ids=["c1"],
        weight=1.0,
        created_at=datetime.now(UTC),
    )
    nodes = [tool, process, low_source, low_target]
    deterministic_edges, deterministic_coverage = orient_edges(
        nodes,
        [medium_edge, low_confidence_edge],
    )
    assert deterministic_edges == [medium_edge, low_confidence_edge]
    assert deterministic_coverage.coverage == 0.5
    assert deterministic_coverage.n_low == 1
    assert deterministic_coverage.low_confidence_edge_ids == [low_confidence_edge.edge_id]

    corrected_edge = low_confidence_edge.model_copy(
        update={"source_node_id": low_target.node_id, "target_node_id": low_source.node_id}
    )
    corrected_edges = [medium_edge, corrected_edge]

    local_llm = _FakeLocalLLM()
    post_orientation_calls: list[list[GraphEdge]] = []

    def fake_post_fallback_orient_edges(
        nodes_arg: list[GraphNode],
        edges_arg: list[GraphEdge],
    ) -> Any:
        post_orientation_calls.append(edges_arg)
        return orient_edges(nodes_arg, edges_arg)

    async def fake_build_graph_for_book(
        **kwargs: Any,
    ) -> tuple[GraphExtractionResult, GraphBuildStats]:
        assert kwargs["llm_call"] == "extraction-call"
        return (
            GraphExtractionResult(
                book_id="book-local-reorient",
                nodes=nodes,
                edges=deterministic_edges,
                validation_errors=[],
                prompt_hash="h",
                input_tokens=1,
                output_tokens=1,
                wall_time_ms=1,
            ),
            GraphBuildStats(
                book_id="book-local-reorient",
                n_nodes_extracted=len(nodes),
                n_edges_extracted=len(deterministic_edges),
                n_nodes_after_dedup=len(nodes),
                n_edges_after_dedup=len(deterministic_edges),
                validation_error_count=0,
                wall_time_ms=1,
                direction_coverage=deterministic_coverage.coverage,
                n_low_confidence_edges=deterministic_coverage.n_low,
                low_confidence_edge_ids=deterministic_coverage.low_confidence_edge_ids,
            ),
        )

    async def fake_orient_low_confidence_via_llm(
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        low_confidence_edge_ids: list[str],
        llm_call: Any,
    ) -> list[GraphEdge]:
        assert nodes == [tool, process, low_source, low_target]
        assert edges == [medium_edge, low_confidence_edge]
        assert low_confidence_edge_ids == [low_confidence_edge.edge_id]
        assert llm_call == "local-reorientation-call"
        return corrected_edges

    async def fake_llm_call_from_client(client: Any, *, guided_json: Any = None) -> str:
        assert client is local_llm
        if guided_json is None:
            return "local-reorientation-call"
        return "extraction-call"

    async def fake_load_chunks(*_args: Any, **_kwargs: Any) -> list[RetrievalChunk]:
        return [_fake_chunk("book-local-reorient")]

    async def fake_ensure_graph_collections(*_args: Any, **_kwargs: Any) -> tuple[bool, bool]:
        return False, False

    captured_persist_args: dict[str, Any] = {}

    async def fake_persist_graph(**kwargs: Any) -> tuple[int, int]:
        captured_persist_args.update(kwargs)
        return len(kwargs["nodes"]), len(kwargs["edges"])

    monkeypatch.setattr("vo_rag.cli.build.get_settings", _fake_settings)
    monkeypatch.setattr("vo_rag.cli.build.QdrantStore", lambda *args, **kwargs: _FakeStore())
    monkeypatch.setattr(
        "vo_rag.cli.build.build_embedding_client",
        lambda *args, **kwargs: _FakeEmbedder(),
    )
    monkeypatch.setattr("vo_rag.cli.build.LLMClient", lambda *args, **kwargs: local_llm)
    monkeypatch.setattr("vo_rag.cli.build.build_graph_for_book", fake_build_graph_for_book)
    monkeypatch.setattr(
        "vo_rag.cli.build.orient_edges",
        fake_post_fallback_orient_edges,
        raising=False,
    )
    monkeypatch.setattr(
        "vo_rag.cli.build.orient_low_confidence_via_llm",
        fake_orient_low_confidence_via_llm,
    )
    monkeypatch.setattr("vo_rag.cli.build._llm_call_from_client", fake_llm_call_from_client)
    monkeypatch.setattr("vo_rag.cli.build._load_chunks_for_source", fake_load_chunks)
    monkeypatch.setattr("vo_rag.cli.build.ensure_graph_collections", fake_ensure_graph_collections)
    monkeypatch.setattr("vo_rag.cli.build.persist_graph", fake_persist_graph)

    summary = await _build_graph_impl("book-local-reorient")

    assert captured_persist_args["edges"] == corrected_edges
    assert post_orientation_calls == []
    assert summary["direction_coverage"] == deterministic_coverage.coverage
    assert summary["n_reoriented"] == 1
    assert summary["reorientation_status"] == "applied_local"
    assert summary["n_low_confidence_edges"] == deterministic_coverage.n_low
    assert summary["low_confidence_edge_ids"] == deterministic_coverage.low_confidence_edge_ids
    assert "direction_coverage_pre" not in summary
    assert "direction_coverage_post" not in summary

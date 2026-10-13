from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.graph.schemas import GraphEdge, GraphEdgeType, GraphNode, GraphNodeType
from vo_rag.core.llm.client import LLMClient
from vo_rag.core.storage.qdrant import QdrantStore
from vo_rag.integrations.llamaindex.build_workflow import (
    AdvancedBuildWorkflow,
    GraphBuiltEvent,
    GraphOrientedEvent,
    LayerBuiltEvent,
    RaptorBuildEvent,
    RaptorPreflightEvent,
)


class _FakeStore:
    def __init__(self) -> None:
        self._client = object()


@pytest.fixture
def workflow(monkeypatch: pytest.MonkeyPatch) -> AdvancedBuildWorkflow:
    store = _FakeStore()
    embed = AsyncMock()
    llm = AsyncMock(spec=LLMClient)
    llm_clients = cast(dict[str, LLMClient], {"text": llm, "dsi": llm})
    settings = MagicMock()
    settings.serverless = None
    wf = AdvancedBuildWorkflow(
        store=cast(QdrantStore, store),
        embedding_client=embed,
        llm_clients=llm_clients,
        settings=settings,
    )
    # Patch _persist_graph so tests never touch Qdrant
    monkeypatch.setattr(wf, "_persist_graph", AsyncMock(return_value=(2, 3)))
    return wf


# ---------------------------------------------------------------------------
# Graph direction gate
# ---------------------------------------------------------------------------


def _make_graph_built_event(
    *, coverage: float, low_confidence_edge_ids: list[str]
) -> GraphBuiltEvent:
    node = GraphNode(
        node_id="n1",
        node_type=GraphNodeType.CONCEPT,
        label="A",
        book_ids=["b1"],
        chunk_ids=["c1"],
        created_at=__import__("datetime").datetime.utcnow(),
    )
    edge = GraphEdge(
        edge_id="e1",
        source_node_id="n1",
        target_node_id="n2",
        edge_type=GraphEdgeType.PART_OF,
        book_ids=["b1"],
        evidence_chunk_ids=["c1"],
        created_at=__import__("datetime").datetime.utcnow(),
    )
    return GraphBuiltEvent(
        nodes=[node],
        edges=[edge],
        coverage=coverage,
        low_confidence_edge_ids=low_confidence_edge_ids,
        summary={"book_id": "b1", "n_low_confidence_edges": len(low_confidence_edge_ids)},
    )


@pytest.mark.asyncio
async def test_graph_gate_persists_as_is_when_coverage_ge_threshold(
    workflow: AdvancedBuildWorkflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ev = _make_graph_built_event(coverage=0.95, low_confidence_edge_ids=["e1"])
    monkeypatch.setattr(workflow, "_runpod_text_available", lambda: True)

    orient_mock = AsyncMock(return_value=ev.edges)
    monkeypatch.setattr(
        "vo_rag.integrations.llamaindex.build_workflow.orient_low_confidence_via_llm",
        orient_mock,
    )

    result = await workflow.graph_direction_gate(MagicMock(), ev)

    assert isinstance(result, GraphOrientedEvent)
    assert result.summary["direction_coverage"] == 0.95
    assert result.summary["reorientation_status"] == "not_needed"
    assert result.summary["n_reoriented"] == 0
    orient_mock.assert_not_awaited()
    cast(AsyncMock, workflow._persist_graph).assert_awaited_once_with(ev.nodes, ev.edges)


@pytest.mark.asyncio
async def test_graph_gate_uses_fallback_when_enabled_and_available(
    workflow: AdvancedBuildWorkflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow.graph_27b_fallback = True
    ev = _make_graph_built_event(coverage=0.5, low_confidence_edge_ids=["e1"])

    # Build a corrected edge list (source/target swapped)
    corrected_edge = ev.edges[0].model_copy(update={"source_node_id": "n2", "target_node_id": "n1"})
    orient_mock = AsyncMock(return_value=[corrected_edge])
    monkeypatch.setattr(
        "vo_rag.integrations.llamaindex.build_workflow.orient_low_confidence_via_llm",
        orient_mock,
    )

    fake_client = MagicMock(spec=LLMClient)
    fake_client.aclose = AsyncMock()
    monkeypatch.setattr(workflow, "_build_runpod_fallback_client", lambda: fake_client)
    monkeypatch.setattr(workflow, "_runpod_text_available", lambda: True)

    result = await workflow.graph_direction_gate(MagicMock(), ev)

    assert isinstance(result, GraphOrientedEvent)
    assert result.summary["reorientation_status"] == "applied"
    assert result.summary["n_reoriented"] == 1
    orient_mock.assert_awaited_once()
    cast(AsyncMock, workflow._persist_graph).assert_awaited_once_with(ev.nodes, [corrected_edge])
    fake_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_graph_gate_closes_fallback_client_when_orientation_fails(
    workflow: AdvancedBuildWorkflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow.graph_27b_fallback = True
    ev = _make_graph_built_event(coverage=0.5, low_confidence_edge_ids=["e1"])

    orient_mock = AsyncMock(side_effect=RuntimeError("fallback orientation failed"))
    monkeypatch.setattr(
        "vo_rag.integrations.llamaindex.build_workflow.orient_low_confidence_via_llm",
        orient_mock,
    )

    fake_client = MagicMock(spec=LLMClient)
    fake_client.aclose = AsyncMock()
    monkeypatch.setattr(workflow, "_build_runpod_fallback_client", lambda: fake_client)
    monkeypatch.setattr(workflow, "_runpod_text_available", lambda: True)

    with pytest.raises(RuntimeError, match="fallback orientation failed"):
        await workflow.graph_direction_gate(MagicMock(), ev)

    orient_mock.assert_awaited_once()
    fake_client.aclose.assert_awaited_once()
    cast(AsyncMock, workflow._persist_graph).assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_gate_persists_as_is_when_fallback_disabled(
    workflow: AdvancedBuildWorkflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow.graph_27b_fallback = False
    ev = _make_graph_built_event(coverage=0.5, low_confidence_edge_ids=["e1"])
    monkeypatch.setattr(workflow, "_runpod_text_available", lambda: True)

    orient_mock = AsyncMock(return_value=ev.edges)
    monkeypatch.setattr(
        "vo_rag.integrations.llamaindex.build_workflow.orient_low_confidence_via_llm",
        orient_mock,
    )

    result = await workflow.graph_direction_gate(MagicMock(), ev)

    assert isinstance(result, GraphOrientedEvent)
    assert result.summary["reorientation_status"] == "disabled"
    assert result.summary["n_reoriented"] == 0
    orient_mock.assert_not_awaited()
    cast(AsyncMock, workflow._persist_graph).assert_awaited_once_with(ev.nodes, ev.edges)


@pytest.mark.asyncio
async def test_graph_gate_skips_fallback_when_endpoint_unavailable(
    workflow: AdvancedBuildWorkflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback enabled but the RunPod text endpoint is DOWN -> no re-orient
    (fail-safe): status=unavailable, persist the un-reoriented graph as-is."""
    workflow.graph_27b_fallback = True
    ev = _make_graph_built_event(coverage=0.5, low_confidence_edge_ids=["e1"])
    monkeypatch.setattr(workflow, "_runpod_text_available", lambda: False)

    orient_mock = AsyncMock(return_value=ev.edges)
    monkeypatch.setattr(
        "vo_rag.integrations.llamaindex.build_workflow.orient_low_confidence_via_llm",
        orient_mock,
    )

    result = await workflow.graph_direction_gate(MagicMock(), ev)

    assert isinstance(result, GraphOrientedEvent)
    assert result.summary["reorientation_status"] == "unavailable"
    assert result.summary["n_reoriented"] == 0
    orient_mock.assert_not_awaited()
    cast(AsyncMock, workflow._persist_graph).assert_awaited_once_with(ev.nodes, ev.edges)


# ---------------------------------------------------------------------------
# RAPTOR preflight gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raptor_preflight_skips_degenerate_corpus(
    workflow: AdvancedBuildWorkflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_mock = AsyncMock(return_value={"n_nodes": 1})
    monkeypatch.setattr(workflow, "_build_raptor", build_mock)

    ev = RaptorPreflightEvent(source_id="src-1", chunks=[])
    result = await workflow.raptor_preflight_gate(MagicMock(), ev)

    assert isinstance(result, LayerBuiltEvent)
    assert result.layer == "raptor"
    assert result.summary.get("skipped") is True
    assert "too small" in result.summary.get("reason", "")
    build_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_raptor_preflight_proceeds_for_sufficient_corpus(
    workflow: AdvancedBuildWorkflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_mock = AsyncMock(return_value={"n_nodes": 1})
    monkeypatch.setattr(workflow, "_build_raptor", build_mock)

    chunks = [
        RetrievalChunk(
            chunk_id=f"c-{i}",
            chunk_type="text",
            document_id="d1",
            source_id="src-1",
            section_path="",
            section_title="",
            chapter_path="",
            content="x",
            embedding_text="x",
            token_count=1,
        )
        for i in range(3)
    ]
    ev = RaptorPreflightEvent(source_id="src-1", chunks=chunks)
    result = await workflow.raptor_preflight_gate(MagicMock(), ev)

    assert isinstance(result, RaptorBuildEvent)
    assert result.source_id == "src-1"
    assert len(result.chunks) == 3


@pytest.mark.asyncio
async def test_raptor_build_invokes_builder(
    workflow: AdvancedBuildWorkflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_mock = AsyncMock(return_value={"book_id": "src-1", "n_nodes": 5})
    monkeypatch.setattr(workflow, "_build_raptor", build_mock)

    chunks = [
        RetrievalChunk(
            chunk_id="c-1",
            chunk_type="text",
            document_id="d1",
            source_id="src-1",
            section_path="",
            section_title="",
            chapter_path="",
            content="x",
            embedding_text="x",
            token_count=1,
        )
    ]
    ev = RaptorBuildEvent(source_id="src-1", chunks=chunks)
    result = await workflow.raptor_build(MagicMock(), ev)

    assert isinstance(result, LayerBuiltEvent)
    assert result.layer == "raptor"
    assert result.summary == {"book_id": "src-1", "n_nodes": 5}
    build_mock.assert_awaited_once_with(source_id="src-1", chunks=chunks)

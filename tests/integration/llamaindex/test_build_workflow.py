from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast, get_args
from unittest.mock import AsyncMock

import pytest
from llama_index.core.workflow import StopEvent
from pydantic import ValidationError
from workflows.errors import WorkflowRuntimeError

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.graph.builder import GraphBuildStats
from vo_rag.core.graph.schemas import GraphEdge, GraphEdgeType, GraphNode, GraphNodeType
from vo_rag.core.llm.client import LLMClient
from vo_rag.core.storage.qdrant import QdrantStore
from vo_rag.integrations.llamaindex.build_workflow import (
    SUPPORTED_BUILD_LAYERS,
    AdvancedBuildWorkflow,
    BuildLayer,
    BuildLayerEvent,
    BuildStartEvent,
    BuildState,
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
    settings = object()
    wf = AdvancedBuildWorkflow(
        store=cast(QdrantStore, store),
        embedding_client=embed,
        llm_clients=llm_clients,
        settings=settings,
    )
    return wf


def test_build_layer_event_layer_literal_drives_supported_layers() -> None:
    annotation = cast(object, BuildLayerEvent.model_fields["layer"].annotation)
    assert annotation is BuildLayer
    start_layers = cast(object, BuildStartEvent.model_fields["layers"].annotation)
    assert start_layers == list[BuildLayer]
    literal_layers = frozenset(get_args(BuildLayer.__value__))

    assert literal_layers == frozenset({"dsi", "summary", "raptor", "graph"})
    assert literal_layers == SUPPORTED_BUILD_LAYERS

    with pytest.raises(ValidationError):
        BuildLayerEvent(layer=cast(Any, "unknown"))


def test_build_state_has_no_unused_layer_count() -> None:
    assert "n_layers" not in BuildState.model_fields


def _chunk(chunk_id: str, *, source_id: str = "src-1") -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        chunk_type="text",
        document_id="doc-1",
        source_id=source_id,
        section_path="Ch1",
        section_title="Chapter 1",
        chapter_path="Ch1",
        content="body",
        embedding_text="body",
        token_count=2,
    )


def _graph_stats(source_id: str, *, coverage: float = 1.0) -> GraphBuildStats:
    return GraphBuildStats(
        book_id=source_id,
        n_nodes_extracted=1,
        n_edges_extracted=1,
        n_nodes_after_dedup=1,
        n_edges_after_dedup=1,
        validation_error_count=0,
        wall_time_ms=1,
        direction_coverage=coverage,
        n_low_confidence_edges=0,
        low_confidence_edge_ids=[],
    )


def _graph_payload() -> tuple[list[GraphNode], list[GraphEdge]]:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    nodes = [
        GraphNode(
            node_id="node-1",
            node_type=GraphNodeType.CONCEPT,
            label="Active suspension",
            book_ids=["src-1"],
            chunk_ids=["c-1"],
            created_at=created_at,
        ),
        GraphNode(
            node_id="node-2",
            node_type=GraphNodeType.CONCEPT,
            label="1994 regulations",
            book_ids=["src-1"],
            chunk_ids=["c-1"],
            created_at=created_at,
        ),
    ]
    edges = [
        GraphEdge(
            edge_id="edge-1",
            source_node_id="node-1",
            target_node_id="node-2",
            edge_type=GraphEdgeType.RELATES_TO,
            book_ids=["src-1"],
            evidence_chunk_ids=["c-1"],
            created_at=created_at,
        )
    ]
    return nodes, edges


class _CollectStore:
    def __init__(self, state: BuildState) -> None:
        self._state = state

    async def get_state(self) -> BuildState:
        return self._state


class _CollectCtx:
    def __init__(self, state: BuildState) -> None:
        self.store = _CollectStore(state)
        self.collect_buffer_ids: list[str | None] = []
        self.collect_expected: list[type[Any]] = []

    def collect_events(
        self,
        ev: Any,
        expected: list[type[Any]],
        buffer_id: str | None = None,
    ) -> None:
        self.collect_buffer_ids.append(buffer_id)
        self.collect_expected = expected
        return None


async def test_build_workflow_rejects_unknown_layer_before_dispatch(
    workflow: AdvancedBuildWorkflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_load_chunks(
        source_id: str, version_hash: str | None = None
    ) -> list[RetrievalChunk]:
        return [
            RetrievalChunk(
                chunk_id="c-1",
                chunk_type="text",
                document_id="doc-1",
                source_id="book-1",
                section_path="Ch1",
                section_title="Chapter 1",
                chapter_path="Ch1",
                content="body",
                embedding_text="body",
                token_count=2,
            )
        ]

    monkeypatch.setattr(workflow, "_load_chunks", _fake_load_chunks)

    with pytest.raises(ValueError, match="unknown build layer"):
        workflow.run(source_id="book-1", layers=["summry"])


async def test_build_workflow_rejects_empty_layers_before_loading_chunks(
    workflow: AdvancedBuildWorkflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _unexpected_load_chunks(
        source_id: str, version_hash: str | None = None
    ) -> list[RetrievalChunk]:
        raise AssertionError("_load_chunks should not be called for empty layers")

    monkeypatch.setattr(workflow, "_load_chunks", _unexpected_load_chunks)

    with pytest.raises(ValueError, match="at least one build layer"):
        workflow.run(source_id="book-1", layers=[])


async def test_build_workflow_rejects_duplicate_layers_before_loading_chunks(
    workflow: AdvancedBuildWorkflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _unexpected_load_chunks(
        source_id: str, version_hash: str | None = None
    ) -> list[RetrievalChunk]:
        raise AssertionError("_load_chunks should not be called for duplicate layers")

    monkeypatch.setattr(workflow, "_load_chunks", _unexpected_load_chunks)

    with pytest.raises(ValueError, match="duplicate build layer"):
        workflow.run(source_id="book-1", layers=["dsi", "dsi"])


async def test_build_workflow_rejects_string_layers_before_loading_chunks(
    workflow: AdvancedBuildWorkflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _unexpected_load_chunks(
        source_id: str, version_hash: str | None = None
    ) -> list[RetrievalChunk]:
        raise AssertionError("_load_chunks should not be called for string layers")

    monkeypatch.setattr(workflow, "_load_chunks", _unexpected_load_chunks)

    with pytest.raises(ValueError, match="layers must be a sequence of layer names"):
        workflow.run(source_id="book-1", layers="dsi")


def test_build_start_event_rejects_empty_layers() -> None:
    with pytest.raises(ValueError, match="at least one build layer"):
        BuildStartEvent(source_id="book-1", layers=[])


def test_build_start_event_rejects_duplicate_layers() -> None:
    with pytest.raises(ValueError, match="duplicate build layer"):
        BuildStartEvent(source_id="book-1", layers=["dsi", "dsi"])


async def test_dsi_summary_raptor_reaches_stop_event(
    workflow: AdvancedBuildWorkflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        RetrievalChunk(
            chunk_id=f"c-{i}",
            chunk_type="text",
            document_id="doc-1",
            source_id="src-1",
            section_path="Ch1",
            section_title="Chapter 1",
            chapter_path="Ch1",
            content="body",
            embedding_text="body",
            token_count=2,
        )
        for i in range(3)
    ]

    async def _fake_load_chunks(
        source_id: str, version_hash: str | None = None
    ) -> list[RetrievalChunk]:
        return chunks

    def _fake_book_meta(chunks: list[RetrievalChunk]) -> tuple[str, str, list[str]]:
        return ("Title", "Author", ["cat-1"])

    async def _fake_build_dsi(st: BuildState) -> dict[str, Any]:
        return {"book_id": st.source_id, "n_records": 1}

    async def _fake_build_summary(st: BuildState) -> dict[str, Any]:
        return {"book_id": st.source_id, "n_nodes": 5}

    async def _fake_build_raptor(*, source_id: str, chunks: list[RetrievalChunk]) -> dict[str, Any]:
        return {"book_id": source_id, "n_nodes": len(chunks)}

    monkeypatch.setattr(workflow, "_load_chunks", _fake_load_chunks)
    monkeypatch.setattr(workflow, "_book_meta", _fake_book_meta)
    monkeypatch.setattr(workflow, "_build_dsi", _fake_build_dsi)
    monkeypatch.setattr(workflow, "_build_summary", _fake_build_summary)
    monkeypatch.setattr(workflow, "_build_raptor", _fake_build_raptor)

    result = await workflow.run(source_id="src-1", layers=["dsi", "summary", "raptor"])

    assert result == {
        "dsi": {"book_id": "src-1", "n_records": 1},
        "summary": {"book_id": "src-1", "n_nodes": 5},
        "raptor": {"book_id": "src-1", "n_nodes": 3},
    }


async def test_raptor_build_passes_explicit_source_and_chunks(
    workflow: AdvancedBuildWorkflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}
    chunks = [_chunk("c-1"), _chunk("c-2"), _chunk("c-3")]

    async def _fake_build_raptor(*, source_id: str, chunks: list[RetrievalChunk]) -> dict[str, Any]:
        seen["source_id"] = source_id
        seen["chunks"] = chunks
        return {"book_id": source_id, "n_nodes": len(chunks)}

    monkeypatch.setattr(workflow, "_build_raptor", _fake_build_raptor)

    result = await workflow.raptor_build(
        cast(Any, None),
        RaptorBuildEvent(source_id="src-raptor", chunks=chunks),
    )

    assert result.summary == {"book_id": "src-raptor", "n_nodes": 3}
    assert seen == {"source_id": "src-raptor", "chunks": chunks}


async def test_collect_uses_source_scoped_event_buffer_id(
    workflow: AdvancedBuildWorkflow,
) -> None:
    state = BuildState(
        source_id="src-1",
        version_hash="vh-1",
        layers=["dsi", "graph"],
    )
    ctx = _CollectCtx(state)

    result = await workflow.collect(
        cast(Any, ctx),
        LayerBuiltEvent(layer="dsi", summary={"book_id": "src-1"}),
    )

    assert result is None
    assert ctx.collect_expected == [LayerBuiltEvent, GraphOrientedEvent]
    assert ctx.collect_buffer_ids == ["advanced_build.collect:src-1:vh-1:dsi,graph"]


async def test_collect_ignores_per_attempt_handler_scope(
    workflow: AdvancedBuildWorkflow,
) -> None:
    state = BuildState(
        source_id="src-1",
        version_hash="vh-1",
        layers=["dsi", "graph"],
    )
    ctx = _CollectCtx(state)
    cast(Any, ctx).vo_rag_build_collect_buffer_scope = "attempt-run-id"

    result = await workflow.collect(
        cast(Any, ctx),
        LayerBuiltEvent(layer="dsi", summary={"book_id": "src-1"}),
    )

    assert result is None
    assert ctx.collect_buffer_ids == ["advanced_build.collect:src-1:vh-1:dsi,graph"]


async def test_graph_only_reaches_collect_through_orientation_gate(
    workflow: AdvancedBuildWorkflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [_chunk("c-1")]
    graph_nodes, graph_edges = _graph_payload()
    extract_graph = AsyncMock(return_value=(graph_nodes, graph_edges, _graph_stats("src-1")))
    persist_graph = AsyncMock(return_value=(1, 1))

    async def _fake_load_chunks(
        source_id: str, version_hash: str | None = None
    ) -> list[RetrievalChunk]:
        assert source_id == "src-1"
        assert version_hash is None
        return chunks

    monkeypatch.setattr(workflow, "_load_chunks", _fake_load_chunks)
    monkeypatch.setattr(workflow, "_book_meta", lambda _: ("Title", "Author", ["cat-1"]))
    monkeypatch.setattr(workflow, "_extract_graph", extract_graph)
    monkeypatch.setattr(workflow, "_persist_graph", persist_graph)

    result = await workflow.run(source_id="src-1", layers=["graph"])

    assert result == {
        "graph": {
            "book_id": "src-1",
            "n_low_confidence_edges": 0,
            "n_nodes": 1,
            "n_edges": 1,
            "direction_coverage": 1.0,
            "n_reoriented": 0,
            "reorientation_status": "not_needed",
        }
    }
    extract_graph.assert_awaited_once()
    persist_graph.assert_awaited_once_with(graph_nodes, graph_edges)


async def test_graph_and_dsi_reach_collect_with_subclass_fan_in(
    workflow: AdvancedBuildWorkflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [_chunk("c-1")]
    graph_nodes, graph_edges = _graph_payload()

    async def _fake_load_chunks(
        source_id: str, version_hash: str | None = None
    ) -> list[RetrievalChunk]:
        assert source_id == "src-1"
        assert version_hash == "vh-1"
        return chunks

    async def _fake_build_dsi(st: BuildState) -> dict[str, Any]:
        assert st.version_hash == "vh-1"
        return {"book_id": st.source_id, "n_records": 1}

    monkeypatch.setattr(workflow, "_load_chunks", _fake_load_chunks)
    monkeypatch.setattr(workflow, "_book_meta", lambda _: ("Title", "Author", ["cat-1"]))
    monkeypatch.setattr(
        workflow,
        "_extract_graph",
        AsyncMock(return_value=(graph_nodes, graph_edges, _graph_stats("src-1"))),
    )
    monkeypatch.setattr(workflow, "_build_dsi", _fake_build_dsi)
    monkeypatch.setattr(workflow, "_persist_graph", AsyncMock(return_value=(1, 1)))

    result = await workflow.run(source_id="src-1", version_hash="vh-1", layers=["graph", "dsi"])

    assert result == {
        "graph": {
            "book_id": "src-1",
            "n_low_confidence_edges": 0,
            "n_nodes": 1,
            "n_edges": 1,
            "direction_coverage": 1.0,
            "n_reoriented": 0,
            "reorientation_status": "not_needed",
        },
        "dsi": {"book_id": "src-1", "n_records": 1},
    }


async def test_load_corpus_raises_when_no_chunks(
    workflow: AdvancedBuildWorkflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_load_chunks(
        source_id: str, version_hash: str | None = None
    ) -> list[RetrievalChunk]:
        return []

    monkeypatch.setattr(workflow, "_load_chunks", _fake_load_chunks)

    with pytest.raises(WorkflowRuntimeError, match="no chunks for source_id=src-1"):
        await workflow.run(source_id="src-1", layers=["dsi"])


async def test_reader_first_build_captures_raw_pointer_before_loading_chunks(
    workflow: AdvancedBuildWorkflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    store: Any = workflow.store
    store.source_visibility_enabled = True
    store.source_publication_enabled = False

    async def _active_revision(collection: str, source_id: str) -> int:
        calls.append("pointer")
        assert collection == "vo_rag_books_raw_v1"
        assert source_id == "src-1"
        return 17

    store.active_source_revision = _active_revision

    async def _load_chunks(source_id: str, version_hash: str | None = None) -> list[RetrievalChunk]:
        calls.append("chunks")
        return [_chunk("c-1", source_id=source_id)]

    async def _build_dsi(st: BuildState) -> dict[str, Any]:
        assert st.dsi_dependency_revision == 17
        return {"book_id": st.source_id}

    monkeypatch.setattr(workflow, "_load_chunks", _load_chunks)
    monkeypatch.setattr(workflow, "_book_meta", lambda _chunks: ("Title", "Author", []))
    monkeypatch.setattr(workflow, "_build_dsi", _build_dsi)

    result = await workflow.run(source_id="src-1", layers=["dsi"])

    assert calls == ["pointer", "chunks"]
    assert result == {"dsi": {"book_id": "src-1"}}


async def test_reader_first_build_skips_unscoped_shared_graph_before_model(
    workflow: AdvancedBuildWorkflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store: Any = workflow.store
    store.source_visibility_enabled = True
    store.source_publication_enabled = False
    store.active_source_revision = AsyncMock(return_value=None)
    monkeypatch.setattr(
        workflow,
        "_load_chunks",
        AsyncMock(return_value=[_chunk("c-1")]),
    )
    monkeypatch.setattr(workflow, "_book_meta", lambda _chunks: ("Title", "Author", []))
    extract = AsyncMock(side_effect=AssertionError("unscoped graph model work must be skipped"))
    monkeypatch.setattr(workflow, "_extract_graph", extract)

    result = await workflow.run(source_id="src-1", layers=["graph"])

    assert result["graph"]["skipped"] is True
    assert "complete corpus dependency snapshot" in result["graph"]["reason"]
    extract.assert_not_awaited()


def test_workflow_routing_graph_validates(workflow: AdvancedBuildWorkflow) -> None:
    steps = workflow._get_steps()

    def _step_config(name: str) -> Any:
        method = steps[name]
        return getattr(method, "__step_config", None)

    load_cfg = _step_config("load_corpus")
    assert load_cfg is not None
    assert BuildStartEvent in load_cfg.accepted_events
    assert BuildLayerEvent in load_cfg.return_types

    build_cfg = _step_config("build_layer")
    assert build_cfg is not None
    assert BuildLayerEvent in build_cfg.accepted_events
    assert LayerBuiltEvent in build_cfg.return_types
    assert GraphBuiltEvent in build_cfg.return_types
    assert RaptorPreflightEvent in build_cfg.return_types
    assert build_cfg.num_workers == 4

    gate_cfg = _step_config("graph_direction_gate")
    assert gate_cfg is not None
    assert GraphBuiltEvent in gate_cfg.accepted_events
    assert GraphOrientedEvent in gate_cfg.return_types

    raptor_preflight_cfg = _step_config("raptor_preflight_gate")
    assert raptor_preflight_cfg is not None
    assert RaptorPreflightEvent in raptor_preflight_cfg.accepted_events
    assert LayerBuiltEvent in raptor_preflight_cfg.return_types
    assert RaptorBuildEvent in raptor_preflight_cfg.return_types

    raptor_build_cfg = _step_config("raptor_build")
    assert raptor_build_cfg is not None
    assert RaptorBuildEvent in raptor_build_cfg.accepted_events
    assert LayerBuiltEvent in raptor_build_cfg.return_types

    collect_cfg = _step_config("collect")
    assert collect_cfg is not None
    assert LayerBuiltEvent in collect_cfg.accepted_events
    assert GraphOrientedEvent in collect_cfg.accepted_events
    assert StopEvent in collect_cfg.return_types

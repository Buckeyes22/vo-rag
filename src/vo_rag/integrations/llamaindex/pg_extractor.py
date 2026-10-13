"""vo-rag graph extraction wrapper for LlamaIndex PropertyGraphIndex."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Final, cast

from llama_index.core.schema import BaseNode, TextNode, TransformComponent
from pydantic import Field, PrivateAttr

from vo_rag.core.graph.direction import orient_edges, orient_low_confidence_via_llm
from vo_rag.core.graph.extraction import (
    GraphExtractionConfig,
    GraphExtractionResult,
    extract_graph_from_book,
)
from vo_rag.core.llm.response import LLMResponse
from vo_rag.integrations.llamaindex._async import run_coro_sync
from vo_rag.integrations.llamaindex.convert import text_node_to_chunk

DEFAULT_DIRECTION_COVERAGE_FLOOR: Final[float] = 0.85
FallbackLLMCall = Callable[[str, str, int, float], Awaitable[Any]]


class VoRagPathExtractor(TransformComponent):
    """Run guided vo-rag graph extraction plus deterministic edge orientation.

    The injected ``llm_call`` is expected to call the graph model with
    ``graph_response_json_schema()`` using the configured structured-output mode
    (default ``response_format`` JSON schema, or ``guided_json`` when
    ``VORAG_GUIDED_DECODING_MODE=guided_json``). This adapter delegates
    extraction to ``extract_graph_from_book`` and then always runs
    ``orient_edges`` before exposing graph metadata to LlamaIndex's
    property-graph pipeline.
    """

    book_id: str
    title: str
    author: str
    config: GraphExtractionConfig
    coverage_floor: float = Field(
        default=DEFAULT_DIRECTION_COVERAGE_FLOOR,
        ge=0.0,
        le=1.0,
    )

    _llm_call: Callable[[str, str, int, float], Awaitable[LLMResponse]] = PrivateAttr()
    _fallback_llm_call: FallbackLLMCall | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        book_id: str,
        title: str,
        author: str,
        config: GraphExtractionConfig,
        llm_call: Callable[[str, str, int, float], Awaitable[LLMResponse]],
        coverage_floor: float = DEFAULT_DIRECTION_COVERAGE_FLOOR,
        fallback_llm_call: FallbackLLMCall | None = None,
    ) -> None:
        model_data: dict[str, Any] = {
            "book_id": book_id,
            "title": title,
            "author": author,
            "config": config,
            "coverage_floor": coverage_floor,
        }
        super().__init__(**model_data)
        self._llm_call = llm_call
        self._fallback_llm_call = fallback_llm_call

    @classmethod
    def class_name(cls) -> str:
        return "VoRagPathExtractor"

    async def acall(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
        llama_nodes = list(nodes)
        chunks = [text_node_to_chunk(cast(TextNode, node)) for node in llama_nodes]
        result: GraphExtractionResult = await extract_graph_from_book(
            book_id=self.book_id,
            title=self.title,
            author=self.author,
            chunks=chunks,
            config=self.config,
            llm_call=self._llm_call,
        )
        oriented, cov = orient_edges(result.nodes, result.edges)

        graph_edges_for_metadata = oriented
        orientation_fallback_ran = False
        if cov.coverage < self.coverage_floor and self._fallback_llm_call is not None:
            graph_edges_for_metadata = await orient_low_confidence_via_llm(
                result.nodes,
                oriented,
                cov.low_confidence_edge_ids,
                self._fallback_llm_call,
            )
            orientation_fallback_ran = True

        graph_nodes = [node.model_dump(mode="json") for node in result.nodes]
        graph_edges = [edge.model_dump(mode="json") for edge in graph_edges_for_metadata]
        graph_payload_node_id = llama_nodes[0].node_id if llama_nodes else None
        for index, node in enumerate(llama_nodes):
            node.metadata["vo_graph_payload_role"] = "primary" if index == 0 else "reference"
            node.metadata["vo_graph_payload_node_id"] = graph_payload_node_id
            node.metadata["vo_graph_node_count"] = len(graph_nodes)
            node.metadata["vo_graph_edge_count"] = len(graph_edges)
            node.metadata["vo_graph_coverage"] = cov.coverage
            node.metadata["vo_graph_low_confidence_edge_ids"] = cov.low_confidence_edge_ids
            node.metadata["vo_graph_orientation_fallback_ran"] = orientation_fallback_ran
            if index == 0:
                node.metadata["vo_graph_nodes"] = graph_nodes
                node.metadata["vo_graph_edges"] = graph_edges
        return llama_nodes

    def __call__(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
        return run_coro_sync(
            lambda: self.acall(nodes, **kwargs),
            component="VoRagPathExtractor.__call__",
        )


__all__ = ["DEFAULT_DIRECTION_COVERAGE_FLOOR", "VoRagPathExtractor"]

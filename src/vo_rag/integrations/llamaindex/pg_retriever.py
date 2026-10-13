"""vo-rag property-graph sub-retriever for LlamaIndex PropertyGraphIndex.

Wraps ``graph_query_to_evidence_chunks`` (seed-resolve -> BFS -> evidence-union)
as a ``CustomPGRetriever.custom_retrieve`` implementation.
"""

from __future__ import annotations

from typing import Any

from llama_index.core.indices.property_graph import CustomPGRetriever
from llama_index.core.schema import NodeWithScore, TextNode

from vo_rag.core.graph.retrieval import graph_query_to_evidence_chunks
from vo_rag.integrations.llamaindex._async import run_coro_sync


class VoRagPGRetriever(CustomPGRetriever):
    """custom_retrieve -> graph_query_to_evidence_chunks (seed -> BFS -> evidence union)."""

    def init(self, **kwargs: Any) -> None:
        # Expected kwargs: store (required), top_k (default 10).
        store = kwargs.get("store")
        if store is None:
            raise ValueError("VoRagPGRetriever requires store=GraphQdrantStore")
        top_k = kwargs.get("top_k", 10)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("VoRagPGRetriever top_k must be a positive integer")
        self._store = store
        self._top_k = top_k

    def custom_retrieve(self, query_str: str) -> list[NodeWithScore]:
        chunks = run_coro_sync(
            lambda: graph_query_to_evidence_chunks(
                query_text=query_str,
                # The query string IS the entity label seeds resolve from:
                # graph_query_to_evidence_chunks resolves seeds ONLY from
                # entity_labels (query_text is logging-only), so entity_labels=None
                # -> labels=[] -> no seeds -> always []. For the graph-lookup use
                # case (VOR-222) the caller queries by entity name. (doc-10 §3's
                # entity_labels=None sketch was a no-op; corrected here.)
                entity_labels=[query_str],
                qdrant=self._store,
                top_k=self._top_k,
            ),
            component="VoRagPGRetriever.custom_retrieve",
        )
        return [
            NodeWithScore(
                node=TextNode(
                    id_=c.chunk_id,
                    text=c.content,
                    metadata={
                        "source_id": c.source_id,
                        "section_path": c.section_path,
                    },
                ),
                score=None,
            )
            for c in chunks
        ]


__all__ = ["VoRagPGRetriever"]

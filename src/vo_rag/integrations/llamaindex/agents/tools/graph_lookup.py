"""graph_lookup tool — trusted property-graph QueryEngineTool for the research agent.

SECURITY: This tool is the TRUSTED arm of the agent. It reads vetted internal
property-graph data and therefore does NOT need the protected wrapper. The
factory must NEVER allow it to be aimed at an untrusted scratch collection —
the graph store and store are provided by the caller and must reference
internal graph collections only.
"""

from __future__ import annotations

from typing import Any

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.tools import QueryEngineTool

from vo_rag.core.graph.persistence import GraphQdrantStore
from vo_rag.integrations.llamaindex.graph_store import VoRagPropertyGraphStore
from vo_rag.integrations.llamaindex.llm import VoRagLLM
from vo_rag.integrations.llamaindex.pg_retriever import VoRagPGRetriever

_GRAPH_LOOKUP_NAME = "graph_lookup"

_GRAPH_LOOKUP_DESCRIPTION = (
    "Given one or more entity names, return connected entities and "
    "the relationships between them, plus the evidence chunks that "
    "support each edge. Use to expand a hop in a multi-step question."
)


def build_graph_lookup_tool(
    *,
    graph_store: VoRagPropertyGraphStore,
    store: GraphQdrantStore,
    top_k: int = 10,
    llm: VoRagLLM,
    response_synthesizer: BaseSynthesizer,
    **kwargs: Any,
) -> QueryEngineTool:
    """Build a ``graph_lookup`` ``QueryEngineTool`` over the property-graph retriever.

    The returned tool wraps ``VoRagPGRetriever`` in a ``RetrieverQueryEngine``
    so the agent can dispatch entity/relationship traversal as a standard tool
    call.  ``return_direct`` is intentionally **not** set — the agent's value is
    synthesis across multi-hop tool calls.
    """
    if response_synthesizer is None:
        raise ValueError("graph_lookup requires a vo-rag response_synthesizer")
    if llm is None:
        raise ValueError("graph_lookup requires a vo-rag llm")
    if not isinstance(llm, VoRagLLM):
        raise TypeError(f"graph_lookup llm must be VoRagLLM, got {type(llm).__name__}")
    synthesizer_llm = getattr(response_synthesizer, "_llm", None)
    if synthesizer_llm is not llm:
        raise ValueError("graph_lookup response_synthesizer llm must be the supplied llm")

    retriever = VoRagPGRetriever(
        graph_store=graph_store,
        store=store,
        top_k=top_k,
        **kwargs,
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    return QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name=_GRAPH_LOOKUP_NAME,
        description=_GRAPH_LOOKUP_DESCRIPTION,
    )


__all__ = [
    "build_graph_lookup_tool",
]

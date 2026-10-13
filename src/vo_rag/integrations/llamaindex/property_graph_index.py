"""Factory to assemble a LlamaIndex PropertyGraphIndex over a live vo-rag graph.

This module wires :class:`VoRagPropertyGraphStore`, :class:`VoRagPathExtractor`,
and :class:`VoRagPGRetriever` into a read-only ``PropertyGraphIndex``
via ``from_existing`` (no re-extraction, no re-ingest).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vo_rag.core.graph.persistence import GraphQdrantStore
from vo_rag.integrations.llamaindex import require_llamaindex

if TYPE_CHECKING:
    from llama_index.core import PropertyGraphIndex
    from llama_index.core.schema import TransformComponent

    from vo_rag.integrations.llamaindex.pg_retriever import VoRagPGRetriever


def build_vo_rag_property_graph_index(
    *,
    store: GraphQdrantStore,
    kg_extractors: list[TransformComponent] | None = None,
    embed_kg_nodes: bool = False,
    **kwargs: Any,
) -> PropertyGraphIndex:
    """Assemble a ``PropertyGraphIndex`` reattached over an existing vo-rag graph.

    Parameters
    ----------
    store:
        A ``GraphQdrantStore`` whose collections are already populated.
    kg_extractors:
        Extractors to register on the index (required to avoid LlamaIndex's
        default OpenAI-dependent extractors).  They are **not** run during
        ``from_existing`` because no documents are inserted.
    embed_kg_nodes:
        Passed through to ``PropertyGraphIndex``.  Defaults to ``False`` so
        the factory does not require a default embedding model.
    **kwargs:
        Additional keyword arguments forwarded to
        ``PropertyGraphIndex.from_existing``.

    Returns
    -------
    PropertyGraphIndex
        A read-only view over the live graph.

    Notes
    -----
    ``PropertyGraphIndex.as_retriever()`` returns LlamaIndex's generic
    property-graph retriever. For vo-rag's entity-label graph lookup and
    evidence-union behavior, use ``build_vo_rag_pg_retriever()`` instead.
    """
    require_llamaindex()
    from llama_index.core import PropertyGraphIndex

    from vo_rag.integrations.llamaindex.graph_store import VoRagPropertyGraphStore

    pg_store = VoRagPropertyGraphStore(store=store)
    return PropertyGraphIndex.from_existing(
        property_graph_store=pg_store,
        kg_extractors=kg_extractors,
        embed_kg_nodes=embed_kg_nodes,
        **kwargs,
    )


def build_vo_rag_pg_retriever(
    *,
    store: GraphQdrantStore,
    top_k: int = 10,
    **kwargs: Any,
) -> VoRagPGRetriever:
    """Return a ``VoRagPGRetriever`` wired to the same graph store.

    This is the public graph-search path for vo-rag label lookup. It resolves
    the query string through vo-rag's graph seed/BFS/evidence-union flow rather
    than LlamaIndex's optional vector graph search.

    Parameters
    ----------
    store:
        The ``GraphQdrantStore`` backing the graph.
    top_k:
        Number of evidence chunks to retrieve.
    **kwargs:
        Forwarded to ``VoRagPGRetriever``.

    Returns
    -------
    VoRagPGRetriever
        The composed sub-retriever.
    """
    require_llamaindex()
    from vo_rag.integrations.llamaindex.graph_store import VoRagPropertyGraphStore
    from vo_rag.integrations.llamaindex.pg_retriever import VoRagPGRetriever

    pg_store = VoRagPropertyGraphStore(store=store)
    return VoRagPGRetriever(graph_store=pg_store, store=store, top_k=top_k, **kwargs)


__all__ = [
    "build_vo_rag_property_graph_index",
    "build_vo_rag_pg_retriever",
]

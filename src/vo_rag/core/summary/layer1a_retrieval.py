"""Layer-1A retrieval arm — retrieve summary nodes, then selectively expand.

``search_summary_nodes`` does not accept a ``QdrantStore`` directly. It takes a
``HybridRetriever`` (or any object that satisfies the same
``hybrid_search(query, collection, top_k, filters, ...) -> list[SearchResult]``
public method). The summary layer NEVER reaches past the retriever into the
store. This keeps the retriever the only path to Qdrant, so any future
license/visual-mode/annotation logic added in the retriever automatically
benefits the +Layer1A eval arm.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, cast, runtime_checkable

import structlog

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.retrieval.bm25_tokenizer import tokenize, tokens_to_bm25_sparse
from vo_rag.core.retrieval.search import HybridRetriever, SearchResult
from vo_rag.core.storage.passage_contract import model_payload
from vo_rag.core.storage.publication_read_view import publication_query
from vo_rag.core.storage.qdrant import QdrantStore
from vo_rag.core.summary.persistence import SUMMARY_TREE_COLLECTION
from vo_rag.core.summary.schemas import NodeLevel, SummaryNode
from vo_rag.core.summary.selective_expansion import default_m_for, expand_summary_node

log = structlog.get_logger(__name__)

SearchFn = Callable[[str, int], Awaitable[list[str]]]


@runtime_checkable
class _RetrieverLike(Protocol):
    """Structural type for the retriever seam ``search_summary_nodes`` depends on.

    Mirrors the landed ``HybridRetriever.hybrid_search`` signature in
    ``vo_rag.core.retrieval.search``. Anything with a method of this shape —
    the real ``HybridRetriever`` or a test fake — is accepted.
    """

    async def hybrid_search(
        self,
        *,
        query: str,
        collection: str,
        top_k: int = ...,
        filters: dict[str, Any] | None = ...,
        prefetch_depth: int = ...,
        mode: Any = ...,
    ) -> list[SearchResult]: ...


def _node_from_payload(payload: dict[str, Any]) -> SummaryNode | None:
    """Reconstruct a SummaryNode from a stored payload (node.model_dump()).

    Tolerant: a payload that doesn't validate (schema drift, a non-node point)
    is dropped with a warning rather than killing the whole retrieval.
    """
    try:
        return SummaryNode.model_validate(model_payload(payload, SummaryNode.model_fields))
    except Exception as exc:  # noqa: BLE001 — one bad node must not break retrieval
        log.warning("layer1a.node_reconstruct_failed", error=f"{type(exc).__name__}: {exc}")
        return None


async def search_summary_nodes(
    *,
    query_text: str,
    retriever: HybridRetriever | _RetrieverLike,
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
    collection: str = SUMMARY_TREE_COLLECTION,
) -> list[tuple[SummaryNode, float]]:
    """Hybrid-search the summary tree for the top-k most relevant nodes.

    Delegates entirely to the injected ``retriever`` (a ``HybridRetriever`` or
    any duck-typed equivalent). The summary layer does NOT touch the
    ``QdrantStore`` directly — the retriever must remain the only path to the store.

    Returns ``(node, score)`` pairs in descending relevance. ``filters``
    narrows the search (e.g. ``{"book_id": ...}`` or
    ``{"source_collection_version": ...}``).
    """
    results = await retriever.hybrid_search(
        query=query_text,
        collection=collection,
        top_k=top_k,
        filters=filters,
    )
    out: list[tuple[SummaryNode, float]] = []
    for result in results:
        node = _node_from_payload(result.payload or {})
        if node is not None:
            out.append((node, result.score))
    return out


@publication_query
async def layer1a_retrieve(
    *,
    query_text: str,
    retriever: HybridRetriever | _RetrieverLike,
    store: Any | None = None,
    query_dense: list[float] | None = None,
    query_sparse: dict[int, float] | None = None,
    query_bm25: dict[int, float] | None = None,
    query_colbert: list[list[float]] | None = None,
    summary_collection: str = SUMMARY_TREE_COLLECTION,
    raw_collection: str = "vo_rag_books_raw_v1",
    top_k_nodes: int = 10,
    top_k_chunks: int = 10,
    filters: dict[str, Any] | None = None,
    m_by_level: dict[NodeLevel, int] | None = None,
) -> list[RetrievalChunk]:
    """Retrieve summary nodes, expand each to its top-m chunks, merge + rank.

    The ``retriever`` is the public search seam. The
    ``store`` is the raw expansion seam used by ``expand_summary_node`` to
    look up ``node.source_chunk_ids`` in the raw collection. ``store``
    defaults to ``retriever`` for callers that wire a single duck-typed
    stand-in in tests; production callers typically pass the real
    ``QdrantStore`` (the same one the retriever was built with) so the raw
    expansion hits the same physical database.

    The four query vectors (``query_dense``, ``query_sparse``, ``query_bm25``,
    ``query_colbert``) are forwarded to ``expand_summary_node`` for the raw
    hybrid search; they are only required when at least one summary node
    is returned AND ``top_k_chunks`` > 0. The default empty values keep the
    test surface (where the fake ignores vectors) friction-free; production
    callers should pass real embeddings.

    Merge policy: each expanded chunk inherits its node's relevance
    discounted by its within-node rank (``node_score / (rank + 1)``). A
    chunk reached via several nodes keeps its single highest score (dedup
    by chunk_id). The final list is the global top-``top_k_chunks`` by that
    combined score.

    The invariant (expand only within a node's own ``source_chunk_ids``) is
    enforced inside ``expand_summary_node``, not here.
    """
    expansion_store: Any = store if store is not None else retriever
    _q_dense = query_dense if query_dense is not None else []
    _q_sparse = query_sparse if query_sparse is not None else {}
    _q_bm25 = query_bm25 if query_bm25 is not None else {}
    _q_colbert = query_colbert if query_colbert is not None else []
    _expansion_qdrant = cast("QdrantStore", expansion_store)
    nodes = await search_summary_nodes(
        query_text=query_text,
        retriever=retriever,
        top_k=top_k_nodes,
        filters=filters,
        collection=summary_collection,
    )

    best: dict[str, tuple[RetrievalChunk, float]] = {}
    for node, node_score in nodes:
        m = (m_by_level or {}).get(node.level) or default_m_for(node.level)
        chunks = await expand_summary_node(
            node=node,
            query_dense=_q_dense,
            query_sparse=_q_sparse,
            query_bm25=_q_bm25,
            query_colbert=_q_colbert,
            m=m,
            qdrant=_expansion_qdrant,
            raw_collection=raw_collection,
        )
        for rank, chunk in enumerate(chunks):
            score = node_score / (rank + 1)
            current = best.get(chunk.chunk_id)
            if current is None or score > current[1]:
                best[chunk.chunk_id] = (chunk, score)

    ranked = sorted(best.values(), key=lambda cs: cs[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_k_chunks]]


def build_layer1a_search_fn(
    *,
    embed: Any,
    retriever: HybridRetriever | _RetrieverLike,
    store: Any | None = None,
    summary_collection: str = SUMMARY_TREE_COLLECTION,
    raw_collection: str = "vo_rag_books_raw_v1",
    top_k_nodes: int = 10,
    filters: dict[str, Any] | None = None,
    m_by_level: dict[NodeLevel, int] | None = None,
) -> SearchFn:
    """Build a harness-compatible ``search_fn(query_text, top_k) -> [chunk_id]``
    for the +Layer1A arm.

    Mirrors ``HybridRetriever.hybrid_search``'s query embedding (dense + sparse
    + ColBERT from the embedding server, BM25 from the local tokenizer), then
    runs ``layer1a_retrieve``.
    """

    async def _search_fn(query_text: str, top_k: int) -> list[str]:
        emb = await embed.embed(
            [query_text], return_dense=True, return_sparse=True, return_colbert=True
        )
        q_dense = emb["dense"][0]
        q_sparse = dict(zip(emb["sparse"][0]["indices"], emb["sparse"][0]["values"], strict=True))
        q_colbert = emb["colbert"][0]
        q_bm25 = tokens_to_bm25_sparse(tokenize(query_text))

        chunks = await layer1a_retrieve(
            query_text=query_text,
            query_dense=q_dense,
            query_sparse=q_sparse,
            query_bm25=q_bm25,
            query_colbert=q_colbert,
            retriever=retriever,
            store=store,
            summary_collection=summary_collection,
            raw_collection=raw_collection,
            top_k_nodes=top_k_nodes,
            top_k_chunks=top_k,
            filters=filters,
            m_by_level=m_by_level,
        )
        return [c.chunk_id for c in chunks]

    return _search_fn


__all__ = [
    "SearchFn",
    "build_layer1a_search_fn",
    "layer1a_retrieve",
    "search_summary_nodes",
]

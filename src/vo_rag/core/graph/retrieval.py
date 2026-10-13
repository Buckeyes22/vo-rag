"""Graph-layer retrieval primitives for Layer 3 Evidence-linked entity graph.

Provides:
  - GraphRetrievalConfig    — BFS traversal parameters
  - GraphTraversalResult    — traversal output (nodes, edges, evidence chunk IDs)
  - resolve_seed_nodes      — map entity label strings → GraphNode seeds
  - expand_neighborhood     — BFS expansion up to N hops
  - graph_query_to_evidence_chunks — end-to-end: labels → traversal → raw chunks

Evidence chunks are collected ONLY from the union of:
  - edge.evidence_chunk_ids on every traversed GraphEdge
  - node.chunk_ids on every visited GraphNode

No evidence is invented beyond that union — provenance always traces to actual
graph edges and node records.

Collections are GATED: only populated after the graph smoke test passes.
"""

from __future__ import annotations

import asyncio
import inspect
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.graph.persistence import GraphQdrantStore, get_neighbors, get_node
from vo_rag.core.graph.schemas import GraphEdge, GraphEdgeType, GraphNode
from vo_rag.core.models.common import VoBaseModel
from vo_rag.core.storage.passage_contract import model_payload
from vo_rag.core.storage.publication_read_view import publication_query

log = structlog.get_logger(__name__)

KEYWORD_SCROLL_FALLBACK = "keyword_scroll_fallback"
LEXICAL_RANKING_FALLBACK = "token_overlap_fallback"
SUPPLIED_RANKER = "supplied_ranker"

RankChunks = Callable[[str, list[str], int], Awaitable[list[str]] | list[str]]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class GraphRetrievalConfig(VoBaseModel):
    """Parameters governing BFS expansion and edge filtering."""

    max_hops: int = 2
    max_neighbors_per_hop: int = 10
    edge_type_whitelist: list[GraphEdgeType] | None = None  # None = all
    min_edge_weight: float = 0.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class GraphTraversalResult(VoBaseModel):
    """Output of expand_neighborhood.

    evidence_chunk_ids: sorted, deduplicated union of:
      - evidence_chunk_ids from every traversed GraphEdge
      - chunk_ids from every visited GraphNode
    """

    seed_node_ids: list[str]
    nodes_visited: list[GraphNode]
    edges_traversed: list[GraphEdge]
    evidence_chunk_ids: list[str]
    hop_depth_per_node: dict[str, int]


# ---------------------------------------------------------------------------
# Seed resolution
# ---------------------------------------------------------------------------


async def resolve_seed_nodes(
    *,
    entity_labels: list[str],
    qdrant: GraphQdrantStore,
    nodes_collection: str = "vo_rag_books_graph_nodes_v1",
    limit: int = 10,
) -> list[GraphNode]:
    """Resolve entity label strings to GraphNode seeds.

    For each label in entity_labels, scroll the nodes collection and match
    on label or aliases (case-insensitive substring).  Returns deduplicated
    nodes by node_id across all labels.

    Scroll is used because the graph nodes collection may not expose a full
    text-search index at pilot scale.  Beyond that, replace the scroll loop
    with a BM25/sparse search once the index is confirmed to be populated.
    """
    seen_ids: set[str] = set()
    matched: list[GraphNode] = []

    for label in entity_labels:
        label_lower = label.strip().lower()
        if not label_lower:
            continue

        next_offset: Any = None
        candidates_found = 0

        while candidates_found < limit:
            results, next_offset = await qdrant._client.scroll(
                collection_name=nodes_collection,
                limit=200,
                with_payload=True,
                offset=next_offset,
            )

            for point in results:
                if point.payload is None:
                    continue
                node = GraphNode.model_validate(
                    model_payload(point.payload, GraphNode.model_fields)
                )
                if node.node_id in seen_ids:
                    continue

                # Match on label or any alias, case-insensitive substring.
                node_label_lower = node.label.lower()
                alias_lowers = [a.lower() for a in node.aliases]

                if label_lower in node_label_lower or any(label_lower in a for a in alias_lowers):
                    seen_ids.add(node.node_id)
                    matched.append(node)
                    candidates_found += 1
                    if candidates_found >= limit:
                        break

            if next_offset is None:
                break

        log.debug(
            "graph.resolve_seed_nodes",
            label=label,
            candidates_found=candidates_found,
            resolution_mode=KEYWORD_SCROLL_FALLBACK,
        )

    return matched


# ---------------------------------------------------------------------------
# BFS neighborhood expansion
# ---------------------------------------------------------------------------


async def expand_neighborhood(
    *,
    seed_node_ids: list[str],
    config: GraphRetrievalConfig,
    qdrant: GraphQdrantStore,
    nodes_collection: str = "vo_rag_books_graph_nodes_v1",
    edges_collection: str = "vo_rag_books_graph_edges_v1",
) -> GraphTraversalResult:
    """BFS expansion from seed nodes up to config.max_hops hops.

    At each hop, neighbors are retrieved via get_neighbors() filtered by
    edge_type_whitelist and min_edge_weight, then capped at
    max_neighbors_per_hop.

    Each (neighbor_node, edge) pair is collected; nodes already visited are
    not re-expanded.  Evidence chunk IDs are accumulated from every traversed
    edge and every visited node.

    Returns a GraphTraversalResult with the full traversal state.
    """
    # Resolve seed GraphNode objects first (seeds count as hop depth 0).
    seed_fetch_tasks = [
        get_node(nid, store=qdrant, collection=nodes_collection) for nid in seed_node_ids
    ]
    seed_results = await asyncio.gather(*seed_fetch_tasks)

    hop_depth: dict[str, int] = {}
    nodes_by_id: dict[str, GraphNode] = {}
    edges_traversed: list[GraphEdge] = []
    evidence_chunk_ids: set[str] = set()

    for nid, node in zip(seed_node_ids, seed_results, strict=True):
        if node is None:
            log.warning("graph.expand_neighborhood.seed_not_found", node_id=nid)
            continue
        hop_depth[nid] = 0
        nodes_by_id[nid] = node
        evidence_chunk_ids.update(node.chunk_ids)

    # BFS queue: entries are (node_id, current_hop_depth)
    queue: deque[tuple[str, int]] = deque(
        (nid, 0)
        for nid in hop_depth  # only seeds that resolved
    )

    while queue:
        current_id, current_depth = queue.popleft()

        if current_depth >= config.max_hops:
            continue

        next_depth = current_depth + 1

        # Fetch all neighbors (in + out) for this node.
        try:
            neighbor_pairs = await get_neighbors(
                current_id,
                store=qdrant,
                direction="both",
                nodes_collection=nodes_collection,
                edges_collection=edges_collection,
            )
        except Exception as exc:
            log.warning(
                "graph.expand_neighborhood.get_neighbors_error",
                node_id=current_id,
                error=str(exc),
            )
            continue

        # Filter by edge_type_whitelist and min_edge_weight.
        if config.edge_type_whitelist is not None:
            whitelist_set = set(config.edge_type_whitelist)
            neighbor_pairs = [(n, e) for n, e in neighbor_pairs if e.edge_type in whitelist_set]

        if config.min_edge_weight > 0.0:
            neighbor_pairs = [
                (n, e) for n, e in neighbor_pairs if e.weight >= config.min_edge_weight
            ]

        # Cap at max_neighbors_per_hop.
        neighbor_pairs = neighbor_pairs[: config.max_neighbors_per_hop]

        for neighbor_node, edge in neighbor_pairs:
            # Collect edge evidence regardless of whether we re-visit the node.
            edges_traversed.append(edge)
            evidence_chunk_ids.update(edge.evidence_chunk_ids)

            if neighbor_node.node_id in hop_depth:
                # Already visited — still collect evidence but don't re-expand.
                evidence_chunk_ids.update(neighbor_node.chunk_ids)
                continue

            # New node.
            hop_depth[neighbor_node.node_id] = next_depth
            nodes_by_id[neighbor_node.node_id] = neighbor_node
            evidence_chunk_ids.update(neighbor_node.chunk_ids)
            queue.append((neighbor_node.node_id, next_depth))

    log.info(
        "graph.expand_neighborhood.done",
        seed_count=len(seed_node_ids),
        nodes_visited=len(nodes_by_id),
        edges_traversed=len(edges_traversed),
        evidence_chunks=len(evidence_chunk_ids),
    )

    return GraphTraversalResult(
        seed_node_ids=seed_node_ids,
        nodes_visited=list(nodes_by_id.values()),
        edges_traversed=edges_traversed,
        evidence_chunk_ids=sorted(evidence_chunk_ids),
        hop_depth_per_node=hop_depth,
    )


# ---------------------------------------------------------------------------
# End-to-end: entity labels → raw evidence chunks
# ---------------------------------------------------------------------------


@publication_query
async def graph_query_to_evidence_chunks(
    *,
    query_text: str,
    entity_labels: list[str] | None = None,
    seed_labels: list[str] | None = None,
    config: GraphRetrievalConfig | None = None,
    qdrant: GraphQdrantStore | None = None,
    store: GraphQdrantStore | None = None,
    rank_chunks: RankChunks | None = None,
    top_k: int = 10,
    raw_collection: str = "vo_rag_books_raw_v1",
    nodes_collection: str = "vo_rag_books_graph_nodes_v1",
    edges_collection: str = "vo_rag_books_graph_edges_v1",
) -> list[RetrievalChunk]:
    """Resolve entity labels → graph traversal → raw evidence chunks.

    Step 1: resolve_seed_nodes — map entity label strings to GraphNode seeds.
    Step 2: expand_neighborhood — BFS up to config.max_hops hops.
    Step 3: scroll raw_collection for points whose chunk_id appears in the
            traversal's evidence_chunk_ids set.

    Returns RetrievalChunk objects built from the raw collection payloads.
    Evidence is strictly limited to the traversal union — no chunks are
    invented beyond what the graph edges and nodes reference (invariant 2).
    """
    resolved_store = _resolve_store(qdrant=qdrant, store=store)
    resolved_config = config or GraphRetrievalConfig()
    labels = seed_labels if seed_labels is not None else entity_labels
    if labels is None:
        labels = []

    # Step 1: resolve seeds.
    seed_nodes = await resolve_seed_nodes(
        entity_labels=labels,
        qdrant=resolved_store,
        nodes_collection=nodes_collection,
    )

    if not seed_nodes:
        log.info(
            "graph.query_to_chunks.no_seeds",
            query_text=query_text,
            entity_labels=labels,
            resolution_mode=KEYWORD_SCROLL_FALLBACK,
        )
        return []

    seed_ids = [n.node_id for n in seed_nodes]

    # Step 2: expand neighborhood.
    traversal = await expand_neighborhood(
        seed_node_ids=seed_ids,
        config=resolved_config,
        qdrant=resolved_store,
        nodes_collection=nodes_collection,
        edges_collection=edges_collection,
    )

    if not traversal.evidence_chunk_ids:
        log.info(
            "graph.query_to_chunks.no_evidence_chunks",
            query_text=query_text,
            seed_count=len(seed_ids),
        )
        return []

    # Step 3: retrieve raw chunks whose chunk_id is in the traversal evidence set.
    # We scroll the raw collection and filter client-side against the evidence set,
    # then score by a simple text-overlap heuristic against query_text so callers
    # get a sensibly ordered list.  A full hybrid re-rank requires embedding the
    # query, which is handled by the caller's planner; here we provide a
    # lightweight relevance ordering that works without an embedding call.
    candidate_chunk_ids = traversal.evidence_chunk_ids
    evidence_set = set(candidate_chunk_ids)
    raw_chunks = await _load_raw_chunks(
        store=resolved_store,
        raw_collection=raw_collection,
        evidence_set=evidence_set,
    )

    if rank_chunks is None:
        ordered_chunks = _order_by_token_overlap(query_text=query_text, chunks=raw_chunks)[:top_k]
        final_ranking_mode = LEXICAL_RANKING_FALLBACK
        log.info(
            "graph.query_to_chunks.final_rank_fallback",
            query_text=query_text,
            final_ranking_mode=final_ranking_mode,
        )
    else:
        ranked_ids = await _ranked_chunk_ids(
            rank_chunks=rank_chunks,
            query_text=query_text,
            candidate_chunk_ids=candidate_chunk_ids,
            top_k=top_k,
        )
        ordered_chunks = _order_chunks_by_ids(raw_chunks, ranked_ids, candidate_chunk_ids)[:top_k]
        final_ranking_mode = SUPPLIED_RANKER

    tagged_chunks = _tag_retrieval_modes(
        ordered_chunks,
        seed_resolution_mode=KEYWORD_SCROLL_FALLBACK,
        final_ranking_mode=final_ranking_mode,
    )

    log.info(
        "graph.query_to_chunks.done",
        query_text=query_text,
        evidence_ids_in_traversal=len(evidence_set),
        chunks_returned=len(tagged_chunks),
        seed_resolution_mode=KEYWORD_SCROLL_FALLBACK,
        final_ranking_mode=final_ranking_mode,
    )

    return tagged_chunks


def _resolve_store(
    *,
    qdrant: GraphQdrantStore | None,
    store: GraphQdrantStore | None,
) -> GraphQdrantStore:
    resolved = store if store is not None else qdrant
    if resolved is None:
        msg = "graph_query_to_evidence_chunks requires either store= or qdrant="
        raise ValueError(msg)
    return resolved


async def _load_raw_chunks(
    *,
    store: GraphQdrantStore,
    raw_collection: str,
    evidence_set: set[str],
) -> list[RetrievalChunk]:
    raw_chunks: list[RetrievalChunk] = []
    next_offset: Any = None
    while True:
        results, next_offset = await store._client.scroll(
            collection_name=raw_collection,
            limit=500,
            with_payload=True,
            offset=next_offset,
        )

        for point in results:
            if point.payload is None:
                continue
            chunk_id = point.payload.get("chunk_id")
            if chunk_id is None or chunk_id not in evidence_set:
                continue
            try:
                # Raw writers intentionally omit duplicate embedding_text for
                # prose, and carry runtime/filter fields outside the native
                # chunk model. Keep native fields and reconstruct its required
                # embedding view without adding storage duplication.
                native = {
                    key: value
                    for key, value in point.payload.items()
                    if key in RetrievalChunk.model_fields
                }
                native.setdefault("embedding_text", str(native.get("content") or ""))
                raw_chunks.append(RetrievalChunk.model_validate(native))
            except Exception as exc:
                log.warning(
                    "graph.query_to_chunks.payload_parse_error",
                    chunk_id=chunk_id,
                    error=str(exc),
                )

        if next_offset is None:
            break
    return raw_chunks


def _order_by_token_overlap(
    *, query_text: str, chunks: list[RetrievalChunk]
) -> list[RetrievalChunk]:
    query_tokens = set(query_text.lower().split())

    def relevance_score(chunk: RetrievalChunk) -> tuple[float, str]:
        if not query_tokens:
            return 0.0, chunk.chunk_id
        chunk_tokens = set(chunk.content.lower().split())
        return len(query_tokens & chunk_tokens) / len(query_tokens), chunk.chunk_id

    return sorted(chunks, key=relevance_score, reverse=True)


async def _ranked_chunk_ids(
    *,
    rank_chunks: RankChunks,
    query_text: str,
    candidate_chunk_ids: list[str],
    top_k: int,
) -> list[str]:
    maybe_ids = rank_chunks(query_text, candidate_chunk_ids, top_k)
    if inspect.isawaitable(maybe_ids):
        return await maybe_ids
    return maybe_ids


def _order_chunks_by_ids(
    chunks: list[RetrievalChunk],
    ranked_ids: list[str],
    candidate_chunk_ids: list[str],
) -> list[RetrievalChunk]:
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for chunk_id in [*ranked_ids, *candidate_chunk_ids]:
        if chunk_id in seen or chunk_id not in by_id:
            continue
        ordered_ids.append(chunk_id)
        seen.add(chunk_id)
    return [by_id[chunk_id] for chunk_id in ordered_ids]


def _tag_retrieval_modes(
    chunks: list[RetrievalChunk],
    *,
    seed_resolution_mode: str,
    final_ranking_mode: str,
) -> list[RetrievalChunk]:
    return [
        chunk.model_copy(
            update={
                "metadata": {
                    **chunk.metadata,
                    "graph_seed_resolution_mode": seed_resolution_mode,
                    "graph_final_ranking_mode": final_ranking_mode,
                }
            }
        )
        for chunk in chunks
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "GraphRetrievalConfig",
    "GraphTraversalResult",
    "KEYWORD_SCROLL_FALLBACK",
    "LEXICAL_RANKING_FALLBACK",
    "SUPPLIED_RANKER",
    "RankChunks",
    "resolve_seed_nodes",
    "expand_neighborhood",
    "graph_query_to_evidence_chunks",
]

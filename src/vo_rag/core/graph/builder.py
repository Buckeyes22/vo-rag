"""Graph build pipeline orchestrator.

Per-book end-to-end graph build: extract → dedupe across books → embed nodes + edges → persist.
Follows the same shape as ``core/summary/builder.py`` and ``core/dsi/``.

Persistence is gated by the graph smoke test.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, cast

import structlog
from pydantic import Field

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.embedding.point_vectors import (
    downsample_colbert_vectors,
    fallback_colbert_vectors,
    include_colbert_at_ingest,
    sparse_payload_to_int_map,
)
from vo_rag.core.graph.direction import orient_edges
from vo_rag.core.graph.extraction import (
    GraphExtractionConfig,
    GraphExtractionResult,
    extract_graph_from_book,
)
from vo_rag.core.graph.persistence import (
    GRAPH_EDGES_COLLECTION,
    GRAPH_NODES_COLLECTION,
    GraphQdrantStore,
    upsert_edge,
    upsert_node,
)
from vo_rag.core.graph.schemas import GraphEdge, GraphNode
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.models.common import VoBaseModel

log = structlog.get_logger(__name__)

_EDGE_WEIGHT_CAP = 100.0
GRAPH_MAX_SKIP_FRACTION = 0.5


class GraphEmbeddingClient(Protocol):
    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = False,
    ) -> dict[str, Any]: ...


class GraphBuildConfig(VoBaseModel):
    extraction_config: GraphExtractionConfig
    embed_batch_size: int = 32
    # Node/edge count per batch scales with chunk count, so this — not max_tokens —
    # is the real lever on per-call output size + decode time. Measured on the
    # pilot book (qwen3-6-27b, 600-char chunks): 8 chunks overran the 2500-token
    # cap (truncated mid-edge → lost nodes); 6 lands at ~1700 output tokens with
    # headroom and reliably completes. All concepts are still captured — they're
    # just spread across more (smaller, faster) batches, so no graph-quality loss.
    chunks_per_extraction_call: int = 6
    max_concurrent_books: int = 4
    # Within-book: extract chunk batches concurrently up to this bound. A
    # sequential await-loop over batches serializes the LLM calls and leaves the
    # endpoint's other workers idle (the cause of the per-book slowness).
    max_concurrent_extraction_calls: int = 8
    max_skip_fraction: float = GRAPH_MAX_SKIP_FRACTION


class GraphBuildStats(VoBaseModel):
    book_id: str
    n_nodes_extracted: int
    n_edges_extracted: int
    n_nodes_after_dedup: int
    n_edges_after_dedup: int
    validation_error_count: int
    wall_time_ms: int
    direction_coverage: float = 1.0
    n_low_confidence_edges: int = 0
    low_confidence_edge_ids: list[str] = Field(default_factory=list)
    n_skipped_batches: int = 0


def dedupe_nodes_across_books(
    nodes_per_book: dict[str, list[GraphNode]],
) -> list[GraphNode]:
    """Merge GraphNode lists from multiple books by node_id.

    Duplicate resolution:
    - book_ids: union of all book_ids lists
    - chunk_ids: union of all chunk_ids lists, sorted
    - aliases: union of all aliases, deduped and sorted
    - created_at: earliest across all duplicates
    """
    merged: dict[str, GraphNode] = {}

    for _book_id, nodes in nodes_per_book.items():
        for node in nodes:
            if node.node_id not in merged:
                merged[node.node_id] = node.model_copy(deep=True)
            else:
                existing = merged[node.node_id]
                # Union book_ids
                combined_book_ids = sorted(set(existing.book_ids) | set(node.book_ids))
                # Union chunk_ids
                combined_chunk_ids = sorted(set(existing.chunk_ids) | set(node.chunk_ids))
                # Union aliases, deduped and sorted
                combined_aliases = sorted(set(existing.aliases) | set(node.aliases))
                # Earliest created_at
                earliest_created_at = (
                    existing.created_at
                    if existing.created_at <= node.created_at
                    else node.created_at
                )
                merged[node.node_id] = existing.model_copy(
                    update={
                        "book_ids": combined_book_ids,
                        "chunk_ids": combined_chunk_ids,
                        "aliases": combined_aliases,
                        "created_at": earliest_created_at,
                    }
                )

    return list(merged.values())


def dedupe_edges_across_books(
    edges_per_book: dict[str, list[GraphEdge]],
) -> list[GraphEdge]:
    """Merge GraphEdge lists from multiple books by edge_id.

    Duplicate resolution:
    - evidence_chunk_ids: union of all evidence_chunk_ids lists, sorted
    - weight: sum of all weights, capped at _EDGE_WEIGHT_CAP
    """
    merged: dict[str, GraphEdge] = {}

    for _book_id, edges in edges_per_book.items():
        for edge in edges:
            if edge.edge_id not in merged:
                merged[edge.edge_id] = edge.model_copy(deep=True)
            else:
                existing = merged[edge.edge_id]
                combined_evidence = sorted(
                    set(existing.evidence_chunk_ids) | set(edge.evidence_chunk_ids)
                )
                combined_book_ids = sorted(set(existing.book_ids) | set(edge.book_ids))
                summed_weight = min(existing.weight + edge.weight, _EDGE_WEIGHT_CAP)
                merged[edge.edge_id] = existing.model_copy(
                    update={
                        "evidence_chunk_ids": combined_evidence,
                        "book_ids": combined_book_ids,
                        "weight": summed_weight,
                    }
                )

    return list(merged.values())


async def build_graph_for_book(
    *,
    book_id: str,
    title: str,
    author: str,
    chunks: list[RetrievalChunk],
    config: GraphBuildConfig,
    llm_call: Callable[[str, str, int, float], Awaitable[LLMResponse]],
) -> tuple[GraphExtractionResult, GraphBuildStats]:
    """Extract, then dedupe within-book graph from chunk batches.

    Splits chunks into batches of size config.chunks_per_extraction_call,
    calls extract_graph_from_book for each batch, concatenates, and dedupes
    within the book before returning.

    Returns (combined_result, stats).
    """
    t0 = time.monotonic()

    batch_size = config.chunks_per_extraction_call
    batches = [chunks[i : i + batch_size] for i in range(0, max(len(chunks), 1), batch_size)]

    all_nodes: list[GraphNode] = []
    all_edges: list[GraphEdge] = []
    total_validation_errors: list[str] = []
    total_input_tokens = 0
    total_output_tokens = 0
    combined_prompt_hash = config.extraction_config.prompt_hash

    sem = asyncio.Semaphore(config.max_concurrent_extraction_calls)

    async def _extract_batch(batch: list[RetrievalChunk]) -> GraphExtractionResult:
        async with sem:
            return await extract_graph_from_book(
                book_id=book_id,
                title=title,
                author=author,
                chunks=batch,
                config=config.extraction_config,
                llm_call=llm_call,
            )

    # Concurrent (bounded) — gather preserves batch order so the downstream
    # dedup stays deterministic.
    batch_jobs = list(enumerate(batches))
    batch_results = await asyncio.gather(
        *(_extract_batch(batch) for _idx, batch in batch_jobs),
        return_exceptions=True,
    )
    successful_batch_results: list[GraphExtractionResult] = []
    for (batch_index, _batch), batch_result in zip(batch_jobs, batch_results, strict=True):
        if isinstance(batch_result, BaseException):
            log.warning(
                "graph_batch_skipped",
                book_id=book_id,
                batch_index=batch_index,
                error=str(batch_result),
            )
            continue
        successful_batch_results.append(batch_result)

    n_total_batches = len(batch_jobs)
    n_skipped_batches = n_total_batches - len(successful_batch_results)
    if len(successful_batch_results) == 0:
        raise RuntimeError(
            f"all graph extraction batches were skipped for book_id={book_id} "
            f"({n_skipped_batches}/{n_total_batches}); refusing to continue"
        )
    skip_fraction = n_skipped_batches / n_total_batches
    if skip_fraction > config.max_skip_fraction:
        raise RuntimeError(
            f"graph batch skip fraction {skip_fraction:.3f} exceeded "
            f"max_skip_fraction {config.max_skip_fraction:.3f} for "
            f"book_id={book_id} ({n_skipped_batches}/{n_total_batches})"
        )

    for result in successful_batch_results:
        all_nodes.extend(result.nodes)
        all_edges.extend(result.edges)
        total_validation_errors.extend(result.validation_errors)
        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens

    # Dedupe within-book: treat all batch results as a single "book"
    nodes_deduped = dedupe_nodes_across_books({book_id: all_nodes})
    edges_deduped = dedupe_edges_across_books({book_id: all_edges})
    edges_oriented, direction_coverage = orient_edges(nodes_deduped, edges_deduped)
    n_swapped = sum(
        1
        for before, after in zip(edges_deduped, edges_oriented, strict=True)
        if before.source_node_id != after.source_node_id
        or before.target_node_id != after.target_node_id
    )

    wall_time_ms = int((time.monotonic() - t0) * 1000)

    combined_result = GraphExtractionResult(
        book_id=book_id,
        nodes=nodes_deduped,
        edges=edges_oriented,
        validation_errors=total_validation_errors,
        prompt_hash=combined_prompt_hash,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        wall_time_ms=wall_time_ms,
    )

    stats = GraphBuildStats(
        book_id=book_id,
        n_nodes_extracted=len(all_nodes),
        n_edges_extracted=len(all_edges),
        n_nodes_after_dedup=len(nodes_deduped),
        n_edges_after_dedup=len(edges_oriented),
        validation_error_count=len(total_validation_errors),
        wall_time_ms=wall_time_ms,
        direction_coverage=direction_coverage.coverage,
        n_low_confidence_edges=direction_coverage.n_low,
        low_confidence_edge_ids=direction_coverage.low_confidence_edge_ids,
        n_skipped_batches=n_skipped_batches,
    )

    log.info(
        "graph.direction.oriented",
        book_id=book_id,
        n_directed=direction_coverage.n_directed,
        n_swapped=n_swapped,
        n_high=direction_coverage.n_high,
        n_medium=direction_coverage.n_medium,
        n_low=direction_coverage.n_low,
        coverage=direction_coverage.coverage,
        stage="build_graph_for_book",
    )

    log.info(
        "graph.build_book.complete",
        book_id=book_id,
        node_count=len(nodes_deduped),
        edge_count=len(edges_deduped),
        n_skipped_batches=n_skipped_batches,
        wall_time_ms=wall_time_ms,
        stage="build_graph_for_book",
    )

    return combined_result, stats


async def build_graph_for_corpus(
    *,
    books: list[tuple[str, str, str, list[RetrievalChunk]]],
    config: GraphBuildConfig,
    llm_call: Callable[[str, str, int, float], Awaitable[LLMResponse]],
) -> tuple[list[GraphNode], list[GraphEdge], list[GraphBuildStats]]:
    """Build graph for all books concurrently, then dedupe across books.

    Processes up to config.max_concurrent_books books in parallel.
    Each element of books is (book_id, title, author, chunks).

    Returns (deduped_nodes, deduped_edges, per_book_stats).
    """
    semaphore = asyncio.Semaphore(config.max_concurrent_books)
    all_stats: list[GraphBuildStats] = []
    nodes_per_book: dict[str, list[GraphNode]] = {}
    edges_per_book: dict[str, list[GraphEdge]] = {}

    async def process_book(
        book_id: str, title: str, author: str, chunks: list[RetrievalChunk]
    ) -> None:
        async with semaphore:
            result, stats = await build_graph_for_book(
                book_id=book_id,
                title=title,
                author=author,
                chunks=chunks,
                config=config,
                llm_call=llm_call,
            )
            nodes_per_book[book_id] = result.nodes
            edges_per_book[book_id] = result.edges
            all_stats.append(stats)

    await asyncio.gather(
        *(process_book(book_id, title, author, chunks) for book_id, title, author, chunks in books)
    )

    deduped_nodes = dedupe_nodes_across_books(nodes_per_book)
    deduped_edges = dedupe_edges_across_books(edges_per_book)

    log.info(
        "graph.build_corpus.complete",
        book_count=len(books),
        node_count=len(deduped_nodes),
        edge_count=len(deduped_edges),
        stage="build_graph_for_corpus",
    )

    return deduped_nodes, deduped_edges, all_stats


async def persist_graph(
    *,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    embedding_client: GraphEmbeddingClient,
    store: GraphQdrantStore,
    nodes_collection: str = GRAPH_NODES_COLLECTION,
    edges_collection: str = GRAPH_EDGES_COLLECTION,
    embed_batch_size: int = 32,
) -> tuple[int, int]:
    """Embed and upsert graph nodes + edges into Qdrant.

    Node embedding text: label + aliases joined by ", "
    Edge embedding text: "source_label + edge_type + target_label"
    (best effort; IDs used if labels unavailable)

    Node vectors: dense + sparse + colbert
    Edge vectors: dense only

    Returns (n_nodes_persisted, n_edges_persisted).
    """
    if getattr(
        store,
        "source_visibility_enabled",
        getattr(store, "source_publication_enabled", False),
    ):
        raise ValueError(
            "per-book graph persistence is unsupported with source publication enabled; "
            "the complete shared graph dependency roster is not available here"
        )

    n_nodes_persisted = 0

    # Build node ID → label map for edge text construction
    node_label_map: dict[str, str] = {n.node_id: n.label for n in nodes}

    # Persist nodes in batches
    for batch_start in range(0, max(len(nodes), 1), embed_batch_size):
        node_batch = nodes[batch_start : batch_start + embed_batch_size]
        if not node_batch:
            break

        node_texts = [
            n.label + (", " + ", ".join(n.aliases) if n.aliases else "") for n in node_batch
        ]

        embeddings = await embedding_client.embed(
            node_texts,
            return_dense=True,
            return_sparse=True,
            return_colbert=include_colbert_at_ingest(),  # opt-in; fallback below
        )
        dense_vecs = _dense_vectors(embeddings, len(node_batch))
        sparse_vecs = _sparse_vectors(embeddings, len(node_batch))
        colbert_vecs = _colbert_vectors(embeddings, dense_vecs, len(node_batch))

        for node, dense, sparse, colbert in zip(
            node_batch,
            dense_vecs,
            sparse_vecs,
            colbert_vecs,
            strict=True,
        ):
            await upsert_node(
                node=node,
                dense_embedding=dense,
                sparse_embedding=sparse,
                colbert_embedding=colbert,
                store=store,
                collection=nodes_collection,
            )
            n_nodes_persisted += 1

    log.info(
        "graph.persist.nodes_done",
        n_nodes_persisted=n_nodes_persisted,
        stage="persist_graph",
    )

    n_edges_persisted = 0

    # Persist edges in batches (dense only)
    for batch_start in range(0, max(len(edges), 1), embed_batch_size):
        edge_batch = edges[batch_start : batch_start + embed_batch_size]
        if not edge_batch:
            break

        edge_texts = [
            " ".join(
                [
                    node_label_map.get(e.source_node_id, e.source_node_id),
                    e.edge_type.value,
                    node_label_map.get(e.target_node_id, e.target_node_id),
                ]
            )
            for e in edge_batch
        ]

        embeddings = await embedding_client.embed(
            edge_texts,
            return_dense=True,
            return_sparse=False,
            return_colbert=False,
        )
        dense_vecs = _dense_vectors(embeddings, len(edge_batch))

        for edge, dense in zip(edge_batch, dense_vecs, strict=True):
            await upsert_edge(
                edge=edge,
                dense_embedding=dense,
                store=store,
                collection=edges_collection,
            )
            n_edges_persisted += 1

    log.info(
        "graph.persist.edges_done",
        n_edges_persisted=n_edges_persisted,
        stage="persist_graph",
    )

    return n_nodes_persisted, n_edges_persisted


def _response_rows(response: dict[str, Any], key: str, expected_len: int) -> list[Any]:
    value = response.get(key)
    if not isinstance(value, list) or len(value) != expected_len:
        msg = f"embedding response {key!r} must contain {expected_len} rows"
        raise ValueError(msg)
    return value


def _dense_vectors(response: dict[str, Any], expected_len: int) -> list[list[float]]:
    rows = _response_rows(response, "dense", expected_len)
    return [[float(value) for value in row] for row in cast(list[list[Any]], rows)]


def _sparse_vectors(response: dict[str, Any], expected_len: int) -> list[dict[int, float]]:
    return [
        sparse_payload_to_int_map(row) for row in _response_rows(response, "sparse", expected_len)
    ]


def _colbert_vectors(
    response: dict[str, Any],
    dense_vecs: list[list[float]],
    expected_len: int,
) -> list[list[list[float]]]:
    rows = response.get("colbert")
    if not isinstance(rows, list) or len(rows) != expected_len:
        return [fallback_colbert_vectors(dense) for dense in dense_vecs]

    vectors: list[list[list[float]]] = []
    for dense, row in zip(dense_vecs, rows, strict=True):
        colbert = downsample_colbert_vectors(row)
        vectors.append(colbert or fallback_colbert_vectors(dense))
    return vectors


__all__ = [
    "GraphBuildConfig",
    "GraphBuildStats",
    "GraphEmbeddingClient",
    "dedupe_nodes_across_books",
    "dedupe_edges_across_books",
    "build_graph_for_book",
    "build_graph_for_corpus",
    "persist_graph",
]

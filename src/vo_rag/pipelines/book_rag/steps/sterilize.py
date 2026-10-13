"""book_rag sterilize step — thin orchestrator around core sterilizer.

Wires ``vo_rag.core.compliance.sterilizer.sterilize_chunks``,
builds ``VoPoint`` records for ``vo_rag_books_sterile_v1``, and upserts via
``QdrantStore``.  Handles the visual-sidecar sterile path
(``figure_composite`` chunks) and enforces the
``min_visual_sterile_yield`` floor.

First-class branch: ``config.sterilize_enabled`` controls whether the
step actually runs sterilization.  When ``False`` the step short-circuits
with a logged skip — the DAG node still exists.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.compliance.sterilizer import sterilize_chunks
from vo_rag.core.embedding.point_vectors import (
    downsample_colbert_vectors,
    fallback_colbert_vectors,
    sparse_payload_to_int_map,
)
from vo_rag.core.ingest.source_publication import SourcePublicationJob
from vo_rag.core.llm.client import LLMClient
from vo_rag.core.retrieval.bm25_tokenizer import tokenize, tokens_to_bm25_sparse
from vo_rag.core.storage.points import VoPoint
from vo_rag.core.storage.qdrant import QdrantStore, deterministic_point_id
from vo_rag.pipelines.book_rag.config import BookRagConfig

logger = logging.getLogger(__name__)

STERILE_COLLECTION = "vo_rag_books_sterile_v1"


async def do_sterilize(
    *,
    raw_chunks: list[RetrievalChunk],
    llm_client: LLMClient,
    config: BookRagConfig,
    store: QdrantStore,
    embed: Callable[[list[str]], Awaitable[dict[str, Any]]],
    generation_id: str | None = None,
    ingest_state: str = "active",
    publication_job: SourcePublicationJob | None = None,
    expected_dependencies: Mapping[str, int | None] | None = None,
    point_ids_out: list[str] | None = None,
) -> list[RetrievalChunk]:
    """Sterilize raw chunks, build vectors, and upsert to the sterile collection.

    Parameters
    ----------
    raw_chunks:
        Chunks produced by the upstream chunking step (may include
        ``figure_composite`` visual-sidecar chunks).
    llm_client:
        Client passed through to ``sterilize_chunks``.
    config:
        Pipeline config.  When ``sterilize_enabled`` is ``False`` the
        step logs a skip and returns ``raw_chunks`` unchanged.
    store:
        Qdrant store for upserting sterile vectors.
    embed:
        Embedding callable (same shape as ``EmbeddingClient.embed``).

    Returns
    -------
    list[RetrievalChunk]
        The sterile chunks (or ``raw_chunks`` when disabled).
    """
    if not config.sterilize_enabled:
        logger.info("book_rag.sterilize skipped sterilize_enabled=False")
        return raw_chunks

    if publication_job is not None and expected_dependencies is None:
        raise ValueError("published sterile output requires a captured raw-book revision")

    if not raw_chunks:
        return []

    sterile_chunks = await sterilize_chunks(raw_chunks, llm_client)

    if sterile_chunks:
        points = await _build_sterile_points(
            sterile_chunks,
            embed,
            generation_id=generation_id,
            ingest_state=ingest_state,
            expected_dependencies=(expected_dependencies if publication_job is not None else None),
        )
        if publication_job is None:
            written_ids = await store.upsert_vectors(STERILE_COLLECTION, points)
        else:
            written_ids = await store.upsert_vectors(
                STERILE_COLLECTION,
                points,
                publication_job=publication_job,
                expected_dependencies=dict(expected_dependencies or {}),
            )
        if point_ids_out is not None:
            if written_ids is None:
                if publication_job is not None:
                    raise RuntimeError("publication-aware sterile writer returned no point IDs")
                point_ids_out.extend(
                    deterministic_point_id(chunk.source_id, index, "sterile", generation_id)
                    for index, chunk in enumerate(sterile_chunks)
                )
            else:
                if len(written_ids) != len(points):
                    raise RuntimeError("sterile writer returned point IDs with wrong cardinality")
                point_ids_out.extend(str(point_id) for point_id in written_ids)

    _check_visual_yield(raw_chunks, sterile_chunks, config)

    return sterile_chunks


async def _build_sterile_points(
    chunks: list[RetrievalChunk],
    embed: Callable[[list[str]], Awaitable[dict[str, Any]]],
    *,
    generation_id: str | None = None,
    ingest_state: str = "active",
    expected_dependencies: Mapping[str, int | None] | None = None,
) -> list[VoPoint]:
    """Embed sterile chunks and build ``VoPoint`` records."""
    texts = [chunk.embedding_text for chunk in chunks]
    resp = await embed(texts)

    dense_batch = resp.get("dense") or []
    sparse_batch = resp.get("sparse") or []
    colbert_batch = resp.get("colbert") or []

    points: list[VoPoint] = []
    for i, chunk in enumerate(chunks):
        dense_i = [float(v) for v in (dense_batch[i] if i < len(dense_batch) else [])]
        colbert_i = downsample_colbert_vectors(colbert_batch[i] if i < len(colbert_batch) else [])
        if not colbert_i:
            colbert_i = fallback_colbert_vectors(dense_i)

        # Denormalize book identity to TOP-LEVEL payload keys (parity with the raw
        # collection — do_upsert_raw). model_dump only carries the nested
        # metadata.* copy; license/identity filtering on the sterile collection
        # needs top-level fields too (the raw and sterile payloads share one shape).
        payload = chunk.model_dump(mode="json")
        meta = chunk.metadata or {}
        payload["source_book"] = meta.get("source_book")
        payload["author"] = meta.get("author")
        payload["license_class"] = meta.get("license_class")
        payload["canonical_source_id"] = meta.get("canonical_source_id")
        payload["project_id"] = meta.get("project_id")
        payload["domain_tags"] = meta.get("domain_tags")
        payload["generation_id"] = generation_id
        payload["ingest_state"] = ingest_state
        if expected_dependencies is not None:
            payload["dependency_revisions"] = dict(expected_dependencies)

        points.append(
            VoPoint(
                source_id=chunk.source_id,
                chunk_index=i,
                stage="sterile",
                version_hash=generation_id,
                dense=dense_i,
                sparse=sparse_payload_to_int_map(sparse_batch[i] if i < len(sparse_batch) else {}),
                bm25=tokens_to_bm25_sparse(tokenize(chunk.embedding_text)),
                colbert=colbert_i,
                payload=payload,
            )
        )

    return points


def _check_visual_yield(
    raw_chunks: list[RetrievalChunk],
    sterile_chunks: list[RetrievalChunk],
    config: BookRagConfig,
) -> None:
    """Log a warning when the visual-sidecar sterile yield drops below floor."""
    raw_visual = [c for c in raw_chunks if c.chunk_type == "figure_composite"]
    if not raw_visual:
        return

    sterile_visual = [c for c in sterile_chunks if c.chunk_type == "figure_composite"]
    yield_ratio = len(sterile_visual) / len(raw_visual) if raw_visual else 1.0

    if yield_ratio < config.min_visual_sterile_yield:
        source_id = raw_chunks[0].source_id if raw_chunks else "unknown"
        logger.warning(
            ("book_rag.sterilize visual_yield_below_floor source_id=%s yield=%.2f floor=%.2f"),
            source_id,
            yield_ratio,
            config.min_visual_sterile_yield,
        )


__all__ = ["do_sterilize", "STERILE_COLLECTION"]

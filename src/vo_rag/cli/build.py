"""Build CLI commands for DSI, graph, summary, and RAPTOR corpus drivers.

These commands construct builders + stores directly (heavy RunPod/Qdrant ops),
unlike the ingest/admin commands which are API-client based.

OPERATOR CONTRACT: This CLI is a PRODUCTION surface that MUST import and
run with the ``llamaindex`` extra ABSENT.  It executes the core builders directly
via the ``_build_*_impl`` coroutines (which are llama_index-free).
``AdvancedBuildWorkflow`` in ``src/vo_rag/integrations/llamaindex/build_workflow.py``
is the OPTIONAL adapter-tier orchestration that lifts the same core coroutines
for the agent/adapter surfaces.  Routing the CLI through the workflow would force
the production CLI to require the optional extra — an architecture inversion and
a break of the optional-extra contract.  Therefore the CLI intentionally stays
on the direct core path.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import Any, cast

import click
import structlog

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.chunking.models import payload_embedding_text
from vo_rag.core.config import get_settings
from vo_rag.core.dsi.builder import (
    _DSI_COLLECTION_DEFAULT,
    DsiBuildConfig,
    admit_dsi_source_revision,
    build_dsi_for_book,
    ensure_dsi_collection,
    persist_dsi_records,
)
from vo_rag.core.dsi.extraction import ExtractionConfig
from vo_rag.core.dsi.prompts import DSI_EXTRACTION_PROMPT_HASH
from vo_rag.core.embedding import build_embedding_client
from vo_rag.core.embedding.point_vectors import include_colbert_at_ingest
from vo_rag.core.graph.builder import (
    GraphBuildConfig,
    build_graph_for_book,
    persist_graph,
)
from vo_rag.core.graph.direction import orient_low_confidence_via_llm
from vo_rag.core.graph.extraction import GraphExtractionConfig, graph_response_json_schema
from vo_rag.core.graph.persistence import (
    GRAPH_EDGES_COLLECTION,
    GRAPH_NODES_COLLECTION,
    GraphQdrantClient,
    ensure_graph_collections,
)
from vo_rag.core.graph.prompts import GRAPH_EXTRACTION_PROMPT_HASH
from vo_rag.core.llm.client import LLMClient
from vo_rag.core.llm.guided import structured_output_extra
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.raptor.builder import RaptorBuildConfig, build_raptor_for_corpus
from vo_rag.core.raptor.persistence import RAPTOR_COLLECTION
from vo_rag.core.storage.qdrant import QdrantStore
from vo_rag.core.summary.builder import BuildConfig, SummaryTreeBuilder, build_summary_for_corpus
from vo_rag.core.summary.persistence import SUMMARY_TREE_COLLECTION

log = structlog.get_logger(__name__)
_RAW_COLLECTION = "vo_rag_books_raw_v1"
_LEAF_EMBED_BATCH_SIZE = 8
_GRAPH_DIRECTION_MIN_COVERAGE_DEFAULT = 0.85


def _collection(name: str) -> str:
    return name + os.environ.get("VORAG_BUILD_COLLECTION_SUFFIX", "")


def _graph_direction_min_coverage() -> float:
    raw_value = os.environ.get(
        "VORAG_GRAPH_DIRECTION_MIN_COVERAGE",
        str(_GRAPH_DIRECTION_MIN_COVERAGE_DEFAULT),
    )
    try:
        return float(raw_value)
    except ValueError:
        log.warning(
            "graph.direction.threshold_invalid",
            raw_value=raw_value,
            default=_GRAPH_DIRECTION_MIN_COVERAGE_DEFAULT,
        )
        return _GRAPH_DIRECTION_MIN_COVERAGE_DEFAULT


def _graph_27b_fallback_enabled() -> bool:
    raw_value = os.environ.get("VORAG_GRAPH_27B_FALLBACK", "0").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def _graph_local_reorient_enabled() -> bool:
    raw_value = os.environ.get("VORAG_GRAPH_LOCAL_REORIENT", "1").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def _runpod_text_fallback_available(settings: Any) -> bool:
    serverless = getattr(settings, "serverless", None)
    if serverless is None:
        log.warning("graph.direction.fallback_unavailable", reason="missing_serverless_settings")
        return False
    if not getattr(serverless, "api_key", ""):
        log.warning("graph.direction.fallback_unavailable", reason="missing_runpod_api_key")
        return False
    try:
        serverless.base_url("text", ignore_override=True)
    except (KeyError, ValueError) as exc:
        log.warning(
            "graph.direction.fallback_unavailable",
            reason="missing_text_endpoint",
            error=str(exc),
        )
        return False
    return True


def _build_runpod_text_client() -> LLMClient:
    from vo_rag.core.cloud.serverless import ServerlessClient
    from vo_rag.core.llm.providers.runpod_serverless_provider import (
        RunpodServerlessProvider,
    )

    serverless = get_settings().serverless
    client = ServerlessClient(
        base_url=serverless.base_url("text", ignore_override=True),
        api_key=serverless.api_key,
        model=serverless.model_name("text"),
        timeout_s=serverless.request_timeout_s,
    )
    return LLMClient(provider=RunpodServerlessProvider(client=client))


def _structured_output_extra(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Local alias for :func:`vo_rag.core.llm.guided.structured_output_extra`."""
    return structured_output_extra(schema)


async def _llm_call_from_client(
    client: LLMClient,
    *,
    guided_json: dict[str, Any] | None = None,
) -> Callable[[str, str, int, float], Awaitable[LLMResponse]]:
    """Return an adapter matching the book_rag workflow's ``_llm_call`` shape.

    ``guided_json`` remains the caller-facing parameter name, but it carries a
    JSON Schema dict. ``VORAG_GUIDED_DECODING_MODE`` controls whether that schema
    is expressed as portable ``response_format: json_schema`` (default), legacy
    top-level ``guided_json``, or schema-free ``response_format: json_object``.
    The default is ``json_schema`` because stock vLLM ignores top-level
    ``guided_json`` while honoring OpenAI-compatible ``response_format``.
    The advanced-layer builders take a 4-arg llm_call with no ``extra`` slot, so
    these settings and ``enable_thinking=False`` are baked into the adapter here.
    """
    extra = _structured_output_extra(guided_json)

    async def _llm_call(system: str, user: str, max_tokens: int, temperature: float) -> LLMResponse:
        resp: LLMResponse = await client.complete(
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
            extra=extra,
        )
        return resp

    return _llm_call


async def _load_chunks_for_source(
    store: QdrantStore,
    source_id: str,
    collection: str = _RAW_COLLECTION,
    *,
    version_hash: str | None = None,
) -> list[RetrievalChunk]:
    """Scroll *collection* filtered by ``source_id`` and reconstruct chunks.

    When ``version_hash`` is given, the scroll is additionally filtered to that
    version so advanced-layer builds don't ingest a degraded mix of
    old- and current-version chunks for a re-ingested book. Default ``None`` keeps
    the historical source_id-only behavior (correct for vo-rag's unversioned
    book_rag chunks; the filter matters for the migrated golden corpus).
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    chunks: list[RetrievalChunk] = []
    offset: Any = None
    # list[Any]: Filter.must is an invariant list of a Condition union — a bare
    # list[FieldCondition] doesn't satisfy it under mypy strict.
    must: list[Any] = [FieldCondition(key="source_id", match=MatchValue(value=source_id))]
    if version_hash is not None:
        must.append(FieldCondition(key="version_hash", match=MatchValue(value=version_hash)))
    scroll_filter = Filter(must=must)
    while True:
        results, next_offset = await store._client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for point in results:
            payload = point.payload or {}
            meta = payload.get("metadata") or {}
            if not isinstance(meta, dict):
                meta = {}

            chunks.append(
                RetrievalChunk(
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    chunk_type=str(payload.get("chunk_type", "text")),
                    document_id=str(payload.get("document_id", source_id)),
                    source_id=str(payload.get("source_id", source_id)),
                    page_number=payload.get("page_number"),
                    page_range=payload.get("page_range"),
                    section_path=str(payload.get("section_path", "")),
                    section_title=str(payload.get("section_title", "")),
                    chapter_path=str(payload.get("chapter_path", "")),
                    content=str(payload.get("content", "")),
                    embedding_text=payload_embedding_text(payload),
                    token_count=int(payload.get("token_count", 0) or 0),
                    has_image=bool(payload.get("has_image", False)),
                    image_id=payload.get("image_id"),
                    image_ref=payload.get("image_ref"),
                    vlm_description=payload.get("vlm_description"),
                    metadata=dict(meta),
                )
            )
        if next_offset is None:
            break
        offset = next_offset
    return chunks


def _book_meta_from_chunks(chunks: list[RetrievalChunk]) -> tuple[str, str, list[str]]:
    """Extract (title, author, category_ids) from the first chunk's metadata."""
    if not chunks:
        return "", "", []
    meta = chunks[0].metadata
    if not isinstance(meta, dict):
        return "", "", []
    # Book identity is stamped by do_build_chunks. Accept either the clean
    # `title` key or the `source_book` ("Title (Author)") key used by the
    # migrated golden corpus.
    title = str(meta.get("title") or meta.get("source_book") or "")
    author = str(meta.get("author") or "")
    domain_tags = meta.get("domain_tags") or []
    if not isinstance(domain_tags, list):
        domain_tags = []
    return title, author, [str(t) for t in domain_tags]


async def _build_dsi_impl(
    source_id: str,
    collection_version: str,
    *,
    n_representative_chunks: int = 16,
    max_tokens: int = 3000,
    version_hash: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    store = QdrantStore(
        settings.qdrant.url,
        settings.qdrant.api_key or None,
        prefer_grpc=settings.qdrant.prefer_grpc,
    )
    embedding_client = build_embedding_client(
        settings.embedding.batch_url,
        timeout_s=float(getattr(settings.embedding, "timeout_s", 30.0)),
    )
    llm_client = LLMClient(provider="runpod_serverless", role="dsi")
    try:
        dsi_collection = _collection(_DSI_COLLECTION_DEFAULT)
        expected_dependencies = (
            {
                source_id: await store.active_source_revision(_RAW_COLLECTION, source_id),
            }
            if getattr(store, "source_visibility_enabled", False)
            else {}
        )
        if expected_dependencies and collection_version != _RAW_COLLECTION:
            raise click.ClickException(
                "published DSI currently requires the canonical raw Books collection"
            )
        chunks = await _load_chunks_for_source(
            store, source_id, collection_version, version_hash=version_hash
        )
        if not chunks:
            raise click.ClickException(
                f"No chunks found for source_id={source_id} in {collection_version}"
            )

        title, author, category_ids = _book_meta_from_chunks(chunks)

        config = DsiBuildConfig(
            extraction_config=ExtractionConfig(
                summary_model="qwen3.6-27b-fp8",
                temperature=0.2,
                max_tokens=max_tokens,
                n_representative_chunks=n_representative_chunks,
                prompt_hash=DSI_EXTRACTION_PROMPT_HASH,
            ),
        )

        publication_job = await admit_dsi_source_revision(
            store=store,
            source_id=source_id,
            collection=dsi_collection,
            config=config,
            source_collection_version=collection_version,
            expected_dependencies=expected_dependencies,
        )

        try:
            llm_call = await _llm_call_from_client(llm_client)
            record, stats = await build_dsi_for_book(
                book_id=source_id,
                title=title,
                author=author,
                category_ids=category_ids,
                chunks=chunks,
                config=config,
                llm_call=llm_call,
                source_collection_version=collection_version,
            )

            n_records = 0
            if record is not None:
                await ensure_dsi_collection(store._client, collection=dsi_collection)
                n_records = await persist_dsi_records(
                    records=[record],
                    embed=lambda texts: embedding_client.embed(
                        texts,
                        return_dense=True,
                        return_sparse=True,
                        return_colbert=include_colbert_at_ingest(),
                    ),
                    store=store,
                    collection=dsi_collection,
                    publication_job=publication_job,
                    expected_dependencies=expected_dependencies,
                )
            elif publication_job is not None:
                await store.abort_source_revision(
                    publication_job, "DSI extraction produced no record"
                )
        except BaseException:
            if publication_job is not None:
                await store.abort_source_revision(publication_job, "DSI producer failed")
            raise
        return {
            "book_id": source_id,
            "n_records": n_records,
            "n_chunks": len(chunks),
            "stats": stats,
        }
    finally:
        await embedding_client.aclose()
        await llm_client.aclose()


async def _build_graph_impl(
    source_id: str,
    *,
    max_tokens: int = 4000,
    chunks_per_call: int = 6,
    max_concurrent: int = 24,
    version_hash: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    store = QdrantStore(
        settings.qdrant.url,
        settings.qdrant.api_key or None,
        prefer_grpc=settings.qdrant.prefer_grpc,
    )
    if getattr(store, "source_visibility_enabled", False):
        raise click.ClickException(
            "graph persistence is disabled while source publication readers are enabled; "
            "the shared graph dependency closure is not available before model work"
        )
    embedding_client = build_embedding_client(
        settings.embedding.batch_url,
        timeout_s=float(getattr(settings.embedding, "timeout_s", 30.0)),
    )
    llm_client = LLMClient(provider="runpod_serverless", role="text")
    try:
        nodes_collection = _collection(GRAPH_NODES_COLLECTION)
        edges_collection = _collection(GRAPH_EDGES_COLLECTION)
        await ensure_graph_collections(
            cast(GraphQdrantClient, store._client),
            nodes_collection=nodes_collection,
            edges_collection=edges_collection,
        )

        chunks = await _load_chunks_for_source(
            store, source_id, _RAW_COLLECTION, version_hash=version_hash
        )
        if not chunks:
            raise click.ClickException(
                f"No chunks found for source_id={source_id} in {_RAW_COLLECTION}"
            )

        title, author, _category_ids = _book_meta_from_chunks(chunks)

        config = GraphBuildConfig(
            extraction_config=GraphExtractionConfig(
                summary_model="qwen3.6-27b-fp8",
                temperature=0.2,
                max_tokens=max_tokens,
                prompt_hash=GRAPH_EXTRACTION_PROMPT_HASH,
            ),
            chunks_per_extraction_call=chunks_per_call,
            max_concurrent_extraction_calls=max_concurrent,
        )

        # Constrain extraction output with the configured structured-output mode;
        # the parameter name is historical, but it carries the graph JSON Schema.
        llm_call = await _llm_call_from_client(llm_client, guided_json=graph_response_json_schema())
        result, stats = await build_graph_for_book(
            book_id=source_id,
            title=title,
            author=author,
            chunks=chunks,
            config=config,
            llm_call=llm_call,
        )
        direction_threshold = _graph_direction_min_coverage()
        direction_status = (
            "sufficient"
            if stats.direction_coverage >= direction_threshold
            else (
                f"insufficient: {stats.n_low_confidence_edges} low-confidence edges "
                "are LLM reorientation candidates"
            )
        )
        log.info(
            "graph.direction.coverage",
            book_id=source_id,
            coverage=stats.direction_coverage,
            threshold=direction_threshold,
            status=direction_status,
            n_low_confidence_edges=stats.n_low_confidence_edges,
        )

        edges_to_persist = result.edges
        n_reoriented = 0
        if stats.direction_coverage >= direction_threshold:
            reorientation_status = "not_needed"
            log.info(
                "graph.direction.fallback",
                book_id=source_id,
                status=reorientation_status,
                coverage=stats.direction_coverage,
                threshold=direction_threshold,
                n_candidates=stats.n_low_confidence_edges,
            )
        elif not stats.low_confidence_edge_ids:
            reorientation_status = "no_candidates"
            log.info(
                "graph.direction.fallback",
                book_id=source_id,
                status=reorientation_status,
                coverage=stats.direction_coverage,
                threshold=direction_threshold,
            )
        else:
            runpod_enabled = _graph_27b_fallback_enabled()
            runpod_available = runpod_enabled and _runpod_text_fallback_available(settings)
            local_reorient_enabled = _graph_local_reorient_enabled()

            if runpod_available:
                runpod_client = _build_runpod_text_client()
                try:
                    fallback_call = await _llm_call_from_client(runpod_client, guided_json=None)
                    corrected_edges = await orient_low_confidence_via_llm(
                        result.nodes,
                        result.edges,
                        stats.low_confidence_edge_ids,
                        fallback_call,
                    )
                    n_reoriented = sum(
                        1
                        for before, after in zip(result.edges, corrected_edges, strict=True)
                        if before.source_node_id != after.source_node_id
                        or before.target_node_id != after.target_node_id
                    )
                    edges_to_persist = corrected_edges
                    reorientation_status = "applied"
                    log.info(
                        "graph.direction.fallback",
                        book_id=source_id,
                        status=reorientation_status,
                        n_candidates=len(stats.low_confidence_edge_ids),
                        n_reoriented=n_reoriented,
                    )
                finally:
                    await runpod_client.aclose()
            elif local_reorient_enabled:
                local_fallback_call = await _llm_call_from_client(llm_client, guided_json=None)
                corrected_edges = await orient_low_confidence_via_llm(
                    result.nodes,
                    result.edges,
                    stats.low_confidence_edge_ids,
                    local_fallback_call,
                )
                n_reoriented = sum(
                    1
                    for before, after in zip(result.edges, corrected_edges, strict=True)
                    if before.source_node_id != after.source_node_id
                    or before.target_node_id != after.target_node_id
                )
                edges_to_persist = corrected_edges
                reorientation_status = "applied_local"
                log.info(
                    "graph.direction.fallback",
                    book_id=source_id,
                    status=reorientation_status,
                    n_candidates=len(stats.low_confidence_edge_ids),
                    n_reoriented=n_reoriented,
                )
            elif runpod_enabled:
                reorientation_status = "unavailable"
                log.info(
                    "graph.direction.fallback",
                    book_id=source_id,
                    status=reorientation_status,
                    coverage=stats.direction_coverage,
                    threshold=direction_threshold,
                    n_candidates=stats.n_low_confidence_edges,
                )
            else:
                reorientation_status = "disabled"
                log.info(
                    "graph.direction.fallback",
                    book_id=source_id,
                    status=reorientation_status,
                    n_candidates=stats.n_low_confidence_edges,
                )

        n_nodes, n_edges = await persist_graph(
            nodes=result.nodes,
            edges=edges_to_persist,
            embedding_client=embedding_client,
            store=store,
            nodes_collection=nodes_collection,
            edges_collection=edges_collection,
        )
        return {
            "book_id": source_id,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "direction_coverage": stats.direction_coverage,
            "n_reoriented": n_reoriented,
            "reorientation_status": reorientation_status,
            "n_low_confidence_edges": stats.n_low_confidence_edges,
            "low_confidence_edge_ids": stats.low_confidence_edge_ids,
            "stats": stats,
        }
    finally:
        await embedding_client.aclose()
        await llm_client.aclose()


async def _build_summary_impl(
    source_id: str,
    *,
    max_concurrent: int = 8,
    version_hash: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    store = QdrantStore(
        settings.qdrant.url,
        settings.qdrant.api_key or None,
        prefer_grpc=settings.qdrant.prefer_grpc,
    )
    embedding_client = build_embedding_client(
        settings.embedding.batch_url,
        timeout_s=float(getattr(settings.embedding, "timeout_s", 30.0)),
    )
    llm_client = LLMClient(provider="runpod_serverless", role="text")
    try:
        raw_dependency_revision = (
            await store.active_source_revision(_RAW_COLLECTION, source_id)
            if getattr(store, "source_visibility_enabled", False)
            else None
        )
        chunks = await _load_chunks_for_source(
            store, source_id, _RAW_COLLECTION, version_hash=version_hash
        )
        if not chunks:
            raise click.ClickException(
                f"No chunks found for source_id={source_id} in {_RAW_COLLECTION}"
            )

        config = BuildConfig(
            max_concurrent_llm_calls=max_concurrent,
            summary_collection=_collection(SUMMARY_TREE_COLLECTION),
        )
        builder = SummaryTreeBuilder(
            llm_client=llm_client,
            embed_client=embedding_client,
            store=store,
            config=config,
        )
        nodes, _stats = await build_summary_for_corpus(
            books=[(source_id, chunks)],
            builder=builder,
            max_concurrent_books=1,
            expected_dependencies={source_id: raw_dependency_revision},
        )
        return {
            "book_id": source_id,
            "n_nodes": len(nodes),
            "n_chunks": len(chunks),
        }
    finally:
        await embedding_client.aclose()
        await llm_client.aclose()


async def _embed_leaf_embeddings(
    chunks: list[RetrievalChunk],
    embedding_client: Any,
    *,
    embed_batch: int = _LEAF_EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """Return dense leaf vectors aligned 1:1 with *chunks* for RAPTOR clustering."""
    texts = [chunk.embedding_text or chunk.content for chunk in chunks]
    embed_batch = max(1, embed_batch)
    leaf_embeddings: list[list[float]] = []

    for start in range(0, len(texts), embed_batch):
        batch_texts = texts[start : start + embed_batch]
        resp: dict[str, Any] = await embedding_client.embed(
            batch_texts,
            return_dense=True,
            return_sparse=False,
            return_colbert=False,
        )
        dense_batch = resp.get("dense") or []
        for vector in dense_batch:
            leaf_embeddings.append([float(value) for value in vector])

    if len(leaf_embeddings) != len(chunks):
        raise RuntimeError(
            f"leaf embedding count ({len(leaf_embeddings)}) does not match "
            f"chunk count ({len(chunks)})"
        )
    return leaf_embeddings


async def _build_raptor_impl(
    source_id: str,
    *,
    max_tokens: int = 2500,
    version_hash: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    store = QdrantStore(
        settings.qdrant.url,
        settings.qdrant.api_key or None,
        prefer_grpc=settings.qdrant.prefer_grpc,
    )
    embedding_client = build_embedding_client(
        settings.embedding.batch_url,
        timeout_s=float(getattr(settings.embedding, "timeout_s", 30.0)),
    )
    llm_client = LLMClient(provider="runpod_serverless", role="text")
    try:
        raw_dependency_revision = (
            await store.active_source_revision(_RAW_COLLECTION, source_id)
            if getattr(store, "source_visibility_enabled", False)
            else None
        )
        chunks = await _load_chunks_for_source(
            store, source_id, _RAW_COLLECTION, version_hash=version_hash
        )
        if not chunks:
            raise click.ClickException(
                f"No chunks found for source_id={source_id} in {_RAW_COLLECTION}"
            )

        leaf_embeddings = await _embed_leaf_embeddings(chunks, embedding_client)
        if len(leaf_embeddings) != len(chunks):
            raise RuntimeError(
                f"leaf embedding count ({len(leaf_embeddings)}) does not match "
                f"chunk count ({len(chunks)})"
            )

        config = RaptorBuildConfig(
            summary_model="qwen3.6-27b-fp8",
            summary_max_tokens=max_tokens,
            raptor_collection=_collection(RAPTOR_COLLECTION),
        )
        llm_call = await _llm_call_from_client(llm_client)
        results, _stats = await build_raptor_for_corpus(
            books=[(source_id, chunks, leaf_embeddings)],
            config=config,
            llm_call=llm_call,
            embed_client=embedding_client,
            store=store,
            max_concurrent_books=1,
            expected_dependencies={source_id: raw_dependency_revision},
        )
        n_nodes = sum(len(result.nodes) for result in results)
        return {
            "book_id": source_id,
            "n_nodes": n_nodes,
            "n_chunks": len(chunks),
        }
    finally:
        await embedding_client.aclose()
        await llm_client.aclose()


def register_build_commands(root: click.Group) -> None:
    @root.group("build")
    def build() -> None:
        """Direct builder commands for DSI, graph, summary, and RAPTOR corpora."""

    @build.command("dsi")
    @click.option("--source-id", required=True, help="Book source_id to build DSI for.")
    @click.option(
        "--collection-version",
        default=_RAW_COLLECTION,
        show_default=True,
        help="Raw collection to read chunks from.",
    )
    @click.option(
        "--n-representative-chunks",
        default=16,
        show_default=True,
        help="Representative chunks fed to the extraction prompt (each truncated to 600 chars). "
        "Tune down to shrink the prompt for tighter-context models / lower cost.",
    )
    @click.option(
        "--max-tokens",
        default=3000,
        show_default=True,
        help="Max output tokens for the DSI extraction call (the faceted JSON record).",
    )
    @click.option(
        "--version-hash",
        default=None,
        help="Only read chunks with this version_hash. Omit for "
        "unversioned vo-rag corpora; set it to isolate a version on the migrated corpus.",
    )
    def build_dsi(
        source_id: str,
        collection_version: str,
        n_representative_chunks: int,
        max_tokens: int,
        version_hash: str | None,
    ) -> None:
        """Build DSI record for a single book and persist to Qdrant."""
        summary = asyncio.run(
            _build_dsi_impl(
                source_id,
                collection_version,
                n_representative_chunks=n_representative_chunks,
                max_tokens=max_tokens,
                version_hash=version_hash,
            )
        )
        click.echo(
            "{book_id: "
            + str(summary["book_id"])
            + ", n_records: "
            + str(summary["n_records"])
            + ", n_chunks: "
            + str(summary["n_chunks"])
            + "}"
        )

    @build.command("graph")
    @click.option("--source-id", required=True, help="Book source_id to build graph for.")
    @click.option(
        "--max-tokens",
        default=4000,
        show_default=True,
        help="Max output tokens per extraction call. 4000 covers dense 6-chunk batches; "
        "2500 truncated ~40% and 3500 ~3% (output scales with node count, not chunk size).",
    )
    @click.option(
        "--chunks-per-call",
        default=6,
        show_default=True,
        help="Chunks per extraction batch. Smaller = smaller/faster calls (more batches). "
        "6 keeps qwen3-6-27b output within the 4000-token cap.",
    )
    @click.option(
        "--max-concurrent",
        default=24,
        show_default=True,
        help="In-flight extraction calls. Higher than worker count on purpose: each vLLM "
        "worker continuous-batches up to MAX_CONCURRENCY requests, so oversubscribing "
        "lifts aggregate throughput well past 1-request-per-worker.",
    )
    @click.option(
        "--version-hash",
        default=None,
        help="Only read chunks with this version_hash. Omit for "
        "unversioned vo-rag corpora; set it to isolate a version on the migrated corpus.",
    )
    def build_graph(
        source_id: str,
        max_tokens: int,
        chunks_per_call: int,
        max_concurrent: int,
        version_hash: str | None,
    ) -> None:
        """Build knowledge graph for a single book and persist to Qdrant."""
        summary = asyncio.run(
            _build_graph_impl(
                source_id,
                max_tokens=max_tokens,
                chunks_per_call=chunks_per_call,
                max_concurrent=max_concurrent,
                version_hash=version_hash,
            )
        )
        click.echo(
            "{book_id: "
            + str(summary["book_id"])
            + ", n_nodes: "
            + str(summary["n_nodes"])
            + ", n_edges: "
            + str(summary["n_edges"])
            + ", direction_coverage: "
            + str(summary["direction_coverage"])
            + ", n_low_confidence_edges: "
            + str(summary["n_low_confidence_edges"])
            + "}"
        )

    @build.command("summary")
    @click.option("--source-id", required=True, help="Book source_id to build summary tree for.")
    @click.option(
        "--max-concurrent",
        default=8,
        show_default=True,
        help="In-flight summary LLM calls (maps to BuildConfig.max_concurrent_llm_calls).",
    )
    @click.option(
        "--version-hash",
        default=None,
        help="Only read chunks with this version_hash. Omit for "
        "unversioned vo-rag corpora; set it to isolate a version on the migrated corpus.",
    )
    def build_summary(
        source_id: str,
        max_concurrent: int,
        version_hash: str | None,
    ) -> None:
        """Build summary tree for a single book and persist to Qdrant."""
        summary = asyncio.run(
            _build_summary_impl(
                source_id,
                max_concurrent=max_concurrent,
                version_hash=version_hash,
            )
        )
        click.echo(
            "{book_id: "
            + str(summary["book_id"])
            + ", n_nodes: "
            + str(summary["n_nodes"])
            + ", n_chunks: "
            + str(summary["n_chunks"])
            + "}"
        )

    @build.command("raptor")
    @click.option("--source-id", required=True, help="Book source_id to build RAPTOR tree for.")
    @click.option(
        "--max-tokens",
        default=2500,
        show_default=True,
        help="Max output tokens per cluster-summary call. 800 truncated large clusters; "
        "2500 covers faithful summaries.",
    )
    @click.option(
        "--version-hash",
        default=None,
        help="Only read chunks with this version_hash. Omit for "
        "unversioned vo-rag corpora; set it to isolate a version on the migrated corpus.",
    )
    def build_raptor(
        source_id: str,
        max_tokens: int,
        version_hash: str | None,
    ) -> None:
        """Build RAPTOR tree for a single book and persist to Qdrant."""
        summary = asyncio.run(
            _build_raptor_impl(
                source_id,
                max_tokens=max_tokens,
                version_hash=version_hash,
            )
        )
        click.echo(
            "{book_id: "
            + str(summary["book_id"])
            + ", n_nodes: "
            + str(summary["n_nodes"])
            + ", n_chunks: "
            + str(summary["n_chunks"])
            + "}"
        )

"""RAPTOR tree builder for Layer 1B.

Builds a recursive abstractive tree over semantic clusters of chunk embeddings.
Mirrors ``vo_rag.core.dsi.builder`` and ``vo_rag.core.summary.builder`` patterns.

Public API
----------
RaptorBuildConfig
    Wire-config for a RAPTOR build run.
RaptorBuildResult
    All nodes + root produced by a book build.
RaptorBuildStats
    Per-book outcome record emitted after ``build_raptor_for_book`` completes.
build_raptor_for_book(...)
    Cluster -> summarize -> embed -> persist for one book.
build_raptor_for_corpus(...)
    Semaphore-bounded corpus-level driver.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

import numpy as np
import structlog

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.embedding.point_vectors import (
    fallback_colbert_vectors,
    include_colbert_at_ingest,
)
from vo_rag.core.ingest.representation_receipt import canonical_sha256, representation_receipt
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.models.common import VoBaseModel
from vo_rag.core.raptor.clustering import ClusterAssignment, ClusteringResult, cluster_chunks
from vo_rag.core.raptor.extract_support import (
    assemble_recursive_evidence,
    build_level_extract_support_prompt,
)
from vo_rag.core.raptor.persistence import (
    RAPTOR_COLLECTION,
    cluster_node_point,
    ensure_raptor_collection,
    get_book_tree,
    get_cluster_node,
    upsert_cluster_node,
)
from vo_rag.core.raptor.prompts import PROMPT_HASHES
from vo_rag.core.raptor.schemas import (
    ClusterLevel,
    ClusterNode,
    ClusterReproducibilityConfig,
    cluster_node_id_for,
)
from vo_rag.core.storage.derived_publication import (
    DerivedSourceBinding,
    abort_derived_source,
    admit_derived_source,
    publish_derived_source,
)
from vo_rag.core.storage.source_revision_writer import SourceRevisionWriter
from vo_rag.core.summary.builder import _parse_summary_response, _strip_reasoning

log = structlog.get_logger(__name__)

RETRY_DELAYS = (10.0, 30.0, 90.0)
SUMMARY_LLM_EXTRA: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}


class _ClusterFunc(Protocol):
    def __call__(
        self,
        chunk_ids: list[str],
        embeddings: Any,
        *,
        config: ClusterReproducibilityConfig,
    ) -> ClusteringResult: ...


# ---------------------------------------------------------------------------
# Config + result + stats
# ---------------------------------------------------------------------------


class RaptorBuildConfig(VoBaseModel):
    """Wire-configuration for a RAPTOR build run."""

    summary_model: str
    embedding_model: str = "BAAI/bge-m3"
    embedding_version: str = "bge-m3@unversioned"
    source_collection_version: str = "vo_rag_books_raw_v1"
    raptor_collection: str = RAPTOR_COLLECTION
    max_concurrent_llm_calls: int = 8
    max_tree_levels: int = 5
    max_skip_fraction: float = 0.5
    summary_temperature: float = 0.2
    summary_max_tokens: int = 2500
    # 16,384 ctx - 2,500 output - template overhead, sized for token-DENSE text.
    # Measured: 8a67 (Motorcycle Dynamics, math-heavy) is ~3.46 chars/token, so
    # 48k chars = 13,885 tok and 13,885 + 2,500 overflowed 16,384 by 1 token.
    # 32k chars ~= 9,972 tok on that content -> comfortable headroom. Only
    # pathological dense clusters hit this cap. (VOR-129)
    max_summary_input_chars: int = 32000
    reproducibility_config: ClusterReproducibilityConfig = ClusterReproducibilityConfig(
        umap_random_state=42,
        gmm_random_state=42,
    )


class RaptorBuildResult(VoBaseModel):
    """All nodes and root produced by ``build_raptor_for_book``."""

    book_id: str
    nodes: list[ClusterNode]
    root: ClusterNode | None


class RaptorBuildStats(VoBaseModel):
    """Per-book outcome record emitted after ``build_raptor_for_book`` completes."""

    book_id: str
    level_count: int
    node_count: int
    input_tokens: int
    output_tokens: int
    wall_time_ms: int
    n_skipped_clusters: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _level_for_num(n: int) -> ClusterLevel:
    levels = [ClusterLevel.LEVEL_0, ClusterLevel.LEVEL_1, ClusterLevel.LEVEL_2]
    return levels[min(n, len(levels) - 1)]


def _prompt_hash_for(level: ClusterLevel) -> str:
    return PROMPT_HASHES.get(level, "unknown")


def _item_id(item: RetrievalChunk | ClusterNode) -> str:
    if isinstance(item, RetrievalChunk):
        return item.chunk_id
    return item.node_id


def _union_raw_chunks(
    members: list[RetrievalChunk | ClusterNode],
    raw_map: dict[str, list[RetrievalChunk]],
) -> list[RetrievalChunk]:
    """Union raw chunks for a list of member items (chunks or nodes)."""
    result: list[RetrievalChunk] = []
    seen: set[str] = set()
    for member in members:
        key = _item_id(member)
        chunks = raw_map.get(key, [])
        for chunk in chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                result.append(chunk)
    return result


def _child_texts(members: list[RetrievalChunk | ClusterNode]) -> list[str]:
    """Primary input texts for members (raw content or child summary)."""
    texts: list[str] = []
    for member in members:
        if isinstance(member, RetrievalChunk):
            texts.append(member.content)
        else:
            texts.append(member.summary_text)
    return texts


def _bounded_summary_input(
    child_primary_texts: list[str],
    *,
    max_chars: int,
    book_id: str,
    level: ClusterLevel,
) -> str:
    separator = "\n\n"
    original_chars = sum(len(text) for text in child_primary_texts) + len(separator) * max(
        0, len(child_primary_texts) - 1
    )
    if original_chars <= max_chars:
        return separator.join(child_primary_texts)

    per_child_chars = max(1, max_chars // len(child_primary_texts))
    primary_input = separator.join(text[:per_child_chars] for text in child_primary_texts)
    log.warning(
        "raptor_input_truncated",
        book_id=book_id,
        level=level.value,
        n_children=len(child_primary_texts),
        original_chars=original_chars,
        capped_chars=len(primary_input),
    )
    return primary_input


async def _call_llm_with_retry(
    llm_call: Callable[[str, str, int, float], Awaitable[LLMResponse]],
    *,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    node_id: str,
    level: ClusterLevel,
    book_id: str,
) -> LLMResponse:
    last_error: Exception | None = None
    for attempt, delay in enumerate(RETRY_DELAYS):
        try:
            resp = await llm_call(system, user, max_tokens, temperature)
            return resp.model_copy(update={"text": _strip_reasoning(resp.text)})
        except Exception as exc:
            last_error = exc
            log.warning(
                "raptor_llm_retry",
                node_id=node_id,
                level=level.value,
                book_id=book_id,
                attempt=attempt + 1,
                delay=delay,
                error=str(exc),
            )
            if attempt < len(RETRY_DELAYS) - 1:
                await asyncio.sleep(delay)
    log.error(
        "raptor_llm_failed",
        node_id=node_id,
        level=level.value,
        book_id=book_id,
        error=str(last_error),
    )
    raise RuntimeError(
        f"LLM call failed for node {node_id} after {len(RETRY_DELAYS)} attempts: {last_error}"
    ) from last_error


async def _maybe_load_existing(
    node_id: str,
    store: Any,
    collection: str,
) -> ClusterNode | None:
    try:
        return await get_cluster_node(node_id, store=store, collection=collection)
    except Exception:
        return None


async def _embed_and_persist_cluster_node(
    node: ClusterNode,
    embed_client: Any,
    store: Any,
    collection: str,
) -> None:
    embedding_text = f"{node.node_id}\n\n{node.summary_text}"
    emb = await embed_client.embed(
        texts=[embedding_text],
        return_dense=True,
        return_sparse=True,
        return_colbert=include_colbert_at_ingest(),
    )
    dense = emb["dense"][0]
    sparse_raw = emb["sparse"][0]
    if isinstance(sparse_raw, dict):
        sparse = {
            int(i): v for i, v in zip(sparse_raw["indices"], sparse_raw["values"], strict=True)
        }
    else:
        sparse = {int(i): v for i, v in zip(sparse_raw.indices, sparse_raw.values, strict=True)}
    colbert = emb["colbert"][0] if emb.get("colbert") else []
    if not colbert:
        colbert = fallback_colbert_vectors(dense)
    await upsert_cluster_node(
        node,
        dense_embedding=dense,
        sparse_embedding=sparse,
        colbert_embedding=colbert,
        store=store,
        collection=collection,
    )


async def _embed_cluster_node(node: ClusterNode, embed_client: Any) -> Any:
    embedding_text = f"{node.node_id}\n\n{node.summary_text}"
    emb = await embed_client.embed(
        texts=[embedding_text],
        return_dense=True,
        return_sparse=True,
        return_colbert=include_colbert_at_ingest(),  # opt-in; fallback below
    )
    dense = emb["dense"][0]
    sparse_raw = emb["sparse"][0]
    if isinstance(sparse_raw, dict):
        sparse = {
            int(i): v for i, v in zip(sparse_raw["indices"], sparse_raw["values"], strict=True)
        }
    else:
        sparse = {int(i): v for i, v in zip(sparse_raw.indices, sparse_raw.values, strict=True)}
    colbert = emb["colbert"][0] if emb.get("colbert") else []
    if not colbert:
        colbert = fallback_colbert_vectors(dense)  # schema needs ≥1 colbert vector
    return cluster_node_point(
        node,
        dense_embedding=dense,
        sparse_embedding=sparse,
        colbert_embedding=colbert,
    )


async def _build_cluster_node(
    *,
    cluster_index: int,
    members: list[RetrievalChunk | ClusterNode],
    level: ClusterLevel,
    book_id: str,
    config: RaptorBuildConfig,
    llm_call: Callable[[str, str, int, float], Awaitable[LLMResponse]],
    sem: asyncio.Semaphore,
    raw_map: dict[str, list[RetrievalChunk]],
    assignment_by_id: dict[str, Any],
) -> ClusterNode:
    member_ids = sorted([_item_id(m) for m in members])
    prompt_hash = _prompt_hash_for(level)
    node_id = cluster_node_id_for(
        book_id=book_id,
        level=level,
        member_chunk_ids=member_ids,
        prompt_hash=prompt_hash,
    )

    # Note: store/collection are not accessed here because idempotency is
    # handled by the caller (build_raptor_for_book) which checks _maybe_load_existing
    # before calling this helper.  This helper always builds a *new* node.

    raw_member_chunks = _union_raw_chunks(members, raw_map)
    child_primary_texts = _child_texts(members)
    primary_input = _bounded_summary_input(
        child_primary_texts,
        max_chars=config.max_summary_input_chars,
        book_id=book_id,
        level=level,
    )

    evidence: list[dict[str, str | float]] = []
    if raw_member_chunks:
        _, evidence = assemble_recursive_evidence(
            level=level,
            child_primary_text=primary_input,
            all_underlying_raw_chunks=raw_member_chunks,
            k_evidence=min(10, len(raw_member_chunks)),
        )

    system_prompt, user_prompt = build_level_extract_support_prompt(
        level=level,
        primary_input=primary_input,
        supporting_evidence=evidence,
        book_id=book_id,
    )

    async with sem:
        resp = await _call_llm_with_retry(
            llm_call,
            system=system_prompt,
            user=user_prompt,
            max_tokens=config.summary_max_tokens,
            temperature=config.summary_temperature,
            node_id=node_id,
            level=level,
            book_id=book_id,
        )

    summary_text, summary_meta = _parse_summary_response(resp.text)

    member_umap_coords = [
        assignment_by_id[mid].umap_coords for mid in member_ids if mid in assignment_by_id
    ]
    member_probs = [
        assignment_by_id[mid].probability for mid in member_ids if mid in assignment_by_id
    ]
    centroid = (
        [
            float(sum(coord[i] for coord in member_umap_coords) / len(member_umap_coords))
            for i in range(len(member_umap_coords[0]))
        ]
        if member_umap_coords
        else []
    )
    avg_prob = sum(member_probs) / len(member_probs) if member_probs else 1.0

    node = ClusterNode(
        node_id=node_id,
        book_id=book_id,
        level=level,
        cluster_index_within_level=cluster_index,
        parent_node_id=None,
        child_node_ids=[_item_id(m) for m in members],
        source_chunk_ids=sorted({c.chunk_id for c in raw_member_chunks}),
        direct_child_chunk_ids=member_ids,
        evidence_spans=evidence,
        cluster_centroid_umap=centroid,
        gmm_cluster_probability=avg_prob,
        summary_text=summary_text,
        summary_model=config.summary_model,
        prompt_hash=prompt_hash,
        prompt_version=f"raptor_{level.value}_v1",
        embedding_model=config.embedding_model,
        embedding_version=config.embedding_version,
        umap_random_state=config.reproducibility_config.umap_random_state,
        gmm_random_state=config.reproducibility_config.gmm_random_state,
        gmm_cluster_selection_criterion=config.reproducibility_config.gmm_cluster_selection_criterion,
        created_at=datetime.now(UTC),
        source_collection_version=config.source_collection_version,
        metadata={
            "input_token_count": resp.input_tokens,
            "output_token_count": resp.output_tokens,
            "summary_parse_fallback": summary_meta.get("summary_parse_fallback"),
        },
    )

    return node


# ---------------------------------------------------------------------------
# Per-book builder
# ---------------------------------------------------------------------------


async def build_raptor_for_book(
    *,
    book_id: str,
    chunks: list[RetrievalChunk],
    leaf_embeddings: list[list[float]],
    config: RaptorBuildConfig,
    llm_call: Callable[[str, str, int, float], Awaitable[LLMResponse]],
    embed_client: Any,
    store: Any,
    cluster_func: _ClusterFunc | None = None,
    expected_dependencies: Mapping[str, int | None] | None = None,
) -> tuple[RaptorBuildResult, RaptorBuildStats]:
    """Build one per-book RAPTOR tree and publish it as a complete revision."""
    if len(chunks) != len(leaf_embeddings):
        raise ValueError("chunks and leaf_embeddings must have the same length")
    if not getattr(
        store,
        "source_visibility_enabled",
        getattr(store, "_publication_enabled", False),
    ):
        return await _build_raptor_for_book_impl(
            book_id=book_id,
            chunks=chunks,
            leaf_embeddings=leaf_embeddings,
            config=config,
            llm_call=llm_call,
            embed_client=embed_client,
            store=store,
            cluster_func=cluster_func,
        )
    if expected_dependencies is None:
        raise ValueError(
            "publication-enabled RAPTOR builds require raw dependency revisions captured "
            "before loading chunks"
        )

    input_manifest = {
        "chunk_manifest_sha256": canonical_sha256(
            [{"chunk_id": chunk.chunk_id, "content": chunk.content} for chunk in chunks]
        ),
        "leaf_embeddings_sha256": canonical_sha256(leaf_embeddings),
    }
    binding = await admit_derived_source(
        store,
        collection=config.raptor_collection,
        source_id=book_id,
        dependency_sources=[book_id],
        expected_dependencies=dict(expected_dependencies),
        producer="raptor-tree-v1",
        input_manifest=input_manifest,
        producer_recipe={
            "summary_model": config.summary_model,
            "embedding_model": config.embedding_model,
            "embedding_version": config.embedding_version,
            "reproducibility": config.reproducibility_config.model_dump(mode="json"),
            "max_tree_levels": config.max_tree_levels,
            "summary_temperature": config.summary_temperature,
            "summary_max_tokens": config.summary_max_tokens,
            "prompt_hashes": PROMPT_HASHES,
        },
    )
    if binding is None:
        if not getattr(store, "source_publication_enabled", False):
            return await _build_raptor_for_book_impl(
                book_id=book_id,
                chunks=chunks,
                leaf_embeddings=leaf_embeddings,
                config=config,
                llm_call=llm_call,
                embed_client=embed_client,
                store=store,
                cluster_func=cluster_func,
            )
        raise RuntimeError("RAPTOR publication admission unexpectedly disabled")
    current = await binding.coordinator.read_active(binding.collection, book_id)
    durable_job = await binding.coordinator.read_job(binding.job.job_id)
    if durable_job is not None and durable_job.state.value == "published":
        if current is None or current.revision != durable_job.revision:
            raise RuntimeError("published RAPTOR build was superseded by a newer revision")
        nodes = await get_book_tree(book_id, store=store, collection=binding.collection)
        root = next((node for node in nodes if node.parent_node_id is None), None)
        result = RaptorBuildResult(book_id=book_id, nodes=nodes, root=root)
        stats = RaptorBuildStats(
            book_id=book_id,
            level_count=len({node.level for node in nodes}),
            node_count=len(nodes),
            input_tokens=0,
            output_tokens=0,
            wall_time_ms=0,
            n_skipped_clusters=len(nodes),
        )
        return result, stats

    try:
        before = await SourceRevisionWriter(store._raw_client, binding.coordinator).census(
            binding.collection, book_id, current.revision if current is not None else None
        )
        before_receipts = {str(row.id): representation_receipt(row) for row in before}
        return await _build_raptor_for_book_impl(
            book_id=book_id,
            chunks=chunks,
            leaf_embeddings=leaf_embeddings,
            config=config,
            llm_call=llm_call,
            embed_client=embed_client,
            store=store,
            cluster_func=cluster_func,
            publication_binding=binding,
            expected_before_receipts=before_receipts,
        )
    except BaseException:
        await abort_derived_source(store, binding, "RAPTOR build failed before publication")
        raise


async def _build_raptor_for_book_impl(
    *,
    book_id: str,
    chunks: list[RetrievalChunk],
    leaf_embeddings: list[list[float]],
    config: RaptorBuildConfig,
    llm_call: Callable[[str, str, int, float], Awaitable[LLMResponse]],
    embed_client: Any,
    store: Any,
    cluster_func: _ClusterFunc | None = None,
    publication_binding: DerivedSourceBinding | None = None,
    expected_before_receipts: Mapping[str, Any] | None = None,
) -> tuple[RaptorBuildResult, RaptorBuildStats]:
    """Build a RAPTOR tree for a single book.

    Level 0 = the leaf chunks with their ``leaf_embeddings``. Recurse:
    cluster the current level's embeddings; for each cluster, concatenate
    member texts and LLM-summarize into a ``ClusterNode`` (deterministic id
    via ``cluster_node_id_for``, parent/child links, level, source_chunk_ids);
    embed the summary via ``embed_client``; ``upsert_cluster_node``. The
    cluster summaries (+ their embeddings) become the next level. Stop when a
    level has a single node (root) OR ``max_tree_levels`` reached.

    Idempotency: skip a node whose deterministic id already exists.
    """
    t0 = time.monotonic()

    if len(chunks) != len(leaf_embeddings):
        raise ValueError(
            f"chunks and leaf_embeddings must be same length: "
            f"{len(chunks)} != {len(leaf_embeddings)}"
        )

    # Some corpus books carry re-ingested duplicate chunk_ids (the 3,374-chunk
    # pilot book had 163). cluster_chunks requires unique ids and the whole tree
    # keys on chunk_id, so dedupe up front (keep first occurrence + its aligned
    # embedding) instead of crashing. No-op for books with unique ids.
    if len({c.chunk_id for c in chunks}) != len(chunks):
        seen: set[str] = set()
        deduped_chunks: list[RetrievalChunk] = []
        deduped_embeddings: list[list[float]] = []
        for chunk, embedding in zip(chunks, leaf_embeddings, strict=True):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            deduped_chunks.append(chunk)
            deduped_embeddings.append(embedding)
        log.info(
            "raptor_deduped_chunk_ids",
            book_id=book_id,
            original=len(chunks),
            unique=len(deduped_chunks),
        )
        chunks = deduped_chunks
        leaf_embeddings = deduped_embeddings

    _cluster_func = cluster_func if cluster_func is not None else cluster_chunks
    sem = asyncio.Semaphore(config.max_concurrent_llm_calls)

    # Map from item id to the raw RetrievalChunks it ultimately covers.
    raw_map: dict[str, list[RetrievalChunk]] = {}
    for chunk in chunks:
        raw_map[chunk.chunk_id] = [chunk]

    current_items: list[Any] = list(chunks)
    current_embeddings = np.asarray(leaf_embeddings, dtype=np.float32)

    all_nodes: list[ClusterNode] = []
    child_to_parent: dict[str, str] = {}

    total_input_tokens = 0
    total_output_tokens = 0
    total_skipped = 0

    for level_num in range(config.max_tree_levels):
        level = _level_for_num(level_num)
        current_ids = [_item_id(item) for item in current_items]

        # UMAP/GMM need enough points: UMAP degenerates when n_samples <=
        # n_neighbors (zero-size graph.data.max()) and BIC component selection
        # needs >= 3 chunks. When the level is too small to cluster, the
        # remaining items ARE the top of the tree — emit a single all-items
        # cluster so the node-build below produces one root and the loop
        # terminates. (Real-data pilot caught the UMAP crash at a ~12-node
        # upper level; the in-loop `len(built_this_level) == 1` break then ends
        # the tree.)
        repro = config.reproducibility_config
        if _cluster_func is cluster_chunks and len(current_items) <= repro.umap_n_neighbors:
            clustering = ClusteringResult(
                assignments=[
                    ClusterAssignment(
                        chunk_id=cid, cluster_index=0, probability=1.0, umap_coords=[]
                    )
                    for cid in current_ids
                ],
                n_clusters_selected=1,
                bic_scores={},
                aic_scores={},
                umap_random_state=repro.umap_random_state,
                gmm_random_state=repro.gmm_random_state,
            )
        else:
            clustering = _cluster_func(current_ids, current_embeddings, config=repro)

        # Group members by cluster index.
        cluster_assignments: dict[int, list[Any]] = {}
        for assignment in clustering.assignments:
            idx = assignment.cluster_index
            item = next(
                (it for it in current_items if _item_id(it) == assignment.chunk_id),
                None,
            )
            if item is not None:
                cluster_assignments.setdefault(idx, []).append(item)

        assignment_by_id = {a.chunk_id: a for a in clustering.assignments}

        # Build every cluster in this level CONCURRENTLY (bounded by `sem` inside
        # _build_cluster_node). A sequential `await` loop here serializes the LLM
        # calls and leaves the endpoint's other workers idle — the cause of the
        # ~1-call-at-a-time slowness. raw_map is only READ during the build
        # (prior-level keys); new entries are written after the gather, so there
        # is no write race. level / assignment_by_id are passed as args (not
        # closed over) so the per-level coroutine is loop-variable-safe.
        async def _process_cluster(
            idx: int,
            members: list[Any],
            lvl: ClusterLevel,
            abi: dict[str, ClusterAssignment],
        ) -> tuple[ClusterNode, list[Any], bool]:
            node_id = cluster_node_id_for(
                book_id=book_id,
                level=lvl,
                member_chunk_ids=sorted([_item_id(m) for m in members]),
                prompt_hash=_prompt_hash_for(lvl),
            )
            existing = await _maybe_load_existing(node_id, store, config.raptor_collection)
            if existing is not None:
                log.info("raptor_node_skipped_existing", node_id=node_id, level=lvl.value)
                return existing, members, True
            node = await _build_cluster_node(
                cluster_index=idx,
                members=members,
                level=lvl,
                book_id=book_id,
                config=config,
                llm_call=llm_call,
                sem=sem,
                raw_map=raw_map,
                assignment_by_id=abi,
            )
            return node, members, False

        cluster_jobs = list(cluster_assignments.items())
        cluster_results = await asyncio.gather(
            *(
                _process_cluster(idx, members, level, assignment_by_id)
                for idx, members in cluster_jobs
            ),
            return_exceptions=True,
        )

        built_this_level: list[ClusterNode] = []
        for (idx, _members), cluster_result in zip(cluster_jobs, cluster_results, strict=True):
            if isinstance(cluster_result, BaseException):
                log.warning(
                    "raptor_cluster_skipped",
                    book_id=book_id,
                    level=level.value,
                    cluster_index=idx,
                    error=str(cluster_result),
                )
                continue

            built_node, built_members, was_existing = cluster_result
            if not was_existing:
                total_input_tokens += built_node.metadata.get("input_token_count", 0)
                total_output_tokens += built_node.metadata.get("output_token_count", 0)
            built_this_level.append(built_node)
            raw_map[built_node.node_id] = _union_raw_chunks(built_members, raw_map)

        n_total = len(cluster_jobs)
        n_skipped = n_total - len(built_this_level)
        if len(built_this_level) == 0:
            raise RuntimeError(
                f"all RAPTOR clusters were skipped for book_id={book_id} "
                f"level={level.value} ({n_skipped}/{n_total}); refusing to continue"
            )
        skip_fraction = n_skipped / n_total
        if skip_fraction > config.max_skip_fraction:
            raise RuntimeError(
                f"RAPTOR cluster skip fraction {skip_fraction:.3f} exceeded "
                f"max_skip_fraction {config.max_skip_fraction:.3f} for "
                f"book_id={book_id} level={level.value} ({n_skipped}/{n_total})"
            )
        total_skipped += n_skipped

        # Update parent mapping and raw_map for next level.
        item_by_id = {_item_id(it): it for it in current_items}
        for node in built_this_level:
            for child_id in node.direct_child_chunk_ids:
                child_to_parent[child_id] = node.node_id
            child_items = [
                item_by_id[cid] for cid in node.direct_child_chunk_ids if cid in item_by_id
            ]
            raw_map[node.node_id] = _union_raw_chunks(child_items, raw_map)

        all_nodes.extend(built_this_level)

        if len(built_this_level) == 1:
            break

        # Embed summaries for next level clustering.
        texts = [n.summary_text for n in built_this_level]
        emb = await embed_client.embed(
            texts=texts,
            return_dense=True,
            return_sparse=False,
            return_colbert=False,
        )
        current_embeddings = np.asarray(emb["dense"], dtype=np.float32)
        current_items = built_this_level

    # Set parent_node_id on all nodes.
    for node in all_nodes:
        node.parent_node_id = child_to_parent.get(node.node_id)

    # Publication mode stages the complete per-book representation before a
    # single pointer switch. Legacy mode retains its original per-node writer.
    if publication_binding is None:
        for node in all_nodes:
            await _embed_and_persist_cluster_node(
                node, embed_client, store, config.raptor_collection
            )
    else:
        points = [await _embed_cluster_node(node, embed_client) for node in all_nodes]
        await publish_derived_source(
            store,
            publication_binding,
            points,
            expected_before_receipts=expected_before_receipts,
        )

    root = next(
        (n for n in all_nodes if n.node_id not in child_to_parent),
        None,
    )

    wall_time_ms = int((time.monotonic() - t0) * 1000)
    stats = RaptorBuildStats(
        book_id=book_id,
        level_count=len({n.level for n in all_nodes}),
        node_count=len(all_nodes),
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        wall_time_ms=wall_time_ms,
        n_skipped_clusters=total_skipped,
    )

    log.info(
        "raptor.build_book.complete",
        book_id=book_id,
        node_count=len(all_nodes),
        level_count=stats.level_count,
        n_skipped_clusters=total_skipped,
        wall_time_ms=wall_time_ms,
        stage="build_raptor_for_book",
    )
    return RaptorBuildResult(book_id=book_id, nodes=all_nodes, root=root), stats


# ---------------------------------------------------------------------------
# Corpus-level builder
# ---------------------------------------------------------------------------


async def build_raptor_for_corpus(
    *,
    books: list[tuple[str, list[RetrievalChunk], list[list[float]]]],
    config: RaptorBuildConfig,
    llm_call: Callable[[str, str, int, float], Awaitable[LLMResponse]],
    embed_client: Any,
    store: Any,
    max_concurrent_books: int = 4,
    cluster_func: _ClusterFunc | None = None,
    expected_dependencies: Mapping[str, int | None] | None = None,
) -> tuple[list[RaptorBuildResult], list[RaptorBuildStats]]:
    """Build RAPTOR trees for a corpus of books with bounded concurrency.

    Each element of ``books`` is ``(book_id, chunks, leaf_embeddings)``.

    Processes up to ``max_concurrent_books`` books in parallel via an
    ``asyncio.Semaphore``.

    Returns ``(list_of_results, list_of_per_book_stats)``.
    """
    await ensure_raptor_collection(
        store._client,
        collection=config.raptor_collection,
    )

    semaphore = asyncio.Semaphore(max_concurrent_books)
    all_results: list[RaptorBuildResult] = []
    all_stats: list[RaptorBuildStats] = []

    async def _process(
        book_id: str,
        chunks: list[RetrievalChunk],
        leaf_embeddings: list[list[float]],
    ) -> None:
        async with semaphore:
            result, stats = await build_raptor_for_book(
                book_id=book_id,
                chunks=chunks,
                leaf_embeddings=leaf_embeddings,
                config=config,
                llm_call=llm_call,
                embed_client=embed_client,
                store=store,
                cluster_func=cluster_func,
                expected_dependencies=(
                    {book_id: expected_dependencies[book_id]}
                    if expected_dependencies is not None and book_id in expected_dependencies
                    else None
                ),
            )
            all_results.append(result)
            all_stats.append(stats)

    await asyncio.gather(
        *(_process(book_id, chunks, embeddings) for book_id, chunks, embeddings in books)
    )

    log.info(
        "raptor.build_corpus.complete",
        book_count=len(books),
        stage="build_raptor_for_corpus",
    )
    return all_results, all_stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "RaptorBuildConfig",
    "RaptorBuildResult",
    "RaptorBuildStats",
    "build_raptor_for_book",
    "build_raptor_for_corpus",
]

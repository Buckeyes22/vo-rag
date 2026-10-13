"""DSI build pipeline orchestrator.

Per-book extraction + validation + persistence pipeline. Follows the shape of
``vo_rag.core.summary.builder`` and ``vo_rag.core.graph.builder``.

Routing is query-time (``vo_rag.core.dsi.routing.allocate_budgets``); the builder
produces the per-book ``FacetedBookRecord`` artifacts that routing consults.

Public API
----------
DsiBuildConfig
    Wire-config for a DSI build run.
DsiBuildStats
    Per-book outcome record emitted by every build call.
build_dsi_for_book(...)
    Extract -> validate -> decide for one book; returns ``(record|None, stats)``.
build_dsi_for_corpus(...)
    Semaphore-bounded corpus-level driver; returns ``(records, stats)``.
persist_dsi_records(...)
    Embed + upsert a list of accepted ``FacetedBookRecord`` objects.
dsi_search_text(...)
    Assemble the BGE-M3 search text for one record (abstract + every facet).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal, cast

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    Modifier,
    MultiVectorComparator,
    MultiVectorConfig,
    PayloadSchemaType,
    SparseVectorParams,
    VectorParams,
)

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.dsi.extraction import ExtractionConfig, ExtractionResult, extract_book
from vo_rag.core.dsi.schemas import FacetedBookRecord
from vo_rag.core.dsi.validation import (
    ProvenanceViolation,
    decide_rejection,
    validate_provenance_offline,
)
from vo_rag.core.embedding.point_vectors import (
    downsample_colbert_vectors,
    fallback_colbert_vectors,
    sparse_payload_to_int_map,
)
from vo_rag.core.idempotency import deterministic_uuid
from vo_rag.core.ingest.source_publication import SourcePublicationJob
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.models.common import VoBaseModel
from vo_rag.core.retrieval.bm25_tokenizer import tokenize, tokens_to_bm25_sparse
from vo_rag.core.storage.points import VoPoint

log = structlog.get_logger(__name__)

_DSI_COLLECTION_DEFAULT = "vo_rag_books_dsi_v1"
_CANONICAL_BOOK_ID = re.compile(r"^vo-rag-corpus-sha256:([0-9a-f]{64})$")


# ---------------------------------------------------------------------------
# Config + stats
# ---------------------------------------------------------------------------


class DsiBuildConfig(VoBaseModel):
    """Wire-configuration for a DSI build run."""

    extraction_config: ExtractionConfig
    max_concurrent_books: int = 4
    max_validation_attempts: int = 2


class DsiBuildStats(VoBaseModel):
    """Per-book outcome record emitted after ``build_dsi_for_book`` completes."""

    book_id: str
    extraction_attempts: int
    extraction_succeeded: bool
    validation_passed: bool
    validation_violations: int
    final_action: Literal["accept", "re_extract", "manual_review", "extraction_failed"]
    input_tokens: int
    output_tokens: int
    wall_time_ms: int


# ---------------------------------------------------------------------------
# Per-book builder
# ---------------------------------------------------------------------------


async def build_dsi_for_book(
    *,
    book_id: str,
    title: str,
    author: str,
    category_ids: list[str],
    chunks: list[RetrievalChunk],
    config: DsiBuildConfig,
    llm_call: Callable[[str, str, int, float], Awaitable[LLMResponse]],
    source_collection_version: str,
    known_chunk_ids: set[str] | None = None,
    known_section_paths: set[str] | None = None,
) -> tuple[FacetedBookRecord | None, DsiBuildStats]:
    """Extract, validate, and decide for a single book.

    Loop up to ``config.max_validation_attempts`` times:

    1. Extract via ``extract_book`` -> ``ExtractionResult``
    2. If extraction failed (record is None): log, increment attempt, retry.
    3. If extraction succeeded:
       - Optionally run ``validate_provenance_offline`` when
         ``known_chunk_ids`` is provided.
       - Call ``decide_rejection`` to produce an accept / re_extract /
         manual_review verdict.
       - ``accept`` -> return ``(record, stats)``.
       - ``re_extract`` -> continue to next attempt.
       - ``manual_review`` -> return ``(None, stats)``.

    Returns ``(None, stats)`` when extraction fails on all attempts or the
    record is flagged for manual review.
    """
    t0 = time.monotonic()

    attempt = 0
    extraction_attempts = 0
    total_input_tokens = 0
    total_output_tokens = 0

    last_result: ExtractionResult | None = None
    last_violations: list[ProvenanceViolation] = []
    final_action: Literal["accept", "re_extract", "manual_review", "extraction_failed"] = (
        "extraction_failed"
    )

    while attempt < config.max_validation_attempts:
        attempt += 1
        extraction_attempts += 1

        result = await extract_book(
            book_id=book_id,
            title=title,
            author=author,
            category_ids=category_ids,
            chunks=chunks,
            config=config.extraction_config,
            llm_call=llm_call,
            source_collection_version=source_collection_version,
        )

        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens
        last_result = result

        if result.record is None:
            log.warning(
                "dsi.build_book.extraction_failed",
                book_id=book_id,
                attempt=attempt,
                validation_errors=result.validation_errors,
                stage="build_dsi_for_book",
            )
            final_action = "extraction_failed"
            continue

        # Extraction succeeded — run provenance validation if sets are provided.
        violations: list[ProvenanceViolation] = []
        if known_chunk_ids is not None:
            violations = validate_provenance_offline(
                result.record,
                known_chunk_ids=known_chunk_ids,
                known_section_paths=known_section_paths or set(),
            )

        last_violations = violations
        decision = decide_rejection(
            violations,
            attempt=attempt,
            max_attempts=config.max_validation_attempts,
        )

        log.info(
            "dsi.build_book.decision",
            book_id=book_id,
            attempt=attempt,
            action=decision.action,
            violation_count=len(violations),
            stage="build_dsi_for_book",
        )

        if decision.action == "accept":
            final_action = "accept"
            wall_time_ms = int((time.monotonic() - t0) * 1000)
            stats = DsiBuildStats(
                book_id=book_id,
                extraction_attempts=extraction_attempts,
                extraction_succeeded=True,
                validation_passed=True,
                validation_violations=len(violations),
                final_action=final_action,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                wall_time_ms=wall_time_ms,
            )
            log.info(
                "dsi.build_book.complete",
                book_id=book_id,
                final_action=final_action,
                wall_time_ms=wall_time_ms,
                stage="build_dsi_for_book",
            )
            return result.record, stats

        if decision.action == "re_extract":
            final_action = "re_extract"
            log.info(
                "dsi.build_book.re_extract",
                book_id=book_id,
                attempt=attempt,
                violation_count=len(violations),
                stage="build_dsi_for_book",
            )
            continue

        # manual_review
        final_action = "manual_review"
        break

    # Loop exhausted — build terminal stats.
    extraction_succeeded = last_result is not None and last_result.record is not None
    validation_passed = extraction_succeeded and len(last_violations) == 0

    wall_time_ms = int((time.monotonic() - t0) * 1000)
    stats = DsiBuildStats(
        book_id=book_id,
        extraction_attempts=extraction_attempts,
        extraction_succeeded=extraction_succeeded,
        validation_passed=validation_passed,
        validation_violations=len(last_violations),
        final_action=final_action,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        wall_time_ms=wall_time_ms,
    )

    if len(chunks) > 40 and not extraction_succeeded:
        reason = last_result.validation_errors if last_result else ["all attempts failed"]
        log.warning(
            "dsi.zero_records_on_substantial_book",
            book_id=book_id,
            n_chunks=len(chunks),
            reason=reason[:3],
            stage="build_dsi_for_book",
        )
        # TODO(VOR-239): Consider a single retry with the same prompt when
        # zero records are produced for a substantial book and all attempts
        # have been exhausted.

    log.info(
        "dsi.build_book.complete",
        book_id=book_id,
        final_action=final_action,
        wall_time_ms=wall_time_ms,
        stage="build_dsi_for_book",
    )
    return None, stats


# ---------------------------------------------------------------------------
# Corpus-level builder
# ---------------------------------------------------------------------------


async def build_dsi_for_corpus(
    *,
    books: list[tuple[str, str, str, list[str], list[RetrievalChunk]]],
    config: DsiBuildConfig,
    llm_call: Callable[[str, str, int, float], Awaitable[LLMResponse]],
    source_collection_version: str,
    known_chunk_ids_per_book: dict[str, set[str]] | None = None,
    known_section_paths_per_book: dict[str, set[str]] | None = None,
) -> tuple[list[FacetedBookRecord], list[DsiBuildStats]]:
    """Build DSI records for a corpus of books with bounded concurrency.

    Each element of ``books`` is ``(book_id, title, author, category_ids, chunks)``.

    Processes up to ``config.max_concurrent_books`` books in parallel via an
    ``asyncio.Semaphore``.

    Returns ``(list_of_accepted_records, list_of_per_book_stats)``.
    """
    semaphore = asyncio.Semaphore(config.max_concurrent_books)
    all_records: list[FacetedBookRecord] = []
    all_stats: list[DsiBuildStats] = []

    async def process_book(
        book_id: str,
        title: str,
        author: str,
        category_ids: list[str],
        chunks: list[RetrievalChunk],
    ) -> None:
        async with semaphore:
            record, stats = await build_dsi_for_book(
                book_id=book_id,
                title=title,
                author=author,
                category_ids=category_ids,
                chunks=chunks,
                config=config,
                llm_call=llm_call,
                source_collection_version=source_collection_version,
                known_chunk_ids=(
                    known_chunk_ids_per_book.get(book_id) if known_chunk_ids_per_book else None
                ),
                known_section_paths=(
                    known_section_paths_per_book.get(book_id)
                    if known_section_paths_per_book
                    else None
                ),
            )
            if record is not None:
                all_records.append(record)
            all_stats.append(stats)

    await asyncio.gather(
        *(
            process_book(book_id, title, author, category_ids, chunks)
            for book_id, title, author, category_ids, chunks in books
        )
    )

    log.info(
        "dsi.build_corpus.complete",
        book_count=len(books),
        accepted_count=len(all_records),
        stage="build_dsi_for_corpus",
    )
    return all_records, all_stats


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _dsi_source_sha256(source_id: str) -> str:
    """Return the verified file digest encoded by a canonical book identity."""
    match = _CANONICAL_BOOK_ID.fullmatch(source_id)
    if match is None:
        raise ValueError("published DSI requires a canonical SHA-256 book_id")
    return match.group(1)


def _dsi_producer_recipe(
    config: DsiBuildConfig, source_collection_version: str
) -> dict[str, str | int | float]:
    extraction = config.extraction_config
    return {
        "producer": "dsi-faceted-book-v1",
        "summary_model": extraction.summary_model,
        "prompt_hash": extraction.prompt_hash,
        "temperature": extraction.temperature,
        "max_tokens": extraction.max_tokens,
        "representative_chunk_count": extraction.n_representative_chunks,
        "max_validation_attempts": config.max_validation_attempts,
        "source_collection_version": source_collection_version,
        "search_text": "dsi-search-text-v1",
        "embedding_model": "BAAI/bge-m3",
    }


async def admit_dsi_source_revision(
    *,
    store: Any,
    source_id: str,
    collection: str,
    config: DsiBuildConfig,
    source_collection_version: str,
    expected_dependencies: Mapping[str, int | None],
) -> SourcePublicationJob | None:
    """Admit derived DSI work against its captured raw-book revision.

    Call this before the LLM producer runs. The content-addressed source ID is
    the evidence for the original input file digest; no hash is synthesized
    from the derived record or its vectors.
    """
    if not getattr(
        store,
        "source_visibility_enabled",
        getattr(store, "source_publication_enabled", False),
    ):
        return None
    if dict(expected_dependencies) != {source_id: expected_dependencies.get(source_id)}:
        raise ValueError("DSI requires exactly one captured raw-book dependency")
    source_sha256 = _dsi_source_sha256(source_id)
    recipe = _dsi_producer_recipe(config, source_collection_version)
    dependency_json = json.dumps(dict(expected_dependencies), sort_keys=True, separators=(",", ":"))
    recipe_json = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
    job_id = deterministic_uuid(
        "dsi-source-publication-v1",
        collection,
        source_id,
        source_sha256,
        dependency_json,
        recipe_json,
    )
    job = cast(
        SourcePublicationJob | None,
        await store.admit_source_revision(
            collection,
            source_id,
            job_id,
            source_sha256,
            expected_dependencies=expected_dependencies,
            producer_recipe=recipe,
        ),
    )
    if job is None and not getattr(store, "source_publication_enabled", False):
        # Reader-first mode still performs QdrantStore's source/dependency
        # fence admission, then writes through the guarded legacy-ID path.
        return None
    if job is None:
        raise RuntimeError("DSI publication admission unexpectedly returned no job")
    return job


def dsi_search_text(record: FacetedBookRecord) -> str:
    """Assemble the BGE-M3 search text for one DSI record (abstract + every
    facet value). One source of truth for both persistence and any re-embed."""
    parts: list[str] = [record.short_abstract]
    parts.extend(record.technical_domains)
    parts.extend(entry.value for entry in record.key_concepts)
    parts.extend(entry.value for entry in record.key_methods)
    parts.extend(entry.value for entry in record.key_equations_or_models)
    parts.extend(entry.value for entry in record.figure_heavy_topics)
    parts.extend(entry.value for entry in record.synonyms_and_aliases)
    return " ".join(p for p in parts if p)


_DSI_DENSE_DIM = 1024
_DSI_COLBERT_DIM = 1024


def _dsi_vector_config() -> dict[str, VectorParams]:
    return {
        "dense": VectorParams(size=_DSI_DENSE_DIM, distance=Distance.COSINE),
        "colbert": VectorParams(
            size=_DSI_COLBERT_DIM,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
            hnsw_config=HnswConfigDiff(m=0),
        ),
    }


async def ensure_dsi_collection(
    client: AsyncQdrantClient,
    *,
    collection: str = _DSI_COLLECTION_DEFAULT,
) -> bool:
    """Idempotent DSI collection setup. Returns True if newly created.

    Same vector config as ``vo_rag_books_raw_v1`` / the summary tree (BGE-M3
    dense 1024 + sparse + bm25 + ColBERT) — matching the named vectors
    ``persist_dsi_records`` writes per ``VoPoint``. Indexes ``book_id`` (keyword).
    The DSI layer shipped without this (it was never run end-to-end), so the
    build path 404'd on upsert until the collection was created here.
    """
    existing = {c.name for c in (await client.get_collections()).collections}
    if collection in existing:
        return False

    await client.create_collection(
        collection_name=collection,
        vectors_config=_dsi_vector_config(),
        sparse_vectors_config={
            "sparse": SparseVectorParams(modifier=Modifier.IDF),
            "bm25": SparseVectorParams(modifier=Modifier.IDF),
        },
    )
    await client.create_payload_index(
        collection_name=collection,
        field_name="book_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    return True


async def persist_dsi_records(
    *,
    records: list[FacetedBookRecord],
    embed: Callable[[list[str]], Awaitable[dict[str, Any]]],
    store: Any,
    collection: str = _DSI_COLLECTION_DEFAULT,
    embed_batch: int = 16,
    publication_job: SourcePublicationJob | None = None,
    expected_dependencies: Mapping[str, int | None] | None = None,
) -> int:
    """Embed and upsert accepted ``FacetedBookRecord`` objects into Qdrant.

    One point per book — ``id = deterministic_point_id(
        book_id, 0, f"dsi_{record.summary_model}", record.prompt_hash
    )``,
    payload = the record. Vectors mirror the book_rag raw-ingestion path exactly
    (shared ``vo_rag.core.embedding.point_vectors`` helpers): dense + sparse from
    BGE-M3, bm25 from the local tokenizer, ColBERT downsampled with a dense
    fallback.

    ``embed`` is an injected callable returning the BGE-M3 server response
    (``{"dense": [...], "sparse": [...], "colbert": [...]}``) for a batch of
    texts — e.g. ``lambda ts: client.embed(ts, return_dense=True,
    return_sparse=True, return_colbert=True)``. ``store`` is a ``QdrantStore``;
    vectors go through its real ``upsert_vectors(collection, list[VoPoint])``.

    Returns the count of records persisted.
    """
    if not records:
        return 0

    source_ids = {record.book_id for record in records}
    publication_enabled = bool(getattr(store, "source_publication_enabled", False))
    if publication_enabled:
        if len(source_ids) != 1:
            raise ValueError("published DSI persistence requires one complete book source")
        source_id = next(iter(source_ids))
        _dsi_source_sha256(source_id)
        if publication_job is None:
            raise ValueError("published DSI persistence requires admission before inference")
        dependencies = dict(expected_dependencies or {})
        if set(dependencies) != {source_id}:
            raise ValueError("published DSI persistence requires its captured raw-book revision")
        producer_recipe = publication_job.recipe.get("producer_recipe", {})
        if any(
            producer_recipe.get("summary_model") != record.summary_model
            or producer_recipe.get("prompt_hash") != record.prompt_hash
            or producer_recipe.get("source_collection_version") != record.source_collection_version
            for record in records
        ):
            raise ValueError("DSI output differs from its admitted producer recipe")
    else:
        dependencies = {}

    texts = [dsi_search_text(record) for record in records]
    points: list[VoPoint] = []
    for start in range(0, len(records), embed_batch):
        batch_records = records[start : start + embed_batch]
        batch_texts = texts[start : start + embed_batch]
        resp = await embed(batch_texts)
        dense = resp.get("dense") or []
        sparse = resp.get("sparse") or []
        colbert = resp.get("colbert") or []
        for i, record in enumerate(batch_records):
            dense_i = [float(v) for v in dense[i]]
            colbert_i = downsample_colbert_vectors(colbert[i] if i < len(colbert) else [])
            if not colbert_i:
                colbert_i = fallback_colbert_vectors(dense_i)
            points.append(
                VoPoint(
                    source_id=record.book_id,
                    chunk_index=0,
                    stage=f"dsi_{record.summary_model}",
                    version_hash=record.prompt_hash,
                    dense=dense_i,
                    sparse=sparse_payload_to_int_map(sparse[i]),
                    bm25=tokens_to_bm25_sparse(tokenize(batch_texts[i])),
                    colbert=colbert_i,
                    payload={
                        **record.model_dump(mode="json"),
                        **({"dependency_revisions": dependencies} if publication_enabled else {}),
                    },
                )
            )

    try:
        if publication_enabled:
            await store.upsert_vectors(
                collection,
                points,
                publication_job=publication_job,
                expected_dependencies=dependencies,
            )
        else:
            await store.upsert_vectors(collection, points)
    except Exception:
        if publication_enabled:
            await store.abort_source_revision(publication_job, "DSI persistence failed")
        raise

    log.info(
        "dsi.persist.complete",
        collection=collection,
        count=len(points),
        stage="persist_dsi_records",
    )
    return len(points)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DsiBuildConfig",
    "DsiBuildStats",
    "admit_dsi_source_revision",
    "build_dsi_for_book",
    "build_dsi_for_corpus",
    "dsi_search_text",
    "persist_dsi_records",
]

"""book_rag chunk + embed + upsert step module (VOR-7).

Exports three step functions:
- ``do_build_chunks`` — HierarchicalChunker over ParsedDocument (+ fallback).
- ``do_embed_raw`` — BGE-M3 embed with ColBERT-192 downsampling / fallback.
- ``do_upsert_raw`` — VoPoint assembly + Qdrant upsert to ``vo_rag_books_raw_v1``.

All three are designed to be wired into the book_rag DAG (no Hatchet
dependency).  External dependencies are injected (EmbeddingClient, QdrantStore)
so the unit test path uses fakes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Callable, Coroutine, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Final

import structlog

from vo_rag.core.chunking.composite_emitter import CompositeChunk
from vo_rag.core.chunking.hierarchical import HierarchicalChunker
from vo_rag.core.chunking.models import (
    PAYLOAD_EMBEDDING_TEXT_KEY,
    DropReason,
    PageDrop,
    RetrievalChunk,
    figure_payload_embedding_text,
)
from vo_rag.core.chunking.semantic_drift import SemanticDriftSplitter
from vo_rag.core.chunking.token_ceiling import TokenCeilingBackstop
from vo_rag.core.config.settings import Settings, get_settings
from vo_rag.core.embedding.client import DEFAULT_EMBED_MAX_LENGTH, MAX_EMBED_BATCH_TEXTS
from vo_rag.core.embedding.point_vectors import (
    downsample_colbert_vectors,
    fallback_colbert_vectors,
    include_colbert_at_ingest,
    max_colbert_vectors_per_point,
    sparse_payload_to_int_map,
)
from vo_rag.core.idempotency import deterministic_uuid
from vo_rag.core.ingest.source_publication import SourcePublicationJob
from vo_rag.core.parsing.contract import ElementType, ParsedDocument
from vo_rag.core.parsing.text_normalization import normalize_extracted_text
from vo_rag.core.retrieval.bm25_tokenizer import tokenize, tokens_to_bm25_sparse
from vo_rag.core.storage.points import VoPoint
from vo_rag.core.storage.qdrant import deterministic_point_id
from vo_rag.pipelines.book_rag.boilerplate import chunk_without_boilerplate, is_boilerplate
from vo_rag.pipelines.book_rag.domain_tagger import heuristic_domain_tags
from vo_rag.pipelines.book_rag.ocr_quality import assess
from vo_rag.pipelines.book_rag.quality_warnings import quality_warning
from vo_rag.pipelines.book_rag.validator import (
    chunk_output_stats_from_chunks,
    is_degenerate_chunk_output,
    is_runaway_chunk_output,
)

_COLLECTION_RAW = "vo_rag_books_raw_v1"
_EMBEDDING_INPUT_RECIPE = "book_rag.embedding_input.v1"
_SPARSE_NORMALIZATION_RECIPE = "bge-m3.sparse-int-map.v1"
_RUNAWAY_REPACK_MAX_CHUNKS = 5_000
_RUNAWAY_REPACK_MAX_TEXT_CHARS = 2_000_000

log = structlog.get_logger(__name__)

#: Chunk types the OCR gate never judges: a figure composite's content is a
#: caption plus neighbour text, and its value is the image.
_OCR_GATE_EXEMPT_TYPES = frozenset({"figure_composite"})


def _ocr_gate(chunk: RetrievalChunk) -> RetrievalChunk | None:
    """Apply the per-chunk OCR quality gate (VOR-1452): ``None`` for a chunk
    judged unmistakable noise, the chunk unchanged otherwise."""
    if chunk.chunk_type in _OCR_GATE_EXEMPT_TYPES:
        return chunk
    return None if assess(chunk.content or "") == "drop" else chunk


def _embedding_recipe(max_length: int, request_colbert: bool) -> dict[str, Any]:
    """Describe request modes and deterministic vector transformations."""
    return {
        "input_recipe": _EMBEDDING_INPUT_RECIPE,
        "max_length": max_length,
        "return_dense": True,
        "return_sparse": True,
        "return_colbert": request_colbert,
        "sparse_normalization": _SPARSE_NORMALIZATION_RECIPE,
        "colbert_max_vectors": max_colbert_vectors_per_point(),
        "colbert_empty_fallback": os.environ.get("VORAG_COLBERT_EMPTY_FALLBACK", "dense")
        .strip()
        .lower(),
    }


def _warn_chunk(chunk: RetrievalChunk, code: str, details: str) -> None:
    metadata = dict(chunk.metadata or {})
    existing = metadata.get("quality_warnings", [])
    warnings = list(existing) if isinstance(existing, list) else []
    warning = quality_warning("embedding", code, details)
    if warning not in warnings:
        warnings.append(warning)
    metadata["quality_warnings"] = warnings
    chunk.metadata = metadata


def _warn_chunks(chunks: list[RetrievalChunk], code: str, details: str) -> None:
    """Attach one source-free chunking warning to every passage in a book."""
    warning = quality_warning("chunking", code, details)
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        existing = metadata.get("quality_warnings", [])
        warnings = list(existing) if isinstance(existing, list) else []
        if warning not in warnings:
            warnings.append(warning)
        metadata["quality_warnings"] = warnings
        chunk.metadata = metadata


def _run_embed_coro_sync(coro: Coroutine[Any, Any, dict[str, object]]) -> dict[str, object]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    def run_in_thread() -> dict[str, object]:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="semantic-chunk-embed") as executor:
        return executor.submit(run_in_thread).result()


async def _embed_dense_async(embedding_client: Any, texts: list[str]) -> dict[str, object]:
    embed = getattr(embedding_client, "embed", None)
    if not callable(embed):
        raise RuntimeError("semantic chunking requires the async embed bridge; see VOR-305")

    response: object = await embed(
        texts,
        return_dense=True,
        return_sparse=False,
        return_colbert=False,
    )
    if not isinstance(response, dict):
        raise RuntimeError("semantic chunking requires the async embed bridge; see VOR-305")

    normalized: dict[str, object] = {}
    for key, value in response.items():
        if isinstance(key, str):
            normalized[key] = value
    return normalized


def _dense_embed_fn(embedding_client: Any) -> Callable[[list[str]], list[Sequence[float]]]:
    def embed_texts(texts: list[str]) -> list[Sequence[float]]:
        response = _run_embed_coro_sync(_embed_dense_async(embedding_client, texts))
        dense = response.get("dense")
        if not isinstance(dense, list):
            raise RuntimeError("semantic chunking requires dense embeddings; see VOR-305")

        vectors: list[Sequence[float]] = []
        for row in dense:
            if not isinstance(row, Sequence) or isinstance(row, str | bytes):
                raise RuntimeError("semantic chunking requires dense embeddings; see VOR-305")
            vector: list[float] = []
            for value in row:
                if not isinstance(value, int | float):
                    raise RuntimeError("semantic chunking requires dense embeddings; see VOR-305")
                vector.append(float(value))
            vectors.append(vector)
        return vectors

    return embed_texts


def _build_chunker(settings: Settings, embedding_client: Any | None) -> HierarchicalChunker:
    chunker = HierarchicalChunker()
    if settings.chunking.strategy != "semantic":
        return chunker

    if embedding_client is None:
        raise RuntimeError("semantic chunking requires the async embed bridge; see VOR-305")

    splitter = SemanticDriftSplitter(
        embed_fn=_dense_embed_fn(embedding_client),
        counter=chunker.count,
        max_tokens=chunker.packer.hard_token_ceiling(),
        breakpoint_percentile=settings.chunking.semantic_breakpoint_percentile,
        buffer_size=settings.chunking.semantic_buffer_size,
    )
    return HierarchicalChunker(counter=chunker.count, prose_splitter=splitter)


def _page_range(values: list[int]) -> str | None:
    if not values:
        return None
    lo = min(values)
    hi = max(values)
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _fallback_text_chunks(
    parsed: ParsedDocument,
    *,
    counter: Any,
    drop_counts: dict[str, int] | None = None,
    dropped: list[RetrievalChunk] | None = None,
    target_tokens: int | None = None,
    max_tokens: int | None = None,
    max_chunks: int | None = None,
    max_source_chars: int | None = None,
) -> list[RetrievalChunk]:
    """Recover low-yield form-style documents that parse but over-filter to zero chunks.

    Every recovered chunk passes the OCR quality gate (these are exactly the
    scanned forms it exists for); drops are counted into *drop_counts* under
    ``"ocr_garbage"`` and, when given, appended to *dropped*.
    """
    eligible = {
        ElementType.TITLE,
        ElementType.NARRATIVE_TEXT,
        ElementType.LIST_ITEM,
        ElementType.TABLE,
        ElementType.FIGURE_CAPTION,
        ElementType.EQUATION,
        ElementType.CODE_BLOCK,
    }
    target_tokens = (
        target_tokens
        if target_tokens is not None
        else int(os.environ.get("VORAG_FALLBACK_TEXT_CHUNK_TARGET", "350"))
    )
    max_tokens = (
        max_tokens
        if max_tokens is not None
        else int(os.environ.get("VORAG_FALLBACK_TEXT_CHUNK_MAX", "500"))
    )
    section_title = parsed.title or parsed.document_id
    section_path = section_title
    ceiling = TokenCeilingBackstop(counter=counter, max_tokens=max_tokens, overlap=0)
    chunks: list[RetrievalChunk] = []
    parts: list[str] = []
    pages: list[int] = []
    accepted_chars = 0

    def flush() -> None:
        if not parts:
            return
        content = normalize_extracted_text("\n\n".join(parts)).strip()
        parts.clear()
        chunk_pages = list(pages)
        pages.clear()
        if not content or not any(ch.isalpha() for ch in content):
            return
        if max_chunks is not None and len(chunks) >= max_chunks:
            raise RuntimeError(
                f"bounded chunk fallback exceeded its {max_chunks}-chunk resource limit"
            )
        embedding_text = f"{section_path}\n\n{content}"
        index = len(chunks)
        candidate = RetrievalChunk(
            chunk_id=deterministic_uuid(
                parsed.source_id,
                parsed.document_id,
                "fallback_text",
                str(index),
            ),
            chunk_type="text",
            document_id=parsed.document_id,
            source_id=parsed.source_id,
            page_number=chunk_pages[0] if chunk_pages else None,
            page_range=_page_range(chunk_pages),
            section_path=section_path,
            section_title=section_title,
            chapter_path=section_path,
            content=content,
            embedding_text=embedding_text,
            token_count=counter(embedding_text),
            metadata={
                "chunking_fallback": "parsed_text_zero_chunk_recovery",
                "fallback_chunk_index": index,
            },
        )
        chunk = _ocr_gate(candidate)
        if chunk is None:
            if dropped is not None:
                dropped.append(candidate)
            if drop_counts is not None:
                drop_counts["ocr_garbage"] = drop_counts.get("ocr_garbage", 0) + 1
            return
        chunks.append(chunk)

    for element in parsed.elements:
        if element.element_type not in eligible:
            continue
        if element.page_number in parsed.quarantined_pages:
            continue
        text = normalize_extracted_text(element.text).strip()
        if not text or not any(ch.isalpha() for ch in text):
            continue
        accepted_chars += len(text)
        if max_source_chars is not None and accepted_chars > max_source_chars:
            raise RuntimeError(
                f"bounded chunk fallback exceeded its {max_source_chars}-character source limit"
            )
        # Split individual oversized source elements too; otherwise a single
        # long OCR/table element can bypass the bounded repack target.
        for piece in (
            ceiling.split_table(text, section_path)
            if element.element_type == ElementType.TABLE
            else ceiling.split_text(text, section_path)
        ):
            candidate = "\n\n".join([*parts, piece])
            if parts and counter(candidate) > max_tokens:
                flush()
            parts.append(piece)
            if element.page_number is not None:
                pages.append(element.page_number)
            if counter("\n\n".join(parts)) >= target_tokens:
                flush()
    flush()
    return chunks


def _record_ocr_drops(
    chunker: HierarchicalChunker,
    parsed_doc: ParsedDocument,
    dropped: list[RetrievalChunk],
    fallback_drops: dict[str, int],
) -> None:
    """Fold OCR-gate drops into the chunker's drop accounting
    (``last_drop_summary`` / ``last_page_drop_ledger``, reason
    ``"ocr_garbage"``) and log the gate's outcome."""
    reason: DropReason = "ocr_garbage"
    count = len(dropped) + fallback_drops.get("ocr_garbage", 0)
    if count:
        summary: Counter[DropReason] = Counter(chunker.last_drop_summary)
        summary[reason] += count
        chunker.last_drop_summary = dict(summary)
        chunker.last_page_drop_ledger = [
            *chunker.last_page_drop_ledger,
            *(
                PageDrop(page_number=c.page_number, reason=reason)
                for c in dropped
                if c.page_number is not None
            ),
        ]
    if count:
        log.info(
            "book_rag.chunk.ocr_quality",
            source_id=parsed_doc.source_id,
            dropped=count,
            dropped_by_reason=chunker.last_drop_summary,
        )


#: A title that is only a content hash: a PDF stored as ``verified/sha256/<sha>.pdf``
#: parses with its file stem as the title.
_HASH_TITLE: Final = re.compile(r"[0-9a-f]{40,64}", re.IGNORECASE)


def _with_book_title(
    parsed_doc: ParsedDocument, book_metadata: dict[str, Any] | None
) -> ParsedDocument:
    """``parsed_doc`` titled with the book's own title when its parsed title is
    empty or a bare content hash (VOR-1454): the title is the root of every
    section path, and the tagger reads those paths."""
    title = str((book_metadata or {}).get("title") or "").strip()
    parsed_title = (parsed_doc.title or "").strip()
    if not title or (parsed_title and not _HASH_TITLE.fullmatch(parsed_title)):
        return parsed_doc
    return parsed_doc.model_copy(update={"title": title})


def do_build_chunks(
    parsed_doc: ParsedDocument,
    composites: list[CompositeChunk] | None = None,
    *,
    book_metadata: dict[str, Any] | None = None,
    settings: Settings | None = None,
    embedding_client: Any | None = None,
    ocr_dropped: list[RetrievalChunk] | None = None,
) -> list[RetrievalChunk]:
    """Run HierarchicalChunker over *parsed_doc*; recover with fallback if zero chunks.

    ``ocr_dropped``, when given, receives every chunk the OCR quality gate
    dropped (main path and fallback), for review tools such as
    ``corpus rechunk-preview`` (VOR-1452).

    ``book_metadata`` (source_book/title/author/domain_tags/license_class/
    canonical_source_id) is stamped onto every chunk's metadata so downstream
    consumers (DSI/graph extraction, the ingestion validator, retrieval filters)
    have book identity. Fresh ingests otherwise carry none — only the migrated
    golden corpus had it.
    """
    chunker = _build_chunker(settings or get_settings(), embedding_client)
    parsed_doc = _with_book_title(parsed_doc, book_metadata)

    # Boilerplate policy (VOR-1427): body elements filtered and
    # composites remapped before chunking, whole-boilerplate chunks dropped
    # after -- one helper shared with the visual sidecar's chunk path.
    original_element_count = len(parsed_doc.elements)
    chunks, parsed_doc = chunk_without_boilerplate(chunker, parsed_doc, composites)
    # Per-chunk OCR quality gate (VOR-1452): noise dropped and counted
    # in the chunker's drop accounting; everything else is kept.
    gate_dropped: list[RetrievalChunk] = []
    gated: list[RetrievalChunk] = []
    for chunk in chunks:
        kept = _ocr_gate(chunk)
        if kept is None:
            gate_dropped.append(chunk)
        else:
            gated.append(kept)
    chunks = gated
    if ocr_dropped is not None:
        ocr_dropped.extend(gate_dropped)
    fallback_used = False
    fallback_drops: dict[str, int] = {}
    if not chunks:
        chunks = _fallback_text_chunks(
            parsed_doc, counter=chunker.count, drop_counts=fallback_drops, dropped=ocr_dropped
        )
        chunks = [c for c in chunks if not is_boilerplate(c.content or "")]
        fallback_used = bool(chunks)
    _record_ocr_drops(chunker, parsed_doc, gate_dropped, fallback_drops)

    if not chunks:
        raise RuntimeError(
            f"book_rag.build_chunks produced zero chunks for {parsed_doc.source_id} "
            f"(parsed_elements={original_element_count}, "
            f"after_boilerplate_filter={len(parsed_doc.elements)}, "
            f"ocr_garbage_dropped={len(gate_dropped) + fallback_drops.get('ocr_garbage', 0)}, "
            f"pages={parsed_doc.total_pages})"
        )

    # Preserve forward_path metadata if present; otherwise create a minimal one.
    forward_path: dict[str, Any] = dict(parsed_doc.metadata.get("book_rag_forward_path") or {})
    # Only the PDF parse path builds a forward path; stamp the parser for the
    # rest too (EPUB/MOBI: "epub"/"mobi"), so the payload's parser_used tells
    # an ebook from a legacy PDF without a Postgres lookup.
    forward_path.setdefault("parser_used", parsed_doc.parser_used)
    forward_path["chunk_count"] = len(chunks)
    if fallback_used:
        forward_path["chunking_fallback"] = "parsed_text_zero_chunk_recovery"

    book_meta = dict(book_metadata or {})
    # Wire heuristic domain tags (VOR-296): fresh ingests carry none. Compute
    # deterministic book-level tags from a bounded text sample and let the
    # per-chunk setdefault below stamp them. Never overwrite tags the caller
    # already supplied (migrated/golden corpus, operator backfill).
    if not book_meta.get("domain_tags"):
        sample = "\n".join((c.embedding_text or c.content or "") for c in chunks[:50])[:20000]
        heuristic = heuristic_domain_tags(sample)
        if heuristic:
            book_meta["domain_tags"] = heuristic
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        metadata["book_rag_forward_path"] = forward_path
        for key, value in book_meta.items():
            metadata.setdefault(key, value)
        chunk.metadata = metadata

    # Degeneracy is a data-quality warning, not a reason to discard usable
    # extracted passages. The workflow can publish them with bounded warnings.
    stats = chunk_output_stats_from_chunks(chunks).model_copy(
        update={"total_pages": parsed_doc.total_pages}
    )
    degenerate, reason = is_degenerate_chunk_output(stats)
    if degenerate:
        _warn_chunks(
            chunks,
            "chunk_output_degenerate",
            f"{reason} (chunks={stats.chunk_count}, pages={stats.effective_pages}, "
            f"tokens={stats.total_tokens})",
        )

    # Runaway guard (2026-06-08 incident; VOR-1392 EPUB fix): the degenerate check
    # above catches the LOW end (collapse to too-few chunks); a scanned/garbage doc
    # can explode the other way into hundreds of tiny chunks/page. is_runaway_chunk_output
    # requires BOTH a high chunks/page ratio AND abnormally small chunks, so reflowable
    # EPUBs (synthetic-low page counts, normal chunk size) are not false-flagged.
    # Judge chunk size on CONTENT tokens: token_count covers embedding_text, which
    # prepends the section path, so a bloated path made 297k one-character chunks
    # read as 782 tok/chunk and slip past the mean-size floor (VOR-1416).
    content_stats = stats.model_copy(
        update={"total_tokens": sum(chunker.count(c.content) for c in chunks)}
    )
    runaway, runaway_detail = is_runaway_chunk_output(content_stats)
    if runaway:
        repack_drops: dict[str, int] = {}
        repack_ocr_dropped: list[RetrievalChunk] = []
        repacked = _fallback_text_chunks(
            parsed_doc,
            counter=chunker.count,
            drop_counts=repack_drops,
            dropped=repack_ocr_dropped,
            target_tokens=350,
            max_tokens=500,
            max_chunks=_RUNAWAY_REPACK_MAX_CHUNKS,
            max_source_chars=_RUNAWAY_REPACK_MAX_TEXT_CHARS,
        )
        repacked = [c for c in repacked if not is_boilerplate(c.content or "")]
        _record_ocr_drops(chunker, parsed_doc, repack_ocr_dropped, repack_drops)
        if not repacked:
            raise RuntimeError(
                f"book_rag.build_chunks runaway chunk output for {parsed_doc.source_id}: "
                f"{runaway_detail}; bounded text repack produced no usable chunks"
            )
        repacked_stats = chunk_output_stats_from_chunks(repacked).model_copy(
            update={
                "total_pages": parsed_doc.total_pages,
                "total_tokens": sum(chunker.count(c.content) for c in repacked),
            }
        )
        still_runaway, repack_detail = is_runaway_chunk_output(repacked_stats)
        chunks = repacked
        forward_path["chunk_count"] = len(chunks)
        forward_path["chunking_fallback"] = "bounded_text_repack_after_runaway"
        for chunk in chunks:
            metadata = dict(chunk.metadata or {})
            metadata["chunking_fallback"] = "bounded_text_repack_after_runaway"
            chunk.metadata = metadata
        _warn_chunks(
            chunks,
            "runaway_chunk_output_repacked_with_residual_warning"
            if still_runaway
            else "runaway_chunk_output_repacked",
            (
                f"Original output was unsafe ({runaway_detail}); bounded text repack remains "
                f"above the heuristic runaway threshold ({repack_detail})."
                if still_runaway
                else f"Original output was unsafe ({runaway_detail}); used bounded text repack."
            ),
        )
        for chunk in chunks:
            metadata = dict(chunk.metadata or {})
            metadata["book_rag_forward_path"] = forward_path
            for key, value in book_meta.items():
                metadata.setdefault(key, value)
            chunk.metadata = metadata

    return chunks


async def do_embed_raw(
    chunks: list[RetrievalChunk],
    embedding_client: Any,
    *,
    # The client's full batch: batch 32 measured 2.5x the throughput of 8 on a
    # loaded bge-m3 worker (VOR-1427).
    embed_batch: int = MAX_EMBED_BATCH_TEXTS,
) -> dict[str, Any]:
    """Embed *chunks* via *embedding_client* and return processed vectors.

    Returns a dict with keys ``dense``, ``sparse``, ``colbert``.  Each key
    holds a list with one entry per chunk.  Sparse vectors are already
    normalised to ``dict[int, float]``.  ColBERT vectors are downsampled to
    192 (``VORAG_QDRANT_MAX_COLBERT_VECTORS``) and empty results fall back
    to the dense vector (never silent-drop).
    """
    if not chunks:
        return {"dense": [], "sparse": [], "colbert": [], "embedding_provenance": []}

    texts = [c.embedding_text or c.content for c in chunks]
    embed_batch = max(1, embed_batch)
    max_length = int(getattr(embedding_client, "_max_length", DEFAULT_EMBED_MAX_LENGTH))
    request_colbert = include_colbert_at_ingest()
    recipe = _embedding_recipe(max_length, request_colbert)
    recipe_sha256 = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    all_dense: list[list[float]] = []
    all_sparse: list[dict[int, float]] = []
    all_colbert: list[list[list[float]]] = []
    all_provenance: list[dict[str, Any]] = []
    observed_identities: set[str] = set()

    for start in range(0, len(texts), embed_batch):
        batch_texts = texts[start : start + embed_batch]
        resp = await embedding_client.embed(
            batch_texts,
            return_dense=True,
            return_sparse=True,
            # ColBERT at ingest is opt-in (default off) — points still carry a
            # dense-derived fallback ColBERT for schema compat (below). See
            # include_colbert_at_ingest.
            return_colbert=request_colbert,
        )
        if not isinstance(resp, dict):
            raise RuntimeError("embedding provider returned a malformed response")
        dense = resp.get("dense")
        sparse = resp.get("sparse")
        if not isinstance(dense, list) or len(dense) != len(batch_texts):
            raise RuntimeError(
                "embedding provider returned a dense response with the wrong cardinality"
            )
        if not isinstance(sparse, list) or len(sparse) != len(batch_texts):
            raise RuntimeError(
                "embedding provider returned a sparse response with the wrong cardinality"
            )
        colbert = resp.get("colbert") or []
        runtime_values = resp.get("runtime_identity_per_text")
        if not isinstance(runtime_values, list) or len(runtime_values) != len(batch_texts):
            runtime_values = [resp.get("runtime_identity")] * len(batch_texts)
        token_values = resp.get("input_token_counts_per_text")
        if not isinstance(token_values, list) or len(token_values) != len(batch_texts):
            token_values = [None] * len(batch_texts)
        truncated_values = resp.get("input_truncated_per_text")
        if not isinstance(truncated_values, list) or len(truncated_values) != len(batch_texts):
            truncated_values = [None] * len(batch_texts)
        metadata_errors = resp.get("embedding_metadata_errors", [])
        if not isinstance(metadata_errors, list):
            metadata_errors = ["worker_metadata_invalid"]
        runtime_mixed = bool(resp.get("runtime_identity_mixed"))

        for i in range(len(batch_texts)):
            dense_row = dense[i]
            if not isinstance(dense_row, Sequence) or isinstance(dense_row, str | bytes):
                raise RuntimeError("embedding provider returned a malformed dense vector")
            dense_i = [float(v) for v in dense_row]
            sparse_row = sparse[i]
            if not isinstance(sparse_row, dict):
                raise RuntimeError("embedding provider returned a malformed sparse vector")
            sparse_i = sparse_payload_to_int_map(sparse_row)
            colbert_i = colbert[i] if i < len(colbert) else []
            colbert_vectors = downsample_colbert_vectors(colbert_i)
            original_colbert_count = len(colbert_vectors) if not colbert_i else len(colbert_i)
            colbert_limit = max_colbert_vectors_per_point()
            if not colbert_vectors:
                colbert_vectors = fallback_colbert_vectors(dense_i)
            colbert_transform = (
                "dense_fallback"
                if not colbert_i
                else "uniform_sample"
                if len(colbert_i) > colbert_limit
                else "as_returned"
            )
            all_dense.append(dense_i)
            all_sparse.append(sparse_i)
            all_colbert.append(colbert_vectors)
            identity = runtime_values[i]
            if isinstance(identity, dict):
                identity_key = json.dumps(identity, sort_keys=True, separators=(",", ":"))
                observed_identities.add(identity_key)
                identity_status = str(identity.get("identity_status") or "unknown")
            else:
                identity = None
                identity_status = "unknown"
            input_token_count = token_values[i]
            if type(input_token_count) is not int or input_token_count < 0:
                input_token_count = None
            input_truncated = truncated_values[i]
            if type(input_truncated) is not bool:
                input_truncated = None
            provenance = {
                "input_sha256": hashlib.sha256(batch_texts[i].encode("utf-8")).hexdigest(),
                "input_char_count": len(batch_texts[i]),
                "input_token_count": input_token_count,
                "input_truncated": input_truncated,
                "input_metadata_errors": list(metadata_errors),
                "runtime_identity": identity,
                "runtime_identity_status": identity_status,
                "request_recipe": recipe,
                "request_recipe_sha256": recipe_sha256,
                "output_manifest": {
                    "dense_dimension": len(dense_i),
                    "sparse_nonzero_count": len(sparse_i),
                    "colbert_vectors_original": original_colbert_count,
                    "colbert_vectors_stored": len(colbert_vectors),
                    "colbert_transform": colbert_transform,
                },
            }
            all_provenance.append(provenance)
            if identity_status != "observed_config":
                _warn_chunk(
                    chunks[start + i],
                    "embedding_runtime_identity_unknown",
                    "the embedding worker did not provide a verified loaded configuration revision",
                )
            if input_truncated is True:
                _warn_chunk(
                    chunks[start + i],
                    "embedding_input_truncated",
                    "the embedding worker reports truncating the exact stored input",
                )
            if input_truncated is None or input_token_count is None:
                _warn_chunk(
                    chunks[start + i],
                    "embedding_input_metadata_unknown",
                    "worker input token count or truncation metadata is unavailable or invalid",
                )
            if metadata_errors:
                _warn_chunk(
                    chunks[start + i],
                    "embedding_worker_metadata_invalid",
                    ",".join(str(error) for error in metadata_errors),
                )
            if runtime_mixed:
                _warn_chunk(
                    chunks[start + i],
                    "embedding_runtime_identity_mixed",
                    "one embedding request combined responses with different runtime identities",
                )

    if len(observed_identities) > 1:
        for chunk in chunks:
            _warn_chunk(
                chunk,
                "embedding_runtime_identity_mixed",
                "one book was embedded by requests with different observed runtime identities",
            )

    return {
        "dense": all_dense,
        "sparse": all_sparse,
        "colbert": all_colbert,
        "embedding_provenance": all_provenance,
    }


def build_raw_points(
    chunks: list[RetrievalChunk],
    embeddings: dict[str, Any],
    source_id: str,
    *,
    collection: str | None = None,
    trust_level: str = "quarantined",
    generation_id: str | None = None,
    ingest_state: str = "active",
) -> list[VoPoint]:
    """Build complete raw passage representations using the production writer recipe.

    This pure seam is shared by ordinary intake and scoped corpus repair; it
    constructs payloads and lexical vectors from actual embedding responses.
    """
    if not chunks:
        return []

    dense_list = embeddings.get("dense")
    sparse_list = embeddings.get("sparse")
    if not isinstance(dense_list, list) or len(dense_list) != len(chunks):
        raise ValueError("raw point dense embeddings must match chunk cardinality")
    if not isinstance(sparse_list, list) or len(sparse_list) != len(chunks):
        raise ValueError("raw point sparse embeddings must match chunk cardinality")
    colbert_list: list[list[list[float]]] = embeddings.get("colbert") or []

    points: list[VoPoint] = []
    provenance_list = embeddings.get("embedding_provenance") or []
    for chunk_index, chunk in enumerate(chunks):
        dense = dense_list[chunk_index]
        sparse = sparse_list[chunk_index]
        if not isinstance(dense, list):
            raise ValueError(f"raw point dense embedding {chunk_index} is malformed")
        if not isinstance(sparse, dict):
            raise ValueError(f"raw point sparse embedding {chunk_index} is malformed")
        colbert = colbert_list[chunk_index] if chunk_index < len(colbert_list) else []
        bm25 = tokens_to_bm25_sparse(tokenize(chunk.content))
        embedding_text = chunk.embedding_text or chunk.content
        provenance = (
            dict(provenance_list[chunk_index])
            if chunk_index < len(provenance_list) and isinstance(provenance_list[chunk_index], dict)
            else {}
        )
        if not provenance:
            recipe = _embedding_recipe(DEFAULT_EMBED_MAX_LENGTH, include_colbert_at_ingest())
            provenance = {
                "input_sha256": hashlib.sha256(embedding_text.encode("utf-8")).hexdigest(),
                "input_char_count": len(embedding_text),
                "input_token_count": None,
                "input_truncated": None,
                "input_metadata_errors": [],
                "runtime_identity": None,
                "runtime_identity_status": "unknown",
                "request_recipe": recipe,
                "request_recipe_sha256": hashlib.sha256(
                    json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "output_manifest": {
                    "dense_dimension": len(dense),
                    "sparse_nonzero_count": len(sparse),
                    "colbert_vectors_original": (
                        len(colbert_list[chunk_index]) if chunk_index < len(colbert_list) else 0
                    ),
                    "colbert_vectors_stored": len(colbert),
                    "colbert_transform": "caller_supplied_vectors_identity_unknown",
                },
            }

        metadata = dict(chunk.metadata or {})
        forward_path = metadata.get("book_rag_forward_path")
        if not isinstance(forward_path, dict):
            forward_path = {}

        payload: dict[str, Any] = {
            "chunk_id": chunk.chunk_id,
            "source_id": chunk.source_id,
            "document_id": chunk.document_id,
            "section_path": chunk.section_path,
            "section_title": chunk.section_title,
            "chapter_path": chunk.chapter_path,
            "chunk_type": chunk.chunk_type,
            "content": chunk.content,
            "stage": "raw",
            "trust_level": trust_level,
            "generation_id": generation_id,
            "ingest_state": ingest_state,
            "page_number": chunk.page_number,
            "page_range": chunk.page_range,
            "token_count": chunk.token_count,
            "has_image": chunk.has_image,
            "image_id": chunk.image_id,
            "image_ref": chunk.image_ref,
            "vlm_description": chunk.vlm_description,
            "metadata": metadata,
            # Book identity — top-level for validator + filtering (mirrors the
            # nested metadata stamped by do_build_chunks / the golden corpus).
            "source_book": metadata.get("source_book"),
            "author": metadata.get("author"),
            "license_class": metadata.get("license_class"),
            "canonical_source_id": metadata.get("canonical_source_id"),
            "project_id": metadata.get("project_id"),
            "domain_tags": metadata.get("domain_tags"),
            "forward_path_decision_doc": forward_path.get("decision_doc"),
            "ocr_stack": forward_path.get("ocr_stack"),
            "parser_backend": forward_path.get("parser_backend"),
            "parser_used": forward_path.get("parser_used"),
            "colbert_fallback": len(colbert) == 1 and colbert == [dense],
            "colbert_vector_count_original": (
                len(colbert_list[chunk_index]) if chunk_index < len(colbert_list) else 0
            ),
            "colbert_vector_count_stored": len(colbert),
            "embedding_provenance": provenance,
        }
        # A figure's neighbour window, so rebuilds re-embed what ingest did.
        figure_window = figure_payload_embedding_text(chunk)
        if embedding_text != chunk.content:
            payload[PAYLOAD_EMBEDDING_TEXT_KEY] = embedding_text
        elif figure_window is not None:
            payload[PAYLOAD_EMBEDDING_TEXT_KEY] = figure_window

        points.append(
            VoPoint(
                source_id=source_id,
                chunk_index=chunk_index,
                stage="raw",
                version_hash=generation_id,
                dense=list(dense),
                sparse=dict(sparse),
                bm25=bm25,
                colbert=list(colbert),
                payload=payload,
            )
        )

    return points


async def do_upsert_raw(
    chunks: list[RetrievalChunk],
    embeddings: dict[str, Any],
    qdrant_store: Any,
    source_id: str,
    *,
    collection: str | None = None,
    trust_level: str = "quarantined",
    generation_id: str | None = None,
    ingest_state: str = "active",
    publication_job: SourcePublicationJob | None = None,
) -> dict[str, Any]:
    """Build raw passage representations and publish through the normal writer."""
    points = build_raw_points(
        chunks,
        embeddings,
        source_id,
        collection=collection,
        trust_level=trust_level,
        generation_id=generation_id,
        ingest_state=ingest_state,
    )
    if not points:
        return {"point_count": 0, "point_ids": []}
    if publication_job is None:
        result = await qdrant_store.upsert_vectors(collection or _COLLECTION_RAW, points)
    else:
        result = await qdrant_store.upsert_vectors(
            collection or _COLLECTION_RAW,
            points,
            publication_job=publication_job,
        )
    if result is None:
        if publication_job is not None:
            raise RuntimeError("publication-aware Qdrant writer returned no point IDs")
        point_ids = [
            deterministic_point_id(source_id, index, "raw", generation_id)
            for index in range(len(points))
        ]
    else:
        if not isinstance(result, list) or len(result) != len(points):
            raise RuntimeError("Qdrant writer returned point IDs with the wrong cardinality")
        point_ids = [str(point_id) for point_id in result]
    return {"point_count": len(points), "point_ids": point_ids}

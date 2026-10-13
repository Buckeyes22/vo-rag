"""book_rag IngestionValidator — 8-check audit over raw/sterile points (VOR-7).

Design notes:

- chunking imports come from ``vo_rag.core.chunking``;
- collections are ``vo_rag_books_raw_v1`` and ``vo_rag_books_sterile_v1``;
- env vars use the ``VO_RAG_BOOK_RAG_*`` prefix;
- the artifacts-bucket default is ``vorag-artifacts``
  (matches ``vo_rag.core.config.settings.GarageBuckets.artifacts``);
- the audit runs 8 checks. There is no ``nonzero_points`` check because it
  is fully subsumed by ``chunk_output_not_degenerate`` failing on zero
  payloads and ``sample_query_returns`` failing on zero hits;
- ``artifact_id_exists`` is conditional (only emitted when the
  caller passes an explicit ``artifact_id``) and is NOT counted in the
  8-check audit;
- the JSONB ``@>`` artifact lookup casts the parameter to ``text`` via
  ``::text`` so the planner uses a GIN-friendly predicate on the canonical
  ledger;
- degeneracy floors are 11 chunks, 25 pages, 0.08 chunks/page and 20
  tokens/page;
- ``non_book_skip_by_pages`` stays fail-open for unknown page counts
  (a page-count read error must never silently drop a real book);
- ``license_class == "unknown"`` is treated as missing in
  ``metadata_field_completeness`` (the canonical
  ``LicenseClass.UNKNOWN`` means "missing").
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, Field
from qdrant_client.models import Condition, FieldCondition, Filter, MatchValue

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.storage.qdrant import retry_read

BUCKET_ARTIFACTS = "vorag-artifacts"
RAW_COLLECTION = "vo_rag_books_raw_v1"
STERILE_COLLECTION = "vo_rag_books_sterile_v1"
_SCROLL_LIMIT = 512
_PAGE_NUMBER_RE = re.compile(r"\d+")

# 2026-06 structure scan: the false-PASS failures had 1-10 chunks while the
# healthy corpus median was 166. For book-scale documents, require at least the
# first non-degenerate count above that bad band.
DEGENERATE_MIN_CHUNKS_DEFAULT = 11
# Keep the guard scoped to books/booklets, not tiny one-page smoke fixtures.
DEGENERATE_MIN_PAGES_DEFAULT = 25
# Pre-parse non-book skip: a PDF with fewer than this many pages is almost
# certainly not a book (waiver form, log sheet, classroom sheet, telemetry
# export). Skip it BEFORE spending OCR/chunk/embed — far cheaper than the
# post-chunk degenerate reject, and the source stays in the canonical ledger.
MIN_BOOK_PAGES_DEFAULT = 5
# Conservative density floor: fewer than one retrieval chunk per 12.5 pages is
# implausible for parsed book prose, but still far below normal corpus density.
DEGENERATE_MIN_CHUNKS_PER_PAGE_DEFAULT = 0.08
# Token floor catches near-empty parser/chunker output without being a quality
# bar: 20 tokens/page is far below real prose, tables, or captions.
DEGENERATE_MIN_TOKENS_PER_PAGE_DEFAULT = 20
METADATA_REQUIRED_DEFAULT = ("source_book", "license_class", "canonical_source_id")
METADATA_WARN_ONLY_FIELDS = ("author", "domain_tags")
_METADATA_MISSING_VALUE = "<missing>"
_LICENSE_CLASS_UNKNOWN = "unknown"
_SKIPPED_STERILIZE_DISABLED = "skipped: sterilize disabled"
_SKIPPED_NO_PG_POOL = "skipped: no pg pool"
_SKIPPED_COMPLIANCE_SUB_CHUNKING_NOT_RUN = "skipped: compliance sub-chunking not run"


class ValidationCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class ValidationReport(BaseModel):
    document_id: str
    source_id: str
    checks: list[ValidationCheck] = Field(default_factory=list)
    passed: bool = False


class _ArtifactRecord(BaseModel):
    id: str
    title: str | None = None


class _GarageLike(Protocol):
    async def head_object(self, bucket: str, key: str) -> Any: ...


class _RetrieverLike(Protocol):
    async def hybrid_search(
        self,
        query: str,
        collection: str,
        *,
        top_k: int,
        filters: dict[str, Any],
    ) -> Sequence[Any]: ...


class ChunkOutputStats(BaseModel):
    chunk_count: int
    total_tokens: int
    distinct_section_paths: int
    total_pages: int | None = None
    page_span: int | None = None

    @property
    def effective_pages(self) -> int | None:
        candidates = [page for page in (self.total_pages, self.page_span) if page is not None]
        return max(candidates) if candidates else None


class MetadataFieldCompleteness(BaseModel):
    field: str
    total_count: int
    missing_count: int
    dominant_value: str
    dominant_count: int
    dominant_missing: bool


class IngestionValidator:
    def __init__(
        self,
        qdrant: Any,
        pool: Any,
        garage: _GarageLike,
        retriever: _RetrieverLike,
        *,
        figures: Any,
        artifacts_bucket: str = BUCKET_ARTIFACTS,
        sterilize_enabled: bool = True,
        raw_collection: str = RAW_COLLECTION,
    ) -> None:
        self.qdrant = qdrant
        self.pool = pool
        self.garage = garage
        self.retriever = retriever
        self.figures = figures
        self.artifacts_bucket = artifacts_bucket
        self.sterilize_enabled = sterilize_enabled
        self.raw_collection = raw_collection

    def _qdrant_client(self) -> Any:
        return getattr(self.qdrant, "_client", self.qdrant)

    @staticmethod
    def _source_filter(
        source_id: str,
        generation_id: str | None = None,
        ingest_state: str | None = None,
    ) -> Filter:
        must: list[Condition] = [FieldCondition(key="source_id", match=MatchValue(value=source_id))]
        if generation_id is not None:
            must.append(FieldCondition(key="generation_id", match=MatchValue(value=generation_id)))
        if ingest_state is not None:
            must.append(FieldCondition(key="ingest_state", match=MatchValue(value=ingest_state)))
        return Filter(
            must=must,
        )

    async def _count(
        self,
        collection: str,
        source_id: str,
        generation_id: str | None = None,
        ingest_state: str | None = None,
    ) -> int:
        try:
            response = await retry_read(
                lambda: self._qdrant_client().count(
                    collection_name=collection,
                    count_filter=self._source_filter(source_id, generation_id, ingest_state),
                    exact=True,
                )
            )
        except Exception as exc:
            if _is_missing_collection_error(exc):
                return 0
            raise
        return int(getattr(response, "count", 0))

    async def _sample_payloads(
        self,
        collection: str,
        source_id: str,
        n: int = 5,
        generation_id: str | None = None,
        ingest_state: str | None = None,
    ) -> list[dict[str, Any]]:
        try:
            points, _ = await self._qdrant_client().scroll(
                collection_name=collection,
                scroll_filter=self._source_filter(source_id, generation_id, ingest_state),
                limit=n,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            if _is_missing_collection_error(exc):
                return []
            raise
        payloads: list[dict[str, Any]] = []
        for point in points:
            payload = getattr(point, "payload", None)
            if isinstance(payload, dict):
                payloads.append(payload)
        return payloads

    async def _all_payloads(
        self,
        collection: str,
        source_id: str,
        generation_id: str | None = None,
        ingest_state: str | None = None,
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        offset: Any = None
        while True:
            points, offset = await self._qdrant_client().scroll(
                collection_name=collection,
                scroll_filter=self._source_filter(source_id, generation_id, ingest_state),
                limit=_SCROLL_LIMIT,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = getattr(point, "payload", None)
                if isinstance(payload, dict):
                    payloads.append(payload)
            if offset is None:
                return payloads

    async def _chunk_output_not_degenerate_check(
        self,
        source_id: str,
        generation_id: str | None = None,
        ingest_state: str | None = None,
    ) -> ValidationCheck:
        payloads = await self._all_payloads(
            self.raw_collection, source_id, generation_id, ingest_state
        )
        stats = chunk_output_stats_from_payloads(payloads)

        if not payloads:
            return ValidationCheck(
                name="chunk_output_not_degenerate",
                passed=False,
                detail="chunks=0 payloads missing",
            )

        is_degenerate, detail = is_degenerate_chunk_output(stats)
        return ValidationCheck(
            name="chunk_output_not_degenerate",
            passed=not is_degenerate,
            detail=detail,
        )

    async def _metadata_completeness_check(
        self,
        source_id: str,
        generation_id: str | None = None,
        ingest_state: str | None = None,
    ) -> ValidationCheck:
        payloads = await self._all_payloads(
            self.raw_collection, source_id, generation_id, ingest_state
        )
        required_fields = _metadata_required_fields()
        guard = _metadata_guard_mode()
        required_statuses = [
            metadata_field_completeness(payloads, field) for field in required_fields
        ]
        warn_statuses = [
            metadata_field_completeness(payloads, field)
            for field in METADATA_WARN_ONLY_FIELDS
            if field not in required_fields
        ]
        missing_required = [status for status in required_statuses if status.dominant_missing]
        warn_missing = [status for status in warn_statuses if status.dominant_missing]
        return ValidationCheck(
            name="metadata_completeness",
            passed=not missing_required or guard == "warn",
            detail=_metadata_completeness_detail(
                payload_count=len(payloads),
                required_fields=required_fields,
                guard=guard,
                missing_required=missing_required,
                warn_missing=warn_missing,
            ),
        )

    async def validate(
        self,
        document_id: str,
        source_id: str,
        artifact_id: str | None = None,
        generation_id: str | None = None,
        ingest_state: str | None = None,
    ) -> ValidationReport:
        checks: list[ValidationCheck] = []

        raw_n = await self._count(self.raw_collection, source_id, generation_id, ingest_state)
        if self.sterilize_enabled:
            ster_n = await self._count(STERILE_COLLECTION, source_id, generation_id, ingest_state)
            checks.append(
                ValidationCheck(
                    name="raw_sterile_1to1",
                    passed=raw_n == ster_n,
                    detail=f"{raw_n}/{ster_n}",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name="raw_sterile_1to1",
                    passed=True,
                    detail=_SKIPPED_STERILIZE_DISABLED,
                )
            )
        checks.append(
            await self._chunk_output_not_degenerate_check(source_id, generation_id, ingest_state)
        )
        checks.append(
            await self._metadata_completeness_check(source_id, generation_id, ingest_state)
        )

        sample = await self._sample_payloads(
            self.raw_collection,
            source_id,
            n=5,
            generation_id=generation_id,
            ingest_state=ingest_state,
        )
        required = {"source_id", "page_number", "section_title"}
        missing = [payload for payload in sample if not required <= payload.keys()]
        checks.append(
            ValidationCheck(
                name="payload_fields_complete",
                passed=not missing,
                detail=f"{len(missing)} missing",
            )
        )

        img_missing = 0
        for payload in sample:
            key = payload.get("image_ref")
            if not key:
                continue
            resolved = self.figures is not None and bool(self.figures.exists(str(key)))
            if not resolved:
                img_missing += 1
        checks.append(
            ValidationCheck(
                name="images_resolve",
                passed=img_missing == 0,
                detail=f"{img_missing} missing",
            )
        )

        artifact: _ArtifactRecord | None = None
        if self.pool is None:
            checks.append(
                ValidationCheck(
                    name="artifact_exists",
                    passed=True,
                    detail=_SKIPPED_NO_PG_POOL,
                )
            )
            if artifact_id is not None:
                checks.append(
                    ValidationCheck(
                        name="artifact_id_exists",
                        passed=True,
                        detail=_SKIPPED_NO_PG_POOL,
                    )
                )
        else:
            artifact = await _get_artifact_by_source(self.pool, source_id)
            checks.append(
                ValidationCheck(
                    name="artifact_exists",
                    passed=artifact is not None,
                    detail=(artifact.id if artifact else ""),
                )
            )
            if artifact_id is not None:
                artifact_by_id = await _get_artifact_by_id(self.pool, artifact_id)
                checks.append(
                    ValidationCheck(
                        name="artifact_id_exists",
                        passed=artifact_by_id is not None,
                        detail=artifact_id if artifact_by_id is not None else "",
                    )
                )

        if self.pool is None:
            checks.append(
                ValidationCheck(
                    name="sub_chunks_present",
                    passed=True,
                    detail=_SKIPPED_NO_PG_POOL,
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name="sub_chunks_present",
                    passed=True,
                    detail=_SKIPPED_COMPLIANCE_SUB_CHUNKING_NOT_RUN,
                )
            )

        sample_collection = STERILE_COLLECTION if self.sterilize_enabled else self.raw_collection
        query_text = (artifact.title if artifact else "") or "introduction"
        retrieval_filters: dict[str, Any] = {"source_id": source_id}
        if generation_id is not None:
            retrieval_filters["generation_id"] = generation_id
        retrieval_kwargs: dict[str, Any] = {
            "top_k": 5,
            "filters": retrieval_filters,
        }
        if ingest_state is not None:
            retrieval_kwargs["ingest_state_override"] = str(ingest_state)
        hits = await self.retriever.hybrid_search(
            query_text,
            sample_collection,
            **retrieval_kwargs,
        )
        checks.append(
            ValidationCheck(
                name="sample_query_returns",
                passed=len(hits) > 0,
                detail=f"{len(hits)} hits @ {sample_collection}",
            )
        )

        passed = all(check.passed for check in checks)
        return ValidationReport(
            document_id=document_id,
            source_id=source_id,
            checks=checks,
            passed=passed,
        )


def _is_missing_collection_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    content = getattr(exc, "content", b"")
    if isinstance(content, bytes):
        content_text = content.decode("utf-8", errors="ignore")
    elif content is None:
        content_text = ""
    else:
        content_text = str(content)

    error_text = f"{exc} {content_text}".lower()
    return (
        status_code == 404
        and "collection" in error_text
        and (
            "doesn't exist" in error_text
            or "does not exist" in error_text
            or "not found" in error_text
        )
    )


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return max(minimum, default)
    try:
        return max(minimum, int(raw))
    except ValueError:
        return max(minimum, default)


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return max(minimum, default)
    try:
        return max(minimum, float(raw))
    except ValueError:
        return max(minimum, default)


def _metadata_required_fields() -> tuple[str, ...]:
    raw = os.environ.get("VO_RAG_BOOK_RAG_METADATA_REQUIRED")
    if raw is None:
        return METADATA_REQUIRED_DEFAULT
    fields = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    return fields or METADATA_REQUIRED_DEFAULT


def _metadata_guard_mode() -> str:
    raw = os.environ.get("VO_RAG_BOOK_RAG_METADATA_GUARD", "fail").strip().lower()
    if raw == "warn":
        return "warn"
    return "fail"


def chunk_output_stats_from_chunks(chunks: Sequence[RetrievalChunk]) -> ChunkOutputStats:
    page_numbers = [chunk.page_number for chunk in chunks if chunk.page_number is not None]
    page_span = None
    if page_numbers:
        page_span = max(page_numbers) - min(page_numbers) + 1
    section_paths = {chunk.section_path for chunk in chunks if chunk.section_path}
    return ChunkOutputStats(
        chunk_count=len(chunks),
        total_tokens=sum(max(0, chunk.token_count) for chunk in chunks),
        distinct_section_paths=len(section_paths),
        page_span=page_span,
    )


def non_book_skip_by_pages(page_count: int | None) -> tuple[bool, str]:
    """Pre-parse gate: should this PDF be skipped as a non-book by page count?

    Returns ``(True, reason)`` when ``page_count`` is known and below
    ``VO_RAG_BOOK_RAG_MIN_BOOK_PAGES`` (default ``MIN_BOOK_PAGES_DEFAULT`` = 5) —
    a document that short is almost certainly a form/log/sheet, not a book.
    Callers skip parsing entirely (no OCR/chunk/embed) while LEAVING the source
    in the canonical ledger. An unknown page count (``None``) never skips —
    fail-open so a page-count read error can't silently drop a real book.
    """
    threshold = _env_int(
        "VO_RAG_BOOK_RAG_MIN_BOOK_PAGES",
        MIN_BOOK_PAGES_DEFAULT,
        minimum=1,
    )
    if page_count is not None and page_count < threshold:
        return (True, f"non_book_too_few_pages:{page_count}<{threshold}")
    return (False, "")


RUNAWAY_MAX_CHUNKS_PER_PAGE_DEFAULT = 60
RUNAWAY_MIN_MEAN_TOKENS_DEFAULT = 40


def is_runaway_chunk_output(
    stats: ChunkOutputStats,
) -> tuple[bool, str]:
    """Detect a parse *explosion* — a scanned/garbage doc fragmented into far too
    many chunks (e.g. a 611-page scan -> 194k chunks) — before any embed/Qdrant
    spend.

    The chunks/page ratio alone false-positives on reflowable EPUB/MOBI, whose
    page numbers are synthetic and tiny: a valid 731-chunk book reads as
    ``122/page`` (VOR-1392). A real explosion ALSO fragments into abnormally
    small chunks, so we require BOTH a high chunks/page ratio AND a low mean
    tokens-per-chunk; a dense-but-legitimate book (normal chunk size) passes.
    Both thresholds are env-tunable (``VORAG_MAX_CHUNKS_PER_PAGE`` /
    ``VORAG_RUNAWAY_MIN_MEAN_TOKENS``). Never fires below the book-page floor
    (too few pages to judge a per-page rate). Returns ``(is_runaway, detail)``.
    """
    per_page_cap = _env_int(
        "VORAG_MAX_CHUNKS_PER_PAGE",
        RUNAWAY_MAX_CHUNKS_PER_PAGE_DEFAULT,
        minimum=1,
    )
    min_mean_tokens = _env_float(
        "VORAG_RUNAWAY_MIN_MEAN_TOKENS",
        RUNAWAY_MIN_MEAN_TOKENS_DEFAULT,
        minimum=0.0,
    )
    page_count = stats.effective_pages
    chunk_count = stats.chunk_count
    if not page_count or page_count < MIN_BOOK_PAGES_DEFAULT or chunk_count <= 0:
        return (False, f"{chunk_count} chunks over {page_count or 0} pages (below runaway floor)")

    per_page = chunk_count / page_count
    mean_tokens = stats.total_tokens / chunk_count
    is_runaway = per_page > per_page_cap and mean_tokens < min_mean_tokens
    detail = (
        f"{chunk_count} chunks over {page_count} pages "
        f"({per_page:.0f}/page > {per_page_cap}/page cap, "
        f"mean {mean_tokens:.0f} tok/chunk < {min_mean_tokens:.0f} floor)"
    )
    return (is_runaway, detail)


def is_degenerate_chunk_output(
    stats: ChunkOutputStats,
) -> tuple[bool, str]:
    min_pages = _env_int(
        "VO_RAG_BOOK_RAG_DEGENERATE_MIN_PAGES",
        DEGENERATE_MIN_PAGES_DEFAULT,
        minimum=1,
    )
    page_count = stats.effective_pages
    if page_count is None or page_count < min_pages:
        return (
            False,
            _degenerate_detail(
                stats=stats,
                min_chunks=1,
                min_total_tokens=0,
                min_pages=min_pages,
            ),
        )

    min_chunk_floor = _env_int(
        "VO_RAG_BOOK_RAG_DEGENERATE_MIN_CHUNKS",
        DEGENERATE_MIN_CHUNKS_DEFAULT,
        minimum=1,
    )
    min_chunks_per_page = _env_float(
        "VO_RAG_BOOK_RAG_DEGENERATE_MIN_CHUNKS_PER_PAGE",
        DEGENERATE_MIN_CHUNKS_PER_PAGE_DEFAULT,
        minimum=0.0,
    )
    min_tokens_per_page = _env_int(
        "VO_RAG_BOOK_RAG_DEGENERATE_MIN_TOKENS_PER_PAGE",
        DEGENERATE_MIN_TOKENS_PER_PAGE_DEFAULT,
        minimum=0,
    )
    min_chunks = max(min_chunk_floor, math.ceil(page_count * min_chunks_per_page))
    min_total_tokens = page_count * min_tokens_per_page
    is_degenerate = stats.chunk_count < min_chunks or stats.total_tokens < min_total_tokens
    return (
        is_degenerate,
        _degenerate_detail(
            stats=stats,
            min_chunks=min_chunks,
            min_total_tokens=min_total_tokens,
            min_pages=min_pages,
        ),
    )


def chunk_output_stats_from_payloads(payloads: list[dict[str, Any]]) -> ChunkOutputStats:
    section_paths: set[str] = set()
    page_numbers: list[int] = []
    total_pages_candidates: list[int] = []
    total_tokens = 0
    for payload in payloads:
        total_tokens += _payload_token_count(payload)
        section_path = payload.get("section_path")
        if section_path:
            section_paths.add(str(section_path))
        page_numbers.extend(_payload_page_numbers(payload))
        total_pages = _payload_total_pages(payload)
        if total_pages is not None:
            total_pages_candidates.append(total_pages)

    page_span = None
    if page_numbers:
        page_span = max(page_numbers) - min(page_numbers) + 1
    total_pages = max(total_pages_candidates) if total_pages_candidates else None
    return ChunkOutputStats(
        chunk_count=len(payloads),
        total_tokens=total_tokens,
        distinct_section_paths=len(section_paths),
        total_pages=total_pages,
        page_span=page_span,
    )


def _payload_token_count(payload: dict[str, Any]) -> int:
    token_count = _as_int(payload.get("token_count"))
    if token_count is not None:
        return token_count
    text = str(payload.get("embedding_text") or payload.get("content") or "")
    return max(0, len(text) // 4)


def _payload_page_numbers(payload: dict[str, Any]) -> list[int]:
    numbers: list[int] = []
    page_number = _as_int(payload.get("page_number"))
    if page_number is not None and page_number > 0:
        numbers.append(page_number)
    page_range = payload.get("page_range")
    if page_range is not None:
        numbers.extend(int(match.group(0)) for match in _PAGE_NUMBER_RE.finditer(str(page_range)))
    return [number for number in numbers if number > 0]


def _payload_total_pages(payload: dict[str, Any]) -> int | None:
    for value in (
        payload.get("total_pages"),
        _nested_payload_value(payload, "metadata", "book_rag_forward_path", "total_pages"),
        _nested_payload_value(payload, "metadata", "forward_path", "total_pages"),
        _nested_payload_value(payload, "book_rag_forward_path", "total_pages"),
        _nested_payload_value(payload, "forward_path", "total_pages"),
    ):
        total_pages = _as_int(value)
        if total_pages is not None and total_pages > 0:
            return total_pages
    return None


def metadata_field_completeness(
    payloads: list[dict[str, Any]], field: str
) -> MetadataFieldCompleteness:
    if not payloads:
        return MetadataFieldCompleteness(
            field=field,
            total_count=0,
            missing_count=0,
            dominant_value=_METADATA_MISSING_VALUE,
            dominant_count=0,
            dominant_missing=True,
        )

    counts: Counter[str] = Counter()
    for payload in payloads:
        counts[_metadata_value_key(field, _payload_contract_metadata_value(payload, field))] += 1

    dominant_value, dominant_count = counts.most_common(1)[0]
    missing_count = counts[_METADATA_MISSING_VALUE]
    return MetadataFieldCompleteness(
        field=field,
        total_count=len(payloads),
        missing_count=missing_count,
        dominant_value=dominant_value,
        dominant_count=dominant_count,
        dominant_missing=missing_count >= dominant_count,
    )


def _payload_contract_metadata_value(payload: dict[str, Any], field: str) -> Any:
    if field in payload:
        return payload.get(field)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get(field)
    return None


def _metadata_value_key(field: str, value: Any) -> str:
    if value is None:
        return _METADATA_MISSING_VALUE
    if isinstance(value, (list, tuple, set)):
        values = sorted(str(item).strip() for item in value if str(item).strip())
        return "|".join(values) if values else _METADATA_MISSING_VALUE
    text = str(value).strip()
    if not text:
        return _METADATA_MISSING_VALUE
    if field == "license_class" and text.lower() == _LICENSE_CLASS_UNKNOWN:
        return _METADATA_MISSING_VALUE
    return text


def _metadata_completeness_detail(
    *,
    payload_count: int,
    required_fields: tuple[str, ...],
    guard: str,
    missing_required: list[MetadataFieldCompleteness],
    warn_missing: list[MetadataFieldCompleteness],
) -> str:
    return (
        f"chunks={payload_count} guard={guard} "
        f"required={','.join(required_fields)} "
        f"missing_required={_metadata_statuses_text(missing_required)} "
        f"warn_missing={_metadata_statuses_text(warn_missing)}"
    )


def _metadata_statuses_text(statuses: list[MetadataFieldCompleteness]) -> str:
    if not statuses:
        return "none"
    return ";".join(
        f"{status.field}(missing={status.missing_count}/{status.total_count},"
        f"dominant={status.dominant_value}:{status.dominant_count})"
        for status in statuses
    )


def _nested_payload_value(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _degenerate_detail(
    *,
    stats: ChunkOutputStats,
    min_chunks: int,
    min_total_tokens: int,
    min_pages: int,
) -> str:
    page_text = str(stats.effective_pages) if stats.effective_pages is not None else "unknown"
    return (
        f"chunks={stats.chunk_count} min_chunks={min_chunks} "
        f"total_tokens={stats.total_tokens} min_total_tokens={min_total_tokens} "
        f"pages={page_text} min_pages={min_pages} "
        f"page_span={stats.page_span or 'unknown'} "
        f"sections={stats.distinct_section_paths}"
    )


async def _pool_fetchrow(pool: Any, query: str, *args: Any) -> Any:
    if hasattr(pool, "fetchrow"):
        return await pool.fetchrow(query, *args)
    async with pool.acquire() as conn:
        return await conn.fetchrow(query, *args)


async def _pool_fetchval(pool: Any, query: str, *args: Any) -> Any:
    if hasattr(pool, "fetchval"):
        return await pool.fetchval(query, *args)
    async with pool.acquire() as conn:
        return await conn.fetchval(query, *args)


async def _get_artifact_by_source(pool: Any, source_id: str) -> _ArtifactRecord | None:
    row = await _pool_fetchrow(
        pool,
        """
        SELECT id, title
        FROM knowledge_artifacts
        WHERE source_provenance @> jsonb_build_array(jsonb_build_object('source_id', $1::text))
        ORDER BY created_at DESC
        LIMIT 1
        """,
        source_id,
    )
    if row is None:
        return None
    return _ArtifactRecord(id=str(row["id"]), title=row.get("title"))


async def _get_artifact_by_id(pool: Any, artifact_id: str) -> _ArtifactRecord | None:
    row = await _pool_fetchrow(
        pool,
        """
        SELECT id, title
        FROM knowledge_artifacts
        WHERE id = $1
        LIMIT 1
        """,
        artifact_id,
    )
    if row is None:
        return None
    return _ArtifactRecord(id=str(row["id"]), title=row.get("title"))


async def _count_sub_chunks_for_source(pool: Any, source_id: str) -> int:
    table_exists = await _pool_fetchval(
        pool,
        "SELECT to_regclass('public.compliance_sub_chunks') IS NOT NULL",
    )
    if not table_exists:
        return 0

    count = await _pool_fetchval(
        pool,
        "SELECT COUNT(*) FROM compliance_sub_chunks WHERE source_id = $1",
        source_id,
    )
    return int(count or 0)


__all__ = [
    "BUCKET_ARTIFACTS",
    "DEGENERATE_MIN_CHUNKS_DEFAULT",
    "DEGENERATE_MIN_CHUNKS_PER_PAGE_DEFAULT",
    "DEGENERATE_MIN_PAGES_DEFAULT",
    "DEGENERATE_MIN_TOKENS_PER_PAGE_DEFAULT",
    "ChunkOutputStats",
    "IngestionValidator",
    "METADATA_REQUIRED_DEFAULT",
    "MIN_BOOK_PAGES_DEFAULT",
    "MetadataFieldCompleteness",
    "RAW_COLLECTION",
    "STERILE_COLLECTION",
    "ValidationCheck",
    "ValidationReport",
    "_ArtifactRecord",
    "_get_artifact_by_id",
    "_get_artifact_by_source",
    "chunk_output_stats_from_chunks",
    "chunk_output_stats_from_payloads",
    "is_degenerate_chunk_output",
    "metadata_field_completeness",
    "non_book_skip_by_pages",
]

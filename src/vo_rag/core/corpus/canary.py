"""Canary-run gates (VOR-1434): read-only pass/fail/unmeasured
verdicts over a pre-batch sample of already-ingested envelopes, plus a
simple wall-time projection for the full batch.

Every function here is pure or takes an already-open ``FetchPool``/Qdrant
client, so tests use fakes -- nothing here opens a live Postgres or Qdrant
connection itself (``cli/corpus_ops.py``'s job). Per-book data comes from
``intake_envelopes`` (``status``, ``updated_at`` against
``metrics.claimed_at_epoch_s``, and
``metadata->'quality'`` -- see ``core/corpus/health.book_quality_metadata``
for that shape) plus a read-only check of Qdrant's declared payload indexes.

A gate whose input isn't persisted
anywhere is reported ``unmeasured``, never faked as a pass, and any
``unmeasured`` gate forces the whole verdict to ``INCOMPLETE`` rather than
``PASS``.

Three gates read what the ingest persists per envelope under
``metadata->'metrics'`` (shape documented in
``core/observability/metrics.py``), recorded where the work happens:

- **stage wall time vs. budget** -- ``metrics.stages.<stage>.seconds``, timed
  around each ``BookRagPipeline`` step (``pipelines/book_rag/workflow.py``),
  against ``stage_budget_s(metrics.pages)``.
- **embedder 429 retry rate** -- ``metrics.embed.{calls,retries_429}``,
  counted by ``EmbeddingClient`` (``core/embedding/client.py``) per logical
  embed request; pooled over the sample.
- **Qdrant call p95** -- ``metrics.qdrant.p95_s``, from every
  ``QdrantStore`` call's latency (``core/storage/qdrant.py``); the worst
  book's p95 is the gate's figure.

Each is still ``unmeasured`` for any sampled envelope that lacks its data
(ingested before the metrics were recorded, or the best-effort recording failed).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from vo_rag.core.corpus.source_format import FetchPool
from vo_rag.core.storage.qdrant import (
    COLLECTION_NAMES,
    COMMON_PAYLOAD_INDEXES,
    _extra_indexes_for,
)

GateStatus = Literal["pass", "fail", "unmeasured"]
Verdict = Literal["PASS", "FAIL", "INCOMPLETE"]

FAILED_STATUS = "failed"
COMPLETED_STATUS = "completed"

#: The stage-time-budget formula: no stage may exceed
#: max(15 minutes, pages * 6 seconds). Exposed so alerting and the canary run
#: apply exactly this formula, not a re-derived one.
STAGE_BUDGET_FLOOR_S = 15 * 60.0
STAGE_BUDGET_PER_PAGE_S = 6.0

#: Gate thresholds.
MAX_EMBED_429_RATE = 0.01
MAX_QDRANT_P95_S = 1.0


def stage_budget_s(pages: int | None) -> float:
    """``max(15 min, pages * 6s)``. Unknown ``pages`` -> just the floor."""
    return max(STAGE_BUDGET_FLOOR_S, (pages or 0) * STAGE_BUDGET_PER_PAGE_S)


@dataclass(frozen=True)
class GateResult:
    name: str
    status: GateStatus
    detail: str


@dataclass(frozen=True)
class CanaryBook:
    """One envelope's canary row. ``findings is None`` means the
    completion hook never recorded ``metadata->'quality'`` for this
    envelope at all (not completed yet, or the best-effort check swallowed
    an error) -- distinct from ``findings == ()``, a clean book."""

    envelope_id: str
    status: str
    source_id: str | None
    title: str | None
    error: str | None
    wall_time_s: float | None
    pages: int | None
    chunks: int | None
    pages_per_min: float | None
    findings: tuple[str, ...] | None
    # From metadata->'metrics'; None when never recorded.
    ingest_pages: int | None = None
    stage_seconds: dict[str, float] | None = None
    embed_calls: int | None = None
    embed_retries_429: int | None = None
    qdrant_p95_s: float | None = None


@dataclass(frozen=True)
class Projection:
    total_pages: int
    pages_per_min_per_lane: float
    lane_count: int
    projected_wall_time_s: float
    formula: str
    lane_count_note: str


@dataclass(frozen=True)
class CanaryReport:
    books: tuple[CanaryBook, ...]
    gates: tuple[GateResult, ...]
    verdict: Verdict
    projection: Projection | None

    def ok(self) -> bool:
        return self.verdict == "PASS"


# ---------------------------------------------------------------------------
# Postgres read
# ---------------------------------------------------------------------------

ENVELOPE_CANARY_SQL = """
    SELECT id, status, error, created_at, updated_at, metadata
    FROM intake_envelopes
    WHERE id = ANY($1::text[])
"""


async def fetch_canary_envelopes(
    pool: FetchPool, envelope_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """Raw ``intake_envelopes`` rows for ``envelope_ids``, in whatever order
    the database returns them (``_book_from_row``/callers don't rely on
    order). A thin wrapper so the pure logic below never imports asyncpg
    itself -- tests pass a ``FetchPool`` fake."""
    if not envelope_ids:
        return []
    rows = await pool.fetch(ENVELOPE_CANARY_SQL, list(envelope_ids))
    return [dict(row) for row in rows]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _preflight_pages(metadata: Any) -> float | None:
    """``metadata->'preflight'->'pages'`` (written at intake): a
    number or a digit string; anything else is unknown."""
    value = _as_dict(_as_dict(metadata).get("preflight")).get("pages")
    if isinstance(value, str) and value.isdigit():
        return float(value)
    return _number(value)


def _ingest_metrics(metadata: Any) -> dict[str, Any]:
    """The ``CanaryBook`` fields read from ``metadata->'metrics'``; a missing
    or malformed value reads as ``None`` (unmeasured), never as zero."""
    metrics = _as_dict(_as_dict(metadata).get("metrics"))

    stages_raw = metrics.get("stages")
    stage_seconds: dict[str, float] | None = None
    if isinstance(stages_raw, dict) and stages_raw:
        stage_seconds = {}
        for name, entry in stages_raw.items():
            seconds = _number(entry.get("seconds")) if isinstance(entry, dict) else None
            if seconds is not None:
                stage_seconds[str(name)] = seconds
        stage_seconds = stage_seconds or None

    embed = _as_dict(metrics.get("embed"))
    calls = _number(embed.get("calls"))
    retries = _number(embed.get("retries_429"))
    if calls is None or retries is None:  # both or neither
        calls = retries = None

    qdrant = _as_dict(metrics.get("qdrant"))
    pages = _number(metrics.get("pages")) or _preflight_pages(metadata)
    return {
        "ingest_pages": int(pages) if pages else None,
        "stage_seconds": stage_seconds,
        "embed_calls": None if calls is None else int(calls),
        "embed_retries_429": None if retries is None else int(retries),
        "qdrant_p95_s": _number(qdrant.get("p95_s")),
    }


def _wall_time_s(row: dict[str, Any], *, include_queue_wait: bool) -> float | None:
    """A completed book's lane time: ``metadata->'metrics'->'claimed_at_epoch_s'``
    (stamped by the claim that ran it) to ``updated_at``. ``created_at`` also
    counts the time the envelope sat in the queue, so it is used only when
    the claim was never recorded AND the caller opted in with
    ``include_queue_wait``; otherwise ``None``."""
    updated_at = row.get("updated_at")
    if row.get("status") != COMPLETED_STATUS or not isinstance(updated_at, datetime):
        return None
    claimed = _number(
        _as_dict(_as_dict(row.get("metadata")).get("metrics")).get("claimed_at_epoch_s")
    )
    if claimed is not None:
        return updated_at.timestamp() - claimed
    created_at = row.get("created_at")
    if include_queue_wait and isinstance(created_at, datetime):
        return (updated_at - created_at).total_seconds()
    return None


def _book_from_row(row: dict[str, Any], *, include_queue_wait: bool = False) -> CanaryBook:
    metadata = row.get("metadata") or {}
    quality = metadata.get("quality") if isinstance(metadata, dict) else None
    stats: dict[str, Any] = (quality or {}).get("stats") or {}
    findings: tuple[str, ...] | None = (
        tuple(quality["findings"]) if quality is not None and "findings" in quality else None
    )

    wall_time_s = _wall_time_s(row, include_queue_wait=include_queue_wait)

    pages = stats.get("max_page")
    chunks = stats.get("chunks")
    pages_per_min = (
        pages / (wall_time_s / 60.0) if pages and wall_time_s and wall_time_s > 0 else None
    )

    ingest = _ingest_metrics(metadata)
    return CanaryBook(
        **ingest,
        envelope_id=str(row["id"]),
        status=str(row.get("status")),
        source_id=stats.get("source_id"),
        title=stats.get("source_book"),
        error=row.get("error"),
        wall_time_s=wall_time_s,
        pages=pages,
        chunks=chunks,
        pages_per_min=pages_per_min,
        findings=findings,
    )


# ---------------------------------------------------------------------------
# Qdrant read-only index check
# ---------------------------------------------------------------------------


#: ``missing_payload_indexes``' entry for a raw-collection override that
#: doesn't exist at all.
OVERRIDE_COLLECTION_MISSING = "<collection missing>"


def _raw_collection_override() -> str | None:
    """The production ``VORAG_RAW_COLLECTION`` reading (lazy import: the
    workflow module pulls in the whole ingest pipeline)."""
    from vo_rag.pipelines.book_rag.workflow import _raw_collection_override as override

    return override()


async def missing_payload_indexes(client: Any) -> dict[str, list[str]]:
    """Read-only twin of ``qdrant.reconcile_payload_indexes``: same
    declarations (``COLLECTION_NAMES`` / ``COMMON_PAYLOAD_INDEXES`` /
    ``_extra_indexes_for``), same "collection must already exist" guard, but
    never calls ``create_payload_index`` -- it only reports what's missing,
    so a canary run before the batch can never mutate a live collection.
    """
    present = {c.name for c in (await client.get_collections()).collections}
    missing: dict[str, list[str]] = {}
    for name in COLLECTION_NAMES:
        if name not in present:
            continue
        info = await client.get_collection(name)
        have = set((info.payload_schema or {}).keys())
        declared = {**COMMON_PAYLOAD_INDEXES, **_extra_indexes_for(name)}
        gaps = [field for field in declared if field not in have]
        if gaps:
            missing[name] = gaps

    # A batch ingested with VORAG_RAW_COLLECTION set writes
    # its raw chunks there, so that collection is gated too -- on the raw
    # book declarations it stands in for. It must exist: the canary's books
    # were written to it.
    override = _raw_collection_override()
    if override and override not in COLLECTION_NAMES:
        if override not in present:
            missing[override] = [OVERRIDE_COLLECTION_MISSING]
        else:
            info = await client.get_collection(override)
            have = set((info.payload_schema or {}).keys())
            declared = {**COMMON_PAYLOAD_INDEXES, **_extra_indexes_for("vo_rag_books_raw_v1")}
            gaps = [field for field in declared if field not in have]
            if gaps:
                missing[override] = gaps
    return missing


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def _failed_envelopes_gate(books: Sequence[CanaryBook]) -> GateResult:
    failed = sorted(b.envelope_id for b in books if b.status == FAILED_STATUS)
    if failed:
        return GateResult("failed_envelopes", "fail", f"{len(failed)} failed: {', '.join(failed)}")
    return GateResult("failed_envelopes", "pass", "0 failed")


def _stage_budget_gate(books: Sequence[CanaryBook]) -> GateResult:
    """No stage over ``stage_budget_s(pages)`` (``metrics.pages``, else
    ``preflight.pages``; unknown pages -> the 15-minute floor). A completed
    book must have a time for every stage in ``INGEST_STEP_ORDER``; one that
    lacks any is unmeasured, never a pass."""
    # Lazy import: `events.py` builds pydantic models; importing it at module
    # level would make every CLI command pay for them, not just canary-report.
    from vo_rag.pipelines.book_rag.events import INGEST_STEP_ORDER

    unrecorded: list[str] = []
    for book in sorted(books, key=lambda b: b.envelope_id):
        if book.stage_seconds is None:
            unrecorded.append(book.envelope_id)
        elif book.status == COMPLETED_STATUS:
            missing = [s for s in INGEST_STEP_ORDER if s not in book.stage_seconds]
            if missing:
                unrecorded.append(f"{book.envelope_id} (missing {'/'.join(missing)})")
    over: list[str] = []
    for book in sorted(books, key=lambda b: b.envelope_id):
        budget = stage_budget_s(book.ingest_pages)
        for stage, seconds in sorted((book.stage_seconds or {}).items()):
            if seconds > budget:
                over.append(f"{book.envelope_id} {stage} {seconds:.0f}s > {budget:.0f}s")
    if over:
        return GateResult("stage_budget", "fail", "; ".join(over))
    if unrecorded or not books:
        return GateResult(
            "stage_budget",
            "unmeasured",
            f"{len(unrecorded)} envelope(s) lack metadata->'metrics'->'stages' "
            f"timings (ingested before stage timing was recorded, or recording failed): "
            f"{', '.join(unrecorded)}",
        )
    return GateResult("stage_budget", "pass", "every stage within max(15 min, pages x 6 s)")


def _embed_429_gate(books: Sequence[CanaryBook]) -> GateResult:
    """Embedder 429 retries < 1% of embed calls, pooled over the sample."""
    unrecorded = sorted(
        b.envelope_id for b in books if b.embed_calls is None or b.embed_retries_429 is None
    )
    if unrecorded or not books:
        return GateResult(
            "embed_429_rate",
            "unmeasured",
            f"{len(unrecorded)} envelope(s) have no metadata->'metrics'->'embed' "
            f"counts: {', '.join(unrecorded)}",
        )
    calls = sum(b.embed_calls or 0 for b in books)
    retries = sum(b.embed_retries_429 or 0 for b in books)
    if calls <= 0:
        return GateResult("embed_429_rate", "unmeasured", "no embed calls recorded in the sample")
    rate = retries / calls
    detail = f"{retries}/{calls} = {rate:.2%} (limit < {MAX_EMBED_429_RATE:.0%})"
    return GateResult("embed_429_rate", "pass" if rate < MAX_EMBED_429_RATE else "fail", detail)


def _qdrant_p95_gate(books: Sequence[CanaryBook]) -> GateResult:
    """Qdrant call p95 < 1 s; each book's own p95, the worst one decides."""
    slow = sorted(
        f"{b.envelope_id} {b.qdrant_p95_s:.3f}s"
        for b in books
        if b.qdrant_p95_s is not None and b.qdrant_p95_s >= MAX_QDRANT_P95_S
    )
    if slow:
        return GateResult(
            "qdrant_p95", "fail", f"p95 >= {MAX_QDRANT_P95_S:.0f}s: {'; '.join(slow)}"
        )
    unrecorded = sorted(b.envelope_id for b in books if b.qdrant_p95_s is None)
    if unrecorded or not books:
        return GateResult(
            "qdrant_p95",
            "unmeasured",
            f"{len(unrecorded)} envelope(s) have no metadata->'metrics'->'qdrant' "
            f"latency: {', '.join(unrecorded)}",
        )
    worst = max(b.qdrant_p95_s or 0.0 for b in books)
    return GateResult("qdrant_p95", "pass", f"worst book p95 {worst:.3f}s")


async def _payload_indexes_gate(client: Any) -> GateResult:
    missing = await missing_payload_indexes(client)
    if missing:
        detail = "; ".join(f"{name}: {fields}" for name, fields in sorted(missing.items()))
        return GateResult("payload_indexes", "fail", detail)
    return GateResult("payload_indexes", "pass", "nothing missing")


def _quality_gate(books: Sequence[CanaryBook]) -> GateResult:
    unrecorded = sorted(b.envelope_id for b in books if b.findings is None)
    failing = sorted(b.envelope_id for b in books if b.findings)
    if failing:
        return GateResult(
            "quality", "fail", f"{len(failing)} book(s) with findings: {', '.join(failing)}"
        )
    if unrecorded:
        return GateResult(
            "quality",
            "unmeasured",
            f"{len(unrecorded)} envelope(s) have no metadata->'quality' yet "
            f"(not completed, or the best-effort check never ran): {', '.join(unrecorded)}",
        )
    return GateResult("quality", "pass", "every book clean")


def _verdict(gates: Sequence[GateResult]) -> Verdict:
    if any(g.status == "fail" for g in gates):
        return "FAIL"
    if any(g.status == "unmeasured" for g in gates):
        return "INCOMPLETE"
    return "PASS"


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def measured_pages_per_min_per_lane(books: Sequence[CanaryBook]) -> float | None:
    """The canary run's measured throughput per lane: total pages across
    every book with both a page count and a measured wall time, divided by
    the total of those wall times. Each book's wall time is the time one
    lane spent on it (claim to completion), so the sum is lane-minutes and
    the result is already per lane -- dividing by the lane count here too
    would count the lanes twice. ``None`` when nothing in
    the sample has both figures (e.g. every book is still processing)."""
    measured = [b for b in books if b.pages and b.wall_time_s and b.wall_time_s > 0]
    if not measured:
        return None
    total_pages = sum(b.pages or 0 for b in measured)
    total_minutes = sum((b.wall_time_s or 0.0) / 60.0 for b in measured)
    if total_minutes <= 0:
        return None
    return total_pages / total_minutes


def project_batch(
    *, total_pages: int, pages_per_min_per_lane: float, lane_count: int
) -> Projection:
    """Projected wall time = total
    batch pages / measured pages-per-minute per lane / lane count.

    The lane-count "suggestion" is left at the input ``lane_count``: the
    ``embed_429_rate`` gate measures whether the embedder kept up at *this*
    lane count, not how it would behave at a higher one, so raising it here
    would be a guess -- re-run the canary at the higher count instead.
    """
    if lane_count <= 0:
        raise ValueError("lane_count must be >= 1")
    if pages_per_min_per_lane <= 0:
        raise ValueError("pages_per_min_per_lane must be > 0")
    minutes = total_pages / pages_per_min_per_lane / lane_count
    return Projection(
        total_pages=total_pages,
        pages_per_min_per_lane=pages_per_min_per_lane,
        lane_count=lane_count,
        projected_wall_time_s=minutes * 60.0,
        formula="projected_wall_time = total_pages / pages_per_min_per_lane / lane_count",
        lane_count_note=(
            f"lane_count is left unchanged at {lane_count}: raise it only while the "
            "embed_429_rate gate (embedder 429 retries vs embed calls, pooled over "
            "this canary) stays under its limit, and re-run the canary at the new count"
        ),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def run_canary_report(
    pool: FetchPool,
    qdrant_client: Any,
    envelope_ids: Sequence[str],
    *,
    total_pages: int | None = None,
    lane_count: int = 1,
    include_queue_wait: bool = False,
) -> CanaryReport:
    """Read every gate's evidence and build the full report. Read-only:
    never writes to Postgres or Qdrant (the payload-index check only reads
    collection schemas). ``include_queue_wait`` lets a book with no recorded
    claim time fall back to ``created_at`` (queue wait included)."""
    rows = await fetch_canary_envelopes(pool, envelope_ids)
    books = tuple(_book_from_row(row, include_queue_wait=include_queue_wait) for row in rows)

    gates = (
        _failed_envelopes_gate(books),
        _stage_budget_gate(books),
        _embed_429_gate(books),
        _qdrant_p95_gate(books),
        await _payload_indexes_gate(qdrant_client),
        _quality_gate(books),
    )

    projection: Projection | None = None
    if total_pages is not None:
        pages_per_min = measured_pages_per_min_per_lane(books)
        if pages_per_min:
            projection = project_batch(
                total_pages=total_pages,
                pages_per_min_per_lane=pages_per_min,
                lane_count=lane_count,
            )

    return CanaryReport(books=books, gates=gates, verdict=_verdict(gates), projection=projection)


CANARY_GATE_NAMES = (
    "failed_envelopes",
    "stage_budget",
    "embed_429_rate",
    "qdrant_p95",
    "payload_indexes",
    "quality",
)

__all__ = [
    "CANARY_GATE_NAMES",
    "CanaryBook",
    "CanaryReport",
    "GateResult",
    "MAX_EMBED_429_RATE",
    "MAX_QDRANT_P95_S",
    "Projection",
    "STAGE_BUDGET_FLOOR_S",
    "STAGE_BUDGET_PER_PAGE_S",
    "fetch_canary_envelopes",
    "measured_pages_per_min_per_lane",
    "missing_payload_indexes",
    "project_batch",
    "run_canary_report",
    "stage_budget_s",
]

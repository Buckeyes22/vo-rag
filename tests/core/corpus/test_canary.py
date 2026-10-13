"""Pure unit tests for `vo_rag.core.corpus.canary` (VOR-1434 task 3).

No live Postgres/Qdrant: `_FakePool`/`_FakeQdrantClient` stand in for the
real `FetchPool`/`AsyncQdrantClient`, matching the fake-pool convention in
`tests/cli/test_corpus_duplicates.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from vo_rag.core.corpus import canary


@pytest.fixture(autouse=True)
def _no_raw_collection_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VORAG_RAW_COLLECTION", raising=False)


def _dt(seconds: float) -> datetime:
    return datetime(2026, 9, 27, tzinfo=UTC) + timedelta(seconds=seconds)


#: A sentinel distinct from ``None`` -- ``quality=None`` means "no
#: metadata->'quality' recorded at all"; the default means "use the clean,
#: fully-recorded book" below.
_UNSET: dict[str, Any] = {}


def _row(
    envelope_id: str,
    *,
    status: str = "completed",
    error: str | None = None,
    created_s: float = 0.0,
    updated_s: float = 600.0,
    claimed_s: float | None = None,
    quality: dict[str, Any] | None = _UNSET,
) -> dict[str, Any]:
    """``claimed_s`` (default: ``created_s``) is when a lane claimed the
    envelope -- ``metadata->'metrics'->'claimed_at_epoch_s'``; ``None`` via
    ``_row_unclaimed`` leaves it unrecorded."""
    metadata: dict[str, Any] = {
        "metrics": {
            "claimed_at_epoch_s": _dt(created_s if claimed_s is None else claimed_s).timestamp()
        }
    }
    if quality is _UNSET:
        # default: a clean, fully-recorded book
        metadata["quality"] = {
            "findings": [],
            "stats": {
                "source_id": f"src-{envelope_id}",
                "source_book": "Book",
                "chunks": 42,
                "max_page": 100,
            },
        }
    elif quality is not None:
        metadata["quality"] = quality
    return {
        "id": envelope_id,
        "status": status,
        "error": error,
        "created_at": _dt(created_s),
        "updated_at": _dt(updated_s),
        "metadata": metadata,
    }


class _FakePool:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, Any]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.calls.append((query, args))
        return self.rows


class _FakeQdrantClient:
    """`payload_schema_by_collection` maps a present collection name to its
    payload schema keys; a collection not in the dict does not exist."""

    def __init__(self, payload_schema_by_collection: dict[str, set[str]]) -> None:
        self._schemas = payload_schema_by_collection

    async def get_collections(self) -> Any:
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self._schemas])

    async def get_collection(self, name: str) -> Any:
        return SimpleNamespace(payload_schema=dict.fromkeys(self._schemas[name]))


def _full_schema_client() -> _FakeQdrantClient:
    """Every declared index present on `vo_rag_books_raw_v1` -- the
    payload-indexes gate should report clean."""
    from vo_rag.core.storage.qdrant import COMMON_PAYLOAD_INDEXES, _extra_indexes_for

    fields = set(COMMON_PAYLOAD_INDEXES) | set(_extra_indexes_for("vo_rag_books_raw_v1"))
    return _FakeQdrantClient({"vo_rag_books_raw_v1": fields})


# ---------------------------------------------------------------------------
# stage_budget_s
# ---------------------------------------------------------------------------


def test_stage_budget_floor_for_small_page_counts() -> None:
    assert canary.stage_budget_s(10) == 15 * 60.0


def test_stage_budget_scales_past_the_floor() -> None:
    # 1000 pages * 6s = 6000s > the 900s floor.
    assert canary.stage_budget_s(1000) == 6000.0


def test_stage_budget_none_pages_is_the_floor() -> None:
    assert canary.stage_budget_s(None) == 15 * 60.0


# ---------------------------------------------------------------------------
# fetch_canary_envelopes
# ---------------------------------------------------------------------------


async def test_fetch_canary_envelopes_empty_ids_skips_the_query() -> None:
    pool = _FakePool(rows=[])
    result = await canary.fetch_canary_envelopes(pool, [])
    assert result == []
    assert pool.calls == []


async def test_fetch_canary_envelopes_passes_ids_through() -> None:
    pool = _FakePool(rows=[_row("env-1")])
    await canary.fetch_canary_envelopes(pool, ["env-1"])
    [(query, args)] = pool.calls
    assert "intake_envelopes" in query
    assert args == (["env-1"],)


# ---------------------------------------------------------------------------
# _book_from_row (via run_canary_report, the public surface)
# ---------------------------------------------------------------------------


async def test_completed_book_gets_wall_time_and_pages_per_min() -> None:
    pool = _FakePool(rows=[_row("env-1", created_s=0.0, updated_s=600.0)])
    report = await canary.run_canary_report(pool, _full_schema_client(), ["env-1"])
    [book] = report.books
    assert book.wall_time_s == 600.0
    assert book.pages == 100
    assert book.chunks == 42
    # 100 pages / (600s / 60) = 10 pages/min
    assert book.pages_per_min == 10.0
    assert book.findings == ()


async def test_wall_time_runs_from_the_claim_not_from_intake() -> None:
    """Queue wait (created_at -> claim) is not lane time."""
    pool = _FakePool(rows=[_row("env-1", created_s=0.0, claimed_s=3000.0, updated_s=3600.0)])
    report = await canary.run_canary_report(pool, _full_schema_client(), ["env-1"])
    [book] = report.books
    assert book.wall_time_s == 600.0
    assert book.pages_per_min == 10.0


async def test_unclaimed_book_has_no_wall_time_unless_queue_wait_is_allowed() -> None:
    row = _row("env-1", created_s=0.0, updated_s=600.0)
    del row["metadata"]["metrics"]["claimed_at_epoch_s"]

    strict = await canary.run_canary_report(_FakePool(rows=[row]), _full_schema_client(), ["env-1"])
    assert strict.books[0].wall_time_s is None

    lenient = await canary.run_canary_report(
        _FakePool(rows=[row]), _full_schema_client(), ["env-1"], include_queue_wait=True
    )
    assert lenient.books[0].wall_time_s == 600.0


async def test_projection_scales_as_one_over_lanes() -> None:
    """Doubling the lanes halves the projected wall time."""
    rows = [_row("env-1", created_s=0.0, updated_s=600.0)]
    one = await canary.run_canary_report(
        _FakePool(rows=rows), _full_schema_client(), ["env-1"], total_pages=1000, lane_count=1
    )
    four = await canary.run_canary_report(
        _FakePool(rows=rows), _full_schema_client(), ["env-1"], total_pages=1000, lane_count=4
    )
    assert one.projection is not None and four.projection is not None
    # 100 pages in 10 lane-minutes = 10 pages/min/lane; 1000 pages / 10 = 100 min.
    assert one.projection.pages_per_min_per_lane == pytest.approx(10.0)
    assert four.projection.pages_per_min_per_lane == pytest.approx(10.0)
    assert one.projection.projected_wall_time_s == pytest.approx(6000.0)
    assert four.projection.projected_wall_time_s == pytest.approx(1500.0)


async def test_non_completed_book_has_no_wall_time() -> None:
    pool = _FakePool(rows=[_row("env-1", status="processing", quality=None)])
    report = await canary.run_canary_report(pool, _full_schema_client(), ["env-1"])
    [book] = report.books
    assert book.wall_time_s is None
    assert book.pages_per_min is None
    assert book.findings is None


async def test_missing_quality_metadata_is_none_not_empty() -> None:
    pool = _FakePool(rows=[_row("env-1", quality=None)])
    report = await canary.run_canary_report(pool, _full_schema_client(), ["env-1"])
    [book] = report.books
    assert book.findings is None


# ---------------------------------------------------------------------------
# failed_envelopes gate
# ---------------------------------------------------------------------------


async def test_failed_envelope_fails_the_gate_and_the_verdict() -> None:
    pool = _FakePool(rows=[_row("env-1", status="failed", error="boom", quality=None)])
    report = await canary.run_canary_report(pool, _full_schema_client(), ["env-1"])
    gate = next(g for g in report.gates if g.name == "failed_envelopes")
    assert gate.status == "fail"
    assert "env-1" in gate.detail
    assert report.verdict == "FAIL"


async def test_zero_failed_envelopes_passes_that_gate() -> None:
    pool = _FakePool(rows=[_row("env-1")])
    report = await canary.run_canary_report(pool, _full_schema_client(), ["env-1"])
    gate = next(g for g in report.gates if g.name == "failed_envelopes")
    assert gate.status == "pass"


# ---------------------------------------------------------------------------
# quality gate
# ---------------------------------------------------------------------------


async def test_book_with_findings_fails_the_quality_gate() -> None:
    pool = _FakePool(
        rows=[
            _row(
                "env-1",
                quality={
                    "findings": ["empty_figures"],
                    "stats": {"source_id": "s", "chunks": 1, "max_page": 5},
                },
            )
        ]
    )
    report = await canary.run_canary_report(pool, _full_schema_client(), ["env-1"])
    gate = next(g for g in report.gates if g.name == "quality")
    assert gate.status == "fail"
    assert "env-1" in gate.detail
    assert report.verdict == "FAIL"


async def test_book_missing_quality_metadata_makes_the_gate_unmeasured() -> None:
    pool = _FakePool(rows=[_row("env-1", status="processing", quality=None)])
    report = await canary.run_canary_report(pool, _full_schema_client(), ["env-1"])
    gate = next(g for g in report.gates if g.name == "quality")
    assert gate.status == "unmeasured"


async def test_every_book_clean_passes_the_quality_gate() -> None:
    pool = _FakePool(rows=[_row("env-1"), _row("env-2")])
    report = await canary.run_canary_report(pool, _full_schema_client(), ["env-1", "env-2"])
    gate = next(g for g in report.gates if g.name == "quality")
    assert gate.status == "pass"


# ---------------------------------------------------------------------------
# payload_indexes gate / missing_payload_indexes
# ---------------------------------------------------------------------------


async def test_missing_payload_indexes_reports_the_gap() -> None:
    client = _FakeQdrantClient({"vo_rag_books_raw_v1": {"project_id", "source_id"}})
    missing = await canary.missing_payload_indexes(client)
    assert "vo_rag_books_raw_v1" in missing
    assert "author" in missing["vo_rag_books_raw_v1"]


async def test_missing_payload_indexes_never_creates_anything() -> None:
    """The fake has no `create_payload_index` method at all -- if the gate
    ever tried to mutate, this would raise AttributeError."""
    client = _FakeQdrantClient({"vo_rag_books_raw_v1": {"project_id"}})
    await canary.missing_payload_indexes(client)  # must not raise


async def test_missing_payload_indexes_skips_absent_collections() -> None:
    client = _FakeQdrantClient({})
    missing = await canary.missing_payload_indexes(client)
    assert missing == {}


async def test_missing_payload_indexes_checks_the_raw_collection_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch written to ``VORAG_RAW_COLLECTION`` is gated
    on that collection's indexes, not only the canonical ones."""
    from vo_rag.core.storage.qdrant import COMMON_PAYLOAD_INDEXES, _extra_indexes_for

    monkeypatch.setenv("VORAG_RAW_COLLECTION", "vo_rag_books_raw_canary")
    full = set(COMMON_PAYLOAD_INDEXES) | set(_extra_indexes_for("vo_rag_books_raw_v1"))
    client = _FakeQdrantClient(
        {"vo_rag_books_raw_v1": full, "vo_rag_books_raw_canary": {"project_id"}}
    )
    missing = await canary.missing_payload_indexes(client)
    assert list(missing) == ["vo_rag_books_raw_canary"]
    assert "author" in missing["vo_rag_books_raw_canary"]


async def test_missing_payload_indexes_flags_an_absent_override_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VORAG_RAW_COLLECTION", "vo_rag_books_raw_canary")
    missing = await canary.missing_payload_indexes(_full_schema_client())
    assert missing == {"vo_rag_books_raw_canary": ["<collection missing>"]}


async def test_full_schema_reports_nothing_missing() -> None:
    missing = await canary.missing_payload_indexes(_full_schema_client())
    assert missing == {}


async def test_missing_indexes_fails_the_gate_and_verdict() -> None:
    pool = _FakePool(rows=[_row("env-1")])
    client = _FakeQdrantClient({"vo_rag_books_raw_v1": {"project_id"}})
    report = await canary.run_canary_report(pool, client, ["env-1"])
    gate = next(g for g in report.gates if g.name == "payload_indexes")
    assert gate.status == "fail"
    assert report.verdict == "FAIL"


# ---------------------------------------------------------------------------
# unmeasured gates -> INCOMPLETE, never PASS
# ---------------------------------------------------------------------------


async def test_unmeasured_gates_are_always_present() -> None:
    pool = _FakePool(rows=[_row("env-1")])
    report = await canary.run_canary_report(pool, _full_schema_client(), ["env-1"])
    unmeasured_names = {g.name for g in report.gates if g.status == "unmeasured"}
    assert unmeasured_names == {"stage_budget", "embed_429_rate", "qdrant_p95"}


async def test_a_clean_canary_run_is_incomplete_never_pass() -> None:
    """Every measurable gate clean, but three gates have no persisted input
    at all -- the verdict must never lie and say PASS."""
    pool = _FakePool(rows=[_row("env-1"), _row("env-2")])
    report = await canary.run_canary_report(pool, _full_schema_client(), ["env-1", "env-2"])
    assert report.verdict == "INCOMPLETE"
    assert report.ok() is False


async def test_fail_outranks_incomplete() -> None:
    pool = _FakePool(rows=[_row("env-1", status="failed", quality=None)])
    report = await canary.run_canary_report(pool, _full_schema_client(), ["env-1"])
    assert report.verdict == "FAIL"


def test_verdict_names_cover_all_six_brief_gates() -> None:
    assert canary.CANARY_GATE_NAMES == (
        "failed_envelopes",
        "stage_budget",
        "embed_429_rate",
        "qdrant_p95",
        "payload_indexes",
        "quality",
    )


async def test_gate_order_matches_canary_gate_names() -> None:
    pool = _FakePool(rows=[_row("env-1")])
    report = await canary.run_canary_report(pool, _full_schema_client(), ["env-1"])
    assert tuple(g.name for g in report.gates) == canary.CANARY_GATE_NAMES


# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------


def test_project_batch_formula() -> None:
    projection = canary.project_batch(total_pages=10_000, pages_per_min_per_lane=50.0, lane_count=4)
    # 10000 / 50 / 4 = 50 minutes = 3000s
    assert projection.projected_wall_time_s == pytest.approx(3000.0)
    assert "total_pages" in projection.formula


def test_project_batch_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        canary.project_batch(total_pages=100, pages_per_min_per_lane=0.0, lane_count=1)
    with pytest.raises(ValueError):
        canary.project_batch(total_pages=100, pages_per_min_per_lane=10.0, lane_count=0)


def test_measured_pages_per_min_per_lane_aggregates_and_divides_by_lanes() -> None:
    books = (
        canary.CanaryBook(
            envelope_id="e1",
            status="completed",
            source_id="s1",
            title="t1",
            error=None,
            wall_time_s=600.0,
            pages=100,
            chunks=10,
            pages_per_min=10.0,
            findings=(),
        ),
        canary.CanaryBook(
            envelope_id="e2",
            status="completed",
            source_id="s2",
            title="t2",
            error=None,
            wall_time_s=600.0,
            pages=200,
            chunks=20,
            pages_per_min=20.0,
            findings=(),
        ),
    )
    # Each book's wall time is one lane's time on it: 300 pages over 20
    # lane-minutes = 15 pages/min per lane, whatever the lane count.
    result = canary.measured_pages_per_min_per_lane(books)
    assert result == pytest.approx(15.0)


def test_measured_pages_per_min_per_lane_none_when_nothing_measurable() -> None:
    books = (
        canary.CanaryBook(
            envelope_id="e1",
            status="processing",
            source_id=None,
            title=None,
            error=None,
            wall_time_s=None,
            pages=None,
            chunks=None,
            pages_per_min=None,
            findings=None,
        ),
    )
    assert canary.measured_pages_per_min_per_lane(books) is None


async def test_run_canary_report_includes_projection_when_measurable() -> None:
    pool = _FakePool(rows=[_row("env-1", created_s=0.0, updated_s=600.0)])
    report = await canary.run_canary_report(
        pool, _full_schema_client(), ["env-1"], total_pages=1000, lane_count=1
    )
    assert report.projection is not None
    assert report.projection.total_pages == 1000


async def test_run_canary_report_omits_projection_without_total_pages() -> None:
    pool = _FakePool(rows=[_row("env-1")])
    report = await canary.run_canary_report(pool, _full_schema_client(), ["env-1"])
    assert report.projection is None


async def test_run_canary_report_omits_projection_when_nothing_measurable() -> None:
    pool = _FakePool(rows=[_row("env-1", status="processing", quality=None)])
    report = await canary.run_canary_report(
        pool, _full_schema_client(), ["env-1"], total_pages=1000, lane_count=1
    )
    assert report.projection is None


# ---------------------------------------------------------------------------
# The three gates fed by the persisted metadata->'metrics'
# ---------------------------------------------------------------------------


def _metrics(
    *,
    pages: int | None = 100,
    stages: dict[str, float] | None = None,
    embed: tuple[int, int] | None = (1000, 0),
    qdrant_p95_s: float | None = 0.2,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if pages is not None:
        metrics["pages"] = pages
    if stages is None:
        from vo_rag.pipelines.book_rag.events import INGEST_STEP_ORDER

        stages = dict.fromkeys(INGEST_STEP_ORDER, 60.0)
    metrics["stages"] = {name: {"seconds": seconds} for name, seconds in stages.items()}
    if embed is not None:
        metrics["embed"] = {"calls": embed[0], "retries_429": embed[1]}
    if qdrant_p95_s is not None:
        metrics["qdrant"] = {"calls": 50, "p95_s": qdrant_p95_s, "max_s": qdrant_p95_s}
    return metrics


def _measured_row(envelope_id: str, **metrics_kwargs: Any) -> dict[str, Any]:
    row = _row(envelope_id)
    row["metadata"]["metrics"].update(_metrics(**metrics_kwargs))
    return row


def _gate(report: canary.CanaryReport, name: str) -> canary.GateResult:
    [gate] = [g for g in report.gates if g.name == name]
    return gate


async def _report(*rows: dict[str, Any]) -> canary.CanaryReport:
    ids = [str(r["id"]) for r in rows]
    return await canary.run_canary_report(_FakePool(rows=list(rows)), _full_schema_client(), ids)


async def test_book_exposes_its_persisted_ingest_metrics() -> None:
    report = await _report(
        _measured_row("env-1", stages={"parse": 61.0, "chunk_embed": 300.0}, embed=(200, 3))
    )
    [book] = report.books
    assert book.stage_seconds == {"parse": 61.0, "chunk_embed": 300.0}  # partial: see below
    assert (book.embed_calls, book.embed_retries_429) == (200, 3)
    assert book.qdrant_p95_s == 0.2
    assert book.ingest_pages == 100


async def test_a_fully_measured_clean_canary_passes() -> None:
    report = await _report(_measured_row("env-1"), _measured_row("env-2"))
    assert {g.name: g.status for g in report.gates} == dict.fromkeys(
        canary.CANARY_GATE_NAMES, "pass"
    )
    assert report.verdict == "PASS"


async def test_stage_over_its_budget_fails() -> None:
    # 100 pages -> budget max(900, 600) = 900 s; 901 s is over.
    report = await _report(_measured_row("env-1", stages={"parse": 10.0, "visual": 901.0}))
    gate = _gate(report, "stage_budget")
    assert gate.status == "fail"
    assert "env-1" in gate.detail and "visual" in gate.detail
    assert report.verdict == "FAIL"


async def test_stage_budget_scales_with_pages() -> None:
    # 1000 pages -> budget 6000 s.
    from vo_rag.pipelines.book_rag.events import INGEST_STEP_ORDER

    stages = {**dict.fromkeys(INGEST_STEP_ORDER, 60.0), "visual": 5999.0}
    report = await _report(_measured_row("env-1", pages=1000, stages=stages))
    assert _gate(report, "stage_budget").status == "pass"


async def test_stage_budget_without_pages_uses_the_floor() -> None:
    report = await _report(_measured_row("env-1", pages=None, stages={"parse": 950.0}))
    assert _gate(report, "stage_budget").status == "fail"


async def test_stage_budget_unmeasured_when_a_book_has_no_stage_times() -> None:
    no_metrics = _row("env-2")
    report = await _report(_measured_row("env-1"), no_metrics)
    gate = _gate(report, "stage_budget")
    assert gate.status == "unmeasured"
    assert "env-2" in gate.detail


async def test_stage_budget_fail_outranks_unmeasured() -> None:
    report = await _report(_measured_row("env-1", stages={"parse": 5000.0}), _row("env-2"))
    assert _gate(report, "stage_budget").status == "fail"


async def test_embed_429_rate_under_one_percent_passes() -> None:
    report = await _report(_measured_row("env-1", embed=(1000, 9)))
    assert _gate(report, "embed_429_rate").status == "pass"


async def test_embed_429_rate_at_one_percent_fails() -> None:
    # Pooled over the sample: (5 + 5) / (500 + 500) = 1%.
    report = await _report(
        _measured_row("env-1", embed=(500, 5)), _measured_row("env-2", embed=(500, 5))
    )
    gate = _gate(report, "embed_429_rate")
    assert gate.status == "fail"
    assert "10/1000" in gate.detail


async def test_embed_429_rate_unmeasured_without_embed_counts() -> None:
    report = await _report(_measured_row("env-1"), _measured_row("env-2", embed=None))
    gate = _gate(report, "embed_429_rate")
    assert gate.status == "unmeasured"
    assert "env-2" in gate.detail


async def test_embed_429_rate_unmeasured_when_nothing_was_embedded() -> None:
    report = await _report(_measured_row("env-1", embed=(0, 0)))
    assert _gate(report, "embed_429_rate").status == "unmeasured"


async def test_qdrant_p95_under_a_second_passes() -> None:
    report = await _report(_measured_row("env-1", qdrant_p95_s=0.99))
    assert _gate(report, "qdrant_p95").status == "pass"


async def test_qdrant_p95_of_a_second_or_more_fails() -> None:
    report = await _report(
        _measured_row("env-1", qdrant_p95_s=0.1), _measured_row("env-2", qdrant_p95_s=1.0)
    )
    gate = _gate(report, "qdrant_p95")
    assert gate.status == "fail"
    assert "env-2" in gate.detail


async def test_qdrant_p95_unmeasured_without_latency_data() -> None:
    report = await _report(_measured_row("env-1", qdrant_p95_s=None))
    assert _gate(report, "qdrant_p95").status == "unmeasured"


async def test_completed_book_missing_a_stage_is_unmeasured_not_pass() -> None:
    from vo_rag.pipelines.book_rag.events import INGEST_STEP_ORDER

    stages = dict.fromkeys(INGEST_STEP_ORDER, 60.0)
    del stages["chunk_embed"]
    report = await _report(_measured_row("env-1", stages=stages))
    gate = _gate(report, "stage_budget")
    assert gate.status == "unmeasured"
    assert "env-1" in gate.detail and "chunk_embed" in gate.detail
    assert report.verdict == "INCOMPLETE"


async def test_a_missing_stage_never_hides_an_over_budget_one() -> None:
    report = await _report(_measured_row("env-1", stages={"parse": 5000.0}))
    assert _gate(report, "stage_budget").status == "fail"


async def test_preflight_pages_set_the_stage_budget_when_parse_recorded_none() -> None:
    from vo_rag.pipelines.book_rag.events import INGEST_STEP_ORDER

    row = _measured_row(
        "env-1", pages=None, stages={**dict.fromkeys(INGEST_STEP_ORDER, 60.0), "parse": 5000.0}
    )
    row["metadata"]["preflight"] = {"pages": 1000}  # budget 6000 s
    report = await _report(row)
    assert report.books[0].ingest_pages == 1000
    assert _gate(report, "stage_budget").status == "pass"


def test_lane_count_note_no_longer_claims_429s_are_unmeasured() -> None:
    projection = canary.project_batch(total_pages=1000, pages_per_min_per_lane=10.0, lane_count=4)
    assert "unmeasured" not in projection.lane_count_note
    assert "embed_429_rate" in projection.lane_count_note

"""`vo-rag corpus duplicates --report CSV` CLI (VOR-1427).

`_resources` and `collect_stats` are monkeypatched; nothing real (Qdrant/
Postgres) is touched. `plan_duplicates` and `envelope_ingested_at` run for
real over the fake `BookStats`/pool-row data so this also exercises the
CLI's Postgres-query wiring end to end.
"""

from __future__ import annotations

import csv
import json
from contextlib import asynccontextmanager
from typing import Any

import pytest
from click.testing import CliRunner

from vo_rag.cli import corpus_ops
from vo_rag.cli.main import cli
from vo_rag.core.corpus.health import BookStats


def _lines(output: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in output.splitlines() if line.strip()]


class _FakePool:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, Any]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.calls.append((query, args))
        return self.rows


@pytest.fixture
def fake_resources(monkeypatch: pytest.MonkeyPatch) -> _FakePool:
    pool = _FakePool(rows=[])

    @asynccontextmanager
    async def _fake_resources() -> Any:
        yield pool, "fake-store"

    monkeypatch.setattr(corpus_ops, "_resources", _fake_resources)
    return pool


@pytest.fixture
def fake_stats(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def _fake_collect_stats(
        store: Any, *, source_ids: list[str] | None = None, boilerplate_matcher: Any = None
    ) -> list[BookStats]:
        calls.append(
            {"store": store, "source_ids": source_ids, "boilerplate_matcher": boilerplate_matcher}
        )
        return [
            BookStats(source_id="A", source_book="Book X", chunks=900, tiny=500),
            BookStats(source_id="B", source_book="Book X.reencoded", chunks=800, tiny=0),
            BookStats(source_id="C", source_book="Book X", chunks=700, tiny=0),
        ]

    monkeypatch.setattr(corpus_ops, "collect_stats", _fake_collect_stats)
    return calls


def test_corpus_group_lists_duplicates() -> None:
    result = CliRunner().invoke(cli, ["corpus", "--help"])
    assert result.exit_code == 0, result.output
    assert "duplicates" in result.output


def test_report_is_required() -> None:
    result = CliRunner().invoke(cli, ["corpus", "duplicates"])
    assert result.exit_code != 0


def test_writes_csv_and_prints_groups_and_delete_counts(
    tmp_path: Any, fake_resources: _FakePool, fake_stats: list[dict[str, Any]]
) -> None:
    report_path = tmp_path / "dupes.csv"

    result = CliRunner().invoke(cli, ["corpus", "duplicates", "--report", str(report_path)])

    assert result.exit_code == 0, result.output
    assert _lines(result.output) == [{"groups": 1, "delete": 2}]

    # No id carries a sha256, so there is no envelope to date them by.
    assert fake_resources.calls == []

    with report_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["source_id"] for row in rows} == {"A", "B", "C"}
    by_id = {row["source_id"]: row for row in rows}
    assert by_id["C"]["action"] == "keep"
    assert by_id["A"]["action"] == by_id["B"]["action"] == "delete"
    assert by_id["A"]["reason"] == "more degenerate/boilerplate chunks than keeper"
    assert by_id["B"]["reason"] == "reencoded twin"
    assert set(rows[0].keys()) == {
        "group",
        "source_id",
        "source_book",
        "chunks",
        "bad_chunks",
        "reencoded",
        "ingested_at",
        "action",
        "reason",
    }


def test_no_books_writes_empty_report_and_zero_counts(
    tmp_path: Any, fake_resources: _FakePool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _empty_stats(
        store: Any, *, source_ids: list[str] | None = None, boilerplate_matcher: Any = None
    ) -> list[BookStats]:
        return []

    monkeypatch.setattr(corpus_ops, "collect_stats", _empty_stats)
    report_path = tmp_path / "dupes.csv"

    result = CliRunner().invoke(cli, ["corpus", "duplicates", "--report", str(report_path)])

    assert result.exit_code == 0, result.output
    assert _lines(result.output) == [{"groups": 0, "delete": 0}]
    assert fake_resources.calls == []  # never queries Postgres with an empty id list
    with report_path.open(newline="") as handle:
        assert list(csv.DictReader(handle)) == []


# --- author from the Qdrant payload, dates from envelopes ---

_LEGACY_SHA = "a" * 64
_NEW_SHA = "b" * 64
_LEGACY_ID = _LEGACY_SHA.upper()  # a pre-prefix legacy source id
_NEW_ID = f"vo-rag-corpus-sha256:{_NEW_SHA}"


class _RoutingPool:
    """Answers the legacy `sources` query (stale, as in production) and the
    `intake_envelopes` query differently, so the test shows which one the
    command reads."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.calls.append((query, args))
        if "FROM sources" in query:
            # Only the legacy copy was ever written to `sources`.
            return [{"id": _LEGACY_ID, "author": "Legacy Author", "created_at": "2026-06-01"}]
        if "intake_envelopes" in query:
            from datetime import UTC, datetime

            return [
                {"sha256": _NEW_SHA, "ingested_at": datetime(2026, 3, 1, tzinfo=UTC)},
                {"sha256": _LEGACY_SHA, "ingested_at": datetime(2026, 8, 1, tzinfo=UTC)},
            ]
        return []


def test_legacy_and_new_copies_group_by_payload_author_and_keep_the_earliest_envelope(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _RoutingPool()

    @asynccontextmanager
    async def _fake_resources() -> Any:
        yield pool, "fake-store"

    async def _fake_collect_stats(
        store: Any, *, source_ids: list[str] | None = None, boilerplate_matcher: Any = None
    ) -> list[BookStats]:
        # Identical copies: only the ingest date can separate them.
        return [
            BookStats(source_id=_LEGACY_ID, source_book="Book Y", chunks=500, author="Smith"),
            BookStats(source_id=_NEW_ID, source_book="Book Y", chunks=500, author="Smith"),
        ]

    monkeypatch.setattr(corpus_ops, "_resources", _fake_resources)
    monkeypatch.setattr(corpus_ops, "collect_stats", _fake_collect_stats)
    report_path = tmp_path / "dupes.csv"

    result = CliRunner().invoke(cli, ["corpus", "duplicates", "--report", str(report_path)])

    assert result.exit_code == 0, result.output
    assert _lines(result.output) == [{"groups": 1, "delete": 1}]
    assert not [q for q, _ in pool.calls if "FROM sources" in q]
    with report_path.open(newline="") as handle:
        by_id = {row["source_id"]: row for row in csv.DictReader(handle)}
    assert by_id[_NEW_ID]["action"] == "keep"
    assert by_id[_NEW_ID]["ingested_at"].startswith("2026-03-01")
    assert by_id[_LEGACY_ID]["action"] == "delete"
    assert by_id[_LEGACY_ID]["reason"] == "ingested later than keeper"

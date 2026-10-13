"""`vo-rag corpus canary-report ENVELOPE_ID...` CLI (VOR-1434).

`_resources` is monkeypatched to a fake Postgres pool + a fake Qdrant store
with every declared payload index present -- nothing real is touched.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

from vo_rag.cli import corpus_ops
from vo_rag.cli.main import cli


def _lines(output: str) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in output.splitlines() if line.strip() and line.startswith("{")
    ]


def _dt(seconds: float) -> datetime:
    return datetime(2026, 9, 27, tzinfo=UTC) + timedelta(seconds=seconds)


def _row(envelope_id: str, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": envelope_id,
        "status": "completed",
        "error": None,
        "created_at": _dt(0.0),
        "updated_at": _dt(600.0),
        "metadata": {
            # Lane time runs from the claim.
            "metrics": {"claimed_at_epoch_s": _dt(0.0).timestamp()},
            "quality": {
                "findings": [],
                "stats": {
                    "source_id": f"src-{envelope_id}",
                    "source_book": "Book",
                    "chunks": 42,
                    "max_page": 100,
                },
            },
        },
    }
    base.update(overrides)
    return base


class _FakePool:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, Any]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.calls.append((query, args))
        return self.rows


class _FakeQdrantClient:
    def __init__(self, payload_schema_by_collection: dict[str, set[str]]) -> None:
        self._schemas = payload_schema_by_collection

    async def get_collections(self) -> Any:
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self._schemas])

    async def get_collection(self, name: str) -> Any:
        return SimpleNamespace(payload_schema=dict.fromkeys(self._schemas[name]))


class _FakeStore:
    def __init__(self, client: _FakeQdrantClient) -> None:
        self._client = client


def _full_schema_store() -> _FakeStore:
    from vo_rag.core.storage.qdrant import COMMON_PAYLOAD_INDEXES, _extra_indexes_for

    fields = set(COMMON_PAYLOAD_INDEXES) | set(_extra_indexes_for("vo_rag_books_raw_v1"))
    return _FakeStore(_FakeQdrantClient({"vo_rag_books_raw_v1": fields}))


@pytest.fixture
def fake_resources(monkeypatch: pytest.MonkeyPatch) -> _FakePool:
    pool = _FakePool(rows=[])

    @asynccontextmanager
    async def _fake_resources() -> Any:
        yield pool, _full_schema_store()

    monkeypatch.setattr(corpus_ops, "_resources", _fake_resources)
    return pool


def test_canary_report_is_listed() -> None:
    result = CliRunner().invoke(cli, ["corpus", "--help"])
    assert result.exit_code == 0, result.output
    assert "canary-report" in result.output


def test_requires_at_least_one_envelope_id() -> None:
    result = CliRunner().invoke(cli, ["corpus", "canary-report"])
    assert result.exit_code != 0


def test_incomplete_verdict_exits_nonzero_and_prints_gates(fake_resources: _FakePool) -> None:
    fake_resources.rows = [_row("env-1")]

    result = CliRunner().invoke(cli, ["corpus", "canary-report", "env-1"])

    assert result.exit_code != 0, result.output
    assert "verdict: INCOMPLETE" in result.output
    assert "stage_budget: unmeasured" in result.output
    assert "embed_429_rate: unmeasured" in result.output
    assert "qdrant_p95: unmeasured" in result.output
    assert "payload_indexes: pass" in result.output
    assert "quality: pass" in result.output
    assert "failed_envelopes: pass" in result.output


def test_prints_one_json_line_per_book(fake_resources: _FakePool) -> None:
    fake_resources.rows = [_row("env-1"), _row("env-2")]

    result = CliRunner().invoke(cli, ["corpus", "canary-report", "env-1", "env-2"])

    books = _lines(result.output)
    assert {b["envelope_id"] for b in books} == {"env-1", "env-2"}
    assert books[0]["pages"] == 100
    assert books[0]["chunks"] == 42
    assert books[0]["pages_per_min"] == 10.0


def test_failed_envelope_fails_the_verdict(fake_resources: _FakePool) -> None:
    fake_resources.rows = [_row("env-1", status="failed", error="boom", metadata={})]

    result = CliRunner().invoke(cli, ["corpus", "canary-report", "env-1"])

    assert result.exit_code != 0
    assert "verdict: FAIL" in result.output
    assert "failed_envelopes: fail" in result.output


def test_json_mode_emits_one_object(fake_resources: _FakePool) -> None:
    fake_resources.rows = [_row("env-1")]

    result = CliRunner().invoke(cli, ["corpus", "canary-report", "env-1", "--json"])

    assert result.exit_code != 0  # still INCOMPLETE
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["verdict"] == "INCOMPLETE"
    assert len(payload["books"]) == 1
    assert payload["books"][0]["envelope_id"] == "env-1"
    assert {g["name"] for g in payload["gates"]} == {
        "failed_envelopes",
        "stage_budget",
        "embed_429_rate",
        "qdrant_p95",
        "payload_indexes",
        "quality",
    }


def test_projection_printed_when_total_pages_given(fake_resources: _FakePool) -> None:
    fake_resources.rows = [_row("env-1")]

    result = CliRunner().invoke(
        cli, ["corpus", "canary-report", "env-1", "--total-pages", "1000", "--lanes", "2"]
    )

    assert "projection:" in result.output
    assert "total_pages=1000" in result.output
    assert "lane_count=2" in result.output


def test_include_queue_wait_flag_falls_back_to_created_at(fake_resources: _FakePool) -> None:
    row = _row("env-1")
    del row["metadata"]["metrics"]
    fake_resources.rows = [row]

    strict = CliRunner().invoke(cli, ["corpus", "canary-report", "env-1", "--total-pages", "1000"])
    assert "projection:" not in strict.output

    lenient = CliRunner().invoke(
        cli,
        ["corpus", "canary-report", "env-1", "--total-pages", "1000", "--include-queue-wait"],
    )
    assert "projection:" in lenient.output


def test_no_projection_line_without_total_pages(fake_resources: _FakePool) -> None:
    fake_resources.rows = [_row("env-1")]

    result = CliRunner().invoke(cli, ["corpus", "canary-report", "env-1"])

    assert "projection:" not in result.output


def test_missing_payload_index_fails_the_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _FakePool(rows=[_row("env-1")])
    store = _FakeStore(_FakeQdrantClient({"vo_rag_books_raw_v1": {"project_id"}}))

    @asynccontextmanager
    async def _fake_resources() -> Any:
        yield pool, store

    monkeypatch.setattr(corpus_ops, "_resources", _fake_resources)

    result = CliRunner().invoke(cli, ["corpus", "canary-report", "env-1"])

    assert result.exit_code != 0
    assert "verdict: FAIL" in result.output
    assert "payload_indexes: fail" in result.output

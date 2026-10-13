"""`vo-rag corpus health` CLI (VOR-1427).

``run_health`` and ``_resources`` are monkeypatched; nothing real
(Qdrant/Postgres) is touched.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import pytest
from click.testing import CliRunner

from vo_rag.cli import corpus_ops
from vo_rag.cli.main import cli
from vo_rag.core.corpus.boilerplate_ops import boilerplate_payload_matcher
from vo_rag.core.corpus.health import HealthReport


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> None:
    @asynccontextmanager
    async def _fake_resources() -> Any:
        yield "fake-pool", "fake-store"

    monkeypatch.setattr(corpus_ops, "_resources", _fake_resources)


@pytest.fixture
def recorded_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def _fake_run_health(
        store: Any,
        *,
        source_ids: list[str] | None = None,
        boilerplate_matcher: Any = None,
        pool: Any = None,
    ) -> HealthReport:
        calls.append(
            {
                "store": store,
                "source_ids": source_ids,
                "boilerplate_matcher": boilerplate_matcher,
                "pool": pool,
            }
        )
        return HealthReport(books=3)

    monkeypatch.setattr(corpus_ops, "run_health", _fake_run_health)
    return calls


def test_corpus_group_lists_health() -> None:
    result = CliRunner().invoke(cli, ["corpus", "--help"])
    assert result.exit_code == 0, result.output
    assert "health" in result.output


def test_health_with_no_source_id_scans_everything(
    fake_store: None, recorded_calls: list[dict[str, Any]]
) -> None:
    result = CliRunner().invoke(cli, ["corpus", "health"])

    assert result.exit_code == 0, result.output
    assert len(recorded_calls) == 1
    assert recorded_calls[0]["store"] == "fake-store"
    assert recorded_calls[0]["source_ids"] is None
    # The pool tells legacy (null parser_used) PDFs from ebooks.
    assert recorded_calls[0]["pool"] == "fake-pool"
    # `health` wires the shared boilerplate matcher into `run_health` (VOR-1427),
    # not just leaves the hook at its `None` default.
    matcher = recorded_calls[0]["boilerplate_matcher"]
    assert matcher is boilerplate_payload_matcher
    assert matcher({"content": "This page intentionally left blank"}) is True
    assert matcher({"content": "A perfectly ordinary sentence."}) is False


def test_health_source_id_may_repeat(
    fake_store: None, recorded_calls: list[dict[str, Any]]
) -> None:
    result = CliRunner().invoke(
        cli, ["corpus", "health", "--source-id", "book-a", "--source-id", "book-b"]
    )

    assert result.exit_code == 0, result.output
    assert len(recorded_calls) == 1
    assert recorded_calls[0]["store"] == "fake-store"
    assert recorded_calls[0]["source_ids"] == ["book-a", "book-b"]


def test_health_json_prints_one_object(
    fake_store: None, recorded_calls: list[dict[str, Any]]
) -> None:
    result = CliRunner().invoke(cli, ["corpus", "health", "--json"])

    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["books"] == 3


def test_health_exits_1_when_report_is_not_ok(
    fake_store: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _fake_run_health(
        store: Any,
        *,
        source_ids: list[str] | None = None,
        boilerplate_matcher: Any = None,
        pool: Any = None,
    ) -> HealthReport:
        return HealthReport(books=1, degenerate=["book-a"])

    monkeypatch.setattr(corpus_ops, "run_health", _fake_run_health)

    result = CliRunner().invoke(cli, ["corpus", "health"])

    assert result.exit_code == 1, result.output


def test_health_exits_0_when_report_is_ok(
    fake_store: None, recorded_calls: list[dict[str, Any]]
) -> None:
    result = CliRunner().invoke(cli, ["corpus", "health"])
    assert result.exit_code == 0, result.output


# --- VOR-1434: the four quality findings (+ no_chunks) print in
# --- the plain-text branch and pass through the JSON branch ---


def test_health_text_output_prints_quality_findings_and_exits_1(
    fake_store: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _fake_run_health(
        store: Any,
        *,
        source_ids: list[str] | None = None,
        boilerplate_matcher: Any = None,
        pool: Any = None,
    ) -> HealthReport:
        return HealthReport(
            books=5,
            empty_figures=["book-a"],
            low_text_density=["book-b"],
            image_only_suspect=["book-c"],
            too_few_chunks=["book-d"],
            no_chunks=["book-e"],
        )

    monkeypatch.setattr(corpus_ops, "run_health", _fake_run_health)

    result = CliRunner().invoke(cli, ["corpus", "health"])

    assert result.exit_code == 1, result.output
    assert "empty_figures: ['book-a']" in result.output
    assert "low_text_density: ['book-b']" in result.output
    assert "image_only_suspect: ['book-c']" in result.output
    assert "too_few_chunks: ['book-d']" in result.output
    assert "no_chunks: ['book-e']" in result.output


def test_health_json_includes_quality_finding_fields(
    fake_store: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _fake_run_health(
        store: Any,
        *,
        source_ids: list[str] | None = None,
        boilerplate_matcher: Any = None,
        pool: Any = None,
    ) -> HealthReport:
        return HealthReport(books=1, too_few_chunks=["book-x"], no_chunks=["book-y"])

    monkeypatch.setattr(corpus_ops, "run_health", _fake_run_health)

    result = CliRunner().invoke(cli, ["corpus", "health", "--json"])

    assert result.exit_code == 1, result.output
    lines = [line for line in result.output.splitlines() if line.strip()]
    payload = json.loads(lines[0])
    assert payload["too_few_chunks"] == ["book-x"]
    assert payload["no_chunks"] == ["book-y"]
    # Existing keys are still present (additive-only, VOR-1434).
    assert payload["degenerate"] == []
    assert payload["duplicate_groups"] == 0

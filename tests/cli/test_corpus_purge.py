"""`vo-rag corpus purge` CLI (VOR-1427).

``purge_source`` and ``_resources`` are monkeypatched; nothing real
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
from vo_rag.core.corpus.purge import PurgeReport


def _lines(output: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in output.splitlines() if line.strip()]


@pytest.fixture
def fake_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    @asynccontextmanager
    async def _fake_resources() -> Any:
        yield "fake-pool", "fake-store"

    monkeypatch.setattr(corpus_ops, "_resources", _fake_resources)


@pytest.fixture
def recorded_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def _fake_purge_source(
        store: Any, pool: Any, source_id: str, *, apply: bool, keep_categories: bool = True
    ) -> PurgeReport:
        calls.append(
            {
                "store": store,
                "pool": pool,
                "source_id": source_id,
                "apply": apply,
                "keep_categories": keep_categories,
            }
        )
        return PurgeReport(
            source_id=source_id,
            qdrant={"vo_rag_books_raw_v1": 3},
            graph_nodes_detached=0,
            postgres={"sources": 1},
        )

    monkeypatch.setattr(corpus_ops, "purge_source", _fake_purge_source)
    return calls


def test_corpus_group_lists_purge() -> None:
    result = CliRunner().invoke(cli, ["corpus", "--help"])
    assert result.exit_code == 0, result.output
    assert "purge" in result.output


def test_without_yes_dry_runs_and_reports(
    fake_resources: None, recorded_calls: list[dict[str, Any]]
) -> None:
    result = CliRunner().invoke(cli, ["corpus", "purge", "book-a", "book-b"])

    assert result.exit_code == 0, result.output
    assert [c["apply"] for c in recorded_calls] == [False, False]
    assert [c["source_id"] for c in recorded_calls] == ["book-a", "book-b"]
    assert [c["keep_categories"] for c in recorded_calls] == [True, True]
    assert [c["store"] for c in recorded_calls] == ["fake-store", "fake-store"]
    assert [c["pool"] for c in recorded_calls] == ["fake-pool", "fake-pool"]

    lines = _lines(result.output)
    assert len(lines) == 3
    assert lines[0]["source_id"] == "book-a"
    assert lines[1]["source_id"] == "book-b"
    assert lines[-1] == {"applied": False}


def test_yes_applies(fake_resources: None, recorded_calls: list[dict[str, Any]]) -> None:
    result = CliRunner().invoke(cli, ["corpus", "purge", "book-a", "--yes"])

    assert result.exit_code == 0, result.output
    assert recorded_calls[0]["apply"] is True
    assert _lines(result.output)[-1] == {"applied": True}


def test_drop_categories_passes_keep_categories_false(
    fake_resources: None, recorded_calls: list[dict[str, Any]]
) -> None:
    result = CliRunner().invoke(cli, ["corpus", "purge", "book-a", "--yes", "--drop-categories"])

    assert result.exit_code == 0, result.output
    assert recorded_calls[0]["keep_categories"] is False


def test_requires_at_least_one_source_id(fake_resources: None) -> None:
    result = CliRunner().invoke(cli, ["corpus", "purge"])
    assert result.exit_code != 0


def test_earlier_reports_are_not_lost_when_a_later_id_raises(
    fake_resources: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each id's report is emitted as it finishes, not batched at the end --
    so a later id raising doesn't swallow an already-applied purge's report."""

    async def _fake_purge_source(
        store: Any, pool: Any, source_id: str, *, apply: bool, keep_categories: bool = True
    ) -> PurgeReport:
        if source_id == "book-b":
            raise RuntimeError("simulated failure on the second id")
        return PurgeReport(source_id=source_id, qdrant={}, graph_nodes_detached=0, postgres={})

    monkeypatch.setattr(corpus_ops, "purge_source", _fake_purge_source)

    result = CliRunner().invoke(cli, ["corpus", "purge", "book-a", "book-b", "--yes"])

    assert result.exit_code != 0
    lines = _lines(result.output)
    assert len(lines) == 1
    assert lines[0]["source_id"] == "book-a"

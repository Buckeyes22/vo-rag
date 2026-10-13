"""`vo-rag corpus boilerplate --scan|--delete` CLI (VOR-1427).

`_boilerplate_point_ids` and `_delete_boilerplate_points` are monkeypatched;
nothing real (Qdrant) is touched. See `tests/core/corpus/test_boilerplate_integration.py`
for the live-Qdrant coverage of the real scan/delete helpers.

Exactly one of `--scan`/`--delete` is required:
Both are explicit, always-typed flags (`corpus boilerplate --scan`, `corpus
boilerplate --delete`), and the final line printed for `--scan` is
`{"points": N}`.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest
from click.testing import CliRunner

from vo_rag.cli import corpus_ops
from vo_rag.cli.main import cli


def _lines(output: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in output.splitlines() if line.strip()]


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> None:
    @asynccontextmanager
    async def _fake_store_only() -> Any:
        yield "fake-store"

    monkeypatch.setattr(corpus_ops, "_store_only", _fake_store_only)


@pytest.fixture
def recorded_scan(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def _fake_boilerplate_point_ids(
        store: Any, *, source_ids: list[str] | None
    ) -> dict[str, list[str]]:
        calls.append({"store": store, "source_ids": source_ids})
        return {"book-a": ["id-1", "id-2"], "book-b": ["id-3"]}

    monkeypatch.setattr(corpus_ops, "_boilerplate_point_ids", _fake_boilerplate_point_ids)
    return calls


_STERILE = "vo_rag_books_sterile_v1"
_CONTEXTUALIZED = "vo_rag_books_contextualized_v1"
_RAW = "vo_rag_books_raw_v1"


@dataclass
class _Derived:
    """Sterile/contextualized scans (the `source_ids` asked for) and deletes."""

    scans: list[list[str] | None] = field(default_factory=list)
    deletes: list[tuple[str, list[str]]] = field(default_factory=list)


@pytest.fixture
def derived(monkeypatch: pytest.MonkeyPatch) -> _Derived:
    record = _Derived()

    async def _fake_derived(
        store: Any, *, source_ids: list[str] | None
    ) -> dict[str, dict[str, list[str]] | None]:
        record.scans.append(source_ids)
        # The sterile copy of book-a still holds one boilerplate point; the
        # contextualized collection doesn't exist on this deployment.
        return {_STERILE: {"book-a": ["s-1"]}, _CONTEXTUALIZED: None}

    monkeypatch.setattr(corpus_ops, "_derived_boilerplate_point_ids", _fake_derived)
    return record


@pytest.fixture
def recorded_delete(monkeypatch: pytest.MonkeyPatch, derived: _Derived) -> list[list[str]]:
    """Raw-collection deletes (the ids, per call); derived ones go to `derived`."""
    calls: list[list[str]] = []

    async def _fake_delete_boilerplate_points(
        store: Any, ids: list[str], *, collection: str = _RAW
    ) -> int:
        if collection == _RAW:
            calls.append(list(ids))
        else:
            derived.deletes.append((collection, list(ids)))
        return len(ids)

    monkeypatch.setattr(corpus_ops, "_delete_boilerplate_points", _fake_delete_boilerplate_points)
    return calls


def test_corpus_group_lists_boilerplate() -> None:
    result = CliRunner().invoke(cli, ["corpus", "--help"])
    assert result.exit_code == 0, result.output
    assert "boilerplate" in result.output


def test_scan_reports_per_book_counts_and_points(
    fake_store: None, recorded_scan: list[dict[str, Any]]
) -> None:
    result = CliRunner().invoke(cli, ["corpus", "boilerplate", "--scan"])

    assert result.exit_code == 0, result.output
    assert recorded_scan == [{"store": "fake-store", "source_ids": None}]

    lines = _lines(result.output)
    by_book = {line["source_id"]: line["boilerplate"] for line in lines if "source_id" in line}
    assert by_book == {"book-a": 2, "book-b": 1}
    assert lines[-1] == {"points": 3}


def test_neither_scan_nor_delete_is_a_usage_error(
    fake_store: None, recorded_scan: list[dict[str, Any]]
) -> None:
    """A bare `corpus boilerplate` invocation (neither
    flag) used to silently behave like `--scan` -- `--scan` was accepted but
    never actually read. Exactly one of --scan/--delete is now required."""
    result = CliRunner().invoke(cli, ["corpus", "boilerplate"])

    assert result.exit_code != 0, result.output
    assert recorded_scan == []  # never even reached the scan helper


def test_both_scan_and_delete_together_is_a_usage_error(
    fake_store: None, recorded_scan: list[dict[str, Any]]
) -> None:
    result = CliRunner().invoke(cli, ["corpus", "boilerplate", "--scan", "--delete"])

    assert result.exit_code != 0, result.output
    assert recorded_scan == []


def test_scan_respects_source_id_filter(
    fake_store: None, recorded_scan: list[dict[str, Any]]
) -> None:
    result = CliRunner().invoke(cli, ["corpus", "boilerplate", "--scan", "--source-id", "book-a"])

    assert result.exit_code == 0, result.output
    assert recorded_scan == [{"store": "fake-store", "source_ids": ["book-a"]}]


def test_delete_without_yes_is_a_dry_run_and_reports_counts(
    fake_store: None,
    recorded_scan: list[dict[str, Any]],
    recorded_delete: list[list[str]],
) -> None:
    result = CliRunner().invoke(cli, ["corpus", "boilerplate", "--delete"])

    assert result.exit_code == 0, result.output
    assert recorded_delete == []  # never actually deletes without --yes
    lines = _lines(result.output)
    assert lines[-1] == {"points": 3}


def test_delete_with_yes_applies_and_prints_deleted_count(
    fake_store: None,
    recorded_scan: list[dict[str, Any]],
    recorded_delete: list[list[str]],
) -> None:
    result = CliRunner().invoke(cli, ["corpus", "boilerplate", "--delete", "--yes"])

    assert result.exit_code == 0, result.output
    assert len(recorded_delete) == 1
    assert set(recorded_delete[0]) == {"id-1", "id-2", "id-3"}
    assert _lines(result.output)[-1] == {"deleted": 3}


def test_delete_with_yes_also_clears_the_sterile_and_contextualized_copies(
    fake_store: None,
    recorded_scan: list[dict[str, Any]],
    recorded_delete: list[list[str]],
    derived: _Derived,
) -> None:
    # Purge clears these two collections too, and publication/partner
    # queries read sterile, so boilerplate must not survive there.
    result = CliRunner().invoke(
        cli, ["corpus", "boilerplate", "--delete", "--yes", "--source-id", "book-a"]
    )

    assert result.exit_code == 0, result.output
    assert derived.scans == [["book-a"]]
    assert derived.deletes == [(_STERILE, ["s-1"])]
    assert _lines(result.output) == [
        {"collection": _STERILE, "deleted": 1},
        {"collection": _CONTEXTUALIZED, "deleted": "absent"},
        {"deleted": 3},
    ]


def test_delete_dry_run_reports_the_derived_collections_without_deleting(
    fake_store: None,
    recorded_scan: list[dict[str, Any]],
    recorded_delete: list[list[str]],
    derived: _Derived,
) -> None:
    result = CliRunner().invoke(cli, ["corpus", "boilerplate", "--delete"])

    assert result.exit_code == 0, result.output
    assert recorded_delete == [] and derived.deletes == []
    lines = _lines(result.output)
    assert {"collection": _STERILE, "boilerplate": 1} in lines
    assert {"collection": _CONTEXTUALIZED, "boilerplate": "absent"} in lines
    assert lines[-1] == {"points": 3}


def test_yes_without_delete_does_not_apply(
    fake_store: None,
    recorded_scan: list[dict[str, Any]],
    recorded_delete: list[list[str]],
) -> None:
    """`--yes` alone (with `--scan`, not `--delete`) must not delete anything
    -- matching `corpus purge`'s pattern where the destructive flag is the
    gate and `--yes` only ever confirms it."""
    result = CliRunner().invoke(cli, ["corpus", "boilerplate", "--scan", "--yes"])

    assert result.exit_code == 0, result.output
    assert recorded_delete == []
    lines = _lines(result.output)
    assert lines[-1] == {"points": 3}


def test_scan_with_no_matches_prints_zero_points(
    fake_store: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _empty(store: Any, *, source_ids: list[str] | None) -> dict[str, list[str]]:
        return {}

    monkeypatch.setattr(corpus_ops, "_boilerplate_point_ids", _empty)

    result = CliRunner().invoke(cli, ["corpus", "boilerplate", "--scan"])

    assert result.exit_code == 0, result.output
    assert _lines(result.output) == [{"points": 0}]

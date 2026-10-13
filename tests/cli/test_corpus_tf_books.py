"""`vo-rag corpus tf-books inputs|load-drafts|sheet|load` (VOR-1443).

`_resources`, `_pool_only`, `tf_source_ids`, `book_chunks`, `_book_identities` and
`upsert_book` are monkeypatched, so nothing real (Qdrant/Postgres) is touched.
`book_prompt`, `validate_book` and `_book_identities`' own SQL are exercised for
real, so the job files the lane consumes and the records that are stored are
both produced by the actual code.
"""

from __future__ import annotations

import csv
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

from vo_rag.cli import corpus_tf
from vo_rag.cli.main import cli
from vo_rag.core.corpus.source_format import SHA_ENVELOPE_MATCH

DRAFT = {
    "tf_source_type": "coaching_manual",
    "tf_evidence_type": "coaching_practice",
    "tf_pub_year": 1987,
    "tf_book_athlete_level": ["general"],
    "tf_book_age_group": ["not_stated"],
    "tf_book_sex": "not_stated",
    "tf_limitations": "Pre-1990 methods.",
    "rationale": "A federation coaching booklet.",
}

SHA = "a" * 64
SID = f"vo-rag-corpus-sha256:{SHA}"
SHA2 = "b" * 64
SID2 = f"vo-rag-corpus-sha256:{SHA2}"


def _write_results(path: Path, *records: dict[str, Any]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _book_result(source_id: str, result: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "job_id": f"book:{source_id}",
        "kind": "book",
        "source_id": source_id,
        "result": DRAFT if result is None else result,
    }


class RecordingPool:
    """A pool that records every ``fetch`` (query, args) and replays canned rows."""

    def __init__(self, rows: list[dict[str, str]] | None = None) -> None:
        self.rows = rows if rows is not None else []
        self.fetched: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self.fetched.append((query, args))
        return self.rows


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"upserts": [], "written": True}

    @asynccontextmanager
    async def fake_resources() -> Any:
        yield "pool", SimpleNamespace(_client="client")

    @asynccontextmanager
    async def fake_pool_only() -> Any:
        yield "pool"

    async def fake_ids(client: Any) -> list[str]:
        return ["s1"]

    async def fake_chunks(client: Any, sid: str) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": "c1",
                "section_path": "Preface",
                "content": "Preface text",
                "content_sha256": "h",
            }
        ]

    async def fake_idents(pool: Any, source_ids: list[str]) -> dict[str, tuple[str, str]]:
        return dict.fromkeys(source_ids, ("Field Guide to Sprint Mechanics", "Jordan Avery"))

    async def fake_upsert(pool: Any, sid: str, book: Any, *, reviewed: bool) -> bool:
        state["upserts"].append((sid, book, reviewed))
        return bool(state["written"])

    monkeypatch.setattr(corpus_tf, "_resources", fake_resources)
    monkeypatch.setattr(corpus_tf, "_pool_only", fake_pool_only)
    monkeypatch.setattr(corpus_tf, "tf_source_ids", fake_ids)
    monkeypatch.setattr(corpus_tf, "book_chunks", fake_chunks)
    monkeypatch.setattr(corpus_tf, "_book_identities", fake_idents)
    monkeypatch.setattr(corpus_tf, "upsert_book", fake_upsert)
    return state


def test_inputs_writes_one_book_job_per_source(tmp_path: Path, fakes: dict[str, Any]) -> None:
    out = tmp_path / "jobs.jsonl"
    result = CliRunner().invoke(cli, ["corpus", "tf-books", "inputs", "--out", str(out)])
    assert result.exit_code == 0, result.output
    jobs = [json.loads(line) for line in out.read_text().splitlines()]
    assert [j["job_id"] for j in jobs] == ["book:s1"] and jobs[0]["kind"] == "book"
    assert "Field Guide to Sprint Mechanics" in jobs[0]["prompt"]
    # No catalogue year is stored anywhere, so the prompt must not claim one.
    assert "Catalogue year: unknown" in jobs[0]["prompt"]


async def test_book_identities_return_the_title_and_author_of_the_matched_envelope() -> None:
    pool = RecordingPool(
        [{"source_id": SID, "title": "Field Guide to Sprint Mechanics", "author": "Jordan Avery"}]
    )
    idents = await corpus_tf._book_identities(pool, [SID])
    assert idents == {SID: ("Field Guide to Sprint Mechanics", "Jordan Avery")}


async def test_book_identities_fall_back_to_the_source_id_without_an_envelope() -> None:
    pool = RecordingPool([])
    assert await corpus_tf._book_identities(pool, [SID]) == {SID: (SID, "")}


async def test_book_identities_fall_back_when_the_envelope_names_nothing() -> None:
    pool = RecordingPool([{"source_id": SID, "title": "", "author": ""}])
    assert await corpus_tf._book_identities(pool, [SID]) == {SID: (SID, "")}


async def test_book_identities_never_query_for_an_id_without_a_sha256() -> None:
    pool = RecordingPool(
        [{"source_id": SID, "title": "Field Guide to Sprint Mechanics", "author": "Jordan Avery"}]
    )
    assert await corpus_tf._book_identities(pool, ["legacy-book", ""]) == {
        "legacy-book": ("legacy-book", ""),
        "": ("", ""),
    }
    assert pool.fetched == []


async def test_book_identities_resolve_every_book_in_one_batched_query() -> None:
    pool = RecordingPool(
        [
            {
                "source_id": SID,
                "title": "Field Guide to Sprint Mechanics",
                "author": "Jordan Avery",
            },
            {"source_id": SID2, "title": "Stories of the Track Years", "author": "R. Quill"},
        ]
    )
    idents = await corpus_tf._book_identities(pool, [SID, SID2, "legacy-book"])
    assert idents == {
        SID: ("Field Guide to Sprint Mechanics", "Jordan Avery"),
        SID2: ("Stories of the Track Years", "R. Quill"),
        "legacy-book": ("legacy-book", ""),
    }
    assert len(pool.fetched) == 1
    query, args = pool.fetched[0]
    assert SHA_ENVELOPE_MATCH in query
    assert "status = 'completed'" in query
    assert "DISTINCT ON (k.source_id)" in query
    assert "ORDER BY k.source_id, e.updated_at DESC" in query
    assert args == ([SID, SID2], [SHA, SHA2])


def test_load_drafts_stores_valid_drafts_unreviewed_without_a_sheet(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    results = tmp_path / "r.jsonl"
    _write_results(
        results,
        _book_result("s1"),
        _book_result("s2", {**DRAFT, "tf_source_type": "blog"}),
        {"job_id": "chunk:s1", "kind": "chunk", "source_id": "s1", "result": {}},
    )
    result = CliRunner().invoke(cli, ["corpus", "tf-books", "load-drafts", str(results)])
    out = json.loads(result.output.strip().splitlines()[-1])
    assert out["loaded"] == 1 and out["skipped_reviewed"] == 0 and out["skipped_other"] == 1
    assert out["rejected"][0]["source_id"] == "s2"
    sid, book, reviewed = fakes["upserts"][0]
    assert sid == "s1" and reviewed is False and book.tf_pub_year == 1987


def test_load_drafts_counts_a_record_without_a_kind_as_other(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    results = tmp_path / "r.jsonl"
    _write_results(results, {"source_id": "s1", "result": DRAFT})
    result = CliRunner().invoke(cli, ["corpus", "tf-books", "load-drafts", str(results)])
    out = json.loads(result.output.strip().splitlines()[-1])
    assert out == {"loaded": 0, "skipped_reviewed": 0, "skipped_other": 1, "rejected": []}


def test_load_drafts_reports_writes_only(tmp_path: Path, fakes: dict[str, Any]) -> None:
    results = tmp_path / "r.jsonl"
    _write_results(results, _book_result("s1"))
    fakes["written"] = False
    result = CliRunner().invoke(cli, ["corpus", "tf-books", "load-drafts", str(results)])
    out = json.loads(result.output.strip().splitlines()[-1])
    assert out == {"loaded": 0, "skipped_reviewed": 1, "skipped_other": 0, "rejected": []}


@pytest.mark.parametrize(
    "source_id", [None, "", "   "], ids=["missing", "blank", "whitespace_only"]
)
def test_load_drafts_rejects_a_record_without_a_usable_source_id(
    tmp_path: Path, fakes: dict[str, Any], source_id: str | None
) -> None:
    results = tmp_path / "r.jsonl"
    record: dict[str, Any] = {"job_id": "book:", "kind": "book", "result": DRAFT}
    if source_id is not None:
        record["source_id"] = source_id
    _write_results(results, record)
    result = CliRunner().invoke(cli, ["corpus", "tf-books", "load-drafts", str(results)])
    out = json.loads(result.output.strip().splitlines()[-1])
    assert out["loaded"] == 0
    assert out["rejected"] == [
        {"source_id": "", "job_id": "book:", "error": "source_id is required"}
    ]
    assert fakes["upserts"] == []


@pytest.mark.parametrize(
    "raw_result",
    [
        pytest.param(None, id="missing"),
        pytest.param("coaching_manual", id="string"),
        pytest.param([DRAFT], id="list"),
    ],
)
def test_load_drafts_rejects_a_record_whose_result_is_not_an_object(
    tmp_path: Path, fakes: dict[str, Any], raw_result: Any
) -> None:
    results = tmp_path / "r.jsonl"
    record: dict[str, Any] = {"job_id": "book:s1", "kind": "book", "source_id": "s1"}
    if raw_result is not None:
        record["result"] = raw_result
    _write_results(results, record)
    result = CliRunner().invoke(cli, ["corpus", "tf-books", "load-drafts", str(results)])
    out = json.loads(result.output.strip().splitlines()[-1])
    assert out["loaded"] == 0
    assert out["rejected"] == [
        {
            "source_id": "s1",
            "job_id": "book:s1",
            "error": f"result must be a JSON object, got {type(raw_result).__name__}",
        }
    ]
    assert fakes["upserts"] == []


def test_load_drafts_rejection_includes_job_id_when_present(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    results = tmp_path / "r.jsonl"
    _write_results(results, _book_result("s2", {**DRAFT, "tf_source_type": "blog"}))
    result = CliRunner().invoke(cli, ["corpus", "tf-books", "load-drafts", str(results)])
    out = json.loads(result.output.strip().splitlines()[-1])
    assert out["rejected"][0]["source_id"] == "s2"
    assert out["rejected"][0]["job_id"] == "book:s2"


def test_load_drafts_rejection_omits_job_id_when_absent(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    results = tmp_path / "r.jsonl"
    record = {"kind": "book", "source_id": "s2", "result": {**DRAFT, "tf_source_type": "blog"}}
    _write_results(results, record)
    result = CliRunner().invoke(cli, ["corpus", "tf-books", "load-drafts", str(results)])
    out = json.loads(result.output.strip().splitlines()[-1])
    assert out["rejected"][0]["source_id"] == "s2"
    assert "job_id" not in out["rejected"][0]


def test_sheet_then_load_round_trip(tmp_path: Path, fakes: dict[str, Any]) -> None:
    results = tmp_path / "r.jsonl"
    _write_results(results, _book_result("s1"))
    sheet = tmp_path / "review.tsv"
    runner = CliRunner()
    assert (
        runner.invoke(
            cli, ["corpus", "tf-books", "sheet", str(results), "--out", str(sheet)]
        ).exit_code
        == 0
    )
    rows = list(csv.DictReader(sheet.open(), delimiter="\t"))
    assert (
        rows[0]["title"] == "Field Guide to Sprint Mechanics"
        and rows[0]["tf_book_athlete_level"] == "general"
    )
    rows[0]["tf_book_athlete_level"] = "youth|high_school"  # the operator's correction
    with sheet.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=corpus_tf._SHEET_COLUMNS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    result = runner.invoke(cli, ["corpus", "tf-books", "load", str(sheet)])
    out = json.loads(result.output.strip().splitlines()[-1])
    assert out == {"loaded": 1, "skipped_reviewed": 0, "rejected": []}
    sid, book, reviewed = fakes["upserts"][0]
    assert sid == "s1" and reviewed and book.tf_book_athlete_level == ["high_school", "youth"]


def test_the_pool_only_commands_never_open_qdrant(
    tmp_path: Path, fakes: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[bool] = []

    @asynccontextmanager
    async def spy_resources() -> Any:
        opened.append(True)
        yield "pool", SimpleNamespace(_client="client")

    monkeypatch.setattr(corpus_tf, "_resources", spy_resources)
    results = tmp_path / "r.jsonl"
    _write_results(results, _book_result("s1"))
    sheet = tmp_path / "review.tsv"
    runner = CliRunner()
    assert (
        runner.invoke(
            cli, ["corpus", "tf-books", "sheet", str(results), "--out", str(sheet)]
        ).exit_code
        == 0
    )
    assert runner.invoke(cli, ["corpus", "tf-books", "load", str(sheet)]).exit_code == 0
    assert runner.invoke(cli, ["corpus", "tf-books", "load-drafts", str(results)]).exit_code == 0
    assert opened == []


def test_sheet_help_says_title_and_author_are_informational() -> None:
    result = CliRunner().invoke(cli, ["corpus", "tf-books", "sheet", "--help"])
    assert result.exit_code == 0, result.output
    assert "informational" in result.output


def test_sheet_gives_a_rejected_draft_a_row_carrying_its_error(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    results = tmp_path / "r.jsonl"
    _write_results(
        results,
        _book_result("s1"),
        _book_result("s2", {**DRAFT, "tf_source_type": "blog"}),
        {"job_id": "chunk:s1", "kind": "chunk", "source_id": "s1", "result": {}},
    )
    sheet = tmp_path / "review.tsv"
    result = CliRunner().invoke(
        cli, ["corpus", "tf-books", "sheet", str(results), "--out", str(sheet)]
    )
    assert result.exit_code == 0, result.output
    out = json.loads(result.output.strip().splitlines()[-1])
    assert out["rows"] == 2 and out["skipped_other"] == 1
    assert [r["source_id"] for r in out["rejected"]] == ["s2"]
    rows = list(csv.DictReader(sheet.open(), delimiter="\t"))
    assert [r["source_id"] for r in rows] == ["s1", "s2"]
    assert rows[0]["error"] == ""
    assert "tf_source_type" in rows[1]["error"] and rows[1]["tf_source_type"] == "blog"


def test_sheet_rejection_includes_job_id_when_present(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    results = tmp_path / "r.jsonl"
    _write_results(results, _book_result("s2", {**DRAFT, "tf_source_type": "blog"}))
    sheet = tmp_path / "review.tsv"
    result = CliRunner().invoke(
        cli, ["corpus", "tf-books", "sheet", str(results), "--out", str(sheet)]
    )
    out = json.loads(result.output.strip().splitlines()[-1])
    assert out["rejected"][0]["source_id"] == "s2"
    assert out["rejected"][0]["job_id"] == "book:s2"


def test_sheet_strips_source_id_before_looking_up_identity(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    results = tmp_path / "r.jsonl"
    _write_results(results, _book_result("  s1  "))
    sheet = tmp_path / "review.tsv"
    result = CliRunner().invoke(
        cli, ["corpus", "tf-books", "sheet", str(results), "--out", str(sheet)]
    )
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(sheet.open(), delimiter="\t"))
    assert rows[0]["source_id"] == "s1"
    # The identity lookup key must be the stripped id, or the padded lookup
    # key never matches and the row falls back to a blank title/author.
    assert rows[0]["title"] == "Field Guide to Sprint Mechanics"
    assert rows[0]["author"] == "Jordan Avery"


def test_sheet_then_load_reports_writes_only(tmp_path: Path, fakes: dict[str, Any]) -> None:
    results = tmp_path / "r.jsonl"
    _write_results(results, _book_result("s1"))
    sheet = tmp_path / "review.tsv"
    CliRunner().invoke(cli, ["corpus", "tf-books", "sheet", str(results), "--out", str(sheet)])
    fakes["written"] = False
    result = CliRunner().invoke(cli, ["corpus", "tf-books", "load", str(sheet)])
    out = json.loads(result.output.strip().splitlines()[-1])
    assert out == {"loaded": 0, "skipped_reviewed": 1, "rejected": []}


def _sheet_row(**overrides: str) -> dict[str, str]:
    row = dict.fromkeys(corpus_tf._SHEET_COLUMNS, "")
    row.update(
        source_id="s1",
        title="Field Guide to Sprint Mechanics",
        author="Jordan Avery",
        tf_source_type="coaching_manual",
        tf_evidence_type="coaching_practice",
        tf_pub_year="1987",
        tf_book_athlete_level="general",
        tf_book_age_group="not_stated",
        tf_book_sex="not_stated",
        tf_limitations="Pre-1990 methods.",
        rationale="A federation coaching booklet.",
    )
    row.update(overrides)
    return row


def _write_sheet(path: Path, *rows: dict[str, str], columns: list[str] | None = None) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns or corpus_tf._SHEET_COLUMNS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def test_load_rejects_a_row_with_an_unknown_value(tmp_path: Path, fakes: dict[str, Any]) -> None:
    sheet = tmp_path / "review.tsv"
    _write_sheet(sheet, _sheet_row(tf_source_type="blog"))
    result = CliRunner().invoke(cli, ["corpus", "tf-books", "load", str(sheet)])
    out = json.loads(result.output.strip().splitlines()[-1])
    assert out["loaded"] == 0 and out["rejected"][0]["source_id"] == "s1"
    assert fakes["upserts"] == []


@pytest.mark.parametrize(
    "source_id", [None, "", "   "], ids=["missing_column", "blank", "whitespace_only"]
)
def test_load_rejects_a_row_without_a_usable_source_id(
    tmp_path: Path, fakes: dict[str, Any], source_id: str | None
) -> None:
    sheet = tmp_path / "review.tsv"
    row = _sheet_row()
    columns = corpus_tf._SHEET_COLUMNS
    if source_id is None:
        columns = [c for c in corpus_tf._SHEET_COLUMNS if c != "source_id"]
        row.pop("source_id")
    else:
        row["source_id"] = source_id
    _write_sheet(sheet, row, columns=columns)
    result = CliRunner().invoke(cli, ["corpus", "tf-books", "load", str(sheet)])
    out = json.loads(result.output.strip().splitlines()[-1])
    assert out["loaded"] == 0
    assert out["rejected"] == [{"source_id": "", "error": "source_id is required"}]
    assert fakes["upserts"] == []


def test_load_rejects_a_row_with_an_empty_list_segment(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    sheet = tmp_path / "review.tsv"
    _write_sheet(sheet, _sheet_row(tf_book_athlete_level="youth||masters"))
    result = CliRunner().invoke(cli, ["corpus", "tf-books", "load", str(sheet)])
    out = json.loads(result.output.strip().splitlines()[-1])
    assert out["loaded"] == 0 and "tf_book_athlete_level" in out["rejected"][0]["error"]
    assert fakes["upserts"] == []


def test_load_rejects_a_row_whose_list_is_blank(tmp_path: Path, fakes: dict[str, Any]) -> None:
    sheet = tmp_path / "review.tsv"
    _write_sheet(sheet, _sheet_row(tf_book_age_group=""))
    result = CliRunner().invoke(cli, ["corpus", "tf-books", "load", str(sheet)])
    out = json.loads(result.output.strip().splitlines()[-1])
    assert out["loaded"] == 0 and "tf_book_age_group" in out["rejected"][0]["error"]


def test_load_rejects_a_row_whose_pub_year_is_not_a_plain_integer(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    sheet = tmp_path / "review.tsv"
    _write_sheet(sheet, _sheet_row(tf_pub_year="circa 1987"))
    result = CliRunner().invoke(cli, ["corpus", "tf-books", "load", str(sheet)])
    out = json.loads(result.output.strip().splitlines()[-1])
    assert out["loaded"] == 0
    assert out["rejected"] == [
        {"source_id": "s1", "error": "tf_pub_year: not a plain integer 'circa 1987'"}
    ]
    assert fakes["upserts"] == []


def test_load_accepts_a_blank_pub_year_as_none(tmp_path: Path, fakes: dict[str, Any]) -> None:
    sheet = tmp_path / "review.tsv"
    _write_sheet(sheet, _sheet_row(tf_pub_year=""))
    result = CliRunner().invoke(cli, ["corpus", "tf-books", "load", str(sheet)])
    out = json.loads(result.output.strip().splitlines()[-1])
    assert out == {"loaded": 1, "skipped_reviewed": 0, "rejected": []}
    _sid, book, _reviewed = fakes["upserts"][0]
    assert book.tf_pub_year is None


def test_load_tolerates_a_leading_utf8_bom_in_the_sheet(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    """A sheet re-saved by a Windows-side editor may carry a BOM; it must not
    sit in front of the header's first column name and break every row."""
    sheet = tmp_path / "review.tsv"
    row = _sheet_row()
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=corpus_tf._SHEET_COLUMNS, delimiter="\t")
    writer.writeheader()
    writer.writerow(row)
    sheet.write_bytes(b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8"))
    result = CliRunner().invoke(cli, ["corpus", "tf-books", "load", str(sheet)])
    out = json.loads(result.output.strip().splitlines()[-1])
    assert out == {"loaded": 1, "skipped_reviewed": 0, "rejected": []}
    sid, _book, _reviewed = fakes["upserts"][0]
    assert sid == "s1"


def test_load_ignores_the_error_column_and_the_informational_columns(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    sheet = tmp_path / "review.tsv"
    _write_sheet(
        sheet,
        _sheet_row(
            error="tf_source_type: not allowed ['blog'] (fixed by hand)",
            title="Some other title",
            author="Someone else",
        ),
    )
    result = CliRunner().invoke(cli, ["corpus", "tf-books", "load", str(sheet)])
    out = json.loads(result.output.strip().splitlines()[-1])
    assert out == {"loaded": 1, "skipped_reviewed": 0, "rejected": []}
    _sid, book, _reviewed = fakes["upserts"][0]
    assert book.tf_source_type == "coaching_manual"

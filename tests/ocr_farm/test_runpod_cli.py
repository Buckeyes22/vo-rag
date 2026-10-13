"""`vo-rag corpus ocr-farm runpod plan|verify` (VOR-1434).

pitwall comes from a FakeRunner and Garage from a FakeCache; nothing real is touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from tests.ocr_farm._runpod_fakes import FakeCache, FakeRunner, pdf_item, pitwall_replies
from vo_rag.cli import corpus_ocr
from vo_rag.cli.main import cli
from vo_rag.ocr_farm.coordinator import ocr_key, ocr_meta_key


def _backlog(tmp_path: Path, pages: int) -> Path:
    item, _ = pdf_item(pages, "b")
    path = tmp_path / "backlog.jsonl"
    path.write_text(json.dumps(item.to_dict()) + "\n" + json.dumps({"books": 1}) + "\n")
    return path


def _lines(output: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in output.splitlines() if line.startswith("{")]


def _plan(tmp_path: Path, remaining: str, monkeypatch: pytest.MonkeyPatch) -> tuple[int, list[Any]]:
    runner = FakeRunner(pitwall_replies(remaining=remaining, price="0.40"))
    monkeypatch.setattr(corpus_ocr, "_runner", lambda: runner)
    result = CliRunner().invoke(
        cli,
        ["corpus", "ocr-farm", "runpod", "plan", "--backlog", str(_backlog(tmp_path, 1000)),
         "--job-dir", str(tmp_path / "job"), "--pages-per-gpu-hour", "100", "--pods", "2"],
    )  # fmt: skip
    return result.exit_code, _lines(result.output)


def test_plan_shows_pages_hours_rate_estimate_and_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, lines = _plan(tmp_path, "20.00", monkeypatch)

    assert code == 0
    [line] = lines
    assert line["pages"] == 1000 and line["gpu_hours"] == "10.50"  # 10 h + 2 x 0.25 h
    assert line["rate_usd_per_gpu_hour"] == "0.40" and line["cost_usd"] == "4.20"
    assert line["remaining_budget_usd"] == "20.00" and line["within_budget"] is True
    assert (tmp_path / "job" / "job.json").exists()


def test_plan_refuses_over_budget_with_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, lines = _plan(tmp_path, "4.19", monkeypatch)

    assert code == 2
    assert lines[0]["cost_usd"] == "4.20" and lines[0]["within_budget"] is False
    assert "exceeds the remaining budget" in str(lines[1]["refused"])
    assert not (tmp_path / "job").exists()


def test_verify_reports_each_book_and_fails_on_a_bad_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    books = [{"envelope_id": "E1", "sha256": "1" * 64, "pages": 2},
             {"envelope_id": "E2", "sha256": "2" * 64, "pages": 2}]  # fmt: skip
    (job / "books.json").write_text(json.dumps(books))
    cache = FakeCache(
        {ocr_key("1" * 64): b"{}", ocr_meta_key("1" * 64): b'{"pipeline": "ocr", "pages": 2}'}
    )
    monkeypatch.setattr(corpus_ocr, "_garage", lambda: (cache, "vorag-artifacts"))

    result = CliRunner().invoke(cli, ["corpus", "ocr-farm", "runpod", "verify", "--job-dir",
                                      str(job)])  # fmt: skip

    assert result.exit_code == 1
    lines = _lines(result.output)
    assert [line.get("ok") for line in lines[:2]] == [True, False]
    assert lines[-1] == {"books": 2, "verified": 1, "bad": 1}


def test_plan_defaults_to_6000_pages_per_gpu_hour_and_shows_reservations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(corpus_ocr, "_runner", lambda: FakeRunner(pitwall_replies()))
    args = ["corpus", "ocr-farm", "runpod", "plan", "--backlog", str(_backlog(tmp_path, 1000)),
            "--job-dir", str(tmp_path / "job")]  # fmt: skip

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 0, result.output
    [line] = _lines(result.output)
    assert line["pages_per_gpu_hour"] == "6000"
    assert {"ttl_minutes", "reservation_usd", "reserved_total_usd", "per_request_max_usd"} <= set(
        line
    )
    assert "measured 2026-09-26" in CliRunner().invoke(cli, [*args[:4], "--help"]).output.lower()


def test_plan_refuses_a_staged_job_dir_without_force_replan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(corpus_ocr, "_runner", lambda: FakeRunner(pitwall_replies()))
    (tmp_path / "job").mkdir()
    (tmp_path / "job" / "units.json").write_text("[]")
    args = ["corpus", "ocr-farm", "runpod", "plan", "--backlog", str(_backlog(tmp_path, 10)),
            "--job-dir", str(tmp_path / "job")]  # fmt: skip

    refused = CliRunner().invoke(cli, args)
    forced = CliRunner().invoke(cli, [*args, "--force-replan"])

    assert refused.exit_code == 1 and "already staged" in refused.output
    assert forced.exit_code == 0
    assert "warning" in _lines(forced.output)[0]


def test_plan_setup_concurrency_flag_sizes_the_queue_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(corpus_ocr, "_runner", lambda: FakeRunner(pitwall_replies()))
    args = ["corpus", "ocr-farm", "runpod", "plan", "--backlog", str(_backlog(tmp_path, 1200)),
            "--job-dir", str(tmp_path / "job"), "--pods", "3",
            "--setup-concurrency", "1"]  # fmt: skip

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 0, result.output
    [line] = _lines(result.output)
    assert line["setup_concurrency"] == 1 and line["queue_minutes"] == 30
    saved = json.loads((tmp_path / "job" / "job.json").read_text())
    assert saved["estimate"]["setup_concurrency"] == 1


def test_run_with_an_unreachable_pitwall_reports_an_error_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing pitwall binary is a one-line JSON error and
    exit 1, raised before anything is staged."""
    from vo_rag.ocr_farm.coordinator import subprocess_runner

    code, _ = _plan(tmp_path, "20.00", monkeypatch)
    assert code == 0
    cache = FakeCache()
    monkeypatch.setattr(corpus_ocr, "_garage", lambda: (cache, "vorag-artifacts"))
    monkeypatch.setattr(corpus_ocr, "_runner", lambda: subprocess_runner)

    result = CliRunner().invoke(
        cli,
        ["corpus", "ocr-farm", "runpod", "run", "--job-dir", str(tmp_path / "job"),
         "--pitwall", "/nonexistent/pitwall-binary"],
    )  # fmt: skip

    assert result.exit_code == 1, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    [line] = _lines(result.output)
    assert "cannot run" in str(line["error"])
    assert not (tmp_path / "job" / "units.json").exists()
    assert cache.gets == []

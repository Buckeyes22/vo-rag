"""`vo-rag corpus ocr-farm plan|bootstrap|run` (VOR-1427).

Postgres/Garage come from a monkeypatched ``_resources`` seam and ssh/scp
from a recorder; nothing real is touched.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from tests.ocr_farm._farm_fakes import FakeGarage, FakePool, FakeQdrant, Recorder
from vo_rag.cli import corpus_ocr
from vo_rag.cli.main import cli
from vo_rag.ocr_farm.coordinator import FarmItem, FarmReport, FarmResult
from vo_rag.ocr_farm.hosts import FarmHost


def _items() -> list[FarmItem]:
    return [
        FarmItem("E1", "k1.pdf", "1" * 64, "Big", 5000),
        FarmItem("E2", "k2.pdf", "2" * 64, "Small", 100),
        FarmItem("E3", "k3.pdf", "3" * 64, "Medium", 2400),
        FarmItem("E4", "k4.pdf", "4" * 64, "Cached", 900, cached=True),
        FarmItem("E5", "k5.bin", "5" * 64, "Odd", 0, problem="unsupported_format"),
    ]


@pytest.fixture
def seams(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    captured: dict[str, Any] = {"recorder": Recorder(), "qdrant": FakeQdrant()}

    @asynccontextmanager
    async def fake_resources():
        yield FakePool(), FakeGarage(), "vorag-artifacts", captured["qdrant"]

    async def fake_plan(pool: Any, garage: Any, qdrant: Any, *, bucket: str) -> list[FarmItem]:
        _ = (pool, garage, bucket)
        assert qdrant is captured["qdrant"]
        return _items()

    async def fake_run(
        items: list[FarmItem],
        hosts: tuple[FarmHost, ...],
        garage: Any,
        pool: Any,
        *,
        requeue: bool,
        bucket: str,
        runner: Any,
    ) -> FarmReport:
        captured.update(items=items, hosts=hosts, requeue=requeue, bucket=bucket)
        results = [
            FarmResult(i.envelope_id, i.title, i.sha256, "lin", i.pages, 1.5, "ok", requeue)
            for i in items
        ]
        return FarmReport(results)

    monkeypatch.setattr(corpus_ocr, "_resources", fake_resources)
    monkeypatch.setattr(corpus_ocr, "plan_backlog", fake_plan)
    monkeypatch.setattr(corpus_ocr, "run_farm", fake_run)
    monkeypatch.setattr(corpus_ocr, "_runner", lambda: captured["recorder"])
    monkeypatch.setenv("VORAG_OCR_FARM_STATE", str(tmp_path / "hosts.json"))
    return captured


def _lines(output: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in output.splitlines() if line.strip()]


def test_corpus_group_lists_ocr_farm() -> None:
    result = CliRunner().invoke(cli, ["corpus", "ocr-farm", "--help"])
    assert result.exit_code == 0, result.output
    for sub in ("plan", "bootstrap", "run"):
        assert sub in result.output


def test_plan_prints_books_pages_and_estimated_hours(seams: dict[str, Any]) -> None:
    result = CliRunner().invoke(cli, ["corpus", "ocr-farm", "plan"])

    assert result.exit_code == 0, result.output
    lines = _lines(result.stdout)
    assert [line["envelope_id"] for line in lines[:-1]] == ["E1", "E3", "E4", "E2", "E5"]
    summary = lines[-1]
    assert summary == {
        "books": 5,
        "to_ocr": 3,
        "cached": 1,
        "problems": 1,
        "pages": 7500,
        "s_per_page": 0.48,
        "est_hours": 1.0,
    }


def test_plan_summary_counts_already_ingested_as_a_problem(
    seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``already_ingested`` items count in ``problems``
    and are excluded from ``to_ocr``, same as any other plan problem."""
    already = FarmItem(
        "E2",
        "k2.pdf",
        "2" * 64,
        "",
        300,
        problem="already_ingested",
        duplicate_of="vo-rag-corpus-sha256:" + "2" * 64,
    )
    dup = FarmItem(
        "E3", "k3.pdf", "3" * 64, "Dup", 50, problem="duplicate_sha256", duplicate_of="E1"
    )

    async def fake_plan(pool: Any, garage: Any, qdrant: Any, *, bucket: str) -> list[FarmItem]:
        _ = (pool, garage, qdrant, bucket)
        return [FarmItem("E1", "k1.pdf", "1" * 64, "Big", 5000), already, dup]

    monkeypatch.setattr(corpus_ocr, "plan_backlog", fake_plan)
    result = CliRunner().invoke(cli, ["corpus", "ocr-farm", "plan"])

    assert result.exit_code == 0, result.output
    summary = _lines(result.stdout)[-1]
    assert summary["problems"] == 2
    assert summary["to_ocr"] == 1
    assert summary["books"] == 3


def test_run_limit_picks_smallest_pending_books_and_honours_flags(seams: dict[str, Any]) -> None:
    result = CliRunner().invoke(
        cli, ["corpus", "ocr-farm", "run", "--limit", "2", "--no-requeue", "--hosts", "lin"]
    )

    assert result.exit_code == 0, result.output
    assert [i.envelope_id for i in seams["items"]] == ["E2", "E3"]
    assert seams["requeue"] is False
    assert [h.alias for h in seams["hosts"]] == ["lin"]
    assert seams["bucket"] == "vorag-artifacts"
    lines = _lines(result.stdout)
    assert lines[-1]["ok"] == 2
    assert lines[-1]["failed"] == 0


def test_run_without_limit_passes_everything_and_requeues(seams: dict[str, Any]) -> None:
    result = CliRunner().invoke(cli, ["corpus", "ocr-farm", "run"])

    assert result.exit_code == 0, result.output
    assert len(seams["items"]) == 5
    assert seams["requeue"] is True
    assert [h.alias for h in seams["hosts"]] == ["win", "lin"]


def test_run_exits_1_when_a_book_failed(
    seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing_run(items: list[FarmItem], *args: Any, **kwargs: Any) -> FarmReport:
        _ = (args, kwargs)
        return FarmReport(
            [FarmResult("E1", "Big", "1" * 64, "lin", 5000, 3.0, "failed(worker_exit_4)", False)]
        )

    monkeypatch.setattr(corpus_ocr, "run_farm", failing_run)
    result = CliRunner().invoke(cli, ["corpus", "ocr-farm", "run"])

    assert result.exit_code == 1
    assert _lines(result.stdout)[0]["status"] == "failed(worker_exit_4)"


def test_unknown_host_is_a_usage_error(seams: dict[str, Any]) -> None:
    result = CliRunner().invoke(cli, ["corpus", "ocr-farm", "run", "--hosts", "nope"])
    assert result.exit_code == 2
    assert "nope" in result.output


def test_bootstrap_prints_one_line_per_host(seams: dict[str, Any]) -> None:
    result = CliRunner().invoke(cli, ["corpus", "ocr-farm", "bootstrap"])

    assert result.exit_code == 0, result.output
    lines = _lines(result.stdout)
    assert [line["host"] for line in lines] == ["win", "lin"]
    assert all(line["ok"] for line in lines)
    assert lines[0]["check"].startswith("torch 2.12.0+cu130 cuda True")


def test_bootstrap_exits_1_on_failure(seams: dict[str, Any]) -> None:
    seams["recorder"].fail_commands["pip install"] = 1
    result = CliRunner().invoke(cli, ["corpus", "ocr-farm", "bootstrap", "--hosts", "lin"])

    assert result.exit_code == 1
    assert _lines(result.stdout)[0]["ok"] is False


def test_bootstrap_timeout_on_one_host_leaves_the_other_unaffected(
    seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from vo_rag.ocr_farm.coordinator import BootstrapTimeouts, bootstrap_worker

    fast = BootstrapTimeouts(command_s=0.2, install_s=0.2, check_s=0.2, backstop_s=0.0)

    async def fast_bootstrap(host: FarmHost, *, runner: Any) -> Any:
        return await bootstrap_worker(host, runner=runner, timeouts=fast)

    monkeypatch.setattr(corpus_ocr, "bootstrap_worker", fast_bootstrap)
    seams["recorder"].hang_commands.append("~/.local/bin/uv pip install")

    result = CliRunner().invoke(cli, ["corpus", "ocr-farm", "bootstrap"])

    assert result.exit_code == 1
    lines = {line["host"]: line for line in _lines(result.stdout)}
    assert lines["win"]["ok"] is True
    assert lines["lin"]["ok"] is False
    assert "pip install timeout" in lines["lin"]["error"]


def test_ocr_farm_commands_show_coordinator_progress_on_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """book_done is INFO; without a handler it never appeared until the summary."""
    import logging

    corpus_ocr._enable_progress_logging()
    corpus_ocr._enable_progress_logging()  # idempotent: one handler, one line
    logging.getLogger("vo_rag.ocr_farm.coordinator").info("ocr_farm.book_done host=lin x=1")
    logging.getLogger("httpx").info("GET /noise")

    err = capsys.readouterr().err
    assert err.count("ocr_farm.book_done host=lin x=1") == 1
    assert "/noise" not in err


def test_sigterm_cancels_the_run_so_cleanup_runs() -> None:
    """systemctl stop (SIGTERM) must reach run_farm's cancellation path."""
    import asyncio
    import os
    import signal

    cleaned: list[str] = []

    async def main() -> str:
        try:
            await asyncio.sleep(0)
            os.kill(os.getpid(), signal.SIGTERM)
            await asyncio.sleep(30)
            return "not cancelled"
        except asyncio.CancelledError:
            await asyncio.sleep(0)  # cleanup can still await (like _kill_remote)
            cleaned.append("killed remote worker")
            raise

    with pytest.raises(asyncio.CancelledError):
        corpus_ocr._run_until_signalled(main)
    assert cleaned == ["killed remote worker"]


def test_cancelled_run_exits_130_after_emitting_cancelled(
    seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    async def cancelled_run(*args: Any, **kwargs: Any) -> Any:
        raise asyncio.CancelledError

    monkeypatch.setattr(corpus_ocr, "run_farm", cancelled_run)
    result = CliRunner().invoke(cli, ["corpus", "ocr-farm", "run"])

    assert result.exit_code == 130
    assert _lines(result.output)[-1] == {"cancelled": True}

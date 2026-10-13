"""OCR-farm coordinator: scheduling, failures, cache skip, host commands (VOR-1427).

ssh/scp are faked by a recorder, Garage and Postgres by in-memory fakes. No
real host, bucket or database is touched.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.ocr_farm._farm_fakes import (
    BUCKET,
    FakeGarage,
    FakePool,
    Recorder,
    farm_items,
    resolved_hosts,
)
from vo_rag.ocr_farm import coordinator
from vo_rag.ocr_farm.coordinator import (
    REQUEUE_SQL,
    FarmItem,
    bootstrap_worker,
    resolve_hosts,
    run_farm,
)
from vo_rag.ocr_farm.hosts import DEFAULT_HOSTS, FarmHost


def test_default_hosts() -> None:
    win, lin = DEFAULT_HOSTS
    assert win == FarmHost(
        "win", "windows", "%TEMP%\\vorag-ocr", 2, r".venv\Scripts\python.exe", uv="uv"
    )
    assert lin == FarmHost(
        "lin",
        "linux",
        "/tmp/vorag-ocr",
        2,
        ".venv/bin/python",
        uv="~/.local/bin/uv",
        gpu_mem_fraction=0.25,
    )


async def test_resolve_hosts_resolves_windows_temp_once_and_keeps_linux_path() -> None:
    recorder = Recorder()
    win, lin = await resolved_hosts(recorder)

    assert win.scp_workdir == "C:/Users/Demo/AppData/Local/Temp/vorag-ocr"
    assert lin.scp_workdir == "/tmp/vorag-ocr"
    assert recorder.calls == [["ssh", "win", "echo %TEMP%"]]


async def test_resolve_hosts_uses_and_writes_state_file(tmp_path: Path) -> None:
    state = tmp_path / "hosts.json"
    recorder = Recorder()
    await resolve_hosts(DEFAULT_HOSTS, recorder, state_path=state)
    assert json.loads(state.read_text())["win"] == ("C:/Users/Demo/AppData/Local/Temp/vorag-ocr")

    again = Recorder()
    win, _ = await resolve_hosts(DEFAULT_HOSTS, again, state_path=state)
    assert win.scp_workdir == "C:/Users/Demo/AppData/Local/Temp/vorag-ocr"
    assert again.calls == []


async def test_longest_first_fills_win_then_z2_and_respects_slots() -> None:
    pages = [120, 900, 40, 700, 300, 850, 60, 500, 220, 610]
    items, objects = farm_items(pages)
    garage = FakeGarage(objects)
    pool = FakePool()
    recorder = Recorder(pages_by_sha={i.sha256: i.pages for i in items})
    hosts = await resolved_hosts(recorder)

    # Whole-book scheduling: no book is split here (the stand-in bytes aren't
    # real PDFs); test_farm_split.py covers books over SPLIT_PAGES.
    report = await run_farm(
        items,
        hosts,
        garage,
        pool,
        requeue=True,
        bucket=BUCKET,
        runner=recorder,
        split_pages=max(pages),
    )

    by_pages = sorted(items, key=lambda i: i.pages, reverse=True)
    host_of = {r.envelope_id: r.host for r in report.results}
    # The report lists books in dispatch order: longest first.
    assert [r.envelope_id for r in report.results][:4] == [i.envelope_id for i in by_pages[:4]]
    assert [host_of[i.envelope_id] for i in by_pages[:4]] == ["win"] * 2 + ["lin"] * 2
    assert recorder.peak == {"win": 2, "lin": 2}
    assert [r.status for r in report.results].count("ok") == len(items)
    for item in items:
        assert (BUCKET, f"ocr/{item.sha256}/docling.json") in garage.objects
    assert len(pool.execute_calls) == len(items)
    assert {args for _, args in pool.execute_calls} == {(i.envelope_id,) for i in items}


async def test_every_host_slot_limit_holds_with_many_books() -> None:
    items, objects = farm_items([10 * n + 5 for n in range(25)])
    recorder = Recorder(pages_by_sha={i.sha256: i.pages for i in items})
    hosts = await resolved_hosts(recorder)

    await run_farm(
        items, hosts, FakeGarage(objects), FakePool(), requeue=False, bucket=BUCKET, runner=recorder
    )

    assert recorder.peak["win"] <= 4
    assert recorder.peak["lin"] <= 2
    assert len(recorder.worker_starts) == 25


async def test_worker_failure_marks_failed_cleans_frees_slot_and_does_not_requeue() -> None:
    items, objects = farm_items([300, 200, 100])
    bad = items[0]
    recorder = Recorder(pages_by_sha={i.sha256: i.pages for i in items}, exit_codes={bad.sha256: 4})
    one_slot = (replace(DEFAULT_HOSTS[1], slots=1),)
    hosts = await resolved_hosts(recorder, one_slot)
    garage = FakeGarage(objects)
    pool = FakePool()

    report = await run_farm(
        items, hosts, garage, pool, requeue=True, bucket=BUCKET, runner=recorder
    )

    results = {r.envelope_id: r for r in report.results}
    assert results[bad.envelope_id].status == "failed(worker_exit_4)"
    assert results[bad.envelope_id].requeued is False
    assert results[bad.envelope_id].host == "lin"
    # The slot was freed: the other two books still ran on the single slot.
    assert results[items[1].envelope_id].status == "ok"
    assert results[items[2].envelope_id].status == "ok"
    assert recorder.peak == {"lin": 1}
    # Worker files for the failed book were removed.
    cleanup = [c for c in recorder.commands_for("lin") if c.startswith("rm -f") and bad.sha256 in c]
    jobs = f"/tmp/vorag-ocr/jobs/{bad.sha256}"
    assert cleanup == [f"rm -f {jobs}.pdf {jobs}.json {jobs}.json.part {jobs}.pid"]
    # Never re-queued, never retried, nothing stored.
    assert (bad.envelope_id,) not in {args for _, args in pool.execute_calls}
    assert [sha for _, sha in recorder.worker_starts].count(bad.sha256) == 1
    assert (BUCKET, f"ocr/{bad.sha256}/docling.json") not in garage.objects


async def test_scp_failure_is_a_named_failure_and_cleans_up() -> None:
    items, objects = farm_items([50])
    recorder = Recorder(fail_commands={f"jobs/{items[0].sha256}.pdf": 1})
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))

    report = await run_farm(
        items, hosts, FakeGarage(objects), FakePool(), requeue=True, bucket=BUCKET, runner=recorder
    )

    assert report.results[0].status == "failed(scp_upload_exit_1)"
    assert any(c.startswith("rm -f") for c in recorder.commands_for("lin"))


async def test_book_with_existing_ocr_object_is_skipped() -> None:
    items, objects = farm_items([400, 100])
    cached = items[0]
    objects[f"ocr/{cached.sha256}/docling.json"] = b'{"schema_name": "DoclingDocument"}'
    recorder = Recorder(pages_by_sha={i.sha256: i.pages for i in items})
    hosts = await resolved_hosts(recorder)
    garage = FakeGarage(objects)

    report = await run_farm(
        items, hosts, garage, FakePool(), requeue=False, bucket=BUCKET, runner=recorder
    )

    results = {r.envelope_id: r for r in report.results}
    assert results[cached.envelope_id].status == "cached"
    assert results[items[1].envelope_id].status == "ok"
    assert all(cached.sha256 not in " ".join(argv) for argv in recorder.calls)
    assert (BUCKET, cached.garage_key) not in garage.gets


async def test_cached_book_is_requeued_when_requested() -> None:
    items, objects = farm_items([400])
    objects[f"ocr/{items[0].sha256}/docling.json"] = b"{}"
    pool = FakePool()
    recorder = Recorder()
    hosts = await resolved_hosts(recorder)

    report = await run_farm(
        items, hosts, FakeGarage(objects), pool, requeue=True, bucket=BUCKET, runner=recorder
    )

    assert report.results[0].status == "cached"
    assert report.results[0].requeued is True
    assert pool.execute_calls == [(REQUEUE_SQL, (items[0].envelope_id,))]


async def test_requeue_sql_keeps_failed_guard_and_no_requeue_skips_it() -> None:
    assert REQUEUE_SQL.startswith(
        "UPDATE intake_envelopes SET status='received', error=NULL, updated_at=NOW(), "
    )
    assert REQUEUE_SQL.endswith("WHERE id=$1 AND status='failed'")
    # VOR-1434: a requeued envelope must not keep a streaming ingest's flag
    # (the dispatcher's claim clears it the same way).
    assert "'{metrics,active_stream}', 'false'::jsonb" in REQUEUE_SQL
    items, objects = farm_items([10])
    pool = FakePool()
    recorder = Recorder()
    hosts = await resolved_hosts(recorder)

    report = await run_farm(
        items, hosts, FakeGarage(objects), pool, requeue=False, bucket=BUCKET, runner=recorder
    )

    assert report.results[0].status == "ok"
    assert report.results[0].requeued is False
    assert pool.execute_calls == []


async def test_sha_mismatch_fails_without_shipping() -> None:
    items, objects = farm_items([10])
    objects[items[0].garage_key] = b"%PDF-1.4\n% changed\n"
    recorder = Recorder()
    hosts = await resolved_hosts(recorder)

    report = await run_farm(
        items, hosts, FakeGarage(objects), FakePool(), requeue=True, bucket=BUCKET, runner=recorder
    )

    assert report.results[0].status == "failed(sha256_mismatch)"
    assert recorder.worker_starts == []


async def test_item_with_plan_problem_fails_with_that_reason() -> None:
    items, objects = farm_items([10])
    items[0] = replace(items[0], problem="unsupported_format")
    recorder = Recorder()
    hosts = await resolved_hosts(recorder)

    report = await run_farm(
        items, hosts, FakeGarage(objects), FakePool(), requeue=True, bucket=BUCKET, runner=recorder
    )

    assert report.results[0].status == "failed(unsupported_format)"
    assert recorder.worker_starts == []


async def test_already_ingested_item_is_never_ocrd_or_requeued() -> None:
    """Bytes already ingested under another envelope must
    not be OCR'd, and must stay failed rather than being re-queued."""
    items, objects = farm_items([10])
    items[0] = replace(
        items[0],
        problem="already_ingested",
        duplicate_of="vo-rag-corpus-sha256:" + items[0].sha256,
    )
    recorder = Recorder()
    hosts = await resolved_hosts(recorder)
    pool = FakePool()

    report = await run_farm(
        items, hosts, FakeGarage(objects), pool, requeue=True, bucket=BUCKET, runner=recorder
    )

    assert report.results[0].status == "failed(already_ingested)"
    assert report.results[0].requeued is False
    assert recorder.worker_starts == []
    assert pool.execute_calls == []


async def test_invalid_worker_output_is_a_named_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    items, objects = farm_items([10])
    recorder = Recorder()
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))
    original = recorder.__call__

    async def garbage_download(argv):
        result = await original(argv)
        if argv[0] == "scp" and argv[-2].endswith(".json"):
            Path(argv[-1]).write_text("not json", encoding="utf-8")
        return result

    garage = FakeGarage(objects)
    report = await run_farm(
        items, hosts, garage, FakePool(), requeue=True, bucket=BUCKET, runner=garbage_download
    )

    assert report.results[0].status == "failed(invalid_output)"
    assert (BUCKET, f"ocr/{items[0].sha256}/docling.json") not in garage.objects


async def test_epub_item_is_converted_to_pdf_before_shipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    epub = b"PK\x03\x04 fake epub"
    sha = hashlib.sha256(epub).hexdigest()
    item = FarmItem("ENVEPUB", "uploads/scan.epub", sha, "Scan", 12, fmt="epub")
    shipped: list[bytes] = []

    def fake_epub_to_pdf(data: bytes) -> bytes:
        assert data == epub
        return b"%PDF-1.4\n% synthesized\n"

    monkeypatch.setattr(coordinator, "epub_to_pdf_bytes", fake_epub_to_pdf)
    recorder = Recorder()
    original = recorder.__call__

    async def capture(argv):
        if argv[0] == "scp" and argv[-2].endswith(".pdf"):
            shipped.append(Path(argv[-2]).read_bytes())
        return await original(argv)

    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))
    garage = FakeGarage({item.garage_key: epub})
    report = await run_farm(
        [item], hosts, garage, FakePool(), requeue=False, bucket=BUCKET, runner=capture
    )

    assert report.results[0].status == "ok"
    assert shipped == [b"%PDF-1.4\n% synthesized\n"]
    # Stored under the EPUB's own hash -- the one the parse step computes.
    assert (BUCKET, f"ocr/{sha}/docling.json") in garage.objects


async def test_stores_docling_json_before_meta_sidecar() -> None:
    """VOR-1434: watchers flip an envelope when meta.json appears, so the
    document must already be there."""
    items, objects = farm_items([10])
    recorder = Recorder()
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))
    garage = FakeGarage(objects)

    await run_farm(items, hosts, garage, FakePool(), requeue=False, bucket=BUCKET, runner=recorder)

    sha = items[0].sha256
    assert [key for _, key, _ in garage.puts] == [
        f"ocr/{sha}/docling.json",
        f"ocr/{sha}/meta.json",
    ]
    meta = json.loads(garage.objects[(BUCKET, f"ocr/{sha}/meta.json")])
    assert meta["host"] == "lin"
    # The router tells farm OCR apart from an off-box native pre-parse by this.
    assert meta["pipeline"] == "ocr"
    assert meta["envelope_id"] == items[0].envelope_id
    assert meta["worker"] == "ocr_worker: ok status=success gpu_peak_bytes=4096"


# ---------------------------------------------------------------------------
# Host command syntax
# ---------------------------------------------------------------------------


async def test_windows_commands_use_cmd_syntax_and_temp() -> None:
    items, objects = farm_items([10])
    sha = items[0].sha256
    recorder = Recorder()
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[0],))

    await run_farm(
        items, hosts, FakeGarage(objects), FakePool(), requeue=False, bucket=BUCKET, runner=recorder
    )

    commands = recorder.commands_for("win")
    assert 'if not exist "%TEMP%\\vorag-ocr\\jobs" mkdir "%TEMP%\\vorag-ocr\\jobs"' in commands
    assert (
        'cd /d "%TEMP%\\vorag-ocr" && set HF_HOME=%TEMP%\\vorag-ocr\\hf&& set HF_TOKEN=&& '
        f".venv\\Scripts\\python.exe ocr_worker.py jobs\\{sha}.pdf jobs\\{sha}.json"
    ) in commands
    jobs = f"%TEMP%\\vorag-ocr\\jobs\\{sha}"
    assert (f'del /f /q "{jobs}.pdf" "{jobs}.json" "{jobs}.json.part" "{jobs}.pid"') in commands
    scp = recorder.scp_calls()
    remote = "win:C:/Users/Demo/AppData/Local/Temp/vorag-ocr"
    assert any(dst == f"{remote}/jobs/{sha}.pdf" for _, dst in scp)
    assert any(src == f"{remote}/jobs/{sha}.json" for src, _ in scp)
    assert any(dst == f"{remote}/ocr_worker.py" for _, dst in scp)
    assert all("/tmp/vorag-ocr" not in c for c in commands)


async def test_linux_commands_use_tmp_and_unset_hf_token() -> None:
    items, objects = farm_items([10])
    sha = items[0].sha256
    recorder = Recorder()
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))

    await run_farm(
        items, hosts, FakeGarage(objects), FakePool(), requeue=False, bucket=BUCKET, runner=recorder
    )

    commands = recorder.commands_for("lin")
    assert "mkdir -p /tmp/vorag-ocr/jobs" in commands
    assert (
        "cd /tmp/vorag-ocr && env -u HF_TOKEN HF_HOME=/tmp/vorag-ocr/hf "
        "VORAG_OCR_GPU_MEM_FRACTION=0.25 "
        f".venv/bin/python ocr_worker.py jobs/{sha}.pdf jobs/{sha}.json"
    ) in commands
    scp = recorder.scp_calls()
    assert any(dst == f"lin:/tmp/vorag-ocr/jobs/{sha}.pdf" for _, dst in scp)
    assert any(src == f"lin:/tmp/vorag-ocr/jobs/{sha}.json" for src, _ in scp)
    assert all("%TEMP%" not in c for c in commands)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_PIP = (
    "pip install --python {py} --extra-index-url https://download.pytorch.org/whl/cu130 "
    "--index-strategy unsafe-best-match -r requirements.txt"
)


async def test_bootstrap_windows_worker() -> None:
    recorder = Recorder()
    (win,) = await resolved_hosts(recorder, (DEFAULT_HOSTS[0],))

    result = await bootstrap_worker(win, runner=recorder)

    assert result.ok is True
    assert result.check == "torch 2.12.0+cu130 cuda True NVIDIA GeForce RTX 5070"
    commands = recorder.commands_for("win")
    # Every long step runs through the shipped farm_step.ps1 wrapper, which
    # enforces the timeout on the remote side (it stops that exact PID).
    step = (
        'cd /d "%TEMP%\\vorag-ocr" && powershell -NoProfile -ExecutionPolicy Bypass '
        '-File farm_step.ps1 -TimeoutSeconds {secs} -FilePath {exe} -Arguments "{args}"'
    )
    assert commands[1:] == [
        'if not exist "%TEMP%\\vorag-ocr\\jobs" mkdir "%TEMP%\\vorag-ocr\\jobs"',
        step.format(secs=3600, exe="uv", args="venv --allow-existing --python 3.12 .venv"),
        step.format(secs=3600, exe="uv", args=_PIP.format(py=".venv\\Scripts\\python.exe")),
        step.format(secs=300, exe=".venv\\Scripts\\python.exe", args="ocr_worker.py --check"),
    ]
    remote = "win:C:/Users/Demo/AppData/Local/Temp/vorag-ocr"
    assert {dst for _, dst in recorder.scp_calls()} == {
        f"{remote}/ocr_worker.py",
        f"{remote}/requirements.txt",
        f"{remote}/farm_step.ps1",
    }


async def test_bootstrap_linux_worker() -> None:
    recorder = Recorder(check_output="torch 2.12.0+cu130 cuda True NVIDIA T1000 8GB")
    (lin,) = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))

    result = await bootstrap_worker(lin, runner=recorder)

    assert result.ok is True
    assert result.check == "torch 2.12.0+cu130 cuda True NVIDIA T1000 8GB"
    # coreutils timeout enforces each bound on the remote side.
    wd = "cd /tmp/vorag-ocr && timeout --kill-after=30 "
    assert recorder.commands_for("lin") == [
        "mkdir -p /tmp/vorag-ocr/jobs",
        wd + "3600 ~/.local/bin/uv venv --allow-existing --python 3.12 .venv",
        wd + "3600 ~/.local/bin/uv " + _PIP.format(py=".venv/bin/python"),
        wd + "300 .venv/bin/python ocr_worker.py --check",
    ]
    assert {dst for _, dst in recorder.scp_calls()} == {
        "lin:/tmp/vorag-ocr/ocr_worker.py",
        "lin:/tmp/vorag-ocr/requirements.txt",
    }


async def test_bootstrap_reports_failed_step() -> None:
    recorder = Recorder(fail_commands={"pip install": 2})
    (lin,) = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))

    result = await bootstrap_worker(lin, runner=recorder)

    assert result.ok is False
    assert "pip" in result.error
    assert not any("--check" in c for c in recorder.commands_for("lin"))


async def test_bootstrap_rejects_cuda_false() -> None:
    recorder = Recorder(check_output="torch 2.12.0+cu130 cuda False")
    (lin,) = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))

    result = await bootstrap_worker(lin, runner=recorder)

    assert result.ok is False

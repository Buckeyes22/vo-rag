"""OCR farm resilience (VOR-1427): host faults, timeouts, cancellation, duplicates.

ssh/scp are faked by the recorder; the one real subprocess is a local
``sh``/``sleep`` used to prove the runner kills its child on timeout.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tests.ocr_farm._farm_fakes import (
    BUCKET,
    FakeGarage,
    FakePool,
    Recorder,
    farm_items,
    resolved_hosts,
)
from vo_rag.ocr_farm.coordinator import (
    REQUEUE_SQL,
    BootstrapTimeouts,
    FarmItem,
    FarmTimeouts,
    bootstrap_worker,
    part_key,
    run_farm,
    subprocess_runner,
)
from vo_rag.ocr_farm.hosts import DEFAULT_HOSTS, FarmHost

#: Short command/probe bounds: the fake answers every command without
#: suspending, so only a hung one (``hang_commands``) can spend them. The fake
#: worker does sleep, so its budget is several seconds: a loaded CI runner
#: can't time out a job that should succeed (VOR-1434: a 0.3 s budget lost to
#: a stalled loop and a host-fault exit read as a timeout). Workers meant to
#: time out do so on cue (``Recorder.expire_*``), not by racing the clock.
FAST = FarmTimeouts(
    command_s=1.0,
    ocr_startup_s=0.0,
    ocr_s_per_page=0.0,
    ocr_min_s=10.0,
    backstop_s=0.0,
    probe_s=0.5,
    pid_retry_s=0.0,
)


def _attempt_key(item: FarmItem, gen: int) -> str:
    """The worker file stem of an unsplit book's attempt ``gen``: its sha for
    the first attempt, then scoped to the attempt (a stalled host's revival)."""
    return item.sha256 if gen == 0 else part_key(item.sha256, 1, item.pages, gen)


# ---------------------------------------------------------------------------
# 1. host health
# ---------------------------------------------------------------------------


async def test_prepare_host_requires_cuda_check() -> None:
    items, objects = farm_items([50, 40, 30])
    recorder = Recorder(check_by_host={"lin": "torch 2.12.0+cu130 cuda False"})
    hosts = await resolved_hosts(recorder)

    report = await run_farm(
        items, hosts, FakeGarage(objects), FakePool(), requeue=False, bucket=BUCKET, runner=recorder
    )

    assert "lin" in report.host_errors
    assert "cuda False" in report.host_errors["lin"]
    assert {r.host for r in report.results} == {"win"}
    assert all(r.status == "ok" for r in report.results)
    assert not [a for a, _ in recorder.worker_starts if a == "lin"]


async def test_host_that_always_exits_3_is_dropped_and_its_books_go_elsewhere() -> None:
    items, objects = farm_items([900, 800, 700, 600, 500, 400, 300, 200, 100, 50])
    recorder = Recorder(
        pages_by_sha={i.sha256: i.pages for i in items}, host_exit_codes={"lin": 69}
    )
    hosts = await resolved_hosts(recorder)
    pool = FakePool()

    # Whole books (the stand-in bytes aren't real PDFs); test_farm_split.py
    # covers a host fault on one part of a split book.
    report = await run_farm(
        items,
        hosts,
        FakeGarage(objects),
        pool,
        requeue=True,
        bucket=BUCKET,
        runner=recorder,
        split_pages=900,
    )

    assert "lin" in report.host_errors
    assert "worker_exit_69" in report.host_errors["lin"]
    assert [r.status for r in report.results] == ["ok"] * len(items)
    assert {r.host for r in report.results} == {"win"}
    assert {args for _, args in pool.execute_calls} == {(i.envelope_id,) for i in items}
    # lin got its initial 2 books, faulted, and was never dispatched to again.
    assert [a for a, _ in recorder.worker_starts].count("lin") == 2
    assert recorder.peak["win"] <= 4
    # The report still lists every book exactly once.
    assert sorted(r.envelope_id for r in report.results) == sorted(i.envelope_id for i in items)


@pytest.mark.parametrize(("bad_host", "code"), [("lin", 127), ("win", 9009), ("lin", 255)])
async def test_missing_interpreter_or_unreachable_host_is_a_host_fault(
    bad_host: str, code: int
) -> None:
    items, objects = farm_items([300, 200, 100, 50, 40, 30, 20])
    recorder = Recorder(host_exit_codes={bad_host: code})
    hosts = await resolved_hosts(recorder)

    report = await run_farm(
        items, hosts, FakeGarage(objects), FakePool(), requeue=False, bucket=BUCKET, runner=recorder
    )

    assert f"worker_exit_{code}" in report.host_errors[bad_host]
    assert all(r.status == "ok" for r in report.results)
    assert bad_host not in {r.host for r in report.results}


async def test_book_specific_worker_failure_still_fails_the_book_not_the_host() -> None:
    items, objects = farm_items([300, 200, 100])
    bad = items[0]
    recorder = Recorder(exit_codes={bad.sha256: 4})
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))

    report = await run_farm(
        items, hosts, FakeGarage(objects), FakePool(), requeue=True, bucket=BUCKET, runner=recorder
    )

    statuses = {r.envelope_id: r.status for r in report.results}
    assert statuses[bad.envelope_id] == "failed(worker_exit_4)"
    assert report.host_errors == {}
    assert [statuses[i.envelope_id] for i in items[1:]] == ["ok", "ok"]


async def test_all_hosts_faulted_marks_remaining_books_failed_without_raising() -> None:
    items, objects = farm_items([300, 200, 100, 50])
    recorder = Recorder(host_exit_codes={"win": 69, "lin": 69})
    hosts = await resolved_hosts(recorder)

    report = await run_farm(
        items, hosts, FakeGarage(objects), FakePool(), requeue=True, bucket=BUCKET, runner=recorder
    )

    assert set(report.host_errors) == {"win", "lin"}
    assert [r.status for r in report.results] == ["failed(no_usable_host)"] * 4
    assert all(not r.requeued for r in report.results)


async def test_no_usable_host_at_start_reports_instead_of_raising() -> None:
    items, objects = farm_items([10])
    recorder = Recorder(check_output="torch 2.12.0+cu130 cuda False")
    hosts = await resolved_hosts(recorder)

    report = await run_farm(
        items, hosts, FakeGarage(objects), FakePool(), requeue=True, bucket=BUCKET, runner=recorder
    )

    assert set(report.host_errors) == {"win", "lin"}
    assert report.results[0].status == "failed(no_usable_host)"
    assert recorder.worker_starts == []


async def test_prepare_host_exception_is_a_host_error_and_other_host_continues() -> None:
    items, objects = farm_items([30, 20])
    recorder = Recorder(raise_commands={"mkdir -p /tmp/vorag-ocr/jobs": FileNotFoundError("ssh")})
    hosts = await resolved_hosts(recorder)

    report = await run_farm(
        items, hosts, FakeGarage(objects), FakePool(), requeue=False, bucket=BUCKET, runner=recorder
    )

    assert "FileNotFoundError" in report.host_errors["lin"]
    assert all(r.status == "ok" and r.host == "win" for r in report.results)


# ---------------------------------------------------------------------------
# 4. timeouts and cancellation
# ---------------------------------------------------------------------------


def _linux_kill(sha: str, pid: int = 4242) -> str:
    return (
        f"grep -qa ocr_worker.py /proc/{pid}/cmdline && grep -qa {sha} /proc/{pid}/cmdline "
        f"&& kill {pid}"
    )


def _windows_kill(sha: str, pid: int = 4242) -> str:
    return (
        'powershell -NoProfile -Command "'
        f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; "
        "if ($p -and $p.CommandLine -like '*ocr_worker.py*' "
        f"-and $p.CommandLine -like '*{sha}*') "
        f'{{ Stop-Process -Id {pid} -Force }} else {{ exit 1 }}"'
    )


def _no_kill(commands: list[str]) -> bool:
    return not [c for c in commands if c.startswith(("cat ", "type ", "grep ", "powershell"))]


def test_timeout_bounds() -> None:
    timeouts = FarmTimeouts()
    assert timeouts.command_s == 300.0
    assert timeouts.ocr_s(1) == 300.0
    assert timeouts.ocr_s(30) == 420.0  # 120 s model load + 30 * 10 s
    assert timeouts.ocr_s(100) == 1120.0


async def test_hung_linux_worker_times_out_kills_remote_cleans_and_keeps_host() -> None:
    items, objects = farm_items([300, 100])
    hung = items[0]
    recorder = Recorder(expire_shas={hung.sha256})
    hosts = await resolved_hosts(recorder, (replace(DEFAULT_HOSTS[1], slots=1),))
    pool = FakePool()

    report = await run_farm(
        items,
        hosts,
        FakeGarage(objects),
        pool,
        requeue=True,
        bucket=BUCKET,
        runner=recorder,
        timeouts=FAST,
    )

    statuses = {r.envelope_id: r.status for r in report.results}
    assert statuses[hung.envelope_id] == "failed(timeout)"
    assert statuses[items[1].envelope_id] == "ok"
    assert report.host_errors == {}
    assert (hung.envelope_id,) not in {args for _, args in pool.execute_calls}
    assert recorder.expired == [("lin", hung.sha256)]
    commands = recorder.commands_for("lin")
    jobs = f"/tmp/vorag-ocr/jobs/{hung.sha256}"
    read_pid = commands.index(f"cat {jobs}.pid")
    kill = commands.index(_linux_kill(hung.sha256))
    cleanup = commands.index(f"rm -f {jobs}.pdf {jobs}.json {jobs}.json.part {jobs}.pid")
    assert read_pid < kill < cleanup


async def test_hung_windows_worker_is_killed_by_verified_pid() -> None:
    items, objects = farm_items([300])
    sha = items[0].sha256
    recorder = Recorder(expire_shas={sha})
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[0],))

    report = await run_farm(
        items,
        hosts,
        FakeGarage(objects),
        FakePool(),
        requeue=False,
        bucket=BUCKET,
        runner=recorder,
        timeouts=FAST,
    )

    assert report.results[0].status == "failed(timeout)"
    commands = recorder.commands_for("win")
    assert f'type "%TEMP%\\vorag-ocr\\jobs\\{sha}.pid"' in commands
    assert _windows_kill(sha) in commands
    assert not [c for c in commands if "taskkill" in c]


async def test_hung_scp_times_out_as_failed_timeout() -> None:
    items, objects = farm_items([10])
    recorder = Recorder(hang_commands=[f"jobs/{items[0].sha256}.pdf"])
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))

    started = time.monotonic()
    report = await run_farm(
        items,
        hosts,
        FakeGarage(objects),
        FakePool(),
        requeue=False,
        bucket=BUCKET,
        runner=recorder,
        timeouts=FAST,
    )

    assert report.results[0].status == "failed(timeout)"
    assert time.monotonic() - started < 5
    assert report.host_errors == {}


async def test_unparseable_pid_skips_remote_kill_but_still_cleans() -> None:
    items, objects = farm_items([10])
    recorder = Recorder(expire_shas={items[0].sha256}, pid_output="The system cannot find the file")
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))

    report = await run_farm(
        items,
        hosts,
        FakeGarage(objects),
        FakePool(),
        requeue=False,
        bucket=BUCKET,
        runner=recorder,
        timeouts=FAST,
    )

    assert report.results[0].status == "failed(timeout)"
    commands = recorder.commands_for("lin")
    assert not [c for c in commands if "grep -qa" in c or " kill " in f" {c}"]
    assert any(c.startswith("rm -f") for c in commands)


async def test_cancel_kills_remote_worker_and_cleans_up() -> None:
    items, objects = farm_items([10])
    sha = items[0].sha256
    recorder = Recorder(hang_shas={sha})
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))

    task = asyncio.create_task(
        run_farm(
            items,
            hosts,
            FakeGarage(objects),
            FakePool(),
            requeue=False,
            bucket=BUCKET,
            runner=recorder,
        )
    )
    for _ in range(200):
        if recorder.worker_starts:
            break
        await asyncio.sleep(0.01)
    assert recorder.worker_starts == [("lin", sha)]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    commands = recorder.commands_for("lin")
    jobs = f"/tmp/vorag-ocr/jobs/{sha}"
    assert _linux_kill(sha) in commands
    assert f"rm -f {jobs}.pdf {jobs}.json {jobs}.json.part {jobs}.pid" in commands
    assert len(recorder.cancelled) == 1


async def test_subprocess_runner_kills_its_child_on_timeout(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    argv = ["sh", "-c", f'echo $$ > "{pid_file}"; exec sleep 30']

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(subprocess_runner(argv), 1.0)

    pid = int(pid_file.read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


# ---------------------------------------------------------------------------
# 5. duplicate sha256 envelopes
# ---------------------------------------------------------------------------


def _duplicate(of: FarmItem, envelope_id: str) -> FarmItem:
    return replace(
        of, envelope_id=envelope_id, problem="duplicate_sha256", duplicate_of=of.envelope_id
    )


async def test_duplicate_envelopes_are_requeued_with_their_ocrd_twin() -> None:
    items, objects = farm_items([100])
    dup = _duplicate(items[0], "ENVDUP")
    recorder = Recorder()
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))
    pool = FakePool()

    report = await run_farm(
        [*items, dup],
        hosts,
        FakeGarage(objects),
        pool,
        requeue=True,
        bucket=BUCKET,
        runner=recorder,
    )

    results = {r.envelope_id: r for r in report.results}
    assert results[items[0].envelope_id].status == "ok"
    assert results["ENVDUP"].status == "duplicate"
    assert results["ENVDUP"].requeued is True
    assert results["ENVDUP"].failed is False
    assert pool.execute_calls == [
        (REQUEUE_SQL, (items[0].envelope_id,)),
        (REQUEUE_SQL, ("ENVDUP",)),
    ]
    assert [sha for _, sha in recorder.worker_starts] == [items[0].sha256]
    assert report.summary()["duplicate"] == 1


async def test_duplicate_of_a_failed_book_is_not_requeued() -> None:
    items, objects = farm_items([100])
    dup = _duplicate(items[0], "ENVDUP")
    recorder = Recorder(exit_codes={items[0].sha256: 4})
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))
    pool = FakePool()

    report = await run_farm(
        [*items, dup],
        hosts,
        FakeGarage(objects),
        pool,
        requeue=True,
        bucket=BUCKET,
        runner=recorder,
    )

    results = {r.envelope_id: r for r in report.results}
    assert results["ENVDUP"].status == "failed(duplicate_sha256)"
    assert results["ENVDUP"].requeued is False
    assert pool.execute_calls == []


async def test_duplicate_without_requeue_is_reported_but_not_requeued() -> None:
    items, objects = farm_items([100])
    dup = _duplicate(items[0], "ENVDUP")
    recorder = Recorder()
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))
    pool = FakePool()

    report = await run_farm(
        [*items, dup],
        hosts,
        FakeGarage(objects),
        pool,
        requeue=False,
        bucket=BUCKET,
        runner=recorder,
    )

    results = {r.envelope_id: r for r in report.results}
    assert results["ENVDUP"].status == "duplicate"
    assert results["ENVDUP"].requeued is False
    assert pool.execute_calls == []


# ---------------------------------------------------------------------------
# The remote kill only targets our own, still-running worker
# ---------------------------------------------------------------------------


async def _run_one(recorder: Recorder, host: FarmHost, pages: int = 10) -> tuple[Any, str]:
    items, objects = farm_items([pages])
    hosts = await resolved_hosts(recorder, (host,))
    report = await run_farm(
        items,
        hosts,
        FakeGarage(objects),
        FakePool(),
        requeue=False,
        bucket=BUCKET,
        runner=recorder,
        timeouts=FAST,
    )
    return report, items[0].sha256


@pytest.mark.parametrize("host", [DEFAULT_HOSTS[0], DEFAULT_HOSTS[1]], ids=["windows", "linux"])
async def test_timeout_during_download_never_kills_a_pid(host: FarmHost) -> None:
    """The worker has already exited; its PID may be recycled by then."""
    items, _ = farm_items([10])
    sha = items[0].sha256
    remote = (
        "win:C:/Users/Demo/AppData/Local/Temp/vorag-ocr"
        if host.is_windows
        else "lin:/tmp/vorag-ocr"
    )
    recorder = Recorder(hang_commands=[f"{remote}/jobs/{sha}.json"])

    report, _ = await _run_one(recorder, host)

    assert report.results[0].status == "failed(timeout)"
    commands = recorder.commands_for(host.alias)
    assert _no_kill(commands)
    assert any(c.startswith(("rm -f", "del /f /q")) for c in commands)


async def test_timeout_during_upload_never_kills_a_pid() -> None:
    items, _ = farm_items([10])
    recorder = Recorder(hang_commands=[f"lin:/tmp/vorag-ocr/jobs/{items[0].sha256}.pdf"])

    report, _ = await _run_one(recorder, DEFAULT_HOSTS[1])

    assert report.results[0].status == "failed(timeout)"
    assert _no_kill(recorder.commands_for("lin"))


async def test_cancel_during_upload_never_kills_a_pid() -> None:
    items, objects = farm_items([10])
    sha = items[0].sha256
    recorder = Recorder(hang_commands=[f"lin:/tmp/vorag-ocr/jobs/{sha}.pdf"])
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))
    task = asyncio.create_task(
        run_farm(
            items,
            hosts,
            FakeGarage(objects),
            FakePool(),
            requeue=False,
            bucket=BUCKET,
            runner=recorder,
        )
    )
    for _ in range(200):
        if recorder.cancelled or any(
            argv[0] == "scp" and argv[-1].endswith(f"{sha}.pdf") for argv in recorder.calls
        ):
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    commands = recorder.commands_for("lin")
    assert _no_kill(commands)
    assert any(c.startswith("rm -f") for c in commands)


def test_linux_kill_command_only_kills_a_process_running_our_worker_for_that_sha() -> None:
    """Run the real Linux kill command locally against two local processes."""
    sha = "b" * 64
    host = DEFAULT_HOSTS[1]
    ours = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", "ocr_worker.py", f"jobs/{sha}.pdf"]
    )
    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    other_book = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", "ocr_worker.py", "jobs/cafe.pdf"]
    )
    try:
        for proc in (stranger, other_book):
            miss = subprocess.run(
                ["sh", "-c", host.kill_worker_command(proc.pid, sha)], check=False
            )
            assert miss.returncode != 0
            assert proc.poll() is None  # untouched
        hit = subprocess.run(["sh", "-c", host.kill_worker_command(ours.pid, sha)], check=False)
        assert hit.returncode == 0
        assert ours.wait(timeout=10) != 0
    finally:
        for proc in (ours, stranger, other_book):
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def test_kill_commands_verify_before_killing() -> None:
    sha = "c" * 64
    assert DEFAULT_HOSTS[1].kill_worker_command(4242, sha) == _linux_kill(sha)
    assert DEFAULT_HOSTS[0].kill_worker_command(4242, sha) == _windows_kill(sha)
    with pytest.raises(ValueError):
        DEFAULT_HOSTS[1].kill_worker_command(4242, "not-a-sha; rm -rf /")


# ---------------------------------------------------------------------------
# Bootstrap bounds are enforced on the remote side
# ---------------------------------------------------------------------------

FAST_BOOT = BootstrapTimeouts(command_s=0.2, install_s=0.2, check_s=0.2, backstop_s=0.0)
FARM_SRC = Path(__file__).resolve().parents[2] / "src" / "vo_rag" / "ocr_farm"


def test_bootstrap_timeout_defaults() -> None:
    timeouts = BootstrapTimeouts()
    assert timeouts.command_s == 300.0
    assert timeouts.install_s == 3600.0
    assert timeouts.check_s == 300.0
    assert timeouts.backstop_s == 90.0


@pytest.mark.parametrize("host", [DEFAULT_HOSTS[0], DEFAULT_HOSTS[1]], ids=["windows", "linux"])
def test_long_remote_commands_carry_their_own_timeout(host: FarmHost) -> None:
    commands = [host.venv_command(3600), host.install_command(3600), host.check_command(300)]
    for command, secs in zip(commands, (3600, 3600, 300), strict=True):
        if host.is_windows:
            assert f"-File farm_step.ps1 -TimeoutSeconds {secs} " in command
        else:
            assert f"timeout --kill-after=30 {secs} " in command


def test_farm_step_wrapper_stops_only_the_tree_it_started() -> None:
    script = (FARM_SRC / "worker" / "farm_step.ps1").read_text(encoding="utf-8")
    assert "Start-Process" in script and "-PassThru" in script and "-NoNewWindow" in script
    assert "Wait-Process -Id $p.Id -Timeout $TimeoutSeconds" in script
    # The whole tree rooted at the exact PID we started.
    assert "taskkill /T /F /PID $($p.Id)" in script
    assert "Stop-Process" not in script
    assert "exit 124" in script
    assert "exit $p.ExitCode" in script


def test_farm_step_wrapper_validates_the_timeout_before_starting_anything() -> None:
    """Wait-Process -Timeout accepts 1..32767 only."""
    script = (FARM_SRC / "worker" / "farm_step.ps1").read_text(encoding="utf-8")
    check = script.index("$TimeoutSeconds -lt 1 -or $TimeoutSeconds -gt 32767")
    assert check < script.index("Start-Process")
    assert "exit 2" in script[check : script.index("Start-Process")]


def test_no_pattern_based_kill_anywhere_in_the_farm_code() -> None:
    for path in FARM_SRC.rglob("*"):
        if path.suffix not in {".py", ".ps1"}:
            continue
        text = path.read_text(encoding="utf-8")
        for banned in ("pkill", "killall", "Where-Object", "/IM", "/FI"):
            assert banned not in text, f"{banned!r} in {path}"
        for line in text.splitlines():
            if "Stop-Process" in line:
                assert "Stop-Process -Id" in line, f"unanchored Stop-Process in {path}: {line}"
            if "taskkill" in line:
                # Only ever an explicit PID (the tree rooted at a process we started).
                assert re.search(
                    r"taskkill /T /F /PID \$\(\$p\.Id\)|taskkill /T /F /PID \{", line
                ), f"taskkill without an explicit PID in {path}: {line}"


async def test_bootstrap_local_backstop_fires_without_any_remote_kill() -> None:
    recorder = Recorder(hang_commands=["pip install"])
    (lin,) = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))

    started = time.monotonic()
    result = await bootstrap_worker(lin, runner=recorder, timeouts=FAST_BOOT)

    assert time.monotonic() - started < 5
    assert result.ok is False
    assert "pip install timeout" in result.error
    assert len(recorder.cancelled) == 1
    commands = recorder.commands_for("lin")
    assert "pip install" in commands[-1]  # the hung step is the last command: no kill after it
    assert not any("--check" in c for c in commands)


async def test_bootstrap_self_timeout_exit_is_a_failed_step() -> None:
    recorder = Recorder(fail_commands={"pip install": 124})
    (lin,) = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))

    result = await bootstrap_worker(lin, runner=recorder, timeouts=FAST_BOOT)

    assert result.ok is False
    assert "pip install exit 124" in result.error


async def test_bootstrap_exception_is_a_failed_result_not_a_crash() -> None:
    recorder = Recorder(raise_commands={"mkdir -p": FileNotFoundError("ssh")})
    (lin,) = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))

    result = await bootstrap_worker(lin, runner=recorder, timeouts=FAST_BOOT)

    assert result.ok is False
    assert "FileNotFoundError" in result.error


async def test_prepare_host_check_is_self_bounded_and_a_hang_drops_the_host() -> None:
    items, objects = farm_items([30, 20])
    recorder = Recorder(hang_commands=["cd /tmp/vorag-ocr && timeout --kill-after=30"])
    hosts = await resolved_hosts(recorder)

    report = await run_farm(
        items,
        hosts,
        FakeGarage(objects),
        FakePool(),
        requeue=False,
        bucket=BUCKET,
        runner=recorder,
        timeouts=FAST,
    )

    assert "timeout" in report.host_errors["lin"]
    assert recorder.commands_for("lin")[-1].startswith(
        "cd /tmp/vorag-ocr && timeout --kill-after=30 1 .venv/bin/python ocr_worker.py --check"
    )
    assert all(r.status == "ok" and r.host == "win" for r in report.results)


# ---------------------------------------------------------------------------
# An scp failure probes the host
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("probe", ["exit_255", "hang"])
async def test_scp_failure_on_unreachable_host_is_a_host_fault(probe: str) -> None:
    items, objects = farm_items([300, 200, 100])
    first = items[0]
    recorder = Recorder(
        fail_commands={f"lin:/tmp/vorag-ocr/jobs/{first.sha256}.pdf": 1},
    )
    if probe == "exit_255":
        recorder.fail_commands["ssh lin exit"] = 255
    else:
        recorder.hang_commands.append("ssh lin exit")
    # lin first so the longest book goes there.
    hosts = await resolved_hosts(
        recorder, (replace(DEFAULT_HOSTS[1], slots=1), replace(DEFAULT_HOSTS[0], slots=1))
    )

    report = await run_farm(
        items,
        hosts,
        FakeGarage(objects),
        FakePool(),
        requeue=False,
        bucket=BUCKET,
        runner=recorder,
        timeouts=FAST,
    )

    assert "scp_upload_exit_1" in report.host_errors["lin"]
    assert all(r.status == "ok" and r.host == "win" for r in report.results)
    assert ["ssh", "lin", "exit"] in recorder.calls


async def test_scp_failure_on_reachable_host_is_a_book_failure() -> None:
    items, objects = farm_items([300, 100])
    first = items[0]
    recorder = Recorder(fail_commands={f"lin:/tmp/vorag-ocr/jobs/{first.sha256}.pdf": 1})
    hosts = await resolved_hosts(recorder, (replace(DEFAULT_HOSTS[1], slots=1),))

    report = await run_farm(
        items,
        hosts,
        FakeGarage(objects),
        FakePool(),
        requeue=False,
        bucket=BUCKET,
        runner=recorder,
        timeouts=FAST,
    )

    statuses = {r.envelope_id: r.status for r in report.results}
    assert statuses[first.envelope_id] == "failed(scp_upload_exit_1)"
    assert statuses[items[1].envelope_id] == "ok"
    assert report.host_errors == {}
    assert ["ssh", "lin", "exit"] in recorder.calls


async def test_download_failure_on_unreachable_host_is_a_host_fault() -> None:
    items, objects = farm_items([300])
    sha = items[0].sha256
    recorder = Recorder(
        fail_commands={f"lin:/tmp/vorag-ocr/jobs/{sha}.json /": 1, "ssh lin exit": 255}
    )
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1], DEFAULT_HOSTS[0]))

    report = await run_farm(
        items,
        hosts,
        FakeGarage(objects),
        FakePool(),
        requeue=False,
        bucket=BUCKET,
        runner=recorder,
        timeouts=FAST,
    )

    assert "scp_download_exit_1" in report.host_errors["lin"]
    assert report.results[0].status == "ok"
    assert report.results[0].host == "win"


# ---------------------------------------------------------------------------
# Two timeouts in a row on one host drop the host
# ---------------------------------------------------------------------------


async def test_two_consecutive_timeouts_drop_the_host_and_requeue_its_books() -> None:
    items, objects = farm_items([300, 200, 100])
    recorder = Recorder(expire_hosts={"lin"})
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1], replace(DEFAULT_HOSTS[0], slots=1)))
    pool = FakePool()

    report = await run_farm(
        items,
        hosts,
        FakeGarage(objects),
        pool,
        requeue=True,
        bucket=BUCKET,
        runner=recorder,
        timeouts=FAST,
    )

    assert "2 consecutive timeouts" in report.host_errors["lin"]
    assert [r.status for r in report.results] == ["ok", "ok", "ok"]
    assert {r.host for r in report.results} == {"win"}
    assert {args for _, args in pool.execute_calls} == {(i.envelope_id,) for i in items}
    assert [a for a, _ in recorder.worker_starts].count("lin") == 2


async def test_a_success_resets_the_timeout_streak() -> None:
    items, objects = farm_items([300, 200, 100])
    recorder = Recorder(expire_shas={items[0].sha256, items[2].sha256})
    hosts = await resolved_hosts(recorder, (replace(DEFAULT_HOSTS[1], slots=1),))

    report = await run_farm(
        items,
        hosts,
        FakeGarage(objects),
        FakePool(),
        requeue=False,
        bucket=BUCKET,
        runner=recorder,
        timeouts=FAST,
    )

    assert report.host_errors == {}
    assert [r.status for r in report.results] == ["failed(timeout)", "ok", "failed(timeout)"]


async def test_a_book_requeued_for_a_timeout_streak_fails_on_its_next_timeout() -> None:
    """Two books that stall everywhere drop one host, not the whole farm."""
    items, objects = farm_items([300, 200])
    # Both attempts stall: the first (files named by the sha) and the
    # revived one (files scoped to attempt 1).
    recorder = Recorder(expire_shas={_attempt_key(i, gen) for i in items for gen in (0, 1)})
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1], replace(DEFAULT_HOSTS[0], slots=1)))

    report = await run_farm(
        items,
        hosts,
        FakeGarage(objects),
        FakePool(),
        requeue=False,
        bucket=BUCKET,
        runner=recorder,
        timeouts=FAST,
    )

    assert set(report.host_errors) == {"lin"}
    assert [r.status for r in report.results] == ["failed(timeout)", "failed(timeout)"]
    assert {r.host for r in report.results} == {"win"}


# ---------------------------------------------------------------------------
# host-drop ordering, wrapper timeout limits and PID-file retries
# ---------------------------------------------------------------------------


async def test_flagged_book_timing_out_on_an_already_dropped_host_is_requeued() -> None:
    """Host-dropped handling comes before the "already requeued once" rule.

    X and Y stall on lin (the streak drops lin; both are flagged and requeued).
    Both then run on win: Y exits 69 there, dropping win while X
    is still running, and X then times out on the dropped host. X goes back
    to the queue (with no host left it ends ``failed(no_usable_host)``); it
    must not end ``failed(timeout)``.
    """
    items, objects = farm_items([300, 250, 100], prefix="fl")
    x, y, w = items
    x1, y1 = _attempt_key(x, 1), _attempt_key(y, 1)  # after the revival
    # X's timeout on win is cued by win's drop, not by a timer
    # racing Y's exit (VOR-1434).
    win_dropped = asyncio.Event()

    class _DropWatch(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "host_dropped host=win" in record.getMessage():
                win_dropped.set()

    async def x1_outlives_the_drop(argv: list[str]) -> None:
        if argv[0] == "ssh" and "ocr_worker.py" in argv[-1] and x1 in argv[-1]:
            await asyncio.wait_for(win_dropped.wait(), 5)

    recorder = Recorder(
        expire_hosts={"lin"},
        expire_shas={x1},
        exit_codes={y1: 69},
        hook=x1_outlives_the_drop,
    )
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1], replace(DEFAULT_HOSTS[0], slots=2)))
    watch = _DropWatch()
    coordinator_log = logging.getLogger("vo_rag.ocr_farm.coordinator")
    coordinator_log.addHandler(watch)
    try:
        report = await run_farm(
            items,
            hosts,
            FakeGarage(objects),
            FakePool(),
            requeue=False,
            bucket=BUCKET,
            runner=recorder,
            timeouts=FAST,
        )
    finally:
        coordinator_log.removeHandler(watch)

    statuses = {r.envelope_id: r.status for r in report.results}
    assert ("win", x1) in recorder.worker_starts  # X did run on win
    assert "2 consecutive timeouts" in report.host_errors["lin"]
    assert "worker_exit_69" in report.host_errors["win"]
    assert statuses[w.envelope_id] == "ok"
    assert statuses[x.envelope_id] == "failed(no_usable_host)"
    assert statuses[y.envelope_id] == "failed(no_usable_host)"


def test_timeouts_are_validated_against_the_wrapper_limit() -> None:
    """farm_step.ps1's Wait-Process accepts 1..32767 s only."""
    assert BootstrapTimeouts(install_s=32767.0).install_s == 32767.0
    for bad in ({"install_s": 32768.0}, {"check_s": 40000.0}, {"install_s": 0.0}):
        with pytest.raises(ValueError, match="32767"):
            BootstrapTimeouts(**bad)
    with pytest.raises(ValueError, match="32767"):
        FarmTimeouts(command_s=32768.0)
    for host in DEFAULT_HOSTS:
        assert "32767" in host.check_command(32767)
        with pytest.raises(ValueError, match="32767"):
            host.check_command(32768)
        with pytest.raises(ValueError, match="32767"):
            host.install_command(-5)


async def test_pid_file_appearing_on_the_second_read_is_still_killed() -> None:
    """The worker may still be starting when its command times out."""
    items, _ = farm_items([10])
    sha = items[0].sha256
    recorder = Recorder(expire_shas={sha}, pid_missing_reads={sha: 1})

    report, _ = await _run_one(recorder, DEFAULT_HOSTS[1])

    assert report.results[0].status == "failed(timeout)"
    commands = recorder.commands_for("lin")
    reads = [c for c in commands if c == f"cat /tmp/vorag-ocr/jobs/{sha}.pid"]
    assert len(reads) == 2
    assert _linux_kill(sha) in commands


async def test_no_pid_after_three_reads_logs_a_possible_orphan(
    caplog: pytest.LogCaptureFixture,
) -> None:
    items, _ = farm_items([10])
    sha = items[0].sha256
    recorder = Recorder(expire_shas={sha}, pid_missing_reads={sha: 99})

    with caplog.at_level(logging.ERROR, logger="vo_rag.ocr_farm.coordinator"):
        report, _ = await _run_one(recorder, DEFAULT_HOSTS[1])

    assert report.results[0].status == "failed(timeout)"
    commands = recorder.commands_for("lin")
    assert [c for c in commands if c.startswith("cat ")] == [
        f"cat /tmp/vorag-ocr/jobs/{sha}.pid"
    ] * 3
    assert not [c for c in commands if c.startswith("grep ")]
    assert any("may still be running" in r.getMessage() for r in caplog.records)


def test_pid_retry_defaults() -> None:
    timeouts = FarmTimeouts()
    assert timeouts.pid_read_attempts == 3
    assert timeouts.pid_retry_s == 5.0


@pytest.mark.parametrize("stage", ["upload", "download"])
async def test_unreachable_host_gets_no_cleanup_and_the_leftover_is_recorded(stage: str) -> None:
    """After a failed probe, don't spend command_s on a dead host."""
    items, objects = farm_items([300])
    sha = items[0].sha256
    needle = (
        f"lin:/tmp/vorag-ocr/jobs/{sha}.pdf"
        if stage == "upload"
        else f"lin:/tmp/vorag-ocr/jobs/{sha}.json /"
    )
    recorder = Recorder(fail_commands={needle: 1, "ssh lin exit": 255})
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1], DEFAULT_HOSTS[0]))

    report = await run_farm(
        items,
        hosts,
        FakeGarage(objects),
        FakePool(),
        requeue=False,
        bucket=BUCKET,
        runner=recorder,
        timeouts=FAST,
    )

    probe_at = recorder.calls.index(["ssh", "lin", "exit"])
    after_probe = [argv for argv in recorder.calls[probe_at + 1 :] if argv[:2] == ["ssh", "lin"]]
    assert after_probe == []
    assert not [c for c in recorder.commands_for("lin") if c.startswith("rm -f")]
    assert f"worker files for {sha} may remain on lin" in report.host_errors["lin"]
    assert report.results[0].status == "ok"
    assert report.results[0].host == "win"


# ---------------------------------------------------------------------------
# ssh exit-255 probing and per-book PID-read tracking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("probe", ["exit_255", "hang"])
async def test_worker_exit_255_on_unreachable_host_skips_cleanup(probe: str) -> None:
    """An ssh exit 255 from the worker command probes the host first.

    A dropped connection must not cost ``command_s`` on a cleanup that can't
    reach the host: the fault is ``unreachable``, nothing more goes to lin, and
    the leftover files are recorded. The book is re-done on win.
    """
    items, objects = farm_items([300])
    sha = items[0].sha256
    recorder = Recorder(host_exit_codes={"lin": 255})
    if probe == "exit_255":
        recorder.fail_commands["ssh lin exit"] = 255
    else:
        recorder.hang_commands.append("ssh lin exit")
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1], DEFAULT_HOSTS[0]))
    pool = FakePool()

    report = await run_farm(
        items,
        hosts,
        FakeGarage(objects),
        pool,
        requeue=True,
        bucket=BUCKET,
        runner=recorder,
        timeouts=FAST,
    )

    probe_at = recorder.calls.index(["ssh", "lin", "exit"])
    after_probe = [argv for argv in recorder.calls[probe_at + 1 :] if argv[:2] == ["ssh", "lin"]]
    assert after_probe == []
    assert not [c for c in recorder.commands_for("lin") if c.startswith("rm -f")]
    assert "worker_exit_255" in report.host_errors["lin"]
    assert f"worker files for {sha} may remain on lin" in report.host_errors["lin"]
    assert report.results[0].status == "ok"
    assert report.results[0].host == "win"
    assert pool.execute_calls[0][1] == (items[0].envelope_id,)


async def test_worker_exit_255_on_reachable_host_is_dropped_and_cleaned() -> None:
    """A reachable host that still exits 255 is dropped and cleaned."""
    items, objects = farm_items([300])
    sha = items[0].sha256
    recorder = Recorder(host_exit_codes={"lin": 255})
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1], DEFAULT_HOSTS[0]))

    report = await run_farm(
        items,
        hosts,
        FakeGarage(objects),
        FakePool(),
        requeue=False,
        bucket=BUCKET,
        runner=recorder,
        timeouts=FAST,
    )

    commands = recorder.commands_for("lin")
    jobs = f"/tmp/vorag-ocr/jobs/{sha}"
    assert commands.index("exit") < commands.index(
        f"rm -f {jobs}.pdf {jobs}.json {jobs}.json.part {jobs}.pid"
    )
    assert "worker_exit_255" in report.host_errors["lin"]
    assert "may remain" not in report.host_errors["lin"]
    assert report.results[0].status == "ok"
    assert report.results[0].host == "win"


#: A whole-loop pause longer than FAST's old 0.3 s worker budget.
LOOP_STALL_S = 0.4


@pytest.mark.parametrize(("bad_host", "code"), [("lin", 69), ("lin", 127), ("win", 9009)])
async def test_other_host_fault_exits_are_not_probed(bad_host: str, code: int) -> None:
    """Exit codes 3 / 127 / 9009 come from a reachable host; no probe, cleanup runs.

    The event loop stalls while the worker runs, as on a loaded CI runner
    (VOR-1434): the exit must still be read as the host's fault, not as a
    timeout, so no test budget may be short enough for a stall to spend it.
    """
    items, objects = farm_items([300])

    async def stall_during_worker(argv: list[str]) -> None:
        if argv[0] == "ssh" and "ocr_worker.py" in argv[-1] and items[0].sha256 in argv[-1]:
            # Blocks the whole loop once the fake worker is asleep, like a
            # runner that deschedules the process: its sleep and its budget
            # then expire together, and the budget must not win.
            asyncio.get_running_loop().call_soon(time.sleep, LOOP_STALL_S)

    recorder = Recorder(host_exit_codes={bad_host: code}, hook=stall_during_worker)
    # The bad host first, so the book lands there before going elsewhere.
    ordered = sorted(DEFAULT_HOSTS, key=lambda h: h.alias != bad_host)
    hosts = await resolved_hosts(recorder, tuple(ordered))

    report = await run_farm(
        items,
        hosts,
        FakeGarage(objects),
        FakePool(),
        requeue=False,
        bucket=BUCKET,
        runner=recorder,
        timeouts=FAST,
    )

    assert recorder.worker_starts[0][0] == bad_host
    assert ["ssh", bad_host, "exit"] not in recorder.calls
    assert [c for c in recorder.commands_for(bad_host) if c.startswith(("rm -f", "del /f /q"))]
    assert f"worker_exit_{code}" in report.host_errors[bad_host]
    assert report.results[0].status == "ok"


async def test_concurrent_timeouts_track_pid_reads_per_book(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """PID-read retries are per sha, not one counter for the run.

    A and B time out together on lin (2 slots). A's PID file appears on its
    second read and A is killed; B's never appears and B is logged as an
    orphan risk. The two must not consume each other's reads.
    """
    items, objects = farm_items([10, 10], prefix="pid")
    a, b = (i.sha256 for i in items)
    recorder = Recorder(expire_shas={a, b}, pid_missing_reads={a: 1, b: 99})
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))

    with caplog.at_level(logging.ERROR, logger="vo_rag.ocr_farm.coordinator"):
        report = await run_farm(
            items,
            hosts,
            FakeGarage(objects),
            FakePool(),
            requeue=False,
            bucket=BUCKET,
            runner=recorder,
            timeouts=FAST,
        )

    # Both start at once on lin's two slots; which one starts first is up to the scheduler.
    assert sorted(recorder.worker_starts) == sorted([("lin", a), ("lin", b)])
    assert recorder.pid_reads == {a: 2, b: 3}
    commands = recorder.commands_for("lin")
    assert _linux_kill(a) in commands
    assert _linux_kill(b) not in commands
    assert not [c for c in commands if c.startswith("grep ") and b in c]
    orphans = [r.getMessage() for r in caplog.records if "may still be running" in r.getMessage()]
    assert len(orphans) == 1
    assert b in orphans[0]
    assert a not in orphans[0]
    # Two timeouts in a row on the only host drop it; both books are then unplaceable.
    assert "2 consecutive timeouts" in report.host_errors["lin"]
    assert [r.status for r in report.results] == ["failed(no_usable_host)"] * 2


async def test_a_worker_crash_exit_3_fails_the_book_not_the_host() -> None:
    """Windows ``abort()`` (e.g. ``std::bad_alloc``) exits 3. That is the book's
    crash: the host keeps its slot and runs the next book (2026-09-26 run)."""
    items, objects = farm_items([300, 200], prefix="crash")
    crashed, healthy = items
    recorder = Recorder(exit_codes={crashed.sha256: 3})
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[0],))

    report = await run_farm(
        items,
        hosts,
        FakeGarage(objects),
        FakePool(),
        requeue=False,
        bucket=BUCKET,
        runner=recorder,
        timeouts=FAST,
    )

    statuses = {r.envelope_id: r.status for r in report.results}
    assert "win" not in report.host_errors
    assert statuses[crashed.envelope_id].startswith("failed(")
    assert statuses[healthy.envelope_id] == "ok"

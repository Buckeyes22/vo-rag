"""OCR farm: large books split into page-range parts on the coordinator (VOR-1427).

The worker keeps every 100-page batch's DoclingDocument until it concatenates
them, so its memory grows with book length: on 2026-09-26 two win
workers on a 4,178- and a 2,036-page book reached ~8 GB of commit each after
40 minutes. The coordinator now ships any book over ``SPLIT_PAGES`` pages as
contiguous sub-PDF parts (independent jobs, any host, any slot), and merges
the part JSONs with ``DoclingDocument.concatenate`` once every part is in.

ssh/scp are faked by the recorder; the PDFs are real (pymupdf), so each
part's shipped pages can be checked. The merge is ``fake_merge`` (same page
shift as docling-core) except in the last test, which uses docling-core itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import vo_rag.ocr_farm.coordinator as coordinator
from tests.ocr_farm._farm_fakes import (
    BUCKET,
    FakeGarage,
    FakePool,
    Recorder,
    fake_merge,
    farm_items,
    pdf_bytes,
    pdf_item,
    resolved_hosts,
)
from vo_rag.ocr_farm.coordinator import (
    REQUEUE_SQL,
    SPLIT_PAGES,
    FarmItem,
    FarmTimeouts,
    merge_docling_parts,
    part_key,
    part_ranges,
    run_farm,
)
from vo_rag.ocr_farm.hosts import DEFAULT_HOSTS, FarmHost

win, Z2 = DEFAULT_HOSTS
#: Hung workers must not time out on their own while a test cancels them.
NO_TIMEOUT = FarmTimeouts(
    command_s=5.0,
    ocr_startup_s=0.0,
    ocr_s_per_page=0.0,
    ocr_min_s=60.0,
    backstop_s=0.0,
    probe_s=1.0,
    pid_retry_s=0.0,
)
#: Jobs meant to time out do so on cue (``Recorder.expire_*``, no timer);
#: every other job keeps this budget, several seconds, so a loaded CI box
#: can't time out a job that should succeed.
EXPIRE_ON_CUE = replace(NO_TIMEOUT, ocr_min_s=10.0)


def _keys(item: FarmItem, split: int = SPLIT_PAGES, gen: int = 0) -> list[str]:
    """The worker file stems of ``item``'s parts in attempt ``gen``."""
    return [
        part_key(item.sha256, first, last, gen) for first, last in part_ranges(item.pages, split)
    ]


def _book_done(caplog: pytest.LogCaptureFixture) -> dict[str, list[str]]:
    """envelope_id -> the status of each ``ocr_farm.book_done`` logged for it."""
    done: dict[str, list[str]] = {}
    for record in caplog.records:
        message = record.getMessage()
        if message.startswith("ocr_farm.book_done "):
            fields = dict(part.split("=", 1) for part in message.split()[1:])
            done.setdefault(fields["envelope_id"], []).append(fields["status"])
    return done


def _linux_kill(job: str) -> str:
    return (
        f"grep -qa ocr_worker.py /proc/4242/cmdline && grep -qa {job} /proc/4242/cmdline "
        "&& kill 4242"
    )


def _cleanup(host: FarmHost, job: str) -> str:
    if host.is_windows:
        jobs = f"%TEMP%\\vorag-ocr\\jobs\\{job}"
        return f'del /f /q "{jobs}.pdf" "{jobs}.json" "{jobs}.json.part" "{jobs}.pid"'
    jobs = f"/tmp/vorag-ocr/jobs/{job}"
    return f"rm -f {jobs}.pdf {jobs}.json {jobs}.json.part {jobs}.pid"


def _stored(garage: FakeGarage, item: FarmItem, name: str) -> Any:
    return json.loads(garage.objects[(BUCKET, f"ocr/{item.sha256}/{name}")])


# ---------------------------------------------------------------------------
# the split itself
# ---------------------------------------------------------------------------


def test_split_pages_is_400_and_ranges_are_contiguous() -> None:
    assert SPLIT_PAGES == 400
    assert part_ranges(1000, 400) == [(1, 400), (401, 800), (801, 1000)]
    assert part_ranges(400, 400) == [(1, 400)]
    assert part_ranges(401, 400) == [(1, 400), (401, 401)]
    assert part_ranges(7, 400) == [(1, 7)]
    assert len(part_ranges(4178, 400)) == 11


def test_part_keys_are_sha256_shaped_distinct_and_never_the_book_sha() -> None:
    sha = "a" * 64
    keys = [part_key(sha, first, last) for first, last in part_ranges(1000, 400)]
    assert len(set(keys)) == 3
    assert sha not in keys
    # kill_worker_command only accepts a sha256-shaped id and verifies it on
    # the remote command line, so a part's kill can never hit a sibling part.
    for key in keys:
        assert len(key) == 64 and int(key, 16) >= 0
        Z2.kill_worker_command(4242, key)


def test_worker_file_names_are_scoped_to_the_attempt() -> None:
    """A revived book's new attempt never reuses the old attempt's
    jobs/<key>.pdf|.json|.pid, which the old attempt may still be killing or
    removing."""
    sha = "b" * 64
    first, last = 401, 800
    assert part_key(sha, first, last, 0) != part_key(sha, first, last, 1)
    assert part_key(sha, first, last, 1) != part_key(sha, first, last, 2)
    item = FarmItem("ENVKEY", "k.pdf", sha, "Key", 800)
    whole = coordinator._Job(item, 0, 1, 800, 1, gen=0)
    assert whole.key == sha  # an unsplit book's first attempt: its sha, as before
    assert replace(whole, gen=1).key == part_key(sha, 1, 800, 1) != sha
    part = coordinator._Job(item, 1, first, last, 2, gen=0)
    assert part.key == part_key(sha, first, last, 0) != replace(part, gen=1).key


async def test_an_old_attempt_unwinding_never_touches_the_new_attempts_files(
    tmp_path: Path,
) -> None:
    """The old attempt of part 2 is cancelled and is between its PID reads
    when the new attempt of part 2 starts on the same host: the old kill and
    cleanup name only the old attempt's files."""
    item = FarmItem("ENVATTEMPT", "k.pdf", "c" * 64, "Attempt", 20)
    old = coordinator._Job(item, 1, 11, 20, 2, gen=0)
    new = replace(old, gen=1)
    old_started, old_pid_read, new_started = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def hook(argv: list[str]) -> None:
        command = argv[-1]
        if "ocr_worker.py" in command:
            (old_started if f"{old.key}.pdf" in command else new_started).set()
        if command == f"cat /tmp/vorag-ocr/jobs/{old.key}.pid":
            old_pid_read.set()

    recorder = Recorder(hang_shas={old.key, new.key}, pid_missing_reads={old.key: 1}, hook=hook)
    (lin,) = await resolved_hosts(recorder, (Z2,))
    timeouts = replace(NO_TIMEOUT, pid_retry_s=0.2)
    pdf = tmp_path / "part.pdf"
    pdf.write_bytes(pdf_bytes(10, text=False))

    old_task = asyncio.create_task(
        coordinator._ocr_on_host(old.key, 10, pdf, tmp_path / "old.json", lin, recorder, timeouts)
    )
    await old_started.wait()
    old_task.cancel()
    await old_pid_read.wait()  # the old attempt is now killing its worker
    new_task = asyncio.create_task(
        coordinator._ocr_on_host(new.key, 10, pdf, tmp_path / "new.json", lin, recorder, timeouts)
    )
    await new_started.wait()
    with pytest.raises(asyncio.CancelledError):
        await old_task

    commands = recorder.commands_for("lin")
    assert _linux_kill(old.key) in commands
    assert _cleanup(Z2, old.key) in commands
    # Nothing the old attempt sent names the new attempt's files...
    unwinding = [c for c in commands if c.startswith(("cat ", "grep ", "rm -f"))]
    assert unwinding and not [c for c in unwinding if new.key in c]
    # ...and the new attempt's worker is still running, untouched.
    assert recorder.running["lin"] == 1
    assert not [argv for argv in recorder.cancelled if new.key in argv[-1]]
    new_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await new_task


async def test_thousand_page_book_runs_as_three_parallel_part_jobs_and_merges(
    caplog: pytest.LogCaptureFixture,
) -> None:
    item, objects = pdf_item(1000)
    keys = _keys(item)
    # lin's part (the last, 200 pages) finishes first; the host list must
    # still be in part order.
    recorder = Recorder(delay_by_sha={keys[0]: 0.3, keys[1]: 0.3, keys[2]: 0.05})
    hosts = await resolved_hosts(recorder)
    garage = FakeGarage(objects)
    pool = FakePool()

    with caplog.at_level(logging.INFO, logger="vo_rag.ocr_farm.coordinator"):
        report = await run_farm(
            [item],
            hosts,
            garage,
            pool,
            requeue=True,
            bucket=BUCKET,
            runner=recorder,
            merge=fake_merge,
        )

    # One line per book, with the book's totals.
    (result,) = report.results
    assert (result.envelope_id, result.status, result.pages) == (item.envelope_id, "ok", 1000)
    assert result.requeued is True
    # Three separate jobs, each shipped exactly its own pages as a sub-PDF.
    assert sorted(job for _, job in recorder.worker_starts) == sorted(keys)
    assert item.sha256 not in {job for _, job in recorder.worker_starts}
    assert [recorder.job_pages[k] for k in keys] == [400, 400, 200]
    for key, (first, last) in zip(keys, part_ranges(1000, 400), strict=True):
        assert recorder.job_texts[key][0] == f"page {first}"
        assert recorder.job_texts[key][-1] == f"page {last}"
    # They ran at the same time, across both hosts.
    assert recorder.peak_total == 3
    assert {host for host, _ in recorder.worker_starts} == {"win", "lin"}
    # Every host that ran a part, in part order (parts 1-2 win, 3 lin).
    assert result.host == "win,lin"
    assert _stored(garage, item, "meta.json")["host"] == "win,lin"
    # Merged: absolute pages 1..1000, every text on its own page.
    merged = _stored(garage, item, "docling.json")
    assert sorted(int(n) for n in merged["pages"]) == list(range(1, 1001))
    assert [t["text"] for t in merged["texts"]] == [f"page {n}" for n in range(1, 1001)]
    assert [t["prov"][0]["page_no"] for t in merged["texts"]] == list(range(1, 1001))
    meta = _stored(garage, item, "meta.json")
    assert (meta["pipeline"], meta["parts"], meta["pages"]) == ("ocr", 3, 1000)
    assert [key for _, key, _ in garage.puts] == [
        f"ocr/{item.sha256}/docling.json",
        f"ocr/{item.sha256}/meta.json",
    ]
    # Requeued once for the book, not once per part.
    assert pool.execute_calls == [(REQUEUE_SQL, (item.envelope_id,))]
    # Every part's worker files were removed from the host that ran it.
    for host, job in recorder.worker_starts:
        target = win if host == "win" else Z2
        assert _cleanup(target, job) in recorder.commands_for(host)
    assert report.summary()["ok"] == 1
    # Progress: one part_done per part, one book_done for the book.
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    part_done = [m for m in messages if m.startswith("ocr_farm.part_done")]
    assert len(part_done) == 3
    assert any("part=2/3 pages=401-800" in m for m in part_done)
    book_done = [m for m in messages if m.startswith("ocr_farm.book_done")]
    assert len(book_done) == 1
    assert "status=ok pages=1000" in book_done[0]


async def test_parts_are_scheduled_longest_first_on_part_size() -> None:
    big, big_objects = pdf_item(1000, n=0, text=False)
    mid, mid_objects = pdf_item(300, n=1, text=False)
    recorder = Recorder()
    hosts = await resolved_hosts(recorder, (replace(Z2, slots=1),))

    report = await run_farm(
        [mid, big],
        hosts,
        FakeGarage({**big_objects, **mid_objects}),
        FakePool(),
        requeue=False,
        bucket=BUCKET,
        runner=recorder,
        merge=fake_merge,
    )

    k1, k2, k3 = _keys(big)
    # 400, 400, then the 300-page book ahead of the 200-page last part.
    assert [job for _, job in recorder.worker_starts] == [k1, k2, mid.sha256, k3]
    assert [r.status for r in report.results] == ["ok", "ok"]
    assert [r.envelope_id for r in report.results] == [big.envelope_id, mid.envelope_id]


async def test_timeout_budget_is_per_part(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[list[str], float]] = []
    real_call = coordinator._call

    async def spy(runner: Any, argv: list[str], timeout: float) -> Any:
        seen.append((argv, timeout))
        return await real_call(runner, argv, timeout)

    monkeypatch.setattr(coordinator, "_call", spy)
    item, objects = pdf_item(1000, text=False)
    recorder = Recorder()
    hosts = await resolved_hosts(recorder, (Z2,))

    report = await run_farm(
        [item],
        hosts,
        FakeGarage(objects),
        FakePool(),
        requeue=False,
        bucket=BUCKET,
        runner=recorder,
        merge=fake_merge,
    )

    assert report.results[0].status == "ok"
    timeouts = FarmTimeouts()
    budgets = {
        job: timeout
        for argv, timeout in seen
        for job in _keys(item)
        if argv[:2] == ["ssh", "lin"] and "ocr_worker.py" in argv[2] and f"{job}.pdf" in argv[2]
    }
    k1, k2, k3 = _keys(item)
    # ocr_startup_s is budgeted once per part: every part loads the models.
    assert budgets == {k1: timeouts.ocr_s(400), k2: timeouts.ocr_s(400), k3: timeouts.ocr_s(200)}
    assert budgets[k1] == 120.0 + 400 * 10.0


# ---------------------------------------------------------------------------
# books that aren't split
# ---------------------------------------------------------------------------


def _no_merge(part_paths: Sequence[Path], out_path: Path, name: str) -> list[int]:
    raise AssertionError(f"a single-part book must not be merged: {part_paths} {out_path} {name}")


@pytest.mark.parametrize("pages", [SPLIT_PAGES, 12])
async def test_book_of_split_pages_or_fewer_is_one_unsplit_job(pages: int) -> None:
    item, objects = pdf_item(pages, text=False)
    recorder = Recorder()
    shipped: list[bytes] = []
    original = recorder.__call__

    async def capture(argv: Sequence[str]) -> Any:
        if argv[0] == "scp" and str(argv[-1]).endswith(".pdf"):
            shipped.append(Path(argv[-2]).read_bytes())
        return await original(argv)

    hosts = await resolved_hosts(recorder, (Z2,))
    garage = FakeGarage(objects)

    report = await run_farm(
        [item],
        hosts,
        garage,
        FakePool(),
        requeue=False,
        bucket=BUCKET,
        runner=capture,
        merge=_no_merge,
    )

    assert report.results[0].status == "ok"
    # The job is the book: named by its sha, the source bytes shipped untouched.
    assert recorder.worker_starts == [("lin", item.sha256)]
    assert shipped == [objects[item.garage_key]]
    # The worker's JSON is stored as-is.
    stored = _stored(garage, item, "docling.json")
    assert stored["name"] == item.sha256
    meta = _stored(garage, item, "meta.json")
    assert (meta["pipeline"], meta["host"], meta["parts"]) == ("ocr", "lin", 1)
    assert meta["worker"] == "ocr_worker: ok status=success gpu_peak_bytes=4096"


async def test_one_page_over_split_pages_makes_a_one_page_part() -> None:
    item, objects = pdf_item(SPLIT_PAGES + 1, text=False)
    recorder = Recorder()
    hosts = await resolved_hosts(recorder, (Z2,))
    garage = FakeGarage(objects)

    report = await run_farm(
        [item],
        hosts,
        garage,
        FakePool(),
        requeue=False,
        bucket=BUCKET,
        runner=recorder,
        merge=fake_merge,
    )

    assert report.results[0].status == "ok"
    assert [recorder.job_pages[k] for k in _keys(item)] == [400, 1]
    merged = _stored(garage, item, "docling.json")
    assert sorted(int(n) for n in merged["pages"]) == list(range(1, 402))
    assert _stored(garage, item, "meta.json")["parts"] == 2


# ---------------------------------------------------------------------------
# failures
# ---------------------------------------------------------------------------


async def test_part_failure_fails_the_book_and_kills_in_flight_siblings() -> None:
    item, objects = pdf_item(1000, text=False)
    k1, k2, k3 = _keys(item)
    siblings_running = asyncio.Event()
    started: set[str] = set()

    async def hook(argv: list[str]) -> None:
        command = argv[-1]
        job = next((k for k in (k1, k2, k3) if f"{k}.pdf" in command), None)
        if argv[0] != "ssh" or "ocr_worker.py" not in command or job is None:
            return  # not a worker run (prepare, scp, kill, cleanup)
        started.add(job)
        if {k1, k3} <= started:
            siblings_running.set()
        if job == k2:
            await siblings_running.wait()  # part 2 fails only once both siblings run

    recorder = Recorder(hang_shas={k1, k3}, exit_codes={k2: 1}, hook=hook)
    hosts = await resolved_hosts(recorder)
    garage = FakeGarage(objects)
    pool = FakePool()

    report = await run_farm(
        [item],
        hosts,
        garage,
        pool,
        requeue=True,
        bucket=BUCKET,
        runner=recorder,
        timeouts=NO_TIMEOUT,
        merge=fake_merge,
    )

    (result,) = report.results
    assert result.status == "failed(worker_exit_1)"
    assert result.requeued is False
    assert report.host_errors == {}
    # Nothing partial is stored, nothing requeued, nothing retried.
    assert garage.puts == []
    assert pool.execute_calls == []
    assert sorted(job for _, job in recorder.worker_starts) == sorted([k1, k2, k3])
    # Both hung siblings were cancelled and killed by their verified PID.
    assert len(recorder.cancelled) == 2
    host_of = {job: host for host, job in recorder.worker_starts}
    for job in (k1, k3):
        commands = recorder.commands_for(host_of[job])
        if host_of[job] == "lin":
            assert _linux_kill(job) in commands
        else:
            assert any("Stop-Process -Id 4242" in c and job in c for c in commands)
    for host, job in recorder.worker_starts:
        assert _cleanup(win if host == "win" else Z2, job) in (recorder.commands_for(host))


async def test_a_part_still_killing_its_timed_out_worker_finishes_the_kill_when_cancelled() -> None:
    """Part 2 times out and is between its PID-file reads when part 1 fails
    the book; the cancellation must not cut the kill short."""
    item, objects = pdf_item(600, text=False)
    k1, k2 = _keys(item)
    k2_killing = asyncio.Event()

    async def hook(argv: list[str]) -> None:
        if argv[-1] == f"cat /tmp/vorag-ocr/jobs/{k2}.pid":
            k2_killing.set()  # part 2 timed out and started its kill
        if "ocr_worker.py" in argv[-1] and f"{k1}.pdf" in argv[-1]:
            await k2_killing.wait()  # part 1 fails only then

    # Part 2 times out on cue and waits pid_retry_s between its PID-file
    # reads; part 1 keeps a budget of several seconds.
    timeouts = replace(EXPIRE_ON_CUE, pid_retry_s=0.5)
    recorder = Recorder(expire_shas={k2}, pid_missing_reads={k2: 1}, exit_codes={k1: 1}, hook=hook)
    hosts = await resolved_hosts(recorder, (Z2,))
    garage = FakeGarage(objects)

    report = await run_farm(
        [item],
        hosts,
        garage,
        FakePool(),
        requeue=True,
        bucket=BUCKET,
        runner=recorder,
        timeouts=timeouts,
        merge=fake_merge,
    )

    assert report.results[0].status == "failed(worker_exit_1)"
    assert recorder.expired == [("lin", k2)]
    commands = recorder.commands_for("lin")
    assert commands.count(f"cat /tmp/vorag-ocr/jobs/{k2}.pid") == 2
    assert _linux_kill(k2) in commands
    assert _cleanup(Z2, k2) in commands
    assert garage.puts == []


async def test_a_sibling_cancelled_during_its_upload_is_cleaned_up_and_never_killed() -> None:
    item, objects = pdf_item(1000, text=False)
    k1, k2, k3 = _keys(item)
    k3_uploading = asyncio.Event()

    async def hook(argv: list[str]) -> None:
        if argv[0] == "scp" and argv[-1].endswith(f"/jobs/{k3}.pdf"):
            k3_uploading.set()
        if "ocr_worker.py" in argv[-1] and f"{k2}.pdf" in argv[-1]:
            await k3_uploading.wait()  # part 2 fails while part 3 is uploading

    recorder = Recorder(
        hang_commands=[f"lin:/tmp/vorag-ocr/jobs/{k3}.pdf"], exit_codes={k2: 1}, hook=hook
    )
    hosts = await resolved_hosts(recorder)
    garage = FakeGarage(objects)

    report = await run_farm(
        [item],
        hosts,
        garage,
        FakePool(),
        requeue=True,
        bucket=BUCKET,
        runner=recorder,
        timeouts=NO_TIMEOUT,
        merge=fake_merge,
    )

    assert report.results[0].status == "failed(worker_exit_1)"
    # Part 3's upload was cancelled: it never ran a worker, so no PID read and
    # no kill, but its files were removed.
    assert [argv for argv in recorder.cancelled if argv[0] == "scp"]
    assert ("lin", k3) not in recorder.worker_starts
    commands = recorder.commands_for("lin")
    assert not [c for c in commands if k3 in c and c.startswith(("cat ", "grep "))]
    assert _cleanup(Z2, k3) in commands
    assert garage.puts == []


async def test_part_failure_drops_the_books_parts_that_have_not_started() -> None:
    item, objects = pdf_item(1000, text=False)
    k1, k2, k3 = _keys(item)
    recorder = Recorder(exit_codes={k1: 5})
    hosts = await resolved_hosts(recorder, (replace(Z2, slots=1),))
    garage = FakeGarage(objects)

    report = await run_farm(
        [item],
        hosts,
        garage,
        FakePool(),
        requeue=True,
        bucket=BUCKET,
        runner=recorder,
        merge=fake_merge,
    )

    assert report.results[0].status == "failed(worker_exit_5)"
    assert recorder.worker_starts == [("lin", k1)]
    assert garage.puts == []


async def test_host_fault_on_a_part_requeues_only_that_part() -> None:
    item, objects = pdf_item(1000)
    k1, k2, k3 = _keys(item)
    recorder = Recorder(host_exit_codes={"lin": 69})
    hosts = await resolved_hosts(recorder)
    garage = FakeGarage(objects)
    pool = FakePool()

    report = await run_farm(
        [item],
        hosts,
        garage,
        pool,
        requeue=True,
        bucket=BUCKET,
        runner=recorder,
        merge=fake_merge,
    )

    (result,) = report.results
    assert result.status == "ok"
    assert "worker_exit_69" in report.host_errors["lin"]
    # Parts 1 and 2 went to win first and ran once; part 3 hit lin's
    # fault and ran again on win. The book never restarted.
    assert [s for s in recorder.worker_starts if s[1] in (k1, k2)] == [
        ("win", k1),
        ("win", k2),
    ]
    assert [s for s in recorder.worker_starts if s[1] == k3] == [("lin", k3), ("win", k3)]
    assert result.host == "win"
    merged = _stored(garage, item, "docling.json")
    assert [t["text"] for t in merged["texts"]] == [f"page {n}" for n in range(1, 1001)]
    assert pool.execute_calls == [(REQUEUE_SQL, (item.envelope_id,))]


async def test_part_timeout_fails_the_book() -> None:
    item, objects = pdf_item(1000, text=False)
    k1, k2, k3 = _keys(item)
    recorder = Recorder(expire_shas={k2})
    hosts = await resolved_hosts(recorder, (Z2,))
    garage = FakeGarage(objects)

    report = await run_farm(
        [item],
        hosts,
        garage,
        FakePool(),
        requeue=True,
        bucket=BUCKET,
        runner=recorder,
        timeouts=EXPIRE_ON_CUE,
        merge=fake_merge,
    )

    assert report.results[0].status == "failed(timeout)"
    assert recorder.expired == [("lin", k2)]
    assert _linux_kill(k2) in recorder.commands_for("lin")
    assert garage.puts == []


async def test_a_stalled_host_revives_every_part_of_its_timed_out_split_books(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two books time out in a row on lin (a stalled host): lin is dropped and
    both books go back to the queue whole, every part, not just the part
    that timed out -- as a new attempt with its own worker files."""
    a, a_objects = pdf_item(20, n=0)
    b, b_objects = pdf_item(15, n=1)
    a1, a2 = _keys(a, 10)
    b1, b2 = _keys(b, 10)
    new = _keys(a, 10, gen=1) + _keys(b, 10, gen=1)
    win_first_jobs_done = asyncio.Event()
    cleaned: set[str] = set()
    old_dirs_at_new_upload: list[bool] = []

    async def hook(argv: list[str]) -> None:
        command = argv[-1]
        if argv[:2] == ["ssh", "win"] and command.startswith("del /f /q"):
            cleaned.update(job for job in (a1, b2) if job in command)
            if cleaned == {a1, b2}:
                win_first_jobs_done.set()
        if argv[0] == "scp" and command.startswith("lin:") and command.endswith(".pdf"):
            # lin stalls only once win's first-attempt jobs are in.
            await win_first_jobs_done.wait()
        if argv[0] == "scp" and any(f"/jobs/{job}.pdf" in command for job in new):
            # A new attempt ships: the old attempt's local files are gone.
            old_dirs_at_new_upload.append((Path(argv[-2]).parent.parent / "g0").exists())

    recorder = Recorder(expire_hosts={"lin"}, hook=hook)
    hosts = await resolved_hosts(recorder, (replace(win, slots=1), Z2))
    garage = FakeGarage({**a_objects, **b_objects})
    pool = FakePool()

    with caplog.at_level(logging.INFO, logger="vo_rag.ocr_farm.coordinator"):
        report = await run_farm(
            [a, b],
            hosts,
            garage,
            pool,
            requeue=True,
            bucket=BUCKET,
            runner=recorder,
            timeouts=EXPIRE_ON_CUE,
            split_pages=10,
            merge=fake_merge,
        )

    assert "2 consecutive timeouts" in report.host_errors["lin"]
    # Jobs a1, a2, b1, b2: win took a1 then b2; lin stalled on a2 and b1.
    assert sorted(job for host, job in recorder.worker_starts if host == "lin") == sorted([a2, b1])
    # Revived whole, as new files: a1 and b2 (done before the timeouts) too.
    win_jobs = [job for host, job in recorder.worker_starts if host == "win"]
    assert sorted(win_jobs) == sorted([a1, b2, *new])
    assert not {a1, a2, b1, b2} & set(new)
    for argv in recorder.calls:
        if any(job in argv[-1] for job in (a1, a2, b1, b2)):
            assert not [job for job in new if job in argv[-1]]
    assert old_dirs_at_new_upload == [False] * 4
    assert [r.status for r in report.results] == ["ok", "ok"]
    assert {r.host for r in report.results} == {"win"}
    for item, pages in ((a, 20), (b, 15)):
        merged = _stored(garage, item, "docling.json")
        assert [t["text"] for t in merged["texts"]] == [f"page {n}" for n in range(1, pages + 1)]
    assert sorted(args for _, args in pool.execute_calls) == [(a.envelope_id,), (b.envelope_id,)]
    # One book_done per book, for its final outcome (not for the revived failure).
    assert _book_done(caplog) == {a.envelope_id: ["ok"], b.envelope_id: ["ok"]}


async def test_merged_pages_not_1_to_n_fail_the_book_and_store_nothing() -> None:
    item, objects = pdf_item(1000, text=False)
    k3 = _keys(item)[2]
    recorder = Recorder()
    original = recorder.__call__

    async def short_last_part(argv: Sequence[str]) -> Any:
        result = await original(argv)
        if argv[0] == "scp" and str(argv[-2]).endswith(f"{k3}.json"):
            path = Path(argv[-1])
            part = json.loads(path.read_text(encoding="utf-8"))
            part["pages"].pop("200")
            path.write_text(json.dumps(part), encoding="utf-8")
        return result

    hosts = await resolved_hosts(recorder, (Z2,))
    garage = FakeGarage(objects)
    pool = FakePool()

    report = await run_farm(
        [item],
        hosts,
        garage,
        pool,
        requeue=True,
        bucket=BUCKET,
        runner=short_last_part,
        merge=fake_merge,
    )

    assert report.results[0].status == "failed(merged_pages)"
    assert garage.puts == []
    assert pool.execute_calls == []


async def test_source_with_a_different_page_count_than_planned_fails_before_shipping() -> None:
    item, objects = pdf_item(999, text=False)
    item = replace(item, pages=1000)
    recorder = Recorder()
    hosts = await resolved_hosts(recorder, (Z2,))

    report = await run_farm(
        [item],
        hosts,
        FakeGarage(objects),
        FakePool(),
        requeue=True,
        bucket=BUCKET,
        runner=recorder,
        merge=fake_merge,
    )

    assert report.results[0].status == "failed(page_count_mismatch)"
    assert recorder.worker_starts == []


async def test_image_only_epub_is_converted_then_split(monkeypatch: pytest.MonkeyPatch) -> None:
    epub = b"PK\x03\x04 fake epub"
    sha = hashlib.sha256(epub).hexdigest()
    item = FarmItem("ENVEPUBSPLIT", "uploads/scan.epub", sha, "Scan", 900, fmt="epub")
    synthesized = pdf_bytes(900)
    monkeypatch.setattr(coordinator, "epub_to_pdf_bytes", lambda data: synthesized)
    recorder = Recorder()
    hosts = await resolved_hosts(recorder)
    garage = FakeGarage({item.garage_key: epub})

    report = await run_farm(
        [item],
        hosts,
        garage,
        FakePool(),
        requeue=False,
        bucket=BUCKET,
        runner=recorder,
        merge=fake_merge,
    )

    assert report.results[0].status == "ok"
    assert [recorder.job_pages[k] for k in _keys(item)] == [400, 400, 100]
    merged = _stored(garage, item, "docling.json")
    assert [t["text"] for t in merged["texts"]] == [f"page {n}" for n in range(1, 901)]
    assert _stored(garage, item, "meta.json")["fmt"] == "epub"


async def test_cancelling_the_run_kills_every_in_flight_part() -> None:
    item, objects = pdf_item(1000, text=False)
    keys = _keys(item)
    recorder = Recorder(hang_shas=set(keys))
    hosts = await resolved_hosts(recorder)

    task = asyncio.create_task(
        run_farm(
            [item],
            hosts,
            FakeGarage(objects),
            FakePool(),
            requeue=False,
            bucket=BUCKET,
            runner=recorder,
            timeouts=NO_TIMEOUT,
            merge=fake_merge,
        )
    )
    for _ in range(500):
        if len(recorder.worker_starts) == 3:
            break
        await asyncio.sleep(0.01)
    assert sorted(job for _, job in recorder.worker_starts) == sorted(keys)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(recorder.cancelled) == 3
    host_of = {job: host for host, job in recorder.worker_starts}
    for job in keys:
        commands = recorder.commands_for(host_of[job])
        target = win if host_of[job] == "win" else Z2
        if target is Z2:
            assert _linux_kill(job) in commands
        else:
            assert any("Stop-Process -Id 4242" in c and job in c for c in commands)
        assert _cleanup(target, job) in commands


# ---------------------------------------------------------------------------
# the real merge (docling-core)
# ---------------------------------------------------------------------------


def test_real_docling_core_merge_gives_absolute_pages_across_three_parts(tmp_path: Path) -> None:
    """Each part starts at page 1 (the worker refills dropped pages, so a part
    has exactly its page count); concatenate's shift makes them absolute."""
    doc_mod = pytest.importorskip("docling_core.types.doc")
    from PIL import Image

    sizes = [4, 4, 3]
    paths: list[Path] = []
    first = 1
    for index, count in enumerate(sizes):
        part = doc_mod.DoclingDocument(name=f"part{index}")
        for page_no in range(1, count + 1):
            part.add_page(page_no=page_no, size=doc_mod.Size(width=200, height=300))
            part.add_text(
                label=doc_mod.DocItemLabel.TEXT,
                text=f"page {first + page_no - 1}",
                prov=doc_mod.ProvenanceItem(
                    page_no=page_no,
                    bbox=doc_mod.BoundingBox(l=1, t=2, r=3, b=4),
                    charspan=(0, 6),
                ),
            )
        part.add_picture(
            image=doc_mod.ImageRef.from_pil(Image.new("RGB", (4, 4), "red"), dpi=72),
            prov=doc_mod.ProvenanceItem(
                page_no=2, bbox=doc_mod.BoundingBox(l=1, t=2, r=3, b=4), charspan=(0, 0)
            ),
        )
        path = tmp_path / f"part-{index}.json"
        part.save_as_json(path, image_mode=doc_mod.ImageRefMode.EMBEDDED)
        paths.append(path)
        first += count

    out = tmp_path / "merged.json"
    pages = merge_docling_parts(paths, out, "a" * 64)

    assert pages == list(range(1, 12))
    merged = doc_mod.DoclingDocument.load_from_json(out)
    assert merged.name == "a" * 64
    assert sorted(merged.pages) == list(range(1, 12))
    assert [(t.text, t.prov[0].page_no) for t in merged.texts] == [
        (f"page {n}", n) for n in range(1, 12)
    ]
    # Each part's picture sits on its part's page 2, now absolute: 2, 6, 10.
    assert [p.prov[0].page_no for p in merged.pictures] == [2, 6, 10]
    assert all(str(p.image.uri).startswith("data:image/png;base64,") for p in merged.pictures)


# ---------------------------------------------------------------------------
# book_done: exactly once per book, for its final status
# ---------------------------------------------------------------------------


async def test_book_done_once_for_books_left_without_a_usable_host(
    caplog: pytest.LogCaptureFixture,
) -> None:
    items, objects = farm_items([300, 200, 100, 50])
    recorder = Recorder(host_exit_codes={"win": 69, "lin": 69})
    hosts = await resolved_hosts(recorder)

    with caplog.at_level(logging.INFO, logger="vo_rag.ocr_farm.coordinator"):
        report = await run_farm(
            items,
            hosts,
            FakeGarage(objects),
            FakePool(),
            requeue=True,
            bucket=BUCKET,
            runner=recorder,
        )

    assert [r.status for r in report.results] == ["failed(no_usable_host)"] * 4
    assert _book_done(caplog) == {i.envelope_id: ["failed(no_usable_host)"] for i in items}


async def test_book_done_once_for_timeouts_whether_the_streak_resets_or_not(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A timed-out book may still be revived while its host's streak is open;
    it is logged once the streak resolves (a success, or the run's end)."""
    items, objects = farm_items([300, 200, 100])
    recorder = Recorder(expire_shas={items[0].sha256, items[2].sha256})
    hosts = await resolved_hosts(recorder, (replace(Z2, slots=1),))

    with caplog.at_level(logging.INFO, logger="vo_rag.ocr_farm.coordinator"):
        report = await run_farm(
            items,
            hosts,
            FakeGarage(objects),
            FakePool(),
            requeue=False,
            bucket=BUCKET,
            runner=recorder,
            timeouts=EXPIRE_ON_CUE,
        )

    assert [r.status for r in report.results] == ["failed(timeout)", "ok", "failed(timeout)"]
    assert recorder.expired == [("lin", items[0].sha256), ("lin", items[2].sha256)]
    assert _book_done(caplog) == {
        items[0].envelope_id: ["failed(timeout)"],
        items[1].envelope_id: ["ok"],
        items[2].envelope_id: ["failed(timeout)"],
    }


# ---------------------------------------------------------------------------
# _to_completion
# ---------------------------------------------------------------------------


async def test_to_completion_reraises_a_cancel_that_lands_as_the_step_finishes() -> None:
    """The step finishes on the same loop iteration the caller is cancelled:
    the cancellation must still reach the caller."""
    gate: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    async def step() -> None:
        await gate

    caller = asyncio.create_task(coordinator._to_completion(step()))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    gate.set_result(None)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert caller.cancelled()

"""RunPod OCR job: estimate, budget refusal, pod lifecycle, collect (VOR-1434).

pitwall, ssh and scp are all ``FakeRunner`` replies: no pod is created and
nothing is spent.
"""

from __future__ import annotations

import asyncio
import io
import json
import tarfile
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tests.ocr_farm._runpod_fakes import FakeCache, FakeRunner, pdf_item, pitwall_replies
from vo_rag.ocr_farm.coordinator import CommandResult, ocr_key, ocr_meta_key
from vo_rag.ocr_farm.runpod.job import (
    BudgetExceeded,
    JobDir,
    JobError,
    RunOptions,
    collect_job,
    estimate,
    extract_results,
    plan_job,
    run_job,
)
from vo_rag.ocr_farm.runpod.pitwall import Budget, Pitwall

BUCKET = "vorag-artifacts"
OK_NOTE = "ocr_worker: ok status=success pages=1"


async def _no_sleep(_: float) -> None:
    return None


def _budget(remaining: str = "20.00", cap: str = "10.00") -> Budget:
    return Budget(Decimal(remaining), Decimal(cap))


def _est(remaining: str = "20.00", cap: str = "10.00") -> Any:
    return estimate(
        [1000, 500], pages_per_gpu_hour=Decimal(300), setup_minutes=15,
        rate_usd_per_gpu_hour=Decimal("0.40"), budget=_budget(remaining, cap),
    )  # fmt: skip


def test_estimate_rounds_up_and_sizes_each_pods_ttl_from_its_shard() -> None:
    est = _est()

    # 1500/300 = 5 h OCR + 2 x 0.25 h setup = 5.5 GPU-h; x $0.40 = $2.20
    assert (est.gpu_hours, est.cost_usd) == (Decimal("5.50"), Decimal("2.20"))
    # pod 0: (1000/300 + 0.25) h x 60 x 1.25 = 268.75 -> 269 min; pod 1: 143.75 -> 144
    assert est.ttl_minutes == (269, 144)
    # reservation = TTL x rate, rounded up: 269/60 x 0.40 = 1.793 -> 1.80; 144/60 x 0.40 = 0.96
    assert est.reservation_usd == (Decimal("1.80"), Decimal("0.96"))
    assert est.reserved_total_usd == Decimal("2.76") and est.within_budget


def test_largest_shard_ttl_covers_its_pages_at_the_configured_rate() -> None:
    est = _est()
    biggest = max(range(est.pods), key=lambda k: est.shard_pages[k])

    ocr_minutes = Decimal(est.shard_pages[biggest]) / est.pages_per_gpu_hour * 60
    assert est.ttl_minutes[biggest] >= (ocr_minutes + est.setup_minutes) * Decimal("1.25")
    assert est.ttl_minutes[biggest] == max(est.ttl_minutes)


def test_budget_is_checked_on_reservations_not_only_raw_cost() -> None:
    est = _est(remaining="2.50")  # raw cost $2.20 fits, reservations $2.76 do not

    assert est.cost_usd <= est.remaining_budget_usd
    assert not est.within_budget
    assert est.refusals == [
        "pod reservations $2.76 (TTL x rate, per pod) exceed the remaining budget $2.50"
    ]


def test_each_pod_reservation_must_fit_the_per_request_cap() -> None:
    est = _est(cap="1.50")

    assert est.refusals == ["pod 0 reserves $1.80 > the per-request cap $1.50"]


async def test_plan_refuses_over_budget_and_writes_nothing(tmp_path: Path) -> None:
    item, _ = pdf_item(4000, "big")
    runner = FakeRunner(pitwall_replies(remaining="15.00", price="0.40"))
    job = JobDir(tmp_path / "job")

    with pytest.raises(BudgetExceeded) as caught:
        await plan_job(
            [item], job, Pitwall(runner), pods=4, gpu_type="NVIDIA A40", cloud="secure",
            pages_per_gpu_hour=Decimal(100),
        )  # fmt: skip

    est = caught.value.estimate
    assert est is not None and est.cost_usd == Decimal("16.40")  # (40 + 4x0.25) h x 0.40
    assert est.shard_pages == (1200, 1200, 800, 800)
    assert est.remaining_budget_usd == Decimal("15.00")
    assert not job.root.exists()
    assert runner.matching("pods", "create") == []


async def test_plan_refuses_when_reservations_exceed_budget_or_cap(tmp_path: Path) -> None:
    item, _ = pdf_item(1500, "b")  # units 400,400,400,300 -> 2 pods of 800 / 700 pages
    for remaining, cap, why in (("2.40", "10.00", "reservations"), ("20", "1.00", "cap")):
        runner = FakeRunner(pitwall_replies(remaining=remaining, per_request=cap, price="0.40"))
        with pytest.raises(BudgetExceeded, match=why):
            await plan_job(
                [item], JobDir(tmp_path / why), Pitwall(runner), pods=2, gpu_type="NVIDIA A40",
                cloud="secure", pages_per_gpu_hour=Decimal(300),
            )  # fmt: skip


async def test_plan_uses_the_catalogue_rate_and_caps_pods_at_units(tmp_path: Path) -> None:
    item, _ = pdf_item(500, "b")  # 2 units
    runner = FakeRunner(pitwall_replies(remaining="20.00", price="0.44"))
    job = JobDir(tmp_path)

    est, warnings = await plan_job(
        [item], job, Pitwall(runner), pods=5, gpu_type="NVIDIA A40", cloud="secure",
        pages_per_gpu_hour=Decimal(500), job_id="j1",
    )  # fmt: skip

    assert est.pods == 2 and est.rate_usd_per_gpu_hour == Decimal("0.44") and warnings == []
    meta = json.loads((tmp_path / "job.json").read_text())
    assert meta["job_id"] == "j1" and meta["estimate"]["cost_usd"] == str(est.cost_usd)
    assert meta["catalogue_rate_usd_per_gpu_hour"] == "0.44"


async def test_plan_refuses_a_staged_job_dir_unless_force_replan(tmp_path: Path) -> None:
    item, _ = pdf_item(5, "b")
    job = JobDir(tmp_path / "job")
    job.save("units.json", [])
    job.save("pods.json", [{"shard": 0, "pod_id": "p9", "name": "old", "state": "terminated"}])
    pitwall = Pitwall(FakeRunner(pitwall_replies()))
    kwargs: Any = {"pods": 1, "gpu_type": "NVIDIA A40", "cloud": "secure",
                   "pages_per_gpu_hour": Decimal(500)}  # fmt: skip

    with pytest.raises(JobError, match="already staged"):
        await plan_job([item], job, pitwall, **kwargs)

    _, warnings = await plan_job([item], job, pitwall, force_replan=True, **kwargs)
    assert not (job.root / "units.json").exists() and not (job.root / "pods.json").exists()
    assert (job.root / "job.json").exists()
    assert "moved to" in warnings[0]


async def test_force_replan_is_refused_while_any_pod_is_up(tmp_path: Path) -> None:
    item, _ = pdf_item(5, "b")
    job = JobDir(tmp_path / "job")
    job.save("job.json", {"job_id": "old"})
    job.save("units.json", [])
    job.save(
        "pods.json",
        [
            {"shard": 0, "pod_id": "p9", "name": "old-0", "state": "started"},
            {"shard": 1, "pod_id": "p8", "name": "old-1", "state": "terminated"},
        ],
    )

    with pytest.raises(JobError, match=r"pods still up for this job: old-0 \(p9\)") as caught:
        await plan_job(
            [item], job, Pitwall(FakeRunner(pitwall_replies())), pods=1, gpu_type="NVIDIA A40",
            cloud="secure", pages_per_gpu_hour=Decimal(500), force_replan=True,
        )  # fmt: skip

    assert "collect" in str(caught.value)
    assert sorted(p.name for p in job.root.iterdir()) == ["job.json", "pods.json", "units.json"]


async def test_refused_force_replan_leaves_the_old_plan_untouched(tmp_path: Path) -> None:
    item, _ = pdf_item(4000, "big")
    job = JobDir(tmp_path / "job")
    job.save("job.json", {"job_id": "old"})
    job.save("units.json", [{"unit": "u"}])
    job.save("pods.json", [{"shard": 0, "pod_id": "p9", "name": "old-0", "state": "terminated"}])
    before = {p.name: p.read_text() for p in job.root.iterdir()}

    with pytest.raises(BudgetExceeded):
        await plan_job(
            [item], job, Pitwall(FakeRunner(pitwall_replies(remaining="1.00"))), pods=4,
            gpu_type="NVIDIA A40", cloud="secure", pages_per_gpu_hour=Decimal(100),
            force_replan=True,
        )  # fmt: skip

    assert {p.name: p.read_text() for p in job.root.iterdir()} == before


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


async def _planned_job(tmp_path: Path, pages: list[int]) -> tuple[JobDir, FakeCache, RunOptions]:
    objects: dict[str, bytes] = {}
    items = []
    for n, count in enumerate(pages):
        item, data = pdf_item(count, f"b{n}")
        objects[item.garage_key] = data
        items.append(item)
    job = JobDir(tmp_path / "job")
    await plan_job(
        items, job, Pitwall(FakeRunner(pitwall_replies())), pods=2, gpu_type="NVIDIA A40",
        cloud="secure", pages_per_gpu_hour=Decimal(1000), job_id="j1",
    )  # fmt: skip
    key = tmp_path / "id_test"
    key.write_text("private")
    Path(f"{key}.pub").write_text("ssh-ed25519 AAAA test")
    return job, FakeCache(objects), RunOptions(ssh_key=key, poll_seconds=0)


def _ssh_commands(runner: FakeRunner) -> list[str]:
    return [c[-1] for c in runner.calls if c[0] == "ssh"]


async def test_run_creates_pods_with_ttl_bootstraps_and_starts_pools(tmp_path: Path) -> None:
    job, cache, opts = await _planned_job(tmp_path, [3, 2])
    runner = FakeRunner(pitwall_replies())

    lines = await run_job(job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts,
                          sleep=_no_sleep)  # fmt: skip

    creates = runner.matching("pods", "create")
    assert len(creates) == 2
    requests = [json.loads(c[c.index("--request-json") + 1]) for c in creates]
    request = requests[0]
    # each pod's own TTL, from its shard (units of 3 and 2 pages -> equal setup-bound TTLs)
    est = json.loads((job.root / "job.json").read_text())["estimate"]
    assert [r["ttl_minutes"] for r in requests] == est["ttl_minutes"]
    assert all(r["max_cost_per_hour"] == "0.40" for r in requests)
    assert request["gpu_type_ids"] == ["NVIDIA A40"] and request["ports"] == ["22/tcp"]
    assert request["env"] == {"PUBLIC_KEY": "ssh-ed25519 AAAA test"}
    assert creates[0][creates[0].index("--confirm") + 1] == request["name"]
    assert [line["state"] for line in lines if "pod" in line] == ["started", "started"]
    config = job.ssh_config.read_text()
    assert "Host vorag-ocr-j1-0" in config and "Port 40000" in config
    commands = _ssh_commands(runner)
    assert any(c.endswith("bash bootstrap.sh") for c in commands)
    starts = [c for c in commands if "run_units.sh run" in c]
    assert len(starts) == 2 and all("setsid nohup" in c and "< /dev/null &" in c for c in starts)
    assert not any("|" in c for c in commands), "nothing is piped through ssh"
    shipped = [c[-1] for c in runner.calls if c[0] == "scp"]
    assert "vorag-ocr-j1-0:/workspace/units.tar" in shipped


async def test_a_pod_without_cuda_is_terminated_and_reported(tmp_path: Path) -> None:
    job, cache, opts = await _planned_job(tmp_path, [3, 2])

    def no_cuda_on_pod1(argv: list[str]) -> CommandResult | None:
        if argv[0] == "ssh" and "vorag-ocr-j1-1" in argv and "bootstrap.sh" in argv[-1]:
            return CommandResult(3, "", "bootstrap: torch.cuda.is_available() is False")
        return None

    runner = FakeRunner(no_cuda_on_pod1, pitwall_replies())
    lines = await run_job(job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts,
                          sleep=_no_sleep)  # fmt: skip

    by_pod = {line["pod"]: line for line in lines if "pod" in line}
    assert by_pod["vorag-ocr-j1-0"]["state"] == "started"
    assert by_pod["vorag-ocr-j1-1"]["state"] == "terminated"
    assert "cuda.is_available() is False" in by_pod["vorag-ocr-j1-1"]["problem"]
    terminates = runner.matching("pods", "terminate")
    assert [t[4] for t in terminates] == ["pod1"]
    assert not any("vorag-ocr-j1-1" in c and "run_units.sh run" in c[-1]
                   for c in runner.calls if c[0] == "ssh")  # fmt: skip


async def test_run_rechecks_the_budget_before_creating_pods(tmp_path: Path) -> None:
    job, cache, opts = await _planned_job(tmp_path, [3])
    runner = FakeRunner(pitwall_replies(remaining="0.00"))

    with pytest.raises(BudgetExceeded):
        await run_job(job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts,
                      sleep=_no_sleep)  # fmt: skip

    assert runner.matching("pods", "create") == []


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------


def _docling(pages: int) -> bytes:
    size = {"width": 100.0, "height": 100.0}
    return json.dumps(
        {"schema_name": "DoclingDocument", "version": "1.7.0", "name": "u",
         "pages": {str(p): {"size": size, "page_no": p} for p in range(1, pages + 1)}}
    ).encode()  # fmt: skip


def _results_tar(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _pod_results(job: JobDir, bad_unit: str | None = None) -> dict[str, bytes]:
    """Per-alias results.tar contents: every unit ok except ``bad_unit``."""
    units = json.loads((job.root / "units.json").read_text())
    per_alias: dict[str, dict[str, bytes]] = {}
    for unit in units:
        files = per_alias.setdefault(f"vorag-ocr-j1-{unit['shard']}", {})
        name = unit["unit"]
        if name == bad_unit:
            files[f"units/{name}.note"] = b"Killed\n"
        else:
            files[f"units/{name}.json"] = _docling(int(unit["pages"]))
            files[f"units/{name}.note"] = OK_NOTE.encode()
    return {alias: _results_tar(files) for alias, files in per_alias.items()}


def _scp_results(tars: dict[str, bytes]) -> Any:
    def handle(argv: list[str]) -> CommandResult | None:
        if argv[0] == "scp" and argv[-2].endswith(":/workspace/results.tar"):
            Path(argv[-1]).write_bytes(tars[argv[-2].split(":")[0]])
            return CommandResult(0)
        return None

    return handle


async def _started_job(tmp_path: Path) -> tuple[JobDir, FakeCache, RunOptions]:
    job, cache, opts = await _planned_job(tmp_path, [3, 2])
    runner = FakeRunner(pitwall_replies())
    await run_job(job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts, sleep=_no_sleep)
    return job, cache, opts


async def test_collect_uploads_verifies_then_terminates_and_drops_aliases(tmp_path: Path) -> None:
    job, cache, opts = await _started_job(tmp_path)
    runner = FakeRunner(_scp_results(_pod_results(job)), pitwall_replies())

    lines, complete = await collect_job(
        job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts
    )

    assert complete, lines
    books = json.loads((job.root / "books.json").read_text())
    for book in books:
        sha = book["sha256"]
        assert cache.uploads.index(ocr_key(sha)) < cache.uploads.index(ocr_meta_key(sha))
    assert sorted(t[4] for t in runner.matching("pods", "terminate")) == ["pod0", "pod1"]
    assert not job.ssh_config.exists()
    packs = [c[-1] for c in runner.calls if c[0] == "ssh"]
    assert packs == ["cd /workspace && bash run_units.sh pack"] * 2
    assert lines[-1]["pods"] == "terminated"


async def test_partial_collect_leaves_pods_up_and_names_the_failed_unit(tmp_path: Path) -> None:
    job, cache, opts = await _started_job(tmp_path)
    units = json.loads((job.root / "units.json").read_text())
    bad = units[0]["unit"]
    runner = FakeRunner(_scp_results(_pod_results(job, bad_unit=bad)), pitwall_replies())

    lines, complete = await collect_job(
        job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts
    )

    assert not complete
    assert lines[-1]["units_failed"] == {bad: "Killed"}
    assert lines[-1]["pods"] == "left running"
    assert runner.matching("pods", "terminate") == []
    assert job.ssh_config.exists()
    sha = units[0]["sha256"]
    assert ocr_meta_key(sha) not in cache.uploads


def test_extract_results_ignores_paths_outside_units(tmp_path: Path) -> None:
    tar_path = tmp_path / "r.tar"
    tar_path.write_bytes(
        _results_tar(
            {"units/a.json": b"{}", "units/a.note": b"ok", "../evil.json": b"x",
             "units/../../evil.note": b"x", "units/b.pdf": b"x", "other/c.json": b"x"}
        )
    )  # fmt: skip
    out = tmp_path / "out"

    assert extract_results(tar_path, out) == 2
    assert sorted(p.name for p in out.iterdir()) == ["a.json", "a.note"]
    assert not (tmp_path / "evil.json").exists()


@pytest.mark.parametrize(
    ("status", "restarted"),
    [
        ("STATUS pool=4 ok=3 failed=1 pending=0 running=no alldone=yes", True),
        ("STATUS pool=4 ok=4 failed=0 pending=0 running=no alldone=yes", False),
        ("STATUS pool=4 ok=2 failed=1 pending=1 running=yes alldone=no", False),
        # the pool died (no ALLDONE) with units never started: restart too
        ("STATUS pool=4 ok=2 failed=0 pending=2 running=no alldone=no", True),
    ],
)
async def test_rerun_restarts_only_an_ended_pool_with_unfinished_units(
    tmp_path: Path, status: str, restarted: bool
) -> None:
    job, cache, opts = await _started_job(tmp_path)

    def status_reply(argv: list[str]) -> CommandResult | None:
        if argv[0] == "ssh" and argv[-1].endswith("run_units.sh status"):
            return CommandResult(0, status + "\n")
        return None

    runner = FakeRunner(status_reply, pitwall_replies())
    lines = await run_job(job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts,
                          sleep=_no_sleep)  # fmt: skip

    assert runner.matching("pods", "create") == []
    starts = [c for c in _ssh_commands(runner) if "run_units.sh run" in c]
    assert len(starts) == (2 if restarted else 0)
    assert all(line.get("restarted", False) is restarted for line in lines if "pod" in line)


async def test_price_override_never_lowers_the_pods_max_cost_per_hour(tmp_path: Path) -> None:
    item, data = pdf_item(3, "b")
    job = JobDir(tmp_path / "job")
    kwargs: Any = {"pods": 1, "gpu_type": "NVIDIA A40", "cloud": "secure",
                   "pages_per_gpu_hour": Decimal(500), "job_id": "j1"}  # fmt: skip
    est, _ = await plan_job(
        [item], job, Pitwall(FakeRunner(pitwall_replies(price="0.44"))),
        price_per_gpu_hour=Decimal("0.30"), **kwargs,
    )  # fmt: skip
    assert est.rate_usd_per_gpu_hour == Decimal("0.30")  # the estimate uses the override
    key = tmp_path / "k"
    key.write_text("x")
    Path(f"{key}.pub").write_text("ssh-ed25519 K")
    runner = FakeRunner(pitwall_replies())

    opts = RunOptions(ssh_key=key, poll_seconds=0)
    cache = FakeCache({item.garage_key: data})
    await run_job(job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts, sleep=_no_sleep)

    [create] = runner.matching("pods", "create")
    assert json.loads(create[create.index("--request-json") + 1])["max_cost_per_hour"] == "0.44"


async def test_a_lost_pod_is_recreated_for_its_pending_shard(tmp_path: Path) -> None:
    job, cache, opts = await _started_job(tmp_path)
    runner = FakeRunner(pitwall_replies(gone={"pod1"}))

    lines = await run_job(job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts,
                          sleep=_no_sleep)  # fmt: skip

    [create] = runner.matching("pods", "create")
    request = json.loads(create[create.index("--request-json") + 1])
    assert request["name"] == "vorag-ocr-j1-1-r1"
    assert create[create.index("--idempotency-key") + 1] == "vorag-ocr-j1-1-r1-create"
    pods = json.loads((job.root / "pods.json").read_text())
    assert [(p["name"], p["state"]) for p in pods] == [
        ("vorag-ocr-j1-0", "started"), ("vorag-ocr-j1-1", "terminated"),
        ("vorag-ocr-j1-1-r1", "started"),
    ]  # fmt: skip
    assert pods[1]["lost"] is True
    assert {"pod": "vorag-ocr-j1-1", "state": "terminated", "lost": True} in lines
    assert "Host vorag-ocr-j1-1-r1" in job.ssh_config.read_text()


async def test_recreating_a_lost_pod_rechecks_the_budget(tmp_path: Path) -> None:
    job, cache, opts = await _started_job(tmp_path)
    runner = FakeRunner(pitwall_replies(gone={"pod1"}, remaining="0.01"))

    with pytest.raises(BudgetExceeded):
        await run_job(job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts,
                      sleep=_no_sleep)  # fmt: skip

    assert runner.matching("pods", "create") == []


async def test_run_rechecks_the_per_request_cap_before_creating_pods(tmp_path: Path) -> None:
    job, cache, opts = await _planned_job(tmp_path, [3])
    runner = FakeRunner(pitwall_replies(per_request="0.01"))

    with pytest.raises(BudgetExceeded, match="per-request cap"):
        await run_job(job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts,
                      sleep=_no_sleep)  # fmt: skip

    assert runner.matching("pods", "create") == []


async def test_pods_bootstrap_concurrently_within_the_bound(tmp_path: Path) -> None:
    job, cache, opts = await _planned_job(tmp_path, [3, 2, 4])
    job_meta = json.loads((job.root / "job.json").read_text())
    assert job_meta["estimate"]["pods"] == 2  # _planned_job plans 2 pods
    inner = FakeRunner(pitwall_replies())
    state = {"now": 0, "peak": 0}
    both_in = asyncio.Event()

    async def runner(argv: Any, /) -> CommandResult:
        if argv[0] == "ssh" and argv[-1].endswith("bash bootstrap.sh"):
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            if state["now"] == 2:
                both_in.set()
            await asyncio.wait_for(both_in.wait(), 5)  # deadlocks if bootstraps are serial
            state["now"] -= 1
        return await inner(argv)

    lines = await run_job(job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts,
                          sleep=_no_sleep)  # fmt: skip

    assert state["peak"] == 2
    assert [line["state"] for line in lines if "pod" in line] == ["started", "started"]

    job2, cache2, _ = await _planned_job(tmp_path / "serial", [3, 2])
    order: list[str] = []

    async def serial(argv: Any, /) -> CommandResult:
        if argv[0] == "ssh" and argv[-1].endswith("bash bootstrap.sh"):
            order.append("in")
            await asyncio.sleep(0)
            order.append("out")
        return await inner(argv)

    one = RunOptions(ssh_key=opts.ssh_key, poll_seconds=0, setup_concurrency=1)
    await run_job(job2, Pitwall(serial), cache2, serial, bucket=BUCKET, opts=one, sleep=_no_sleep)
    assert order == ["in", "out", "in", "out"]


def test_every_pod_ttl_and_reservation_include_the_bootstrap_queue_wait() -> None:
    kwargs: Any = {
        "pages_per_gpu_hour": Decimal(300),
        "setup_minutes": 15,
        "rate_usd_per_gpu_hour": Decimal("0.40"),
        "budget": _budget("999", "999"),
    }
    fits = estimate([300] * 8, setup_concurrency=8, **kwargs)
    queued = estimate([300] * 9, setup_concurrency=8, **kwargs)
    serial = estimate([300] * 3, setup_concurrency=1, **kwargs)

    assert fits.queue_minutes == 0 and fits.ttl_minutes[0] == 94  # (60 + 15) x 1.25 = 93.75
    # 9 pods, 8 slots: ceil(9/8) - 1 = 1 extra setup allowance for every pod
    assert queued.queue_minutes == 15 and queued.ttl_minutes[0] == 113  # (60 + 30) x 1.25 = 112.5
    assert serial.queue_minutes == 30 and serial.ttl_minutes[0] == 132  # (60 + 45) x 1.25
    assert queued.reservation_usd[0] == Decimal("0.76")  # 113/60 x 0.40 = 0.7533 -> 0.76
    # the queued time bills too: 9 x (1 h + 0.5 h) = 13.5 GPU-h
    assert queued.gpu_hours == Decimal("13.50")


async def test_run_uses_the_planned_concurrency_for_queue_wait_in_ttl(tmp_path: Path) -> None:
    item, data = pdf_item(1200, "b")  # 3 units -> 3 pods
    job = JobDir(tmp_path / "job")
    await plan_job(
        [item], job, Pitwall(FakeRunner(pitwall_replies())), pods=3, gpu_type="NVIDIA A40",
        cloud="secure", pages_per_gpu_hour=Decimal(400), job_id="j1", setup_concurrency=2,
    )  # fmt: skip
    key = tmp_path / "k"
    key.write_text("x")
    Path(f"{key}.pub").write_text("ssh-ed25519 K")
    runner = FakeRunner(pitwall_replies())

    await run_job(job, Pitwall(runner), FakeCache({item.garage_key: data}), runner,
                  bucket=BUCKET, opts=RunOptions(ssh_key=key, poll_seconds=0),
                  sleep=_no_sleep)  # fmt: skip

    ttls = [json.loads(c[c.index("--request-json") + 1])["ttl_minutes"]
            for c in runner.matching("pods", "create")]  # fmt: skip
    # 400 pages / 400 per h = 60 min + 15 setup + 15 queue (3 pods, 2 slots) = 90 x 1.25
    assert ttls == [113, 113, 113]


async def test_an_exited_pod_is_terminated_through_pitwall(tmp_path: Path) -> None:
    job, cache, opts = await _started_job(tmp_path)
    exited: dict[str, Any] = {
        "id": "pod1",
        "name": "vorag-ocr-j1-1",
        "status": "exited",
        "public_ip": None,
        "port_mappings": {},
    }
    runner = FakeRunner(pitwall_replies(pods={"pod1": exited}))

    await run_job(job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts, sleep=_no_sleep)

    terminates = runner.matching("pods", "terminate")
    assert [t[4] for t in terminates] == ["pod1"]
    assert terminates[0][terminates[0].index("--idempotency-key") + 1] == "vorag-ocr-j1-1-terminate"
    pods = json.loads((job.root / "pods.json").read_text())
    assert pods[1]["state"] == "terminated" and pods[1]["lost"] is True


async def test_one_bring_up_raising_terminates_that_pod_and_the_rest_continue(
    tmp_path: Path,
) -> None:
    job, cache, opts = await _planned_job(tmp_path, [3, 2])
    inner = FakeRunner(pitwall_replies())

    async def runner(argv: Any, /) -> CommandResult:
        if argv[0] == "scp" and "vorag-ocr-j1-1:/workspace/units.tar" in argv:
            raise OSError("disk full while copying")
        if argv[0] == "ssh" and argv[-1].endswith("bash bootstrap.sh"):
            await asyncio.sleep(0)  # let the other pod's bring-up fail meanwhile
        return await inner(argv)

    lines = await run_job(job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts,
                          sleep=_no_sleep)  # fmt: skip

    by_pod = {line["pod"]: line for line in lines if "pod" in line}
    assert by_pod["vorag-ocr-j1-0"]["state"] == "started"
    assert by_pod["vorag-ocr-j1-1"]["state"] == "terminated"
    assert "OSError: disk full" in by_pod["vorag-ocr-j1-1"]["problem"]
    assert [t[4] for t in inner.matching("pods", "terminate")] == ["pod1"]
    assert lines[-1]["failed_shards"] == [1]
    pods = json.loads((job.root / "pods.json").read_text())
    assert [p["state"] for p in pods] == ["started", "terminated"]

    # a re-run recovers the failed shard with a new pod
    rerun = FakeRunner(pitwall_replies())
    await run_job(job, Pitwall(rerun), cache, rerun, bucket=BUCKET, opts=opts, sleep=_no_sleep)
    [create] = rerun.matching("pods", "create")
    assert json.loads(create[create.index("--request-json") + 1])["name"] == "vorag-ocr-j1-1-r1"


async def test_a_pod_whose_ssh_wait_raises_is_terminated_and_the_rest_continue(
    tmp_path: Path,
) -> None:
    job, cache, opts = await _planned_job(tmp_path, [3, 2])
    runner = FakeRunner(pitwall_replies(get_errors={"pod1"}))

    lines = await run_job(job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts,
                          sleep=_no_sleep)  # fmt: skip

    by_pod = {line["pod"]: line for line in lines if "pod" in line}
    assert by_pod["vorag-ocr-j1-0"]["state"] == "started"
    assert by_pod["vorag-ocr-j1-1"]["state"] == "terminated"
    assert "RunPod API timed out" in by_pod["vorag-ocr-j1-1"]["problem"]
    terminate = runner.matching("pods", "terminate")
    assert [t[4] for t in terminate] == ["pod1"]
    assert terminate[0][terminate[0].index("--idempotency-key") + 1] == "vorag-ocr-j1-1-terminate"
    assert lines[-1]["failed_shards"] == [1]
    pods = json.loads((job.root / "pods.json").read_text())
    assert [(p["state"], p.get("lost")) for p in pods] == [("started", None), ("terminated", True)]
    assert not any("vorag-ocr-j1-1" in c for c in runner.calls if c[0] in ("ssh", "scp"))

    # recoverable on re-run: the shard gets a new pod
    rerun = FakeRunner(pitwall_replies())
    await run_job(job, Pitwall(rerun), cache, rerun, bucket=BUCKET, opts=opts, sleep=_no_sleep)
    [create] = rerun.matching("pods", "create")
    assert json.loads(create[create.index("--request-json") + 1])["name"] == "vorag-ocr-j1-1-r1"


async def test_a_pod_that_never_gets_an_ssh_address_is_terminated(tmp_path: Path) -> None:
    job, cache, _ = await _planned_job(tmp_path, [3, 2])
    key = tmp_path / "id_test"
    opts = RunOptions(ssh_key=key, poll_seconds=0, ssh_ready_attempts=2)
    no_address: dict[str, Any] = {"pod1": {"id": "pod1", "status": "running", "public_ip": None,
                                           "port_mappings": {}}}  # fmt: skip
    runner = FakeRunner(pitwall_replies(pods=no_address))

    lines = await run_job(job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts,
                          sleep=_no_sleep)  # fmt: skip

    by_pod = {line["pod"]: line for line in lines if "pod" in line}
    assert by_pod["vorag-ocr-j1-1"]["problem"] == "no ssh address from pitwall"
    assert [t[4] for t in runner.matching("pods", "terminate")] == ["pod1"]
    assert lines[-1]["failed_shards"] == [1]


async def test_a_failed_terminate_after_ssh_wait_is_reported_as_still_billing(
    tmp_path: Path,
) -> None:
    job, cache, opts = await _planned_job(tmp_path, [3, 2])
    runner = FakeRunner(pitwall_replies(get_errors={"pod1"}, fail={"pods terminate": 1}))

    lines = await run_job(job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts,
                          sleep=_no_sleep)  # fmt: skip

    pod1 = [line for line in lines if line.get("pod") == "vorag-ocr-j1-1"]
    assert pod1 and "TERMINATE FAILED" in str(pod1[-1]["problem"])
    assert 1 in lines[-1]["failed_shards"]
    pods = json.loads((job.root / "pods.json").read_text())
    assert pods[1]["state"] != "started"


# ---------------------------------------------------------------------------
# Unattended billing
# ---------------------------------------------------------------------------


def _fail_nth_create(n: int, *, timeout: bool = False) -> Any:
    """Fail the ``n``-th (0-based) ``pods create``: rc=1, or a client-side
    timeout (TimeoutError from the runner) when ``timeout``."""
    seen = {"creates": 0}

    def handle(argv: list[str]) -> CommandResult | None:
        if argv[:4] == ["pitwall", "runpod", "pods", "create"]:
            index = seen["creates"]
            seen["creates"] += 1
            if index == n:
                if timeout:
                    raise TimeoutError
                return CommandResult(1, json.dumps({"error": "provider_unavailable"}), "")
        return None

    return handle


async def test_a_failed_create_terminates_the_pods_this_run_created(tmp_path: Path) -> None:
    job, cache, opts = await _planned_job(tmp_path, [3, 2])
    runner = FakeRunner(_fail_nth_create(1), pitwall_replies())

    with pytest.raises(JobError, match="pod create failed") as caught:
        await run_job(job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts,
                      sleep=_no_sleep)  # fmt: skip

    assert "terminated pod0" in str(caught.value)
    assert [t[4] for t in runner.matching("pods", "terminate")] == ["pod0"]
    pods = json.loads((job.root / "pods.json").read_text())
    assert [(p["pod_id"], p["state"]) for p in pods] == [("pod0", "terminated")]
    assert not any(c[0] == "ssh" for c in runner.calls), "nothing is bootstrapped"


async def test_a_create_that_timed_out_but_succeeded_is_found_and_recorded(
    tmp_path: Path,
) -> None:
    """The create timed out client-side but pitwall made the pod: it is looked
    up by its unique name, recorded, and run like any other pod."""
    job, cache, opts = await _planned_job(tmp_path, [3, 2])
    runner = FakeRunner(
        _fail_nth_create(1, timeout=True),
        pitwall_replies(listed=[{"id": "pod9", "name": "vorag-ocr-j1-1", "status": "running"}]),
    )

    lines = await run_job(job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts,
                          sleep=_no_sleep)  # fmt: skip

    pods = json.loads((job.root / "pods.json").read_text())
    assert [(p["pod_id"], p["name"]) for p in pods] == [
        ("pod0", "vorag-ocr-j1-0"), ("pod9", "vorag-ocr-j1-1"),
    ]  # fmt: skip
    assert [line["state"] for line in lines if "pod" in line] == ["started", "started"]
    assert runner.matching("pods", "terminate") == []


async def test_a_timed_out_create_that_made_no_pod_cleans_up_and_fails(tmp_path: Path) -> None:
    job, cache, opts = await _planned_job(tmp_path, [3, 2])
    runner = FakeRunner(_fail_nth_create(1, timeout=True), pitwall_replies())

    with pytest.raises(JobError, match="pod create failed"):
        await run_job(job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts,
                      sleep=_no_sleep)  # fmt: skip

    assert runner.matching("pods", "list")
    assert [t[4] for t in runner.matching("pods", "terminate")] == ["pod0"]


async def test_collect_tries_every_terminate_and_reports_each_failure(tmp_path: Path) -> None:
    """The first failed terminate no longer stops
    the loop; every pod is tried, every failure is named, and the result is
    not ok so the CLI exits non-zero."""
    job, cache, opts = await _started_job(tmp_path)

    def pod0_terminate_fails(argv: list[str]) -> CommandResult | None:
        if argv[:4] == ["pitwall", "runpod", "pods", "terminate"] and argv[4] == "pod0":
            return CommandResult(1, json.dumps({"error": "provider_unavailable"}), "")
        return None

    runner = FakeRunner(pod0_terminate_fails, _scp_results(_pod_results(job)), pitwall_replies())
    lines, ok = await collect_job(job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts)

    assert not ok
    assert sorted(t[4] for t in runner.matching("pods", "terminate")) == ["pod0", "pod1"]
    summary = lines[-1]
    assert summary["complete"] is True  # every book verified
    assert list(summary["terminate_failed"]) == ["pod0"]
    assert "pod may still bill" in summary["pods"]
    pods = {p["pod_id"]: p["state"] for p in json.loads((job.root / "pods.json").read_text())}
    assert pods == {"pod0": "started", "pod1": "terminated"}
    assert job.ssh_config.exists(), "aliases stay while a pod may still be up"


async def test_run_checks_pitwall_before_staging_anything(tmp_path: Path) -> None:
    """An unreachable pitwall (missing binary) fails the run
    before any unit is staged or any Garage object is read."""
    from vo_rag.ocr_farm.coordinator import subprocess_runner
    from vo_rag.ocr_farm.runpod.pitwall import PitwallError

    job, cache, opts = await _planned_job(tmp_path, [3, 2])

    with pytest.raises(PitwallError):
        await run_job(job, Pitwall(subprocess_runner, "/nonexistent/pitwall-binary"), cache,
                      FakeRunner(), bucket=BUCKET, opts=opts, sleep=_no_sleep)  # fmt: skip

    assert not (job.root / "units.json").exists()
    assert cache.gets == []


async def test_a_timed_out_create_whose_lookup_fails_says_a_pod_may_be_billing(
    tmp_path: Path,
) -> None:
    """When the create timed out AND the name lookup
    failed, nobody knows whether RunPod made the pod. The error says so and
    names the pod to look for."""
    job, cache, opts = await _planned_job(tmp_path, [3, 2])
    runner = FakeRunner(_fail_nth_create(1, timeout=True), pitwall_replies(fail={"pods list": 1}))

    with pytest.raises(JobError) as caught:
        await run_job(job, Pitwall(runner), cache, runner, bucket=BUCKET, opts=opts,
                      sleep=_no_sleep)  # fmt: skip

    message = str(caught.value)
    assert "may exist unrecorded" in message
    assert "billing until its TTL" in message
    assert "vorag-ocr-j1-1" in message
    assert "pitwall runpod pods list" in message
    assert [t[4] for t in runner.matching("pods", "terminate")] == ["pod0"]

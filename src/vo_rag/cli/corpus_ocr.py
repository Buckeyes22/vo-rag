"""`vo-rag corpus ocr-farm plan|bootstrap|run` (VOR-1427) and
`vo-rag corpus ocr-farm runpod plan|run|collect|verify` (VOR-1434).

Run on localhost, which reaches the GPU workers (``win``, ``lin``)
by ssh alias. ``plan`` is read-only (SELECTs + Garage GETs); ``bootstrap``
only touches the worker hosts; ``run`` writes ``ocr/<sha256>/*.json`` to
Garage and, unless ``--no-requeue``, flips each OCR'd envelope from
``failed`` back to ``received`` so the dispatcher ingests it with the cached OCR.

Output: one JSON object per line; exit 1 when any line reports a failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any

import click

from vo_rag.ocr_farm.coordinator import (
    FarmItem,
    Runner,
    bootstrap_worker,
    plan_backlog,
    resolve_hosts,
    run_farm,
    subprocess_runner,
)
from vo_rag.ocr_farm.hosts import DEFAULT_HOSTS, FARM_SECONDS_PER_PAGE, FarmHost
from vo_rag.ocr_farm.runpod import job as runpod_job
from vo_rag.ocr_farm.runpod.pitwall import Pitwall, PitwallError
from vo_rag.ocr_farm.runpod.stage import load_backlog
from vo_rag.ocr_farm.runpod.verify import verify_books

_DEFAULT_STATE = "~/.config/vo-rag/ocr-farm-hosts.json"
#: Exit code of a run stopped by SIGTERM / SIGINT (128 + SIGINT, the shell convention).
EXIT_CANCELLED = 130


class _StderrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Writes to whatever ``sys.stderr`` is at emit time (click / pytest swap it)."""

    @property
    def stream(self) -> Any:
        return sys.stderr

    @stream.setter
    def stream(self, value: Any) -> None:
        _ = value


def _enable_progress_logging() -> None:
    """Show the coordinator's INFO events (``ocr_farm.book_done`` …) on stderr.

    Nothing configured logging before, so INFO was dropped and a run's
    progress was invisible until its final summary (hours for a full run).
    Only the ``vo_rag.ocr_farm`` logger is raised to INFO: httpx and friends
    stay quiet. Idempotent.
    """
    logger = logging.getLogger("vo_rag.ocr_farm")
    logger.setLevel(logging.INFO)
    if not any(isinstance(handler, _StderrHandler) for handler in logger.handlers):
        handler = _StderrHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)


def _run_until_signalled[T](main: Callable[[], Awaitable[T]]) -> T:
    """``asyncio.run(main())``, with SIGTERM / SIGINT cancelling it.

    ``systemctl stop`` sends SIGTERM, which used to end the process outright:
    ``run_farm``'s cancellation path (kill each in-flight worker by its
    verified PID, then clean up its files) never ran, and the remote workers
    kept going on both hosts. Raises ``asyncio.CancelledError`` after cleanup.
    """

    async def _guarded() -> T:
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        if task is None:  # pragma: no cover - asyncio.run always has a task
            raise RuntimeError("_run_until_signalled needs a running task")
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, task.cancel)
        return await main()

    return asyncio.run(_guarded())


def _emit(payload: dict[str, Any]) -> None:
    click.echo(json.dumps(payload, default=str))


def _runner() -> Runner:
    return subprocess_runner


def _garage() -> tuple[Any, str]:
    """``(garage client, artifacts bucket)`` from settings (the RunPod job needs no DB)."""
    from vo_rag.core.config import get_settings
    from vo_rag.core.storage.garage import GarageClient

    settings = get_settings()
    garage = GarageClient(
        endpoint_url=settings.garage.endpoint,
        access_key=settings.garage.access_key,
        secret_key=settings.garage.secret_key,
        region=settings.garage.region,
    )
    return garage, settings.garage.buckets.artifacts


def _state_path() -> Path:
    return Path(os.environ.get("VORAG_OCR_FARM_STATE") or _DEFAULT_STATE).expanduser()


@asynccontextmanager
async def _resources() -> AsyncIterator[tuple[Any, Any, str, Any]]:
    """``(pool, garage, bucket, qdrant_client)`` from settings; closed on exit.

    The Qdrant client (``QdrantStore``'s) is only used for the read-only
    ``source_book`` facet in ``plan_backlog``.
    """
    from vo_rag.core.config import get_settings
    from vo_rag.core.db.postgres import close_pool, get_pool
    from vo_rag.core.storage.garage import GarageClient
    from vo_rag.core.storage.qdrant import QdrantStore

    settings = get_settings()
    garage = GarageClient(
        endpoint_url=settings.garage.endpoint,
        access_key=settings.garage.access_key,
        secret_key=settings.garage.secret_key,
        region=settings.garage.region,
    )
    store = QdrantStore(
        settings.qdrant.url,
        settings.qdrant.api_key or None,
        prefer_grpc=settings.qdrant.prefer_grpc,
    )
    try:
        pool = await get_pool(settings.database.url, min_size=1, max_size=4)
        try:
            yield pool, garage, settings.garage.buckets.artifacts, store._client  # noqa: SLF001
        finally:
            await close_pool()
    finally:
        await store._client.close()  # noqa: SLF001


def _select_hosts(spec: str | None) -> tuple[FarmHost, ...]:
    if not spec:
        return DEFAULT_HOSTS
    by_alias = {host.alias.lower(): host for host in DEFAULT_HOSTS}
    chosen: list[FarmHost] = []
    for alias in (part.strip() for part in spec.split(",")):
        if not alias:
            continue
        host = by_alias.get(alias.lower())
        if host is None:
            known = ", ".join(h.alias for h in DEFAULT_HOSTS)
            raise click.BadParameter(
                f"unknown host {alias!r} (known: {known})", param_hint="--hosts"
            )
        if host not in chosen:
            chosen.append(host)
    if not chosen:
        raise click.BadParameter("no hosts given", param_hint="--hosts")
    return tuple(chosen)


_HOSTS_OPTION = click.option(
    "--hosts",
    "hosts_spec",
    default=None,
    help="Comma-separated worker aliases (default: win,lin).",
)


def register(group: click.Group) -> None:
    @group.group("ocr-farm")
    def ocr_farm() -> None:
        """GPU OCR farm for failed scanned books (configured worker hosts)."""
        _enable_progress_logging()

    @ocr_farm.command("plan")
    def plan() -> None:
        """List the books to OCR with total pages and estimated hours (read-only)."""

        async def _plan() -> list[FarmItem]:
            async with _resources() as (pool, garage, bucket, qdrant):
                return await plan_backlog(pool, garage, qdrant, bucket=bucket)

        items = sorted(asyncio.run(_plan()), key=lambda item: item.pages, reverse=True)
        for item in items:
            _emit(item.to_dict())
        pending = [item for item in items if not item.cached and not item.problem]
        pages = sum(item.pages for item in pending)
        _emit(
            {
                "books": len(items),
                "to_ocr": len(pending),
                "cached": sum(1 for item in items if item.cached),
                "problems": sum(1 for item in items if item.problem),
                "pages": pages,
                "s_per_page": FARM_SECONDS_PER_PAGE,
                "est_hours": round(pages * FARM_SECONDS_PER_PAGE / 3600, 1),
            }
        )

    @ocr_farm.command("bootstrap")
    @_HOSTS_OPTION
    def bootstrap(hosts_spec: str | None) -> None:
        """Create each worker's venv (CUDA wheels) and verify torch sees the GPU."""
        hosts = _select_hosts(hosts_spec)
        runner = _runner()

        async def _bootstrap() -> list[dict[str, Any]]:
            resolved = await resolve_hosts(hosts, runner, state_path=_state_path())
            return [(await bootstrap_worker(host, runner=runner)).to_dict() for host in resolved]

        lines = asyncio.run(_bootstrap())
        for line in lines:
            _emit(line)
        if not all(line["ok"] for line in lines):
            raise click.exceptions.Exit(1)

    @ocr_farm.command("run")
    @click.option(
        "--limit",
        type=click.IntRange(min=1),
        default=None,
        help="Only OCR the N smallest pending books (smoke tests).",
    )
    @click.option(
        "--no-requeue",
        is_flag=True,
        default=False,
        help="Store the OCR in Garage but leave the envelopes failed.",
    )
    @_HOSTS_OPTION
    def run(limit: int | None, no_requeue: bool, hosts_spec: str | None) -> None:
        """OCR the backlog on the workers and store ocr/<sha256>/docling.json."""
        hosts = _select_hosts(hosts_spec)
        runner = _runner()

        async def _run() -> Any:
            async with _resources() as (pool, garage, bucket, qdrant):
                items = await plan_backlog(pool, garage, qdrant, bucket=bucket)
                if limit is not None:
                    pending = [i for i in items if not i.cached and not i.problem]
                    items = sorted(pending, key=lambda item: item.pages)[:limit]
                resolved = await resolve_hosts(hosts, runner, state_path=_state_path())
                return await run_farm(
                    items,
                    resolved,
                    garage,
                    pool,
                    requeue=not no_requeue,
                    bucket=bucket,
                    runner=runner,
                )

        try:
            report = _run_until_signalled(_run)
        except asyncio.CancelledError:
            _emit({"cancelled": True})
            raise click.exceptions.Exit(EXIT_CANCELLED) from None
        for result in report.results:
            _emit(result.to_dict())
        summary = report.summary()
        _emit(summary)
        if summary["failed"] or summary["host_errors"]:
            raise click.exceptions.Exit(1)

    _register_runpod(ocr_farm)


_JOB_DIR = click.option(
    "--job-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Local job directory (plan writes it; run/collect/verify read it).",
)
_PITWALL = click.option(
    "--pitwall", "pitwall_bin", default="pitwall", show_default=True, help="pitwall CLI to call."
)
#: Exit code of a plan refused for budget.
EXIT_OVER_BUDGET = 2


def _decimal(value: float) -> Decimal:
    """``6000.0`` -> ``Decimal("6000")``; other floats via their repr."""
    return Decimal(int(value)) if value.is_integer() else Decimal(str(value))


def _register_runpod(ocr_farm: click.Group) -> None:
    @ocr_farm.group("runpod")
    def runpod() -> None:
        """OCR the backlog on RunPod GPU pods as one budgeted job.

        plan (estimate vs budget) -> run (stage, create pods, bootstrap, start)
        -> collect (pull, merge, upload, verify; terminate pods when all verify).
        """

    @runpod.command("plan")
    @click.option(
        "--backlog",
        type=click.Path(dir_okay=False, exists=True, path_type=Path),
        required=True,
        help="JSONL from `vo-rag corpus ocr-farm plan` (cached/problem books are skipped).",
    )
    @_JOB_DIR
    @click.option(
        "--pages-per-gpu-hour",
        type=click.FloatRange(min=0, min_open=True),
        default=6000,
        show_default=True,
        help="OCR throughput of ONE pod GPU at its full worker pool. Measured 2026-09-26 at "
        "1.7-2.9 pages/s per pod (6,100-10,400 pages/h); 6000 is conservative, so estimates "
        "and pod TTLs err high.",
    )
    @click.option("--pods", type=click.IntRange(min=1), default=4, show_default=True)
    @click.option("--gpu-type", default=runpod_job.DEFAULT_GPU, show_default=True)
    @click.option(
        "--cloud", type=click.Choice(["secure", "community"]), default="secure", show_default=True
    )
    @click.option(
        "--setup-minutes",
        type=click.IntRange(min=0),
        default=runpod_job.DEFAULT_SETUP_MINUTES,
        show_default=True,
        help="Per-pod allowance for bootstrap and pulls (an allowance, not a measurement).",
    )
    @click.option(
        "--price-per-gpu-hour",
        type=click.FloatRange(min=0, min_open=True),
        default=None,
        help="USD per GPU-hour; default: the pitwall catalogue's on-demand price.",
    )
    @click.option(
        "--setup-concurrency",
        type=click.IntRange(min=1),
        default=runpod_job.DEFAULT_SETUP_CONCURRENCY,
        show_default=True,
        help="Pods bootstrapped at once. Pods queued for a slot already bill, so every "
        "pod's TTL and reservation include (ceil(pods / this) - 1) x --setup-minutes.",
    )
    @click.option(
        "--force-replan",
        is_flag=True,
        default=False,
        help="Re-plan an already staged job dir: its units/books/pods state is moved aside "
        "(pods it lists are NOT terminated).",
    )
    @_PITWALL
    def runpod_plan(
        backlog: Path,
        job_dir: Path,
        pages_per_gpu_hour: float,
        pods: int,
        gpu_type: str,
        cloud: str,
        setup_minutes: int,
        price_per_gpu_hour: float | None,
        setup_concurrency: int,
        force_replan: bool,
        pitwall_bin: str,
    ) -> None:
        """Estimate pages, GPU-hours, cost and pod reservations; refuse if the
        reservations exceed the remaining budget or a pod's exceeds the per-request cap."""
        items = load_backlog(backlog)
        try:
            estimate, warnings = asyncio.run(
                runpod_job.plan_job(
                    items,
                    runpod_job.JobDir(job_dir),
                    Pitwall(_runner(), pitwall_bin),
                    pods=pods,
                    gpu_type=gpu_type,
                    cloud=cloud,  # type: ignore[arg-type]
                    pages_per_gpu_hour=_decimal(pages_per_gpu_hour),
                    setup_minutes=setup_minutes,
                    price_per_gpu_hour=(
                        Decimal(str(price_per_gpu_hour)) if price_per_gpu_hour else None
                    ),
                    force_replan=force_replan,
                    setup_concurrency=setup_concurrency,
                )
            )
        except runpod_job.BudgetExceeded as exc:
            if exc.estimate is not None:
                _emit({"books": len(items), **exc.estimate.to_dict()})
            _emit({"refused": str(exc)})
            raise click.exceptions.Exit(EXIT_OVER_BUDGET) from None
        except (PitwallError, runpod_job.JobError) as exc:
            _emit({"error": str(exc)})
            raise click.exceptions.Exit(1) from None
        for warning in warnings:
            _emit({"warning": warning})
        _emit({"books": len(items), **estimate.to_dict(), "job_dir": str(job_dir)})

    @runpod.command("run")
    @_JOB_DIR
    @click.option("--image", default=runpod_job.DEFAULT_IMAGE, show_default=True)
    @click.option(
        "--ssh-key",
        type=click.Path(dir_okay=False, path_type=Path),
        default=Path("~/.ssh/id_ed25519"),
        show_default=True,
        help="Private key for the pods; its .pub is given to each pod.",
    )
    @click.option(
        "--max-workers",
        type=click.IntRange(min=1),
        default=8,
        show_default=True,
        help="Upper bound per pod; the pod sizes its pool from its cgroup memory at 9 GB/worker.",
    )
    @click.option(
        "--setup-concurrency",
        type=click.IntRange(min=1),
        default=None,
        help="Override the planned bootstrap concurrency (default: the plan's; TTLs are "
        "sized for it).",
    )
    @_PITWALL
    def runpod_run(
        job_dir: Path,
        image: str,
        ssh_key: Path,
        max_workers: int,
        setup_concurrency: int | None,
        pitwall_bin: str,
    ) -> None:
        """Stage units, create pods (TTL = estimate + 25%), bootstrap, start the pools."""
        garage, bucket = _garage()
        runner = _runner()
        opts = runpod_job.RunOptions(
            image=image,
            ssh_key=ssh_key,
            max_workers=max_workers,
            setup_concurrency=setup_concurrency,
        )
        try:
            lines = asyncio.run(
                runpod_job.run_job(
                    runpod_job.JobDir(job_dir),
                    Pitwall(runner, pitwall_bin),
                    garage,
                    runner,
                    bucket=bucket,
                    opts=opts,
                )
            )
        except (runpod_job.BudgetExceeded, PitwallError, runpod_job.JobError) as exc:
            _emit({"error": str(exc)})
            raise click.exceptions.Exit(1) from None
        for line in lines:
            _emit(line)
        if any(line.get("problem") for line in lines):
            raise click.exceptions.Exit(1)

    @runpod.command("collect")
    @_JOB_DIR
    @click.option(
        "--overwrite",
        is_flag=True,
        default=False,
        help="Replace an existing ocr/<sha256>/ cache (re-dos).",
    )
    @_PITWALL
    def runpod_collect(job_dir: Path, overwrite: bool, pitwall_bin: str) -> None:
        """Pull results, merge, upload (docling.json then meta.json), verify; terminate
        the pods and remove the ssh aliases only when every book verifies."""
        garage, bucket = _garage()
        runner = _runner()
        try:
            lines, complete = asyncio.run(
                runpod_job.collect_job(
                    runpod_job.JobDir(job_dir),
                    Pitwall(runner, pitwall_bin),
                    garage,
                    runner,
                    bucket=bucket,
                    overwrite=overwrite,
                )
            )
        except (PitwallError, runpod_job.JobError) as exc:
            _emit({"error": str(exc)})
            raise click.exceptions.Exit(1) from None
        for line in lines:
            _emit(line)
        if not complete:
            raise click.exceptions.Exit(1)

    @runpod.command("verify")
    @_JOB_DIR
    def runpod_verify(job_dir: Path) -> None:
        """Check every staged book's ocr/<sha256>/ cache in Garage (read-only)."""
        garage, bucket = _garage()
        try:
            books = runpod_job.JobDir(job_dir).load("books.json")
        except runpod_job.JobError as exc:
            _emit({"error": str(exc)})
            raise click.exceptions.Exit(1) from None
        results = asyncio.run(verify_books(books, garage, bucket=bucket))
        for result in results:
            _emit(result)
        bad = sum(1 for r in results if not r["ok"])
        _emit({"books": len(results), "verified": len(results) - bad, "bad": bad})
        if bad:
            raise click.exceptions.Exit(1)

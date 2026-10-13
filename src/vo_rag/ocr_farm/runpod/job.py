"""The RunPod OCR job: plan, run, collect (VOR-1434).

A job lives in one local directory (``--job-dir``)::

    job.json      plan: job id, GPU, pods, the estimate (written by ``plan``)
    backlog.json  the books to OCR (from ``ocr-farm plan`` output)
    books.json / units.json / units/*.pdf   staged by ``run``
    pods.json     one entry per pod: shard, pod id, ssh alias, state
    ssh_config    temporary ssh aliases for the pods (removed by ``collect``)
    pulled/<k>.tar, out/<unit>.json|.note   results pulled by ``collect``

Cost controls: ``plan`` refuses when the ceiling-rounded estimate exceeds the
GPU broker's remaining monthly budget, and ``run`` re-checks it before any pod
is created; each pod gets a pitwall TTL of the estimate + 25%, so an idle pod
stops billing on its own; results travel pod -> file -> scp (never a tar piped
through ssh); ``collect`` terminates the pods itself, but only once every book verifies.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import tarfile
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

from vo_rag.ocr_farm.coordinator import WORKER_DIR, CommandResult, FarmItem, Runner, part_ranges
from vo_rag.ocr_farm.hosts import WORKER_SCRIPT
from vo_rag.ocr_farm.runpod.merge_upload import CacheStore, merge_and_upload, unit_state
from vo_rag.ocr_farm.runpod.pitwall import Budget, Cloud, Pitwall, PitwallError
from vo_rag.ocr_farm.runpod.stage import shard_units, stage_books
from vo_rag.ocr_farm.runpod.verify import verify_books

RUNPOD_DIR = Path(__file__).resolve().parent
SHELL_FILES = ("bootstrap.sh", "run_units.sh")
REMOTE_DIR = "/workspace"
DEFAULT_GPU = "NVIDIA A40"
#: Any image with python3 (venv) and sshd works: bootstrap.sh installs its own pinned torch.
DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
#: Pod TTL = estimated wall time x this (+25%).
TTL_MARGIN = Decimal("1.25")
MAX_TTL_MINUTES = 10080  # pitwall's PodCreateRequest.ttl_minutes limit
#: Allowance per pod for venv install, model warm-up and result pulls. NOT measured:
#: no setup times were recorded. Override with --setup-minutes.
DEFAULT_SETUP_MINUTES = 15
#: Pods bootstrapped at once. Pods past the first ``setup_concurrency`` wait for a
#: bootstrap slot while already billing, so the TTL and reservation of every pod
#: include the queue wait: (ceil(pods / setup_concurrency) - 1) x setup allowance.
DEFAULT_SETUP_CONCURRENCY = 8

Sleep = Callable[[float], Awaitable[None]]


class BudgetExceeded(RuntimeError):
    """The estimate is above pitwall's remaining monthly budget."""

    def __init__(self, message: str, estimate: Estimate | None = None) -> None:
        super().__init__(message)
        self.estimate = estimate


class JobError(RuntimeError):
    """The job directory is missing a step or a pod could not be set up."""


# ---------------------------------------------------------------------------
# estimate
# ---------------------------------------------------------------------------


def _ceil(value: Decimal, places: str = "0.01") -> Decimal:
    return value.quantize(Decimal(places), rounding=ROUND_CEILING)


@dataclass(frozen=True)
class Estimate:
    """What a set of pods will cost, and what pitwall will reserve for them.

    Each pod gets its own TTL, sized from its own shard: (shard pages / rate +
    setup allowance) x 1.25, in whole minutes. A pitwall pod create reserves
    about TTL x hourly price, so the budget is checked on those reservations
    (always above the raw cost): their sum against the remaining monthly
    budget, and each one against the per-request cap.
    """

    shard_ids: tuple[int, ...]
    shard_pages: tuple[int, ...]
    pages_per_gpu_hour: Decimal
    setup_minutes: int
    rate_usd_per_gpu_hour: Decimal
    remaining_budget_usd: Decimal
    per_request_max_usd: Decimal
    gpu_hours: Decimal
    cost_usd: Decimal
    ttl_minutes: tuple[int, ...]
    reservation_usd: tuple[Decimal, ...]
    reserved_total_usd: Decimal
    setup_concurrency: int = DEFAULT_SETUP_CONCURRENCY
    #: Worst-case wait for a bootstrap slot, included in every pod's TTL.
    queue_minutes: int = 0

    @property
    def pages(self) -> int:
        return sum(self.shard_pages)

    @property
    def pods(self) -> int:
        return len(self.shard_pages)

    @property
    def refusals(self) -> list[str]:
        out: list[str] = []
        remaining, cap = self.remaining_budget_usd, self.per_request_max_usd
        if self.cost_usd > remaining:
            out.append(f"estimate ${self.cost_usd} exceeds the remaining budget ${remaining}")
        if self.reserved_total_usd > remaining:
            out.append(
                f"pod reservations ${self.reserved_total_usd} (TTL x rate, per pod) exceed "
                f"the remaining budget ${remaining}"
            )
        for k, ttl, reserve in zip(
            self.shard_ids, self.ttl_minutes, self.reservation_usd, strict=True
        ):
            if reserve > cap:
                out.append(f"pod {k} reserves ${reserve} > the per-request cap ${cap}")
            if ttl > MAX_TTL_MINUTES:
                out.append(f"pod {k} needs a {ttl} min TTL > pitwall's {MAX_TTL_MINUTES}")
        return out

    @property
    def within_budget(self) -> bool:
        return not self.refusals

    def to_dict(self) -> dict[str, Any]:
        def plain(value: Any) -> Any:
            if isinstance(value, Decimal):
                return str(value)
            if isinstance(value, tuple):
                return [plain(v) for v in value]
            return value

        out = {k: plain(v) for k, v in asdict(self).items()}
        out.update(pages=self.pages, pods=self.pods, within_budget=self.within_budget)
        out["refusals"] = self.refusals
        return out


def estimate(
    shard_pages: Sequence[int],
    *,
    pages_per_gpu_hour: Decimal,
    setup_minutes: int,
    rate_usd_per_gpu_hour: Decimal,
    budget: Budget,
    shard_ids: Sequence[int] | None = None,
    setup_concurrency: int = DEFAULT_SETUP_CONCURRENCY,
) -> Estimate:
    """GPU-hours, cost and reservations, all rounded UP (hours to 0.01 h, USD to
    the cent, TTLs to the minute). One GPU per pod. Every pod's TTL (and every
    pod's billed time in the cost) includes the bootstrap-queue wait."""
    if not shard_pages or pages_per_gpu_hour <= 0 or setup_concurrency < 1:
        raise ValueError("need >= 1 shard, pages_per_gpu_hour > 0, setup_concurrency >= 1")
    queue_minutes = (math.ceil(len(shard_pages) / setup_concurrency) - 1) * setup_minutes
    setup_hours = Decimal(setup_minutes + queue_minutes) / 60
    gpu_hours = _ceil(
        Decimal(sum(shard_pages)) / pages_per_gpu_hour + len(shard_pages) * setup_hours
    )
    ttls = tuple(
        max(1, math.ceil((Decimal(p) / pages_per_gpu_hour + setup_hours) * 60 * TTL_MARGIN))
        for p in shard_pages
    )
    reserves = tuple(_ceil(Decimal(t) / 60 * rate_usd_per_gpu_hour) for t in ttls)
    return Estimate(
        tuple(shard_ids if shard_ids is not None else range(len(shard_pages))),
        tuple(shard_pages), pages_per_gpu_hour, setup_minutes, rate_usd_per_gpu_hour,
        budget.remaining_usd, budget.per_request_max_usd, gpu_hours,
        _ceil(gpu_hours * rate_usd_per_gpu_hour), ttls, reserves, sum(reserves, Decimal(0)),
        setup_concurrency=setup_concurrency, queue_minutes=queue_minutes,
    )  # fmt: skip


def shard_plan(items: Sequence[FarmItem], pods: int) -> list[int]:
    """Pages per pod for ``items`` split and sharded exactly as ``stage_books``
    will (never more pods than units: an idle pod only adds setup cost)."""
    units = [
        {"pages": last - first + 1} for item in items for first, last in part_ranges(item.pages)
    ]
    return shard_units(units, max(1, min(pods, len(units))))


# ---------------------------------------------------------------------------
# job directory
# ---------------------------------------------------------------------------


class JobDir:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.units = root / "units"
        self.out = root / "out"
        self.pulled = root / "pulled"
        self.merged = root / "merged"
        self.ssh_config = root / "ssh_config"

    def load(self, name: str, default: Any = None) -> Any:
        path = self.root / name
        if not path.exists():
            if default is None:
                raise JobError(f"{path} is missing (run the previous step first)")
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, name: str, value: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / f".{name}.tmp"
        tmp.write_text(json.dumps(value, indent=1, default=str), encoding="utf-8")
        tmp.replace(self.root / name)

    def ssh(self, alias: str, command: str) -> list[str]:
        return ["ssh", "-F", str(self.ssh_config), alias, command]

    def scp(self, src: str, dst: str) -> list[str]:
        return ["scp", "-q", "-F", str(self.ssh_config), src, dst]


#: Staged state ``plan --force-replan`` moves aside (it would be orphaned otherwise).
STAGED_STATE = (
    "job.json", "backlog.json", "units.json", "books.json", "pods.json", "units", "out",
    "pulled", "ssh_config",
)  # fmt: skip


async def plan_job(
    items: Sequence[FarmItem],
    job: JobDir,
    pitwall: Pitwall,
    *,
    pods: int,
    gpu_type: str,
    cloud: Cloud,
    pages_per_gpu_hour: Decimal,
    setup_minutes: int = DEFAULT_SETUP_MINUTES,
    price_per_gpu_hour: Decimal | None = None,
    job_id: str | None = None,
    force_replan: bool = False,
    setup_concurrency: int = DEFAULT_SETUP_CONCURRENCY,
) -> tuple[Estimate, list[str]]:
    """Estimate against the budget; write ``job.json`` only when it fits.

    Refuses a job dir that is already staged (``units.json``) unless
    ``force_replan``, and refuses ``force_replan`` too while pods.json lists any
    pod not terminated (they would bill untracked until their TTL). The old
    state is moved aside only once the new estimate has passed the budget.
    """
    if not items:
        raise JobError("the backlog has no books to OCR")
    staged = (job.root / "units.json").exists()
    if staged:
        if not force_replan:
            raise JobError(
                f"{job.root} is already staged (units.json); planning again would orphan its "
                "units and pods. Use a new --job-dir, or --force-replan."
            )
        live = [
            f"{pod['name']} ({pod['pod_id']})"
            for pod in job.load("pods.json", [])
            if pod.get("state") != "terminated"
        ]
        if live:
            raise JobError(
                f"--force-replan refused: pods still up for this job: {', '.join(live)}. "
                "Run `runpod collect` (it terminates them once every book verifies) or "
                "terminate them with pitwall first."
            )
    try:
        catalogue: Decimal | None = await pitwall.gpu_price_usd_per_hour(gpu_type, cloud)
    except PitwallError:
        if price_per_gpu_hour is None:
            raise
        catalogue = None
    rate = price_per_gpu_hour or catalogue
    if rate is None:  # pragma: no cover - the catalogue call raised above
        raise JobError("no GPU price: pass --price-per-gpu-hour")
    est = estimate(
        shard_plan(items, pods),
        pages_per_gpu_hour=pages_per_gpu_hour,
        setup_minutes=setup_minutes,
        rate_usd_per_gpu_hour=rate,
        budget=await pitwall.budget(),
        setup_concurrency=setup_concurrency,
    )
    if not est.within_budget:
        raise BudgetExceeded("; ".join(est.refusals), est)
    warnings = _move_staged_state_aside(job) if staged else []
    job.save("backlog.json", [item.to_dict() for item in items])
    job.save(
        "job.json",
        {"job_id": job_id or time.strftime("%Y%m%d%H%M%S", time.gmtime()), "gpu_type": gpu_type,
         "cloud": cloud,
         # pods' max_cost_per_hour: the catalogue price, never an override below it
         "catalogue_rate_usd_per_gpu_hour": str(catalogue) if catalogue is not None else None,
         "estimate": est.to_dict()},
    )  # fmt: skip
    return est, warnings


def _move_staged_state_aside(job: JobDir) -> list[str]:
    """Only called once no pod of the old plan is up and the new plan fits."""
    stamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    aside = job.root / f"replanned-{stamp}"
    aside.mkdir(parents=True)
    for name in STAGED_STATE:
        if (job.root / name).exists():
            (job.root / name).replace(aside / name)
    return [f"--force-replan: previous plan and staged state moved to {aside}"]


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunOptions:
    image: str = DEFAULT_IMAGE
    ssh_key: Path = Path("~/.ssh/id_ed25519")
    max_workers: int = 8
    disk_gb: int = 100
    ssh_ready_attempts: int = 60
    poll_seconds: float = 10.0
    setup_timeout_s: float = 45 * 60
    copy_timeout_s: float = 60 * 60
    command_timeout_s: float = 120
    #: Pods bootstrapped at once; ``None``: the value the job was planned with
    #: (its TTLs include the queue wait for that value).
    setup_concurrency: int | None = None


async def _call(runner: Runner, argv: list[str], timeout: float) -> CommandResult:
    try:
        return await asyncio.wait_for(runner(argv), timeout)
    except TimeoutError:
        return CommandResult(124, "", f"timed out after {timeout:.0f}s")


def _tail(result: CommandResult) -> str:
    return ((result.stderr or result.stdout).strip().splitlines() or [""])[-1][:300]


def write_ssh_config(job: JobDir, pods: Sequence[dict[str, Any]], key: Path) -> None:
    blocks = [
        f"Host {pod['alias']}\n  HostName {pod['ssh_host']}\n  Port {pod['ssh_port']}\n"
        f"  User root\n  IdentityFile {key}\n  IdentitiesOnly yes\n  BatchMode yes\n"
        f"  StrictHostKeyChecking accept-new\n  UserKnownHostsFile {job.root / 'known_hosts'}\n"
        f"  ServerAliveInterval 30\n"
        for pod in pods
        if pod.get("ssh_host") and pod.get("state") != "terminated"
    ]
    job.ssh_config.write_text("\n".join(blocks), encoding="utf-8")


async def _stage(job: JobDir, garage: CacheStore, bucket: str, pods: int) -> list[dict[str, Any]]:
    if (job.root / "units.json").exists():
        return []
    items = [FarmItem(**row) for row in job.load("backlog.json")]
    staged = await stage_books(items, garage, bucket=bucket, units_dir=job.units, pods=pods)
    job.save("books.json", staged.books)
    job.save("units.json", staged.units)
    return [{"staged": len(staged.books), "units": len(staged.units), "bad": staged.bad,
             "shard_pages": staged.shard_pages}]  # fmt: skip


def _pending_units(job: JobDir, units: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Units without an ok result pulled into ``out/`` yet."""
    return [u for u in units if unit_state(job.out, str(u["unit"]))[0] != "ok"]


#: Pod statuses that will never OCR again. An exited/dead pod can still bill
#: (its disk), so it is terminated through pitwall, not just forgotten.
DEAD_POD_STATUSES = ("terminated", "exited", "dead")


def _concurrency(meta: dict[str, Any], opts: RunOptions) -> int:
    if opts.setup_concurrency is not None:
        return max(1, opts.setup_concurrency)
    return int(meta["estimate"].get("setup_concurrency", DEFAULT_SETUP_CONCURRENCY))


async def _mark_lost_pods(pods: list[dict[str, Any]], pitwall: Pitwall) -> None:
    """A pod pitwall no longer knows (TTL expired, terminated elsewhere) or that
    has exited is terminated (via pitwall when it still exists) and marked lost."""
    for pod in pods:
        if pod["state"] == "terminated":
            continue
        try:
            info = await pitwall.get_pod(pod["pod_id"])
        except PitwallError as exc:
            if exc.code != "resource_not_found":
                raise
            pod.update(state="terminated", lost=True)
            continue
        if info.status in DEAD_POD_STATUSES:
            if info.status != "terminated":
                # idempotent: the same key as every other terminate of this pod
                await pitwall.terminate_pod(pod["pod_id"], f"{pod['name']}-terminate")
            pod.update(state="terminated", lost=True)


async def _create_pods(
    job: JobDir,
    meta: dict[str, Any],
    pitwall: Pitwall,
    opts: RunOptions,
    units: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create a pod for every shard that has pending units and no live pod:
    first run, or a pod terminated / lost during the job (recreated with a new
    name and idempotency key). Budget and per-request cap are checked on the
    reservations of exactly the pods about to be created."""
    pods: list[dict[str, Any]] = job.load("pods.json", [])
    await _mark_lost_pods(pods, pitwall)
    job.save("pods.json", pods)
    live = {pod["shard"] for pod in pods if pod["state"] != "terminated"}
    pending: dict[int, int] = {}
    for unit in _pending_units(job, units):
        k = int(unit["shard"])
        pending[k] = pending.get(k, 0) + int(unit["pages"])
    need = sorted(k for k in pending if k not in live)
    if not need:
        return pods
    planned = meta["estimate"]
    est = estimate(
        [pending[k] for k in need],
        pages_per_gpu_hour=Decimal(planned["pages_per_gpu_hour"]),
        setup_minutes=int(planned["setup_minutes"]),
        rate_usd_per_gpu_hour=Decimal(planned["rate_usd_per_gpu_hour"]),
        budget=await pitwall.budget(),
        shard_ids=need,
        setup_concurrency=_concurrency(meta, opts),
    )
    if not est.within_budget:
        raise BudgetExceeded("; ".join(est.refusals), est)
    try:
        public_key = Path(f"{opts.ssh_key.expanduser()}.pub").read_text().strip()
    except OSError as exc:
        raise JobError(f"cannot read the public key for --ssh-key: {exc}") from exc
    ceiling = meta.get("catalogue_rate_usd_per_gpu_hour")
    created: list[dict[str, Any]] = []
    for k, ttl in zip(need, est.ttl_minutes, strict=True):
        base = f"vorag-ocr-{meta['job_id']}-{k}"
        attempt = sum(1 for pod in pods if pod["shard"] == k)
        name = base if attempt == 0 else f"{base}-r{attempt}"
        request = {
            "name": name, "image": opts.image, "gpu_type_ids": [meta["gpu_type"]],
            "gpu_count": 1, "cloud": meta["cloud"].upper(), "disk_gb": opts.disk_gb,
            "ports": ["22/tcp"], "env": {"PUBLIC_KEY": public_key}, "ttl_minutes": ttl,
        }  # fmt: skip
        if ceiling is not None:
            request["max_cost_per_hour"] = ceiling
        try:
            pod_id = await _create_or_find(pitwall, request, f"{name}-create")
        except PitwallError as exc:
            # Never leave this run's pods billing unbootstrapped until their
            # TTL because a later create failed.
            cleanup = await _terminate_created(job, pods, created, pitwall)
            raise JobError(f"pod create failed for shard {k} ({name}): {exc}; {cleanup}") from exc
        entry = {"shard": k, "pod_id": pod_id, "name": name, "alias": name,
                 "state": "created", "ttl_minutes": ttl}  # fmt: skip
        pods.append(entry)
        created.append(entry)
        job.save("pods.json", pods)  # record each pod the moment it exists
    return pods


async def _create_or_find(pitwall: Pitwall, request: dict[str, Any], key: str) -> str:
    """``pitwall.create_pod``; when it fails (a client-side timeout, a lost
    reply) the pod may still have been made, so look it up by its unique name
    before giving up. Re-raises the create's error when no such pod exists
    (or the lookup fails too)."""
    name = str(request["name"])
    try:
        return await pitwall.create_pod(request, key)
    except PitwallError as create_error:
        try:
            found = await pitwall.find_pod_by_name(name)
        except PitwallError as lookup_error:
            if create_error.code == "timeout":
                # Nobody knows whether RunPod made the pod: say so, loudly.
                raise PitwallError(
                    f"{create_error}; the lookup by name failed too ({lookup_error}): "
                    f"pod {name!r} may exist unrecorded and is billing until its TTL "
                    f"({request.get('ttl_minutes')} min). Check `pitwall runpod pods list "
                    f"--json` for {name!r} and terminate it through pitwall if present",
                    create_error.code,
                ) from create_error
            raise create_error from None
        if found is None:
            raise
        return found


async def _terminate_created(
    job: JobDir, pods: list[dict[str, Any]], created: list[dict[str, Any]], pitwall: Pitwall
) -> str:
    """Terminate every pod in ``created``, each independently; returns a
    summary naming the terminated pods and any still billing."""
    done: list[str] = []
    failed: list[str] = []
    for pod in created:
        try:
            await pitwall.terminate_pod(pod["pod_id"], f"{pod['name']}-terminate")
            pod.update(state="terminated", lost=True)
            done.append(pod["pod_id"])
        except PitwallError as exc:
            failed.append(f"{pod['pod_id']} ({exc})")
    job.save("pods.json", pods)
    summary = f"terminated {', '.join(done)}" if done else "no pods created this run"
    if failed:
        summary += f"; TERMINATE FAILED, may still bill: {'; '.join(failed)}"
    return summary


async def _wait_for_ssh(
    job: JobDir,
    all_pods: list[dict[str, Any]],
    pods: list[dict[str, Any]],
    pitwall: Pitwall,
    opts: RunOptions,
    sleep: Sleep,
) -> dict[str, str]:
    """Poll pitwall until each new pod has an ssh address. A pod whose poll
    raises, or that never gets an address, is terminated now (it would bill
    unbootstrapped until its TTL) and the others carry on. Returns
    ``{pod_id: problem}`` for those pods."""
    failed: dict[str, str] = {}
    for pod in pods:
        problem = ""
        try:
            for _ in range(opts.ssh_ready_attempts):
                if pod.get("ssh_host") or pod["state"] == "terminated":
                    break
                info = await pitwall.get_pod(pod["pod_id"])
                if info.ssh_ready:
                    pod["ssh_host"], pod["ssh_port"] = info.ssh_host, info.ssh_port
                    break
                await sleep(opts.poll_seconds)
            else:
                problem = "no ssh address from pitwall"
        except PitwallError as exc:
            problem = f"ssh wait: {exc}"[:300]
        if problem:
            failed[pod["pod_id"]] = await _terminate_unready(job, all_pods, pod, pitwall, problem)
    return failed


async def _terminate_unready(
    job: JobDir, pods: list[dict[str, Any]], pod: dict[str, Any], pitwall: Pitwall, problem: str
) -> str:
    try:
        await pitwall.terminate_pod(pod["pod_id"], f"{pod['name']}-terminate")
        pod.update(state="terminated", lost=True)
    except PitwallError as exc:
        # still maybe billing: a re-run (or _bring_up) tries the terminate again
        pod["state"] = "no_ssh"
        problem += f"; TERMINATE FAILED, pod may still bill: {exc}"[:300]
    job.save("pods.json", pods)
    return problem


async def _setup_pod(
    job: JobDir, pod: dict[str, Any], units: list[dict[str, Any]], runner: Runner,
    opts: RunOptions, sleep: Sleep,
) -> str:  # fmt: skip
    """Copy scripts + units, bootstrap; ``""`` on success, else the failure."""
    alias = pod["alias"]
    for _ in range(opts.ssh_ready_attempts):
        if (await _call(runner, job.ssh(alias, "true"), opts.command_timeout_s)).returncode == 0:
            break
        await sleep(opts.poll_seconds)
    else:
        return "ssh never answered"
    files = [str(RUNPOD_DIR / name) for name in SHELL_FILES] + [str(WORKER_DIR / WORKER_SCRIPT)]
    shard_tar = job.root / f"shard{pod['shard']}.tar"
    with tarfile.open(shard_tar, "w") as tar:
        for unit in units:
            tar.add(job.units / f"{unit['unit']}.pdf", arcname=f"units/{unit['unit']}.pdf")
    steps = [
        ("copy", job.scp(files[0], f"{alias}:{REMOTE_DIR}/"), opts.copy_timeout_s),
        ("copy", job.scp(files[1], f"{alias}:{REMOTE_DIR}/"), opts.copy_timeout_s),
        ("copy", job.scp(files[2], f"{alias}:{REMOTE_DIR}/"), opts.copy_timeout_s),
        ("units", job.scp(str(shard_tar), f"{alias}:{REMOTE_DIR}/units.tar"), opts.copy_timeout_s),
        ("unpack", job.ssh(alias, f"cd {REMOTE_DIR} && tar xf units.tar && rm units.tar"),
         opts.command_timeout_s),
        ("bootstrap", job.ssh(alias, f"cd {REMOTE_DIR} && bash bootstrap.sh"),
         opts.setup_timeout_s),
    ]  # fmt: skip
    for what, argv, timeout in steps:
        result = await _call(runner, argv, timeout)
        if result.returncode != 0:
            return f"{what} rc={result.returncode}: {_tail(result)}"
    shard_tar.unlink(missing_ok=True)
    return ""


def _start_command(opts: RunOptions) -> str:
    return (
        f"cd {REMOTE_DIR} && MAX_WORKERS={opts.max_workers} setsid nohup bash "
        "run_units.sh run > run.out 2>&1 < /dev/null &"
    )


_STATUS_FIELD = re.compile(r"\b(ok|failed|pending|running)=(\w+)")


async def _restart_if_unfinished(
    job: JobDir, pod: dict[str, Any], runner: Runner, opts: RunOptions
) -> dict[str, Any]:
    """A started pod whose pool is not running but still has failed or pending
    units gets its pool again (it skips ok units): the retry path for
    OOM-killed or crashed units and for a pool that died."""
    line: dict[str, Any] = {"pod": pod["name"], "pod_id": pod["pod_id"], "state": "started"}
    status = await _call(
        runner, job.ssh(pod["alias"], f"cd {REMOTE_DIR} && bash run_units.sh status"),
        opts.command_timeout_s,
    )  # fmt: skip
    fields = dict(_STATUS_FIELD.findall(status.stdout))
    line["status"] = status.stdout.strip() or None
    unfinished = fields.get("failed", "0") != "0" or fields.get("pending", "0") != "0"
    if status.returncode == 0 and fields.get("running") == "no" and unfinished:
        result = await _call(runner, job.ssh(pod["alias"], _start_command(opts)),
                             opts.command_timeout_s)  # fmt: skip
        line["restarted"] = result.returncode == 0
    return line


async def _bring_up(
    job: JobDir,
    pods: list[dict[str, Any]],
    pod: dict[str, Any],
    units: list[dict[str, Any]],
    pitwall: Pitwall,
    runner: Runner,
    opts: RunOptions,
    sleep: Sleep,
    gate: asyncio.Semaphore,
) -> dict[str, Any]:
    """Setup + start one new pod; a pod that cannot OCR is terminated at once."""
    async with gate:
        problem = "no ssh address from pitwall" if pod["state"] == "no_ssh" else ""
        problem = problem or await _setup_pod(job, pod, units, runner, opts, sleep)
        if not problem:
            result = await _call(
                runner, job.ssh(pod["alias"], _start_command(opts)), opts.command_timeout_s
            )
            problem = "" if result.returncode == 0 else f"start rc={result.returncode}"
    if problem:
        # A pod that cannot OCR (no CUDA, no ssh) only bills: terminate it now.
        await pitwall.terminate_pod(pod["pod_id"], f"{pod['name']}-terminate")
        pod["state"] = "terminated"
    else:
        pod["state"] = "started"
    job.save("pods.json", pods)
    return {"pod": pod["name"], "pod_id": pod["pod_id"], "state": pod["state"],
            "ttl_minutes": pod.get("ttl_minutes"), "units": len(units),
            "problem": problem or None}  # fmt: skip


async def run_job(
    job: JobDir,
    pitwall: Pitwall,
    garage: CacheStore,
    runner: Runner,
    *,
    bucket: str,
    opts: RunOptions = RunOptions(),  # noqa: B008 - frozen dataclass
    sleep: Sleep = asyncio.sleep,
) -> list[dict[str, Any]]:
    """Stage, create pods, bootstrap the new ones concurrently and start their
    pools. Re-runnable: staging is kept, a live started pod is only restarted
    if its pool ended unfinished, and a shard whose pod is gone gets a new pod."""
    meta = job.load("job.json")
    # Reach pitwall before staging: an unreachable pitwall
    # fails the run before any Garage read or unit is written.
    await pitwall.budget()
    lines = await _stage(job, garage, bucket, int(meta["estimate"]["pods"]))
    units: list[dict[str, Any]] = job.load("units.json")
    pods = await _create_pods(job, meta, pitwall, opts, units)
    active = [pod for pod in pods if pod["state"] != "terminated"]
    unready = await _wait_for_ssh(job, pods, active, pitwall, opts, sleep)
    job.save("pods.json", pods)
    write_ssh_config(job, pods, opts.ssh_key)
    pending = _pending_units(job, units)
    gate = asyncio.Semaphore(_concurrency(meta, opts))
    jobs: list[Awaitable[dict[str, Any]]] = []
    targets: list[dict[str, Any]] = []
    failed_shards: list[int] = []
    for pod in pods:
        if pod["state"] == "terminated":
            line = {"pod": pod["name"], "state": "terminated", "lost": pod.get("lost")}
            if pod["pod_id"] in unready:
                line.update(pod_id=pod["pod_id"], problem=unready[pod["pod_id"]])
                failed_shards.append(int(pod["shard"]))
            lines.append(line)
            continue
        if pod["state"] == "started":
            jobs.append(_restart_if_unfinished(job, pod, runner, opts))
        else:
            shard = [u for u in pending if int(u["shard"]) == pod["shard"]]
            jobs.append(_bring_up(job, pods, pod, shard, pitwall, runner, opts, sleep, gate))
        targets.append(pod)
    # One pod's exception must neither cancel the other bootstraps nor leave its
    # own pod billing: collect every outcome, then terminate the pods that raised.
    outcomes = await asyncio.gather(*jobs, return_exceptions=True)
    for pod, outcome in zip(targets, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            if not isinstance(outcome, Exception):
                raise outcome  # cancellation / KeyboardInterrupt
            lines.append(await _terminate_after_error(job, pods, pod, pitwall, outcome))
            failed_shards.append(int(pod["shard"]))
        else:
            lines.append(outcome)
            if outcome.get("problem"):
                failed_shards.append(int(pod["shard"]))
    write_ssh_config(job, pods, opts.ssh_key)
    if failed_shards:
        lines.append({"failed_shards": sorted(set(failed_shards)),
                      "hint": "fix the cause and re-run `runpod run`: a shard whose pod was "
                              "terminated gets a new pod"})  # fmt: skip
    return lines


async def _terminate_after_error(
    job: JobDir, pods: list[dict[str, Any]], pod: dict[str, Any], pitwall: Pitwall,
    error: Exception,
) -> dict[str, Any]:  # fmt: skip
    problem = f"{type(error).__name__}: {error}"[:300]
    if pod["state"] == "started":
        # a live pool whose status/restart call failed: leave it OCRing
        return {"pod": pod["name"], "pod_id": pod["pod_id"], "state": "started",
                "problem": problem}  # fmt: skip
    try:
        await pitwall.terminate_pod(pod["pod_id"], f"{pod['name']}-terminate")
        pod["state"] = "terminated"
    except PitwallError as exc:
        problem += f"; TERMINATE FAILED, pod may still bill: {exc}"[:300]
    job.save("pods.json", pods)
    return {"pod": pod["name"], "pod_id": pod["pod_id"], "state": pod["state"],
            "problem": problem}  # fmt: skip


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------


def extract_results(tar_path: Path, out_dir: Path) -> int:
    """Copy ``units/<name>.json|.note`` members into ``out_dir``; nothing else
    (no links, no paths outside ``out_dir``). Returns the files written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            parts = member.name.split("/")
            if (
                not member.isfile()
                or len(parts) != 2
                or parts[0] != "units"
                or not parts[1].endswith((".json", ".note"))
                or parts[1].startswith(".")
            ):
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            (out_dir / parts[1]).write_bytes(source.read())
            written += 1
    return written


async def _pull(job: JobDir, pod: dict[str, Any], runner: Runner, opts: RunOptions) -> str:
    alias = pod["alias"]
    pack = await _call(
        runner, job.ssh(alias, f"cd {REMOTE_DIR} && bash run_units.sh pack"), opts.setup_timeout_s
    )
    if pack.returncode != 0:
        return f"pack rc={pack.returncode}: {_tail(pack)}"
    pod["status"] = pack.stdout.strip().splitlines()[-1] if pack.stdout.strip() else None
    job.pulled.mkdir(parents=True, exist_ok=True)
    local = job.pulled / f"{pod['shard']}.tar"
    copy = await _call(
        runner, job.scp(f"{alias}:{REMOTE_DIR}/results.tar", str(local)), opts.copy_timeout_s
    )
    if copy.returncode != 0:
        return f"scp rc={copy.returncode}: {_tail(copy)}"
    try:
        extract_results(local, job.out)
    except tarfile.TarError as exc:
        return f"bad results.tar: {exc}"
    return ""


async def collect_job(
    job: JobDir,
    pitwall: Pitwall,
    garage: CacheStore,
    runner: Runner,
    *,
    bucket: str,
    overwrite: bool = False,
    opts: RunOptions = RunOptions(),  # noqa: B008
) -> tuple[list[dict[str, Any]], bool]:
    """Pull, merge, upload, verify; terminate the pods only if every book verified.

    Returns ``(lines, ok)``: ``ok`` only when every book verified AND every pod
    terminated. When not complete the pods stay up and the lines name every
    failed or pending unit; a failed terminate is named in
    ``terminate_failed`` (the others are still tried) and re-running collect
    tries it again.
    """
    meta = job.load("job.json")
    books: list[dict[str, Any]] = job.load("books.json")
    units: list[dict[str, Any]] = job.load("units.json")
    pods: list[dict[str, Any]] = job.load("pods.json", [])
    lines: list[dict[str, Any]] = []
    pull_errors = False
    for pod in pods:
        if pod["state"] != "started":
            continue
        problem = await _pull(job, pod, runner, opts)
        pull_errors |= bool(problem)
        lines.append({"pod": pod["name"], "pulled": not problem, "status": pod.get("status"),
                      "problem": problem or None})  # fmt: skip
    outcomes = await merge_and_upload(
        books, units, job.out, job.merged, garage,
        bucket=bucket, job_id=str(meta["job_id"]), overwrite=overwrite,
    )  # fmt: skip
    checks = {r["sha256"]: r for r in await verify_books(books, garage, bucket=bucket)}
    for outcome in outcomes:
        check = checks[outcome["sha256"]]
        outcome["verified"] = check["ok"]
        outcome["verify_reason"] = check["reason"] or None
        lines.append(outcome)
    complete = not pull_errors and all(o["verified"] for o in outcomes)
    terminate_failed: dict[str, str] = {}
    if complete:
        # Each terminate on its own: one failure must not
        # leave the pods after it billing untried.
        for pod in pods:
            if pod["state"] == "terminated":
                continue
            try:
                await pitwall.terminate_pod(pod["pod_id"], f"{pod['name']}-terminate")
                pod["state"] = "terminated"
            except PitwallError as exc:
                terminate_failed[pod["pod_id"]] = str(exc)[:300]
            job.save("pods.json", pods)
        if not terminate_failed:
            job.ssh_config.unlink(missing_ok=True)
    failed = {u: n for o in outcomes for u, n in o["units_failed"].items()}
    pending = [u for o in outcomes for u in o["units_pending"]]
    if not complete:
        pods_state = "left running"
    elif terminate_failed:
        pods_state = "TERMINATE FAILED for some pods, pod may still bill: re-run collect"
    else:
        pods_state = "terminated"
    summary: dict[str, Any] = {
        "complete": complete, "books": len(books),
        "verified": sum(1 for o in outcomes if o["verified"]),
        "units_failed": failed, "units_pending": pending, "pods": pods_state,
    }  # fmt: skip
    if terminate_failed:
        summary["terminate_failed"] = terminate_failed
    lines.append(summary)
    return lines, complete and not terminate_failed


__all__ = [
    "BudgetExceeded",
    "Estimate",
    "JobDir",
    "JobError",
    "PitwallError",
    "RunOptions",
    "collect_job",
    "estimate",
    "extract_results",
    "plan_job",
    "run_job",
]

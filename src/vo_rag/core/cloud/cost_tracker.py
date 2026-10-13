"""Per-job cloud spend tracker + budget gate."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime, timezone

import asyncpg  # type: ignore[import-untyped]

from vo_rag.core.models import VoBaseModel

log = logging.getLogger("vo_rag.cloud.cost_tracker")

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.50, 0.75, 0.90)


class SpendReport(VoBaseModel):
    month: str  # YYYY-MM (UTC)
    total_spend_usd: float
    budget_usd: float
    budget_remaining_usd: float
    budget_pct_used: float
    by_provider: dict[str, float]
    by_workload: dict[str, float]
    by_gpu_type: dict[str, float]
    jobs_this_month: int


class CloudCostTracker:
    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        monthly_budget_usd: float,
        per_job_max_usd: float,
        alert_thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    ) -> None:
        if monthly_budget_usd <= 0 or per_job_max_usd <= 0:
            raise ValueError("budgets must be positive")
        self.pool = pool
        self.monthly = float(monthly_budget_usd)
        self.per_job_max = float(per_job_max_usd)
        self.thresholds = tuple(sorted(float(t) for t in alert_thresholds))

    @staticmethod
    def estimate_cost(rate_per_hour: float, hours: float, gpu_count: int = 1) -> float:
        if rate_per_hour < 0 or hours < 0 or gpu_count < 0:
            raise ValueError("non-negative values required")
        return rate_per_hour * hours * gpu_count

    @staticmethod
    def _month_key(now: datetime | None = None) -> str:
        n = now or datetime.now(timezone.utc)  # noqa: UP017
        return n.strftime("%Y-%m")

    async def _current_month_spend(self) -> float:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT COALESCE(SUM(cost_usd), 0) AS s
                   FROM cloud_cost
                   WHERE date_trunc('month', started_at AT TIME ZONE 'UTC')
                         = date_trunc('month', now() AT TIME ZONE 'UTC')"""
            )
            return float(row["s"])

    async def can_launch(self, estimated_cost_usd: float) -> bool:
        if estimated_cost_usd <= 0:
            return False
        if estimated_cost_usd > self.per_job_max:
            log.warning("per-job cap exceeded: %.2f > %.2f", estimated_cost_usd, self.per_job_max)
            return False
        spend = await self._current_month_spend()
        if spend + estimated_cost_usd > self.monthly:
            log.warning(
                "monthly budget would exceed: %.2f + %.2f > %.2f",
                spend,
                estimated_cost_usd,
                self.monthly,
            )
            return False
        return True

    async def record_launch(
        self,
        *,
        job_id: str,
        provider: str,
        gpu_type: str,
        workload: str,
        reserved_cost_usd: float = 0.0,
        started_at: datetime | None = None,
    ) -> None:
        started = started_at or datetime.now(timezone.utc)  # noqa: UP017
        if reserved_cost_usd < 0:
            raise ValueError("reserved cost must be non-negative")
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO cloud_cost (job_id, provider, gpu_type, workload,
                                           started_at, status, cost_usd)
                   VALUES ($1, $2, $3, $4, $5, 'launching', $6)
                   ON CONFLICT (job_id) DO NOTHING""",
                job_id,
                provider,
                gpu_type,
                workload,
                started,
                reserved_cost_usd,
            )

    # F-018: workload-stable advisory-lock key. All launches contend for the
    # same lock — concurrent burst requests serialize through the gate so
    # the budget check and the cloud_cost INSERT are atomic. Different
    # workloads don't get distinct locks here on purpose: the budget is
    # whole-account, not per-workload.
    _BUDGET_GATE_LOCK_KEY = 0x56_4F_52_41_47_42_55_44  # b"VORAGBUD" as int64

    async def try_launch(
        self,
        *,
        estimated_cost_usd: float,
        job_id: str,
        provider: str,
        gpu_type: str,
        workload: str,
        reserved_cost_usd: float = 0.0,
        started_at: datetime | None = None,
    ) -> bool:
        """F-018: atomic can_launch + record_launch under a Postgres
        advisory lock. Returns True if the launch was admitted (row written),
        False if the budget would be exceeded.

        Without this, two concurrent launches each pass the can_launch read,
        then each insert, exceeding the monthly cap.
        """
        if estimated_cost_usd <= 0:
            return False
        if estimated_cost_usd > self.per_job_max:
            log.warning(
                "per-job cap exceeded: %.2f > %.2f",
                estimated_cost_usd,
                self.per_job_max,
            )
            return False
        if reserved_cost_usd < 0:
            raise ValueError("reserved cost must be non-negative")

        started = started_at or datetime.now(timezone.utc)  # noqa: UP017
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock($1)", self._BUDGET_GATE_LOCK_KEY)
            row = await conn.fetchrow(
                """SELECT COALESCE(SUM(cost_usd), 0) AS s
                   FROM cloud_cost
                   WHERE date_trunc('month', started_at AT TIME ZONE 'UTC')
                         = date_trunc('month', now() AT TIME ZONE 'UTC')"""
            )
            spend = float(row["s"])
            if spend + estimated_cost_usd > self.monthly:
                log.warning(
                    "monthly budget would exceed under advisory lock: %.2f + %.2f > %.2f",
                    spend,
                    estimated_cost_usd,
                    self.monthly,
                )
                return False
            await conn.execute(
                """INSERT INTO cloud_cost (job_id, provider, gpu_type, workload,
                                           started_at, status, cost_usd)
                   VALUES ($1, $2, $3, $4, $5, 'launching', $6)
                   ON CONFLICT (job_id) DO NOTHING""",
                job_id,
                provider,
                gpu_type,
                workload,
                started,
                reserved_cost_usd,
            )
            return True

    async def record_completion(
        self,
        *,
        job_id: str,
        cost_usd: float,
        duration_s: int,
        status: str = "completed",
    ) -> None:
        if status not in ("completed", "failed", "killed"):
            raise ValueError(f"invalid status: {status}")
        before = await self._current_month_spend()
        async with self.pool.acquire() as conn:
            await conn.execute(
                """UPDATE cloud_cost
                   SET cost_usd = $2, duration_s = $3, status = $4,
                       completed_at = now()
                   WHERE job_id = $1""",
                job_id,
                cost_usd,
                duration_s,
                status,
            )
        after = before + max(cost_usd - 0, 0)
        self._maybe_alert(before, after)

    def _maybe_alert(self, before: float, after: float) -> None:
        for t in self.thresholds:
            cap = self.monthly * t
            if before < cap <= after:
                log.warning(
                    "cloud-spend threshold %d%% crossed: %.2f / %.2f",
                    int(t * 100),
                    after,
                    self.monthly,
                )

    async def get_spend_report(self) -> SpendReport:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT provider, workload, gpu_type, cost_usd
                   FROM cloud_cost
                   WHERE date_trunc('month', started_at AT TIME ZONE 'UTC')
                         = date_trunc('month', now() AT TIME ZONE 'UTC')"""
            )
        total = sum(float(r["cost_usd"]) for r in rows)
        by_provider: dict[str, float] = {}
        by_workload: dict[str, float] = {}
        by_gpu: dict[str, float] = {}
        for r in rows:
            by_provider[r["provider"]] = by_provider.get(r["provider"], 0.0) + float(r["cost_usd"])
            by_workload[r["workload"]] = by_workload.get(r["workload"], 0.0) + float(r["cost_usd"])
            by_gpu[r["gpu_type"]] = by_gpu.get(r["gpu_type"], 0.0) + float(r["cost_usd"])
        return SpendReport(
            month=self._month_key(),
            total_spend_usd=total,
            budget_usd=self.monthly,
            budget_remaining_usd=max(self.monthly - total, 0.0),
            budget_pct_used=(total / self.monthly * 100.0) if self.monthly else 0.0,
            by_provider=by_provider,
            by_workload=by_workload,
            by_gpu_type=by_gpu,
            jobs_this_month=len(rows),
        )

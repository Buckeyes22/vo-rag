"""Cloud cost Prometheus exporter.

Exports vo-rag cloud spend as Prometheus gauges.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any

import asyncpg  # type: ignore[import-untyped]
from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest
from starlette.responses import Response

from vo_rag.core.cloud.cost_tracker import CloudCostTracker
from vo_rag.core.cloud.tailscale_auth import TailscaleProvisioner
from vo_rag.core.config import get_settings

log = logging.getLogger("vo_rag.cloud_cost_exporter")

# --------------------------------------------------------------------------- #
# Prometheus gauges
# --------------------------------------------------------------------------- #
cloud_spend_month_usd = Gauge(
    "vo_rag_cloud_spend_month_usd",
    "Cumulative monthly cloud spend (USD)",
    ["workload", "provider", "gpu_type"],
)
cloud_active_workers = Gauge(
    "vo_rag_cloud_active_workers",
    "Active cloud workers (Tailscale devices with tag:vorag-cloud-worker)",
    ["provider", "gpu_type"],
)
cloud_budget_pct = Gauge(
    "vo_rag_cloud_budget_pct",
    "% of monthly budget consumed",
)
cloud_budget_usd = Gauge(
    "vo_rag_cloud_budget_usd",
    "Monthly budget (USD)",
)
cloud_budget_remaining_usd = Gauge(
    "vo_rag_cloud_budget_remaining_usd",
    "Remaining monthly budget (USD)",
)
cloud_jobs_this_month = Gauge(
    "vo_rag_cloud_jobs_this_month",
    "Number of cloud jobs launched this month",
)
cloud_kill_switch_triggers_total = Gauge(
    "vo_rag_cloud_kill_switch_triggers_7d",
    "Kill-switch activations in the last 7 days",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sanitize_label(value: str) -> str:
    """Prometheus label values must match [a-zA-Z0-9_]*."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(value))


def _redact(text: str) -> str:
    """Remove common secret tokens from log messages."""
    for key in (
        "RUNPOD_API_KEY",
        "R2_ACCESS_KEY",
        "R2_SECRET_KEY",
        "GARAGE_ACCESS_KEY",
        "GARAGE_SECRET_KEY",
        "DB_PASS",
        "POSTGRES_PASSWORD",
    ):
        text = text.replace(key, "[REDACTED]")
    return text


# --------------------------------------------------------------------------- #
# Refresh logic
# --------------------------------------------------------------------------- #
async def _refresh(app: FastAPI) -> None:
    tracker: CloudCostTracker = app.state.tracker
    pool: asyncpg.Pool = app.state.pool

    # ---- spend from tracker (no duplicated monthly SQL) ----
    report = await tracker.get_spend_report()
    cloud_budget_usd.set(report.budget_usd)
    cloud_budget_pct.set(report.budget_pct_used)
    cloud_budget_remaining_usd.set(report.budget_remaining_usd)
    cloud_jobs_this_month.set(report.jobs_this_month)

    cloud_spend_month_usd.clear()
    for workload, cost in report.by_workload.items():
        cloud_spend_month_usd.labels(
            workload=_sanitize_label(workload),
            provider="",
            gpu_type="",
        ).set(cost)
    for provider, cost in report.by_provider.items():
        cloud_spend_month_usd.labels(
            workload="",
            provider=_sanitize_label(provider),
            gpu_type="",
        ).set(cost)
    for gpu, cost in report.by_gpu_type.items():
        cloud_spend_month_usd.labels(
            workload="",
            provider="",
            gpu_type=_sanitize_label(gpu),
        ).set(cost)

    # ---- kill-switch from DB (pool required) ----
    kills = 0
    if pool is not None:
        async with pool.acquire() as conn:
            kills = await conn.fetchval(
                "SELECT COUNT(*) FROM cloud_kill_log WHERE triggered_at > now() - interval '7 days'"
            )
    cloud_kill_switch_triggers_total.set(int(kills or 0))

    # ---- active fleet from Tailscale ----
    cloud_active_workers.clear()
    ts = getattr(app.state, "ts", None)
    if ts is None:
        cloud_active_workers.labels(
            provider="not_configured",
            gpu_type="not_configured",
        ).set(0)
    else:
        try:
            devices: list[dict[str, Any]] = await ts.list_devices()
        except Exception as exc:
            log.warning("tailscale list_devices failed: %s", _redact(str(exc)))
            devices = []
        agg: dict[tuple[str, str], int] = {}
        for device in devices:
            host = str(device.get("name", "") or device.get("hostname", ""))
            provider = "runpod" if "-sft-" not in host else "vast"
            gpu = host.split("-")[1] if "-" in host else "unknown"
            agg[(provider, gpu)] = agg.get((provider, gpu), 0) + 1
        for (provider, gpu), count in agg.items():
            cloud_active_workers.labels(
                provider=_sanitize_label(provider),
                gpu_type=_sanitize_label(gpu),
            ).set(count)


async def _poll_loop(app: FastAPI) -> None:
    while True:
        try:
            await _refresh(app)
        except Exception as exc:
            log.exception("refresh failed: %s", _redact(str(exc)))
        await asyncio.sleep(app.state.poll_interval_s)


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(
    settings: Any = None,
    pool: asyncpg.Pool | None = None,
    tracker: CloudCostTracker | None = None,
    tailscale: Any = None,
    poll_interval_s: float = 60.0,
) -> FastAPI:
    """Build a FastAPI exporter app with optional dependency injection.

    When *pool*, *tracker*, or *tailscale* are omitted the lifespan creates
    real connections from *settings* (or the default ``get_settings()``).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        cfg = settings if settings is not None else get_settings()
        app.state.settings = cfg
        app.state.poll_interval_s = poll_interval_s

        # Pool
        if pool is not None:
            app.state.pool = pool
            app.state._owns_pool = False
        else:
            postgres_url = getattr(getattr(cfg, "postgres", None), "url", None) or cfg.database.url
            app.state.pool = await asyncpg.create_pool(postgres_url, min_size=1, max_size=3)
            app.state._owns_pool = True

        # Tracker
        if tracker is not None:
            app.state.tracker = tracker
        else:
            app.state.tracker = CloudCostTracker(
                pool=app.state.pool,
                monthly_budget_usd=float(cfg.cloud.monthly_budget_usd),
                per_job_max_usd=float(cfg.cloud.per_job_max_usd),
            )

        # Tailscale
        if tailscale is not None:
            app.state.ts = tailscale
            app.state._owns_ts = False
        else:
            ts_cfg = cfg.cloud.tailscale
            if ts_cfg.oauth_client_id and ts_cfg.oauth_client_secret and ts_cfg.tailnet:
                app.state.ts = TailscaleProvisioner(
                    ts_cfg.oauth_client_id,
                    ts_cfg.oauth_client_secret,
                    ts_cfg.tailnet,
                )
                app.state._owns_ts = True
            else:
                app.state.ts = None
                app.state._owns_ts = False

        app.state._poll_task = asyncio.create_task(_poll_loop(app))
        try:
            yield
        finally:
            app.state._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await app.state._poll_task
            if app.state._owns_pool:
                await app.state.pool.close()
            if app.state._owns_ts and app.state.ts is not None:
                await app.state.ts.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    return app


# --------------------------------------------------------------------------- #
# Production export
# --------------------------------------------------------------------------- #
app = create_app()


def main() -> int:
    import uvicorn

    port = int(os.environ.get("CLOUD_COST_EXPORTER_PORT", "9109"))
    uvicorn.run(app, host="0.0.0.0", port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

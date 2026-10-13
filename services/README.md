# services

The cloud-cost Prometheus exporter is the only service in this directory.

- `services/cloud_cost_exporter/` — Prometheus exporter for cloud
  spend, monthly budget, active workers (Tailscale), and
  kill-switch telemetry. Registered in the worker roster as
  `cloud_cost_exporter` (workload `misc`, `local`, CPU-only) and
  exposes the metrics listed below.

## Run

The exporter is a plain FastAPI app — no GPU, no model load. It
opens a Postgres pool and a Tailscale client at lifespan-start
(unless injected by tests via `create_app(pool=..., tracker=...,
tailscale=...)`).

```bash
# Local dev
uv run python -m services.cloud_cost_exporter.app

# Container
docker build -f services/cloud_cost_exporter/Dockerfile \
    -t vo-rag-cloud-cost-exporter .
docker run --rm -p 9109:9109 vo-rag-cloud-cost-exporter
```

| Env | Default | Notes |
| --- | --- | --- |
| `CLOUD_COST_EXPORTER_PORT` | `9109` | uvicorn listen port |
| `settings.database.url` | (required) | populated from `vo_rag.core.config.get_settings()`; the shipped settings file fills it from `VORAG_POSTGRES_URL` |
| `settings.cloud.tailscale.{oauth_client_id,oauth_client_secret,tailnet}` | (optional) | Tailscale device listing for `vo_rag_cloud_active_workers`; when unset, the gauge stays at `provider="not_configured"` and `/healthz` still returns 200 |

## Endpoints

| Path | Method | Description |
| --- | --- | --- |
| `/metrics` | `GET` | Prometheus text format |
| `/healthz` | `GET` | `{"ok": true}` — no DB required |

## Metrics

Label values are sanitised to `[a-zA-Z0-9_]*` (Prometheus label
rule). The label set the exporter publishes agrees with the worker
registry's workload tags — see
`tests/integration/test_compute_contracts.py`
(`test_cost_exporter_workload_labels_match_registry`).

| Name | Type | Labels | Source |
| --- | --- | --- | --- |
| `vo_rag_cloud_budget_usd` | Gauge | — | `settings.cloud.monthly_budget_usd` |
| `vo_rag_cloud_budget_pct` | Gauge | — | `CloudCostTracker.get_spend_report().budget_pct_used` |
| `vo_rag_cloud_budget_remaining_usd` | Gauge | — | `CloudCostTracker.get_spend_report().budget_remaining_usd` |
| `vo_rag_cloud_jobs_this_month` | Gauge | — | `CloudCostTracker.get_spend_report().jobs_this_month` |
| `vo_rag_cloud_spend_month_usd` | Gauge | `workload`, `provider`, `gpu_type` | `CloudCostTracker.get_spend_report().by_*` |
| `vo_rag_cloud_active_workers` | Gauge | `provider`, `gpu_type` | `TailscaleProvisioner.list_devices()` |
| `vo_rag_cloud_kill_switch_triggers_7d` | Gauge | — | `cloud_kill_log` 7-day count |

**Secret hygiene.** Log lines are run through `_redact` before
they reach the logger — `RUNPOD_API_KEY`, `R2_ACCESS_KEY`,
`R2_SECRET_KEY`, `GARAGE_ACCESS_KEY`, `GARAGE_SECRET_KEY`,
`DB_PASS`, `POSTGRES_PASSWORD` are stripped to `[REDACTED]`.
See `tests/services/cloud_cost_exporter/test_metrics.py::test_exception_no_secret_leak`.

**No-duplicated-SQL.** The exporter reads spend from
`CloudCostTracker` rather than re-running the monthly aggregation
query — see
`tests/services/cloud_cost_exporter/test_metrics.py::test_tracker_called_no_duplicate_sql`.

## Grafana dashboard

The metric contract above is what a Grafana dashboard should be built against.

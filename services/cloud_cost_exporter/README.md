# services/cloud_cost_exporter

Prometheus exporter for vo-rag cloud-budget telemetry.

## Endpoints

| Path | Description |
|------|-------------|
| `/metrics` | Prometheus text format |
| `/healthz` | `{"ok": true}` — no DB required |

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CLOUD_COST_EXPORTER_PORT` | `9109` | HTTP listen port |

## Metrics

| Name | Type | Labels | Description |
|------|------|--------|-------------|
| `vo_rag_cloud_budget_usd` | Gauge | — | Monthly budget |
| `vo_rag_cloud_budget_pct` | Gauge | — | % consumed |
| `vo_rag_cloud_budget_remaining_usd` | Gauge | — | Remaining budget |
| `vo_rag_cloud_jobs_this_month` | Gauge | — | Jobs launched this month |
| `vo_rag_cloud_spend_month_usd` | Gauge | workload, provider, gpu_type | Spend breakdown |
| `vo_rag_cloud_active_workers` | Gauge | provider, gpu_type | Tailscale-tagged devices |
| `vo_rag_cloud_kill_switch_triggers_7d` | Gauge | — | Kill-switch activations (7d) |

## Build

```bash
docker build -f services/cloud_cost_exporter/Dockerfile -t vo-rag-cloud-cost-exporter .
```

## Grafana dashboard

This service defines only the exporter metric contract above; dashboard JSON
is imported at deploy time.

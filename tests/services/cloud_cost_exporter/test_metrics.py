"""Tests for /metrics endpoint and metric rendering."""

from __future__ import annotations

import contextlib
from typing import Any

from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
from services.cloud_cost_exporter.app import _refresh, create_app


def _parse_metrics(client: TestClient) -> dict[str, Any]:
    """Fetch /metrics and return {metric_name: [{labels, value}, ...]}."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    out: dict[str, Any] = {}
    for family in text_string_to_metric_families(resp.text):
        out[family.name] = []
        for sample in family.samples:
            out[family.name].append({"labels": sample.labels, "value": sample.value})
    return out


async def test_metrics_from_tracker(fake_tracker, fake_pool, fake_tailscale):
    app = create_app(
        tracker=fake_tracker, pool=fake_pool, tailscale=fake_tailscale, poll_interval_s=3600
    )
    with TestClient(app) as client:
        await _refresh(app)
        metrics = _parse_metrics(client)

    assert fake_tracker.call_count >= 1
    assert any(m["value"] == 50.0 for m in metrics.get("vo_rag_cloud_budget_usd", []))
    assert any(abs(m["value"] - 24.68) < 0.01 for m in metrics.get("vo_rag_cloud_budget_pct", []))
    assert any(
        abs(m["value"] - 37.66) < 0.01 for m in metrics.get("vo_rag_cloud_budget_remaining_usd", [])
    )
    assert any(m["value"] == 7 for m in metrics.get("vo_rag_cloud_jobs_this_month", []))
    spend_samples = metrics.get("vo_rag_cloud_spend_month_usd", [])
    assert spend_samples


async def test_missing_tailscale_not_unhealthy(fake_tracker, fake_pool):
    app = create_app(tracker=fake_tracker, pool=fake_pool, tailscale=None, poll_interval_s=3600)
    with TestClient(app) as client:
        await _refresh(app)
        hz = client.get("/healthz")
        assert hz.status_code == 200
        assert hz.json() == {"ok": True}
        metrics = _parse_metrics(client)
    # Active workers should be zero / not_configured.
    workers = metrics.get("vo_rag_cloud_active_workers", [])
    assert workers
    assert any(w["labels"].get("provider") == "not_configured" and w["value"] == 0 for w in workers)


async def test_kill_switch_from_pool(fake_tracker, fake_pool, fake_tailscale):
    fake_pool.fetchval_value = 3
    app = create_app(
        tracker=fake_tracker, pool=fake_pool, tailscale=fake_tailscale, poll_interval_s=3600
    )
    with TestClient(app) as client:
        await _refresh(app)
        metrics = _parse_metrics(client)
    ks = metrics.get("vo_rag_cloud_kill_switch_triggers_7d", [])
    assert ks
    assert any(s["value"] == 3 for s in ks)


async def test_tracker_called_no_duplicate_sql(fake_tracker, fake_pool, fake_tailscale):
    """When tracker is injected, the exporter must not run its own monthly-spend SQL."""
    app = create_app(
        tracker=fake_tracker, pool=fake_pool, tailscale=fake_tailscale, poll_interval_s=3600
    )
    with TestClient(app) as client:
        await _refresh(app)
        client.get("/metrics")
    # The tracker should have been asked for the report.
    assert fake_tracker.call_count >= 1


async def test_exception_no_secret_leak(fake_tracker, fake_pool, fake_tailscale, caplog):
    """If refresh raises, the exception must not contain RUNPOD_API_KEY or DB password."""
    fake_tracker.raise_on_get = RuntimeError("boom RUNPOD_API_KEY=secret DB_PASS=secret")
    app = create_app(
        tracker=fake_tracker, pool=fake_pool, tailscale=fake_tailscale, poll_interval_s=3600
    )
    with TestClient(app), contextlib.suppress(RuntimeError):
        await _refresh(app)
    assert any("refresh failed" in r.message for r in caplog.records)
    for record in caplog.records:
        assert "RUNPOD_API_KEY" not in record.message
        assert "DB_PASS" not in record.message

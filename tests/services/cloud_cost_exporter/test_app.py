"""Unit tests for cloud_cost_exporter app surface."""

from __future__ import annotations

from fastapi.testclient import TestClient
from services.cloud_cost_exporter.app import app as module_app
from services.cloud_cost_exporter.app import create_app


def test_healthz(fake_tracker, fake_pool, fake_tailscale):
    app = create_app(tracker=fake_tracker, pool=fake_pool, tailscale=fake_tailscale)
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_module_level_app_exists():
    assert module_app is not None
    assert hasattr(module_app, "state")

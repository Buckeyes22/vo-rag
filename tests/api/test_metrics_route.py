"""Test the /metrics route on the main vo-rag FastAPI app.

Mirrors the shape of ``tests/api/test_health.py``: build
a hermetic TestClient via ``create_app(settings=, services=)`` from
``conftest.py`` so no network sockets are opened. The metrics route is
auth-exempt (``EXEMPT_PATHS`` in ``vo_rag.api.middleware``), so no
X-API-Key header is required.
"""

from __future__ import annotations

import importlib

from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST

from vo_rag.api.main import create_app
from vo_rag.api.singletons import SurfaceServices

from .conftest import fake_settings

# Import for side effect only: registers the vo_rag_* collectors on the default
# prometheus_client registry so /metrics has families to expose under test.
importlib.import_module("vo_rag.core.observability.metrics")


def _build_client() -> TestClient:
    services = SurfaceServices(settings=fake_settings())
    app = create_app(settings=services.settings, services=services)
    return TestClient(app)


def test_metrics_returns_200_with_prometheus_content_type() -> None:
    client = _build_client()
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == CONTENT_TYPE_LATEST


def test_metrics_body_contains_vo_rag_family() -> None:
    client = _build_client()
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"vo_rag_" in response.content
    assert b"vo_rag_llm_calls_total" in response.content


def test_metrics_is_auth_exempt() -> None:
    """``/metrics`` must respond 200 without an X-API-Key header."""
    client = _build_client()
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "invalid_auth" not in response.text

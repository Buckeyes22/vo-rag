"""``/health``, ``/ready``, ``/status``, ``/api/v1/rag/status``.

These routes are the operator liveness + readiness + status surface.
F-014: secrets are NEVER echoed in the response — the test asserts that
the recognisable placeholder strings from ``fake_settings`` do not
appear anywhere in the body.

The contract:

- ``GET /health`` keeps the ``{"status": "ok", "service": "vo-rag"}`` shape
  so the existing ``test_health.py`` stays green.
- ``GET /ready`` is unauth (already in ``EXEMPT_PATHS``) and returns
  503 when any required dependency slot is unavailable.
- ``GET /status`` and ``GET /api/v1/rag/status`` are auth-gated and
  return the readiness-style envelope with masked secrets.
- ``GET /api/v1/rag/status`` requires the raw collection and reports the
  opt-in sterile collection without requiring it for readiness.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tests.api.conftest import (
    FakeCloudCostTracker,
    FakePool,
    FakeQdrantClient,
    FakeQdrantStore,
    FakeReranker,
    auth_header,
    fake_settings,
    make_services,
)

pytestmark = pytest.mark.security

# Recognisable secret placeholders — must never appear in any response body.
_SECRET_NEEDLES = (
    "test-admin-key",
    "RUNPOD-SECRET-VALUE",
    "ci-key",
    "csrf-secret",
)


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #
def test_health_unauth_returns_status_ok_shape(app_client) -> None:
    """/health returns {"status": "ok", "service": "vo-rag"}."""
    client = app_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "vo-rag"}


def test_health_does_not_require_services() -> None:
    """/health must work even when services are not injected.

    The /health route is in EXEMPT_PATHS; it cannot depend on
    ``get_services`` or it would 503 in any minimal deployment.
    """
    from vo_rag.api.main import create_app

    app = create_app()  # no services — no lifespan path required
    resp = TestClient(app).get("/health")
    assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# /ready
# --------------------------------------------------------------------------- #
def test_ready_returns_200_when_all_required_deps_ok(app_client, mock_embedding_health) -> None:
    """With all required deps configured, /ready returns 200."""
    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
        reranker=FakeReranker(),
    )
    client = app_client(services)
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    # The /ready body surfaces a per-service map; required services must be ok.
    for required in ("db_pool", "qdrant", "embedding_server", "reranker"):
        assert body["services"][required]["ok"] is True
    assert body["services"]["dispatcher"]["ok"] is False


def test_ready_returns_503_when_required_dep_missing(app_client) -> None:
    """Missing required dep (no db_pool) → 503 + explicit reason."""
    services = make_services(
        db_pool=None,
        db_pool_unavailable_reason="db_pool_unavailable: connection refused",
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
        reranker=FakeReranker(),
    )
    client = app_client(services)
    resp = client.get("/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["ok"] is False
    assert body["services"]["db_pool"]["ok"] is False
    assert "db_pool_unavailable" in (body["services"]["db_pool"].get("error") or "")
    assert any("db_pool" in reason for reason in body["reasons"])


def test_ready_returns_503_when_qdrant_down(app_client) -> None:
    """A live Qdrant client that raises → 503 (not a hidden 200)."""
    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(
            _client=FakeQdrantClient(
                raise_get_collections=RuntimeError("qdrant socket closed"),
            )
        ),
        reranker=FakeReranker(),
    )
    client = app_client(services)
    resp = client.get("/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["services"]["qdrant"]["ok"] is False
    assert "qdrant socket closed" in (body["services"]["qdrant"].get("error") or "")


def test_ready_reports_reranker_unavailable(app_client) -> None:
    """Reranker down → /ready says so explicitly with the recorded reason."""
    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
        reranker=None,
        reranker_unavailable_reason="reranker_unavailable: model file missing",
    )
    client = app_client(services)
    resp = client.get("/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["services"]["reranker"]["ok"] is False
    assert "model file missing" in (body["services"]["reranker"].get("error") or "")


def test_query_ready_fails_when_embedding_is_down(app_client) -> None:
    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
        reranker=FakeReranker(),
    )
    body = app_client(services).get("/ready/query").json()
    assert body["ok"] is False
    assert body["surface"] == "query"
    assert body["services"]["embedding_server"]["ok"] is False


def test_remote_reranker_readiness_is_active(app_client, mock_embedding_health) -> None:
    import respx

    settings = fake_settings(
        reranker=SimpleNamespace(url="http://reranker.local/rerank", backend="client")
    )
    services = make_services(
        settings=settings,
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
        reranker=FakeReranker(),
    )
    with respx.mock(assert_all_called=True) as router:
        router.get("http://reranker.local/ready").respond(503)
        response = app_client(services).get("/ready/query")
    assert response.status_code == 503
    assert response.json()["services"]["reranker"]["ok"] is False


def test_query_can_be_ready_while_intake_is_not(
    app_client, mock_embedding_health, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vo_rag.api.routes import system

    monkeypatch.setattr(
        system,
        "dispatcher_health",
        lambda: {"ok": False, "error": "stale_heartbeat", "age_seconds": 300.0},
    )
    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
        reranker=FakeReranker(),
    )
    client = app_client(services)
    assert client.get("/ready/query").status_code == 200
    intake = client.get("/ready/intake")
    assert intake.status_code == 503
    assert intake.json()["surface"] == "intake"


# --------------------------------------------------------------------------- #
# /status
# --------------------------------------------------------------------------- #
def test_status_returns_envelope_and_qdrant_and_cloud_summaries(
    app_client, mock_embedding_health
) -> None:
    """/status surfaces envelope counts, qdrant collections, and cloud spend."""
    pool = FakePool(
        fetch_rows=[
            SimpleNamespace(status="received", count=3),
            SimpleNamespace(status="triaged", count=2),
        ],
        fetchval_value=0,
    )
    services = make_services(
        db_pool=pool,
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
        cloud_cost_tracker=FakeCloudCostTracker(),
        reranker=FakeReranker(),
    )
    client = app_client(services)
    resp = client.get("/status", headers=auth_header())
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["service"] == "vo-rag"
    # Envelope counts surfaced as a per-status map.
    assert body["envelope_counts"] == {"received": 3, "triaged": 2}
    # Qdrant summary lists the two production collections.
    assert "qdrant" in body
    assert "vo_rag_books_raw_v1" in body["qdrant"]["collections"]
    assert "vo_rag_books_sterile_v1" in body["qdrant"]["collections"]
    # Cloud summary surfaces the spend report (masked — no runpod key leak).
    assert body["cloud"]["total_spend_usd"] == 1.5


def test_status_masks_secrets(app_client) -> None:
    """The /status body must not contain api_key, secret_key, RUNPOD_API_KEY,
    Garage creds, or the raw VORAG_API_KEYS_JSON payload."""
    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
        cloud_cost_tracker=FakeCloudCostTracker(),
    )
    client = app_client(services)
    resp = client.get("/status", headers=auth_header())
    assert resp.status_code == 200
    body_text = json.dumps(resp.json())
    for needle in _SECRET_NEEDLES:
        assert needle not in body_text, f"secret leaked: {needle!r}"


def test_status_handles_missing_optional_services_explicitly(app_client) -> None:
    """Missing optional services must surface as not_configured, not crash."""
    services = make_services(
        db_pool=FakePool(),
        qdrant=None,
        qdrant_unavailable_reason="qdrant_unavailable: not configured",
        cloud_cost_tracker=None,
        cloud_cost_unavailable_reason="cloud_cost_unavailable: db_pool_unavailable",
    )
    client = app_client(services)
    resp = client.get("/status", headers=auth_header())
    assert resp.status_code == 200
    body = resp.json()
    assert body["qdrant"]["ok"] is False
    assert body["qdrant"]["error"] == "not_configured"
    assert body["cloud"]["ok"] is False
    assert body["cloud"]["error"] == "not_configured"


# --------------------------------------------------------------------------- #
# /api/v1/rag/status
# --------------------------------------------------------------------------- #
def test_rag_status_checks_both_production_collections(app_client, mock_embedding_health) -> None:
    """The RAG status surface reports both production collections."""
    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
    )
    client = app_client(services)
    resp = client.get("/api/v1/rag/status", headers=auth_header())
    assert resp.status_code == 200
    body = resp.json()
    assert "raw_collection" in body
    assert "sterile_collection" in body
    assert body["raw_collection"]["collection"] == "vo_rag_books_raw_v1"
    assert body["raw_collection"]["points"] == 100
    assert body["sterile_collection"]["collection"] == "vo_rag_books_sterile_v1"
    assert body["sterile_collection"]["points"] == 80
    assert body["readiness_policy"] == "raw_required_sterile_optional"
    assert body["sterile_collection_required"] is False
    assert body["ready"] is True
    assert body["dispatcher"]["ok"] is False


def test_rag_status_ready_when_raw_has_points_and_sterile_is_empty(
    app_client, mock_embedding_health
) -> None:
    """Raw-only production is ready because sterilization is opt-in."""
    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(
            _client=FakeQdrantClient(
                points_by_collection={
                    "vo_rag_books_raw_v1": 100,
                    "vo_rag_books_sterile_v1": 0,
                }
            )
        ),
    )
    client = app_client(services)
    resp = client.get("/api/v1/rag/status", headers=auth_header())
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["raw_collection_has_points"] is True
    assert body["sterile_collection_has_points"] is False
    assert body["sterile_collection_required"] is False


@pytest.mark.parametrize("sterile_points", [0, 80])
def test_rag_status_not_ready_when_raw_collection_empty(
    app_client, mock_embedding_health, sterile_points: int
) -> None:
    """An empty canonical raw collection makes the RAG surface NotReady."""
    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(
            _client=FakeQdrantClient(
                points_by_collection={
                    "vo_rag_books_raw_v1": 0,
                    "vo_rag_books_sterile_v1": sterile_points,
                }
            )
        ),
    )
    client = app_client(services)
    resp = client.get("/api/v1/rag/status", headers=auth_header())
    assert resp.status_code == 503
    body = resp.json()
    assert body["ready"] is False
    assert body["raw_collection_has_points"] is False
    assert body["sterile_collection_has_points"] is bool(sterile_points)


def test_rag_status_masks_secrets(app_client, mock_embedding_health) -> None:
    """/api/v1/rag/status body must not leak any secret needle."""
    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
    )
    client = app_client(services)
    resp = client.get("/api/v1/rag/status", headers=auth_header())
    body_text = json.dumps(resp.json())
    for needle in _SECRET_NEEDLES:
        assert needle not in body_text, f"secret leaked: {needle!r}"


def test_rag_status_returns_503_when_qdrant_unreachable(app_client, mock_embedding_health) -> None:
    """Qdrant down → rag/status must not paper over with a 200."""
    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(
            _client=FakeQdrantClient(
                raise_count=RuntimeError("qdrant count failed"),
            )
        ),
    )
    client = app_client(services)
    resp = client.get("/api/v1/rag/status", headers=auth_header())
    assert resp.status_code == 503
    body = resp.json()
    assert body["ready"] is False
    assert body["raw_collection"]["ok"] is False


# --------------------------------------------------------------------------- #
# Auth gating
# --------------------------------------------------------------------------- #
def test_status_requires_api_key(app_client) -> None:
    """/status is auth-gated — 401 without X-API-Key."""
    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
    )
    client = app_client(services)
    resp = client.get("/status")
    assert resp.status_code == 401


def test_rag_status_requires_api_key(app_client) -> None:
    """/api/v1/rag/status is auth-gated — 401 without X-API-Key."""
    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
    )
    client = app_client(services)
    resp = client.get("/api/v1/rag/status")
    assert resp.status_code == 401


# --------------------------------------------------------------------------- #
# Health: /health + test_health must keep working
# --------------------------------------------------------------------------- #
def test_health_via_imported_app_keeps_status_contract() -> None:
    """The import-time ``app = create_app()`` must serve /health.

    The ``tests/api/test_health.py`` test imports ``app`` directly;
    if the system router overwrites /health or breaks it, that test
    fails. This is a regression guard.
    """
    from vo_rag.api.main import app

    resp = TestClient(app).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "vo-rag"}

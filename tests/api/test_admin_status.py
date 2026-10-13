"""``GET /api/v1/admin/status`` operator surface.

The admin route is the narrow operator console. It is read-only and
returns:

- ``pipeline_roster`` — list of ``PipelineSpec.model_dump()`` from
  :func:`vo_rag.core.config.pipelines.load_pipeline_registry`.
- ``eval_gate`` — :func:`vo_rag.core.eval.gate.run_eval_gate` result
  against the same registry.
- ``compute_placement`` — static workload→location map for the five
  workload classes the compute registry defines.
- ``services`` — readiness summaries (db_pool, qdrant, reranker, …) +
  reranker availability.
- ``cost`` — :class:`CloudCostTracker.get_spend_report` output, or
  ``{"ok": false, "error": "not_configured"}`` when the tracker is
  unavailable.

The route is auth-gated: an unauthenticated request gets 401, an
authenticated request gets 200. The test asserts the body shape
without leaking any of the recognisable secret placeholders.
"""

from __future__ import annotations

import json

import pytest

from tests.api.conftest import (
    FakeCloudCostTracker,
    FakePool,
    FakeQdrantClient,
    FakeQdrantStore,
    FakeReranker,
    auth_header,
    make_services,
)

pytestmark = pytest.mark.security

# Secret placeholders — must not appear anywhere in the admin/status body.
_SECRET_NEEDLES = (
    "test-admin-key",
    "RUNPOD-SECRET-VALUE",
    "ci-key",
    "csrf-secret",
)


# --------------------------------------------------------------------------- #
# Auth gating
# --------------------------------------------------------------------------- #
def test_admin_status_requires_api_key(app_client) -> None:
    """No X-API-Key → 401. F-001: never silent-admin."""
    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
    )
    client = app_client(services)
    resp = client.get("/api/v1/admin/status")
    assert resp.status_code == 401


def test_admin_status_200_with_api_key(app_client) -> None:
    """A valid X-API-Key returns 200 + the full operator envelope."""
    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
        reranker=FakeReranker(),
    )
    client = app_client(services)
    resp = client.get("/api/v1/admin/status", headers=auth_header())
    assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# pipeline_roster
# --------------------------------------------------------------------------- #
def test_admin_status_includes_pipeline_roster_from_registry(app_client) -> None:
    """pipeline_roster mirrors load_pipeline_registry().values().

    The fake settings don't change the pyproject.toml roster, so the
    test asserts the keys/structure the real registry produces.
    """
    from vo_rag.core.config.pipelines import load_pipeline_registry

    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
    )
    client = app_client(services)
    resp = client.get("/api/v1/admin/status", headers=auth_header())
    body = resp.json()
    assert "pipeline_roster" in body
    assert isinstance(body["pipeline_roster"], list)
    # The real registry ships at least book_rag; the test asserts the
    # roster is the real registry, not a fake.
    real = load_pipeline_registry()
    assert {p["name"] for p in body["pipeline_roster"]} == set(real.keys())
    # Each entry is a PipelineSpec.model_dump() — name, status, summary.
    for entry in body["pipeline_roster"]:
        assert set(entry.keys()) >= {"name", "status", "summary"}


# --------------------------------------------------------------------------- #
# eval_gate
# --------------------------------------------------------------------------- #
def test_admin_status_includes_eval_gate_result(app_client) -> None:
    """eval_gate mirrors the registered eval gate result."""
    from vo_rag.core.config.pipelines import load_pipeline_registry
    from vo_rag.core.eval.registered import run_registered_eval_gate

    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
    )
    client = app_client(services)
    resp = client.get("/api/v1/admin/status", headers=auth_header())
    body = resp.json()
    assert "eval_gate" in body
    registry = load_pipeline_registry()
    real = run_registered_eval_gate(registry)
    assert body["eval_gate"]["passed"] == real.passed
    assert body["eval_gate"]["checked"] == real.checked
    assert body["eval_gate"]["skipped"] == real.skipped
    assert body["eval_gate"]["failed"] == real.failed


def test_admin_status_reports_blocked_eval_gate_mode(app_client, monkeypatch) -> None:
    monkeypatch.setenv("VO_RAG_EVAL_MODE", "retrieval_gate")

    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
    )
    client = app_client(services)
    resp = client.get("/api/v1/admin/status", headers=auth_header())
    body = resp.json()

    assert body["eval_gate"]["passed"] is False
    # The snapshot fixture is checked in; the admin route supplies no
    # retriever factory, so retrieval mode reports the retriever as missing.
    assert body["eval_gate"]["blocked_preconditions"] == {"book_rag": "blocked_missing_retriever"}
    assert body["eval_gate"]["failed"] == ["book_rag"]
    assert body["eval_gate"]["skipped"]


# --------------------------------------------------------------------------- #
# compute_placement
# --------------------------------------------------------------------------- #
def test_admin_status_compute_placement_is_static_map(app_client) -> None:
    """compute_placement maps the five workload classes to WorkloadLocation."""
    from vo_rag.core.cloud import WorkloadLocation

    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
    )
    client = app_client(services)
    resp = client.get("/api/v1/admin/status", headers=auth_header())
    body = resp.json()
    assert body["compute_placement"] == {
        "text-gen": WorkloadLocation.RUNPOD_SERVERLESS.value,
        "vlm": WorkloadLocation.RUNPOD_SERVERLESS.value,
        "dsi": WorkloadLocation.RUNPOD_SERVERLESS.value,
        "embedding": WorkloadLocation.LOCAL.value,
        "query-handling": WorkloadLocation.LOCAL.value,
    }


# --------------------------------------------------------------------------- #
# services + reranker availability
# --------------------------------------------------------------------------- #
def test_admin_status_services_block_includes_reranker_availability(app_client) -> None:
    """services.reranker surfaces the reranker slot's state explicitly."""
    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
        reranker=None,
        reranker_unavailable_reason="reranker_unavailable: model file missing",
    )
    client = app_client(services)
    resp = client.get("/api/v1/admin/status", headers=auth_header())
    body = resp.json()
    assert "services" in body
    assert body["services"]["reranker"]["ok"] is False
    assert "model file missing" in (body["services"]["reranker"].get("error") or "")


def test_admin_status_services_block_surfaces_db_pool_state(app_client) -> None:
    """services.db_pool reflects the lifespan-recorded unavailable reason."""
    services = make_services(
        db_pool=None,
        db_pool_unavailable_reason="db_pool_unavailable: connection refused",
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
    )
    client = app_client(services)
    resp = client.get("/api/v1/admin/status", headers=auth_header())
    body = resp.json()
    assert body["services"]["db_pool"]["ok"] is False
    assert "db_pool_unavailable" in (body["services"]["db_pool"].get("error") or "")


# --------------------------------------------------------------------------- #
# cost
# --------------------------------------------------------------------------- #
def test_admin_status_cost_reports_spend_when_tracker_present(app_client) -> None:
    """When cloud_cost_tracker is set, cost mirrors its get_spend_report()."""
    tracker = FakeCloudCostTracker(
        spend_report={
            "month": "2026-06",
            "total_spend_usd": 2.5,
            "budget_usd": 50.0,
            "budget_remaining_usd": 47.5,
            "budget_pct_used": 5.0,
            "by_provider": {"runpod": 2.5},
            "by_workload": {"text-gen": 2.5},
            "by_gpu_type": {"A100": 2.5},
            "jobs_this_month": 5,
        }
    )
    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
        cloud_cost_tracker=tracker,
    )
    client = app_client(services)
    resp = client.get("/api/v1/admin/status", headers=auth_header())
    body = resp.json()
    assert body["cost"]["total_spend_usd"] == 2.5
    assert body["cost"]["budget_usd"] == 50.0


def test_admin_status_cost_reports_not_configured_when_tracker_missing(app_client) -> None:
    """No cloud_cost_tracker → cost = {"ok": False, "error": "not_configured"}."""
    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
        cloud_cost_tracker=None,
        cloud_cost_unavailable_reason="cloud_cost_unavailable: db_pool_unavailable",
    )
    client = app_client(services)
    resp = client.get("/api/v1/admin/status", headers=auth_header())
    body = resp.json()
    assert body["cost"]["ok"] is False
    assert body["cost"]["error"] == "not_configured"


# --------------------------------------------------------------------------- #
# Secret non-disclosure (F-014)
# --------------------------------------------------------------------------- #
def test_admin_status_does_not_leak_secrets(app_client) -> None:
    """The admin/status body must not echo api_key, RUNPOD_API_KEY, Garage
    creds, or raw VORAG_API_KEYS_JSON. F-014: any of these in the body
    is a security test failure.
    """
    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
        cloud_cost_tracker=FakeCloudCostTracker(),
    )
    client = app_client(services)
    resp = client.get("/api/v1/admin/status", headers=auth_header())
    body_text = json.dumps(resp.json())
    for needle in _SECRET_NEEDLES:
        assert needle not in body_text, f"secret leaked: {needle!r}"


def test_admin_status_does_not_leak_secrets_when_cost_tracker_raises(app_client) -> None:
    """Even when the cost tracker raises, the body must not echo secrets.

    A buggy implementation that catches the exception and re-emits the
    raw error message (which might include the runpod api key) would
    fail this test.
    """

    class _RaisingTracker:
        async def get_spend_report(self) -> dict[str, object]:
            raise RuntimeError("auth failed for key RUNPOD-SECRET-VALUE")

    services = make_services(
        db_pool=FakePool(),
        qdrant=FakeQdrantStore(_client=FakeQdrantClient()),
        cloud_cost_tracker=_RaisingTracker(),
    )
    client = app_client(services)
    resp = client.get("/api/v1/admin/status", headers=auth_header())
    body_text = json.dumps(resp.json())
    assert "RUNPOD-SECRET-VALUE" not in body_text

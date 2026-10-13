"""Operator console — narrow read-only admin surface.

vo-rag's ``/api/v1/admin/status`` is NOT a portal clone. It returns the four
artifacts an operator needs at a glance:

- ``pipeline_roster`` — list of ``PipelineSpec.model_dump()`` from
  :func:`vo_rag.core.config.pipelines.load_pipeline_registry`.
- ``eval_gate`` — :func:`vo_rag.core.eval.registered.run_registered_eval_gate`
  result against the same registry.
- ``compute_placement`` — static workload→location map for the five
  workload classes in ``_ADMIN_WORKLOADS``.
- ``services`` — readiness summaries (db_pool, qdrant, reranker, …).
- ``cost`` — :class:`CloudCostTracker.get_spend_report` output, or
  ``{"ok": False, "error": "not_configured"}`` when the tracker is
  unavailable.

The route is read-only. No mutation endpoints live here. F-014:
secrets are NEVER echoed — see ``_safe_cost_payload`` and the
``_SECRET_NEEDLES`` constant in tests.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends

from vo_rag.api.auth import ApiKeyIdentity
from vo_rag.api.dependencies import get_identity, get_services
from vo_rag.api.singletons import SurfaceServices
from vo_rag.core.cloud import workload_location
from vo_rag.core.config import load_pipeline_registry
from vo_rag.core.eval.registered import run_registered_eval_gate

log = logging.getLogger("vo_rag.api.routes.admin")

router = APIRouter()

# Static workload→location map. The list is closed — adding a new
# workload class requires editing this list AND
# ``vo_rag.core.cloud.router._STATIC_MAP``. The duplication is
# intentional: the route's contract is a closed allowlist, not a
# reflection of the underlying router's implementation.
_ADMIN_WORKLOADS: tuple[str, ...] = (
    "text-gen",
    "vlm",
    "dsi",
    "embedding",
    "query-handling",
)


@router.get("/api/v1/admin/status")
async def admin_status(
    identity: ApiKeyIdentity = Depends(get_identity),  # noqa: B008
    services: SurfaceServices = Depends(get_services),  # noqa: B008
) -> dict[str, Any]:
    """Read-only operator status — requires API identity.

    The dependency on :func:`get_identity` is the auth gate. F-001:
    requests without a resolved ``vo_rag_api_identity`` on
    ``request.state`` get 401 from the auth module before this
    handler runs. The identity is referenced so the route signature
    documents the requirement; the body itself does not need to
    inspect it (the operator surface is read-only).
    """
    registry = load_pipeline_registry()
    pipeline_roster = [spec.model_dump() for spec in registry.values()]
    eval_result = await asyncio.to_thread(
        run_registered_eval_gate,
        registry,
    )
    eval_dump = eval_result.model_dump()

    services_block = _services_block(services)
    cost_block = await _cost_block(services)

    return {
        "ok": True,
        "actor": _actor_summary(identity),
        "pipeline_roster": pipeline_roster,
        "eval_gate": eval_dump,
        "compute_placement": _compute_placement(),
        "services": services_block,
        "cost": cost_block,
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _actor_summary(identity: ApiKeyIdentity) -> dict[str, Any]:
    """Identity fields safe to surface to the operator (no secrets)."""
    return {
        "name": identity.name,
        "caller": identity.caller,
        "is_admin": identity.is_admin,
        "auth_source": identity.auth_source,
        "email": identity.email,
    }


def _compute_placement() -> dict[str, str]:
    """Static workload→location map for the ``_ADMIN_WORKLOADS`` allowlist."""
    out: dict[str, str] = {}
    for workload in _ADMIN_WORKLOADS:
        try:
            out[workload] = str(workload_location(workload).value)
        except ValueError:
            log.warning("admin_workload_location_missing", extra={"workload": workload})
            out[workload] = "unknown"
    return out


def _services_block(services: SurfaceServices) -> dict[str, dict[str, Any]]:
    """Per-slot readiness summary. Each slot is read off the container;
    no reflection on the settings object, so secrets never reach the wire."""
    block: dict[str, dict[str, Any]] = {
        "db_pool": _slot_health(services.db_pool, services.db_pool_unavailable_reason),
        "qdrant": _slot_health(services.qdrant, services.qdrant_unavailable_reason),
        "garage": _slot_health(services.garage, services.garage_unavailable_reason),
        "cloud": _slot_health(services.cloud_cost_tracker, services.cloud_cost_unavailable_reason),
        "kill_switch": _slot_health(services.kill_switch, services.kill_switch_unavailable_reason),
    }
    # Reranker is required for the answer lane — flag it explicitly so
    # operators see when it is unavailable (a dev env without the int8
    # ONNX artifact shows up here, not as a hidden 500).
    block["reranker"] = _slot_health(services.reranker, services.reranker_unavailable_reason)
    return block


def _slot_health(value: Any, reason: str) -> dict[str, Any]:
    if value is None:
        return {"ok": False, "error": reason or "not_configured"}
    return {"ok": True}


async def _cost_block(services: SurfaceServices) -> dict[str, Any]:
    if services.cloud_cost_tracker is None:
        return {
            "ok": False,
            "error": "not_configured",
            "reason": services.cloud_cost_unavailable_reason or "not_configured",
        }
    try:
        report = await services.cloud_cost_tracker.get_spend_report()
    except Exception as exc:  # noqa: BLE001 — never echo the raw message
        log.warning("admin_cost_report_failed", extra={"error": str(exc)})
        return {"ok": False, "error": "report_failed"}
    if hasattr(report, "model_dump"):
        return {"ok": True, **report.model_dump()}
    if isinstance(report, dict):
        return {"ok": True, **report}
    return {"ok": True, "report": str(report)}


__all__ = ["router"]

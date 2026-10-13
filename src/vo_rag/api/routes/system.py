"""System liveness, readiness, and status routes.

The inline checks are split into focused helpers under a hardened
secret-masking policy. Three properties worth knowing:

- The ``/health`` payload is the minimal contract
  ``{"status": "ok", "service": "vo-rag"}``. The richer dependency
  checks live on ``/ready`` and ``/status`` so liveness stays
  dependency-free and existing monitoring configs keep working.
- The qdrant store wrapper is ``QdrantStore``; the underlying
  ``AsyncQdrantClient`` is on ``_client`` (no public wrapper for
  ``get_collections`` / ``count`` exists).
- Secret masking is the explicit F-014 contract: no string from
  ``settings.api.api_key``, ``settings.api.api_keys_json``,
  ``settings.cloud.api_key`` (RUNPOD_API_KEY), ``settings.garage.access_key``
  / ``secret_key`` may appear in any body. The /status and
  /api/v1/rag/status responses are constructed from explicit
  attribute reads, never from a ``settings.model_dump()`` call.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from vo_rag.api.dependencies import get_services, get_settings_dep
from vo_rag.api.singletons import SurfaceServices
from vo_rag.pipelines.book_rag.dispatcher_health import dispatcher_health

log = logging.getLogger("vo_rag.api.routes.system")

router = APIRouter()

# Production readiness collections. Raw is the canonical internal-serving
# index. Sterilization is an explicit, cost-bearing opt-in, so its collection
# remains observable without making a raw-only deployment NotReady.
RAW_BOOK_COLLECTION = "vo_rag_books_raw_v1"
STERILE_BOOK_COLLECTION = "vo_rag_books_sterile_v1"
STERILE_BOOK_COLLECTION_REQUIRED = False
_RAG_PRODUCTION_COLLECTIONS: tuple[str, ...] = (RAW_BOOK_COLLECTION, STERILE_BOOK_COLLECTION)


# --------------------------------------------------------------------------- #
# /health — minimal contract, no deps, always 200
# --------------------------------------------------------------------------- #
@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe.

    Pinned to the contract ``{"status": "ok", "service": "vo-rag"}``, which
    ``tests/api/test_health.py`` asserts.
    This route intentionally does NOT depend on ``get_services`` — a
    minimal deployment without injected services must still serve it.
    """
    return {"status": "ok", "service": "vo-rag"}


# --------------------------------------------------------------------------- #
# /ready/query + /ready/intake — dependency-true, surface-specific probes
# --------------------------------------------------------------------------- #
@router.get("/ready")
@router.get("/ready/query")
async def ready(
    services: SurfaceServices = Depends(get_services),  # noqa: B008
    settings: Any = Depends(get_settings_dep),  # noqa: B008
) -> JSONResponse:
    """Query readiness; ``/ready`` remains its compatibility alias."""
    db = await _db_pool_health(services)
    qdrant_health = await _qdrant_health(services)
    reranker_health = await _reranker_health(services, settings)
    embedding_health = await _embedding_health(settings)
    cloud_health = _cloud_health(services)
    garage_health = _garage_health(services)
    dispatcher = dispatcher_health()

    services_map: dict[str, dict[str, Any]] = {
        "db_pool": db,
        "qdrant": qdrant_health,
        "reranker": reranker_health,
        "embedding_server": embedding_health,
        "cloud": cloud_health,
        "garage": garage_health,
        "dispatcher": dispatcher,
    }

    required = ("db_pool", "qdrant", "embedding_server", "reranker")
    reasons = [
        f"{name}_unavailable: {services_map[name].get('error', 'unknown')}"
        for name in required
        if not services_map[name].get("ok")
    ]
    ok = not reasons

    body: dict[str, Any] = {
        "ok": ok,
        "surface": "query",
        "services": services_map,
        "reasons": reasons,
    }
    code = status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=body, status_code=code)


@router.get("/ready/intake")
async def intake_ready(
    services: SurfaceServices = Depends(get_services),  # noqa: B008
) -> JSONResponse:
    """Intake readiness requires durable admission and a live dispatcher."""
    db = await _db_pool_health(services)
    dispatcher = dispatcher_health()
    services_map: dict[str, dict[str, Any]] = {
        "db_pool": db,
        "dispatcher": dispatcher,
    }
    required = ("db_pool", "dispatcher")
    reasons = [
        f"{name}_unavailable: {services_map[name].get('error', 'unknown')}"
        for name in required
        if not services_map[name].get("ok")
    ]
    ok = not reasons
    return JSONResponse(
        content={
            "ok": ok,
            "surface": "intake",
            "services": services_map,
            "reasons": reasons,
        },
        status_code=status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE,
    )


# --------------------------------------------------------------------------- #
# /status — auth-gated, envelope/qdrant/cloud summaries
# --------------------------------------------------------------------------- #
@router.get("/status")
async def status_route(
    request: Request,  # noqa: ARG001 — kept for parity with the lifespan path
    services: SurfaceServices = Depends(get_services),  # noqa: B008
    settings: Any = Depends(get_settings_dep),  # noqa: B008
) -> dict[str, Any]:
    """Aggregate operator status: envelopes, qdrant, cloud spend.

    Auth-gated (NOT in ``EXEMPT_PATHS``). Secrets are masked by
    construction — only the fields explicitly enumerated below are
    emitted; no ``settings.model_dump()`` or reflection.
    """
    pool = services.db_pool
    envelope_counts: dict[str, int] = {}
    if pool is not None:
        try:
            rows = await pool.fetch(
                "SELECT status, COUNT(*) AS count FROM intake_envelopes GROUP BY status"
            )
        except Exception as exc:  # noqa: BLE001 — surface to operator
            log.warning("envelope_count_query_failed", extra={"error": str(exc)})
        else:
            for row in rows:
                envelope_counts[str(row["status"])] = int(row["count"])
    else:
        log.info("envelope_count_skipped: db_pool_unavailable")

    active_states = {"received", "triaging", "triaged", "dispatched", "processing"}
    queue_depth = sum(envelope_counts.get(state, 0) for state in active_states)

    qdrant_summary = await _qdrant_summary(services)
    cloud_summary = await _cloud_summary(services)
    reranker_summary = await _reranker_health(services, settings)
    db_summary = await _db_pool_health(services)
    embedding_summary = await _embedding_health(settings)
    dispatcher = dispatcher_health()

    return {
        "ok": all(
            s.get("ok") for s in (db_summary, qdrant_summary, reranker_summary, embedding_summary)
        ),
        "service": "vo-rag",
        "version": "0.1.0",
        "envelope_counts": envelope_counts,
        "active_workers": 1 if dispatcher.get("ok") is True else 0,
        "queue_depth": queue_depth,
        "qdrant": qdrant_summary,
        "cloud": cloud_summary,
        "services": {
            "db_pool": db_summary,
            "qdrant": qdrant_summary,
            "reranker": reranker_summary,
            "embedding_server": embedding_summary,
            "dispatcher": dispatcher,
        },
    }


# --------------------------------------------------------------------------- #
# /api/v1/rag/status — checks production collections
# --------------------------------------------------------------------------- #
@router.get("/api/v1/rag/status")
async def rag_status(
    services: SurfaceServices = Depends(get_services),  # noqa: B008
    settings: Any = Depends(get_settings_dep),  # noqa: B008
) -> JSONResponse:
    """Production-readiness probe for the vo-rag RAG surface.

    Requires the canonical raw collection (``vo_rag_books_raw_v1``) and
    its runtime dependencies. The opt-in sterile collection
    (``vo_rag_books_sterile_v1``) remains observable but is not required
    by the current raw-only readiness policy. The route does NOT require
    ``book_rag`` to be ACTIVE — collection availability is reported
    regardless of pipeline status, so operators can distinguish missing
    data from a disabled pipeline.
    """
    db = await _db_pool_health(services)
    qdrant_h = await _qdrant_health(services)
    embedding_h = await _embedding_health(settings)
    dispatcher = dispatcher_health()

    raw_collection = await _collection_count(services, RAW_BOOK_COLLECTION)
    sterile_collection = await _collection_count(services, STERILE_BOOK_COLLECTION)

    raw_ok = bool(raw_collection.get("ok")) and int(raw_collection.get("points", 0)) > 0
    sterile_ok = bool(sterile_collection.get("ok")) and int(sterile_collection.get("points", 0)) > 0
    required_collections_ok = raw_ok and (sterile_ok if STERILE_BOOK_COLLECTION_REQUIRED else True)
    ready = (
        bool(db.get("ok"))
        and bool(qdrant_h.get("ok"))
        and bool(embedding_h.get("ok"))
        and required_collections_ok
    )

    body: dict[str, Any] = {
        "ready": ready,
        "db_pool": db,
        "qdrant": qdrant_h,
        "embedding_server": embedding_h,
        "dispatcher": dispatcher,
        "raw_collection": raw_collection,
        "sterile_collection": sterile_collection,
        "readiness_policy": "raw_required_sterile_optional",
        "sterile_collection_required": STERILE_BOOK_COLLECTION_REQUIRED,
        "raw_collection_has_points": raw_ok,
        "sterile_collection_has_points": sterile_ok,
    }
    code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=body, status_code=code)


# --------------------------------------------------------------------------- #
# Helpers — one per check. Splitting inline checks into helpers keeps the
# route bodies focused on the response shape and lets tests/operators
# reach a single helper without a full request cycle.
# --------------------------------------------------------------------------- #
async def _db_pool_health(services: SurfaceServices) -> dict[str, Any]:
    if services.db_pool is None:
        reason = services.db_pool_unavailable_reason or "not_configured"
        return {"ok": False, "error": reason}
    try:
        async with services.db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
    except Exception as exc:  # noqa: BLE001 — surface to operator
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


async def _qdrant_health(services: SurfaceServices) -> dict[str, Any]:
    qdrant = services.qdrant
    if qdrant is None:
        reason = services.qdrant_unavailable_reason or "not_configured"
        return {"ok": False, "error": reason}
    try:
        collections = await qdrant._client.get_collections()  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001 — surface to operator
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "collections": len(collections.collections)}


async def _qdrant_summary(services: SurfaceServices) -> dict[str, Any]:
    """Per-collection point counts for the /status body."""
    qdrant = services.qdrant
    if qdrant is None:
        reason = services.qdrant_unavailable_reason or "not_configured"
        return {"ok": False, "error": "not_configured", "reason": reason, "collections": []}
    try:
        result = await qdrant._client.get_collections()  # noqa: SLF001
        names = [str(c.name) for c in result.collections]
    except Exception as exc:  # noqa: BLE001 — surface to operator
        return {"ok": False, "error": str(exc), "collections": []}
    counts: dict[str, int] = {}
    for name in names:
        try:
            response = await qdrant._client.count(  # noqa: SLF001
                collection_name=name, exact=True
            )
            counts[name] = int(response.count)
        except Exception as exc:  # noqa: BLE001
            counts[name] = -1
            log.warning("qdrant_count_failed", extra={"collection": name, "error": str(exc)})
    return {"ok": True, "collections": names, "points": counts}


async def _collection_count(services: SurfaceServices, collection_name: str) -> dict[str, Any]:
    qdrant = services.qdrant
    if qdrant is None:
        reason = services.qdrant_unavailable_reason or "not_configured"
        return {"ok": False, "collection": collection_name, "error": reason}
    try:
        response = await qdrant._client.count(  # noqa: SLF001
            collection_name=collection_name, exact=True
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "collection": collection_name, "error": str(exc)}
    return {
        "ok": True,
        "collection": collection_name,
        "points": int(response.count),
    }


async def _reranker_health(services: SurfaceServices, settings: Any) -> dict[str, Any]:
    if services.reranker is None:
        reason = services.reranker_unavailable_reason or "not_configured"
        return {"ok": False, "error": reason}
    reranker_settings = getattr(settings, "reranker", None)
    url = str(getattr(reranker_settings, "url", "") or "").rstrip("/")
    backend = str(getattr(reranker_settings, "backend", "") or "")
    if url or backend == "client":
        if not url:
            return {"ok": False, "error": "remote_reranker_url_not_configured"}
        if url.endswith("/rerank"):
            url = url[: -len("/rerank")]
        health_url = url + "/ready"
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                response = await client.get(health_url)
                response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc), "endpoint": health_url}
        return {"ok": True, "status_code": response.status_code, "endpoint": health_url}
    return {"ok": True}


async def _embedding_health(settings: Any) -> dict[str, Any]:
    url = getattr(settings.embedding, "query_url", "") if settings is not None else ""
    if not url:
        return {"ok": False, "error": "not_configured"}
    base = url.rstrip("/")
    # The embedding service exposes /health at its root, not under /embed; the
    # configured query_url points at the /embed endpoint. Strip it before
    # appending /health so the readiness probe hits the right path.
    if base.endswith("/embed"):
        base = base[: -len("/embed")]
    health_url = base + "/health"
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            response = await client.get(health_url)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "endpoint": health_url}
    if payload.get("status") != "ready" or "device" not in payload:
        return {
            "ok": False,
            "error": "unexpected_embedding_health_payload",
            "status_code": response.status_code,
        }
    return {"ok": True, "status_code": response.status_code, "device": payload["device"]}


def _garage_health(services: SurfaceServices) -> dict[str, Any]:
    if services.garage is None:
        reason = services.garage_unavailable_reason or "not_configured"
        return {"ok": False, "error": reason}
    return {"ok": True}


def _cloud_health(services: SurfaceServices) -> dict[str, Any]:
    if services.cloud_cost_tracker is None:
        reason = services.cloud_cost_unavailable_reason or "not_configured"
        return {"ok": False, "error": reason}
    return {"ok": True}


async def _cloud_summary(services: SurfaceServices) -> dict[str, Any]:
    if services.cloud_cost_tracker is None:
        return {
            "ok": False,
            "error": "not_configured",
            "reason": services.cloud_cost_unavailable_reason or "not_configured",
        }
    try:
        report = await services.cloud_cost_tracker.get_spend_report()
    except Exception as exc:  # noqa: BLE001 — never echo exc (could embed secrets)
        log.warning("cloud_cost_report_failed", extra={"error": str(exc)})
        return {"ok": False, "error": "report_failed"}
    if hasattr(report, "model_dump"):
        return {"ok": True, **report.model_dump()}
    if isinstance(report, dict):
        return {"ok": True, **report}
    return {"ok": True, "report": str(report)}


__all__ = [
    "RAW_BOOK_COLLECTION",
    "STERILE_BOOK_COLLECTION",
    "_RAG_PRODUCTION_COLLECTIONS",
    "router",
]

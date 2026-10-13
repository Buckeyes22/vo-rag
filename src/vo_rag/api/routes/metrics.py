"""Prometheus exposition route for the vo-rag HTTP surface.

Mirrors the worker implementation in
``src/vo_rag/workers/embedding/server.py`` (the ``metrics()`` handler
at lines 187-198) — serves the default ``prometheus_client`` registry,
which is where every ``vo_rag_*`` collector registered by
:mod:`vo_rag.core.observability.metrics` and the optional
``prometheus-fastapi-instrumentator`` already lives.

This is a leaf module: it does not import ``get_services``,
``vo_rag.core.storage``, or anything else that opens a socket at import
time. Mounting it on both the services-injected and lifespan
``create_app()`` paths is therefore safe.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_client import REGISTRY as _PROM_DEFAULT_REGISTRY

router = APIRouter()


@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus exposition endpoint.

    Returns the default ``prometheus_client`` registry output. The
    ``vo_rag_*`` collectors are registered onto the default registry at
    import time by :mod:`vo_rag.core.observability.metrics` (and, if
    installed, by ``prometheus-fastapi-instrumentator``); no per-request
    setup is required.
    """
    payload = generate_latest(_PROM_DEFAULT_REGISTRY)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


__all__ = ["router"]

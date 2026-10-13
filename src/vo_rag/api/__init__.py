"""vo-rag HTTP surface (FastAPI).

Public entry points:

- :func:`create_app` — ASGI app factory. Hermetic when ``services``
  is injected (no DB / Qdrant / Garage sockets open at import time
  or first request).
- :data:`app` — the module-level ASGI app, built once at import
  time via ``create_app()`` for ``uvicorn vo_rag.api.main:app``.

All route surfaces (system / admin / query / answer / ingest) are
importable from :mod:`vo_rag.api.routes` and mounted by
:func:`create_app`. The auth/middleware layer is in
:mod:`vo_rag.api.auth` and :mod:`vo_rag.api.middleware`. The
service container that the routes reach into is in
:mod:`vo_rag.api.singletons`.
"""

from __future__ import annotations

from vo_rag.api.auth import (
    ALLOWED_CALLERS,
    AccessTokenError,
    ApiKeyIdentity,
    authenticate_api_key,
    identity_from_request,
    normalize_caller,
)
from vo_rag.api.main import app, create_app
from vo_rag.api.schemas import (
    AdminStatusResponse,
    AnswerRequest,
    AnswerResponse,
    ComplianceInfo,
    EnvelopeResponse,
    IntakeSubmission,
    QueryHit,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
    ServiceHealth,
    StatusResponse,
    UploadUrlRequest,
    UploadUrlResponse,
)
from vo_rag.api.singletons import (
    SurfaceServices,
    build_surface_services,
    close_surface_services,
)

__all__ = [
    "ALLOWED_CALLERS",
    "AccessTokenError",
    "AdminStatusResponse",
    "AnswerRequest",
    "AnswerResponse",
    "ApiKeyIdentity",
    "ComplianceInfo",
    "EnvelopeResponse",
    "IntakeSubmission",
    "QueryHit",
    "QueryRequest",
    "QueryResponse",
    "ReadinessResponse",
    "ServiceHealth",
    "StatusResponse",
    "SurfaceServices",
    "UploadUrlRequest",
    "UploadUrlResponse",
    "app",
    "authenticate_api_key",
    "build_surface_services",
    "close_surface_services",
    "create_app",
    "identity_from_request",
    "normalize_caller",
]

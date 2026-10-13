"""vo-rag HTTP middleware: API key + Cloudflare Access authentication,
CSRF for mutating CF Access requests, structlog request-id binding.

Notes:
- /ready is in ``EXEMPT_PATHS`` (readiness probe)
- the resolved identity lives on the request-state attribute
  ``vo_rag_api_identity``
- settings come from ``vo_rag.core.config.get_settings``
- /webhooks/github/star is intentionally NOT in EXEMPT_PATHS — repo_analysis
  webhooks are not built.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

import structlog
import structlog.contextvars as cv
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from vo_rag.api.auth import (
    AccessTokenError,
    authenticate_api_key,
    authenticate_cloudflare_access,
    validate_csrf_token,
)
from vo_rag.core import config as config_module
from vo_rag.core.models import ulid_new

EXEMPT_PATHS = {
    "/health",
    "/ready",
    "/ready/query",
    "/ready/intake",
    "/docs",
    "/openapi.json",
    "/metrics",
}
CSRF_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path in EXEMPT_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        settings = config_module.get_settings()
        identity = authenticate_api_key(request.headers.get("X-API-Key"), settings)
        if identity is None:
            try:
                identity = await authenticate_cloudflare_access(request, settings)
            except AccessTokenError as exc:
                return JSONResponse({"error": str(exc)}, status_code=401)

        if identity is None:
            return JSONResponse({"error": "invalid_auth"}, status_code=401)

        if (
            identity.auth_source == "cloudflare_access"
            and request.method in CSRF_MUTATING_METHODS
            and not _csrf_valid(request, settings, identity.email)
        ):
            return JSONResponse({"error": "invalid_csrf_token"}, status_code=403)

        request.state.vo_rag_api_identity = identity
        return await call_next(request)


async def logging_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    logger = structlog.get_logger("vo_rag.api")
    request_id = request.headers.get("X-Request-Id") or str(ulid_new())
    cv.clear_contextvars()
    cv.bind_contextvars(request_id=request_id, path=request.url.path, method=request.method)
    t0 = time.perf_counter()
    response: Response | None = None

    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        if response is not None:
            response.headers["X-Request-Id"] = request_id

        logger.info(
            "request",
            duration_ms=duration_ms,
            status=response.status_code if response is not None else None,
        )
        cv.clear_contextvars()


def _csrf_valid(request: Request, settings: object, email: str | None) -> bool:
    if not email:
        return False
    portal = getattr(settings, "portal", None)
    secret = getattr(portal, "csrf_secret", "")
    token = request.headers.get("X-CSRF-Token")
    return validate_csrf_token(token, email=email, secret=str(secret or ""))

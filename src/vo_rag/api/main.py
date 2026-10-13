"""FastAPI app factory for the vo-rag HTTP surface.

``create_app()`` is the app factory and the public entry point for
uvicorn and tests.

Design constraints:

- ``create_app(settings=None, services=None)`` is injectable for tests.
  When ``services`` is passed, no lifespan is registered and the test
  gets a hermetic ASGI app — no DB / Qdrant / Garage sockets open.
- ``app = create_app()`` remains the ASGI export. Importing
  ``vo_rag.api.main`` must not open sockets; only the lifespan does,
  on first request inside an ASGI server.
- The :class:`ApiKeyMiddleware` and request logging middleware are
  installed unconditionally.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from vo_rag.api.lifespan import _mirror_services_to_state, lifespan
from vo_rag.api.middleware import ApiKeyMiddleware, logging_middleware
from vo_rag.api.routes import answer as answer_route
from vo_rag.api.routes import query as query_route
from vo_rag.api.singletons import SurfaceServices

__all__ = ["app", "create_app"]


@asynccontextmanager
async def _noop_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Stand-in lifespan used when services are pre-injected.

    FastAPI rejects ``lifespan=None`` (it falls back to the default
    no-op lifespan that still runs, which can hit the network if a
    route triggers a deferred import). This no-op is the smallest
    possible async context manager — it just yields once and exits.
    """
    yield


def create_app(
    settings: Any = None,
    services: SurfaceServices | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Parameters
    ----------
    settings:
        Optional pre-resolved :class:`Settings` instance. When omitted,
        the lifespan resolves it through ``vo_rag.core.config.get_settings``.
    services:
        Optional pre-built :class:`SurfaceServices` instance for tests.
        When provided, a no-op lifespan is registered (FastAPI does
        not allow ``lifespan=None`` to mean "skip lifespan") and the
        services are mirrored onto ``app.state`` without opening any
        sockets.
    """
    if services is not None:
        # Tests + integration: no network. The injected services are
        # mirrored onto app.state exactly the way the lifespan would
        # do it.
        app = FastAPI(title="vo-rag", version="0.1.0", lifespan=_noop_lifespan)
        app.state.surface_services = services
        app.state.settings = settings if settings is not None else services.settings
        app.state.services_injected = True
        for key, value in _mirror_services_to_state(services).items():
            setattr(app.state, key, value)
    else:
        # Production path: lifespan resolves settings + services on
        # first request (and on TestClient ``__enter__``).
        app = FastAPI(title="vo-rag", version="0.1.0", lifespan=lifespan)
        if settings is not None:
            # Caller-supplied settings still skip the cached
            # ``get_settings()`` lookup in the lifespan — useful for
            # the operator override path.
            app.state.settings = settings

    # Middleware order matters: BaseHTTPMiddleware-based middleware
    # added via ``add_middleware`` runs in reverse-registration order
    # on the way in. So registering logging first means it wraps the
    # auth middleware. Auth must be the OUTERMOST layer so unauth
    # requests never touch a route.
    app.middleware("http")(logging_middleware)
    app.add_middleware(ApiKeyMiddleware)

    # Routers: system + admin first, then query/answer, ingest, figures, etc.
    from vo_rag.api.routes import admin as admin_routes
    from vo_rag.api.routes import system as system_routes

    app.include_router(system_routes.router)
    app.include_router(admin_routes.router)

    # Query/answer routes: grouped together so future route additions do not
    # interleave with the health compatibility path above.
    app.include_router(query_route.router)
    app.include_router(answer_route.router)

    # ---------------------------------------------------------------------------
    # Route lanes
    # ---------------------------------------------------------------------------
    # Ingest: /api/v1/intake + /api/v1/intake/upload-url. Each include is
    # additive and grouped under a comment header.
    from vo_rag.api.routes import ingest as ingest_routes

    app.include_router(ingest_routes.router)

    # Figure-image origin serving.
    from vo_rag.api.routes import figures as figures_routes

    app.include_router(figures_routes.router)

    # Portal-support surface (VOR-244).
    from vo_rag.api.routes import portal as portal_routes

    app.include_router(portal_routes.router)

    from vo_rag.api.routes import catalog as catalog_routes

    app.include_router(catalog_routes.router)

    from vo_rag.api.routes import books as books_routes

    app.include_router(books_routes.router)

    # Prometheus exposition — serves the default prometheus_client
    # registry (where vo_rag_* collectors live). Auth-exempt via
    # EXEMPT_PATHS in middleware.py. Mounted after the route lanes so
    # it appears alongside them in OpenAPI and so the block runs for
    # both the services-injected and lifespan paths.
    from vo_rag.api.routes import metrics as metrics_routes

    app.include_router(metrics_routes.router)

    # OpenTelemetry instrumentation is installed HERE, at construction time, on
    # the production path only — NOT in the lifespan. The FastAPI server-span
    # ASGI middleware must be in the stack before Starlette builds and caches it;
    # the lifespan runs after that, which is too late for server spans (verified
    # live: httpx client spans — a timing-independent global patch — appeared,
    # but FastAPI server spans never did when instrumenting from the lifespan).
    # Gate on the env flag so the test-import path and the services-injected
    # (test) path never load settings or touch global OTel state.
    if services is None and os.environ.get("VORAG_OTEL_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        from vo_rag.core.config import get_settings
        from vo_rag.core.observability.otel import (
            instrument_fastapi_app,
            instrument_httpx,
            setup_otel,
        )

        if setup_otel(settings if settings is not None else get_settings()):
            instrument_fastapi_app(app)
            instrument_httpx()

    return app


# ASGI export — kept for uvicorn and ``tests/api/test_health.py``, which
# imports ``app`` directly. This must remain import-time safe (no socket
# opens).
app = create_app()

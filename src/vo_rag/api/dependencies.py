"""FastAPI dependency accessors for the API surface.

The route handlers never reach into ``app.state`` directly —
they call these accessors via ``Depends``. The split exists so:

- The wiring is one importable name per concern (``get_services``,
  ``get_settings_dep``, ``get_identity``).
- ``rerank_fn`` is a thin adapter over ``services.reranker`` that
  returns ``None`` when the reranker is unavailable, so the answer
  route can call ``execute_retrieval(rerank_fn=...)`` uniformly
  without branching on reranker availability.
- Tests can override any of these with ``app.dependency_overrides``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends, HTTPException, Request, status

from vo_rag.api.auth import ApiKeyIdentity, identity_from_request
from vo_rag.api.singletons import SurfaceServices


def get_services(request: Request) -> SurfaceServices:
    """Return the :class:`SurfaceServices` mounted on ``app.state``.

    Routes that need a service call ``services = Depends(get_services)``.
    If the lifespan was bypassed (no services were injected and the
    app is not running through the lifespan startup) this raises 503
    so the route is honest about its state rather than crashing on
    ``None`` later.
    """
    services: SurfaceServices | None = getattr(request.app.state, "surface_services", None)
    if services is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="surface_services_unavailable",
        )
    return services


def get_settings_dep(request: Request) -> Any:
    """Return the active :class:`Settings` for the running app.

    Prefers ``app.state.settings`` (set by the lifespan or the
    ``create_app(settings=...)`` injection path) and falls back to
    ``vo_rag.core.config.get_settings()`` for direct script-style
    usage (e.g. CLI delegating to the API for ad-hoc checks).
    """
    settings = getattr(request.app.state, "settings", None)
    if settings is not None:
        return settings

    from vo_rag.core.config import get_settings

    return get_settings()


def get_identity(request: Request) -> ApiKeyIdentity:
    """Return the resolved identity for the current request, or 401.

    Thin re-export of :func:`vo_rag.api.auth.identity_from_request`
    so route handlers depend on this module's API surface.
    """
    return identity_from_request(request)


def get_rerank_fn(
    services: SurfaceServices = Depends(get_services),  # noqa: B008
) -> Callable[[str, list[Any]], list[Any]] | None:
    """Return a ``RerankFn`` callable or ``None`` when unavailable.

    The returned callable matches the ``RerankFn`` shape in
    ``vo_rag.core.retrieval.execution``. When the reranker slot is
    ``None`` (GTE init failed at boot), this returns ``None`` and
    the route skips the optional rerank pass — never raises.
    """

    def _fn(query: str, candidates: list[Any]) -> list[Any]:
        reranker = services.reranker
        if reranker is None:
            return list(candidates)
        return list(reranker.rerank(query, candidates))

    return _fn if services.reranker is not None else None


__all__ = [
    "get_identity",
    "get_rerank_fn",
    "get_services",
    "get_settings_dep",
]

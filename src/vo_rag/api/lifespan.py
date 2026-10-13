"""FastAPI lifespan that wires :class:`SurfaceServices` onto ``app.state``.

The lifespan is registered by :func:`vo_rag.api.main.create_app` when
``services`` is not injected. The injected path (``create_app(settings=,
services=)``) skips the lifespan entirely so route tests never open
sockets during construction.

The lifespan is intentionally tolerant:

- If the DB pool cannot be opened, the lifespan records the reason on
  ``app.state`` and continues. ``/health`` does not need DB, and the
  /ready route reports the failure as a 503.
- If the GTE reranker fails to load (missing int8 ONNX file), the
  lifespan records the reason and sets ``app.state.reranker = None``.
- The yield is the only blocking call the lifespan makes; teardown
  is bounded by ``close_surface_services`` and tolerates partial
  state.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from vo_rag.api.operational_sampler import run_operational_sampler
from vo_rag.api.singletons import SurfaceServices, build_surface_services, close_surface_services

log = logging.getLogger("vo_rag.api.lifespan")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Wire ``SurfaceServices`` onto ``app.state`` for the request lifetime."""
    settings = getattr(app.state, "settings", None)
    if settings is None:
        from vo_rag.core.config import get_settings

        settings = get_settings()
        app.state.settings = settings

    # NOTE: OTel instrumentation is installed in api.main.create_app at
    # construction time (before the ASGI middleware stack is built), NOT here —
    # the lifespan runs too late for the FastAPI server-span middleware.
    log.info("lifespan_starting")
    services = await build_surface_services(settings)
    app.state.surface_services = services

    # Mirror the per-slot fields onto ``app.state`` so route code can
    # reach them via attribute access (``app.state.reranker``) or via
    # the dependency accessor. The dataclass remains the single source
    # of truth — these mirrors are convenience.
    app.state.db_pool = services.db_pool
    app.state.qdrant = services.qdrant
    app.state.garage = services.garage
    app.state.embedding_client = services.embedding_client
    app.state.retriever = services.retriever
    app.state.reranker = services.reranker
    app.state.answer_synthesizer = services.answer_synthesizer
    app.state.cloud_cost_tracker = services.cloud_cost_tracker
    app.state.kill_switch = services.kill_switch

    index_task = (
        asyncio.create_task(
            _reconcile_qdrant_indexes(services.qdrant), name="vo-rag-qdrant-indexes"
        )
        if services.qdrant is not None
        else None
    )

    sampler_stop = asyncio.Event()
    sampler_task = asyncio.create_task(
        run_operational_sampler(services, settings, sampler_stop),
        name="vo-rag-operational-sampler",
    )

    app.state.reranker_unavailable_reason = services.reranker_unavailable_reason
    app.state.cloud_cost_unavailable_reason = services.cloud_cost_unavailable_reason
    app.state.garage_unavailable_reason = services.garage_unavailable_reason
    app.state.db_pool_unavailable_reason = services.db_pool_unavailable_reason
    app.state.kill_switch_unavailable_reason = services.kill_switch_unavailable_reason
    app.state.qdrant_unavailable_reason = services.qdrant_unavailable_reason

    try:
        yield
    finally:
        log.info("lifespan_closing")
        if index_task is not None and not index_task.done():
            index_task.cancel()
        if index_task is not None:
            await asyncio.gather(index_task, return_exceptions=True)
        sampler_stop.set()
        await sampler_task
        await close_surface_services(services)


def _mirror_services_to_state(services: SurfaceServices) -> dict[str, Any]:
    """Return a dict of per-slot fields ready to attach to ``app.state``.

    Used by the injected path so the mirrors stay in sync with the
    dataclass without duplicating the field list.
    """
    return {
        "db_pool": services.db_pool,
        "qdrant": services.qdrant,
        "garage": services.garage,
        "embedding_client": services.embedding_client,
        "retriever": services.retriever,
        "reranker": services.reranker,
        "answer_synthesizer": services.answer_synthesizer,
        "cloud_cost_tracker": services.cloud_cost_tracker,
        "kill_switch": services.kill_switch,
        "reranker_unavailable_reason": services.reranker_unavailable_reason,
        "cloud_cost_unavailable_reason": services.cloud_cost_unavailable_reason,
        "garage_unavailable_reason": services.garage_unavailable_reason,
        "db_pool_unavailable_reason": services.db_pool_unavailable_reason,
        "kill_switch_unavailable_reason": services.kill_switch_unavailable_reason,
        "qdrant_unavailable_reason": services.qdrant_unavailable_reason,
    }


__all__ = ["lifespan", "_mirror_services_to_state"]


async def _reconcile_qdrant_indexes(store: Any) -> None:
    """Best-effort, in the background: an index build on millions of points
    takes minutes and must not delay readiness; failure only logs (VOR-1427)."""
    from vo_rag.core.storage.qdrant import reconcile_payload_indexes

    try:
        added = await reconcile_payload_indexes(store._client)  # noqa: SLF001
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - never break API startup over an index
        log.warning("qdrant_index_reconcile_failed: %s: %s", type(exc).__name__, exc)
        return
    if added:
        log.info("qdrant_index_reconcile_added %s", added)

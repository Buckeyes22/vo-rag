"""Surface service container for the API surface.

Lives between the FastAPI app factory (``api/main.py``) and the route
handlers. The container is the single seam the API/CLI/MCP
surfaces share to reach core services.

The split from ``api/main.py`` exists so:

- ``build_surface_services`` is testable in isolation (no FastAPI
  coupling, no DB pool opened at construction time unless you opt in).
- ``close_surface_services`` is idempotent and tolerates partial
  construction (a reranker that failed to load, a kill switch that
  could not be wired because Tailscale is unconfigured, etc.).
- Route handlers receive services through a FastAPI
  ``Depends`` that reads from ``app.state``; the route code never
  calls ``build_surface_services`` directly.

Why the build is split out of the lifespan: the lifespan runs on
ASGI startup, but route tests inject a pre-built ``SurfaceServices``
and skip the lifespan entirely. Both paths must produce the same
state shape on ``app.state``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from vo_rag.core.answer.synthesizer import AnswerSynthesizer
from vo_rag.core.embedding import build_embedding_client
from vo_rag.core.reranking.factory import build_reranker
from vo_rag.core.retrieval.search import HybridRetriever
from vo_rag.core.storage.garage import GarageClient
from vo_rag.core.storage.qdrant import QdrantStore

log = logging.getLogger("vo_rag.api.singletons")


@dataclass
class SurfaceServices:
    """The complete service container the API surface mounts on app.state.

    All slots default to ``None`` so partial construction is representable
    — that is the whole point of the per-slot ``*_unavailable_reason``
    fields. A route handler reading ``services.cloud_cost_tracker`` and
    getting ``None`` should check ``services.cloud_cost_unavailable_reason``
    and surface a 503 in /ready rather than crash.
    """

    settings: Any

    db_pool: Any = None
    qdrant: Any = None
    garage: Any = None
    embedding_client: Any = None
    retriever: Any = None
    reranker: Any = None
    answer_synthesizer: Any = None
    cloud_cost_tracker: Any = None
    kill_switch: Any = None

    reranker_unavailable_reason: str = ""
    cloud_cost_unavailable_reason: str = ""
    garage_unavailable_reason: str = ""
    db_pool_unavailable_reason: str = ""
    kill_switch_unavailable_reason: str = ""
    qdrant_unavailable_reason: str = ""

    # Runtime callbacks are explicit slots so every surface uses the same
    # persistence and status contract rather than probing ``extra`` ad hoc.
    ingest_submit: Any = None
    envelope_lookup: Any = None
    status_provider: Any = None

    # Only objects constructed by this container are closed by its lifecycle.
    owned_resources: list[Any] = field(default_factory=list)

    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
async def build_surface_services(
    settings: Any,
    *,
    db_pool: Any = None,
    qdrant: Any = None,
    garage: Any = None,
    embedding_client: Any = None,
    retriever: Any = None,
) -> SurfaceServices:
    """Construct a :class:`SurfaceServices` from ``settings``.

    Every slot can be pre-supplied (tests, integration, MCP) to skip
    the corresponding real construction. Slots that are passed in are
    returned as-is; slots that are left as ``None`` are built here
    using the real core factories.

    The build never raises when an optional dep is missing — the
    corresponding ``*_unavailable_reason`` field is set and the slot
    is left as ``None``. This is the only way the API can boot on a
    dev box without a live GTE model, Garage, or Postgres.
    """
    services = SurfaceServices(settings=settings)

    try:
        # DB pool — only attempt to build if the caller did not inject one.
        if db_pool is None:
            pool_was_initialized = False
            try:
                from vo_rag.core.db.postgres import get_existing_pool, get_pool

                pool_was_initialized = get_existing_pool() is not None
                db_pool = await get_pool(
                    settings.database.url,
                    min_size=settings.database.pool_min,
                    max_size=settings.database.pool_max,
                )
            except Exception as exc:  # noqa: BLE001
                services.db_pool_unavailable_reason = (
                    f"db_pool_unavailable: {type(exc).__name__}: {exc}"
                )
                log.warning("db_pool_unavailable", extra={"reason": str(exc)})
                db_pool = None
            if db_pool is not None and not pool_was_initialized:
                services.owned_resources.append(db_pool)
        services.db_pool = db_pool

        # Qdrant — sync construction; AsyncQdrantClient is built lazily and
        # the only failure mode here is a malformed URL.
        if qdrant is None:
            try:
                qdrant = QdrantStore(
                    settings.qdrant.url,
                    settings.qdrant.api_key or None,
                    prefer_grpc=settings.qdrant.prefer_grpc,
                )
            except Exception as exc:  # noqa: BLE001
                services.qdrant_unavailable_reason = (
                    f"qdrant_unavailable: {type(exc).__name__}: {exc}"
                )
                log.warning("qdrant_unavailable", extra={"reason": str(exc)})
                qdrant = None
            if qdrant is not None:
                services.owned_resources.append(qdrant)
        services.qdrant = qdrant

        # Garage — only attempt if access_key + secret_key are present.
        if garage is None:
            try:
                if getattr(settings.garage, "access_key", "") and getattr(
                    settings.garage, "secret_key", ""
                ):
                    garage = GarageClient(
                        endpoint_url=settings.garage.endpoint,
                        access_key=settings.garage.access_key,
                        secret_key=settings.garage.secret_key,
                        region=getattr(settings.garage, "region", "garage"),
                    )
                else:
                    services.garage_unavailable_reason = "garage_credentials_unconfigured"
                    log.info("garage_unconfigured: missing access_key/secret_key")
            except Exception as exc:  # noqa: BLE001
                services.garage_unavailable_reason = (
                    f"garage_unavailable: {type(exc).__name__}: {exc}"
                )
                log.warning("garage_unavailable", extra={"reason": str(exc)})
                garage = None
            if garage is not None:
                services.owned_resources.append(garage)
        services.garage = garage

        # Embedding client — the factory tolerates a missing model server
        # by returning the raw client; that is fine, it just won't succeed
        # at runtime.
        if embedding_client is None:
            embedding_client = build_embedding_client(
                settings.embedding.query_url,
                timeout_s=float(getattr(settings.embedding, "timeout_s", 30.0)),
            )
            if embedding_client is not None:
                services.owned_resources.append(embedding_client)
        services.embedding_client = embedding_client

        # Hybrid retriever — needs a qdrant store + an embedding client.
        if retriever is None and qdrant is not None and embedding_client is not None:
            retriever = HybridRetriever(qdrant, embedding_client)
        services.retriever = retriever

        # Reranker — backend selected by settings (VOR-8): ``client`` (remote
        # localhost /rerank), ``gpu`` (in-process CUDA), or ``onnx_cpu``
        # (in-process GteReranker fallback). Same warn-and-record policy: record
        # ``reranker_unavailable_reason``, leave the slot ``None``, let /ready
        # reflect it — never fail the whole boot on a missing model/endpoint.
        try:
            services.reranker = build_reranker(settings.reranker)
            if services.reranker is not None:
                services.owned_resources.append(services.reranker)
        except Exception as exc:  # noqa: BLE001
            services.reranker_unavailable_reason = (
                f"reranker_unavailable: {type(exc).__name__}: {exc}"
            )
            services.reranker = None
            log.warning("reranker_unavailable", extra={"reason": str(exc)})

        # Answer synthesizer — always buildable (no network or filesystem).
        services.answer_synthesizer = AnswerSynthesizer(
            settings.answer,
            anthropic_fallback_key=getattr(settings.triage, "anthropic_api_key", ""),
        )

        # Cloud cost tracker — needs a real asyncpg pool.
        if services.db_pool is not None:
            try:
                from vo_rag.core.cloud.cost_tracker import CloudCostTracker

                services.cloud_cost_tracker = CloudCostTracker(
                    services.db_pool,
                    monthly_budget_usd=float(settings.cloud.monthly_budget_usd),
                    per_job_max_usd=float(settings.cloud.per_job_max_usd),
                )
            except Exception as exc:  # noqa: BLE001
                services.cloud_cost_unavailable_reason = (
                    f"cloud_cost_unavailable: {type(exc).__name__}: {exc}"
                )
                log.warning("cloud_cost_unavailable", extra={"reason": str(exc)})
        else:
            services.cloud_cost_unavailable_reason = "cloud_cost_unavailable: db_pool_unavailable"

        # Kill switch — needs Tailscale credentials. We do not fail boot on
        # missing Tailscale creds; record the reason and leave the slot None.
        if getattr(settings.cloud.tailscale, "oauth_client_id", "") and getattr(
            settings.cloud.tailscale, "oauth_client_secret", ""
        ):
            try:
                from vo_rag.core.cloud.kill_switch import CloudKillSwitch
                from vo_rag.core.cloud.tailscale_auth import TailscaleProvisioner

                tailscale = TailscaleProvisioner(
                    oauth_client_id=settings.cloud.tailscale.oauth_client_id,
                    oauth_client_secret=settings.cloud.tailscale.oauth_client_secret,
                    tailnet=getattr(settings.cloud.tailscale, "tailnet", ""),
                )
                services.kill_switch = CloudKillSwitch(tailscale)
                services.owned_resources.append(services.kill_switch)
            except Exception as exc:  # noqa: BLE001
                services.kill_switch_unavailable_reason = (
                    f"kill_switch_unavailable: {type(exc).__name__}: {exc}"
                )
                log.warning("kill_switch_unavailable", extra={"reason": str(exc)})
        else:
            services.kill_switch_unavailable_reason = (
                "kill_switch_unavailable: tailscale_unconfigured"
            )

        return services
    except BaseException:
        await close_surface_services(services)
        raise


# --------------------------------------------------------------------------- #
# Close
# --------------------------------------------------------------------------- #
async def close_surface_services(services: SurfaceServices) -> None:
    """Release resources owned by ``services``.

    Tolerates a partially-built container — every step is guarded so
    a ``None`` slot is just a no-op. This is what makes ``/health``
    and shutdown work when, for example, the reranker never loaded
    or the DB pool was never opened.
    """
    # Embedding client holds an httpx pool — must be closed first so
    # any in-flight request gets a chance to finish.
    resources = list(reversed(services.owned_resources))
    services.owned_resources.clear()
    for resource in resources:
        try:
            if resource is services.db_pool:
                from vo_rag.core.db.postgres import close_pool

                await close_pool()
                continue
            close = getattr(resource, "aclose", None) or getattr(resource, "close", None)
            if close is None:
                continue
            outcome = close()
            if hasattr(outcome, "__await__"):
                await outcome
        except Exception:  # noqa: BLE001
            log.warning("surface_resource_close_failed", exc_info=True)


__all__ = [
    "SurfaceServices",
    "build_surface_services",
    "close_surface_services",
]

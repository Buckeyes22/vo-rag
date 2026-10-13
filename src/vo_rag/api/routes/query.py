"""Thin /api/v1/query route over the landed retrieval helper."""

from __future__ import annotations

import logging
from typing import Any

import structlog.contextvars as cv
from fastapi import APIRouter, Depends, HTTPException, Request, status

from vo_rag.api.auth import enforce_query_scope, identity_from_request, normalize_caller
from vo_rag.api.dependencies import get_services
from vo_rag.api.schemas import QueryHit, QueryRequest, QueryResponse
from vo_rag.api.singletons import SurfaceServices
from vo_rag.core.db.queries.query_log import insert_query_log
from vo_rag.core.models import ulid_new
from vo_rag.core.observability.otel import get_tracer
from vo_rag.core.retrieval.execution import execute_retrieval
from vo_rag.core.retrieval.query_mode import resolve_collections
from vo_rag.core.retrieval.search import SearchResult

router = APIRouter()
log = logging.getLogger("vo_rag.api.routes.query")


@router.post("/api/v1/query", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    request: Request,
    services: SurfaceServices = Depends(get_services),  # noqa: B008
) -> QueryResponse:
    if services.retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="retriever_unavailable",
        )

    identity = identity_from_request(request)
    caller = _identify_caller(request, identity)
    resolved_collections = resolve_collections(payload.mode, payload.collections)
    cv.bind_contextvars(query_mode=payload.mode.value, collections=resolved_collections)

    enforce_query_scope(
        identity=identity,
        caller=caller,
        mode=payload.mode.value,
        collections=resolved_collections,
    )

    await _try_insert_query_log(
        services=services,
        query_text=payload.query,
        mode=payload.mode.value,
        collections=resolved_collections,
        top_k=payload.top_k,
        caller=caller,
    )

    rerank_fn = (
        services.reranker.rerank if payload.rerank and services.reranker is not None else None
    )
    tracer = get_tracer("vo_rag.query")
    results = await execute_retrieval(
        retriever=services.retriever,
        query=payload.query,
        mode=payload.mode,
        requested_collections=payload.collections,
        top_k=payload.top_k,
        filters=payload.filters,
        visual_mode=payload.visual_mode,
        rerank_fn=rerank_fn,
        phase_context=tracer.start_as_current_span,
        auto_merge=services.settings.reranker.auto_merge_enabled,
        auto_merge_threshold=services.settings.reranker.auto_merge_threshold,
        # fetch_parent_fn lands with VOR-306 Half B (multi-resolution ingest);
        # inert until parents exist.
        fetch_parent_fn=None,
    )
    return QueryResponse(results=[_to_hit(result) for result in results])


async def _try_insert_query_log(
    *,
    services: SurfaceServices,
    query_text: str,
    mode: str,
    collections: list[str],
    top_k: int,
    caller: str,
) -> None:
    if services.db_pool is None:
        return

    try:
        await insert_query_log(
            services.db_pool,
            id=ulid_new(),
            query_text=query_text,
            mode=mode,
            collections=collections,
            top_k=top_k,
            caller=caller,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("query_log_insert_failed", extra={"exc_type": type(exc).__name__})


def _to_hit(result: SearchResult) -> QueryHit:
    if result.payload is None:
        raise HTTPException(status_code=422, detail="search result payload missing")
    return QueryHit.from_search_result(result)


def _identify_caller(request: Request, identity: Any) -> str:
    header_value = request.headers.get("X-VO-RAG-Caller")
    if header_value is None and getattr(identity, "caller", None) is not None:
        return str(identity.caller)
    return normalize_caller(header_value)


__all__ = ["router"]

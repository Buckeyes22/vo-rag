"""Protected /api/v1/answer route over retrieval and AnswerSynthesizer."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from vo_rag.api.auth import enforce_query_scope, identity_from_request, normalize_caller
from vo_rag.api.dependencies import get_services
from vo_rag.api.schemas import AnswerRequest, AnswerResponse, ComplianceInfo, QueryHit
from vo_rag.api.singletons import SurfaceServices
from vo_rag.core.answer.context import (
    AnswerEvidenceUnavailable,
    parent_ids_for_results,
    resolve_answer_collections,
    source_ids,
)
from vo_rag.core.answer.synthesizer import AnswerConfigError
from vo_rag.core.compliance.answer_gate import AnswerComplianceGate, AnswerGateDecision
from vo_rag.core.db.queries.answer_generation import insert_answer_log
from vo_rag.core.db.queries.citation_assignments import insert_citation_assignments
from vo_rag.core.models import ulid_new
from vo_rag.core.observability.otel import get_tracer
from vo_rag.core.retrieval.execution import execute_retrieval
from vo_rag.core.retrieval.query_mode import RAW_COLLECTIONS, QueryMode
from vo_rag.core.retrieval.search import SearchResult

router = APIRouter()
log = logging.getLogger("vo_rag.api.routes.answer")


class NoComplianceGateConfigured(RuntimeError):
    """Raised when public answer modes would need a gate that is absent."""


@router.post("/api/v1/answer", response_model=AnswerResponse)
async def answer(
    payload: AnswerRequest,
    request: Request,
    services: SurfaceServices = Depends(get_services),  # noqa: B008
) -> AnswerResponse:
    if services.retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="retriever_unavailable",
        )
    if services.answer_synthesizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="answer_synthesizer_unavailable",
        )

    identity = identity_from_request(request)
    caller = _identify_caller(request, identity)
    resolved_collections = resolve_answer_collections(
        mode=payload.mode,
        requested=payload.collections,
        context_scope=payload.context_scope,
    )
    enforce_query_scope(
        identity=identity,
        caller=caller,
        mode=payload.mode.value,
        collections=resolved_collections,
    )

    _enforce_no_silent_compliance_allow_all(payload.mode, services)
    _enforce_raw_hosted_egress_guard(
        settings=services.settings,
        resolved_collections=resolved_collections,
    )

    rerank_fn = (
        services.reranker.rerank if payload.rerank and services.reranker is not None else None
    )
    tracer = get_tracer("vo_rag.query")
    filtered_results = await execute_retrieval(
        retriever=services.retriever,
        query=payload.query,
        mode=payload.mode,
        requested_collections=resolved_collections,
        top_k=payload.top_k,
        filters=payload.filters,
        visual_mode=payload.visual_mode,
        rerank_fn=rerank_fn,
        phase_context=tracer.start_as_current_span,
    )

    gate_allowed = _enforce_answer_compliance_gate(
        mode=payload.mode,
        services=services,
        hits=filtered_results,
        caller=caller,
    )

    try:
        with tracer.start_as_current_span("answer_synthesis"):
            synthesis = await services.answer_synthesizer.synthesize(
                query=payload.query,
                context=filtered_results,
                profile=payload.profile,
            )
    except AnswerEvidenceUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from None
    except AnswerConfigError as exc:
        log.warning("answer_generation_not_configured", extra={"exc_type": type(exc).__name__})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="answer_generation_not_configured",
        ) from None
    except Exception as exc:  # noqa: BLE001
        log.error("answer_generation_failed", extra={"exc_type": type(exc).__name__})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="answer_generation_failed",
        ) from None

    answer_id = ulid_new()
    warnings = _warnings_for_synthesis(synthesis)
    compliance = _compliance_info(synthesis, filtered_results, gate_allowed=gate_allowed)
    answer_text = synthesis.draft.answer.strip()
    if not answer_text:
        answer_value: str | None = None
        confidence = 0.0
        citations: list[str] = []
    else:
        answer_value = synthesis.draft.answer
        confidence = synthesis.draft.confidence
        citations = list(synthesis.draft.cited_chunk_ids)

    await _persist_answer(
        services=services,
        answer_id=answer_id,
        payload=payload,
        resolved_collections=resolved_collections,
        model=synthesis.model,
        filtered_results=filtered_results,
        warnings=warnings,
        compliance_outcome=(
            "prompt_injection_suspected" if synthesis.should_quarantine else compliance.status
        ),
        synthesis=synthesis,
        citations=citations,
    )

    retrieval_results = (
        [_to_answer_hit(result) for result in filtered_results] if payload.include_retrieval else []
    )
    return AnswerResponse(
        answer_id=answer_id,
        answer=answer_value,
        confidence=confidence,
        citations=citations,
        sources=source_ids(filtered_results),
        retrieval_results=retrieval_results,
        compliance=compliance,
        warnings=warnings,
        model=synthesis.model,
        mode=payload.mode.value,
    )


def _identify_caller(request: Request, identity: Any) -> str:
    header_value = request.headers.get("X-VO-RAG-Caller")
    if header_value is None and getattr(identity, "caller", None) is not None:
        return str(identity.caller)
    return normalize_caller(header_value)


def _enforce_no_silent_compliance_allow_all(mode: QueryMode, services: SurfaceServices) -> None:
    if _answer_compliance_gate(services) is not None:
        return

    if mode in {QueryMode.PUBLICATION, QueryMode.PARTNER}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="answer_compliance_gate_not_configured",
        )


def _gate_forbidden(detail: str, *, mode: QueryMode, **extra: Any) -> HTTPException:
    log.warning(detail, extra={"mode": mode.value, **extra})
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


def _safe_reason(reason: str) -> str:
    return " ".join(reason.splitlines())[:200]


def _enforce_answer_compliance_gate(
    *,
    mode: QueryMode,
    services: SurfaceServices,
    hits: list[SearchResult],
    caller: str,
) -> bool:
    """Call the gate for public modes; any non-allow outcome is a 403.

    Returns True only when a gate was consulted and explicitly allowed the request.
    """
    if mode not in {QueryMode.PUBLICATION, QueryMode.PARTNER}:
        return False

    gate = _answer_compliance_gate(services)
    if not isinstance(gate, AnswerComplianceGate) or not callable(gate.check):
        raise _gate_forbidden("answer_compliance_gate_invalid", mode=mode)

    try:
        decision = gate.check(mode=mode, hits=tuple(hits), caller=caller)
    except Exception as exc:  # noqa: BLE001
        raise _gate_forbidden(
            "answer_compliance_gate_error", mode=mode, exc_type=type(exc).__name__
        ) from None

    if inspect.isawaitable(decision):
        close = getattr(decision, "close", None)
        if callable(close):
            close()
        raise _gate_forbidden("answer_compliance_gate_invalid", mode=mode)
    if (
        not isinstance(decision, AnswerGateDecision)
        or not isinstance(decision.allowed, bool)
        or not isinstance(decision.reason, str)
    ):
        raise _gate_forbidden("answer_compliance_gate_invalid", mode=mode)
    if decision.allowed is not True:
        raise _gate_forbidden(
            "answer_compliance_gate_denied", mode=mode, reason=_safe_reason(decision.reason)
        )
    return True


def _answer_compliance_gate(services: SurfaceServices) -> Any | None:
    direct = getattr(services, "answer_compliance_gate", None)
    if direct is not None:
        return direct
    return services.extra.get("answer_compliance_gate")


def _enforce_raw_hosted_egress_guard(
    *,
    settings: Any,
    resolved_collections: list[str],
) -> None:
    answer_settings = getattr(settings, "answer", None)
    allow_hosted = bool(getattr(answer_settings, "allow_hosted", False))
    allow_raw = bool(getattr(answer_settings, "allow_raw_hosted_egress", False))
    if (
        allow_hosted
        and not allow_raw
        and any(collection in RAW_COLLECTIONS for collection in resolved_collections)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="raw_context_hosted_egress_forbidden",
        )


def _warnings_for_synthesis(synthesis: Any) -> list[str]:
    if not synthesis.should_quarantine:
        return []

    warnings = ["prompt_injection_suspected"]
    for signal in getattr(synthesis.validation, "signals", []):
        kind = str(getattr(signal, "kind", ""))
        if kind and kind not in warnings:
            warnings.append(kind)
    return warnings


def _compliance_info(
    synthesis: Any, filtered_results: list[SearchResult], *, gate_allowed: bool = False
) -> ComplianceInfo:
    gate_status = "allowed" if gate_allowed else "not_configured"
    details: dict[str, Any] = {
        "parent_ids": parent_ids_for_results(filtered_results),
        "signals": [
            {
                "kind": str(getattr(signal, "kind", "")),
                "confidence": float(getattr(signal, "confidence", 0.0)),
            }
            for signal in getattr(synthesis.validation, "signals", [])
        ],
    }
    if synthesis.should_quarantine:
        return ComplianceInfo(
            status=gate_status,
            should_quarantine=True,
            reason="prompt_injection_suspected",
            pipeline="query_answer",
            details=details,
        )
    return ComplianceInfo(status=gate_status, pipeline="query_answer", details=details)


async def _persist_answer(
    *,
    services: SurfaceServices,
    answer_id: str,
    payload: AnswerRequest,
    resolved_collections: list[str],
    model: str,
    filtered_results: list[SearchResult],
    warnings: list[str],
    compliance_outcome: str,
    synthesis: Any,
    citations: list[str],
) -> None:
    if services.db_pool is None:
        return

    try:
        await insert_answer_log(
            services.db_pool,
            id=answer_id,
            query_log_id=None,
            query_text=payload.query,
            mode=payload.mode.value,
            profile=payload.profile.value,
            model=model,
            prompt_name=getattr(synthesis.llm_response, "prompt_name", None),
            prompt_version=getattr(synthesis.llm_response, "prompt_version", None),
            collections=resolved_collections,
            context_scope=payload.context_scope.value,
            retrieved_count=len(filtered_results),
            source_ids=source_ids(filtered_results),
            latency_ms=float(getattr(synthesis.llm_response, "latency_ms", 0.0)),
            cost_usd=float(getattr(synthesis.llm_response, "cost_usd", 0.0)),
            compliance_outcome=compliance_outcome,
            warnings=warnings,
        )
        await insert_citation_assignments(services.db_pool, answer_id, _citation_rows(citations))
    except Exception as exc:  # noqa: BLE001
        log.warning("answer_persistence_failed", extra={"exc_type": type(exc).__name__})


def _citation_rows(citations: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, citation in enumerate(citations):
        rows.append(
            {
                "text_segment": citation,
                "start_offset": index,
                "end_offset": index + len(citation),
                "tier": "chunk",
                "source_id": citation,
                "source_title": None,
                "confidence": 1.0,
            }
        )
    return rows


def _to_answer_hit(result: SearchResult) -> QueryHit:
    return QueryHit(
        id=result.id,
        score=float(result.score),
        collection=result.collection,
        payload=_redacted_payload(result.payload),
    )


def _redacted_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sensitive_exact = {
        "api_key",
        "authorization",
        "content",
        "content_text",
        "credentials",
        "password",
        "raw_content",
        "secret",
        "summary",
        "text",
        "token",
    }
    out: dict[str, Any] = {}
    for key, value in payload.items():
        normalized = key.lower()
        if normalized in sensitive_exact or any(
            marker in normalized
            for marker in ("api_key", "credential", "password", "secret", "token")
        ):
            continue
        out[key] = value
    return out


__all__ = ["NoComplianceGateConfigured", "router"]

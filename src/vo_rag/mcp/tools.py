"""MCP tool implementations — pure async callables over core seams.

Each function takes a :class:`SurfaceServices` (or duck-typed
equivalent) as the first argument and is independent of FastMCP.
The :mod:`vo_rag.mcp.server` module closes over the services and
registers these as FastMCP tools.

Why split tools from server?  Two reasons:

1. The implementations are the public contract — every assertion
   in ``tests/mcp/test_mcp_tools.py`` runs against these functions
   directly with a fake ``SurfaceServices``. Server tests then
   pin the FastMCP wire-up (the four tool names, the ``vo-rag``
   server name) without dragging in any HTTP / stdio transport.
2. The functions are importable in scripts and notebooks without
   triggering the FastMCP server's import-time side effects
   (logging config, etc.). They are the production seam a non-MCP
   caller (a worker, a one-off job) can reach into.

Invariants pinned by the test suite:

- **F-012 boundary** — query, status, and submit-ingest tools do not
  call an LLM. ``vo_rag_ingest_stream_impl`` executes the BookRag ingest
  pipeline in-process, so any LLM use happens behind that pipeline's
  governed seams rather than through a separate MCP path.
- **F-014** — ``vo_rag_ingest_impl`` runs ``validate_external_url``
  for ``url`` / ``github_star`` source types before any further
  work; unsafe URLs raise :class:`ToolError` and no envelope is
  created.
- **F-031** — error responses surface ``warnings`` rather than
  being silently flattened to success.
- **Deterministic ids preserved** — :class:`McpQueryHit` carries
  the core ``SearchResult.id`` verbatim, never re-mapped.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import structlog

from vo_rag.core.chunking.models import brain_evidence_is_complete, hit_payload
from vo_rag.core.ingest.normalize import IntakeSubmission, normalize
from vo_rag.core.models.envelope import (
    IntakeSource,
    Priority,
    Sovereignty,
)
from vo_rag.core.retrieval.execution import execute_retrieval
from vo_rag.core.retrieval.query_mode import QueryMode
from vo_rag.core.retrieval.search import SearchResult, validate_filters
from vo_rag.mcp.errors import ToolError
from vo_rag.mcp.models import (
    McpEnvelopeView,
    McpIngestArgs,
    McpIngestResult,
    McpQueryArgs,
    McpQueryHit,
    McpQueryResult,
    McpStatusArgs,
    McpStatusResult,
)
from vo_rag.web.url_safety import SSRFBlocked, validate_external_url

if TYPE_CHECKING:
    from vo_rag.core.config import Settings

# Cap on top_k enforced at the MCP boundary. The API surface allows
# up to 200, but the MCP tool is consumed by models that will
# pack every hit into a context window — 50 is the documented
# hard ceiling.
_MCP_TOP_K_CEILING = 50
_MCP_DEFAULT_TOP_K = 5
log = structlog.get_logger(__name__)

# Source types that carry a URL payload and must be SSRF-gated.
_URL_SOURCES: frozenset[str] = frozenset({IntakeSource.URL.value, IntakeSource.GITHUB_STAR.value})


async def stream_book_rag_ingest(*args: Any, **kwargs: Any) -> Any:
    """Lazy proxy kept as the stable monkeypatch seam for streaming tests."""
    from vo_rag.pipelines.book_rag.runner import stream_book_rag_ingest as _stream_book_rag_ingest

    async for event in _stream_book_rag_ingest(*args, **kwargs):
        yield event


async def _fail_envelope_best_effort(
    fail_fn: Any,
    pool: Any,
    envelope_id: str,
    error: str,
) -> None:
    try:
        await fail_fn(pool, envelope_id, error)
    except Exception as exc:
        # Keep this aligned with the SSE sibling in api/routes/ingest.py.
        log.error(
            "mcp_ingest_fail_persist_failed",
            envelope_id=envelope_id,
            error=str(exc),
            original_error=error,
            exc_info=True,
        )
        return


def _exception_error_text(exc: BaseException) -> str:
    return str(exc) or exc.__class__.__name__


# --------------------------------------------------------------------------- #
# vo_rag_query
# --------------------------------------------------------------------------- #
async def vo_rag_query_impl(
    *,
    services: Any,
    query: str,
    collections: list[str] | None = None,
    top_k: int = _MCP_DEFAULT_TOP_K,
    mode: str = QueryMode.INTERNAL.value,
    filters: dict[str, Any] | None = None,
    rerank: bool = True,
    visual_mode: str | None = None,
) -> McpQueryResult:
    """Search the vo-rag knowledge layer and return ranked hits.

    The implementation calls
    :func:`vo_rag.core.retrieval.execution.execute_retrieval` directly
    instead of making an HTTP round-trip.

    filters: payload filters; scalar = exact, list = any-of,
    {"gte": n, "lte": m} = numeric range (e.g. page_number).

    The ``top_k`` parameter is clamped to ``[_MCP_DEFAULT_TOP_K,
    _MCP_TOP_K_CEILING]`` at the boundary so a model caller cannot
    ask for thousands of hits and blow the context window.
    """
    args = McpQueryArgs(
        query=query,
        collections=collections,
        top_k=top_k if top_k else _MCP_DEFAULT_TOP_K,
        mode=mode,
        filters=filters,
        rerank=rerank,
        visual_mode=visual_mode,
    )

    retriever = getattr(services, "retriever", None)
    if retriever is None:
        raise ToolError(
            "retriever_unavailable: services.retriever is not configured",
            code="retriever_unavailable",
        )

    try:
        resolved_mode = QueryMode(args.mode)
    except ValueError as exc:
        raise ToolError(
            f"invalid_mode: {args.mode!r} is not one of {[m.value for m in QueryMode]}",
            code="invalid_mode",
        ) from exc

    try:
        validate_filters(args.filters)
    except ValueError as exc:
        raise ToolError(
            f"invalid_filters: {exc}",
            code="invalid_filters",
        ) from exc

    effective_top_k = min(max(args.top_k, 1), _MCP_TOP_K_CEILING)

    rerank_fn: Any = None
    if args.rerank and getattr(services, "reranker", None) is not None:
        # Preserve the execution layer's scored-pool bound; final result
        # selection applies the MCP limit after evidence redundancy removal.
        base_rerank = services.reranker.rerank

        def _rerank(
            query: str, candidates: list[SearchResult], *, top_k: int = effective_top_k
        ) -> Any:
            # base_rerank may be sync (GteReranker/GpuReranker) or async
            # (RerankerClient); forward the awaitable untouched (execute_retrieval
            # awaits it) and only coerce the sync result to a list.
            out = base_rerank(query, candidates, top_k=top_k)
            return out if inspect.isawaitable(out) else list(out)

        rerank_fn = _rerank

    hits: list[SearchResult] = await execute_retrieval(
        retriever=retriever,
        query=args.query,
        mode=resolved_mode,
        requested_collections=args.collections,
        top_k=effective_top_k,
        filters=args.filters,
        visual_mode=args.visual_mode,
        rerank_fn=rerank_fn,
    )

    result_hits = [_hit_from_search_result(hit) for hit in hits]
    warnings: list[str] = []
    if args.rerank and getattr(services, "reranker", None) is None:
        warnings.append("reranker_unavailable: rerank=True requested but no reranker is configured")

    return McpQueryResult(
        results=result_hits,
        mode=resolved_mode.value,
        top_k=effective_top_k,
        reranked=bool(rerank_fn),
        warnings=warnings,
    )


def _hit_from_search_result(result: SearchResult) -> McpQueryHit:
    """Map a core ``SearchResult`` to an ``McpQueryHit`` without
    mutating ids / score / collection. Title / source / page /
    image are pulled from the Qdrant payload using the same field
    names the TS tool renders.
    """
    payload = result.payload or {}
    page_raw = payload.get("page_number")
    return McpQueryHit(
        id=str(result.id),
        score=float(result.score),
        collection=str(result.collection),
        title=_first_str(payload.get("section_title"), payload.get("title")),
        source=_first_str(payload.get("source_title"), payload.get("source")),
        page=int(page_raw) if isinstance(page_raw, (int, float)) else None,
        image=_first_str(payload.get("image_ref"), payload.get("image_url")),
        payload=hit_payload(payload),
    )


def _first_str(*candidates: Any) -> str | None:
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def render_query_text(hits: list[SearchResult]) -> str:
    """Render ``hits`` as a human-readable text block.

    Each hit gets a rank, score, title, source, page, snippet, and
    optional image line. Returns ``"No results."`` for an empty input.
    """
    if not hits:
        return "No results."

    blocks: list[str] = []
    brain_packet = any(hit.collection == "vo_rag_brain_v1" for hit in hits)
    used = 0
    withheld = 0
    for index, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        title = _first_str(payload.get("section_title"), payload.get("title")) or hit.collection
        source = _first_str(payload.get("source_title"), payload.get("source")) or hit.collection
        page = payload.get("page_number")
        page_str = f" (page {page})" if page not in (None, "") else ""
        snippet = (
            _first_str(payload.get("text"), payload.get("content"), payload.get("content_text"))
            or ""
        )
        brain_context = hit.collection == "vo_rag_brain_v1"
        if brain_context:
            snippet = _first_str(payload.get("content")) or ""
            if not brain_evidence_is_complete(payload, hit.collection):
                snippet = (
                    "[Complete governing evidence unavailable or unattested; "
                    "no source prefix shown.]"
                )
            path = payload.get("path") or payload.get("source_id") or "(missing path)"
            revision = (
                payload.get("source_revision") or payload.get("commit") or "(missing revision)"
            )
            source = f"{hit.collection} {path}@{revision}"

        else:
            snippet = snippet[:500]

        lines = [
            f"#{index} [{hit.score:.3f}] {title}",
            f"  Source: {source}{page_str}",
            f"  {snippet}" if snippet else "  (no snippet)",
        ]
        if brain_context:
            for field in ("source_origin", "status", "updated", "url", "fetched_at"):
                value = payload.get(field)
                if value is not None:
                    lines.append(f"  {field}: {value}")
        image = _first_str(payload.get("image_ref"), payload.get("image_url"))
        if image:
            lines.append(f"  Image: {image}")
        block = "\n".join(lines)
        if brain_packet and used + len(block) + 2 > 65_280:
            withheld += 1
            notice = (
                "[Complete evidence withheld from bounded text view; "
                "full evidence remains in structuredContent.]"
            )
            lines[2] = "  " + notice
            block = "\n".join(lines)
            if used + len(block) + 2 > 65_280:
                continue
        blocks.append(block)
        used += len(block) + 2
    if withheld:
        blocks.append(
            f"Complete evidence text withheld for {withheld} result(s); "
            "consult structuredContent. No partial source prefixes shown."
        )
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# vo_rag_ingest
# --------------------------------------------------------------------------- #
async def vo_rag_ingest_impl(
    *,
    services: Any,
    source_type: str,
    content: str,
    metadata: dict[str, Any] | None = None,
    sovereignty: str = Sovereignty.CLOUD_OK.value,
) -> McpIngestResult:
    """Build an ``IntakeSubmission``, run SSRF on URLs, normalize, submit.

    Returns an :class:`McpIngestResult` with ``dispatch_status``
    either ``"submitted"`` (an ``ingest_submit`` or DB adapter was
    found in ``services``) or ``"not_configured"`` (no dispatcher
    is wired — the envelope is still built and its id is returned
    so the caller can refer to it when a dispatcher is added).
    """
    args = McpIngestArgs(
        source_type=source_type,
        content=content,
        metadata=metadata or {},
        sovereignty=sovereignty,
    )

    try:
        resolved_source = IntakeSource(args.source_type)
    except ValueError as exc:
        valid = [s.value for s in IntakeSource]
        raise ToolError(
            f"invalid_source_type: {args.source_type!r} is not one of {valid}",
            code="invalid_source_type",
        ) from exc

    try:
        resolved_sovereignty = Sovereignty(args.sovereignty)
    except ValueError as exc:
        valid = [s.value for s in Sovereignty]
        raise ToolError(
            f"invalid_sovereignty: {args.sovereignty!r} is not one of {valid}",
            code="invalid_sovereignty",
        ) from exc

    # F-014: gate URL content before doing any further work. An
    # unsafe URL aborts with a tool error BEFORE an envelope is
    # created.
    if args.source_type in _URL_SOURCES:
        try:
            await validate_external_url(args.content)
        except SSRFBlocked as exc:
            raise ToolError(
                f"ssrf_blocked: {exc.reason}",
                code="ssrf_blocked",
            ) from exc

    raw_content = _raw_content_for(resolved_source, args.content, args.metadata)
    source_data: dict[str, Any] = {}
    if "pipeline_hint" in args.metadata:
        source_data["pipeline_hint"] = args.metadata["pipeline_hint"]

    submission = IntakeSubmission(
        source_type=resolved_source,
        source_data=source_data,
        raw_content=raw_content,
        metadata=dict(args.metadata),
        sovereignty=resolved_sovereignty,
        priority=Priority.MEDIUM,
    )

    envelope = normalize(submission)

    dispatch_status, extra = await _dispatch(services, envelope)

    warnings: list[str] = []
    if dispatch_status == "not_configured":
        warnings.append("ingest_dispatcher_unconfigured: envelope was normalized but not submitted")
    elif dispatch_status == "dispatch_outcome_unrecognized":
        outcome_type = str(extra.get("outcome_type", "unknown"))
        warnings.append(f"ingest_dispatch_outcome_unrecognized: dispatcher returned {outcome_type}")

    return McpIngestResult(
        envelope_id=envelope.id,
        status=_enum_str(envelope.status),
        source_type=_enum_str(envelope.source_type),
        dispatch_status=dispatch_status,
        warnings=warnings,
    )


def _enum_str(value: Any) -> str:
    return str(value.value if hasattr(value, "value") else value)


def _raw_content_for(
    source_type: IntakeSource,
    content: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Pack the flat ``content`` field into the core ``raw_content`` dict.

    Callers send the same flat ``content`` string and ``source_type``
    regardless of which surface answers.

    Optional fields (``title``, ``filename``) are only added when
    the caller actually supplied a value — a None must never
    appear in the output, both because the core ``normalize``
    treats absent keys and explicit-None keys differently and
    because a clean shape is easier to test.
    """
    title = metadata.get("title") if isinstance(metadata.get("title"), str) else None
    filename = metadata.get("filename") if isinstance(metadata.get("filename"), str) else None
    out: dict[str, Any]
    if source_type is IntakeSource.URL:
        out = {"url": content}
        if title is not None:
            out["title"] = title
        return out
    if source_type is IntakeSource.GITHUB_STAR:
        return {"url": content}
    if source_type is IntakeSource.TEXT_PROMPT:
        out = {"text": content}
        if title is not None:
            out["title"] = title
        return out
    if source_type is IntakeSource.FILE_DROP:
        out = {"garage_key": content}
        if filename is not None:
            out["filename"] = filename
        return out
    # API: opaque pass-through, content is the caller's raw payload
    # string. Keep it as a dict so the core ``normalize`` step is
    # shape-stable.
    return {"payload": content}


async def _dispatch(services: Any, envelope: Any) -> tuple[str, dict[str, Any]]:
    """Submit the envelope through the configured dispatcher, or
    return ``("not_configured", {})`` when none is wired.

    The dispatcher precedence:

    1. ``services.ingest_submit(envelope)`` — a callable injected
       directly on the dataclass. Used by tests + by the
       production wiring.
    2. ``services.extra["ingest_submit"](envelope)`` — the
       ``SurfaceServices.extra`` dict is the seam for an envelope
       dispatcher service.

    Both paths return a ``{"status": ..., "envelope_id": ...}``-
    shaped dict. The dispatch is allowed to return its own
    ``status`` (e.g. ``"queued"``) which the result echoes back.
    """
    submit = _service_callback(services, "ingest_submit")

    if submit is None:
        return "not_configured", {}

    outcome = submit(envelope)
    if inspect.isawaitable(outcome):
        outcome = await outcome
    if not isinstance(outcome, dict):
        return "dispatch_outcome_unrecognized", {"outcome_type": type(outcome).__name__}

    return "submitted", dict(outcome)


# --------------------------------------------------------------------------- #
# vo_rag_status
# --------------------------------------------------------------------------- #
async def vo_rag_status_impl(
    *,
    services: Any,
    envelope_id: str | None = None,
) -> McpStatusResult:
    """Return system status (no id) or envelope status (with id).

    The system-status path delegates to ``services.status_provider``
    when one is configured; otherwise it falls back to a minimal
    slot-availability rollup built from the service container.
    The envelope-status path delegates to
    ``services.envelope_lookup`` when one is configured; when no
    lookup is available, the tool returns ``"not_configured"`` so
    the caller knows the surface is not yet wired to a
    persistence layer.
    """
    args = McpStatusArgs(envelope_id=envelope_id)

    if args.envelope_id is None:
        system = await _system_status(services)
        warnings = list(system.pop("warnings", []) or [])
        return McpStatusResult(scope="system", system=system, warnings=warnings)

    envelope = await _lookup_envelope(services, args.envelope_id)
    if envelope is None:
        raise ToolError(
            f"envelope_not_found: no envelope with id {args.envelope_id!r}",
            code="envelope_not_found",
        )

    return McpStatusResult(
        scope="envelope",
        envelope=_envelope_view(envelope),
        warnings=[],
    )


async def _system_status(services: Any) -> dict[str, Any]:
    """Build the system status payload.

    The provider, if any, is the primary source. We layer the
    service-container slot-availability rollup on top so the
    response always carries the per-slot ``ok``/``error`` shape
    an operator-side client renders.
    """
    provider = _service_callback(services, "status_provider")

    if provider is not None:
        payload = provider()
        if inspect.isawaitable(payload):
            payload = await payload
        if isinstance(payload, dict):
            return dict(payload)

    # Fallback rollup — the service container's slot availability.
    slot_names = (
        "db_pool",
        "qdrant",
        "garage",
        "embedding_client",
        "retriever",
        "reranker",
        "answer_synthesizer",
        "cloud_cost_tracker",
        "kill_switch",
    )
    services_block: dict[str, dict[str, Any]] = {}
    reasons = {
        "reranker_unavailable_reason": "reranker",
        "db_pool_unavailable_reason": "db_pool",
        "garage_unavailable_reason": "garage",
        "cloud_cost_unavailable_reason": "cloud_cost_tracker",
        "kill_switch_unavailable_reason": "kill_switch",
        "qdrant_unavailable_reason": "qdrant",
    }
    for slot in slot_names:
        ok = getattr(services, slot, None) is not None
        block: dict[str, Any] = {"ok": ok}
        for reason_attr, reason_slot in reasons.items():
            if reason_slot == slot:
                reason = getattr(services, reason_attr, "") or ""
                if reason:
                    block["error"] = reason
                    block["ok"] = False
        services_block[slot] = block

    version = ""
    settings = getattr(services, "settings", None)
    if settings is not None:
        version = str(getattr(settings, "version", "") or "")
    return {
        "service": "vo-rag",
        "version": version,
        "ok": all(block["ok"] for block in services_block.values()),
        "services": services_block,
        "warnings": [
            "readiness reports dependency availability only; no live service probe was configured"
        ],
    }


async def _lookup_envelope(services: Any, envelope_id: str) -> Any:
    lookup = _service_callback(services, "envelope_lookup")
    if lookup is None:
        return None
    outcome = lookup(envelope_id)
    if inspect.isawaitable(outcome):
        outcome = await outcome
    return outcome


def _service_callback(services: Any, name: str) -> Any:
    """Return one callback from the service interface or legacy extras map."""
    callback = getattr(services, name, None)
    if callback is not None:
        return callback
    extra = getattr(services, "extra", {}) or {}
    return extra.get(name) if isinstance(extra, dict) else None


def _envelope_view(envelope: Any) -> McpEnvelopeView:
    def _value(value: Any) -> str:
        return str(value.value if hasattr(value, "value") else value)

    if not hasattr(envelope, "id") or not hasattr(envelope, "status"):
        raise ToolError(
            "envelope_shape_invalid: envelope lookup returned an invalid envelope object",
            code="envelope_shape_invalid",
        )

    created_at = getattr(envelope, "created_at", None) or datetime.now(UTC)
    updated_at = getattr(envelope, "updated_at", None) or created_at
    return McpEnvelopeView(
        envelope_id=str(envelope.id),
        status=_value(envelope.status),
        source_type=_value(getattr(envelope, "source_type", "")),
        priority=_value(getattr(envelope, "priority", Priority.MEDIUM)),
        sovereignty=_value(getattr(envelope, "sovereignty", Sovereignty.CLOUD_OK)),
        created_at=created_at,
        updated_at=updated_at,
    )


# --------------------------------------------------------------------------- #
# vo_rag_ingest_stream
# --------------------------------------------------------------------------- #
async def vo_rag_ingest_stream_impl(
    *,
    services: Any,
    envelope_id: str,
    ctx: Any,
) -> McpIngestResult:
    """Stream ingest progress for ``envelope_id`` and report via ``ctx``.

    This is not an attach-only status view: it executes the BookRag ingest
    pipeline in-process and forwards
    :class:`~vo_rag.pipelines.book_rag.events.IngestProgress` events through
    FastMCP's ``report_progress`` mechanism. Any LLM calls are the pipeline's
    governed calls, not a separate MCP completion path.
    """
    from vo_rag.core.db.queries.envelopes import fail_envelope, get_envelope, update_envelope_status

    # Lazy import: stream execution may load the optional book_rag runtime, but
    # importing this MCP tools module must stay lightweight. The module-level
    # ``stream_book_rag_ingest`` proxy remains the monkeypatch seam for tests.
    from vo_rag.pipelines.book_rag.events import INGEST_STEP_ORDER, IngestProgress
    from vo_rag.pipelines.book_rag.workflow import PipelineOutput

    pool = getattr(services, "db_pool", None)
    if pool is None:
        raise ToolError(
            "db_pool_unavailable: cannot stream ingest without a database pool",
            code="db_pool_unavailable",
        )

    envelope = await get_envelope(pool, envelope_id)
    if envelope is None:
        raise ToolError(
            f"envelope_not_found: no envelope with id {envelope_id!r}",
            code="envelope_not_found",
        )

    total_steps = len(INGEST_STEP_ORDER)
    terminal_output: Any | None = None

    try:
        async for ev in stream_book_rag_ingest(
            envelope,
            settings=cast("Settings", getattr(services, "settings", None)),
            pg_pool=pool,
        ):
            if isinstance(ev, IngestProgress):
                if ev.step in INGEST_STEP_ORDER:
                    step_index = INGEST_STEP_ORDER.index(ev.step)
                else:
                    step_index = total_steps
                await ctx.report_progress(
                    step_index,
                    total_steps,
                    message=f"{ev.step}: {ev.detail}",
                )
            elif isinstance(ev, PipelineOutput):
                terminal_output = ev
    except asyncio.CancelledError as exc:
        drained_output = getattr(exc, "terminal_output", None)
        if isinstance(drained_output, PipelineOutput):
            terminal_output = drained_output
            try:
                await update_envelope_status(
                    pool,
                    str(getattr(envelope, "id", envelope_id)),
                    terminal_output.status,
                )
            except Exception as persist_exc:
                # Keep this aligned with the SSE sibling
                # (book_rag_sse_success_persist_failed): a persistence hiccup
                # on the drained-success write must not replace the
                # cancellation flow with a DB exception.
                log.error(
                    "mcp_ingest_drained_persist_failed",
                    envelope_id=str(getattr(envelope, "id", envelope_id)),
                    status=_enum_str(terminal_output.status),
                    error=str(persist_exc),
                    exc_info=True,
                )
        else:
            await _fail_envelope_best_effort(
                fail_envelope,
                pool,
                str(getattr(envelope, "id", envelope_id)),
                _exception_error_text(exc),
            )
        raise
    except Exception as exc:
        await _fail_envelope_best_effort(
            fail_envelope,
            pool,
            str(getattr(envelope, "id", envelope_id)),
            _exception_error_text(exc),
        )
        raise

    if terminal_output is None:
        error = "ingest_stream_finished_without_output"
        await _fail_envelope_best_effort(
            fail_envelope,
            pool,
            str(getattr(envelope, "id", envelope_id)),
            error,
        )
        raise ToolError(error, code="stream_failed")

    await update_envelope_status(
        pool,
        str(getattr(envelope, "id", envelope_id)),
        terminal_output.status,
    )

    return McpIngestResult(
        envelope_id=str(getattr(envelope, "id", "")),
        status=_enum_str(terminal_output.status),
        source_type=_enum_str(getattr(envelope, "source_type", "")),
        dispatch_status="streamed",
        warnings=[
            f"{item.get('stage', 'quality')}:{item.get('code', 'unknown')}"
            for item in terminal_output.quality_warnings
        ],
    )


__all__ = [
    "render_query_text",
    "vo_rag_ingest_impl",
    "vo_rag_ingest_stream_impl",
    "vo_rag_query_impl",
    "vo_rag_status_impl",
]

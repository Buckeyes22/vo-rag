"""Tests for the Python MCP surface tools.

The tool functions in ``vo_rag.mcp.tools`` are the implementation
seam — they call the landed core retrieval / ingest functions
directly through an injected ``SurfaceServices``-like object (no
HTTP round-trip to the API). The FastMCP server in
``vo_rag.mcp.server`` closes over these functions and registers
them under four tool names: ``vo_rag_query``, ``vo_rag_ingest``,
``vo_rag_ingest_stream``, ``vo_rag_status``.

These tests pin the contract that:

- ``vo_rag_query`` clamps ``top_k`` to 50, applies the mode-based
  collection resolver, threads the rerank hook, and preserves
  ``SearchResult`` ids / collection / score / title / source / page
  / image fields in both the structured result and the rendered
  text.
- ``vo_rag_ingest`` builds an ``IntakeSubmission`` from the flat
  ``content`` field, runs the SSRF gate for ``url`` source types
  (rejecting unsafe URLs WITHOUT creating an envelope — F-014),
  normalizes via ``core.ingest.normalize``, and either submits
  through ``services.ingest_submit`` (or DB adapter) or returns
  ``dispatch_status="not_configured"`` explicitly when no
  dispatcher is configured.
- ``vo_rag_status`` returns the system status when no envelope id
  is given and the envelope status when one is.
- Tool errors are redacted: no API keys, no Garage secrets, no
  stack traces, no raw settings payloads.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from tests.mcp._fakes import (
    FakeReranker,
    FakeRetriever,
    make_fake_services,
    make_intake_envelope,
    make_search_result,
)

pytestmark = [pytest.mark.asyncio]


# --------------------------------------------------------------------------- #
# Query tool
# --------------------------------------------------------------------------- #
async def test_query_calls_execute_retrieval_with_clamps_and_rerank() -> None:
    """vo_rag_query must clamp top_k to 50, pass mode / collections / filters
    through, and call services.reranker.rerank when rerank=True."""
    from vo_rag.mcp.tools import vo_rag_query_impl

    retriever = FakeRetriever(scripted=[make_search_result()])
    reranker = FakeReranker()
    services = make_fake_services(retriever=retriever, reranker=reranker)

    result = await vo_rag_query_impl(
        services=services,
        query="trail braking",
        collections=["vo_rag_books_raw_v1"],
        top_k=999,  # over the 50 ceiling
        mode="internal",
        filters={"source_id": "book-A"},
        rerank=True,
        visual_mode="baseline",
    )

    # top_k clamped to 50 before it reached the retriever
    assert retriever.calls, "retriever was not called"
    call = retriever.calls[0]
    assert call["top_k"] == 400
    assert call["query"] == "trail braking"
    assert call["collections"] == ["vo_rag_books_raw_v1"]
    assert call["filters"] == {"source_id": "book-A"}
    assert call["visual_mode"] == "baseline"

    # rerank hook called
    assert reranker.calls == [{"query": "trail braking", "n": 1, "top_k": 200}]

    # structured result preserves id / collection / score
    assert result.results
    assert result.results[0].id == "point-1"
    assert result.results[0].collection == "vo_rag_books_raw_v1"
    assert result.results[0].score == pytest.approx(0.91)


async def test_query_default_top_k_is_5_and_clamps_too_small_inputs() -> None:
    """When top_k is omitted, default is 5; when zero/None, we coerce
    to 5; when over 50, clamp to 50."""
    from vo_rag.mcp.tools import vo_rag_query_impl

    services = make_fake_services(retriever=FakeRetriever(), reranker=FakeReranker())

    await vo_rag_query_impl(services=services, query="x")
    assert services.retriever.calls[0]["top_k"] == 100

    await vo_rag_query_impl(services=services, query="x", top_k=0)
    assert services.retriever.calls[1]["top_k"] == 100

    await vo_rag_query_impl(services=services, query="x", top_k=None)  # type: ignore[arg-type]
    assert services.retriever.calls[2]["top_k"] == 100

    await vo_rag_query_impl(services=services, query="x", top_k=51)
    assert services.retriever.calls[3]["top_k"] == 400


async def test_query_rerank_false_skips_rerank_hook() -> None:
    """When rerank=False the rerank hook must not be invoked."""
    from vo_rag.mcp.tools import vo_rag_query_impl

    reranker = FakeReranker()
    services = make_fake_services(retriever=FakeRetriever(), reranker=reranker)

    await vo_rag_query_impl(services=services, query="x", rerank=False)
    assert reranker.calls == []


async def test_query_rerank_with_no_reranker_does_not_explode() -> None:
    """When services.reranker is None and rerank=True, the tool
    silently degrades to the unranked result list (matching the API
    surface behavior in ``get_rerank_fn``)."""
    from vo_rag.mcp.tools import vo_rag_query_impl

    services = make_fake_services(retriever=FakeRetriever(scripted=[make_search_result()]))
    services.reranker = None

    result = await vo_rag_query_impl(services=services, query="x")
    assert len(result.results) == 1


async def test_query_text_rendering_preserves_title_source_page_image() -> None:
    """The text content returned alongside the structured result must
    preserve title, source, page, and image fields if present
    (the query tool's documented output contract)."""
    from vo_rag.mcp.tools import render_query_text

    result_obj = make_search_result(
        point_id="point-9",
        score=0.88,
        title="Trail Braking Deep Dive",
        source="Corner Craft",
        page=42,
        image="books/speed-secrets/fig-3.png",
    )
    text = render_query_text([result_obj])

    assert "[0.880]" in text
    assert "Trail Braking Deep Dive" in text
    assert "Corner Craft" in text
    assert "page 42" in text
    assert "books/speed-secrets/fig-3.png" in text


async def test_query_text_rendering_handles_empty_results() -> None:
    from vo_rag.mcp.tools import render_query_text

    assert render_query_text([]) == "No results."


async def test_query_retriever_unavailable_raises_tool_error() -> None:
    """When services.retriever is None, the tool must raise — never
    fabricate results."""
    from vo_rag.mcp.errors import ToolError
    from vo_rag.mcp.tools import vo_rag_query_impl

    services = make_fake_services()
    services.retriever = None

    with pytest.raises(ToolError) as excinfo:
        await vo_rag_query_impl(services=services, query="x")
    assert "retriever_unavailable" in str(excinfo.value)


async def test_query_rejects_a_malformed_filter_with_tool_error() -> None:
    """A filter dict ``_build_filter`` cannot translate (e.g. ``{"between":
    [...]}}``) must raise ``ToolError`` before the retriever is ever called —
    the MCP surface has to validate filters the same way the HTTP API does
    (VOR-1443 r3)."""
    from vo_rag.mcp.errors import ToolError
    from vo_rag.mcp.tools import vo_rag_query_impl

    retriever = FakeRetriever(scripted=[make_search_result()])
    services = make_fake_services(retriever=retriever)

    with pytest.raises(ToolError) as excinfo:
        await vo_rag_query_impl(
            services=services,
            query="x",
            filters={"tf_pub_year": {"between": [1, 2]}},
        )
    assert "invalid_filters" in str(excinfo.value)
    assert retriever.calls == []


# --------------------------------------------------------------------------- #
# Ingest tool
# --------------------------------------------------------------------------- #
async def test_ingest_url_runs_ssrf_gate_and_normalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """vo_rag_ingest(source_type='url', content='https://example.com/x') must
    (a) call validate_external_url, (b) build an IntakeSubmission, (c)
    normalize it, and (d) submit through services.ingest_submit."""
    from vo_rag.core.models.envelope import IntakeSource
    from vo_rag.mcp.tools import vo_rag_ingest_impl

    submitted: list[Any] = []

    def _submit(envelope: Any) -> dict[str, str]:
        submitted.append(envelope)
        return {"envelope_id": envelope.id, "status": "received", "dispatch_status": "submitted"}

    services = make_fake_services(ingest_submit=_submit)

    # Patch the URL safety gate so the test is hermetic.
    resolved = SimpleNamespace(
        url="https://example.com/x",
        scheme="https",
        host="example.com",
        ip="93.184.216.34",
        port=443,
    )
    monkeypatch.setattr(
        "vo_rag.mcp.tools.validate_external_url",
        lambda url: _async_return(resolved),
    )

    result = await vo_rag_ingest_impl(
        services=services,
        source_type="url",
        content="https://example.com/x",
        metadata={"title": "Example"},
    )

    assert submitted, "ingest_submit was not called"
    envelope = submitted[0]
    assert envelope.source_type is IntakeSource.URL
    assert envelope.raw_content == {"url": "https://example.com/x", "title": "Example"}
    assert result.envelope_id == envelope.id
    assert result.status == "received"
    assert result.dispatch_status == "submitted"


async def test_ingest_text_prompt_uses_text_raw_content() -> None:
    """Text-prompt ingest packs content into raw_content.text."""
    from vo_rag.mcp.tools import vo_rag_ingest_impl

    submitted: list[Any] = []

    def _submit(envelope: Any) -> dict[str, str]:
        submitted.append(envelope)
        return {"envelope_id": envelope.id, "status": "received", "dispatch_status": "submitted"}

    services = make_fake_services(ingest_submit=_submit)

    result = await vo_rag_ingest_impl(
        services=services,
        source_type="text_prompt",
        content="a self-driving question",
    )

    assert submitted
    # normalize() preserves the caller-supplied text and may add a
    # ``title`` key — what we pin is the contract that the operator's
    # text round-trips into raw_content["text"].
    assert submitted[0].raw_content["text"] == "a self-driving question"
    assert result.dispatch_status == "submitted"


async def test_ingest_file_drop_uses_garage_key() -> None:
    from vo_rag.mcp.tools import vo_rag_ingest_impl

    submitted: list[Any] = []

    def _submit(envelope: Any) -> dict[str, str]:
        submitted.append(envelope)
        return {"envelope_id": envelope.id, "status": "received", "dispatch_status": "submitted"}

    services = make_fake_services(ingest_submit=_submit)

    result = await vo_rag_ingest_impl(
        services=services,
        source_type="file_drop",
        content="uploads/abc/manifest.json",
    )

    assert submitted
    assert submitted[0].raw_content["garage_key"] == "uploads/abc/manifest.json"
    assert result.dispatch_status == "submitted"


async def test_ingest_no_dispatcher_returns_not_configured() -> None:
    """When no ingest_submit is injected, the tool still normalizes
    the envelope and returns dispatch_status='not_configured'."""
    from vo_rag.mcp.tools import vo_rag_ingest_impl

    services = make_fake_services()  # no ingest_submit
    result = await vo_rag_ingest_impl(
        services=services,
        source_type="text_prompt",
        content="hello",
    )
    assert result.dispatch_status == "not_configured"
    assert result.envelope_id  # envelope was still built and normalized
    assert result.status == "received"


async def test_ingest_unsafe_url_raises_tool_error_no_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-014: unsafe URLs return a tool error WITHOUT creating an
    envelope and WITHOUT calling the dispatcher."""
    from vo_rag.mcp.errors import ToolError
    from vo_rag.mcp.tools import vo_rag_ingest_impl
    from vo_rag.web.url_safety import SSRFBlocked

    submitted: list[Any] = []

    def _submit(envelope: Any) -> dict[str, str]:
        submitted.append(envelope)
        return {"envelope_id": envelope.id, "status": "received", "dispatch_status": "submitted"}

    services = make_fake_services(ingest_submit=_submit)

    async def _raise_blocked(url: str) -> None:
        raise SSRFBlocked(url, "scheme 'javascript' not in ['http', 'https']")

    monkeypatch.setattr("vo_rag.mcp.tools.validate_external_url", _raise_blocked)

    attacker_url = "javascript:alert(1)"

    with pytest.raises(ToolError) as excinfo:
        await vo_rag_ingest_impl(
            services=services,
            source_type="url",
            content=attacker_url,
        )
    # redacted message — must not echo the raw attacker URL or a stack trace
    msg = str(excinfo.value)
    assert "ssrf_blocked" in msg
    assert "javascript:alert(1)" not in msg
    assert "Traceback" not in msg
    assert submitted == []


# --------------------------------------------------------------------------- #
# _dispatch hardening: async submitter support
# --------------------------------------------------------------------------- #
def fake_envelope(envelope_id: str) -> Any:
    return SimpleNamespace(id=envelope_id, envelope_id=envelope_id)


async def test_ingest_dispatch_awaits_async_submitter() -> None:
    """An async submit() coroutine must be awaited so its outcome is
    captured, not dropped as a coroutine object."""
    from vo_rag.mcp.tools import _dispatch

    calls: list[str] = []

    async def submit(envelope: Any) -> dict[str, Any]:
        calls.append(envelope.envelope_id)
        return {"status": "queued", "envelope_id": envelope.envelope_id}

    services = SimpleNamespace(ingest_submit=submit)

    status, outcome = await _dispatch(services, fake_envelope("env-1"))

    assert calls == ["env-1"]
    assert status == "submitted"
    assert outcome["status"] == "queued"


async def test_ingest_dispatch_accepts_sync_submitter() -> None:
    """Sync submitters must continue to work unchanged."""
    from vo_rag.mcp.tools import _dispatch

    def submit(envelope: Any) -> dict[str, Any]:
        return {"status": "queued"}

    status, outcome = await _dispatch(SimpleNamespace(ingest_submit=submit), fake_envelope("env-1"))

    assert status == "submitted"
    assert outcome == {"status": "queued"}


async def test_ingest_dispatch_rejects_unrecognized_submitter_outcome() -> None:
    """A falsy/non-dict submitter result must not be rendered as submitted."""
    from vo_rag.mcp.tools import _dispatch

    def submit(envelope: Any) -> bool:
        return False

    status, outcome = await _dispatch(SimpleNamespace(ingest_submit=submit), fake_envelope("env-1"))

    assert status == "dispatch_outcome_unrecognized"
    assert outcome == {"outcome_type": "bool"}


async def test_ingest_surfaces_unrecognized_dispatch_warning() -> None:
    from vo_rag.mcp.tools import vo_rag_ingest_impl

    def submit(envelope: Any) -> None:
        return None

    services = make_fake_services(ingest_submit=submit)

    result = await vo_rag_ingest_impl(
        services=services,
        source_type="text_prompt",
        content="hello",
    )

    assert result.dispatch_status == "dispatch_outcome_unrecognized"
    assert result.warnings == ["ingest_dispatch_outcome_unrecognized: dispatcher returned NoneType"]


async def test_ingest_result_validates_dispatch_status_vocabulary() -> None:
    from vo_rag.mcp.models import McpIngestResult

    for dispatch_status in ("submitted", "streamed", "not_configured", "custom_unrecognized"):
        result = McpIngestResult(
            envelope_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            status="received",
            source_type="text_prompt",
            dispatch_status=dispatch_status,
        )
        assert result.dispatch_status == dispatch_status

    with pytest.raises(ValueError, match="dispatch_status"):
        McpIngestResult(
            envelope_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            status="received",
            source_type="text_prompt",
            dispatch_status="silently_new_status",
        )


# --------------------------------------------------------------------------- #
# Status tool
# --------------------------------------------------------------------------- #
async def test_status_no_envelope_returns_system_status() -> None:
    """vo_rag_status() with no envelope_id returns the system status
    via the injected status_provider (or a default 'not_configured'
    envelope if none)."""
    from vo_rag.mcp.tools import vo_rag_status_impl

    def _provider() -> dict[str, Any]:
        return {
            "service": "vo-rag",
            "version": "0.1.0",
            "ok": True,
            "services": {"qdrant": {"ok": True}, "db_pool": {"ok": False}},
        }

    services = make_fake_services(status_provider=_provider)

    result = await vo_rag_status_impl(services=services, envelope_id=None)
    assert result.scope == "system"
    assert result.system is not None
    assert result.system["service"] == "vo-rag"
    assert result.system["services"]["qdrant"]["ok"] is True


async def test_status_no_envelope_awaits_async_status_provider() -> None:
    from vo_rag.mcp.tools import vo_rag_status_impl

    async def _provider() -> dict[str, Any]:
        return {
            "service": "vo-rag",
            "version": "0.1.0",
            "ok": False,
            "eval_gate": {"passed": False, "blocked_preconditions": {"book_rag": "blocked"}},
        }

    services = make_fake_services(status_provider=_provider)

    result = await vo_rag_status_impl(services=services, envelope_id=None)

    assert result.scope == "system"
    assert result.system is not None
    assert result.system["eval_gate"] == {
        "passed": False,
        "blocked_preconditions": {"book_rag": "blocked"},
    }


async def test_status_no_provider_falls_back_to_minimal_health() -> None:
    """When no status_provider is injected, the tool still returns a
    well-formed response with a minimal slot-availability rollup."""
    from vo_rag.mcp.tools import vo_rag_status_impl

    services = make_fake_services()
    result = await vo_rag_status_impl(services=services, envelope_id=None)
    assert result.scope == "system"
    assert result.system is not None
    # Minimal rollup must include the retriever/reranker/db_pool slots
    assert "services" in result.system
    assert "retriever" in result.system["services"]


async def test_status_no_provider_hoists_fallback_warning() -> None:
    from vo_rag.mcp.tools import vo_rag_status_impl

    services = make_fake_services()
    result = await vo_rag_status_impl(services=services, envelope_id=None)

    assert result.system is not None
    assert "warnings" not in result.system
    assert result.warnings == [
        "readiness reports dependency availability only; no live service probe was configured"
    ]


async def test_status_with_envelope_id_returns_envelope_status() -> None:
    """vo_rag_status(envelope_id=...) calls the injected envelope_lookup
    and returns the envelope view."""
    from vo_rag.core.models.common import ulid_new
    from vo_rag.core.models.envelope import IntakeSource
    from vo_rag.mcp.tools import vo_rag_status_impl

    envelope_id = ulid_new()  # valid Crockford-base32 ULID
    envelope = make_intake_envelope(
        envelope_id=envelope_id,
        source_type=IntakeSource.URL,
    )

    def _lookup(eid: str) -> Any:
        assert eid == envelope_id
        return envelope

    services = make_fake_services(envelope_lookup=_lookup)
    result = await vo_rag_status_impl(services=services, envelope_id=envelope_id)
    assert result.scope == "envelope"
    assert result.envelope is not None
    assert result.envelope.envelope_id == envelope_id


async def test_status_with_envelope_id_awaits_async_envelope_lookup() -> None:
    """Async envelope_lookup callables must be awaited before rendering
    the envelope view."""
    from vo_rag.core.models.common import ulid_new
    from vo_rag.mcp.tools import vo_rag_status_impl

    envelope_id = ulid_new()
    envelope = make_intake_envelope(envelope_id=envelope_id)
    calls: list[str] = []

    async def _lookup(eid: str) -> Any:
        calls.append(eid)
        return envelope

    services = make_fake_services(envelope_lookup=_lookup)

    result = await vo_rag_status_impl(services=services, envelope_id=envelope_id)

    assert calls == [envelope_id]
    assert result.scope == "envelope"
    assert result.envelope is not None
    assert result.envelope.envelope_id == envelope_id


async def test_status_with_invalid_envelope_shape_raises_tool_error() -> None:
    """Malformed envelope lookup results must fail closed instead of
    fabricating default envelope_id/status values."""
    from vo_rag.core.models.common import ulid_new
    from vo_rag.mcp.errors import ToolError
    from vo_rag.mcp.tools import vo_rag_status_impl

    services = make_fake_services(envelope_lookup=lambda _: SimpleNamespace())

    with pytest.raises(ToolError) as excinfo:
        await vo_rag_status_impl(services=services, envelope_id=ulid_new())

    assert "envelope_shape_invalid" in str(excinfo.value)


async def test_status_envelope_not_found_returns_not_found() -> None:
    from vo_rag.core.models.common import ulid_new
    from vo_rag.mcp.errors import ToolError
    from vo_rag.mcp.tools import vo_rag_status_impl

    services = make_fake_services(envelope_lookup=lambda _: None)
    with pytest.raises(ToolError) as excinfo:
        await vo_rag_status_impl(services=services, envelope_id=ulid_new())
    assert "envelope_not_found" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Tool error redaction
# --------------------------------------------------------------------------- #
async def test_tool_error_redacts_api_keys_and_secrets() -> None:
    """The error path must not leak api keys, Garage secrets, raw
    settings payloads, or stack traces back to the MCP client."""
    from vo_rag.mcp.errors import ToolError, redact_error_message

    leaky = (
        "Connection refused: postgres://user:sk_live_5f4dcc3b5aa765d61d8327deb882cf99"
        "@db.example.com/vorag api_key=sk-ANTHROPIC-1234 "
        "garage.secret_key=garage_secret_value settings.reranker.model_path=/etc/secret\n"
        "Traceback (most recent call last):\n  File '/srv/vo-rag/mcp/server.py', line 42\n"
        "RuntimeError: real cause"
    )
    redacted = redact_error_message(leaky)

    assert "sk_live_5f4dcc3b5aa765d61d8327deb882cf99" not in redacted
    assert "sk-ANTHROPIC-1234" not in redacted
    assert "garage_secret_value" not in redacted
    assert "Traceback" not in redacted
    assert "/srv/vo-rag" not in redacted

    # ToolError itself must carry the redacted message
    err = ToolError(leaky)
    assert "sk_live_5f4dcc3b5aa765d61d8327deb882cf99" not in str(err)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
async def _async_return(value: Any) -> Any:
    return value

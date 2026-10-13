"""Shared fakes + helpers for the Python MCP surface tests.

The MCP tool implementations in ``vo_rag.mcp.tools`` call
core retrieval and ingest seams directly through an injected
``SurfaceServices``-like object. The fakes here are minimal stand-ins
that satisfy the structural shape (duck-typed) and record every call
so assertions can pin behavior.

Why not subclass the real ``HybridRetriever``?  Subclassing would
import the qdrant-client + fastembed weight chain and pull in
asyncpg+ormconfig at import time. The MCP test suite runs in the
hermetic unit-test path (no live infra), so a duck-typed fake that
satisfies the call site is the smaller, faster, and more honest
choice.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from vo_rag.api.singletons import SurfaceServices
from vo_rag.core.models.envelope import (
    EnvelopeStatus,
    IntakeEnvelope,
    IntakeSource,
    Priority,
    Sovereignty,
)
from vo_rag.core.retrieval.search import SearchResult


@dataclass
class FakeRetriever:
    """Duck-typed stand-in for ``HybridRetriever``.

    Records every ``cross_collection_search`` call and returns a
    scripted list of ``SearchResult``. The MCP tool path only uses
    ``cross_collection_search`` (the public ``execute_retrieval``
    entry point), so we do not need to fake ``hybrid_search``.
    """

    scripted: list[SearchResult] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def cross_collection_search(
        self,
        query: str,
        collections: list[str],
        *,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        mode: Any = None,
        visual_mode: str | None = None,
    ) -> list[SearchResult]:
        self.calls.append(
            {
                "query": query,
                "collections": list(collections),
                "top_k": top_k,
                "filters": filters,
                "mode": mode,
                "visual_mode": visual_mode,
            }
        )
        return list(self.scripted)[:top_k]


@dataclass
class FakeReranker:
    """Duck-typed stand-in for ``GteReranker``.

    The MCP query tool calls ``rerank(query, candidates)`` when the
    caller passes ``rerank=True``. The fake just preserves order
    (records the call) and returns the input list unchanged — what
    matters for the test is the call happened, not the ranking.
    """

    calls: list[dict[str, Any]] = field(default_factory=list)

    def rerank(
        self, query: str, candidates: list[SearchResult], top_k: int = 20
    ) -> list[SearchResult]:
        self.calls.append({"query": query, "n": len(candidates), "top_k": top_k})
        return list(candidates)[:top_k]


def make_fake_services(
    *,
    retriever: FakeRetriever | None = None,
    reranker: FakeReranker | None = None,
    ingest_submit: Callable[[IntakeEnvelope], Any] | None = None,
    status_provider: Callable[[], dict[str, Any] | Awaitable[dict[str, Any]]] | None = None,
    envelope_lookup: Callable[[str], Any] | None = None,
) -> SurfaceServices:
    """Build a :class:`SurfaceServices` instance with fakes plugged in.

    ``ingest_submit``, ``status_provider``, and ``envelope_lookup`` are
    passed through the ``extra`` dict because the dataclass does not
    (yet) have dedicated slots for them, so the MCP surface works against
    the container's current shape.
    """
    extra: dict[str, Any] = {}
    if ingest_submit is not None:
        extra["ingest_submit"] = ingest_submit
    if status_provider is not None:
        extra["status_provider"] = status_provider
    if envelope_lookup is not None:
        extra["envelope_lookup"] = envelope_lookup

    return SurfaceServices(
        settings=_fake_settings(),
        retriever=retriever if retriever is not None else FakeRetriever(),
        reranker=reranker if reranker is not None else FakeReranker(),
        extra=extra,
    )


def _fake_settings() -> Any:
    """Minimal ``settings`` namespace sufficient for the MCP tool path.

    Only ``settings.api.api_key`` is read at HTTP-transport start
    (validation), not by the stdio default path or by the tool
    functions themselves. We build a namespace with the attribute
    tree the tool path expects to be importable.
    """

    class _Api:
        api_key = ""
        host = "127.0.0.1"
        port = 8200

    class _Settings:
        api = _Api()
        version = "0.1.0"

    return _Settings()


def make_search_result(
    *,
    point_id: str = "point-1",
    score: float = 0.91,
    collection: str = "vo_rag_books_raw_v1",
    title: str | None = "Chapter 1",
    source: str | None = "Book A",
    page: int | None = 4,
    image: str | None = None,
) -> SearchResult:
    """Build a ``SearchResult`` with the field shape the MCP tool text
    rendering is pinned to preserve.
    """
    payload: dict[str, Any] = {
        "chunk_id": "c-1",
        "source_id": "src-1",
    }
    if title is not None:
        payload["section_title"] = title
    if source is not None:
        payload["source_title"] = source
    if page is not None:
        payload["page_number"] = page
    if image is not None:
        payload["image_ref"] = image
    return SearchResult(id=point_id, score=score, collection=collection, payload=payload)


def make_intake_envelope(
    *,
    envelope_id: str | None = None,
    source_type: IntakeSource = IntakeSource.TEXT_PROMPT,
    raw_content: dict[str, Any] | None = None,
    status: EnvelopeStatus = EnvelopeStatus.RECEIVED,
    sovereignty: Sovereignty = Sovereignty.CLOUD_OK,
) -> IntakeEnvelope:
    """Build a valid ``IntakeEnvelope`` for ingest-tool tests.

    The default ``envelope_id`` is a freshly-minted Crockford-base32
    ULID. Callers may pin an explicit id (it must be 26 chars and
    not contain ``I`` / ``L`` / ``O`` / ``U``).
    """
    from vo_rag.core.models.common import ulid_new

    now = datetime(2026, 6, 3, tzinfo=UTC)
    return IntakeEnvelope(
        id=envelope_id if envelope_id is not None else ulid_new(),
        source_type=source_type,
        source_data={},
        raw_content=raw_content if raw_content is not None else {"text": "hello"},
        metadata={},
        status=status,
        priority=Priority.MEDIUM,
        sovereignty=sovereignty,
        created_at=now,
        updated_at=now,
    )


def find_tool(server: Any, name: str) -> Any:
    """Look up a registered tool on a FastMCP server instance.

    The MCP Python SDK stores tools on ``_tool_manager._tools``. The
    helper hides that implementation detail from the test bodies.
    """
    tool = server._tool_manager._tools.get(name)
    if tool is None:
        raise AssertionError(
            f"tool {name!r} not registered; have {list(server._tool_manager._tools)}"
        )
    return tool


def list_tool_names(server: Any) -> Iterable[str]:
    return list(server._tool_manager._tools)


__all__ = [
    "FakeReranker",
    "FakeRetriever",
    "find_tool",
    "list_tool_names",
    "make_fake_services",
    "make_intake_envelope",
    "make_search_result",
]

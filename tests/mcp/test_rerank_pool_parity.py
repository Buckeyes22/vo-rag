"""MCP preserves the ordinary scored pool before final evidence selection."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, cast

import pytest
from tests.mcp._fakes import FakeReranker, FakeRetriever, make_fake_services

from vo_rag.core.retrieval.execution import RerankFn, execute_retrieval
from vo_rag.core.retrieval.search import HybridRetriever, SearchResult
from vo_rag.mcp.tools import vo_rag_query_impl

ROOT = "# Device\n\nApplies only to Enterprise firmware 3."


def _hit(identifier: str, local: str, *, root: bool = False) -> SearchResult:
    heading = "(page top)" if root else identifier
    ancestors: list[dict[str, Any]] = (
        []
        if root
        else [
            {
                "text": ROOT,
                "source_id": "wiki/device.md",
                "collection": "vo_rag_brain_v1",
                "source_revision": "a" * 40,
                "section_occurrence": 0,
                "point_id": identifier,
                "content_hash": hashlib.sha256(ROOT.encode()).hexdigest(),
            }
        ]
    )
    title = "Device" if root else "Device / " + heading
    return SearchResult(
        identifier,
        1.0,
        "vo_rag_brain_v1",
        {
            "source_id": "wiki/device.md",
            "path": "wiki/device.md",
            "commit": "a" * 40,
            "title": "Device",
            "heading": heading,
            "source_section_text": local,
            "content": title + "\n\n" + "\n\n".join([*(a["text"] for a in ancestors), local]),
            "governing_context_version": 1,
            "governing_context_complete": True,
            "governing_context": ancestors,
            "status": "accepted",
            "edition": "Enterprise",
            "source_origin": "curated",
        },
    )


class AsyncReranker(FakeReranker):
    async def rerank(  # type: ignore[override]
        self, query: str, candidates: list[SearchResult], top_k: int = 20
    ) -> list[SearchResult]:
        return super().rerank(query, candidates, top_k)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_http_callback_and_mcp_keep_same_pool_before_final_redundancy(
    asynchronous: bool,
) -> None:
    children = [_hit(f"detail-{i}", f"## Detail {i}\n\nDistinct mechanism {i}.") for i in range(7)]
    effect = _hit("latency", "## Latency\n\nUp to 2.5x faster for regular traffic.")
    pool = [children[0], _hit("root", ROOT, root=True), *children[1:], effect]
    pool.extend(_hit(f"other-{i}", f"## Other {i}\n\nSeparate fact {i}.") for i in range(31))
    original = deepcopy(pool)
    retriever = FakeRetriever(scripted=pool)
    ranker = AsyncReranker() if asynchronous else FakeReranker()
    direct = await execute_retrieval(
        retriever=cast(HybridRetriever, retriever),
        query="latency and processing",
        requested_collections=["vo_rag_brain_v1"],
        top_k=8,
        rerank_fn=cast(RerankFn, ranker.rerank),
    )
    services = make_fake_services(retriever=retriever, reranker=ranker)
    native = await vo_rag_query_impl(
        services=services,
        query="latency and processing",
        collections=["vo_rag_brain_v1"],
        top_k=8,
        rerank=True,
    )
    assert [hit.id for hit in direct] == [hit.id for hit in native.results]
    assert [hit.id for hit in direct] == [*(f"detail-{i}" for i in range(7)), "latency"]
    assert native.top_k == 8 and len(native.results) == 8
    assert ranker.calls == [{"query": "latency and processing", "n": 40, "top_k": 40}] * 2
    assert [call["top_k"] for call in retriever.calls] == [100, 100]
    assert [hit.payload for hit in direct] == [hit.payload for hit in native.results]
    assert ROOT in native.results[-1].payload["content"]
    assert "regular traffic" in native.results[-1].payload["content"]
    assert pool == original

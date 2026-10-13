"""Brain ranking packets are distinct from complete source evidence (VOR-1448)."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import httpx
import pytest

from vo_rag.core.reranking.client import RerankerClient
from vo_rag.core.reranking.render import _render_candidate
from vo_rag.core.retrieval.search import SearchResult


def _brain(identifier: str, heading: str, local: str) -> SearchResult:
    ancestor = (
        "# API Performance\n\nThese features affect latency and throughput.\n\n"
        + "Additional governing details. " * 80
        + "Enterprise only; up to 2.5x, with regular traffic.\n\n"
    )
    return SearchResult(
        id=identifier,
        score=0.0,
        collection="vo_rag_brain_v1",
        payload={
            "title": "API Performance",
            "heading": heading,
            "source_section_text": local,
            "content": ancestor + local,
            "source_id": "wiki/api-performance.md",
            "source_revision": "a" * 40,
            "commit": "a" * 40,
            "source_origin": "curated wiki",
            "status": "accepted",
            "governing_context_complete": True,
            "governing_context_version": 1,
            "governing_context": [
                {
                    "text": ancestor,
                    "source_id": "wiki/api-performance.md",
                    "collection": "vo_rag_brain_v1",
                    "point_id": identifier,
                    "source_revision": "a" * 40,
                    "content_hash": hashlib.sha256(ancestor.encode()).hexdigest(),
                }
            ],
        },
    )


async def test_remote_wire_ranks_local_fact_and_returns_whole_governing_evidence() -> None:
    navigation = _brain("navigation", "Related Pages", "## Related Pages\n\n- [[responses]]")
    fast = _brain("fast", "Fast Mode", "## Fast Mode\n\nUp to 2.5x faster with regular traffic.")
    before = deepcopy([navigation, fast])

    def handler(request: httpx.Request) -> httpx.Response:
        frame = json.loads(request.content)
        by_id = {row["id"]: row["text"] for row in frame["candidates"]}
        assert (
            by_id["navigation"]
            == "API Performance\nRelated Pages\n## Related Pages\n\n- [[responses]]"
        )
        assert (
            by_id["fast"]
            == "API Performance\nFast Mode\n## Fast Mode\n\nUp to 2.5x faster with regular traffic."
        )
        assert "These features affect latency" not in by_id["navigation"]
        assert "Enterprise only" not in by_id["fast"]
        return httpx.Response(
            200,
            json={"results": [{"id": "fast", "score": 0.9}, {"id": "navigation", "score": 0.1}]},
        )

    client = RerankerClient("http://rank/rerank")
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://rank"
    )
    try:
        results = await client.rerank(
            "How do response schedules affect latency?", [navigation, fast], top_k=1
        )
    finally:
        await client.aclose()
    assert results[0].id == "fast"
    assert results[0].payload == before[1].payload
    assert "Enterprise only" in results[0].payload["content"]
    assert results[0].payload["governing_context"][0]["text"] in results[0].payload["content"]
    assert [navigation, fast] == before


@pytest.mark.parametrize("collection", ["vo_rag_books_raw_v1", "vo_rag_reddit_v1", "other"])
def test_local_field_does_not_change_other_corpus_rendering(collection: str) -> None:
    candidate = _brain("passage", "child", "local brain-like field")
    candidate.collection = collection
    candidate.payload.update(section_title="Existing section", text="Existing text")
    assert _render_candidate(candidate) == "Existing section\nExisting text"
    candidate.payload.pop("text")
    assert _render_candidate(candidate) == "Existing section\n" + candidate.payload["content"]


@pytest.mark.parametrize("local", [None, "", "  ", 42])
def test_legacy_brain_without_local_text_keeps_existing_rendering(local: object) -> None:
    candidate = _brain("legacy", "child", "local")
    candidate.payload["source_section_text"] = local
    candidate.payload.update(section_title="Legacy section", content="Complete legacy text")
    assert _render_candidate(candidate) == "Legacy section\nComplete legacy text"


def test_governing_heading_labels_survive_focal_rendering() -> None:
    candidate = _brain("leaf", "Enterprise / Limits", "## Limits\n\n7 days.")
    assert (
        _render_candidate(candidate) == "API Performance\nEnterprise / Limits\n## Limits\n\n7 days."
    )
    assert "Enterprise only" in candidate.payload["content"]

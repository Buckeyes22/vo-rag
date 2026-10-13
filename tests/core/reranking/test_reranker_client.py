"""RerankerClient tests (VOR-8) — mocked httpx transport, no live server."""

from __future__ import annotations

import httpx

from vo_rag.core.reranking.client import RerankerClient
from vo_rag.core.retrieval.search import SearchResult


def _mock(handler) -> httpx.AsyncClient:  # noqa: ANN001
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x")


async def test_client_posts_and_reorders() -> None:
    captured: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            200,
            json={
                "results": [
                    {"id": "b", "text": "", "payload": {"chunk_id": "cb"}, "score": 0.9},
                    {"id": "a", "text": "", "payload": {"chunk_id": "ca"}, "score": 0.3},
                ]
            },
        )

    client = RerankerClient("http://x/rerank")
    client._client = _mock(handler)
    cands = [
        SearchResult(
            id="a", score=0.0, collection="c", payload={"chunk_id": "ca", "content": "ta"}
        ),
        SearchResult(
            id="b", score=0.0, collection="c", payload={"chunk_id": "cb", "content": "tb"}
        ),
    ]
    out = await client.rerank("q", cands, top_k=2)
    assert [c.id for c in out] == ["b", "a"]
    assert out[0].score == 0.9
    assert out[0].payload["chunk_id"] == "cb"  # payload carried back by id
    assert b'"query"' in captured["body"]
    await client.aclose()


async def test_client_url_normalizes_trailing_rerank() -> None:
    client = RerankerClient("http://x/rerank")
    assert str(client._client.base_url).rstrip("/") == "http://x"
    await client.aclose()

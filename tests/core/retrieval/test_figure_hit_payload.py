"""A raw figure's embedding-only window never reaches a query hit (VOR-1452).

The raw book payload stores a figure chunk's neighbour window under
``embedding_text`` so rebuilds re-embed what ingest did. It duplicates the
prose of the neighbouring chunks, so the hit payloads of the API query and
answer routes and the MCP query tool must not carry it. The sterile
collection's ``embedding_text`` is sterilized text and stays as it was.
"""

from __future__ import annotations

from typing import Any

from qdrant_client.models import ScoredPoint

from vo_rag.api.routes.answer import _to_answer_hit
from vo_rag.api.schemas import QueryHit
from vo_rag.core.retrieval.execution import execute_retrieval
from vo_rag.core.retrieval.query_mode import QueryMode
from vo_rag.core.retrieval.search import HybridRetriever, SearchResult
from vo_rag.mcp.tools import _hit_from_search_result

WINDOW = "[Book > Throws]\n\nPreceding: The athlete starts low at the back of the circle."
RAW_FIGURE = {
    "chunk_id": "f-1",
    "chunk_type": "figure_composite",
    "stage": "raw",
    "content": "Caption: Figure 4.3 The glide.",
    "embedding_text": WINDOW,
    "image_ref": "s/img-1.png",
    "section_title": "Throws",
}
RAW_TEXT = {
    "chunk_id": "t-1",
    "chunk_type": "text",
    "stage": "raw",
    "content": "The glide is taught in three stages.",
    "section_title": "Throws",
}
STERILE_FIGURE = {
    "chunk_id": "sf-1",
    "chunk_type": "figure_composite",
    "content": "Sterile figure summary.",
    "embedding_text": "[Book > Throws]\n\nSterile window summary.",
    "section_title": "Throws",
}


class _Embed:
    async def embed(self, texts: list[str], **_: Any) -> dict[str, Any]:
        return {
            "dense": [[0.1, 0.2, 0.3]],
            "sparse": [{"indices": [1], "values": [0.5]}],
            "colbert": [[[0.7, 0.8]]],
        }


class _Store:
    def __init__(self, payloads: dict[str, list[dict[str, Any]]]) -> None:
        self.payloads = payloads

    async def search_hybrid(self, collection: str, *_: Any, **__: Any) -> list[ScoredPoint]:
        return [
            ScoredPoint(id=f"{collection}-{i}", score=1.0 - i / 10, payload=dict(p), version=0)
            for i, p in enumerate(self.payloads.get(collection, []))
        ]


async def _hits(
    mode: QueryMode, collection: str, payloads: list[dict[str, Any]]
) -> list[SearchResult]:
    retriever = HybridRetriever(
        qdrant=_Store({collection: payloads}),
        embedding_client=_Embed(),
    )
    return await execute_retrieval(
        retriever=retriever,
        query="glide",
        mode=mode,
        requested_collections=[collection],
        top_k=10,
    )


def _without_annotation(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if not k.startswith("_vo_rag_")}


def _by_chunk(payloads: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(p["chunk_id"]): p for p in payloads}


async def test_api_query_hit_of_a_raw_figure_has_no_embedding_text() -> None:
    hits = await _hits(QueryMode.INTERNAL, "vo_rag_books_raw_v1", [RAW_FIGURE, RAW_TEXT])
    payloads = _by_chunk([QueryHit.from_search_result(h).payload for h in hits])

    assert "embedding_text" not in payloads["f-1"]
    assert payloads["f-1"]["content"] == RAW_FIGURE["content"]
    assert payloads["f-1"]["image_ref"] == "s/img-1.png"
    # Text hits are unchanged (the retriever only adds its license annotation).
    assert _without_annotation(payloads["t-1"]) == RAW_TEXT


async def test_answer_hit_of_a_raw_figure_has_no_embedding_text() -> None:
    hits = await _hits(QueryMode.INTERNAL, "vo_rag_books_raw_v1", [RAW_FIGURE])
    answer_hits = [_to_answer_hit(h) for h in hits]
    assert answer_hits
    for hit in answer_hits:
        assert "embedding_text" not in hit.payload
        assert "Preceding:" not in str(hit.payload)


async def test_mcp_hit_of_a_raw_figure_has_no_embedding_text() -> None:
    hits = await _hits(QueryMode.INTERNAL, "vo_rag_books_raw_v1", [RAW_FIGURE, RAW_TEXT])
    payloads = _by_chunk([_hit_from_search_result(h).payload for h in hits])

    assert "embedding_text" not in payloads["f-1"]
    assert _without_annotation(payloads["t-1"]) == RAW_TEXT


async def test_sterile_hit_keeps_its_sterile_embedding_text() -> None:
    hits = await _hits(QueryMode.PUBLICATION, "vo_rag_books_sterile_v1", [STERILE_FIGURE])
    payloads = [QueryHit.from_search_result(h).payload for h in hits]
    assert payloads
    for payload in payloads:
        assert payload["embedding_text"] == STERILE_FIGURE["embedding_text"]


def test_answer_context_chunks_never_carry_the_raw_window() -> None:
    """RAPTOR and summary selective expansion turn raw hits into chunks for
    answer context; the raw figure window stays out of them too."""
    from vo_rag.core.raptor.persistence import _point_to_chunk as raptor_chunk
    from vo_rag.core.summary.selective_expansion import _point_to_chunk as expansion_chunk

    point = ScoredPoint(id="p", score=1.0, payload=dict(RAW_FIGURE), version=0)
    for convert in (raptor_chunk, expansion_chunk):
        chunk = convert(point)
        assert chunk.embedding_text == ""
        assert chunk.content == RAW_FIGURE["content"]

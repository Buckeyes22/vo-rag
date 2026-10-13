"""Wire-contract tests for the remote rerank schema (VOR-8)."""

from __future__ import annotations

from vo_rag.core.reranking.schemas import RerankCandidate, RerankRequest, RerankResponse


def test_request_roundtrips_through_json() -> None:
    req = RerankRequest(
        query="aero load transfer under braking",
        top_k=10,
        candidates=[RerankCandidate(id="p1", text="front axle...", payload={"chunk_id": "c1"})],
    )
    back = RerankRequest.model_validate_json(req.model_dump_json())
    assert back.query == req.query
    assert back.top_k == 10
    assert back.candidates[0].id == "p1"
    assert back.candidates[0].payload["chunk_id"] == "c1"


def test_candidate_defaults() -> None:
    c = RerankCandidate(id="x")
    assert c.text == ""
    assert c.payload == {}
    assert c.score is None


def test_response_preserves_order_and_scores() -> None:
    resp = RerankResponse(
        results=[
            RerankCandidate(id="p2", text="t2", payload={}, score=0.9),
            RerankCandidate(id="p1", text="t1", payload={}, score=0.4),
        ]
    )
    assert [c.id for c in resp.results] == ["p2", "p1"]
    assert resp.results[0].score == 0.9

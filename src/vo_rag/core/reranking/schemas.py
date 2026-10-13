"""Wire contract for the remote rerank endpoint (VOR-8).

Single source of truth shared by the inference server
(``vo_rag.workers.inference.server``) and ``RerankerClient``. A candidate
carries the id + payload + text the cross-encoder needs — ``_render_candidate``
in ``vo_rag.core.reranking.gte`` reads ``section_title`` + ``content``/``text``
off the payload, so the client sends the payload through and the server can
render identically to the in-process path.
"""

from __future__ import annotations

from vo_rag.core.models.common import VoBaseModel


class RerankCandidate(VoBaseModel):
    """One retrieval hit handed to / returned from the reranker."""

    id: str
    text: str = ""
    payload: dict[str, object] = {}
    score: float | None = None


class RerankRequest(VoBaseModel):
    """POST /rerank body: a query + the candidates to reorder."""

    query: str
    candidates: list[RerankCandidate]
    top_k: int = 10


class RerankResponse(VoBaseModel):
    """POST /rerank response: candidates in reranked order, with scores."""

    results: list[RerankCandidate]


__all__ = ["RerankCandidate", "RerankRequest", "RerankResponse"]

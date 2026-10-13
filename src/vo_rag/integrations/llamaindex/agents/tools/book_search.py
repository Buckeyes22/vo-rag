"""book_search tool — trusted INTERNAL-mode QueryEngineTool for the research agent.

SECURITY: This tool is the TRUSTED arm of the agent. It reads vetted internal
collections (raw books in INTERNAL mode) and therefore does NOT need the
protected wrapper. The factory must NEVER allow it to be aimed at an untrusted
scratch collection — mode and collection are locked to INTERNAL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llama_index.core.tools import QueryEngineTool

from vo_rag.core.answer.synthesizer import AnswerProfile, AnswerSynthesizer
from vo_rag.core.retrieval.execution import RerankFn
from vo_rag.core.retrieval.query_mode import QueryMode
from vo_rag.core.retrieval.search import HybridRetriever
from vo_rag.integrations.llamaindex.query_engine import build_vo_rag_query_engine
from vo_rag.integrations.llamaindex.reranker import VoRagReranker
from vo_rag.integrations.llamaindex.retriever import VoRagRetriever

if TYPE_CHECKING:
    pass

_BOOK_SEARCH_NAME = "book_search"

_BOOK_SEARCH_DESCRIPTION = (
    "Search the ingested corpus for passages relevant to a question. "
    "Returns ranked excerpts with source ids and page refs."
)


def build_book_search_tool(
    *,
    retriever: HybridRetriever,
    answer: AnswerSynthesizer,
    reranker: VoRagReranker | None = None,
    profile: AnswerProfile = AnswerProfile.QUALITY,
    top_k: int = 20,
    requested_collections: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    visual_mode: str | None = None,
    rerank_fn: RerankFn | None = None,
) -> QueryEngineTool:
    """Build a ``book_search`` ``QueryEngineTool`` over the INTERNAL-mode query engine.

    The returned tool is wired to ``QueryMode.INTERNAL`` (raw books) with
    ``allow_internal=True`` so the retriever enforces the trusted-collection
    boundary at construction. ``return_direct`` is intentionally **not** set —
    the agent's value is synthesis across multi-hop tool calls.
    """
    vo_rag_retriever = VoRagRetriever(
        retriever=retriever,
        mode=QueryMode.INTERNAL,
        allow_internal=True,
        top_k=top_k,
        requested_collections=requested_collections,
        filters=filters,
        visual_mode=visual_mode,
        rerank_fn=rerank_fn,
    )

    query_engine = build_vo_rag_query_engine(
        retriever=vo_rag_retriever,
        answer=answer,
        reranker=reranker,
        profile=profile,
    )

    return QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name=_BOOK_SEARCH_NAME,
        description=_BOOK_SEARCH_DESCRIPTION,
    )


__all__ = [
    "build_book_search_tool",
]

"""VoRagQueryFusionRetriever — async query-fusion retriever over VoRagRetriever.

Uses a trusted vo-rag-authored prompt to generate query rewrites via
``VoRagLLM.acomplete``, then merges results with reciprocal-rank fusion
(RRF).  Dedup is by logical chunk id (``node.id_`` — the content-addressed
handle set in ``convert.py``).

SECURITY: the rewrite prompt is TRUSTED text; the user query is
interpolated as trusted (it is the user's own question, not retrieved
content).  Synthesis stays on the protected ``VoRagSynthesizer`` path —
``build_fusion_query_engine`` wires the fusion retriever into the same
``RetrieverQueryEngine`` factory used by the plain path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Any

from vo_rag.integrations.llamaindex import import_llamaindex_symbol
from vo_rag.integrations.llamaindex._async import run_coro_sync

if TYPE_CHECKING:
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle

    from vo_rag.core.answer.synthesizer import AnswerProfile, AnswerSynthesizer
    from vo_rag.integrations.llamaindex.llm import VoRagLLM
    from vo_rag.integrations.llamaindex.reranker import VoRagReranker
    from vo_rag.integrations.llamaindex.retriever import VoRagRetriever
else:
    BaseRetriever = import_llamaindex_symbol("llama_index.core.retrievers", "BaseRetriever")
    NodeWithScore = import_llamaindex_symbol("llama_index.core.schema", "NodeWithScore")
    QueryBundle = import_llamaindex_symbol("llama_index.core.schema", "QueryBundle")

logger = logging.getLogger(__name__)

_RRF_K = 60

_REWRITE_PROMPT = (
    "You are a query-rewriting assistant. Given a user question, generate "
    "alternative phrasings of the same question to improve retrieval coverage.\n\n"
    "User question: {query}\n\n"
    "Respond with ONLY a JSON array of strings, each string being one rephrased "
    "query. Do not include any other text, markdown, or explanation outside the "
    'JSON array. Example: ["rephrase 1", "rephrase 2"]'
)


def _extract_rewrites(text: str) -> list[str]:
    """Extract a JSON array of strings from LLM output.

    Survives reasoning-model preamble and markdown code fences.
    NEVER line-parses.
    """
    # Prefer a JSON code block if present.
    block_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if block_match:
        candidate = block_match.group(1)
    else:
        # Fallback: first top-level JSON array.
        array_match = re.search(r"(\[.*?\])", text, re.DOTALL)
        if not array_match:
            return []
        candidate = array_match.group(1)

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    return [str(item) for item in parsed if isinstance(item, str) and str(item).strip()]


class VoRagQueryFusionRetriever(BaseRetriever):
    """Async query-fusion retriever: rewrite + fan-out + RRF merge."""

    def __init__(
        self,
        base_retriever: VoRagRetriever | list[VoRagRetriever],
        llm: VoRagLLM,
        *,
        num_queries: int = 4,
        top_k: int = 20,
        callback_manager: Any | None = None,
    ) -> None:
        if isinstance(base_retriever, list):
            if len(base_retriever) != 1:
                raise ValueError("VoRagQueryFusionRetriever supports a single base retriever")
            base_retriever = base_retriever[0]
        self._base = base_retriever
        self._llm = llm
        self._num_queries = max(1, num_queries)
        self._top_k = top_k
        super().__init__(callback_manager=callback_manager)

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        queries: list[str] = [query_bundle.query_str]

        if self._num_queries > 1:
            prompt = _REWRITE_PROMPT.format(query=query_bundle.query_str)
            completion = await self._llm.acomplete(prompt)
            rewrites = _extract_rewrites(completion.text)
            max_rewrites = self._num_queries - 1
            queries.extend(rewrites[:max_rewrites])

        results_per_query: list[list[NodeWithScore]] = await asyncio.gather(
            *(self._base._aretrieve(QueryBundle(query_str=q)) for q in queries)
        )

        return self._fuse(results_per_query)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return run_coro_sync(
            lambda: self._aretrieve(query_bundle),
            component="VoRagQueryFusionRetriever._retrieve",
        )

    def _fuse(self, results_per_query: list[list[NodeWithScore]]) -> list[NodeWithScore]:
        """RRF merge + dedup by chunk id.  Ties broken by chunk_id (stable)."""
        scores: dict[str, float] = {}
        nodes_by_id: dict[str, NodeWithScore] = {}

        for result_list in results_per_query:
            for rank, nws in enumerate(result_list, start=1):
                chunk_id = nws.node.id_
                rrf = 1.0 / (_RRF_K + rank)
                if chunk_id in scores:
                    scores[chunk_id] += rrf
                else:
                    scores[chunk_id] = rrf
                    nodes_by_id[chunk_id] = nws

        sorted_ids = sorted(scores.keys(), key=lambda cid: (-scores[cid], cid))[: self._top_k]

        fused: list[NodeWithScore] = []
        all_quarantined = True
        for cid in sorted_ids:
            original = nodes_by_id[cid]
            fused.append(NodeWithScore(node=original.node, score=scores[cid]))
            if not original.node.metadata.get("should_quarantine", False):
                all_quarantined = False

        if fused and all_quarantined:
            logger.error("all fused retrieval hits quarantined")

        return fused


def build_fusion_query_engine(
    *,
    base_retriever: VoRagRetriever | list[VoRagRetriever],
    llm: VoRagLLM,
    answer: AnswerSynthesizer,
    num_queries: int = 4,
    top_k: int = 20,
    reranker: VoRagReranker | None = None,
    profile: AnswerProfile | None = None,
) -> RetrieverQueryEngine:
    """Compose a fusion retriever into the defended query engine.

    The retriever fans out rewrites and merges with RRF; synthesis stays on
    ``VoRagSynthesizer`` (complete_json_protected + egress lockdown).
    """
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.query_engine import RetrieverQueryEngine

    from vo_rag.core.answer.synthesizer import AnswerProfile
    from vo_rag.integrations.llamaindex.synthesizer import VoRagSynthesizer

    if profile is None:
        profile = AnswerProfile.QUALITY

    fusion_retriever = VoRagQueryFusionRetriever(
        base_retriever=base_retriever,
        llm=llm,
        num_queries=num_queries,
        top_k=top_k,
    )

    postprocessors: list[BaseNodePostprocessor] = []
    if reranker is not None:
        postprocessors.append(reranker)

    return RetrieverQueryEngine(
        retriever=fusion_retriever,
        node_postprocessors=postprocessors,
        response_synthesizer=VoRagSynthesizer(answer=answer, profile=profile),
    )

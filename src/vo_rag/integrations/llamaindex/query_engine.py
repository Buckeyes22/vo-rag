"""Query engine factory — compose VoRagRetriever + VoRagReranker + VoRagSynthesizer.

The ``RetrieverQueryEngine`` is mode-blind; the mode is already bound to the
``VoRagRetriever`` it holds (VOR-184). The factory must not add a per-call
mode/collection argument that could re-route the public path to raw.
"""

from __future__ import annotations

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine

from vo_rag.core.answer.synthesizer import AnswerProfile, AnswerSynthesizer
from vo_rag.integrations.llamaindex.reranker import VoRagReranker
from vo_rag.integrations.llamaindex.retriever import VoRagRetriever
from vo_rag.integrations.llamaindex.synthesizer import VoRagSynthesizer


def build_vo_rag_query_engine(
    *,
    retriever: VoRagRetriever,
    answer: AnswerSynthesizer,
    reranker: VoRagReranker | None = None,
    profile: AnswerProfile = AnswerProfile.QUALITY,
) -> RetrieverQueryEngine:
    """Compose the supported vo-rag query path as a stock RetrieverQueryEngine.

    retrieve (sparse+dense+ColBERT, vo-rag) -> rerank (node_postprocessor) ->
    DEFENDED synthesis (complete_json_protected + egress lockdown).
    """
    postprocessors: list[BaseNodePostprocessor] = []
    if reranker is not None:
        postprocessors.append(reranker)
    return RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=postprocessors,
        response_synthesizer=VoRagSynthesizer(answer=answer, profile=profile),
    )

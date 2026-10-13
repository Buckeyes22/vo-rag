"""VoRagSynthesizer — the defended synthesis path behind a LlamaIndex BaseSynthesizer.

SECURITY: This adapter is exactly where the trust boundary lives.
Synthesis over untrusted retrieved context goes through ``complete_json_protected``
(canary + output validation + quarantine) and the egress lockdown
(``allowed_domains=[]``, temp 0.0, raw->sterile remap), NOT through the plain
``VoRagLLM.complete`` seam.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Generator, Sequence
from typing import Any, Self, cast

from llama_index.core.base.response.schema import (
    AsyncStreamingResponse,
    PydanticResponse,
    Response,
    StreamingResponse,
)
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import BaseModel

from vo_rag.core.answer.synthesizer import (
    AnswerProfile,
    AnswerSynthesis,
    AnswerSynthesizer,
)
from vo_rag.core.retrieval.search import SearchResult
from vo_rag.integrations.llamaindex._async import run_coro_sync
from vo_rag.integrations.llamaindex.convert import node_with_score_to_search_result

VO_RAG_RESPONSE_METADATA_KEY = "vo_rag"
_NODES_RESPONSE_KWARG = "_vo_rag_nodes"
logger = logging.getLogger(__name__)

type _ResponseText = BaseModel | str | Generator[str, None, None] | AsyncGenerator[str, None]
type _Response = Response | StreamingResponse | AsyncStreamingResponse | PydanticResponse


class _VoRagAnswerText(str):
    synthesis: AnswerSynthesis

    def __new__(cls, value: str, synthesis: AnswerSynthesis) -> Self:
        instance = str.__new__(cls, value)
        instance.synthesis = synthesis
        return instance

    def __deepcopy__(self, memo: dict[int, Any]) -> str:
        text = str(self)
        memo[id(self)] = text
        # Deliberately shed synthesis metadata; callers get the existing
        # metadata-unavailable warning if they try to prepare a copied string.
        return text

    def __reduce__(self) -> tuple[type[str], tuple[str]]:
        return str, (str(self),)


def _node_to_search_result(node: NodeWithScore) -> SearchResult:
    """Lower a LlamaIndex node back to a vo-rag SearchResult.

    Preserves ``chunk_id``, ``source_id``, ``title``, and ``page`` so
    ``format_answer_context`` emits complete ``[S{i}]`` blocks.
    The shared converter restores payload["chunk_id"], payload["content"],
    the collection, and the original qdrant_point_id when available.
    """
    result = node_with_score_to_search_result(node)
    payload = result.payload
    # format_answer_context reads "source_title" or "title"; the converter
    # carries "section_title" in metadata. Normalise here so the label
    # round-trips regardless of which key the node was built with.
    if not payload.get("source_title") and not payload.get("title"):
        section_title = payload.get("section_title")
        if section_title:
            payload["source_title"] = str(section_title)
    return result


def _synthesis_metadata(
    synthesis: AnswerSynthesis,
    *,
    should_quarantine: bool | None = None,
    metadata_source: str = "response",
) -> dict[str, Any]:
    return {
        "should_quarantine": (
            synthesis.should_quarantine if should_quarantine is None else should_quarantine
        ),
        "model": synthesis.model,
        "confidence": synthesis.draft.confidence,
        "cited_chunk_ids": list(synthesis.draft.cited_chunk_ids),
        "metadata_source": metadata_source,
    }


class VoRagSynthesizer(BaseSynthesizer):
    """BaseSynthesizer that delegates to the DEFENDED AnswerSynthesizer.

    This is the whole point: synthesis over untrusted retrieved context goes
    through ``complete_json_protected`` (canary + output validation + quarantine)
    and the egress lockdown (``allowed_domains=[]``, temp 0.0, raw->sterile remap),
    NOT through the plain VoRagLLM.complete seam.
    """

    def __init__(
        self,
        *,
        answer: AnswerSynthesizer,
        profile: AnswerProfile = AnswerProfile.QUALITY,
        callback_manager: Any | None = None,
        llm: Any | None = None,
    ) -> None:
        super().__init__(llm=llm, callback_manager=callback_manager)
        self._answer = answer
        self._profile = profile

    # BaseSynthesizer.synthesize lowers nodes->text_chunks; we need the nodes
    # (for payload fidelity), so pass them through request-scoped kwargs.
    async def asynthesize(
        self,
        query: str | QueryBundle,
        nodes: list[NodeWithScore],
        additional_source_nodes: Sequence[NodeWithScore] | None = None,
        **kwargs: Any,
    ) -> Any:
        captured_nodes = list(nodes)
        kwargs[_NODES_RESPONSE_KWARG] = captured_nodes
        return await super().asynthesize(
            query=query,
            nodes=nodes,
            additional_source_nodes=additional_source_nodes,
            **kwargs,
        )

    def synthesize(
        self,
        query: str | QueryBundle,
        nodes: list[NodeWithScore],
        additional_source_nodes: Sequence[NodeWithScore] | None = None,
        **kwargs: Any,
    ) -> Any:
        captured_nodes = list(nodes)
        kwargs[_NODES_RESPONSE_KWARG] = captured_nodes
        return super().synthesize(
            query=query,
            nodes=nodes,
            additional_source_nodes=additional_source_nodes,
            **kwargs,
        )

    async def aget_response(self, query_str: str, text_chunks: Sequence[str], **kwargs: Any) -> str:
        if _NODES_RESPONSE_KWARG not in kwargs:
            raise RuntimeError(
                "VoRagSynthesizer.aget_response requires request-scoped "
                f"{_NODES_RESPONSE_KWARG}; call synthesize/asynthesize with source nodes"
            )
        nodes = kwargs.pop(_NODES_RESPONSE_KWARG)
        context = [_node_to_search_result(n) for n in nodes]
        synthesis: AnswerSynthesis = await self._answer.synthesize(
            query=query_str,
            context=context,
            profile=self._profile,
        )
        # Return a str subtype so _prepare_response_output can attach Response
        # metadata without relying on instance state shared across requests.
        return _VoRagAnswerText(synthesis.draft.answer, synthesis)

    def get_response(self, query_str: str, text_chunks: Sequence[str], **kwargs: Any) -> str:
        return run_coro_sync(
            lambda: self.aget_response(query_str, text_chunks, **kwargs),
            component="VoRagSynthesizer.get_response",
        )

    def _prepare_response_output(
        self,
        response_str: _ResponseText | None,
        source_nodes: list[NodeWithScore],
    ) -> _Response:
        response = super()._prepare_response_output(response_str, source_nodes)
        vo_rag_metadata: dict[str, Any] | None = None

        if isinstance(response_str, _VoRagAnswerText):
            vo_rag_metadata = _synthesis_metadata(response_str.synthesis)

        if vo_rag_metadata is None:
            if isinstance(response_str, str) and source_nodes:
                logger.warning(
                    "vo_rag_synthesizer_metadata_unavailable: plain string response "
                    "with source nodes has no vo-rag metadata; response wrapper was shed"
                )
            return response

        typed_response = cast(Response, response)
        metadata = dict(typed_response.metadata or {})
        metadata[VO_RAG_RESPONSE_METADATA_KEY] = vo_rag_metadata
        typed_response.metadata = metadata
        return typed_response

    def _get_prompts(self) -> dict[str, Any]:
        return {}  # prompt is owned by AnswerSynthesizer, not the framework

    def _update_prompts(self, prompts: dict[str, Any]) -> None:
        return None

"""RED-first contract tests for VoRagQueryFusionRetriever.

Skip-guard: ``pytest.importorskip("llama_index")`` is present so this module
is safe to collect in the light unit job.  (The subtree conftest also drops
adapter tests when the extra is absent; the importorskip is a belt-and-
suspenders guard for CI jobs that may not use the conftest collect_ignore.)

NOTE: this file must be added to the ``test-llamaindex`` CI job.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

pytest.importorskip("llama_index")

from vo_rag.core.answer.synthesizer import AnswerDraft, AnswerSynthesis
from vo_rag.core.llm.client import LLMClient
from vo_rag.core.llm.output_validator import ValidationResult
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.observability.cost import cost_context
from vo_rag.core.retrieval.query_mode import QueryMode
from vo_rag.core.retrieval.search import HybridRetriever, SearchResult
from vo_rag.integrations.llamaindex.fusion import (
    VoRagQueryFusionRetriever,
    _extract_rewrites,
    build_fusion_query_engine,
)
from vo_rag.integrations.llamaindex.llm import VoRagLLM
from vo_rag.integrations.llamaindex.query_engine import build_vo_rag_query_engine
from vo_rag.integrations.llamaindex.retriever import VoRagRetriever
from vo_rag.integrations.llamaindex.synthesizer import VoRagSynthesizer

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _FakeQdrantStore:
    async def search_hybrid(
        self,
        collection: str,
        query_dense: list[float],
        query_sparse: dict[int, float],
        query_bm25: dict[int, float],
        query_colbert: list[list[float]],
        *,
        top_k: int = 20,
        filters: Any | None = None,
        prefetch_depth: int = 100,
    ) -> list[Any]:
        return []


class _FakeHybridRetriever(HybridRetriever):
    def __init__(self, results: list[SearchResult] | None = None) -> None:
        super().__init__(qdrant=_FakeQdrantStore(), embedding_client=None)
        self.results = results or []
        self.calls: list[str] = []

    async def cross_collection_search(
        self,
        query: str,
        collections: list[str],
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        mode: QueryMode = QueryMode.INTERNAL,
        visual_mode: str | None = None,
    ) -> list[SearchResult]:
        self.calls.append(query)
        return self.results


def _make_node(chunk_id: str, score: float = 0.5, text: str = "body") -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(text=text, id_=chunk_id, metadata={"chunk_id": chunk_id}),
        score=score,
    )


def _fake_llm_client(response_text: str = "[]") -> Any:
    client = MagicMock()
    client.provider.name = "scripted"
    client.provider.model = "scripted-model"
    client.complete = AsyncMock(
        return_value=LLMResponse(
            text=response_text,
            model="scripted-model",
            latency_ms=1.0,
            attempts=1,
        )
    )
    return client


def _answer_synth() -> Any:
    a = MagicMock()
    a.synthesize = AsyncMock(
        return_value=AnswerSynthesis(
            draft=AnswerDraft(answer="ans", confidence=0.8, cited_chunk_ids=["c1"]),
            llm_response=LLMResponse(text="ans", model="m"),
            validation=ValidationResult(ok=True),
            should_quarantine=False,
            model="m",
        )
    )
    return a


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #


def test_extract_rewrites_from_pure_json() -> None:
    assert _extract_rewrites('["q1", "q2"]') == ["q1", "q2"]


def test_extract_rewrites_from_reasoning_preamble() -> None:
    raw = '<thinking>I need to rephrase this.</thinking>\n```json\n["rewrite a", "rewrite b"]\n```'
    assert _extract_rewrites(raw) == ["rewrite a", "rewrite b"]


def test_extract_rewrites_no_array_returns_empty() -> None:
    assert _extract_rewrites("just some text") == []


def test_extract_rewrites_ignores_non_string_items() -> None:
    assert _extract_rewrites('["ok", 123, null, "fine"]') == ["ok", "fine"]


# --------------------------------------------------------------------------- #
# Rewrite cap
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_rewrites_bounded_to_num_queries_cap() -> None:
    """If the LLM returns N rewrites, at most num_queries-1 are issued."""
    base = _FakeHybridRetriever(results=[])
    vorag_base = VoRagRetriever(base, mode=QueryMode.PUBLICATION, top_k=10)
    llm = VoRagLLM(client=_fake_llm_client('["a", "b", "c", "d", "e"]'))

    fusion = VoRagQueryFusionRetriever(
        base_retriever=vorag_base,
        llm=llm,
        num_queries=4,
        top_k=10,
    )

    await fusion._aretrieve(QueryBundle(query_str="original"))

    # original + at most 3 rewrites = at most 4 calls
    assert len(base.calls) <= 4
    assert base.calls[0] == "original"


@pytest.mark.asyncio
async def test_single_query_skips_rewrite_call() -> None:
    base = _FakeHybridRetriever(results=[])
    vorag_base = VoRagRetriever(base, mode=QueryMode.PUBLICATION, top_k=10)
    client = _fake_llm_client("[]")
    llm = VoRagLLM(client=client)

    fusion = VoRagQueryFusionRetriever(
        base_retriever=vorag_base,
        llm=llm,
        num_queries=1,
        top_k=10,
    )

    await fusion._aretrieve(QueryBundle(query_str="q"))

    # num_queries == 1 means NO rewrite LLM call.
    assert client.complete.await_count == 0
    assert len(base.calls) == 1
    assert base.calls[0] == "q"


# --------------------------------------------------------------------------- #
# Dedup + RRF scoring
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dedup_same_chunk_id_rrf_sums() -> None:
    """A chunk appearing in two query branches gets one fused node with summed RRF."""
    base = _FakeHybridRetriever(
        results=[
            SearchResult(
                id="p1",
                score=0.9,
                collection="books",
                payload={"chunk_id": "c1", "content": "alpha"},
            ),
        ]
    )
    vorag_base = VoRagRetriever(
        base, mode=QueryMode.PUBLICATION, top_k=10, requested_collections=["books"]
    )
    llm = VoRagLLM(client=_fake_llm_client('["alt"]'))

    fusion = VoRagQueryFusionRetriever(
        base_retriever=vorag_base,
        llm=llm,
        num_queries=2,
        top_k=10,
    )

    nodes = await fusion._aretrieve(QueryBundle(query_str="q"))

    # original + 1 rewrite = 2 calls, both return the same chunk
    assert len(base.calls) == 2
    assert len(nodes) == 1
    # RRF for rank 1 in each list: 1/61 + 1/61
    assert nodes[0].score == pytest.approx(2.0 / 61.0)
    assert nodes[0].node.id_ == "c1"


@pytest.mark.asyncio
async def test_tie_break_by_chunk_id_deterministic() -> None:
    """Same RRF score → sorted by chunk_id ascending (stable)."""
    # Two different chunks, each appears once at rank 1 in one branch.
    base = _FakeHybridRetriever(
        results=[
            SearchResult(
                id="p1",
                score=0.9,
                collection="books",
                payload={"chunk_id": "z_chunk", "content": "z"},
            ),
        ]
    )
    vorag_base = VoRagRetriever(base, mode=QueryMode.PUBLICATION, top_k=10)
    llm = VoRagLLM(client=_fake_llm_client('["alt"]'))

    fusion = VoRagQueryFusionRetriever(
        base_retriever=vorag_base,
        llm=llm,
        num_queries=2,
        top_k=10,
    )

    nodes = await fusion._aretrieve(QueryBundle(query_str="q"))

    # Each call returns one distinct chunk; RRF scores are equal (1/61 each).
    # Tie-break by chunk_id ascending.
    ids = [n.node.id_ for n in nodes]
    assert ids == sorted(ids)


# --------------------------------------------------------------------------- #
# Quarantine propagation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_all_quarantined_hits_propagated_not_silently_empty() -> None:
    """If every returned node carries should_quarantine=True, fusion logs error
    but still returns the nodes (does not silently drop them)."""
    base = _FakeHybridRetriever(
        results=[
            SearchResult(
                id="p1",
                score=0.9,
                collection="books",
                payload={
                    "chunk_id": "c1",
                    "content": "alpha",
                    "should_quarantine": True,
                },
            ),
        ]
    )
    vorag_base = VoRagRetriever(
        base, mode=QueryMode.PUBLICATION, top_k=10, requested_collections=["books"]
    )
    llm = VoRagLLM(client=_fake_llm_client("[]"))

    fusion = VoRagQueryFusionRetriever(
        base_retriever=vorag_base,
        llm=llm,
        num_queries=1,
        top_k=10,
    )

    nodes = await fusion._aretrieve(QueryBundle(query_str="q"))

    assert len(nodes) == 1
    assert nodes[0].node.metadata.get("should_quarantine") is True


# --------------------------------------------------------------------------- #
# Cost accounting
# --------------------------------------------------------------------------- #


class _CostFakeProvider:
    name = "scripted"

    def __init__(self, model: str = "scripted-model", response_text: str = "[]") -> None:
        self._model = model
        self._response_text = response_text

    @property
    def model(self) -> str:
        return self._model

    async def complete_raw(self, **kwargs: Any) -> LLMResponse:
        return LLMResponse(
            text=self._response_text,
            model=self._model,
            provider=self.name,
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
        )

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_rewrite_call_is_cost_accounted(monkeypatch: pytest.MonkeyPatch) -> None:
    """When wrapped in cost_context, the rewrite LLM call records cost."""
    base = _FakeHybridRetriever(results=[])
    vorag_base = VoRagRetriever(base, mode=QueryMode.PUBLICATION, top_k=10)
    monkeypatch.setattr("vo_rag.core.llm.client._compute_cost", lambda _resp: 0.0042)
    client = LLMClient(provider=_CostFakeProvider(response_text='["rewrite"]'))
    llm = VoRagLLM(client=client)

    fusion = VoRagQueryFusionRetriever(
        base_retriever=vorag_base,
        llm=llm,
        num_queries=2,
        top_k=10,
    )

    with cost_context(envelope_id="env-fusion", pipeline="test_fusion") as acc:
        await fusion._aretrieve(QueryBundle(query_str="q"))

    assert acc.total_usd() == pytest.approx(0.0042)
    assert acc.step_breakdown()["llm|scripted-model"]["calls"] == 1


# --------------------------------------------------------------------------- #
# Protected-path wiring
# --------------------------------------------------------------------------- #


def test_build_fusion_query_engine_uses_vo_rag_synthesizer() -> None:
    """The engine returned by build_fusion_query_engine delegates synthesis to
    VoRagSynthesizer (complete_json_protected), not a bare LLM seam."""
    base = _FakeHybridRetriever(results=[])
    vorag_base = VoRagRetriever(base, mode=QueryMode.PUBLICATION, top_k=10)
    llm = VoRagLLM(client=_fake_llm_client("[]"))

    engine = build_fusion_query_engine(
        base_retriever=vorag_base,
        llm=llm,
        answer=_answer_synth(),
        num_queries=1,
        top_k=10,
    )

    assert isinstance(engine, RetrieverQueryEngine)
    assert isinstance(engine._response_synthesizer, VoRagSynthesizer)


def test_build_fusion_query_engine_matches_plain_engine_shape() -> None:
    """build_fusion_query_engine produces the same structural shape as
    build_vo_rag_query_engine (retriever + optional postprocessors + defended
    synthesizer)."""
    base = _FakeHybridRetriever(results=[])
    vorag_base = VoRagRetriever(base, mode=QueryMode.PUBLICATION, top_k=10)
    llm = VoRagLLM(client=_fake_llm_client("[]"))
    answer = _answer_synth()

    plain = build_vo_rag_query_engine(
        retriever=vorag_base,
        answer=answer,
    )
    fusion = build_fusion_query_engine(
        base_retriever=vorag_base,
        llm=llm,
        answer=answer,
        num_queries=1,
        top_k=10,
    )

    assert type(plain) is type(fusion)
    assert type(plain._response_synthesizer) is type(fusion._response_synthesizer)


# --------------------------------------------------------------------------- #
# Constructor validation
# --------------------------------------------------------------------------- #


def test_list_of_multiple_retrievers_rejected() -> None:
    base = _FakeHybridRetriever(results=[])
    vorag_base = VoRagRetriever(base, mode=QueryMode.PUBLICATION, top_k=10)
    llm = VoRagLLM(client=_fake_llm_client("[]"))

    with pytest.raises(ValueError, match="single base retriever"):
        VoRagQueryFusionRetriever(
            base_retriever=[vorag_base, vorag_base],
            llm=llm,
        )


def test_list_of_one_retriever_accepted() -> None:
    base = _FakeHybridRetriever(results=[])
    vorag_base = VoRagRetriever(base, mode=QueryMode.PUBLICATION, top_k=10)
    llm = VoRagLLM(client=_fake_llm_client("[]"))

    fusion = VoRagQueryFusionRetriever(
        base_retriever=[vorag_base],
        llm=llm,
    )
    assert fusion._base is vorag_base

"""Contract tests for VoRagSynthesizer and build_vo_rag_query_engine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from vo_rag.core.answer.synthesizer import (
    AnswerDraft,
    AnswerProfile,
    AnswerSynthesis,
)
from vo_rag.core.llm.output_validator import ValidationResult
from vo_rag.core.llm.response import LLMResponse
from vo_rag.integrations.llamaindex.synthesizer import (
    _NODES_RESPONSE_KWARG,
    VO_RAG_RESPONSE_METADATA_KEY,
    VoRagSynthesizer,
)


def _node(text="alpha", chunk_id="c1"):
    n = TextNode(text=text, id_=chunk_id, metadata={"chunk_id": chunk_id, "source_id": "s1"})
    return NodeWithScore(node=n, score=0.9)


def _answer_synth(should_quarantine=False):
    a = MagicMock()
    a.synthesize = AsyncMock(
        return_value=AnswerSynthesis(
            draft=AnswerDraft(answer="grounded [S1]", confidence=0.8, cited_chunk_ids=["c1"]),
            llm_response=LLMResponse(text="grounded [S1]", model="claude-x"),
            validation=ValidationResult(ok=True),
            should_quarantine=should_quarantine,
            model="claude-x",
        )
    )
    return a


def test_synthesizer_delegates_to_protected_answer_path():
    a = _answer_synth()
    syn = VoRagSynthesizer(answer=a, profile=AnswerProfile.QUALITY, llm=MagicMock())
    out = syn.get_response("q", ["alpha"], **{_NODES_RESPONSE_KWARG: [_node()]})
    assert out == "grounded [S1]"
    a.synthesize.assert_awaited_once()
    # the egress lock is a property of AnswerSynthesizer, asserted in its own
    # unit test; here we assert the engine routes THROUGH it.


def test_answer_parity_wrapped_equals_native():
    """Wrapped path returns the same answer text and citations as the native path."""
    fixture = AnswerSynthesis(
        draft=AnswerDraft(
            answer="The answer is 42 [S1][S2]",
            confidence=0.95,
            cited_chunk_ids=["chunk-a", "chunk-b"],
            gaps=[],
            follow_up_questions=[],
        ),
        llm_response=LLMResponse(text="The answer is 42 [S1][S2]", model="claude-x"),
        validation=ValidationResult(ok=True),
        should_quarantine=False,
        model="claude-x",
    )

    answer_mock = MagicMock()
    answer_mock.synthesize = AsyncMock(return_value=fixture)

    nodes = [
        _node(text="alpha", chunk_id="chunk-a"),
        _node(text="beta", chunk_id="chunk-b"),
    ]

    # Native path result (deterministic fixture)
    native = fixture

    # Wrapped path
    syn = VoRagSynthesizer(answer=answer_mock, profile=AnswerProfile.QUALITY, llm=MagicMock())
    wrapped = syn.synthesize("What is the answer?", nodes)

    # Answer text parity
    assert str(wrapped) == native.draft.answer
    # Citation parity — chunk_ids round-trip through node metadata
    assert wrapped.metadata is not None
    assert (
        wrapped.metadata[VO_RAG_RESPONSE_METADATA_KEY]["cited_chunk_ids"]
        == native.draft.cited_chunk_ids
    )

    # Verify the wrapped path converted nodes with correct chunk_id fidelity
    call_kwargs = answer_mock.synthesize.call_args.kwargs
    assert call_kwargs["query"] == "What is the answer?"
    assert call_kwargs["profile"] == AnswerProfile.QUALITY
    context = call_kwargs["context"]
    assert len(context) == 2
    assert context[0].payload["chunk_id"] == "chunk-a"
    assert context[1].payload["chunk_id"] == "chunk-b"


def test_quarantine_verdict_propagates():
    syn = VoRagSynthesizer(answer=_answer_synth(should_quarantine=True), llm=MagicMock())
    response = syn.synthesize("q", [_node()])

    assert str(response) == "grounded [S1]"
    assert response.metadata is not None
    assert response.metadata[VO_RAG_RESPONSE_METADATA_KEY]["should_quarantine"] is True


def test_query_engine_composes_retriever_reranker_synthesizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llama_index.core.settings import Settings

    from vo_rag.integrations.llamaindex.query_engine import build_vo_rag_query_engine

    # monkeypatch (not a bare ``Settings._llm = ...``) so the global LLM is
    # restored after this test — a leaked MagicMock here flakes unrelated tests
    # under pytest-randomly when this runs before them.
    monkeypatch.setattr(Settings, "_llm", MagicMock(), raising=False)
    eng = build_vo_rag_query_engine(
        retriever=MagicMock(),
        answer=_answer_synth(),
        reranker=MagicMock(),
    )
    # reranker landed as a node_postprocessor; synthesizer is the defended one.
    assert any(isinstance(p, MagicMock) for p in eng._node_postprocessors)
    assert isinstance(eng._response_synthesizer, VoRagSynthesizer)

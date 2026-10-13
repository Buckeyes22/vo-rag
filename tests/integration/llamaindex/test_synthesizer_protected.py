"""VOR-195 contract tests for the protected LlamaIndex synthesizer path."""

from __future__ import annotations

import copy
import logging
import pickle
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from vo_rag.core.answer.synthesizer import AnswerDraft, AnswerProfile, AnswerSynthesis
from vo_rag.core.llm.output_validator import ValidationResult
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.observability.redaction import get_default_redactor
from vo_rag.integrations.llamaindex.synthesizer import (
    _NODES_RESPONSE_KWARG,
    VO_RAG_RESPONSE_METADATA_KEY,
    VoRagSynthesizer,
    _VoRagAnswerText,
)


def _node(text: str = "alpha", chunk_id: str = "c1") -> NodeWithScore:
    node = TextNode(
        text=text,
        id_=chunk_id,
        metadata={
            "chunk_id": chunk_id,
            "source_id": "s1",
            "source_title": "Source One",
            "page_number": 7,
            "collection": "vo_rag_books_sterile_v1",
            "qdrant_point_id": "point-1",
        },
    )
    return NodeWithScore(node=node, score=0.9)


def _synthesis(*, should_quarantine: bool = False) -> AnswerSynthesis:
    return AnswerSynthesis(
        draft=AnswerDraft(
            answer="grounded [S1]",
            confidence=0.8,
            cited_chunk_ids=["c1"],
        ),
        llm_response=LLMResponse(text="grounded [S1]", model="claude-x"),
        validation=ValidationResult(ok=True),
        should_quarantine=should_quarantine,
        model="claude-x",
    )


def _answer_synth(*, should_quarantine: bool = False) -> MagicMock:
    answer = MagicMock()
    answer.synthesize = AsyncMock(return_value=_synthesis(should_quarantine=should_quarantine))
    return answer


def test_answer_text_copy_protocols_degrade_to_plain_str() -> None:
    text = _VoRagAnswerText("grounded [S1]", _synthesis(should_quarantine=True))

    for clone in (copy.deepcopy(text), copy.copy(text), pickle.loads(pickle.dumps(text))):
        assert clone == "grounded [S1]"
        assert type(clone) is str
        assert not hasattr(clone, "synthesis")


def test_synthesizer_delegates_to_protected_answer_path() -> None:
    answer = _answer_synth()
    synthesizer = VoRagSynthesizer(
        answer=answer,
        profile=AnswerProfile.QUALITY,
        llm=MagicMock(),
    )

    out = synthesizer.get_response("q", ["alpha"], **{_NODES_RESPONSE_KWARG: [_node()]})

    assert out == "grounded [S1]"
    answer.synthesize.assert_awaited_once()
    call_kwargs = answer.synthesize.call_args.kwargs
    assert call_kwargs["query"] == "q"
    assert call_kwargs["profile"] == AnswerProfile.QUALITY
    assert call_kwargs["context"][0].payload["chunk_id"] == "c1"


def test_quarantine_verdict_propagates() -> None:
    synthesizer = VoRagSynthesizer(
        answer=_answer_synth(should_quarantine=True),
        llm=MagicMock(),
    )

    response = synthesizer.synthesize("q", [_node()])

    assert str(response) == "grounded [S1]"
    assert response.metadata is not None
    assert response.metadata[VO_RAG_RESPONSE_METADATA_KEY]["should_quarantine"] is True


def test_synthesizer_response_can_be_redacted_after_metadata_attachment() -> None:
    synthesizer = VoRagSynthesizer(
        answer=_answer_synth(),
        llm=MagicMock(),
    )

    response = synthesizer.synthesize("q", [_node()])
    redacted = get_default_redactor().redact_mapping({"response": response})

    response_payload = redacted["response"]
    assert response_payload["response"] == "grounded [S1]"
    assert response_payload["metadata"][VO_RAG_RESPONSE_METADATA_KEY]["should_quarantine"] is False


def test_plain_str_response_warns_when_vo_rag_metadata_wrapper_is_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    synthesizer = VoRagSynthesizer(
        answer=_answer_synth(should_quarantine=True),
        llm=MagicMock(),
    )
    prior = synthesizer.synthesize("bad", [_node()])
    assert prior.metadata is not None
    assert prior.metadata[VO_RAG_RESPONSE_METADATA_KEY]["should_quarantine"] is True

    with caplog.at_level(logging.WARNING, logger="vo_rag.integrations.llamaindex.synthesizer"):
        response = synthesizer._prepare_response_output("grounded [S1]", [_node()])

    assert response.metadata is None or VO_RAG_RESPONSE_METADATA_KEY not in response.metadata
    assert "vo_rag_synthesizer_metadata_unavailable" in caplog.text


async def test_async_quarantine_verdict_propagates_in_response_metadata() -> None:
    synthesizer = VoRagSynthesizer(
        answer=_answer_synth(should_quarantine=True),
        llm=MagicMock(),
    )

    response = await synthesizer.asynthesize("q", [_node()])

    assert str(response) == "grounded [S1]"
    assert response.metadata is not None
    vo_rag = response.metadata[VO_RAG_RESPONSE_METADATA_KEY]
    assert vo_rag["should_quarantine"] is True
    assert vo_rag["model"] == "claude-x"
    assert vo_rag["confidence"] == 0.8
    assert vo_rag["cited_chunk_ids"] == ["c1"]


async def test_aget_response_requires_request_scoped_nodes_even_with_stale_instance_nodes() -> None:
    answer = _answer_synth()
    synthesizer = VoRagSynthesizer(
        answer=answer,
        profile=AnswerProfile.QUALITY,
        llm=MagicMock(),
    )
    object.__setattr__(synthesizer, "_nodes", [_node(chunk_id="stale")])

    with pytest.raises(RuntimeError, match="_vo_rag_nodes"):
        await synthesizer.aget_response("q", ["alpha"])

    answer.synthesize.assert_not_awaited()


def test_no_vo_rag_llm_complete_in_answer_path(monkeypatch) -> None:
    from vo_rag.integrations.llamaindex.llm import VoRagLLM

    complete_calls = 0

    def spy_complete(*args, **kwargs):
        nonlocal complete_calls
        complete_calls += 1
        raise AssertionError("VoRagLLM.complete must not handle answer synthesis")

    monkeypatch.setattr(VoRagLLM, "complete", spy_complete)
    synthesizer = VoRagSynthesizer(
        answer=_answer_synth(),
        profile=AnswerProfile.QUALITY,
        llm=MagicMock(),
    )

    assert (
        synthesizer.get_response("q", ["alpha"], **{_NODES_RESPONSE_KWARG: [_node()]})
        == "grounded [S1]"
    )
    assert complete_calls == 0


def test_empty_synthesis_has_no_stale_vo_rag_metadata_sync() -> None:
    """Regression: empty-node calls must not retain prior response metadata."""
    answer = _answer_synth(should_quarantine=True)
    synthesizer = VoRagSynthesizer(
        answer=answer,
        profile=AnswerProfile.QUALITY,
        llm=MagicMock(),
    )

    prior = synthesizer.synthesize("bad", [_node()])
    assert prior.metadata is not None
    assert prior.metadata[VO_RAG_RESPONSE_METADATA_KEY]["should_quarantine"] is True

    response = synthesizer.synthesize("empty", [])
    assert response.metadata is None or VO_RAG_RESPONSE_METADATA_KEY not in response.metadata


async def test_empty_synthesis_has_no_stale_vo_rag_metadata_async() -> None:
    """Regression: async empty-node calls must not retain prior response metadata."""
    answer = _answer_synth(should_quarantine=True)
    synthesizer = VoRagSynthesizer(
        answer=answer,
        profile=AnswerProfile.QUALITY,
        llm=MagicMock(),
    )

    prior = await synthesizer.asynthesize("bad", [_node()])
    assert prior.metadata is not None
    assert prior.metadata[VO_RAG_RESPONSE_METADATA_KEY]["should_quarantine"] is True

    response = await synthesizer.asynthesize("empty", [])
    assert response.metadata is None or VO_RAG_RESPONSE_METADATA_KEY not in response.metadata

"""Complete brain source evidence is delivered whole or rejected before synthesis."""

from __future__ import annotations

from typing import Any

import pytest

from vo_rag.core.answer.context import AnswerEvidenceUnavailable, format_answer_context
from vo_rag.core.answer.synthesizer import AnswerProfile, AnswerSynthesizer
from vo_rag.core.config.settings import AnswerSettings
from vo_rag.core.retrieval.search import SearchResult


def _hit(text: str, *, collection: str = "vo_rag_brain_v1", complete: bool = True) -> SearchResult:
    return SearchResult(
        id="p",
        score=1.0,
        collection=collection,
        payload={
            "content": text,
            "source_id": "wiki/device.md",
            "commit": "a" * 40,
            "governing_context": [],
            "governing_context_version": 1,
            "governing_context_complete": complete,
        },
    )


def test_complete_brain_conditions_never_become_a_prefix() -> None:
    text = "Background.\n" * 4000 + "Only laboratory firmware 3 supports retention."
    rendered = format_answer_context([_hit(text)])
    assert text in rendered
    assert rendered.endswith("Only laboratory firmware 3 supports retention.")
    with pytest.raises(AnswerEvidenceUnavailable, match="exceeds_65536"):
        format_answer_context([_hit(text), _hit(text)])
    with pytest.raises(AnswerEvidenceUnavailable, match="incomplete"):
        format_answer_context([_hit("Retention seven days.", complete=False)])


def test_existing_library_context_behavior_is_unchanged() -> None:
    text = "source material " * 6000
    assert text.rstrip() in format_answer_context(
        [_hit(text, collection="vo_rag_books_sterile_v1")]
    )


@pytest.mark.asyncio
async def test_answer_rejects_overbudget_brain_before_provider_or_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthesizer = AnswerSynthesizer(settings=AnswerSettings())
    calls: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append("provider")
        raise AssertionError("must refuse before constructing a provider")

    monkeypatch.setattr(synthesizer, "select_model", forbidden)
    with pytest.raises(AnswerEvidenceUnavailable, match="exceeds_65536"):
        await synthesizer.synthesize(
            query="retention?", context=[_hit("x" * 66000)], profile=AnswerProfile.FAST
        )
    assert calls == []


@pytest.mark.asyncio
async def test_legacy_brain_evidence_rejected_before_any_provider_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = _hit("Retention is seven days.")
    legacy.payload.pop("governing_context_version")
    legacy.payload.pop("governing_context_complete")
    legacy.payload.pop("governing_context")
    synthesizer = AnswerSynthesizer(settings=AnswerSettings())
    calls: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append("provider")
        raise AssertionError("legacy evidence is unattested")

    monkeypatch.setattr(synthesizer, "select_model", forbidden)
    with pytest.raises(AnswerEvidenceUnavailable, match="incomplete"):
        await synthesizer.synthesize(
            query="retention?", context=[legacy], profile=AnswerProfile.LOCAL
        )
    assert calls == []

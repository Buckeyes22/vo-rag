"""Tests for core/compliance sterilizer."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.compliance.sterilizer import (
    SterilizerParseError,
    _check_verbatim_budget,
    sterilize_chunks,
)
from vo_rag.core.llm.client import LLMClient
from vo_rag.core.llm.output_validator import ValidationResult
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.models.artifact import LicenseClass


class _FakeProvider:
    name = "fake"
    model = "fake-model"

    async def complete_raw(self, **kwargs: Any) -> LLMResponse:
        return LLMResponse(
            text='{"sterile_content": "ok"}',
            model=self.model,
            input_tokens=10,
            output_tokens=5,
            provider=self.name,
        )

    async def aclose(self) -> None:
        return None


def _fake_llm_client() -> LLMClient:
    return LLMClient(provider=_FakeProvider())


def _chunk(**overrides: Any) -> RetrievalChunk:
    defaults: dict[str, Any] = {
        "chunk_id": "chunk-1",
        "chunk_type": "text",
        "document_id": "doc-1",
        "source_id": "src-1",
        "page_number": 1,
        "page_range": None,
        "section_path": "1.2 Braking",
        "section_title": "Braking Technique",
        "chapter_path": "Ch1",
        "content": "Trail braking transfers weight to the front axle.",
        "embedding_text": "Trail braking transfers weight to the front axle.",
        "token_count": 10,
        "has_image": False,
        "image_id": None,
        "image_ref": None,
        "vlm_description": None,
        "metadata": {},
    }
    defaults.update(overrides)
    return RetrievalChunk(**defaults)


def _make_fake_complete(
    sterile_content: str = "Sterilized factual summary.",
    facts: list[str] | None = None,
    equations: list[str] | None = None,
    concepts: list[str] | None = None,
    should_quarantine: bool = False,
) -> Any:
    class _Payload(BaseModel):
        sterile_content: str
        facts: list[str] = []
        equations: list[str] = []
        concepts: list[str] = []
        injection_suspected: bool = False
        injection_reason: str | None = None

    payload = _Payload(
        sterile_content=sterile_content,
        facts=facts or [],
        equations=equations or [],
        concepts=concepts or [],
    )
    resp = LLMResponse(
        text=json.dumps(payload.model_dump()),
        model="fake-model",
        input_tokens=10,
        output_tokens=5,
        provider="fake",
    )
    validation = ValidationResult(ok=True, signals=[])

    async def _fake(
        client: LLMClient,
        *,
        schema: type[Any],
        untrusted_content: str,
        pipeline: str,
        allowed_domains: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, LLMResponse, ValidationResult, bool]:
        parsed = schema.model_validate(payload.model_dump())
        return parsed, resp, validation, should_quarantine

    return _fake


# --------------------------------------------------------------------------- #
# Invariant: empty input returns empty output without any LLM call
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_empty_input_returns_empty_without_llm_call() -> None:
    with patch("vo_rag.core.compliance.sterilizer.complete_json_protected") as mock_complete:
        result = await sterilize_chunks([], _fake_llm_client())
        assert result == []
        mock_complete.assert_not_called()


# --------------------------------------------------------------------------- #
# Invariant: output is RetrievalChunk-shaped with redacted content
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sterilize_chunks_returns_retrieval_chunk_shape() -> None:
    chunk = _chunk()
    fake = _make_fake_complete(sterile_content="Weight transfers to front under braking.")

    with patch("vo_rag.core.compliance.sterilizer.complete_json_protected", fake):
        result = await sterilize_chunks([chunk], _fake_llm_client())

    assert len(result) == 1
    sterile = result[0]
    assert isinstance(sterile, RetrievalChunk)
    assert sterile.chunk_id != chunk.chunk_id
    assert sterile.source_id == chunk.source_id
    assert sterile.document_id == chunk.document_id
    assert sterile.content == "Weight transfers to front under braking."
    assert "Weight transfers to front under braking." in sterile.embedding_text
    assert sterile.metadata.get("facts_extracted") == []
    assert sterile.metadata.get("equations") == []
    assert sterile.metadata.get("concepts") == []
    assert sterile.metadata.get("_sterilization_quarantine") is False


# --------------------------------------------------------------------------- #
# Invariant: every LLM call uses complete_json_protected with allowed_domains=[]
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_complete_json_protected_called_with_allowed_domains_empty_list() -> None:
    chunk = _chunk()
    calls: list[dict[str, Any]] = []

    async def _recording_fake(
        client: LLMClient,
        *,
        schema: type[Any],
        untrusted_content: str,
        pipeline: str,
        allowed_domains: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, LLMResponse, ValidationResult, bool]:
        calls.append({"allowed_domains": allowed_domains, "pipeline": pipeline})
        parsed = schema.model_validate(
            {"sterile_content": "ok", "facts": [], "equations": [], "concepts": []}
        )
        resp = LLMResponse(
            text='{"sterile_content": "ok"}',
            model="fake-model",
            input_tokens=10,
            output_tokens=5,
            provider="fake",
        )
        return parsed, resp, ValidationResult(ok=True, signals=[]), False

    with patch("vo_rag.core.compliance.sterilizer.complete_json_protected", _recording_fake):
        await sterilize_chunks([chunk], _fake_llm_client())

    assert len(calls) == 1
    assert calls[0]["allowed_domains"] == []
    assert calls[0]["pipeline"] == "book_rag"


# --------------------------------------------------------------------------- #
# Invariant: verbatim budget (F-029)
# --------------------------------------------------------------------------- #


def test_verbatim_budget_ok_for_short_matches() -> None:
    original = "The quick brown fox jumps over the lazy dog."
    sterile = "A quick animal leaps over a sleepy canine."
    ok, longest, ratio = _check_verbatim_budget(original, sterile)
    assert ok is True
    assert longest <= 100
    assert ratio <= 0.15


def test_verbatim_budget_fails_when_too_much_copied() -> None:
    original = "This is a very long passage that should not be copied verbatim."
    sterile = "This is a very long passage that should not be copied verbatim."
    ok, longest, ratio = _check_verbatim_budget(original, sterile)
    assert ok is False
    assert longest > 100 or ratio > 0.15


def test_verbatim_budget_longest_run_capped_at_100() -> None:
    original = "x" * 150
    sterile = "x" * 150
    ok, longest, ratio = _check_verbatim_budget(original, sterile)
    assert ok is False
    assert longest == 150


def test_verbatim_budget_total_ratio_capped_at_15_percent() -> None:
    original = "alpha beta gamma delta epsilon zeta eta theta"
    # Copy ~50% verbatim
    sterile = "alpha beta gamma delta epsilon zeta eta theta something else"
    ok, longest, ratio = _check_verbatim_budget(original, sterile)
    assert ok is False
    assert ratio > 0.15


@pytest.mark.asyncio
async def test_sterilize_chunks_raises_on_verbatim_budget_violation() -> None:
    chunk = _chunk(content="x" * 200)
    fake = _make_fake_complete(sterile_content="x" * 200)

    with (
        patch("vo_rag.core.compliance.sterilizer.complete_json_protected", fake),
        pytest.raises(SterilizerParseError) as exc_info,
    ):
        await sterilize_chunks([chunk], _fake_llm_client())

    assert "verbatim budget exceeded" in str(exc_info.value)


# --------------------------------------------------------------------------- #
# Invariant: license-aware redaction wires PUBLISHABLE
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_publishable_license_flag_set_for_cleared() -> None:
    chunk = _chunk(
        metadata={"license_class": LicenseClass.CLEARED.value},
    )
    fake = _make_fake_complete()

    with patch("vo_rag.core.compliance.sterilizer.complete_json_protected", fake):
        result = await sterilize_chunks([chunk], _fake_llm_client())

    assert result[0].metadata.get("_sterilization_license_publishable") is True


@pytest.mark.asyncio
async def test_publishable_license_flag_false_for_pending() -> None:
    chunk = _chunk(
        metadata={"license_class": LicenseClass.PENDING.value},
    )
    fake = _make_fake_complete()

    with patch("vo_rag.core.compliance.sterilizer.complete_json_protected", fake):
        result = await sterilize_chunks([chunk], _fake_llm_client())

    assert result[0].metadata.get("_sterilization_license_publishable") is False


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_empty_sterile_content_raises_parse_error() -> None:
    chunk = _chunk()
    fake = _make_fake_complete(sterile_content="")

    with (
        patch("vo_rag.core.compliance.sterilizer.complete_json_protected", fake),
        pytest.raises(SterilizerParseError),
    ):
        await sterilize_chunks([chunk], _fake_llm_client())


# --------------------------------------------------------------------------- #
# normalize_extracted_text is applied before sterilization
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_normalize_extracted_text_applied_before_sterilization() -> None:
    chunk = _chunk(content="co-\noperate")
    calls: list[str] = []

    async def _recording_fake(
        client: LLMClient,
        *,
        schema: type[Any],
        untrusted_content: str,
        pipeline: str,
        allowed_domains: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, LLMResponse, ValidationResult, bool]:
        calls.append(untrusted_content)
        parsed = schema.model_validate(
            {"sterile_content": "ok", "facts": [], "equations": [], "concepts": []}
        )
        resp = LLMResponse(
            text='{"sterile_content": "ok"}',
            model="fake-model",
            input_tokens=10,
            output_tokens=5,
            provider="fake",
        )
        return parsed, resp, ValidationResult(ok=True, signals=[]), False

    with patch("vo_rag.core.compliance.sterilizer.complete_json_protected", _recording_fake):
        await sterilize_chunks([chunk], _fake_llm_client())

    assert len(calls) == 1
    assert "cooperate" in calls[0]
    assert "co-\noperate" not in calls[0]

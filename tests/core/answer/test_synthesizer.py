"""Tests for ``AnswerSynthesizer`` — protected answer synthesis.

The synthesizer invariants live here:

- profile selection rejects hosted-disabled configs
- FAST and QUALITY select Anthropic models from settings
- KIMI_BENCHMARK requires allow_kimi=True
- synthesize() calls complete_json_protected() with pipeline="query_answer",
  the inherited system prompt, temperature=0.0,
  max_tokens=settings.answer.max_output_tokens, and allowed_domains=[]
  (immutable at the call site; closes F-045).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from vo_rag.core.config.settings import AnswerSettings
from vo_rag.core.llm.output_validator import ValidationResult
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.retrieval.search import SearchResult

_ANTHROPIC_TEST_KEY = "test-anthropic-key"  # pragma: allowlist secret
_KIMI_TEST_KEY = "test-kimi-key"  # pragma: allowlist secret
_FALLBACK_TEST_KEY = "test-fallback-key"  # pragma: allowlist secret
_ENV_KIMI_TEST_KEY = "test-env-kimi-key"  # pragma: allowlist secret
_LOCAL_BASE_URL = "http://127.0.0.1:8080"
_LOCAL_MODEL = "unsloth_Qwen3.6-35B-A3B-GGUF_Qwen3.6-35B-A3B-Q8_0.gguf"


def _settings(**overrides: Any) -> AnswerSettings:
    """Build an ``AnswerSettings`` with all defaults except ``overrides``."""
    defaults: dict[str, Any] = {
        "enabled": True,
        "provider": "anthropic",
        "anthropic_api_key": _ANTHROPIC_TEST_KEY,
        "quality_model": "claude-sonnet-4-6",
        "fast_model": "claude-haiku-4-5-20251001",
        "audit_model": "claude-opus-4-7",
        "kimi_model": "kimi-k2.6",
        "kimi_api_key": _KIMI_TEST_KEY,
        "kimi_base_url": "https://api.moonshot.ai/v1",
        "allow_hosted": True,
        "allow_kimi": True,
        "allow_raw_hosted_egress": False,
        "max_context_chunks": 8,
        "max_output_tokens": 1200,
        "timeout_s": 90,
    }
    defaults.update(overrides)
    return AnswerSettings(**defaults)


def _result() -> SearchResult:
    return SearchResult(
        id="point-1",
        score=0.9,
        collection="vo_rag_books_raw_v1",
        payload={
            "chunk_id": "chunk-1",
            "source_id": "source-1",
            "source_title": "Brake Handbook",
            "content": "Brake bias should be adjusted gradually.",
            "page_number": 7,
        },
    )


# --------------------------------------------------------------------------- #
# Profile selection — guards for the config-driven gates
# --------------------------------------------------------------------------- #


def test_fast_profile_selects_anthropic_fast_model() -> None:
    from vo_rag.core.answer.synthesizer import (
        AnswerProfile,
        AnswerSynthesizer,
    )

    synth = AnswerSynthesizer(_settings())
    selection = synth.select_model(AnswerProfile.FAST)
    assert selection.provider == "anthropic"
    assert selection.model == "claude-haiku-4-5-20251001"
    assert selection.api_key == _ANTHROPIC_TEST_KEY


def test_quality_profile_selects_anthropic_quality_model() -> None:
    from vo_rag.core.answer.synthesizer import (
        AnswerProfile,
        AnswerSynthesizer,
    )

    synth = AnswerSynthesizer(_settings())
    selection = synth.select_model(AnswerProfile.QUALITY)
    assert selection.provider == "anthropic"
    assert selection.model == "claude-sonnet-4-6"
    assert selection.api_key == _ANTHROPIC_TEST_KEY


def test_kimi_profile_uses_kimi_provider_and_model() -> None:
    from vo_rag.core.answer.synthesizer import (
        AnswerProfile,
        AnswerSynthesizer,
    )

    synth = AnswerSynthesizer(_settings())
    selection = synth.select_model(AnswerProfile.KIMI_BENCHMARK)
    assert selection.provider == "kimi"
    assert selection.model == "kimi-k2.6"
    assert selection.api_key == _KIMI_TEST_KEY
    assert selection.base_url == "https://api.moonshot.ai/v1"


def test_select_model_rejects_when_hosted_disabled() -> None:
    """F-045 guard: any profile (including kimi) must be rejected if
    the host has not opted into hosted answer generation."""
    from vo_rag.core.answer.synthesizer import (
        AnswerConfigError,
        AnswerProfile,
        AnswerSynthesizer,
    )

    synth = AnswerSynthesizer(_settings(allow_hosted=False, allow_kimi=True))
    with pytest.raises(AnswerConfigError, match="hosted answer generation is disabled"):
        synth.select_model(AnswerProfile.FAST)
    with pytest.raises(AnswerConfigError, match="hosted answer generation is disabled"):
        synth.select_model(AnswerProfile.QUALITY)
    with pytest.raises(AnswerConfigError, match="hosted answer generation is disabled"):
        synth.select_model(AnswerProfile.KIMI_BENCHMARK)


def test_kimi_profile_requires_allow_kimi_flag() -> None:
    """The KIMI_BENCHMARK profile is double-gated: hosted must be on AND
    allow_kimi must be on. Misconfigurations should be loud, not silent."""
    from vo_rag.core.answer.synthesizer import (
        AnswerConfigError,
        AnswerProfile,
        AnswerSynthesizer,
    )

    synth = AnswerSynthesizer(_settings(allow_hosted=True, allow_kimi=False))
    with pytest.raises(AnswerConfigError, match="Kimi benchmark answers are disabled"):
        synth.select_model(AnswerProfile.KIMI_BENCHMARK)


def test_kimi_profile_falls_back_to_kimi_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kimi key resolution: settings.kimi_api_key takes priority; if
    empty, fall back to the MOONSHOT_API_KEY / KIMI_API_KEY env var.
    Pinned to keep exfil paths through misconfigured keys from being
    a silent drop."""
    from vo_rag.core.answer.synthesizer import AnswerProfile, AnswerSynthesizer

    monkeypatch.setenv("KIMI_API_KEY", _ENV_KIMI_TEST_KEY)
    synth = AnswerSynthesizer(_settings(kimi_api_key=""))
    selection = synth.select_model(AnswerProfile.KIMI_BENCHMARK)
    assert selection.api_key == _ENV_KIMI_TEST_KEY


def test_kimi_profile_raises_when_no_key_available(monkeypatch: pytest.MonkeyPatch) -> None:
    from vo_rag.core.answer.synthesizer import (
        AnswerConfigError,
        AnswerProfile,
        AnswerSynthesizer,
    )

    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    synth = AnswerSynthesizer(_settings(kimi_api_key=""))
    with pytest.raises(AnswerConfigError, match="Kimi"):
        synth.select_model(AnswerProfile.KIMI_BENCHMARK)


def test_anthropic_profile_uses_fallback_key_when_settings_empty() -> None:
    """anthropic_fallback_key (e.g. triage.anthropic_api_key) is the
    last-resort source for FAST/QUALITY. Settings key wins when set."""
    from vo_rag.core.answer.synthesizer import AnswerProfile, AnswerSynthesizer

    synth = AnswerSynthesizer(
        _settings(anthropic_api_key=""),
        anthropic_fallback_key=_FALLBACK_TEST_KEY,
    )
    selection = synth.select_model(AnswerProfile.QUALITY)
    assert selection.api_key == _FALLBACK_TEST_KEY


def test_anthropic_profile_raises_when_no_key_available() -> None:
    from vo_rag.core.answer.synthesizer import (
        AnswerConfigError,
        AnswerProfile,
        AnswerSynthesizer,
    )

    synth = AnswerSynthesizer(
        _settings(anthropic_api_key=""),
        anthropic_fallback_key="",
    )
    with pytest.raises(AnswerConfigError, match="ANTHROPIC_API_KEY"):
        synth.select_model(AnswerProfile.QUALITY)
    with pytest.raises(AnswerConfigError, match="ANTHROPIC_API_KEY"):
        synth.select_model(AnswerProfile.FAST)


def test_local_profile_selects_llama_cpp_local_provider() -> None:
    """LOCAL routes to the on-prem llama.cpp daily-driver. It is gated on
    allow_local (NOT allow_hosted) — local inference is not external egress."""
    from vo_rag.core.answer.synthesizer import AnswerProfile, AnswerSynthesizer

    synth = AnswerSynthesizer(
        _settings(
            allow_hosted=False,
            allow_local=True,
            local_base_url=_LOCAL_BASE_URL,
            local_model=_LOCAL_MODEL,
        )
    )
    selection = synth.select_model(AnswerProfile.LOCAL)
    assert selection.provider == "local"
    assert selection.model == _LOCAL_MODEL
    assert selection.base_url == _LOCAL_BASE_URL


def test_local_profile_requires_allow_local_flag() -> None:
    from vo_rag.core.answer.synthesizer import (
        AnswerConfigError,
        AnswerProfile,
        AnswerSynthesizer,
    )

    synth = AnswerSynthesizer(
        _settings(
            allow_local=False,
            local_base_url=_LOCAL_BASE_URL,
            local_model=_LOCAL_MODEL,
        )
    )
    with pytest.raises(AnswerConfigError, match="local answer generation is disabled"):
        synth.select_model(AnswerProfile.LOCAL)


def test_local_profile_requires_base_url_and_model() -> None:
    """A misconfigured local profile must fail loudly, not silently route
    to an empty endpoint."""
    from vo_rag.core.answer.synthesizer import (
        AnswerConfigError,
        AnswerProfile,
        AnswerSynthesizer,
    )

    missing_url = AnswerSynthesizer(
        _settings(allow_local=True, local_base_url="", local_model=_LOCAL_MODEL)
    )
    with pytest.raises(AnswerConfigError, match="required for local answers"):
        missing_url.select_model(AnswerProfile.LOCAL)

    missing_model = AnswerSynthesizer(
        _settings(allow_local=True, local_base_url=_LOCAL_BASE_URL, local_model="")
    )
    with pytest.raises(AnswerConfigError, match="required for local answers"):
        missing_model.select_model(AnswerProfile.LOCAL)


# --------------------------------------------------------------------------- #
# synthesize() — protected-completion contract
# --------------------------------------------------------------------------- #


class _CapturedCall(BaseModel):
    """Snapshot of what synthesize() passed to ``complete_json_protected``."""

    schema_cls: type[BaseModel]
    pipeline: str
    system: str | None
    untrusted_content: str
    messages: list[dict[str, Any]] | None
    temperature: float | None
    max_tokens: int | None
    allowed_domains: list[str] | None


def _capture_protected(
    captured: list[_CapturedCall],
    parsed: BaseModel,
    response: LLMResponse,
    validation: ValidationResult,
    should_quarantine: bool,
) -> Any:
    """Build a fake ``complete_json_protected`` that records its kwargs."""

    async def _fake(
        client: Any,
        *,
        schema: type[BaseModel],
        untrusted_content: str,
        pipeline: str,
        allowed_domains: list[str] | None = None,
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[BaseModel, LLMResponse, ValidationResult, bool]:
        captured.append(
            _CapturedCall(
                schema_cls=schema,
                pipeline=pipeline,
                system=system,
                untrusted_content=untrusted_content,
                messages=list(messages) if messages else None,
                temperature=kwargs.get("temperature"),
                max_tokens=kwargs.get("max_tokens"),
                allowed_domains=allowed_domains,
            )
        )
        return parsed, response, validation, should_quarantine

    return _fake


@pytest.mark.asyncio
async def test_synthesize_calls_protected_with_query_answer_pipeline() -> None:
    from vo_rag.core.answer.synthesizer import (
        AnswerDraft,
        AnswerProfile,
        AnswerSynthesizer,
    )

    captured: list[_CapturedCall] = []
    draft = AnswerDraft(answer="A.", confidence=0.9, cited_chunk_ids=["chunk-1"])
    response = LLMResponse(text='{"answer":"A."}', model="claude-haiku-4-5-20251001")
    validation = ValidationResult(ok=True, signals=[])

    fake = _capture_protected(captured, draft, response, validation, False)

    with patch("vo_rag.core.answer.synthesizer.complete_json_protected", fake):
        synth = AnswerSynthesizer(_settings())
        await synth.synthesize(
            query="What is trail braking?",
            context=[_result()],
            profile=AnswerProfile.FAST,
        )

    assert len(captured) == 1
    call = captured[0]
    assert call.pipeline == "query_answer"
    assert call.schema_cls is AnswerDraft
    assert call.temperature == 0.0
    assert call.max_tokens == 1200  # = settings.answer.max_output_tokens
    assert call.allowed_domains == []  # immutable at call site (F-045)


@pytest.mark.asyncio
async def test_synthesize_uses_inherited_system_prompt_semantics() -> None:
    """Pin the system prompt's load-
    bearing rules so a future edit can't silently weaken the citation
    or untrusted-context guarantees."""
    from vo_rag.core.answer.synthesizer import AnswerProfile, AnswerSynthesizer

    captured: list[_CapturedCall] = []
    from vo_rag.core.answer.synthesizer import AnswerDraft

    fake = _capture_protected(
        captured,
        AnswerDraft(answer="A.", confidence=0.9),
        LLMResponse(text='{"answer":"A."}', model="claude-haiku-4-5-20251001"),
        ValidationResult(ok=True, signals=[]),
        False,
    )

    with patch("vo_rag.core.answer.synthesizer.complete_json_protected", fake):
        synth = AnswerSynthesizer(_settings())
        await synth.synthesize(
            query="q",
            context=[_result()],
            profile=AnswerProfile.FAST,
        )

    system = captured[0].system or ""
    assert "untrusted" in system.lower()  # treat context as untrusted
    assert "JSON" in system  # JSON-only contract
    assert "[S1]" in system  # cite every factual sentence
    assert "no long" in system.lower() or "concise" in system.lower()  # no long quotes


@pytest.mark.asyncio
async def test_synthesize_uses_settings_max_output_tokens_not_hardcoded() -> None:
    """The max_tokens argument MUST be settings.answer.max_output_tokens,
    not a hardcoded fallback. Pinned to catch a regression that would
    silently cap generation."""
    from vo_rag.core.answer.synthesizer import AnswerDraft, AnswerProfile, AnswerSynthesizer

    captured: list[_CapturedCall] = []
    fake = _capture_protected(
        captured,
        AnswerDraft(answer="A.", confidence=0.9),
        LLMResponse(text='{"answer":"A."}', model="claude-haiku-4-5-20251001"),
        ValidationResult(ok=True, signals=[]),
        False,
    )

    with patch("vo_rag.core.answer.synthesizer.complete_json_protected", fake):
        synth = AnswerSynthesizer(_settings(max_output_tokens=2048))
        await synth.synthesize(
            query="q",
            context=[_result()],
            profile=AnswerProfile.QUALITY,
        )

    assert captured[0].max_tokens == 2048


@pytest.mark.asyncio
async def test_synthesize_passes_untrusted_context_via_format_answer_context() -> None:
    """The untrusted_content payload is the formatted answer context
    (S1/S2 markers, point/chunk/source ids, content). complete_json_protected
    wraps it in <document> tags + canary downstream — the synthesizer's
    job is just to ship the formatted block, not to re-validate."""
    from vo_rag.core.answer.synthesizer import AnswerProfile, AnswerSynthesizer

    captured: list[_CapturedCall] = []
    from vo_rag.core.answer.synthesizer import AnswerDraft

    fake = _capture_protected(
        captured,
        AnswerDraft(answer="A.", confidence=0.9),
        LLMResponse(text='{"answer":"A."}', model="claude-haiku-4-5-20251001"),
        ValidationResult(ok=True, signals=[]),
        False,
    )

    with patch("vo_rag.core.answer.synthesizer.complete_json_protected", fake):
        synth = AnswerSynthesizer(_settings())
        await synth.synthesize(
            query="What is trail braking?",
            context=[_result()],
            profile=AnswerProfile.FAST,
        )

    untrusted = captured[0].untrusted_content
    assert "[S1]" in untrusted
    assert "point_id: point-1" in untrusted
    assert "chunk_id: chunk-1" in untrusted
    assert "source_id: source-1" in untrusted
    assert "Brake Handbook" in untrusted  # title rendered
    assert "page: 7" in untrusted
    assert "Brake bias should be adjusted gradually." in untrusted  # content


@pytest.mark.asyncio
async def test_synthesize_builds_query_user_message() -> None:
    """The user message is the question + a JSON-only reminder. The
    system prompt already enforces JSON-only, so this is a belt-and-
    suspenders hint to the model."""
    from vo_rag.core.answer.synthesizer import AnswerDraft, AnswerProfile, AnswerSynthesizer

    captured: list[_CapturedCall] = []
    fake = _capture_protected(
        captured,
        AnswerDraft(answer="A.", confidence=0.9),
        LLMResponse(text='{"answer":"A."}', model="claude-haiku-4-5-20251001"),
        ValidationResult(ok=True, signals=[]),
        False,
    )

    with patch("vo_rag.core.answer.synthesizer.complete_json_protected", fake):
        synth = AnswerSynthesizer(_settings())
        await synth.synthesize(
            query="What is trail braking?",
            context=[_result()],
            profile=AnswerProfile.FAST,
        )

    messages = captured[0].messages
    assert messages is not None
    assert len(messages) == 1
    msg = messages[0]
    assert msg["role"] == "user"
    assert "What is trail braking?" in msg["content"]
    assert "JSON" in msg["content"]


@pytest.mark.asyncio
async def test_synthesize_returns_answer_synthesis_with_model() -> None:
    """The synthesis envelope must carry the model name back to the
    caller (for logging/audit). It comes from the selection, not from
    the LLMResponse — different providers might report different model
    strings than the configured one."""
    from vo_rag.core.answer.synthesizer import (
        AnswerDraft,
        AnswerProfile,
        AnswerSynthesizer,
    )

    draft = AnswerDraft(answer="A.", confidence=0.9)
    response = LLMResponse(text='{"answer":"A."}', model="claude-sonnet-4-6")
    validation = ValidationResult(ok=True, signals=[])

    fake = _capture_protected([], draft, response, validation, False)

    with patch("vo_rag.core.answer.synthesizer.complete_json_protected", fake):
        synth = AnswerSynthesizer(_settings())
        synthesis = await synth.synthesize(
            query="q",
            context=[_result()],
            profile=AnswerProfile.QUALITY,
        )

    assert synthesis.model == "claude-sonnet-4-6"
    assert synthesis.draft is draft
    assert synthesis.llm_response is response
    assert synthesis.validation is validation
    assert synthesis.should_quarantine is False


@pytest.mark.asyncio
async def test_synthesize_propagates_quarantine_flag() -> None:
    """should_quarantine flows through the envelope so callers can
    decide whether to persist to a *_quarantine_v1 table."""
    from vo_rag.core.answer.synthesizer import (
        AnswerDraft,
        AnswerProfile,
        AnswerSynthesizer,
    )

    draft = AnswerDraft(answer="", confidence=0.0)
    response = LLMResponse(text='{"answer":""}', model="claude-haiku-4-5-20251001")
    validation = ValidationResult(ok=False, signals=[])

    fake = _capture_protected([], draft, response, validation, True)

    with patch("vo_rag.core.answer.synthesizer.complete_json_protected", fake):
        synth = AnswerSynthesizer(_settings())
        synthesis = await synth.synthesize(
            query="q",
            context=[_result()],
            profile=AnswerProfile.FAST,
        )

    assert synthesis.should_quarantine is True


@pytest.mark.asyncio
async def test_synthesize_kimi_profile_builds_kimi_provider() -> None:
    """KIMI_BENCHMARK profile routes through ``KimiProvider``; the
    synthesizer must close that client (and not the Anthropic one)
    when the call finishes. The httpx.AsyncClient inside KimiProvider
    is built but never used because we patch the protected call."""
    from vo_rag.core.answer.synthesizer import (
        AnswerDraft,
        AnswerProfile,
        AnswerSynthesizer,
    )

    captured: list[_CapturedCall] = []
    draft = AnswerDraft(answer="A.", confidence=0.9)
    response = LLMResponse(text='{"answer":"A."}', model="kimi-k2.6")
    validation = ValidationResult(ok=True, signals=[])

    fake = _capture_protected(captured, draft, response, validation, False)

    with patch("vo_rag.core.answer.synthesizer.complete_json_protected", fake):
        synth = AnswerSynthesizer(_settings())
        synthesis = await synth.synthesize(
            query="q",
            context=[_result()],
            profile=AnswerProfile.KIMI_BENCHMARK,
        )

    assert len(captured) == 1
    assert synthesis.model == "kimi-k2.6"


@pytest.mark.asyncio
async def test_synthesize_local_profile_builds_llama_cpp_provider() -> None:
    """LOCAL profile routes through ``LlamaCppProvider`` against the on-prem
    daily-driver. Synthesis is parse-and-retry (no guided ``extra``), so the
    provider that drops ``extra`` is the right fit. The httpx client is built
    but unused because the protected call is patched; synthesize() closes it."""
    from vo_rag.core.answer.synthesizer import (
        AnswerDraft,
        AnswerProfile,
        AnswerSynthesizer,
    )

    draft = AnswerDraft(answer="A.", confidence=0.9)
    response = LLMResponse(text='{"answer":"A."}', model=_LOCAL_MODEL)
    validation = ValidationResult(ok=True, signals=[])

    fake = _capture_protected([], draft, response, validation, False)

    with patch("vo_rag.core.answer.synthesizer.complete_json_protected", fake):
        synth = AnswerSynthesizer(
            _settings(
                allow_hosted=False,
                allow_local=True,
                local_base_url=_LOCAL_BASE_URL,
                local_model=_LOCAL_MODEL,
            )
        )
        synthesis = await synth.synthesize(
            query="q",
            context=[_result()],
            profile=AnswerProfile.LOCAL,
        )

    assert synthesis.model == _LOCAL_MODEL

"""Tests for LLMClient provider factory wiring."""

from __future__ import annotations

import pytest

from vo_rag.core.llm.client import _build_provider
from vo_rag.core.llm.providers.anthropic_provider import AnthropicProvider
from vo_rag.core.llm.providers.kimi_provider import KimiProvider
from vo_rag.core.llm.providers.llama_cpp_provider import LlamaCppProvider


def test_build_provider_anthropic() -> None:
    provider = _build_provider(
        "anthropic",
        model="claude-3-sonnet",
        api_key="test-key",
        base_url=None,
    )
    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "claude-3-sonnet"


def test_build_provider_kimi() -> None:
    provider = _build_provider(
        "kimi",
        model="kimi-k2.6",
        api_key="test-key",
        base_url=None,
    )
    assert isinstance(provider, KimiProvider)
    assert provider.model == "kimi-k2.6"


def test_build_provider_llama_cpp() -> None:
    provider = _build_provider(
        "llama_cpp",
        model="qwen3-4b",
        api_key=None,
        base_url="http://localhost:8080",
    )
    assert isinstance(provider, LlamaCppProvider)
    assert provider.model == "qwen3-4b"


def test_build_provider_unknown_raises() -> None:
    with pytest.raises(ValueError):
        _build_provider("unknown", model="x", api_key="k")  # type: ignore[arg-type]


def test_build_provider_passes_through_instance() -> None:
    """Provider instances are returned as-is."""
    existing = AnthropicProvider(api_key="k", model="m")
    result = _build_provider(existing, model=None, api_key=None)
    assert result is existing

"""Capability-flag tests for VoRagLLM."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from vo_rag.integrations.llamaindex.llm import VoRagLLM


def _fake_client(provider_name: str = "anthropic", model: str = "claude-x") -> Any:
    client = MagicMock()
    client.provider.name = provider_name
    client.provider.model = model
    return client


def test_capability_flag_is_false_for_anthropic() -> None:
    llm = VoRagLLM(client=_fake_client("anthropic"))
    assert llm.supports_guided_decoding is False


def test_capability_flag_is_false_for_llama_cpp() -> None:
    llm = VoRagLLM(client=_fake_client("llama_cpp"))
    assert llm.supports_guided_decoding is False


def test_capability_flag_is_true_for_kimi() -> None:
    llm = VoRagLLM(client=_fake_client("kimi"))
    assert llm.supports_guided_decoding is True


def test_capability_flag_is_true_for_runpod_serverless() -> None:
    llm = VoRagLLM(client=_fake_client("runpod_serverless"))
    assert llm.supports_guided_decoding is True

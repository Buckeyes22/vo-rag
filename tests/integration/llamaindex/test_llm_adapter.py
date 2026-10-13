"""Contract tests for VoRagLLM adapter — telemetry, metadata, and the
injection-defense boundary (complete must NOT reach complete_json_protected).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import llama_index_instrumentation as instrument
import pytest
from llama_index.core.llms import ChatMessage, MessageRole

from vo_rag.core.llm.client import LLMClient, LLMStreamEvent
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.observability.cost import cost_context
from vo_rag.integrations.llamaindex.llm import VoRagLLM
from vo_rag.integrations.llamaindex.observability import (
    VoRagBridgeHandler,
    VoRagSpanHandler,
    install_bridge,
)


def _fake_client(
    provider_name: str = "anthropic",
    model: str = "claude-x",
    response_text: str = "hi",
) -> Any:
    async def _stream_complete(**_: Any) -> Any:
        yield LLMStreamEvent(
            delta=response_text,
            text=response_text,
            model=model,
            provider=provider_name,
            raw={"model": model},
        )

    client = MagicMock()
    client.provider.name = provider_name
    client.provider.model = model
    client.complete = AsyncMock(
        return_value=LLMResponse(
            text=response_text,
            model=model,
            provider=provider_name,
            latency_ms=12.0,
            attempts=1,
        )
    )
    client.stream_complete = _stream_complete
    return client


class _ScriptedProvider:
    name = "scripted"

    def __init__(self, model: str = "scripted-model") -> None:
        self._model = model
        self.calls = 0

    @property
    def model(self) -> str:
        return self._model

    async def complete_raw(self, **kwargs: Any) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text="hello",
            model=self._model,
            provider=self.name,
            input_tokens=13,
            output_tokens=8,
            latency_ms=12.0,
        )

    async def aclose(self) -> None:
        return None


def _clear_vo_rag_bridge_handlers() -> None:
    dispatcher = instrument.get_dispatcher()
    dispatcher.event_handlers = [
        h for h in dispatcher.event_handlers if not isinstance(h, VoRagBridgeHandler)
    ]
    dispatcher.span_handlers = [
        h for h in dispatcher.span_handlers if not isinstance(h, VoRagSpanHandler)
    ]


def test_complete_delegates_and_preserves_telemetry() -> None:
    llm = VoRagLLM(client=_fake_client())
    out = llm.complete("ping")
    assert out.text == "hi"
    assert out.raw["latency_ms"] == 12.0
    assert out.raw["attempts"] == 1


def test_metadata_reports_active_provider_model() -> None:
    llm = VoRagLLM(client=_fake_client(model="claude-x"))
    assert llm.metadata.model_name == "claude-x"


def test_capability_flag_is_honest() -> None:
    assert VoRagLLM(client=_fake_client("anthropic")).supports_guided_decoding is False
    assert VoRagLLM(client=_fake_client("kimi")).supports_guided_decoding is True


def test_capability_flag_is_derived_from_provider() -> None:
    client = _fake_client("anthropic")
    llm = VoRagLLM(client=client)
    assert llm.supports_guided_decoding is False

    client.provider.name = "kimi"

    assert llm.supports_guided_decoding is True


def test_complete_warns_on_unconsumed_llamaindex_kwargs(monkeypatch: Any) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        "vo_rag.integrations.llamaindex.llm.logger.warning",
        lambda msg, *args: warnings.append(msg % args),
    )

    VoRagLLM(client=_fake_client()).complete("ping", stop=["END"], custom_flag=True)

    assert warnings == ["VoRagLLM ignored unsupported LlamaIndex kwargs: custom_flag, stop"]


def test_acomplete_runs_inside_event_loop() -> None:
    llm = VoRagLLM(client=_fake_client())

    async def _drive() -> Any:
        return await llm.acomplete("ping")

    assert asyncio.run(_drive()).text == "hi"


def test_chat_works_outside_event_loop() -> None:
    llm = VoRagLLM(client=_fake_client(response_text="hello"))

    response = llm.chat([ChatMessage(role=MessageRole.USER, content="hi")])

    assert response.message.role == MessageRole.ASSISTANT
    assert response.message.content == "hello"
    assert response.raw["latency_ms"] == 12.0


def test_achat_uses_async_client_without_run_coro_sync() -> None:
    llm = VoRagLLM(client=_fake_client(response_text="hello"))

    async def _drive() -> Any:
        return await llm.achat([ChatMessage(role=MessageRole.USER, content="hi")])

    response = asyncio.run(_drive())
    assert response.message.role == MessageRole.ASSISTANT
    assert response.message.content == "hello"
    assert response.raw["latency_ms"] == 12.0


def test_achat_under_cost_context_with_bridge_records_one_cost_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _ScriptedProvider()
    llm = VoRagLLM(client=LLMClient(provider=provider))
    monkeypatch.setattr("vo_rag.core.llm.client._compute_cost", lambda _resp: 0.0051)

    async def _drive() -> tuple[Any, dict[str, dict[str, Any]], dict[str, int], float]:
        with cost_context(envelope_id="env-229", pipeline="test_pipeline") as acc:
            response = await llm.achat([ChatMessage(role=MessageRole.USER, content="hi")])
            return response, acc.step_breakdown(), acc.total_tokens(), acc.total_usd()

    _clear_vo_rag_bridge_handlers()
    try:
        install_bridge()
        response, breakdown, tokens, total_usd = asyncio.run(_drive())
    finally:
        _clear_vo_rag_bridge_handlers()

    assert provider.calls == 1
    assert response.message.role == MessageRole.ASSISTANT
    assert response.message.content == "hello"
    assert breakdown == {
        "llm|scripted-model": {
            "step": "llm",
            "model": "scripted-model",
            "input_tokens": 13,
            "output_tokens": 8,
            "cache_read_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "cost_usd": 0.0051,
            "calls": 1,
        }
    }
    assert tokens == {
        "total_input": 13,
        "total_output": 8,
        "total_cache_read": 0,
        "total_cache_write": 0,
    }
    assert total_usd == pytest.approx(0.0051)


def test_stream_chat_works_outside_event_loop() -> None:
    llm = VoRagLLM(client=_fake_client(response_text="hello"))

    response = next(llm.stream_chat([ChatMessage(role=MessageRole.USER, content="hi")]))

    assert response.message.role == MessageRole.ASSISTANT
    assert response.message.content == "hello"
    assert response.delta == "hello"
    assert response.raw["model"] == "claude-x"


def test_astream_chat_uses_async_client_without_run_coro_sync() -> None:
    llm = VoRagLLM(client=_fake_client(response_text="hello"))

    async def _drive() -> list[Any]:
        return [
            response
            async for response in await llm.astream_chat(
                [ChatMessage(role=MessageRole.USER, content="hi")]
            )
        ]

    responses = asyncio.run(_drive())
    assert len(responses) == 1
    response = responses[0]
    assert response.message.role == MessageRole.ASSISTANT
    assert response.message.content == "hello"
    assert response.delta == "hello"
    assert response.raw["model"] == "claude-x"

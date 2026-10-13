"""Tests for gen_ai OpenTelemetry spans emitted by LLMClient."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from typing import Any

import httpx
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import BaseModel

from vo_rag.core.llm.client import LLMClient
from vo_rag.core.llm.errors import StructuredOutputError
from vo_rag.core.llm.providers.kimi_provider import KimiProvider
from vo_rag.core.llm.response import LLMResponse


class _Answer(BaseModel):
    value: int


class _ScriptedProvider:
    def __init__(
        self,
        *,
        name: str,
        model: str,
        responses: Sequence[LLMResponse],
    ) -> None:
        self.name = name
        self.model = model
        self._responses = list(responses)
        self.calls = 0

    async def complete_raw(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        extra: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        return self._responses.pop(0)

    async def aclose(self) -> None:
        return None


class _RaisingProvider:
    name = "llama_cpp"
    model = "qwen3-test"

    async def complete_raw(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        extra: dict[str, Any] | None = None,
    ) -> LLMResponse:
        raise RuntimeError("provider exploded")

    async def aclose(self) -> None:
        return None


class _SSEByteStream(httpx.AsyncByteStream):
    def __init__(self, payloads: Iterable[dict[str, Any]]) -> None:
        self._chunks = [f"data: {json.dumps(payload)}\n\n".encode() for payload in payloads]
        self._chunks.append(b"data: [DONE]\n\n")

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


@pytest.fixture
def span_exporter(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    once = trace._TRACER_PROVIDER_SET_ONCE

    monkeypatch.setattr(trace, "_TRACER_PROVIDER", None, raising=False)
    monkeypatch.setattr(once, "_done", False, raising=False)
    trace.set_tracer_provider(provider)

    yield exporter

    provider.shutdown()


def _single_span(exporter: InMemorySpanExporter) -> Any:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def _attribute_values(span: Any) -> set[str]:
    return {str(value) for value in span.attributes.values()}


def _kimi_provider(handler: Any) -> KimiProvider:
    return KimiProvider(
        api_key="test-key",  # pragma: allowlist secret
        model="kimi-k2.6",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://api.moonshot.ai/v1",
        ),
    )


@pytest.mark.asyncio
async def test_complete_emits_gen_ai_span_with_string_usage(
    span_exporter: InMemorySpanExporter,
) -> None:
    provider = _ScriptedProvider(
        name="anthropic",
        model="claude-test",
        responses=[
            LLMResponse(
                text="private response text",
                model="claude-test",
                input_tokens=13,
                output_tokens=21,
                provider="anthropic",
            )
        ],
    )
    client = LLMClient(provider=provider)

    resp = await client.complete(
        system="private system prompt",
        messages=[{"role": "user", "content": "private user prompt"}],
    )

    span = _single_span(span_exporter)
    attrs = span.attributes
    assert resp.text == "private response text"
    assert span.name == "chat claude-test"
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.system"] == "anthropic"
    assert attrs["gen_ai.request.model"] == "claude-test"
    assert attrs["gen_ai.usage.input_tokens"] == "13"
    assert attrs["gen_ai.usage.output_tokens"] == "21"
    assert isinstance(attrs["gen_ai.usage.input_tokens"], str)
    assert isinstance(attrs["gen_ai.usage.output_tokens"], str)
    assert "private system prompt" not in _attribute_values(span)
    assert "private user prompt" not in _attribute_values(span)
    assert "private response text" not in _attribute_values(span)


@pytest.mark.asyncio
async def test_complete_json_uses_one_openai_protocol_span_for_schema_retries(
    span_exporter: InMemorySpanExporter,
) -> None:
    provider = _ScriptedProvider(
        name="runpod_serverless",
        model="runpod-model",
        responses=[
            LLMResponse(
                text='{"value": "bad"}',
                model="runpod-model",
                input_tokens=3,
                output_tokens=4,
                provider="runpod_serverless",
            ),
            LLMResponse(
                text='{"value": 42}',
                model="runpod-model",
                input_tokens=5,
                output_tokens=6,
                provider="runpod_serverless",
            ),
        ],
    )
    client = LLMClient(provider=provider)

    parsed, resp = await client.complete_json(
        schema=_Answer,
        system="schema system text",
        messages=[{"role": "user", "content": "schema user text"}],
        max_retries=1,
    )

    span = _single_span(span_exporter)
    attrs = span.attributes
    assert parsed.value == 42
    assert resp.attempts == 2
    assert provider.calls == 2
    assert span.name == "chat runpod-model"
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.system"] == "openai"
    assert attrs["gen_ai.request.model"] == "runpod-model"
    assert attrs["gen_ai.usage.input_tokens"] == "5"
    assert attrs["gen_ai.usage.output_tokens"] == "6"
    assert isinstance(attrs["gen_ai.usage.input_tokens"], str)
    assert isinstance(attrs["gen_ai.usage.output_tokens"], str)
    assert "schema system text" not in _attribute_values(span)
    assert "schema user text" not in _attribute_values(span)
    assert '{"value": 42}' not in _attribute_values(span)


@pytest.mark.asyncio
async def test_error_span_records_exception_without_prompt_attributes(
    span_exporter: InMemorySpanExporter,
) -> None:
    client = LLMClient(provider=_RaisingProvider())

    with pytest.raises(RuntimeError, match="provider exploded"):
        await client.complete(
            system="do not put this system text on the span",
            messages=[{"role": "user", "content": "do not put this user text on the span"}],
        )

    span = _single_span(span_exporter)
    assert span.name == "chat qwen3-test"
    assert span.status.status_code == StatusCode.ERROR
    assert any(event.name == "exception" for event in span.events)
    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "qwen3-test"
    assert "do not put this system text on the span" not in _attribute_values(span)
    assert "do not put this user text on the span" not in _attribute_values(span)


@pytest.mark.asyncio
async def test_complete_json_failure_does_not_leak_response_text_into_span(
    span_exporter: InMemorySpanExporter,
) -> None:
    # Regression: StructuredOutputError embeds the raw
    # model response in its message; span.record_exception(exc) would serialize
    # that into the exception event (exception.message / stacktrace). The error
    # path must record exception TYPE only — never the response text.
    sentinel = "CANARY-model-response-must-not-leak"
    bad = f'{{"value": "{sentinel}"}}'
    provider = _ScriptedProvider(
        name="runpod_serverless",
        model="runpod-model",
        responses=[
            LLMResponse(
                text=bad,
                model="runpod-model",
                input_tokens=1,
                output_tokens=2,
                provider="runpod_serverless",
            )
            for _ in range(2)
        ],
    )
    client = LLMClient(provider=provider)

    with pytest.raises(StructuredOutputError):
        await client.complete_json(
            schema=_Answer,
            system="schema system text",
            messages=[{"role": "user", "content": "schema user text"}],
            max_retries=1,
        )

    span = _single_span(span_exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert any(event.name == "exception" for event in span.events)

    surfaces = [str(span.attributes), str(span.status.description)]
    for event in span.events:
        surfaces.append(event.name)
        surfaces.append(str(dict(event.attributes or {})))
    assert all(sentinel not in surface for surface in surfaces), (
        "model response text leaked into a span surface"
    )


@pytest.mark.asyncio
async def test_stream_complete_keeps_span_open_until_stream_is_consumed(
    span_exporter: InMemorySpanExporter,
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=_SSEByteStream(
                [
                    {
                        "model": "kimi-k2.6",
                        "choices": [{"delta": {"content": "Hel"}, "finish_reason": None}],
                    },
                    {
                        "model": "kimi-k2.6",
                        "choices": [{"delta": {"content": "lo"}, "finish_reason": None}],
                    },
                    {
                        "model": "kimi-k2.6",
                        "choices": [],
                        "usage": {"prompt_tokens": 7, "completion_tokens": 2},
                    },
                ]
            ),
        )

    client = LLMClient(provider=_kimi_provider(handler))
    events = []
    agen = client.stream_complete(
        system="stream system text",
        messages=[{"role": "user", "content": "stream user text"}],
    )

    first = await anext(agen)
    events.append(first)
    assert span_exporter.get_finished_spans() == ()
    events.extend([event async for event in agen])

    span = _single_span(span_exporter)
    attrs = span.attributes
    assert [event.delta for event in events] == ["Hel", "lo"]
    assert events[-1].text == "Hello"
    assert span.name == "chat kimi-k2.6"
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.system"] == "openai"
    assert attrs["gen_ai.request.model"] == "kimi-k2.6"
    assert attrs["gen_ai.usage.input_tokens"] == "7"
    assert attrs["gen_ai.usage.output_tokens"] == "2"
    assert isinstance(attrs["gen_ai.usage.input_tokens"], str)
    assert isinstance(attrs["gen_ai.usage.output_tokens"], str)
    assert "stream system text" not in _attribute_values(span)
    assert "stream user text" not in _attribute_values(span)
    assert "Hello" not in _attribute_values(span)

"""Streaming contract tests for LLMClient.

The provider is exercised through an httpx MockTransport with OpenAI-style
SSE frames. No live endpoint is contacted.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import httpx
import pytest

from vo_rag.core.llm.client import LLMClient
from vo_rag.core.llm.providers.kimi_provider import KimiProvider
from vo_rag.core.llm.providers.llama_cpp_provider import LlamaCppProvider
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.observability.cost import cost_context


class _SSEByteStream(httpx.AsyncByteStream):
    def __init__(self, payloads: Iterable[dict[str, Any]]) -> None:
        self._chunks = [f"data: {json.dumps(payload)}\n\n".encode() for payload in payloads]
        self._chunks.append(b"data: [DONE]\n\n")
        self.closed = False

    async def __aiter__(self) -> Any:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


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
async def test_stream_complete_yields_deltas_and_records_cost_from_final_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
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

    monkeypatch.setattr(
        "vo_rag.core.llm.client._compute_cost",
        lambda resp: 0.051 if isinstance(resp, LLMResponse) else 0.0,
    )
    client = LLMClient(provider=_kimi_provider(handler))

    with cost_context(envelope_id="env-stream", pipeline="test") as acc:
        events = [
            event
            async for event in client.stream_complete(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=8,
            )
        ]
        breakdown = acc.step_breakdown()

    assert [event.delta for event in events] == ["Hel", "lo"]
    assert events[-1].text == "Hello"
    assert requests[0]["stream"] is True
    assert requests[0]["stream_options"] == {"include_usage": True}
    assert breakdown == {
        "llm|kimi-k2.6": {
            "step": "llm",
            "model": "kimi-k2.6",
            "input_tokens": 7,
            "output_tokens": 2,
            "cache_read_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "cost_usd": 0.051,
            "calls": 1,
        }
    }


@pytest.mark.asyncio
async def test_stream_complete_governance_abort_happens_before_provider_call() -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    class KillSwitchBudget:
        async def charge_estimated(self, *_args: Any) -> None:
            raise RuntimeError("kill-switch engaged")

    client = LLMClient(provider=_kimi_provider(handler), budget=KillSwitchBudget())

    with pytest.raises(RuntimeError, match="kill-switch engaged"):
        _ = [
            event
            async for event in client.stream_complete(
                messages=[{"role": "user", "content": "blocked"}],
            )
        ]

    assert calls == 0


@pytest.mark.asyncio
async def test_stream_complete_non_streaming_provider_raises_without_faking_chunk() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "fake"}}]})

    provider = LlamaCppProvider(
        base_url="http://llm",
        model="qwen3-x",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://llm"),
    )
    client = LLMClient(provider=provider)

    with pytest.raises(NotImplementedError, match="does not support real token streaming"):
        _ = [
            event
            async for event in client.stream_complete(
                messages=[{"role": "user", "content": "do not fake"}],
            )
        ]

    assert calls == 0


@pytest.mark.asyncio
async def test_stream_complete_closes_http_stream_when_consumer_exits_early() -> None:
    stream = _SSEByteStream(
        [
            {"choices": [{"delta": {"content": "first"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "second"}, "finish_reason": None}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
        ]
    )

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    client = LLMClient(provider=_kimi_provider(handler))

    agen = client.stream_complete(messages=[{"role": "user", "content": "stop early"}])
    first = await anext(agen)
    await agen.aclose()

    assert first.delta == "first"
    assert stream.closed is True

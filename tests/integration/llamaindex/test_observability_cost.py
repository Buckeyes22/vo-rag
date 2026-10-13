"""Contract tests for LlamaIndex -> vo-rag metric/cost translation.

Verifies that VoRagBridgeHandler turns synthetic LlamaIndex LLM events
into vo-rag Prometheus counter increments and CostAccumulator attributions,
and that poisoned model labels are blocked before they reach /metrics.
"""

from __future__ import annotations

from typing import Any

import pytest
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
)
from llama_index.core.llms import ChatMessage, ChatResponse

from vo_rag.core.observability.cost import CostAccumulator, cost_context
from vo_rag.core.observability.metrics import (
    UnsafeLabel,
    llm_calls_total,
    llm_usage_extraction_failed_total,
)
from vo_rag.integrations.llamaindex.observability import VoRagBridgeHandler


@pytest.fixture
def handler() -> VoRagBridgeHandler:
    return VoRagBridgeHandler()


def test_cost_accumulator_extracts_usage_from_llamaindex_chat_response() -> None:
    response = ChatResponse(
        message=ChatMessage(role="assistant", content="ok"),
        raw={
            "model": "qwen3",
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            "cost_usd": 0.0042,
        },
    )

    acc = CostAccumulator(envelope_id="env-229", pipeline="test_pipeline")
    acc.add(step="llamaindex_llm", response=response)

    item = next(iter(acc.by_key.values()))
    assert item.model == "qwen3"
    assert item.input_tokens == 11
    assert item.output_tokens == 7
    assert item.cost_usd == 0.0042


def test_cost_accumulator_extracts_usage_from_vorag_llm_flat_raw() -> None:
    response = ChatResponse(
        message=ChatMessage(role="assistant", content="ok"),
        raw={
            "model": "qwen3",
            "input_tokens": 13,
            "output_tokens": 8,
            "cost_usd": 0.0051,
        },
    )

    acc = CostAccumulator(envelope_id="env-229", pipeline="test_pipeline")
    acc.add(step="llm", response=response)

    item = next(iter(acc.by_key.values()))
    assert item.model == "qwen3"
    assert item.input_tokens == 13
    assert item.output_tokens == 8
    assert item.cost_usd == 0.0051


def test_cost_accumulator_warns_when_known_model_has_zero_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[dict[str, Any]] = []

    class _Logger:
        def warning(self, event: str, **kwargs: Any) -> None:
            warnings.append({"event": event, **kwargs})

    monkeypatch.setattr("vo_rag.core.observability.cost._log", _Logger())
    response = ChatResponse(
        message=ChatMessage(role="assistant", content="ok"),
        raw={
            "model": "qwen3",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        },
    )

    acc = CostAccumulator(envelope_id="env-229", pipeline="test_pipeline")
    before = llm_usage_extraction_failed_total._value.get()
    acc.add(step="llm", response=response)

    assert warnings == [
        {
            "event": "llm_usage_shape_unknown_zero_tokens",
            "step": "llm",
            "model": "qwen3",
            "response_type": "ChatResponse",
        }
    ]
    assert llm_usage_extraction_failed_total._value.get() == before + 1


def test_llm_chat_end_event_does_not_attribute_cost_inside_cost_context(
    handler: VoRagBridgeHandler,
) -> None:
    response = ChatResponse(
        message=ChatMessage(role="assistant", content="ok"),
        raw={
            "model": "qwen3",
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            "cost_usd": 0.0042,
        },
    )
    event = LLMChatEndEvent(messages=[], response=response)

    with cost_context(envelope_id="env-229", pipeline="test_pipeline") as acc:
        handler.handle(event)

    assert acc.total_usd() == pytest.approx(0.0)
    assert acc.step_breakdown() == {}


def test_llm_chat_start_event_increments_llm_calls_total(
    handler: VoRagBridgeHandler,
) -> None:
    model = "gpt-4"
    labels = {
        "provider": "llamaindex",
        "model": model,
        "prompt_name": "unknown",
        "result": "started",
    }
    before = llm_calls_total.labels(**labels)._value.get()

    event = LLMChatStartEvent(
        messages=[],
        additional_kwargs={},
        model_dict={"model": model},
    )
    handler.handle(event)

    after = llm_calls_total.labels(**labels)._value.get()
    assert after == before + 1


def test_poisoned_model_label_raises_unsafe_label(
    handler: VoRagBridgeHandler,
) -> None:
    event = LLMChatStartEvent(
        messages=[],
        additional_kwargs={},
        model_dict={"model": "user@example.com"},
    )
    with pytest.raises(UnsafeLabel):
        handler.handle(event)


def test_bridge_handle_synthetic_event_returns_safely(
    handler: VoRagBridgeHandler,
) -> None:
    assert handler.handle("synthetic_event") is None

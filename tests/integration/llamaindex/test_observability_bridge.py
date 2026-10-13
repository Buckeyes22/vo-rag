"""Contract tests for the LlamaIndex observability bridge.

Verifies the ingestion seam (VoRagBridgeHandler + VoRagSpanHandler
registered on the root dispatcher) without testing the translation
bodies — those are VOR-229 and VOR-231.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import Any

import llama_index_instrumentation as instrument
import pytest
from llama_index.core.instrumentation.events.llm import LLMChatEndEvent
from llama_index.core.llms import ChatMessage, ChatResponse

from vo_rag.core.observability.cost import cost_context
from vo_rag.core.observability.metrics import (
    UnsafeLabel,
    pipeline_duration_seconds,
)
from vo_rag.integrations.llamaindex.observability import (
    VoRagBridgeHandler,
    VoRagSpanHandler,
    install_bridge,
)


def _clear_vo_rag_handlers() -> None:
    """Remove any vo-rag bridge handlers from the root dispatcher."""
    dispatcher = instrument.get_dispatcher()
    dispatcher.event_handlers = [
        h for h in dispatcher.event_handlers if not isinstance(h, VoRagBridgeHandler)
    ]
    dispatcher.span_handlers = [
        h for h in dispatcher.span_handlers if not isinstance(h, VoRagSpanHandler)
    ]


def _empty_bound_args() -> inspect.BoundArguments:
    return inspect.signature(lambda: None).bind()


def _histogram_sample_value(
    metric: Any,
    sample_name: str,
    labels: dict[str, str],
) -> float:
    for family in metric.collect():
        for sample in family.samples:
            if sample.name == sample_name and sample.labels == labels:
                return float(sample.value)
    return 0.0


def _histogram_has_label(metric: Any, label_name: str, label_value: str) -> bool:
    return any(
        sample.labels.get(label_name) == label_value
        for family in metric.collect()
        for sample in family.samples
    )


@pytest.fixture(autouse=True)
def _cleanup_bridge() -> Iterator[None]:
    """Ensure each test starts with a clean dispatcher."""
    _clear_vo_rag_handlers()
    yield
    _clear_vo_rag_handlers()


def test_install_bridge_attaches_both_handlers_to_root_dispatcher() -> None:
    install_bridge()
    dispatcher = instrument.get_dispatcher()
    event_names = [h.class_name() for h in dispatcher.event_handlers]
    span_names = [h.class_name() for h in dispatcher.span_handlers]
    assert "VoRagBridgeHandler" in event_names
    assert "VoRagSpanHandler" in span_names


def test_install_bridge_is_idempotent() -> None:
    handler1 = install_bridge()
    handler2 = install_bridge()
    dispatcher = instrument.get_dispatcher()
    bridge_handlers = [h for h in dispatcher.event_handlers if isinstance(h, VoRagBridgeHandler)]
    span_handlers = [h for h in dispatcher.span_handlers if isinstance(h, VoRagSpanHandler)]
    assert len(bridge_handlers) == 1
    assert len(span_handlers) == 1
    assert handler1 is handler2


def test_install_bridge_repairs_missing_span_handler_when_event_handler_exists() -> None:
    dispatcher = instrument.get_dispatcher()
    existing = VoRagBridgeHandler()
    dispatcher.add_event_handler(existing)

    handler = install_bridge()

    bridge_handlers = [h for h in dispatcher.event_handlers if isinstance(h, VoRagBridgeHandler)]
    span_handlers = [h for h in dispatcher.span_handlers if isinstance(h, VoRagSpanHandler)]
    assert handler is existing
    assert bridge_handlers == [existing]
    assert len(span_handlers) == 1


def test_bridge_handle_synthetic_event_returns_safely() -> None:
    handler = VoRagBridgeHandler()
    result = handler.handle("synthetic_event")
    assert result is None


def test_llm_chat_end_event_does_not_attribute_cost_to_prevent_double_counting() -> None:
    handler = VoRagBridgeHandler()
    response = ChatResponse(
        message=ChatMessage(role="assistant", content="ok"),
        raw={
            "model": "qwen3",
            "input_tokens": 13,
            "output_tokens": 8,
            "cost_usd": 0.0051,
        },
    )
    event = LLMChatEndEvent(messages=[], response=response)

    with cost_context(envelope_id="env-229", pipeline="test_pipeline") as acc:
        handler.handle(event)

    assert acc.step_breakdown() == {}


def test_span_handler_new_span_noop() -> None:
    handler = VoRagSpanHandler()
    result = handler.new_span(
        id_="test-span",
        bound_args=_empty_bound_args(),
    )
    assert result is None


def test_span_handler_strips_full_uuid_suffix_from_metric_label() -> None:
    handler = VoRagSpanHandler()
    span_id = "Retriever.retrieve-123e4567-e89b-12d3-a456-426614174000"

    assert handler._func_name(span_id) == "Retriever.retrieve"


def test_span_handler_observes_pipeline_duration_with_safe_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = VoRagSpanHandler()
    pipeline = "observability_pipeline_stage"
    span_id = f"{pipeline}-spanid"
    observed_labels = {"pipeline": pipeline}
    before_count = _histogram_sample_value(
        pipeline_duration_seconds,
        "vo_rag_pipeline_duration_seconds_count",
        observed_labels,
    )
    before_sum = _histogram_sample_value(
        pipeline_duration_seconds,
        "vo_rag_pipeline_duration_seconds_sum",
        observed_labels,
    )
    times = iter([10.0, 10.25])
    monkeypatch.setattr(
        "vo_rag.integrations.llamaindex.observability.time.perf_counter",
        lambda: next(times),
    )

    assert handler.new_span(id_=span_id, bound_args=_empty_bound_args()) is None
    result = handler.prepare_to_exit_span(id_=span_id, bound_args=_empty_bound_args())

    assert result is None
    assert (
        _histogram_sample_value(
            pipeline_duration_seconds,
            "vo_rag_pipeline_duration_seconds_count",
            observed_labels,
        )
        == before_count + 1
    )
    assert _histogram_sample_value(
        pipeline_duration_seconds,
        "vo_rag_pipeline_duration_seconds_sum",
        observed_labels,
    ) == pytest.approx(before_sum + 0.25)


def test_span_handler_rejects_poisoned_pipeline_label_before_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = VoRagSpanHandler()
    pipeline = "span-user@example.com"
    span_id = f"{pipeline}-spanid"
    assert not _histogram_has_label(pipeline_duration_seconds, "pipeline", pipeline)
    times = iter([20.0, 20.25])
    monkeypatch.setattr(
        "vo_rag.integrations.llamaindex.observability.time.perf_counter",
        lambda: next(times),
    )

    handler.new_span(id_=span_id, bound_args=_empty_bound_args())
    with pytest.raises(UnsafeLabel):
        handler.prepare_to_exit_span(id_=span_id, bound_args=_empty_bound_args())

    assert not _histogram_has_label(pipeline_duration_seconds, "pipeline", pipeline)


def test_span_handler_prepare_to_exit_span_noop() -> None:
    handler = VoRagSpanHandler()
    result = handler.prepare_to_exit_span(
        id_="test-span",
        bound_args=_empty_bound_args(),
    )
    assert result is None


def test_span_handler_prepare_to_drop_span_noop() -> None:
    handler = VoRagSpanHandler()
    result = handler.prepare_to_drop_span(
        id_="test-span",
        bound_args=_empty_bound_args(),
    )
    assert result is None

"""Contract tests for LlamaIndex export redaction (VOR-231).

Verifies that no LlamaIndex-originated value escapes vo-rag's egress
policy: every exported span attribute is redacted, and every Prometheus
label is checked for PII.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import llama_index.core
import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.callbacks import EventPayload
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType

from vo_rag.core.observability.metrics import UnsafeLabel
from vo_rag.core.observability.redaction import reset_default_redactor
from vo_rag.integrations.llamaindex import observability as obs
from vo_rag.integrations.llamaindex.observability import (
    VoRagBridgeHandler,
    export_span_attr,
)


@pytest.fixture(autouse=True)
def _reset_redactor() -> Iterator[None]:
    """Each test starts with a fresh default redactor."""
    reset_default_redactor()
    obs._reset_otel_export_state()
    llama_index.core.global_handler = None
    yield
    llama_index.core.global_handler = None
    obs._reset_otel_export_state()
    reset_default_redactor()


# ------------------------------------------------------------------
# export_span_attr seam
# ------------------------------------------------------------------


def test_export_span_attr_scrubs_email() -> None:
    raw = "contact me at user@example.com"
    out = export_span_attr(raw)
    assert "user@example.com" not in out
    assert "<EMAIL_ADDRESS>" in out


def test_export_span_attr_scrubs_ssn() -> None:
    raw = "SSN 123-45-6789 filed"
    out = export_span_attr(raw)
    assert "123-45-6789" not in out
    assert "<US_SSN>" in out


def test_export_span_attr_scrubs_api_key() -> None:
    raw = "token: sk-ant-api01-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    out = export_span_attr(raw)
    assert "sk-ant-api01-" not in out
    assert "<API_KEY>" in out


def test_export_span_attr_passthrough_when_redaction_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VORAG_REDACTION_LEVEL", "off")
    reset_default_redactor()
    raw = "contact me at user@example.com"
    out = export_span_attr(raw)
    assert out == raw


# ------------------------------------------------------------------
# Prometheus label PII guard
# ------------------------------------------------------------------


def test_bridge_handler_raises_unsafe_label_on_email_in_model() -> None:
    """A poisoned LlamaIndex event must not leak PII onto /metrics."""
    from llama_index.core.instrumentation.events.llm import LLMChatStartEvent

    event = LLMChatStartEvent(
        messages=[],
        additional_kwargs={},
        model_dict={"model": "user@example.com"},
    )
    handler = VoRagBridgeHandler()
    with pytest.raises(UnsafeLabel):
        handler.handle(event)


# ------------------------------------------------------------------
# Callback handler wrapper
# ------------------------------------------------------------------


def test_redacting_callback_handler_redacts_payload() -> None:
    """The wrapper scrubs PII from payloads before they reach the exporter."""
    from vo_rag.integrations.llamaindex.observability import _RedactingCallbackHandler

    class SpyHandler(BaseCallbackHandler):
        def __init__(self) -> None:
            super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
            self.last_payload: dict[str, Any] | None = None

        def on_event_start(
            self,
            event_type: CBEventType,
            payload: dict[str, Any] | None = None,
            event_id: str = "",
            parent_id: str = "",
            **kwargs: Any,
        ) -> str:
            self.last_payload = payload
            return event_id

        def on_event_end(
            self,
            event_type: CBEventType,
            payload: dict[str, Any] | None = None,
            event_id: str = "",
            **kwargs: Any,
        ) -> None:
            self.last_payload = payload

        def start_trace(self, trace_id: str | None = None) -> None:
            pass

        def end_trace(
            self,
            trace_id: str | None = None,
            trace_map: dict[str, list[str]] | None = None,
        ) -> None:
            pass

    spy = SpyHandler()
    wrapper = _RedactingCallbackHandler(spy)

    wrapper.on_event_start(
        CBEventType.LLM,
        payload={"query": "contact user@example.com", "ssn": "123-45-6789"},
    )
    assert spy.last_payload is not None
    assert "user@example.com" not in spy.last_payload["query"]
    assert "<EMAIL_ADDRESS>" in spy.last_payload["query"]
    assert "123-45-6789" not in spy.last_payload["ssn"]
    assert "<US_SSN>" in spy.last_payload["ssn"]


def test_redacts_chat_message_objects_in_export_payload() -> None:
    payload = {
        EventPayload.MESSAGES: [
            ChatMessage(role="user", content="email maintainer@example.com card 4242424242424242")
        ]
    }

    redacted = obs._redact_payload(payload)

    rendered = str(redacted)
    assert "maintainer@example.com" not in rendered
    assert "4242424242424242" not in rendered


def test_redacts_chat_response_objects_in_export_payload() -> None:
    payload = {
        EventPayload.RESPONSE: ChatResponse(
            message=ChatMessage(role="assistant", content="reply to maintainer@example.com")
        )
    }

    redacted = obs._redact_payload(payload)

    assert "maintainer@example.com" not in str(redacted)


def test_redacting_callback_handler_passes_none_payload() -> None:
    """The wrapper must not crash when payload is None."""
    from vo_rag.integrations.llamaindex.observability import _RedactingCallbackHandler

    class SpyHandler(BaseCallbackHandler):
        def __init__(self) -> None:
            super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
            self.called = False

        def on_event_start(
            self,
            event_type: CBEventType,
            payload: dict[str, Any] | None = None,
            event_id: str = "",
            parent_id: str = "",
            **kwargs: Any,
        ) -> str:
            self.called = True
            return event_id

        def on_event_end(
            self,
            event_type: CBEventType,
            payload: dict[str, Any] | None = None,
            event_id: str = "",
            **kwargs: Any,
        ) -> None:
            pass

        def start_trace(self, trace_id: str | None = None) -> None:
            pass

        def end_trace(
            self,
            trace_id: str | None = None,
            trace_map: dict[str, list[str]] | None = None,
        ) -> None:
            pass

    spy = SpyHandler()
    wrapper = _RedactingCallbackHandler(spy)
    wrapper.on_event_start(CBEventType.LLM, payload=None)
    assert spy.called is True


def test_enable_otel_export_wraps_handler_with_redactor() -> None:
    """After enable_otel_export, the global handler is wrapped."""
    import llama_index.core

    obs._reset_otel_export_state()

    class FakeHandler(BaseCallbackHandler):
        def __init__(self) -> None:
            super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])

        def on_event_start(
            self,
            event_type: CBEventType,
            payload: dict[str, Any] | None = None,
            event_id: str = "",
            parent_id: str = "",
            **kwargs: Any,
        ) -> str:
            return event_id

        def on_event_end(
            self,
            event_type: CBEventType,
            payload: dict[str, Any] | None = None,
            event_id: str = "",
            **kwargs: Any,
        ) -> None:
            pass

        def start_trace(self, trace_id: str | None = None) -> None:
            pass

        def end_trace(
            self,
            trace_id: str | None = None,
            trace_map: dict[str, list[str]] | None = None,
        ) -> None:
            pass

    with patch.object(llama_index.core, "set_global_handler", autospec=True) as mock_set:

        def side_effect(*args: Any, **kwargs: Any) -> None:
            llama_index.core.global_handler = FakeHandler()

        mock_set.side_effect = side_effect

        env = {obs.OTEL_EXPORT_ENV_VAR: "phoenix"}
        obs.enable_otel_export(env=env)

    handler = llama_index.core.global_handler
    assert handler is not None
    assert isinstance(handler, obs._RedactingCallbackHandler)
    assert isinstance(handler._handler, FakeHandler)

    obs._reset_otel_export_state()

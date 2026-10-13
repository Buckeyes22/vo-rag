"""LlamaIndex instrumentation bridge into vo-rag observability.

Wires LlamaIndex's Dispatcher events and spans into vo-rag's Prometheus
metrics, cost accumulator, and PII guards. Metric/cost translation and export
redaction now live here: labels pass ``assert_safe_labels``, exported callback
payloads are wrapped by ``_RedactingCallbackHandler``, and ``export_span_attr``
is the plain-text redaction seam.
"""

from __future__ import annotations

import inspect
import os
import re
import time
from collections.abc import Mapping
from typing import Any, Final

import llama_index.core
import llama_index_instrumentation as instrument
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType
from llama_index.core.instrumentation.events.llm import LLMChatEndEvent, LLMChatStartEvent
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from llama_index_instrumentation.event_handlers.base import BaseEventHandler
from pydantic import PrivateAttr

from vo_rag.core.observability.metrics import (
    assert_safe_labels,
    llm_calls_total,
    llm_latency_seconds,
    pipeline_duration_seconds,
)
from vo_rag.core.observability.redaction import get_default_redactor

_SPAN_UUID_SUFFIX_RE = re.compile(
    r"^(?P<name>.+)-"
    r"[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}$"
)


class VoRagBridgeHandler(BaseEventHandler):
    """Funnels LlamaIndex events into vo-rag's observability stack.

    PII guards (``assert_safe_labels``) wrap every value that reaches a
    Prometheus label.
    """

    @classmethod
    def class_name(cls) -> str:
        return "VoRagBridgeHandler"

    def handle(self, event: Any, **kwargs: Any) -> Any:
        if isinstance(event, LLMChatStartEvent):
            model = str(getattr(event, "model_dict", {}).get("model", "unknown"))
            assert_safe_labels(model=model)
            llm_calls_total.labels(
                provider="llamaindex",
                model=model,
                prompt_name="unknown",
                result="started",
            ).inc()

        elif isinstance(event, LLMChatEndEvent):
            model = str(getattr(event, "model_dict", {}).get("model", "unknown"))
            assert_safe_labels(model=model)

        return None


class VoRagSpanHandler(BaseSpanHandler[Any]):
    """Funnels LlamaIndex spans into vo-rag's duration histograms.

    LLM-shaped spans observe ``llm_latency_seconds``; everything else observes
    ``pipeline_duration_seconds``.  The handler keeps span start times in a
    side dict and returns ``None`` so the dispatcher contract stays satisfied
    without accumulating state in ``open_spans``.
    """

    _span_starts: dict[str, float] = PrivateAttr(default_factory=dict)

    @classmethod
    def class_name(cls) -> str:
        return "VoRagSpanHandler"

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        parent_span_id: str | None = None,
        tags: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._span_starts[id_] = time.perf_counter()
        return None

    def _func_name(self, id_: str) -> str:
        # Dispatcher ids are ``f"{func.__qualname__}-{uuid.uuid4()}"``.
        if match := _SPAN_UUID_SUFFIX_RE.match(id_):
            return match.group("name")
        return id_.rsplit("-", 1)[0]

    def _is_llm_span(self, name: str) -> bool:
        lowered = name.lower()
        return "chat" in lowered or "llm" in lowered

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        result: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        start = self._span_starts.pop(id_, None)
        if start is None:
            return None
        duration = time.perf_counter() - start
        name = self._func_name(id_)

        if self._is_llm_span(name):
            model = "unknown"
            if instance is not None:
                model = str(getattr(instance, "model", "unknown"))
            assert_safe_labels(model=model)
            llm_latency_seconds.labels(provider="llamaindex", model=model).observe(duration)
        else:
            assert_safe_labels(pipeline=name)
            pipeline_duration_seconds.labels(pipeline=name).observe(duration)

        return None

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        err: BaseException | None = None,
        **kwargs: Any,
    ) -> Any:
        self._span_starts.pop(id_, None)
        return None


def _find_existing_bridge(dispatcher: Any) -> VoRagBridgeHandler | None:
    for h in dispatcher.event_handlers:
        if isinstance(h, VoRagBridgeHandler):
            return h
    return None


def _find_existing_span_bridge(dispatcher: Any) -> VoRagSpanHandler | None:
    for h in dispatcher.span_handlers:
        if isinstance(h, VoRagSpanHandler):
            return h
    return None


def install_bridge() -> VoRagBridgeHandler:
    """Register both handlers on the LlamaIndex root dispatcher.

    Idempotent: calling twice does not register duplicate handlers.
    """
    dispatcher = instrument.get_dispatcher()

    event_handler = _find_existing_bridge(dispatcher)
    if event_handler is None:
        event_handler = VoRagBridgeHandler()
        dispatcher.add_event_handler(event_handler)

    if _find_existing_span_bridge(dispatcher) is None:
        dispatcher.add_span_handler(VoRagSpanHandler())

    return event_handler


# --- Phoenix / OTel operator export (VOR-230) -----------------------------
#
# Phoenix / OTel is an *operator-facing extra*.  The
# primary sink is vo-rag's stack via install_bridge() above.  This export
# is opt-in via env ``VORAG_LLAMAINDEX_OTEL_EXPORT`` whose value is the
# LlamaIndex backend name passed to ``set_global_handler(...)``
# (e.g. ``arize_phoenix``, ``openinference``).  Unset / empty → no
# ``set_global_handler`` call, no import side effects, no export.
#
OTEL_EXPORT_ENV_VAR: Final[str] = "VORAG_LLAMAINDEX_OTEL_EXPORT"

# Operator-friendly aliases mapped onto the canonical LlamaIndex
# ``set_global_handler`` backend names.  See
# .venv/.../llama_index/core/callbacks/global_handlers.py for the full
# supported list.
_BACKEND_ALIASES: Final[dict[str, str]] = {
    "phoenix": "arize_phoenix",
    "arize": "arize_phoenix",
    "arize_phoenix": "arize_phoenix",
    "otel": "openinference",
    "opentelemetry": "openinference",
    "openinference": "openinference",
}

_otel_export_active: bool = False


def export_span_attr(text: str) -> str:
    """Redaction seam for any text shipped as a LlamaIndex span attribute.

    Thin shim that delegates to
    ``vo_rag.core.observability.redaction.get_default_redactor().redact(text)``
    (env ``VORAG_REDACTION_LEVEL``). Callback payload dictionaries are redacted
    through ``_RedactingCallbackHandler``; this function remains the scalar-text
    seam for exported span/trace attributes.
    """
    return get_default_redactor().redact(text)


def _resolve_backend(name: str) -> str:
    return _BACKEND_ALIASES.get(name.strip().lower(), name.strip().lower())


def _redact_payload(payload: Mapping[Any, Any] | None) -> dict[str, Any] | None:
    """Redact a callback payload dict before it reaches an exporter."""
    if payload is None:
        return None
    string_keyed = {k if isinstance(k, str) else str(k): v for k, v in payload.items()}
    return get_default_redactor().redact_mapping(string_keyed)


class _RedactingCallbackHandler(BaseCallbackHandler):
    """Wraps a LlamaIndex callback handler so every exported payload is redacted.

    LlamaIndex ships raw prompt/response text to trace backends; vo-rag
    must redact it first.
    """

    def __init__(self, handler: BaseCallbackHandler) -> None:
        super().__init__(
            event_starts_to_ignore=list(handler.event_starts_to_ignore),
            event_ends_to_ignore=list(handler.event_ends_to_ignore),
        )
        self._handler = handler

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handler, name)

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        redacted = _redact_payload(payload)
        return self._handler.on_event_start(event_type, redacted, event_id, parent_id, **kwargs)

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        redacted = _redact_payload(payload)
        self._handler.on_event_end(event_type, redacted, event_id, **kwargs)

    def start_trace(self, trace_id: str | None = None) -> None:
        self._handler.start_trace(trace_id)

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: dict[str, list[str]] | None = None,
    ) -> None:
        self._handler.end_trace(trace_id, trace_map)


def enable_otel_export(
    *,
    env: Mapping[str, str] | None = None,
    handler_kwargs: Mapping[str, Any] | None = None,
) -> bool:
    """Wire LlamaIndex's one-line Phoenix/OTel export, opt-in only.

    Reads ``VORAG_LLAMAINDEX_OTEL_EXPORT`` from ``env`` (default: the
    process environment) and — only when set to a non-empty value — calls
    ``set_global_handler(backend, **handler_kwargs)`` for the chosen
    LlamaIndex backend (e.g. ``arize_phoenix``, ``openinference``).  An
    unset / empty flag is a no-op: ``set_global_handler`` is not called
    and no backend module is imported.

    Exported callback payloads are redacted at ``VORAG_REDACTION_LEVEL`` by
    ``_RedactingCallbackHandler`` before they leave the process. Scalar text
    attributes use ``export_span_attr(...)``.

    Idempotent: after the first successful activation, later calls with the
    flag set return ``True`` without replacing the global handler.

    Returns ``True`` when a handler was attached, ``False`` when the flag
    was unset / empty.
    """
    global _otel_export_active

    source: Mapping[str, str] = os.environ if env is None else env
    raw = source.get(OTEL_EXPORT_ENV_VAR, "").strip()
    if not raw:
        return False

    backend = _resolve_backend(raw)
    if _otel_export_active:
        return True

    # Keep the scalar-text export seam exercised for backend identifiers.
    # Callback payload dictionaries are redacted by _RedactingCallbackHandler
    # after set_global_handler attaches the backend handler.
    export_span_attr(backend)

    kwargs: dict[str, Any] = dict(handler_kwargs) if handler_kwargs else {}

    llama_index.core.set_global_handler(backend, **kwargs)
    # Wrap the bare handler so every exported payload is redacted before
    # it leaves the process.
    raw_handler = getattr(llama_index.core, "global_handler", None)
    if raw_handler is not None:
        llama_index.core.global_handler = _RedactingCallbackHandler(raw_handler)
    _otel_export_active = True
    return True


def is_otel_export_active() -> bool:
    """Whether ``enable_otel_export`` has attached a LlamaIndex handler."""
    return _otel_export_active


def _reset_otel_export_state() -> None:
    """Test-only: clear the module-level activation flag.

    Does NOT detach the LlamaIndex handler (the callback API does not expose a
    clean detach); tests patch ``set_global_handler`` and reset this flag for a
    fresh activation path.
    """
    global _otel_export_active
    _otel_export_active = False


__all__ = [
    "OTEL_EXPORT_ENV_VAR",
    "VoRagBridgeHandler",
    "VoRagSpanHandler",
    "enable_otel_export",
    "export_span_attr",
    "install_bridge",
    "is_otel_export_active",
]

"""OpenTelemetry SDK bootstrap for vo-rag.

This module owns the process-wide SDK install used by the production API
lifespan and by future CLI/pipeline runners. It stays importable from core code:
no FastAPI imports, no route imports, and no app-side sampling policy.
"""

from __future__ import annotations

import logging
import socket
from typing import Any, Protocol

from opentelemetry import trace
from opentelemetry.trace import Tracer

from vo_rag.core.config.settings import OtelSettings

# NOTE: the OpenTelemetry SDK + OTLP-exporter packages are imported lazily
# inside setup_otel() (not at module top level) so that this module — imported
# broadly by the LLM client, pipeline, and API lifespan — stays importable and
# the app keeps serving even if those optional SDK packages are absent/broken.
# Only the always-present opentelemetry API (trace / Tracer) is imported here.

log = logging.getLogger(__name__)

_otel_installed: bool = False
_httpx_instrumented: bool = False


class _SettingsWithOtel(Protocol):
    otel: OtelSettings


def setup_otel(settings: _SettingsWithOtel) -> bool:
    """Install the OpenTelemetry tracer provider once.

    Returns ``False`` when disabled or when SDK/exporter setup fails. OTel is a
    soft runtime dependency: the SDK/exporter packages are imported here (not at
    module top level) and any import or construction error is logged and
    swallowed so application startup can continue normally.
    """
    global _otel_installed

    if not settings.otel.enabled:
        return False
    if _otel_installed:
        return True

    otel = settings.otel
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(
            {
                "host.name": otel.host_name or socket.gethostname(),
                "service.name": otel.service_name,
                "service.namespace": otel.service_namespace,
                "deployment.environment.name": otel.deployment_environment,
                "role": otel.role,
            }
        )
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=f"{otel.endpoint}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
    except Exception:
        log.exception("otel_setup_failed")
        return False

    _otel_installed = True
    return True


def get_tracer(name: str) -> Tracer:
    """Return an OTel tracer, falling back to the API no-op provider if unset."""
    return trace.get_tracer(name)


def _reset_for_tests() -> None:
    """Test-only: clear the module-level installation flag."""
    global _httpx_instrumented, _otel_installed
    if _httpx_instrumented:
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

            HTTPXClientInstrumentor().uninstrument()
        except Exception:
            log.exception("otel_httpx_uninstrument_failed")
    _otel_installed = False
    _httpx_instrumented = False


def instrument_fastapi_app(app: Any) -> None:
    """Install FastAPI request tracing on ``app`` once, best-effort."""
    if getattr(app, "_is_instrumented_by_opentelemetry", False):
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="health,ready,metrics,docs,openapi.json",
        )
    except Exception:
        log.exception("otel_fastapi_instrumentation_failed")


def instrument_httpx() -> None:
    """Install global httpx tracing once, best-effort."""
    global _httpx_instrumented

    if _httpx_instrumented:
        return

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
    except Exception:
        log.exception("otel_httpx_instrumentation_failed")
        return

    _httpx_instrumented = True


__all__ = [
    "get_tracer",
    "instrument_fastapi_app",
    "instrument_httpx",
    "setup_otel",
    "_reset_for_tests",
]

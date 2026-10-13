"""Contract tests for OTel auto-instrumentation helpers."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from tests.api.conftest import fake_settings

from vo_rag.api.main import create_app
from vo_rag.api.singletons import SurfaceServices
from vo_rag.core.config.settings import OtelSettings
from vo_rag.core.observability import otel


@pytest.fixture(autouse=True)
def reset_otel_instrumentation() -> Iterator[None]:
    otel._reset_for_tests()
    yield
    otel._reset_for_tests()


def test_instrument_httpx_is_idempotent() -> None:
    otel.instrument_httpx()
    otel.instrument_httpx()

    assert HTTPXClientInstrumentor().is_instrumented_by_opentelemetry


def test_instrument_fastapi_app_marks_app_instrumented() -> None:
    app = FastAPI()

    otel.instrument_fastapi_app(app)

    assert getattr(app, "_is_instrumented_by_opentelemetry", False) is True


def test_setup_otel_disabled_is_noop() -> None:
    settings = SimpleNamespace(otel=OtelSettings(enabled=False))

    assert otel.setup_otel(settings) is False


def test_setup_otel_enabled_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    # Hermetic: stub the SDK construction + the global set call so the test
    # never installs a real TracerProvider (which OTel allows only once per
    # process and would block other suites' InMemorySpanExporter providers).
    set_calls: list[object] = []
    monkeypatch.setattr(
        "vo_rag.core.observability.otel.trace.set_tracer_provider",
        lambda provider: set_calls.append(provider),
    )
    # setup_otel imports the SDK lazily inside its try block, so patch the SDK
    # at its source modules (not as attributes of otel).
    monkeypatch.setattr("opentelemetry.sdk.trace.TracerProvider", MagicMock())
    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
        MagicMock(),
    )
    monkeypatch.setattr("opentelemetry.sdk.trace.export.BatchSpanProcessor", MagicMock())

    settings = SimpleNamespace(otel=OtelSettings(enabled=True))

    assert otel.setup_otel(settings) is True
    # Second call is a no-op that still reports installed, and does NOT
    # re-install the provider.
    assert otel.setup_otel(settings) is True
    assert len(set_calls) == 1


def test_services_injected_create_app_does_not_auto_instrument() -> None:
    settings = fake_settings(
        otel=SimpleNamespace(
            enabled=True,
            endpoint="http://localhost:4318",
            service_name="vo-rag-test",
            service_namespace="vo",
            deployment_environment="test",
            role="api",
            host_name="test-host",
        )
    )
    services = SurfaceServices(settings=settings)

    app = create_app(settings=settings, services=services)

    assert getattr(app, "_is_instrumented_by_opentelemetry", False) is False
    assert not HTTPXClientInstrumentor().is_instrumented_by_opentelemetry


def test_production_create_app_instruments_fastapi_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: FastAPI instrumentation must be installed at create_app
    # construction time (before Starlette builds + caches the ASGI middleware
    # stack), NOT in the lifespan — which runs too late, so server spans were
    # never emitted in production. With OTel enabled, the production path
    # (services is None) must mark the app instrumented as soon as create_app
    # returns, before any request is served.
    monkeypatch.setenv("VORAG_OTEL_ENABLED", "1")
    # Keep the test hermetic: stub the provider install + the httpx global
    # patch so only the FastAPI-at-construction behaviour is exercised.
    monkeypatch.setattr("vo_rag.core.observability.otel.setup_otel", lambda settings: True)
    monkeypatch.setattr("vo_rag.core.observability.otel.instrument_httpx", lambda: None)

    app = create_app(settings=fake_settings())

    assert getattr(app, "_is_instrumented_by_opentelemetry", False) is True

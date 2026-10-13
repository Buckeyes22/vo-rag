"""Tests for the API app factory, lifespan, and SurfaceServices
container.

These tests pin the public contract:

- ``create_app()`` exists and ``app = create_app()`` remains the ASGI export.
- ``create_app(settings=fake, services=fake_services)`` lets route tests
  inject fakes without touching the network or DB.
- ``/health`` keeps returning the ``{"status": "ok"}`` payload (the import-time contract).
- The service container exposes the slots the routes need.
- GTE reranker init failures are surfaced loudly — not swallowed.
- Pydantic schemas round-trip core ``SearchResult`` and ``IntakeEnvelope``
  shapes without mutating ids or payloads.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

pytestmark = pytest.mark.security


# --------------------------------------------------------------------------- #
# Fake services and settings for injection.
# --------------------------------------------------------------------------- #
@dataclass
class FakeReranker:
    available: bool = True
    reason: str = ""

    def rerank(self, query: str, candidates: list[Any], top_k: int = 20) -> list[Any]:
        return candidates[:top_k]


@dataclass
class FakeSurfaceServices:
    """Bare-minimum stand-in for the real SurfaceServices.

    The point is to keep tests independent of network/DB/Qdrant/Garage
    availability. The shape mirrors the real dataclass so swapping
    fakes for the production container is a 1-line edit in test code.
    """

    settings: Any
    db_pool: Any = None
    qdrant: Any = None
    garage: Any = None
    embedding_client: Any = None
    retriever: Any = None
    reranker: Any = None
    answer_synthesizer: Any = None
    cloud_cost_tracker: Any = None
    kill_switch: Any = None
    reranker_unavailable_reason: str = ""
    cloud_cost_unavailable_reason: str = ""
    garage_unavailable_reason: str = ""
    db_pool_unavailable_reason: str = ""
    kill_switch_unavailable_reason: str = ""


def _fake_settings(**overrides: Any) -> SimpleNamespace:
    """Minimal settings namespace covering the fields SurfaceServices reads."""
    base = {
        "database": SimpleNamespace(
            url="postgresql://localhost:5432/vorag",
            migrations_dir="src/vo_rag/core/db/migrations",
            pool_min=1,
            pool_max=2,
        ),
        "qdrant": SimpleNamespace(
            url="http://localhost:6333",
            api_key="",
            prefer_grpc=False,
        ),
        "garage": SimpleNamespace(
            endpoint="http://localhost:3900",
            access_key="",
            secret_key="",
            region="garage",
            buckets=SimpleNamespace(artifacts="artifacts", models="models", backups="backups"),
        ),
        "embedding": SimpleNamespace(
            query_url="http://localhost:8080",
            timeout_s=30,
            query_api_key="",
        ),
        "reranker": SimpleNamespace(
            model_path="./models/gte-reranker-onnx-int8",
            intra_threads=4,
            batch_size=8,
        ),
        "answer": SimpleNamespace(
            enabled=False,
            provider="anthropic",
            anthropic_api_key="",
            quality_model="claude-sonnet-4-6",
            fast_model="claude-haiku-4-5-20251001",
            kimi_model="kimi-k2.6",
            kimi_api_key="",
            kimi_base_url="https://api.moonshot.ai/v1",
            allow_hosted=False,
            allow_kimi=False,
            max_context_chunks=8,
            max_output_tokens=1200,
            timeout_s=90,
        ),
        "triage": SimpleNamespace(anthropic_api_key=""),
        "cloud": SimpleNamespace(
            monthly_budget_usd=50.0,
            per_job_max_usd=10.0,
            tailscale=SimpleNamespace(
                oauth_client_id="",
                oauth_client_secret="",
                tailnet="",
            ),
        ),
        "portal": SimpleNamespace(upload_max_bytes=2 * 1024 * 1024 * 1024),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------- #
# create_app() contract.
# --------------------------------------------------------------------------- #
def test_create_app_is_importable_and_callable() -> None:
    """The app factory is the public surface for the API."""
    from vo_rag.api import main as main_module

    assert callable(main_module.create_app)
    assert inspect.isfunction(main_module.create_app)


def test_app_asgi_export_kept() -> None:
    """``app = create_app()`` must remain importable from vo_rag.api.main."""
    main_module = importlib.import_module("vo_rag.api.main")
    assert isinstance(main_module.app, FastAPI)
    assert main_module.app.title == "vo-rag"


def test_app_includes_api_key_middleware() -> None:
    from vo_rag.api.main import app

    # The ApiKeyMiddleware must be installed so protected routes gate.
    from vo_rag.api.middleware import ApiKeyMiddleware

    assert any(
        isinstance(m.cls, type) and issubclass(m.cls, ApiKeyMiddleware) for m in app.user_middleware
    )


def test_health_keeps_status_ok_payload() -> None:
    """/health is part of the test_health.py contract — keep its shape."""
    from vo_rag.api.main import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "vo-rag"}


def test_create_app_with_injected_services_skips_lifespan() -> None:
    """Test injection path: passing services must NOT trigger a lifespan
    that would try to open DB / Qdrant / Garage sockets."""
    from vo_rag.api import main as main_module
    from vo_rag.api.singletons import SurfaceServices

    settings = _fake_settings()
    services = SurfaceServices(
        settings=settings,
        db_pool=None,
        qdrant=None,
        garage=None,
        embedding_client=SimpleNamespace(),
        retriever=SimpleNamespace(),
        reranker=FakeReranker(),
        answer_synthesizer=SimpleNamespace(),
        cloud_cost_tracker=None,
        kill_switch=None,
    )

    app = main_module.create_app(settings=settings, services=services)
    assert isinstance(app, FastAPI)
    assert app.state.surface_services is services
    assert app.state.settings is settings
    # No lifespan side-effects when services are injected — tests must
    # not implicitly open sockets.
    assert getattr(app.state, "services_injected", False) is True


def test_create_app_with_no_services_registers_lifespan() -> None:
    from vo_rag.api import main as main_module

    app = main_module.create_app()
    assert app.router.lifespan_context is not None


# --------------------------------------------------------------------------- #
# State wiring — the routes read these slots directly.
# --------------------------------------------------------------------------- #
def test_injected_services_expose_slots_for_routes() -> None:
    from vo_rag.api import main as main_module
    from vo_rag.api.singletons import SurfaceServices

    settings = _fake_settings()
    services = SurfaceServices(
        settings=settings,
        db_pool="pool",
        qdrant="qdrant",
        garage="garage",
        embedding_client="embed",
        retriever="retriever",
        reranker=FakeReranker(),
        answer_synthesizer="answer",
        cloud_cost_tracker="cost",
        kill_switch="kill",
    )

    app = main_module.create_app(settings=settings, services=services)

    # Mirrored onto app.state for direct attribute access.
    assert app.state.settings is settings
    assert app.state.db_pool == "pool"
    assert app.state.qdrant == "qdrant"
    assert app.state.garage == "garage"
    assert app.state.embedding_client == "embed"
    assert app.state.retriever == "retriever"
    assert app.state.reranker is services.reranker
    assert app.state.answer_synthesizer == "answer"
    assert app.state.cloud_cost_tracker == "cost"
    assert app.state.kill_switch == "kill"


# --------------------------------------------------------------------------- #
# GTE reranker failure surfacing.
# --------------------------------------------------------------------------- #
def test_reranker_init_failure_records_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """Boot must not silently swallow a GTE init failure.

    Two designs are possible: fail-closed at boot, or record an unavailable
    reason + status warning. The latter is used so dev environments
    without the GTE ONNX model still boot; the assertion below pins
    that the reason string is recorded loudly and consistently.
    """
    import asyncio

    from vo_rag.api import singletons

    def _explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("model file missing")

    monkeypatch.setattr(singletons, "build_reranker", _explode)

    settings = _fake_settings()
    services = asyncio.run(
        singletons.build_surface_services(
            settings,
            db_pool=None,
            qdrant=None,
            garage=None,
            embedding_client=SimpleNamespace(),
        )
    )
    try:
        assert services.reranker is None
        assert services.reranker_unavailable_reason  # non-empty
        assert "model file missing" in services.reranker_unavailable_reason
    finally:
        # Run the close path on whatever the builder did create. It must
        # tolerate reranker=None.
        asyncio.run(singletons.close_surface_services(services))


def test_reranker_init_success_returns_real_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from vo_rag.api import singletons

    class _StubRerank:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.init_args = args
            self.init_kwargs = kwargs

        def rerank(self, query: str, candidates: list[Any], top_k: int = 20) -> list[Any]:
            return candidates[:top_k]

    monkeypatch.setattr(singletons, "build_reranker", _StubRerank)

    settings = _fake_settings()
    services = asyncio.run(
        singletons.build_surface_services(
            settings,
            db_pool=None,
            qdrant=None,
            garage=None,
            embedding_client=SimpleNamespace(),
        )
    )
    try:
        assert isinstance(services.reranker, _StubRerank)
        assert services.reranker_unavailable_reason == ""
    finally:
        asyncio.run(singletons.close_surface_services(services))


# --------------------------------------------------------------------------- #
# SurfaceServices construction uses the real core clients + real settings.
# --------------------------------------------------------------------------- #
def test_build_surface_services_uses_real_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_surface_services must use QdrantStore / HybridRetriever /
    GteReranker / AnswerSynthesizer against the actual settings fields —
    not a parallel mini-version."""
    import asyncio

    from vo_rag.api import singletons

    captured: dict[str, Any] = {}

    class _StubQdrant:
        def __init__(
            self,
            url: str,
            api_key: str | None = None,
            *,
            prefer_grpc: bool = True,
        ) -> None:
            captured["qdrant"] = (url, api_key, prefer_grpc)

    class _StubRetriever:
        def __init__(self, qdrant: Any, embedding_client: Any) -> None:
            captured["retriever"] = (qdrant, embedding_client)

    class _StubRerank:
        def __init__(
            self,
            model_path: str,
            *,
            intra_threads: int = 16,
            batch_size: int = 32,
        ) -> None:
            captured["reranker"] = (model_path, intra_threads, batch_size)

        def rerank(self, query: str, candidates: list[Any], top_k: int = 20) -> list[Any]:
            return candidates[:top_k]

    class _StubEmbedClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["embedding"] = (args, kwargs)

        async def aclose(self) -> None:
            return None

    class _StubAnswer:
        def __init__(self, settings: Any, *, anthropic_fallback_key: str = "") -> None:
            captured["answer"] = (settings, anthropic_fallback_key)

    monkeypatch.setattr(singletons, "QdrantStore", _StubQdrant)
    monkeypatch.setattr(singletons, "HybridRetriever", _StubRetriever)
    monkeypatch.setattr(singletons, "build_reranker", _StubRerank)
    monkeypatch.setattr(singletons, "build_embedding_client", _StubEmbedClient)
    monkeypatch.setattr(singletons, "AnswerSynthesizer", _StubAnswer)

    settings = _fake_settings()
    services = asyncio.run(
        singletons.build_surface_services(
            settings,
            db_pool="pool",
            qdrant=None,
            garage=None,
        )
    )
    try:
        # QdrantStore was called with the actual settings.qdrant fields.
        assert captured["qdrant"][0] == settings.qdrant.url
        assert captured["qdrant"][1] is None  # api_key "" -> None
        assert captured["qdrant"][2] == settings.qdrant.prefer_grpc
        # Retriever was handed the qdrant store + embedding client.
        assert isinstance(captured["retriever"][0], _StubQdrant)
        assert isinstance(captured["retriever"][1], _StubEmbedClient)
        # build_reranker was handed the reranker settings object (it selects the
        # backend — client/gpu/onnx_cpu — internally).
        assert captured["reranker"][0] is settings.reranker
        # Answer synthesizer was called with the actual answer settings
        # object and the triage anthropic fallback key.
        assert captured["answer"][0] is settings.answer
        assert captured["answer"][1] == settings.triage.anthropic_api_key
    finally:
        asyncio.run(singletons.close_surface_services(services))


# --------------------------------------------------------------------------- #
# Pydantic schemas — these are shared with the CLI and MCP surfaces; field names
# must not be remapped downstream.
# --------------------------------------------------------------------------- #
def test_query_response_serializes_search_result() -> None:
    from vo_rag.api.schemas import QueryHit, QueryResponse
    from vo_rag.core.retrieval.search import SearchResult

    result = SearchResult(
        id="abc-123",
        score=0.87,
        collection="vo_rag_books_raw_v1",
        payload={"chunk_id": "c1", "text": "hello"},
    )
    response = QueryResponse(results=[QueryHit.from_search_result(result)])

    dumped = response.model_dump()
    assert dumped["results"][0]["id"] == "abc-123"
    assert dumped["results"][0]["score"] == 0.87
    assert dumped["results"][0]["collection"] == "vo_rag_books_raw_v1"
    assert dumped["results"][0]["payload"] == {"chunk_id": "c1", "text": "hello"}


def test_query_request_rejects_empty_collections() -> None:
    from vo_rag.api.schemas import QueryRequest

    with pytest.raises(ValidationError):
        QueryRequest(query="hello", collections=[])


def test_query_request_validates_visual_mode() -> None:
    from vo_rag.api.schemas import QueryRequest

    QueryRequest(query="hello", visual_mode="intent_aware")
    QueryRequest(query="hello", visual_mode=None)
    with pytest.raises(ValidationError):
        QueryRequest(query="hello", visual_mode="bogus_mode")


def test_query_request_enforces_top_k_bounds() -> None:
    from vo_rag.api.schemas import QueryRequest

    QueryRequest(query="hello", top_k=1)
    QueryRequest(query="hello", top_k=200)
    with pytest.raises(ValidationError):
        QueryRequest(query="hello", top_k=0)
    with pytest.raises(ValidationError):
        QueryRequest(query="hello", top_k=201)


def test_answer_request_defaults_to_sterile_scope_and_quality_profile() -> None:
    from vo_rag.api.schemas import AnswerRequest
    from vo_rag.core.answer.context import ContextScope
    from vo_rag.core.answer.synthesizer import AnswerProfile

    req = AnswerRequest(query="hello")
    assert req.context_scope == ContextScope.STERILE
    assert req.profile == AnswerProfile.QUALITY
    assert req.include_retrieval is True
    assert req.rerank is True


def test_answer_response_preserves_id_fields() -> None:
    from vo_rag.api.schemas import AnswerResponse, ComplianceInfo

    resp = AnswerResponse(
        answer_id="01KT6XDYX2KH1CVAPB0QNX4B43",  # pragma: allowlist secret
        answer="42",
        confidence=0.5,
        citations=["S1"],
        sources=["src-1"],
        retrieval_results=[],
        compliance=ComplianceInfo(should_quarantine=True, reason="prompt_injection_suspected"),
        warnings=["prompt_injection_suspected"],
        model="claude-sonnet-4-6",
        mode="internal",
    )
    dumped = resp.model_dump()
    assert dumped["answer_id"] == "01KT6XDYX2KH1CVAPB0QNX4B43"  # noqa: S105
    assert dumped["compliance"]["should_quarantine"] is True
    assert dumped["warnings"] == ["prompt_injection_suspected"]


def test_upload_url_request_rejects_oversize() -> None:
    from vo_rag.api.schemas import UploadUrlRequest

    UploadUrlRequest(filename="a.pdf", content_type="application/pdf", size_bytes=1024)
    with pytest.raises(ValidationError):
        UploadUrlRequest(
            filename="a.pdf",
            content_type="application/pdf",
            size_bytes=3 * 1024 * 1024 * 1024,  # > 2 GiB fallback
        )


def test_upload_url_request_sanitizes_filename_via_validator() -> None:
    from vo_rag.api.schemas import UploadUrlRequest

    req = UploadUrlRequest(
        filename="../../etc/passwd",
        content_type="application/octet-stream",
        size_bytes=128,
    )
    # The filename is sanitized via secure_filename in the schema.
    assert ".." not in req.filename
    assert "/" not in req.filename


def test_admin_status_response_shape() -> None:
    from vo_rag.api.schemas import AdminStatusResponse, ServiceHealth

    resp = AdminStatusResponse(
        ok=True,
        services={
            "qdrant": ServiceHealth(ok=True),
            "reranker": ServiceHealth(ok=False, error="model file missing"),
        },
        pipeline_roster=[],
        eval_gate={"ok": True, "checked": 0, "skipped": 0, "failed": 0, "details": []},
        cloud={"workloads": {}},
    )
    assert resp.ok is True
    assert resp.services["reranker"].ok is False
    assert resp.services["reranker"].error == "model file missing"

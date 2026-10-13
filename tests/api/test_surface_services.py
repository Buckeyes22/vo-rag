"""SurfaceServices container + build/close lifecycle.

The service container is the single seam the API/CLI/MCP surfaces share
to reach core services. These tests pin:

- The dataclass shape (all 10 slots + unavailable reasons).
- close_surface_services tolerates a partially-built container (no
  raised errors when retriever/reranker/etc. are None).
- Real settings fields are the input — no parallel mini-config in api/.
- Schema serialization preserves core ids/payloads byte-for-byte.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.security


# --------------------------------------------------------------------------- #
# Test fixtures
# --------------------------------------------------------------------------- #
def _fake_settings(**overrides: Any) -> SimpleNamespace:
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
# SurfaceServices dataclass
# --------------------------------------------------------------------------- #
def test_surface_services_dataclass_shape() -> None:
    """The dataclass is the contract every route handler reads against."""
    from dataclasses import fields

    from vo_rag.api.singletons import SurfaceServices

    names = {f.name for f in fields(SurfaceServices)}
    assert {
        "settings",
        "db_pool",
        "qdrant",
        "garage",
        "embedding_client",
        "retriever",
        "reranker",
        "answer_synthesizer",
        "cloud_cost_tracker",
        "kill_switch",
        "reranker_unavailable_reason",
        "cloud_cost_unavailable_reason",
        "garage_unavailable_reason",
        "db_pool_unavailable_reason",
        "kill_switch_unavailable_reason",
    } <= names


def test_surface_services_unavailable_reasons_default_empty() -> None:
    from vo_rag.api.singletons import SurfaceServices

    services = SurfaceServices(settings=SimpleNamespace())
    assert services.reranker_unavailable_reason == ""
    assert services.cloud_cost_unavailable_reason == ""
    assert services.garage_unavailable_reason == ""
    assert services.db_pool_unavailable_reason == ""
    assert services.kill_switch_unavailable_reason == ""


# --------------------------------------------------------------------------- #
# build_surface_services — accepts injected deps, never opens sockets.
# --------------------------------------------------------------------------- #
def test_build_surface_services_uses_injected_db_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    from vo_rag.api import singletons

    captured: dict[str, Any] = {}

    def _fake_get_pool(*args: Any, **kwargs: Any) -> None:
        captured["get_pool"] = (args, kwargs)
        raise AssertionError("must not call get_pool when db_pool is injected")

    monkeypatch.setattr(singletons, "get_pool", _fake_get_pool, raising=False)

    services = asyncio.run(
        singletons.build_surface_services(
            _fake_settings(),
            db_pool="pre-built",
            qdrant=None,
            garage=None,
        )
    )
    try:
        assert services.db_pool == "pre-built"
        assert services.db_pool_unavailable_reason == ""
    finally:
        asyncio.run(singletons.close_surface_services(services))


def test_build_surface_services_uses_injected_qdrant_and_garage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.api import singletons

    monkeypatch.setattr(singletons, "GteReranker", lambda *a, **kw: _NoOpReranker(), raising=False)

    services = asyncio.run(
        singletons.build_surface_services(
            _fake_settings(),
            db_pool="pool",
            qdrant="pre-built-qdrant",
            garage="pre-built-garage",
        )
    )
    try:
        assert services.qdrant == "pre-built-qdrant"
        assert services.garage == "pre-built-garage"
    finally:
        asyncio.run(singletons.close_surface_services(services))


def test_build_surface_services_garage_built_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.api import singletons

    captured: dict[str, Any] = {}

    class _StubGarage:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["garage"] = (args, kwargs)

    monkeypatch.setattr(singletons, "GarageClient", _StubGarage, raising=False)
    monkeypatch.setattr(singletons, "GteReranker", lambda *a, **kw: _NoOpReranker(), raising=False)

    settings = _fake_settings()
    # Garage needs real credentials to attempt the build; the stub
    # captures the kwargs verbatim.
    settings.garage.access_key = "ak"
    settings.garage.secret_key = "sk"  # pragma: allowlist secret

    services = asyncio.run(
        singletons.build_surface_services(
            settings,
            db_pool="pool",
            qdrant=None,
            garage=None,
        )
    )
    try:
        assert "garage" in captured
        # Built from the actual settings.garage fields (kwargs shape).
        args, kwargs = captured["garage"]
        assert kwargs["endpoint_url"] == settings.garage.endpoint
        assert kwargs["access_key"] == settings.garage.access_key
        assert kwargs["secret_key"] == settings.garage.secret_key
        assert isinstance(services.garage, _StubGarage)
    finally:
        asyncio.run(singletons.close_surface_services(services))


def test_partial_build_failure_closes_resources_already_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.api import singletons
    from vo_rag.core.db import postgres

    closed: list[str] = []

    class _Pool:
        pass

    class _Qdrant:
        async def aclose(self) -> None:
            closed.append("qdrant")

    pool = _Pool()
    monkeypatch.setattr(postgres, "get_existing_pool", lambda: None)

    async def _get_pool(*_args: Any, **_kwargs: Any) -> _Pool:
        return pool

    async def _close_pool() -> None:
        closed.append("pool")

    monkeypatch.setattr(postgres, "get_pool", _get_pool)
    monkeypatch.setattr(postgres, "close_pool", _close_pool)
    monkeypatch.setattr(singletons, "QdrantStore", lambda *_args, **_kwargs: _Qdrant())

    def _fail_embedding(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("fixture construction failure")

    monkeypatch.setattr(singletons, "build_embedding_client", _fail_embedding)

    with pytest.raises(RuntimeError, match="fixture construction failure"):
        asyncio.run(singletons.build_surface_services(_fake_settings()))

    assert closed == ["qdrant", "pool"]


# --------------------------------------------------------------------------- #
# close_surface_services — tolerates partial state.
# --------------------------------------------------------------------------- #
def test_close_surface_services_tolerates_reranker_none() -> None:
    """The whole point of recording reranker_unavailable_reason is that
    the rest of the app can boot. The close path must not raise when the
    reranker is None."""
    from vo_rag.api.singletons import SurfaceServices, close_surface_services

    services = SurfaceServices(
        settings=SimpleNamespace(),
        db_pool=None,
        qdrant=None,
        garage=None,
        embedding_client=SimpleNamespace(aclose=_async_noop),
        retriever=None,
        reranker=None,
        answer_synthesizer=None,
        cloud_cost_tracker=None,
        kill_switch=None,
        reranker_unavailable_reason="missing model",
        owned_resources=[SimpleNamespace(aclose=_async_noop)],
    )

    # Must not raise.
    asyncio.run(close_surface_services(services))


def test_close_surface_services_closes_embedding_client() -> None:
    from vo_rag.api.singletons import SurfaceServices, close_surface_services

    closed: list[bool] = []

    class _Emb:
        async def aclose(self) -> None:
            closed.append(True)

    embedding = _Emb()
    services = SurfaceServices(
        settings=SimpleNamespace(),
        embedding_client=embedding,
        owned_resources=[embedding],
    )

    asyncio.run(close_surface_services(services))
    assert closed == [True]


def test_close_surface_services_leaves_injected_clients_open() -> None:
    from vo_rag.api.singletons import SurfaceServices, close_surface_services

    closed: list[bool] = []

    class _Emb:
        async def aclose(self) -> None:
            closed.append(True)

    services = SurfaceServices(settings=SimpleNamespace(), embedding_client=_Emb())
    asyncio.run(close_surface_services(services))
    assert closed == []


def test_close_surface_services_closes_only_owned_resources_once() -> None:
    from vo_rag.api.singletons import SurfaceServices, close_surface_services

    closed: list[str] = []

    class _Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        async def aclose(self) -> None:
            closed.append(self.name)

    owned = [_Resource("first"), _Resource("second")]
    injected = _Resource("injected")
    services = SurfaceServices(
        settings=SimpleNamespace(),
        embedding_client=injected,
        owned_resources=owned,
    )

    asyncio.run(close_surface_services(services))
    asyncio.run(close_surface_services(services))
    assert closed == ["second", "first"]


# --------------------------------------------------------------------------- #
# Schema serialization — no id remapping, no payload mutation.
# --------------------------------------------------------------------------- #
def test_query_hit_round_trip_does_not_mutate_search_result() -> None:
    from vo_rag.api.schemas import QueryHit
    from vo_rag.core.retrieval.search import SearchResult

    payload = {
        "chunk_id": "c1",
        "parent_chunk_id": "p1",
        "source_id": "src-1",
        "text": "verbatim",
        "section_title": "Intro",
        "page_number": 4,
    }
    result = SearchResult(id="point-xyz", score=0.91, collection="vo_rag_books_v1", payload=payload)
    hit = QueryHit.from_search_result(result)

    # All core ids preserved.
    assert hit.id == "point-xyz"
    assert hit.score == 0.91
    assert hit.collection == "vo_rag_books_v1"
    # Payload passed through unchanged (no field remapping).
    assert hit.payload == payload


def test_envelope_serialization_preserves_id_and_status() -> None:
    from vo_rag.api.schemas import EnvelopeResponse
    from vo_rag.core.models.envelope import (
        EnvelopeStatus,
        IntakeEnvelope,
        IntakeSource,
        Priority,
        Sovereignty,
    )

    envelope = IntakeEnvelope(
        id="01KT6XE8M1MW42M0Y0JJKRZSVP",  # pragma: allowlist secret
        source_type=IntakeSource.TEXT_PROMPT,
        source_data={"pipeline_hint": "book_rag"},
        raw_content={"text": "hello"},
        metadata={},
        status=EnvelopeStatus.RECEIVED,
        priority=Priority.MEDIUM,
        sovereignty=Sovereignty.CLOUD_OK,
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
        updated_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    response = EnvelopeResponse.from_envelope(envelope)
    dumped = response.model_dump()
    assert dumped["envelope_id"] == "01KT6XE8M1MW42M0Y0JJKRZSVP"  # noqa: S105
    assert dumped["status"] == "received"
    assert dumped["source_type"] == "text_prompt"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _NoOpReranker:
    def rerank(self, query: str, candidates: list[Any], top_k: int = 20) -> list[Any]:
        return candidates[:top_k]


async def _async_noop() -> None:
    return None

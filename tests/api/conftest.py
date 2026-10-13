"""Shared fakes + injection helpers for the API route tests.

Tests under ``tests/api/`` use these helpers to build a hermetic FastAPI
app + TestClient without opening any network/DB/Qdrant/Garage sockets.

The shape mirrors the production :class:`SurfaceServices` dataclass so a
real container can be swapped in 1:1 — see ``api/singletons.py`` for
the canonical field list and ``api/lifespan.py`` for the state mirror.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from vo_rag.api.main import create_app
from vo_rag.api.singletons import SurfaceServices


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def fake_settings(**overrides: Any) -> SimpleNamespace:
    """Build a settings namespace covering the fields the route layer reads.

    The values are intentionally non-secret placeholder strings; the
    test asserts that NONE of them leak through the route responses,
    so picking recognisable strings makes leaks easy to grep for.
    """
    base: dict[str, Any] = {
        "api": SimpleNamespace(
            api_key="test-admin-key",  # pragma: allowlist secret
            api_keys_json='{"ci":{"key":"ci-key","caller":"cli"}}',  # pragma: allowlist secret
            cors_origins=["*"],
        ),
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
        "figures": SimpleNamespace(root="/nonexistent/vo-rag-test-figures"),
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
            api_key="RUNPOD-SECRET-VALUE",  # RUNPOD_API_KEY surface  # pragma: allowlist secret
            tailscale=SimpleNamespace(
                oauth_client_id="",
                oauth_client_secret="",
                tailnet="",
            ),
        ),
        "portal": SimpleNamespace(
            upload_max_bytes=2 * 1024 * 1024 * 1024,
            csrf_secret="csrf-secret",  # pragma: allowlist secret
            access_team_domain="",
            access_aud="",
            allowed_emails=[],
        ),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------- #
# Service fakes
# --------------------------------------------------------------------------- #
@dataclass
class FakeQdrantClient:
    """Stand-in for the underlying ``AsyncQdrantClient`` the store wraps.

    The real :class:`vo_rag.core.storage.qdrant.QdrantStore` stores the
    client on ``_client``; the route code goes through ``store._client``
    the same way the system route does. The fake mirrors the two
    methods the route layer needs: ``get_collections`` and ``count``.
    """

    collections: list[str] = field(
        default_factory=lambda: [
            "vo_rag_books_raw_v1",
            "vo_rag_books_sterile_v1",
        ]
    )
    points_by_collection: dict[str, int] = field(
        default_factory=lambda: {
            "vo_rag_books_raw_v1": 100,
            "vo_rag_books_sterile_v1": 80,
        }
    )
    raise_get_collections: Exception | None = None
    raise_count: Exception | None = None

    async def get_collections(self) -> Any:
        if self.raise_get_collections is not None:
            raise self.raise_get_collections
        return SimpleNamespace(
            collections=[SimpleNamespace(name=name) for name in self.collections]
        )

    async def count(self, collection_name: str, exact: bool = True) -> Any:  # noqa: ARG002
        if self.raise_count is not None:
            raise self.raise_count
        return SimpleNamespace(count=self.points_by_collection.get(collection_name, 0))


@dataclass
class FakeQdrantStore:
    _client: Any = None


@dataclass
class _FakeAcquiredConn:
    rows: list[Any] = field(default_factory=list)
    scalar: Any = 0
    error: Exception | None = None

    async def __aenter__(self) -> _FakeAcquiredConn:
        if self.error is not None:
            raise self.error
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def fetch(self, query: str, *args: Any) -> list[Any]:  # noqa: ARG002
        return self.rows

    async def fetchval(self, query: str, *args: Any) -> Any:  # noqa: ARG002
        return self.scalar


def _coerce_row(value: Any) -> Any:
    """Convert a SimpleNamespace-shaped row into a dict.

    asyncpg's ``Record`` type supports both ``row["col"]`` and
    ``row.col``. ``SimpleNamespace`` only supports attribute access,
    so the route code that uses ``row["status"]`` fails. This helper
    adapts one to the other without making the tests depend on
    ``asyncpg.Record`` (which requires a live pool to construct).
    """
    if isinstance(value, SimpleNamespace):
        return {key: getattr(value, key) for key in vars(value)}
    return value


@dataclass
class FakePool:
    """Stand-in for ``asyncpg.Pool`` — only the methods the route layer calls.

    The fake exposes both styles:
    - direct ``pool.fetch(query)`` / ``pool.fetchval(query)`` — used by
      :mod:`vo_rag.api.routes.system` for the envelope-count summary.
    - ``async with pool.acquire() as conn: ...`` — used by
      :func:`vo_rag.api.routes.system._db_pool_health` for the
      ``SELECT 1`` ping. Both styles share the same ``fetch_rows`` /
      ``fetchval_value`` / ``raise_on_acquire`` configuration.
    """

    fetch_rows: list[Any] = field(default_factory=list)
    fetchval_value: Any = 0
    raise_on_acquire: Exception | None = None

    async def fetch(self, query: str, *args: Any) -> list[Any]:  # noqa: ARG002
        return [_coerce_row(row) for row in self.fetch_rows]

    async def fetchval(self, query: str, *args: Any) -> Any:  # noqa: ARG002
        return self.fetchval_value

    def acquire(self) -> _FakeAcquiredConn:
        if self.raise_on_acquire is not None:
            return _FakeAcquiredConn(error=self.raise_on_acquire)
        return _FakeAcquiredConn(
            rows=[_coerce_row(row) for row in self.fetch_rows],
            scalar=self.fetchval_value,
        )


@dataclass
class FakeReranker:
    available: bool = True
    reason: str = ""

    def rerank(self, query: str, candidates: list[Any], top_k: int = 20) -> list[Any]:  # noqa: ARG002
        return candidates[:top_k]


@dataclass
class FakeCloudCostTracker:
    """Stand-in for :class:`vo_rag.core.cloud.cost_tracker.CloudCostTracker`."""

    spend_report: dict[str, Any] = field(
        default_factory=lambda: {
            "month": "2026-06",
            "total_spend_usd": 1.5,
            "budget_usd": 50.0,
            "budget_remaining_usd": 48.5,
            "budget_pct_used": 3.0,
            "by_provider": {"runpod": 1.5},
            "by_workload": {"text-gen": 1.0, "vlm": 0.5},
            "by_gpu_type": {"A100": 1.5},
            "jobs_this_month": 3,
        }
    )
    raise_on_get: Exception | None = None

    async def get_spend_report(self) -> dict[str, Any]:
        if self.raise_on_get is not None:
            raise self.raise_on_get
        return self.spend_report


# --------------------------------------------------------------------------- #
# Surface services
# --------------------------------------------------------------------------- #
def make_services(
    settings: Any | None = None,
    **overrides: Any,
) -> SurfaceServices:
    """Build a real :class:`SurfaceServices` with sane defaults.

    Tests can override any slot (``qdrant=``, ``db_pool=``,
    ``cloud_cost_tracker=``) or any of the ``*_unavailable_reason``
    fields. The settings object is mounted on the dataclass so routes
    that read ``services.settings.cloud.api_key`` see a real attribute.
    """
    return SurfaceServices(
        settings=settings if settings is not None else fake_settings(),
        **overrides,
    )


# --------------------------------------------------------------------------- #
# Test client factory
# --------------------------------------------------------------------------- #
def make_client(
    services: SurfaceServices | None = None,
    *,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> TestClient:
    """Build a hermetic TestClient with system + admin routers wired in.

    Uses the :func:`create_app(settings=, services=)` injection path so
    no lifespan runs and no DB / Qdrant / Garage sockets are opened. The
    routers are included on the new app instance — the import-time
    ``app`` is left untouched so the existing /health contract
    (``tests/api/test_health.py`` + ``test_health_keeps_status_ok_payload``)
    keeps passing through the shared ``app = create_app()`` in
    ``api/main.py``.

    The auth middleware calls :func:`vo_rag.core.config.get_settings` at
    request time, which returns the real ``Settings`` (production env
    values). Tests that hit auth-gated routes must pass a
    ``monkeypatch`` so the middleware sees the fake settings where
    ``api.api_key == "test-admin-key"``.  # pragma: allowlist secret
    """
    if services is None:
        services = make_services()

    if monkeypatch is not None:
        from vo_rag.core import config as config_module

        monkeypatch.setattr(config_module, "get_settings", lambda: services.settings)

    from vo_rag.api.routes import admin, system

    app = create_app(settings=services.settings, services=services)
    app.include_router(system.router)
    app.include_router(admin.router)
    return TestClient(app)


def auth_header(key: str = "test-admin-key") -> dict[str, str]:
    """Return a request header dict carrying a valid X-API-Key."""
    return {"X-API-Key": key}


# --------------------------------------------------------------------------- #
# Pytest fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch):
    """Return a factory that builds a hermetic TestClient.

    The factory signature matches ``make_client(services=...)`` plus a
    ``monkeypatch`` plumbing. The ``monkeypatch`` is applied here so
    the auth middleware's :func:`vo_rag.core.config.get_settings` call
    returns the test's fake settings — without it the middleware reads
    the real production env, fails to match the X-API-Key, and 401s.
    """

    def _factory(
        services: SurfaceServices | None = None,
    ) -> TestClient:
        return make_client(services, monkeypatch=monkeypatch)

    return _factory


@pytest.fixture
def mock_embedding_health() -> Any:
    """Mock the embedding server's ``/health`` endpoint with a 200 + ready.

    The system routes call
    ``httpx.AsyncClient().get(f"{settings.embedding.query_url}/health")``
    on every /ready, /status, and /api/v1/rag/status request. Without
    a live embedding server, that call raises ``ConnectError`` and
    the route reports the embedding slot as not_configured. Tests
    that want a green embedding row opt into this fixture.
    """
    import respx

    with respx.mock(assert_all_called=False) as router:
        router.get("http://localhost:8080/health").respond(
            200, json={"status": "ready", "device": "cpu"}
        )
        yield router


__all__ = [
    "FakeCloudCostTracker",
    "FakePool",
    "FakeQdrantClient",
    "FakeQdrantStore",
    "FakeReranker",
    "app_client",
    "auth_header",
    "fake_settings",
    "make_client",
    "make_services",
    "mock_embedding_health",
]

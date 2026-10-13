"""F-014 / F-048 / F-049: SSRF gate integration at the /api/v1/intake route.

The route must run ``vo_rag.web.url_safety.validate_external_url`` BEFORE any
DB insert and translate ``SSRFBlocked`` into a 422 response. These tests pin
the integration against the real validator (no DNS monkeypatching on the
block paths) and a stubbed validator for the happy path so the test stays
hermetic without depending on the public resolver.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vo_rag.web.url_safety import ResolvedURL, SSRFBlocked, validate_external_url

pytestmark = pytest.mark.security

_TEST_API_KEY = "test-key"  # pragma: allowlist secret


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeAcquire:
    async def __aenter__(self) -> _FakeAcquire:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class _FakePool:
    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire()


def _fake_settings() -> SimpleNamespace:
    return SimpleNamespace(
        api=SimpleNamespace(
            api_key=_TEST_API_KEY,
            api_keys_json="",
            cors_origins=["*"],
            request_timeout_s=60,
        ),
        portal=SimpleNamespace(upload_max_bytes=2 * 1024 * 1024 * 1024),
        garage=SimpleNamespace(
            buckets=SimpleNamespace(
                artifacts="vorag-artifacts",
                models="vorag-models",
                backups="vorag-backups",
            ),
        ),
    )


def _build_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    from vo_rag.api import main as main_module
    from vo_rag.api.singletons import SurfaceServices
    from vo_rag.core import config as config_module

    settings = _fake_settings()
    pool = _FakePool()

    @dataclass
    class _StubRerank:
        def rerank(self, query: str, candidates: list[Any], top_k: int = 20) -> list[Any]:
            return candidates[:top_k]

    services = SurfaceServices(
        settings=settings,
        db_pool=pool,
        qdrant=None,
        garage=None,
        embedding_client=SimpleNamespace(),
        retriever=SimpleNamespace(),
        reranker=_StubRerank(),
        answer_synthesizer=SimpleNamespace(),
        cloud_cost_tracker=None,
        kill_switch=None,
    )
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    app = main_module.create_app(settings=settings, services=services)
    app.state.db_pool = pool
    app.state.settings = settings
    return app


def _client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    return TestClient(_build_app(monkeypatch))


def _patch_route_dependency(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_module: Any,
    attr: str,
    replacement: Any,
) -> None:
    """Patch ``attr`` on both the source module and the route module.

    The route uses ``from X import Y`` which captures the function reference
    at import time. Patching only ``X.Y`` is not enough if the route module
    was already imported by an earlier test (or by the test runner's module
    cache) — the route still holds the original reference. We patch both to
    keep the test order-independent and to keep the local binding consistent
    with the source for future readers.

    Important: ``monkeypatch.setattr`` records the previous value of the
    attribute and restores it on teardown. When the same attribute is
    patched twice, only the second ``setattr`` reverts to the value that
    existed *between* the two sets — which is still the fake. To get a
    correct revert, we patch each attribute at most once.
    """
    importlib.import_module("vo_rag.api.routes.ingest")
    from vo_rag.api.routes import ingest as ingest_module

    monkeypatch.setattr(source_module, attr, replacement)
    if hasattr(ingest_module, attr):
        monkeypatch.setattr(ingest_module, attr, replacement)


# --------------------------------------------------------------------------- #
# URL submissions — block list
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "blocked_url",
    [
        "http://127.0.0.1",
        "http://127.0.0.1:8080",
        "http://10.0.0.5",
        "http://192.168.1.1",
        "http://169.254.169.254",  # cloud metadata
        "http://127.0.0.1",  # Tailscale
        "http://[::1]",
        "ftp://example.com",
        "javascript:alert(1)",
    ],
)
def test_intake_url_rejects_blocked_targets(
    monkeypatch: pytest.MonkeyPatch,
    blocked_url: str,
) -> None:
    """Private/Tailscale/link-local/loopback/bad-scheme URLs are 422'd."""

    from vo_rag.core.db.queries import envelopes as queries_envelopes

    insert_called = {"n": 0}

    async def _track_insert(pool: object, envelope: object) -> None:
        insert_called["n"] += 1

    _patch_route_dependency(
        monkeypatch,
        source_module=queries_envelopes,
        attr="insert_envelope",
        replacement=_track_insert,
    )

    client = _client(monkeypatch)
    response = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json={
            "source_type": "url",
            "raw_content": {"url": blocked_url},
        },
    )

    assert response.status_code == 422
    assert insert_called["n"] == 0, "DB insert must NOT run when SSRF gate blocks the URL"


def test_intake_url_ssrf_blocked_detail_is_operator_readable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 422 detail should mention SSRF so operators can grep logs."""

    from vo_rag.core.db.queries import envelopes as queries_envelopes

    async def _noop_insert(pool: object, envelope: object) -> None:
        return None

    _patch_route_dependency(
        monkeypatch,
        source_module=queries_envelopes,
        attr="insert_envelope",
        replacement=_noop_insert,
    )

    client = _client(monkeypatch)
    response = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json={"source_type": "url", "raw_content": {"url": "http://10.0.0.5"}},
    )

    assert response.status_code == 422
    detail_str = str(response.json()["detail"]).lower()
    assert "ssrf" in detail_str or "block" in detail_str or "10.0.0.5" in detail_str


def test_intake_url_runs_ssrf_gate_before_db_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The order matters: SSRF gate first, then DB. We pin the order."""

    from vo_rag.core.db.queries import envelopes as queries_envelopes
    from vo_rag.web import url_safety as url_safety_module

    order: list[str] = []

    async def _explode(*args: object, **kwargs: object) -> ResolvedURL:
        order.append("ssrf")
        raise SSRFBlocked("http://10.0.0.5", "blocked private IPv4")

    async def _track_insert(pool: object, envelope: object) -> None:
        order.append("db")

    _patch_route_dependency(
        monkeypatch,
        source_module=url_safety_module,
        attr="validate_external_url",
        replacement=_explode,
    )
    _patch_route_dependency(
        monkeypatch,
        source_module=queries_envelopes,
        attr="insert_envelope",
        replacement=_track_insert,
    )

    client = _client(monkeypatch)
    response = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json={"source_type": "url", "raw_content": {"url": "http://10.0.0.5"}},
    )

    assert response.status_code == 422
    # SSRF must have been called; DB must NOT.
    assert order == ["ssrf"]


# --------------------------------------------------------------------------- #
# URL submissions — happy path
# --------------------------------------------------------------------------- #
def test_intake_url_accepts_public_url_when_ssrf_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public URL that passes validate_external_url persists successfully."""

    from vo_rag.core.db.queries import envelopes as queries_envelopes
    from vo_rag.web import url_safety as url_safety_module

    async def _fake_validate(url: str) -> ResolvedURL:
        return ResolvedURL(
            url=url,
            scheme="https",
            host="example.com",
            ip="93.184.216.34",
            port=443,
        )

    captured: dict[str, Any] = {}

    async def _track_insert(pool: object, envelope: object) -> None:
        captured["envelope"] = envelope

    _patch_route_dependency(
        monkeypatch,
        source_module=url_safety_module,
        attr="validate_external_url",
        replacement=_fake_validate,
    )
    _patch_route_dependency(
        monkeypatch,
        source_module=queries_envelopes,
        attr="insert_envelope",
        replacement=_track_insert,
    )

    client = _client(monkeypatch)
    response = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json={
            "source_type": "url",
            "raw_content": {"url": "https://example.com/article", "title": "An article"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["source_type"] == "url"
    assert body["raw_content"]["url"] == "https://example.com/article"
    assert body["raw_content"]["title"] == "An article"
    envelope = captured["envelope"]
    assert envelope.raw_content["url"] == "https://example.com/article"


def test_intake_url_rejects_missing_url_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """A url submission with no url in raw_content is a 422."""

    from vo_rag.core.db.queries import envelopes as queries_envelopes

    async def _noop_insert(pool: object, envelope: object) -> None:
        return None

    _patch_route_dependency(
        monkeypatch,
        source_module=queries_envelopes,
        attr="insert_envelope",
        replacement=_noop_insert,
    )

    client = _client(monkeypatch)
    response = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json={"source_type": "url", "raw_content": {}},
    )

    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Non-URL submissions must not hit the SSRF gate.
# --------------------------------------------------------------------------- #
def test_text_prompt_does_not_invoke_ssrf_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SSRF gate is URL-only — text/file/api submissions must skip it."""

    from vo_rag.core.db.queries import envelopes as queries_envelopes
    from vo_rag.web import url_safety as url_safety_module

    invocations = {"n": 0}

    async def _track(*args: object, **kwargs: object) -> ResolvedURL:
        invocations["n"] += 1
        return ResolvedURL(url="x", scheme="http", host="x", ip="1.1.1.1", port=80)

    async def _noop_insert(pool: object, envelope: object) -> None:
        return None

    _patch_route_dependency(
        monkeypatch,
        source_module=url_safety_module,
        attr="validate_external_url",
        replacement=_track,
    )
    _patch_route_dependency(
        monkeypatch,
        source_module=queries_envelopes,
        attr="insert_envelope",
        replacement=_noop_insert,
    )

    client = _client(monkeypatch)
    response = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json={"source_type": "text_prompt", "raw_content": {"text": "no url here"}},
    )

    assert response.status_code == 200
    assert invocations["n"] == 0


# --------------------------------------------------------------------------- #
# Direct validator sanity: this test makes sure we exercise the real
# url_safety module too, not only through the route stub. If the real
# validator changes behavior, the route must follow.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_real_validator_still_blocks_loopback() -> None:
    """Sanity: the real validate_external_url still blocks 127.0.0.1.

    Pinned so a future refactor of the gate cannot silently regress.
    """
    with pytest.raises(SSRFBlocked):
        await validate_external_url("http://127.0.0.1:8080")

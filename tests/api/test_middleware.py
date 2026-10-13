"""ApiKeyMiddleware: X-API-Key / Cloudflare Access / 401 / CSRF / exempt paths.
F-001: never silent-admin. F-014: mutating CF Access requests need CSRF.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from vo_rag.api.auth import ApiKeyIdentity, make_csrf_token
from vo_rag.api.middleware import EXEMPT_PATHS, ApiKeyMiddleware


def _app(monkeypatch, api_key: str = "k", api_keys_json: str = "") -> FastAPI:
    settings = SimpleNamespace(
        api=SimpleNamespace(api_key=api_key, api_keys_json=api_keys_json, cors_origins=["*"]),
        portal=SimpleNamespace(csrf_secret="csrf-secret"),
    )

    from vo_rag.core import config as config_module

    monkeypatch.setattr(config_module, "get_settings", lambda: settings)

    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware)

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/ready")
    def ready() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/v1/protected")
    def protected() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/api/v1/protected")
    def protected_post() -> dict[str, bool]:
        return {"ok": True}

    return app


def test_health_exempt(monkeypatch) -> None:
    client = TestClient(_app(monkeypatch))
    assert client.get("/health").status_code == 200


def test_ready_exempt(monkeypatch) -> None:
    client = TestClient(_app(monkeypatch))
    assert client.get("/ready").status_code == 200


def test_protected_requires_key(monkeypatch) -> None:
    client = TestClient(_app(monkeypatch))
    assert client.get("/api/v1/protected").status_code == 401


def test_protected_with_key(monkeypatch) -> None:
    client = TestClient(_app(monkeypatch))
    assert client.get("/api/v1/protected", headers={"X-API-Key": "k"}).status_code == 200


def test_protected_with_scoped_key(monkeypatch) -> None:
    scoped = '{"cli":{"key":"cli-key","caller":"cli"}}'
    client = TestClient(_app(monkeypatch, api_keys_json=scoped))
    assert client.get("/api/v1/protected", headers={"X-API-Key": "cli-key"}).status_code == 200


def test_protected_with_cloudflare_access_jwt(monkeypatch) -> None:
    async def fake_access(request, settings):
        return ApiKeyIdentity(
            name="admin@example.com",
            is_admin=True,
            auth_source="cloudflare_access",
            email="admin@example.com",
        )

    from vo_rag.api import middleware as mw_module

    monkeypatch.setattr(mw_module, "authenticate_cloudflare_access", fake_access)
    client = TestClient(_app(monkeypatch))
    response = client.get(
        "/api/v1/protected",
        headers={"Cf-Access-Jwt-Assertion": "jwt"},
    )
    assert response.status_code == 200


def test_cloudflare_access_post_requires_csrf(monkeypatch) -> None:
    async def fake_access(request, settings):
        return ApiKeyIdentity(
            name="admin@example.com",
            is_admin=True,
            auth_source="cloudflare_access",
            email="admin@example.com",
        )

    from vo_rag.api import middleware as mw_module

    monkeypatch.setattr(mw_module, "authenticate_cloudflare_access", fake_access)
    client = TestClient(_app(monkeypatch))
    response = client.post(
        "/api/v1/protected",
        headers={"Cf-Access-Jwt-Assertion": "jwt"},
    )
    assert response.status_code == 403

    ok = client.post(
        "/api/v1/protected",
        headers={
            "Cf-Access-Jwt-Assertion": "jwt",
            "X-CSRF-Token": make_csrf_token("admin@example.com", "csrf-secret"),
        },
    )
    assert ok.status_code == 200


def test_api_key_post_is_csrf_exempt(monkeypatch) -> None:
    client = TestClient(_app(monkeypatch))
    assert client.post("/api/v1/protected", headers={"X-API-Key": "k"}).status_code == 200


def test_exempt_paths_include_ready_and_docs_and_metrics() -> None:
    for path in ("/health", "/ready", "/docs", "/openapi.json", "/metrics"):
        assert path in EXEMPT_PATHS


def test_state_carries_vo_rag_api_identity(monkeypatch) -> None:
    """Sanity: identity is set on request.state with the vo_rag namespace."""
    from vo_rag.api import middleware as mw_module

    captured: dict[str, object] = {}

    async def fake_access(request, settings):
        return ApiKeyIdentity(
            name="admin@example.com",
            is_admin=True,
            auth_source="cloudflare_access",
            email="admin@example.com",
        )

    monkeypatch.setattr(mw_module, "authenticate_cloudflare_access", fake_access)

    from vo_rag.core import config as config_module

    settings = SimpleNamespace(
        api=SimpleNamespace(api_key="k", api_keys_json="", cors_origins=["*"]),
        portal=SimpleNamespace(csrf_secret="csrf-secret"),
    )
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)

    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware)

    @app.get("/who")
    def who(request: Request) -> dict[str, object]:
        identity = getattr(request.state, "vo_rag_api_identity", None)
        if identity is not None:
            captured["identity"] = identity
            return {"name": identity.name, "auth_source": identity.auth_source}
        return {"name": "anonymous"}

    client = TestClient(app)
    response = client.get(
        "/who",
        headers={"Cf-Access-Jwt-Assertion": "jwt"},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "admin@example.com"
    assert response.json()["auth_source"] == "cloudflare_access"
    assert isinstance(captured.get("identity"), ApiKeyIdentity)

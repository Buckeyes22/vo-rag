"""GET /api/v1/portal/session — identity + CSRF + capabilities for the SPA."""

from __future__ import annotations

from tests.api.conftest import auth_header, make_client, make_services


def test_session_with_api_key_identity(monkeypatch) -> None:
    client = make_client(make_services(), monkeypatch=monkeypatch)
    resp = client.get("/api/v1/portal/session", headers=auth_header())
    assert resp.status_code == 200
    body = resp.json()
    assert body["auth_source"] == "api_key"
    assert body["user"]["email"] == ""
    assert body["csrf_token"] is None  # CSRF only applies to Access-cookie sessions
    caps = body["capabilities"]
    assert caps["single_admin"] is True
    assert caps["chunked_upload"] is False
    assert caps["legacy_api_keys"] is True
    assert body["modes"] == ["internal"]


def test_session_unauthenticated_401(monkeypatch) -> None:
    client = make_client(make_services(), monkeypatch=monkeypatch)
    assert client.get("/api/v1/portal/session").status_code == 401

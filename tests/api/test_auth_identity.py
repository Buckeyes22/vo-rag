"""F-001: identity_from_request must refuse with 401, never silent admin.
F-014 callers: ALLOWED_CALLERS restricted to vo-rag surface (api/cli/mcp/portal).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from vo_rag.api.auth import (
    ALLOWED_CALLERS,
    ApiKeyIdentity,
    authenticate_api_key,
    enforce_query_scope,
    identity_from_request,
)

pytestmark = pytest.mark.security

_TEST_API_KEY = "test-api-key"  # pragma: allowlist secret
_TEST_OTHER_KEY = "test-other-key"  # pragma: allowlist secret
_TEST_ADMIN_KEY = "test-admin-key"  # pragma: allowlist secret
_TEST_CI_KEY = "test-ci-key"  # pragma: allowlist secret


def _request_with_state(**state_values: object) -> object:
    request = SimpleNamespace()
    request.state = SimpleNamespace(**state_values)
    return request


def test_returns_identity_when_state_set() -> None:
    identity = ApiKeyIdentity(name="real-user", is_admin=False)
    req = _request_with_state(vo_rag_api_identity=identity)
    assert identity_from_request(req) is identity  # type: ignore[arg-type]


def test_raises_401_when_state_missing() -> None:
    req = _request_with_state()
    with pytest.raises(HTTPException) as excinfo:
        identity_from_request(req)  # type: ignore[arg-type]
    assert excinfo.value.status_code == 401
    assert "identity" in str(excinfo.value.detail).lower()


def test_raises_401_when_state_has_wrong_type() -> None:
    # A bug elsewhere might set vo_rag_api_identity to a string or dict —
    # refuse rather than silently grant admin.
    req = _request_with_state(vo_rag_api_identity="admin")
    with pytest.raises(HTTPException) as excinfo:
        identity_from_request(req)  # type: ignore[arg-type]
    assert excinfo.value.status_code == 401


def test_allowed_callers_does_not_contain_other_orgs() -> None:
    """vo-rag is single-tenant: no other caller organizations are allowed."""
    assert {"api", "cli", "mcp", "portal"} == ALLOWED_CALLERS


def _settings(*, api_key: str = "", api_keys_json: str = "") -> object:
    return SimpleNamespace(api=SimpleNamespace(api_key=api_key, api_keys_json=api_keys_json))


def test_authenticate_api_key_returns_none_for_missing() -> None:
    assert authenticate_api_key(None, _settings()) is None


def test_authenticate_api_key_returns_none_for_wrong_key() -> None:
    assert authenticate_api_key(_TEST_OTHER_KEY, _settings(api_key=_TEST_ADMIN_KEY)) is None


def test_authenticate_api_key_admin_key_is_admin() -> None:
    identity = authenticate_api_key(_TEST_ADMIN_KEY, _settings(api_key=_TEST_ADMIN_KEY))
    assert identity is not None
    assert identity.is_admin is True
    assert identity.auth_source == "api_key"


def test_authenticate_api_key_scoped_key() -> None:
    scoped = json.dumps(
        {
            "ci-runner": {
                "name": "ci-runner",
                "key": _TEST_CI_KEY,
                "caller": "cli",
                "allowed_modes": ["internal"],
                "allowed_collections": ["vorag_books_v1"],
            }
        }
    )
    identity = authenticate_api_key(_TEST_CI_KEY, _settings(api_keys_json=scoped))
    assert identity is not None
    assert identity.is_admin is False
    assert identity.caller == "cli"
    assert "internal" in identity.allowed_modes
    assert "vorag_books_v1" in identity.allowed_collections


def test_enforce_query_scope_admin_bypasses() -> None:
    identity = ApiKeyIdentity(name="admin", is_admin=True)
    enforce_query_scope(
        identity=identity,
        caller="api",
        mode="internal",
        collections=["anything"],
    )


def test_enforce_query_scope_caller_mismatch() -> None:
    identity = ApiKeyIdentity(
        name="ci", caller="cli", is_admin=False, allowed_modes=frozenset({"internal"})
    )
    with pytest.raises(HTTPException) as excinfo:
        enforce_query_scope(
            identity=identity,
            caller="api",
            mode="internal",
            collections=[],
        )
    assert excinfo.value.status_code == 403
    assert "caller" in str(excinfo.value.detail).lower()


def test_enforce_query_scope_mode_not_allowed() -> None:
    identity = ApiKeyIdentity(
        name="ci", caller="cli", is_admin=False, allowed_modes=frozenset({"internal"})
    )
    with pytest.raises(HTTPException) as excinfo:
        enforce_query_scope(
            identity=identity,
            caller="cli",
            mode="public",
            collections=[],
        )
    assert excinfo.value.status_code == 403


def test_enforce_query_scope_collection_not_allowed() -> None:
    identity = ApiKeyIdentity(
        name="ci",
        caller="cli",
        is_admin=False,
        allowed_modes=frozenset({"internal"}),
        allowed_collections=frozenset({"vorag_books_v1"}),
    )
    with pytest.raises(HTTPException) as excinfo:
        enforce_query_scope(
            identity=identity,
            caller="cli",
            mode="internal",
            collections=["vorag_vce_v1"],
        )
    assert excinfo.value.status_code == 403
    detail = excinfo.value.detail
    assert isinstance(detail, dict)
    assert detail.get("error") == "collection_not_allowed_for_api_key"
    assert "vorag_vce_v1" in detail.get("collections", [])

"""vo-rag API authentication: API key + Cloudflare Access JWT, CSRF, query
scope enforcement, and identity resolution.

F-001: ``identity_from_request`` must always raise 401 rather than granting
a silent admin identity when middleware didn't set one. This is the
non-negotiable invariant the test suite pins.

Notes:
- settings come from ``vo_rag.core.config.settings.get_settings``
- API keys come from ``VORAG_API_KEYS_JSON``
- the resolved identity lives on the request-state attribute
  ``vo_rag_api_identity``
- ``ALLOWED_CALLERS`` is ``{api, cli, mcp, portal}`` only — vo-rag is
  single-tenant; multi-tenant callers are out of scope.
"""

from __future__ import annotations

import base64
import hashlib
import hmac as hmac_module
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from hmac import compare_digest
from typing import Any

import httpx
from fastapi import HTTPException, Request

ALLOWED_CALLERS = {
    "api",
    "cli",
    "mcp",
    "portal",
}


@dataclass(frozen=True)
class ApiKeyIdentity:
    name: str
    caller: str | None = None
    allowed_modes: frozenset[str] = frozenset()
    allowed_collections: frozenset[str] = frozenset()
    is_admin: bool = False
    auth_source: str = "api_key"
    email: str | None = None


def normalize_caller(value: str | None, *, default: str = "api") -> str:
    caller = (value or default).strip().lower()
    if caller in ALLOWED_CALLERS:
        return caller
    return "api"


def authenticate_api_key(key: str | None, settings: Any) -> ApiKeyIdentity | None:
    if key is None:
        return None

    configured_admin_key = str(getattr(settings.api, "api_key", "") or "")
    if configured_admin_key and compare_digest(key, configured_admin_key):
        return ApiKeyIdentity(name="legacy", is_admin=True)

    for credential in _iter_scoped_credentials(
        str(getattr(settings.api, "api_keys_json", "") or "")
    ):
        if compare_digest(key, credential.key):
            return credential.identity

    return None


class AccessTokenError(ValueError):
    """Raised when a Cloudflare Access token is present but invalid."""


_JWKS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


async def authenticate_cloudflare_access(
    request: Request,
    settings: Any,
) -> ApiKeyIdentity | None:
    token = request.headers.get("Cf-Access-Jwt-Assertion") or request.cookies.get(
        "CF_Authorization"
    )
    if not token:
        return None

    portal = getattr(settings, "portal", None)
    team_domain = _string_attr(portal, "access_team_domain")
    audience = _string_attr(portal, "access_aud")
    allowed_emails = {
        email.strip().lower()
        for email in getattr(portal, "allowed_emails", [])
        if isinstance(email, str) and email.strip()
    }
    if not team_domain or not audience or not allowed_emails:
        raise AccessTokenError("cloudflare_access_not_configured")

    payload = await verify_cloudflare_access_jwt(
        token,
        team_domain=team_domain,
        audience=audience,
    )
    email = _payload_email(payload)
    if email is None or email.lower() not in allowed_emails:
        raise AccessTokenError("cloudflare_access_email_not_allowed")

    return ApiKeyIdentity(
        name=email,
        caller="portal",
        is_admin=True,
        auth_source="cloudflare_access",
        email=email,
    )


async def verify_cloudflare_access_jwt(
    token: str,
    *,
    team_domain: str,
    audience: str,
) -> dict[str, Any]:
    try:
        jwt = _import_pyjwt()
        header = jwt.get_unverified_header(token)
        kid = str(header.get("kid") or "")
        if not kid:
            raise AccessTokenError("cloudflare_access_missing_kid")

        jwks = await _fetch_access_jwks(team_domain)
        key_payload = next(
            (
                key
                for key in jwks.get("keys", [])
                if isinstance(key, dict) and key.get("kid") == kid
            ),
            None,
        )
        if key_payload is None:
            raise AccessTokenError("cloudflare_access_unknown_kid")

        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key_payload))
        decoded = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=audience,
            issuer=team_domain.rstrip("/"),
        )
        if not isinstance(decoded, dict):
            raise AccessTokenError("cloudflare_access_invalid_payload")
        return decoded
    except AccessTokenError:
        raise
    except Exception as exc:
        raise AccessTokenError("cloudflare_access_jwt_invalid") from exc


def make_csrf_token(email: str, secret: str, *, now: int | None = None) -> str:
    if not secret:
        raise ValueError("csrf_secret_not_configured")
    issued_at = int(time.time() if now is None else now)
    payload = f"{email}:{issued_at}"
    signature = hmac_module.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    encoded_sig = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    return f"{issued_at}.{encoded_sig}"


def validate_csrf_token(
    token: str | None,
    *,
    email: str,
    secret: str,
    max_age_s: int = 8 * 60 * 60,
) -> bool:
    if not token or not secret:
        return False
    try:
        issued_raw, _ = token.split(".", 1)
        issued_at = int(issued_raw)
    except ValueError:
        return False

    now = int(time.time())
    if issued_at > now + 60 or now - issued_at > max_age_s:
        return False

    expected = make_csrf_token(email, secret, now=issued_at)
    return compare_digest(token, expected)


def identity_from_request(request: Request) -> ApiKeyIdentity:
    """Return the resolved identity for the request, or raise 401.

    F-001: never fall back to a silent admin identity. A request that reaches
    a route requiring identity but doesn't have ``vo_rag_api_identity`` set on
    ``request.state`` is a middleware bug or a forgotten auth dep — either way,
    refuse with 401 rather than granting admin scope.
    """
    identity = getattr(request.state, "vo_rag_api_identity", None)
    if isinstance(identity, ApiKeyIdentity):
        return identity
    raise HTTPException(status_code=401, detail="identity unresolved")


def enforce_query_scope(
    *,
    identity: ApiKeyIdentity,
    caller: str,
    mode: str,
    collections: list[str],
) -> None:
    if identity.is_admin:
        return

    if identity.caller is not None and caller != identity.caller:
        raise HTTPException(status_code=403, detail="caller_not_allowed_for_api_key")

    if identity.allowed_modes and mode not in identity.allowed_modes:
        raise HTTPException(status_code=403, detail="mode_not_allowed_for_api_key")

    if identity.allowed_collections:
        blocked = sorted(
            collection
            for collection in collections
            if collection not in identity.allowed_collections
        )
        if blocked:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "collection_not_allowed_for_api_key",
                    "collections": blocked,
                },
            )


@dataclass(frozen=True)
class _ScopedCredential:
    key: str
    identity: ApiKeyIdentity


def _iter_scoped_credentials(raw_json: str) -> list[_ScopedCredential]:
    raw_json = raw_json.strip()
    if not raw_json:
        return []

    parsed = json.loads(raw_json)
    entries: Iterable[tuple[str, Any]]
    if isinstance(parsed, dict):
        entries = list(parsed.items())
    elif isinstance(parsed, list):
        entries = [(str(index), item) for index, item in enumerate(parsed, start=1)]
    else:
        raise ValueError("VORAG_API_KEYS_JSON must be a JSON object or array")

    credentials: list[_ScopedCredential] = []
    for fallback_name, value in entries:
        if not isinstance(value, dict):
            raise ValueError("VORAG_API_KEYS_JSON entries must be objects")

        name = _string_field(value, "name") or str(fallback_name)
        key = _string_field(value, "key")
        if not key:
            raise ValueError(f"VORAG_API_KEYS_JSON entry {name!r} is missing key")

        raw_caller = _string_field(value, "caller")
        caller = normalize_caller(raw_caller or name)
        credentials.append(
            _ScopedCredential(
                key=key,
                identity=ApiKeyIdentity(
                    name=name,
                    caller=caller,
                    allowed_modes=_string_set_field(value, "allowed_modes", "modes"),
                    allowed_collections=_string_set_field(
                        value,
                        "allowed_collections",
                        "collections",
                    ),
                ),
            )
        )
    return credentials


def _string_field(mapping: dict[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"VORAG_API_KEYS_JSON field {key!r} must be a string")
    value = value.strip()
    return value or None


def _string_set_field(mapping: dict[str, Any], *keys: str) -> frozenset[str]:
    value: object | None = None
    for key in keys:
        if key in mapping:
            value = mapping[key]
            break
    if value is None:
        return frozenset()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"VORAG_API_KEYS_JSON field {keys[0]!r} must be an array of strings")
    return frozenset(item.strip().lower() for item in value if item.strip())


async def _fetch_access_jwks(team_domain: str) -> dict[str, Any]:
    domain = team_domain.rstrip("/")
    url = f"{domain}/cdn-cgi/access/certs"
    cached = _JWKS_CACHE.get(url)
    now = time.time()
    if cached is not None and cached[0] > now:
        return cached[1]

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("keys"), list):
        raise AccessTokenError("cloudflare_access_bad_jwks")
    _JWKS_CACHE[url] = (now + 3600, payload)
    return payload


def _payload_email(payload: dict[str, Any]) -> str | None:
    value = payload.get("email") or payload.get("user_email")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _string_attr(obj: Any, name: str) -> str:
    value = getattr(obj, name, "")
    if isinstance(value, str):
        return value.strip()
    return ""


def _import_pyjwt() -> Any:
    try:
        import jwt
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise AccessTokenError("pyjwt_not_installed") from exc
    return jwt

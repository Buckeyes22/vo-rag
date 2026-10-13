"""GET /api/v1/portal/session — the SPA's identity/capability bootstrap.

The portal-session contract the React client expects
(user/auth_source/csrf_token/capabilities/modes). Auth itself is the
app-wide ApiKeyMiddleware + Cloudflare Access machinery in api/auth.py;
this route only REPORTS the resolved identity and mints the CSRF token
for Access-cookie sessions.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from vo_rag.api.auth import identity_from_request, make_csrf_token
from vo_rag.api.dependencies import get_services
from vo_rag.api.singletons import SurfaceServices

router = APIRouter()


@router.get("/api/v1/portal/session")
async def portal_session(
    request: Request,
    services: SurfaceServices = Depends(get_services),  # noqa: B008
) -> dict[str, Any]:
    identity = identity_from_request(request)
    email = identity.email or ""
    auth_source = identity.auth_source or ("cloudflare_access" if email else "api_key")
    csrf_token: str | None = None
    csrf_secret = str(getattr(services.settings.portal, "csrf_secret", "") or "")
    if auth_source == "cloudflare_access" and email and csrf_secret:
        csrf_token = make_csrf_token(email, csrf_secret)
    return {
        "user": {"email": email},
        "auth_source": auth_source,
        "csrf_token": csrf_token,
        "capabilities": {
            "single_admin": True,
            "chunked_upload": False,
            "safe_publication_mode": False,
            "legacy_api_keys": True,
        },
        "modes": ["internal"],
    }

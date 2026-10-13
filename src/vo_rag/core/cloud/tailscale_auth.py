"""Tailscale OAuth client — ephemeral auth-key minting for cloud workers.

Keys are ephemeral (device auto-removed on disconnect), preauthorized (no admin
approval), non-reusable (one enrollment per key), and carry `tag:vorag-cloud-worker`
so the ACL restricts them to Hatchet gRPC + Garage S3 only.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

TAILSCALE_API = "https://api.tailscale.com/api/v2"
DEFAULT_TAG = "tag:vorag-cloud-worker"


@dataclass
class _Token:
    value: str
    expires_at: datetime


class TailscaleProvisioner:
    """Mint ephemeral auth keys + manage cloud-worker devices in a tailnet."""

    def __init__(
        self,
        oauth_client_id: str,
        oauth_client_secret: str,
        tailnet: str,
        *,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        if not (oauth_client_id and oauth_client_secret and tailnet):
            raise ValueError("oauth_client_id, oauth_client_secret, tailnet are required")
        self._client_id = oauth_client_id
        self._client_secret = oauth_client_secret
        self._tailnet = tailnet
        self._http = http or httpx.AsyncClient(timeout=30.0)
        self._token: _Token | None = None
        self._lock = asyncio.Lock()

    async def _access_token(self) -> str:
        async with self._lock:
            now = datetime.now(timezone.utc)  # noqa: UP017
            if self._token and self._token.expires_at > now + timedelta(seconds=30):
                return self._token.value

            resp = await self._http.post(
                f"{TAILSCALE_API}/oauth/token",
                data={"grant_type": "client_credentials"},
                auth=(self._client_id, self._client_secret),
            )
            resp.raise_for_status()
            body = resp.json()
            ttl = int(body.get("expires_in", 3600))
            self._token = _Token(
                value=body["access_token"],
                expires_at=now + timedelta(seconds=ttl),
            )
            return self._token.value

    async def create_auth_key(
        self,
        *,
        ttl_seconds: int = 14400,
        tags: list[str] | tuple[str, ...] | str = (DEFAULT_TAG,),
        reusable: bool = False,
        ephemeral: bool = True,
        preauthorized: bool = True,
    ) -> str:
        """POST /tailnet/{tailnet}/keys; return the bare `tskey-...` string."""
        if isinstance(tags, str):
            tags = [tags]
        tags = list(tags)
        if not tags:
            raise ValueError("at least one tag required")

        token = await self._access_token()
        body: dict[str, Any] = {
            "capabilities": {
                "devices": {
                    "create": {
                        "reusable": reusable,
                        "ephemeral": ephemeral,
                        "preauthorized": preauthorized,
                        "tags": tags,
                    }
                }
            },
            "expirySeconds": ttl_seconds,
        }
        resp = await self._http.post(
            f"{TAILSCALE_API}/tailnet/{self._tailnet}/keys",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return str(resp.json()["key"])

    async def list_devices(self, tag: str = DEFAULT_TAG) -> list[dict[str, Any]]:
        token = await self._access_token()
        resp = await self._http.get(
            f"{TAILSCALE_API}/tailnet/{self._tailnet}/devices",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return [d for d in resp.json().get("devices", []) if tag in (d.get("tags") or [])]

    async def remove_device(self, device_id: str) -> None:
        token = await self._access_token()
        resp = await self._http.delete(
            f"{TAILSCALE_API}/device/{device_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()

    async def revoke_all(self, tag: str = DEFAULT_TAG) -> int:
        """Delete every device carrying `tag`. Returns count removed."""
        devices = await self.list_devices(tag=tag)
        for d in devices:
            await self.remove_device(d["id"])
        return len(devices)

    async def set_tag_deny_all(self, tag: str = DEFAULT_TAG) -> None:
        """Overwrite ACL so `tag` has no allowed destinations."""
        token = await self._access_token()
        get = await self._http.get(
            f"{TAILSCALE_API}/tailnet/{self._tailnet}/acl",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
        )
        get.raise_for_status()
        etag = get.headers.get("ETag", "")
        acl = get.json()
        acl["acls"] = [r for r in acl.get("acls", []) if tag not in r.get("src", [])]
        acl.setdefault("acls", []).append(
            {
                "action": "accept",
                "src": [tag],
                "dst": [],
            }
        )
        post = await self._http.post(
            f"{TAILSCALE_API}/tailnet/{self._tailnet}/acl",
            headers={
                "Authorization": f"Bearer {token}",
                "If-Match": etag,
                "Content-Type": "application/json",
            },
            json=acl,
        )
        post.raise_for_status()

    async def aclose(self) -> None:
        await self._http.aclose()

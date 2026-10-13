"""Emergency teardown of cloud resources. Target runtime <30s.

Steps: (1) deny Tailscale tag, (2) delete all tagged devices, (3) terminate
any live RunPod pods in the account.

R2 credential rotation is not a teardown step. Cloudflare deprecated
`/accounts/{id}/r2/tokens` (the old rotation endpoint) with no REST replacement
— permanent R2 access keys can now only be managed through the dashboard. The replacement pattern is
`/accounts/{id}/r2/temp-access-credentials`, which returns TTL-bounded
S3 STS-style creds that auto-expire — rotation becomes unnecessary because
exposure is inherently time-bounded.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from vo_rag.core.cloud.runpod_client import terminate_all_with_tag
from vo_rag.core.cloud.tailscale_auth import DEFAULT_TAG, TailscaleProvisioner
from vo_rag.core.models import VoBaseModel

log = logging.getLogger("vo_rag.cloud.kill_switch")


class KillReport(VoBaseModel):
    triggered_at: datetime
    reason: str
    tailscale_acl_updated: bool
    devices_removed: int
    # Honest name: counts RunPod pods terminated (was `skypilot_clusters_destroyed`).
    runpod_pods_terminated: int
    total_duration_ms: int
    errors: list[str]


class CloudKillSwitch:
    def __init__(
        self,
        tailscale: TailscaleProvisioner,
        *,
        terminate_compute: bool = True,
        tag: str = DEFAULT_TAG,
    ) -> None:
        self.ts = tailscale
        self.terminate_compute = terminate_compute
        self.tag = tag

    async def aclose(self) -> None:
        """Close the owned Tailscale HTTP client."""
        await self.ts.aclose()

    async def activate(self, reason: str) -> KillReport:
        if not reason:
            raise ValueError("reason is required (audit trail)")

        t0 = time.perf_counter()
        errors: list[str] = []
        acl_ok = False
        devices = 0
        compute_n = 0

        try:
            await self.ts.set_tag_deny_all(self.tag)
            acl_ok = True
        except Exception as exc:  # pragma: no cover
            errors.append(f"acl: {exc}")

        try:
            devices = await self.ts.revoke_all(self.tag)
        except Exception as exc:  # pragma: no cover
            errors.append(f"devices: {exc}")

        if self.terminate_compute:
            try:
                compute_n = await terminate_all_with_tag(name_prefix="vorag-")
            except Exception as exc:  # pragma: no cover
                errors.append(f"compute: {exc}")

        duration_ms = int((time.perf_counter() - t0) * 1000)
        return KillReport(
            triggered_at=datetime.now(timezone.utc),  # noqa: UP017
            reason=reason,
            tailscale_acl_updated=acl_ok,
            devices_removed=devices,
            runpod_pods_terminated=compute_n,
            total_duration_ms=duration_ms,
            errors=errors,
        )

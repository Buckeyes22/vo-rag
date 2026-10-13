"""Tests for CloudKillSwitch."""

from __future__ import annotations

import pytest

from vo_rag.core.cloud.kill_switch import CloudKillSwitch, KillReport


class _FakeTS:
    def __init__(
        self, acl_raises: bool = False, devices: int = 2, revoke_raises: bool = False
    ) -> None:
        self.acl_raises = acl_raises
        self.revoke_raises = revoke_raises
        self.devices = devices

    async def set_tag_deny_all(self, tag: str) -> None:
        if self.acl_raises:
            raise RuntimeError("acl fail")

    async def revoke_all(self, tag: str) -> int:
        if self.revoke_raises:
            raise RuntimeError("revoke fail")
        return self.devices


@pytest.mark.asyncio
async def test_activate_happy_path_under_30s() -> None:
    ks = CloudKillSwitch(_FakeTS(), terminate_compute=False)  # type: ignore[arg-type]
    rep = await ks.activate("test drill")
    assert isinstance(rep, KillReport)
    assert rep.tailscale_acl_updated is True
    assert rep.devices_removed == 2
    assert rep.runpod_pods_terminated == 0
    assert rep.total_duration_ms < 30000
    assert rep.errors == []


@pytest.mark.asyncio
async def test_activate_requires_reason() -> None:
    ks = CloudKillSwitch(_FakeTS(), terminate_compute=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        await ks.activate("")


@pytest.mark.asyncio
async def test_activate_partial_failure_continues() -> None:
    ks = CloudKillSwitch(
        _FakeTS(acl_raises=True),  # type: ignore[arg-type]
        terminate_compute=False,
    )
    rep = await ks.activate("partial")
    assert rep.tailscale_acl_updated is False
    assert rep.devices_removed == 2
    assert any(err.startswith("acl:") for err in rep.errors)


@pytest.mark.asyncio
async def test_activate_collects_all_errors() -> None:
    ks = CloudKillSwitch(
        _FakeTS(acl_raises=True, revoke_raises=True),  # type: ignore[arg-type]
        terminate_compute=False,
    )
    rep = await ks.activate("all fail")
    assert rep.tailscale_acl_updated is False
    assert len(rep.errors) >= 2
    assert any(err.startswith("acl:") for err in rep.errors)
    assert any(err.startswith("devices:") for err in rep.errors)

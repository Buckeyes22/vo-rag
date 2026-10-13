"""Shared fakes for cloud_cost_exporter tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from vo_rag.core.cloud.cost_tracker import SpendReport


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
@dataclass
class FakeTailscale:
    devices: list[dict[str, Any]] = field(default_factory=list)
    raise_on_list: Exception | None = None

    async def list_devices(self, tag: str = "tag:vorag-cloud-worker") -> list[dict[str, Any]]:
        if self.raise_on_list is not None:
            raise self.raise_on_list
        return self.devices

    async def aclose(self) -> None:
        pass


@dataclass
class FakeAcquiredConn:
    rows: list[Any] = field(default_factory=list)
    scalar: Any = 0
    error: Exception | None = None

    async def __aenter__(self) -> FakeAcquiredConn:
        if self.error is not None:
            raise self.error
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def fetch(self, query: str, *args: Any) -> list[Any]:  # noqa: ARG002
        return self.rows

    async def fetchval(self, query: str, *args: Any) -> Any:  # noqa: ARG002
        return self.scalar


@dataclass
class FakePool:
    fetch_rows: list[Any] = field(default_factory=list)
    fetchval_value: Any = 0
    raise_on_acquire: Exception | None = None

    def acquire(self) -> FakeAcquiredConn:
        if self.raise_on_acquire is not None:
            return FakeAcquiredConn(error=self.raise_on_acquire)
        return FakeAcquiredConn(rows=self.fetch_rows, scalar=self.fetchval_value)

    async def close(self) -> None:
        pass


@dataclass
class FakeTracker:
    report: SpendReport = field(
        default_factory=lambda: SpendReport(
            month="2026-06",
            total_spend_usd=12.34,
            budget_usd=50.0,
            budget_remaining_usd=37.66,
            budget_pct_used=24.68,
            by_provider={"runpod": 12.34},
            by_workload={"text-gen": 10.0, "vlm": 2.34},
            by_gpu_type={"A100": 12.34},
            jobs_this_month=7,
        )
    )
    raise_on_get: Exception | None = None
    call_count: int = 0

    async def get_spend_report(self) -> SpendReport:
        self.call_count += 1
        if self.raise_on_get is not None:
            raise self.raise_on_get
        return self.report


@pytest.fixture
def fake_tracker() -> FakeTracker:
    return FakeTracker()


@pytest.fixture
def fake_pool() -> FakePool:
    return FakePool()


@pytest.fixture
def fake_tailscale() -> FakeTailscale:
    return FakeTailscale()

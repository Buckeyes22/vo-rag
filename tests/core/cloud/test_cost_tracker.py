"""Tests for CloudCostTracker budget logic.

Db-touching paths are gated behind ``@pytest.mark.integration``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from vo_rag.core.cloud.cost_tracker import CloudCostTracker, SpendReport


def test_rejects_non_positive_budgets() -> None:
    with pytest.raises(ValueError):
        CloudCostTracker(MagicMock(), monthly_budget_usd=0, per_job_max_usd=10)
    with pytest.raises(ValueError):
        CloudCostTracker(MagicMock(), monthly_budget_usd=50, per_job_max_usd=-1)


def test_estimate_cost_basic() -> None:
    assert CloudCostTracker.estimate_cost(0.39, 2, 1) == pytest.approx(0.78)
    assert CloudCostTracker.estimate_cost(0.22, 3, 2) == pytest.approx(1.32)
    with pytest.raises(ValueError):
        CloudCostTracker.estimate_cost(-0.1, 1, 1)


def test_month_key_format() -> None:
    assert CloudCostTracker._month_key(datetime(2026, 4, 13, tzinfo=timezone.utc)) == "2026-04"  # noqa: UP017


@pytest.mark.asyncio
async def test_can_launch_rejects_over_per_job_cap() -> None:
    pool = _mock_pool(current_spend=0.0)
    t = CloudCostTracker(pool, monthly_budget_usd=50, per_job_max_usd=10)
    assert await t.can_launch(0.5) is True
    assert await t.can_launch(12) is False


@pytest.mark.asyncio
async def test_can_launch_rejects_over_monthly_cap() -> None:
    pool = _mock_pool(current_spend=48.0)
    t = CloudCostTracker(pool, monthly_budget_usd=50, per_job_max_usd=10)
    assert await t.can_launch(1.0) is True
    assert await t.can_launch(3.0) is False


@pytest.mark.asyncio
async def test_can_launch_zero_or_negative_estimate_false() -> None:
    pool = _mock_pool(current_spend=0.0)
    t = CloudCostTracker(pool, monthly_budget_usd=50, per_job_max_usd=10)
    assert await t.can_launch(0) is False
    assert await t.can_launch(-1) is False


def test_threshold_alert_fires_once_per_crossing(caplog: pytest.LogCaptureFixture) -> None:
    t = CloudCostTracker(
        MagicMock(),
        monthly_budget_usd=100,
        per_job_max_usd=50,
        alert_thresholds=(0.50, 0.75, 0.90),
    )
    with caplog.at_level("WARNING"):
        t._maybe_alert(40, 55)
    assert any("50%" in m for m in caplog.messages)
    caplog.clear()
    with caplog.at_level("WARNING"):
        t._maybe_alert(80, 80)
    assert not caplog.messages


@pytest.mark.asyncio
async def test_record_completion_rejects_unknown_status() -> None:
    pool = _mock_pool(current_spend=0.0)
    t = CloudCostTracker(pool, monthly_budget_usd=50, per_job_max_usd=10)
    with pytest.raises(ValueError):
        await t.record_completion(job_id="j", cost_usd=1.0, duration_s=60, status="weird")


@pytest.mark.asyncio
async def test_get_spend_report_aggregates_dimensions() -> None:
    rows = [
        {"provider": "runpod", "workload": "text-gen", "gpu_type": "A10G", "cost_usd": 1.25},
        {"provider": "runpod", "workload": "text-gen", "gpu_type": "A10G", "cost_usd": 2.75},
        {"provider": "runpod", "workload": "vlm", "gpu_type": "RTX3090", "cost_usd": 3.00},
    ]
    pool = _mock_pool(current_spend=0.0, report_rows=rows)
    t = CloudCostTracker(pool, monthly_budget_usd=50, per_job_max_usd=10)

    report = await t.get_spend_report()
    assert isinstance(report, SpendReport)
    assert report.total_spend_usd == pytest.approx(7.0)
    assert report.by_provider == {"runpod": 7.0}
    assert report.by_workload == {"text-gen": 4.0, "vlm": 3.0}
    assert report.by_gpu_type == {"A10G": 4.0, "RTX3090": 3.0}
    assert report.jobs_this_month == 3


@pytest.mark.asyncio
async def test_try_launch_atomic_budget_gate() -> None:
    pool = _mock_pool(current_spend=48.0)
    t = CloudCostTracker(pool, monthly_budget_usd=50, per_job_max_usd=10)
    ok = await t.try_launch(
        estimated_cost_usd=1.5,
        job_id="j1",
        provider="runpod",
        gpu_type="A10G",
        workload="text-gen",
    )
    assert ok is True
    conn = pool.acquire.return_value.__aenter__.return_value
    assert conn.execute.call_count >= 2


@pytest.mark.asyncio
async def test_try_launch_rejects_over_budget() -> None:
    pool = _mock_pool(current_spend=48.0)
    t = CloudCostTracker(pool, monthly_budget_usd=50, per_job_max_usd=10)
    ok = await t.try_launch(
        estimated_cost_usd=5.0,
        job_id="j2",
        provider="runpod",
        gpu_type="A10G",
        workload="text-gen",
    )
    assert ok is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_try_launch_integration_needs_live_db() -> None:
    pytest.skip("integration: requires live Postgres pool")


def _mock_pool(
    current_spend: float, report_rows: list[dict[str, object]] | None = None
) -> MagicMock:
    pool = MagicMock()
    acq = MagicMock()
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value={"s": current_spend})
    conn.fetch = AsyncMock(return_value=report_rows or [])
    conn.execute = AsyncMock(return_value="UPDATE 0")
    acq.__aenter__ = AsyncMock(return_value=conn)
    acq.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=acq)
    return pool

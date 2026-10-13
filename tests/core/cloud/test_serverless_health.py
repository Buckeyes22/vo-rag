"""Tests for the pure serverless-health classifier."""

from __future__ import annotations

from vo_rag.core.cloud.serverless_health import (
    COLD_START_DELAY_MS,
    VERDICT_COLD_STARTING,
    VERDICT_FAILING,
    VERDICT_HEALTHY_COLD,
    VERDICT_HEALTHY_WARM,
    VERDICT_IDLE,
    VERDICT_THROTTLED,
    VERDICT_UNKNOWN,
    classify,
)


def _health(**workers_and_jobs: int) -> dict[str, object]:
    workers = {
        k: v
        for k, v in workers_and_jobs.items()
        if k in {"idle", "ready", "running", "initializing", "throttled", "unhealthy"}
    }
    jobs = {
        k: v for k, v in workers_and_jobs.items() if k in {"inQueue", "inProgress", "completed"}
    }
    return {"workers": workers, "jobs": jobs}


def test_completed_warm_is_healthy_warm() -> None:
    verdict, reasons = classify(
        health=_health(idle=1, ready=1),
        probe_status="COMPLETED",
        delay_ms=800,
        execution_ms=120,
        probe_error=None,
    )
    assert verdict == VERDICT_HEALTHY_WARM
    assert any("warm" in r for r in reasons)


def test_completed_high_delay_is_healthy_cold() -> None:
    verdict, _ = classify(
        health=_health(running=1),
        probe_status="COMPLETED",
        delay_ms=COLD_START_DELAY_MS + 1,
        execution_ms=2000,
        probe_error=None,
    )
    assert verdict == VERDICT_HEALTHY_COLD


def test_failed_probe_is_failing() -> None:
    verdict, reasons = classify(
        health=_health(idle=2, ready=2),
        probe_status="FAILED",
        delay_ms=None,
        execution_ms=None,
        probe_error="OOM",
    )
    assert verdict == VERDICT_FAILING
    assert any("OOM" in r for r in reasons)


def test_all_throttled_no_runnable_is_throttled() -> None:
    verdict, reasons = classify(
        health=_health(throttled=2, inQueue=1),
        probe_status=None,
        delay_ms=None,
        execution_ms=None,
        probe_error=None,
    )
    assert verdict == VERDICT_THROTTLED
    assert any("capacity" in r for r in reasons)


def test_queued_but_workers_up_is_cold_starting() -> None:
    verdict, reasons = classify(
        health=_health(ready=4, inQueue=4, inProgress=0),
        probe_status=None,
        delay_ms=None,
        execution_ms=None,
        probe_error=None,
    )
    assert verdict == VERDICT_COLD_STARTING
    assert any("model weights" in r for r in reasons)


def test_idle_empty_queue_is_idle() -> None:
    verdict, _ = classify(
        health=_health(idle=3, ready=3, inQueue=0),
        probe_status=None,
        delay_ms=None,
        execution_ms=None,
        probe_error=None,
    )
    assert verdict == VERDICT_IDLE


def test_unhealthy_workers_are_reported_even_when_serving() -> None:
    verdict, reasons = classify(
        health=_health(running=1, unhealthy=1),
        probe_status="COMPLETED",
        delay_ms=500,
        execution_ms=100,
        probe_error=None,
    )
    assert verdict == VERDICT_HEALTHY_WARM
    assert any("unhealthy" in r for r in reasons)


def test_probe_timeout_while_queued_falls_through_to_cold_starting() -> None:
    verdict, _ = classify(
        health=_health(initializing=1, inQueue=1, inProgress=0),
        probe_status="IN_QUEUE",
        delay_ms=None,
        execution_ms=None,
        probe_error=None,
    )
    assert verdict == VERDICT_COLD_STARTING


def test_no_signal_is_unknown() -> None:
    verdict, _ = classify(
        health=_health(),
        probe_status=None,
        delay_ms=None,
        execution_ms=None,
        probe_error=None,
    )
    assert verdict == VERDICT_UNKNOWN

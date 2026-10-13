"""RunPod Serverless health diagnostics.

Distinguishes a *cold-starting* endpoint from a *broken* one so operators stop
misreading cold starts as outages. Backs `vo-rag system serverless-health`.

The authoritative probe is a native ``/run`` + ``/status`` poll: it is immune to
client HTTP timeouts and returns ``delayTime`` (worker wait incl. cold start) vs
``executionTime`` (GPU time once picked up). See the worker-state
semantics at https://docs.runpod.io/serverless/workers/overview#worker-states.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from pydantic import BaseModel

# Verdicts. HEALTHY_* mean the endpoint served a job end-to-end.
VERDICT_HEALTHY_WARM = "HEALTHY_WARM"
VERDICT_HEALTHY_COLD = "HEALTHY_COLD"
VERDICT_COLD_STARTING = "COLD_STARTING"
VERDICT_THROTTLED = "THROTTLED"
VERDICT_FAILING = "FAILING"
VERDICT_DEGRADED = "DEGRADED"
VERDICT_IDLE = "IDLE"
VERDICT_UNKNOWN = "UNKNOWN"

# delayTime at/above this (ms) on a completed probe means a cold start was
# absorbed (worker spin-up + weight load), not steady-state latency.
COLD_START_DELAY_MS = 15_000


class ServerlessHealthReport(BaseModel):
    """Result of a single endpoint health check."""

    role: str
    endpoint_id: str
    verdict: str
    reasons: list[str]
    health: dict[str, Any]
    probe_status: str | None = None
    delay_ms: int | None = None
    execution_ms: int | None = None
    probe_error: str | None = None

    @property
    def is_serving(self) -> bool:
        """True if the endpoint demonstrably served a job (or is just warming)."""
        return self.verdict in {
            VERDICT_HEALTHY_WARM,
            VERDICT_HEALTHY_COLD,
            VERDICT_COLD_STARTING,
            VERDICT_IDLE,
            VERDICT_THROTTLED,
        }


def classify(
    *,
    health: dict[str, Any],
    probe_status: str | None,
    delay_ms: int | None,
    execution_ms: int | None,
    probe_error: str | None,
) -> tuple[str, list[str]]:
    """Pure verdict logic. ``probe_status`` is None when no live probe was run.

    Precedence: a FAILED probe or crashed
    workers are real failures; a COMPLETED probe proves the endpoint serves
    (cold or warm); otherwise we read the health counters.
    """

    workers = health.get("workers", {}) if isinstance(health, dict) else {}
    jobs = health.get("jobs", {}) if isinstance(health, dict) else {}
    unhealthy = int(workers.get("unhealthy", 0) or 0)
    throttled = int(workers.get("throttled", 0) or 0)
    idle = int(workers.get("idle", 0) or 0)
    ready = int(workers.get("ready", 0) or 0)
    running = int(workers.get("running", 0) or 0)
    initializing = int(workers.get("initializing", 0) or 0)
    in_queue = int(jobs.get("inQueue", 0) or 0)
    in_progress = int(jobs.get("inProgress", 0) or 0)
    reasons: list[str] = []

    if unhealthy > 0:
        reasons.append(
            f"{unhealthy} unhealthy worker(s) — crashed; RunPod auto-retries up to 7 days"
        )

    # A live probe is the authoritative signal.
    if probe_status == "FAILED":
        reasons.append(f"probe job FAILED: {probe_error or 'no error detail'}")
        return VERDICT_FAILING, reasons
    if probe_status == "COMPLETED":
        if delay_ms is not None and delay_ms >= COLD_START_DELAY_MS:
            reasons.append(
                f"probe COMPLETED after a cold start (delayTime={delay_ms}ms, "
                f"executionTime={execution_ms}ms) — endpoint is fine, just spun up"
            )
            return VERDICT_HEALTHY_COLD, reasons
        reasons.append(
            f"probe COMPLETED warm (delayTime={delay_ms}ms, executionTime={execution_ms}ms)"
        )
        return VERDICT_HEALTHY_WARM, reasons
    if probe_status in {"IN_QUEUE", "IN_PROGRESS"}:
        reasons.append(
            f"probe still {probe_status} at poll timeout — likely a long cold start "
            "(weight load). Not a failure; re-run or wait."
        )
        # fall through to health-based qualifier below

    # No terminal probe result — infer from health counters.
    if throttled > 0 and (idle + ready + running) == 0:
        reasons.append(
            f"{throttled} throttled worker(s), none runnable — host capacity "
            "contention (not billed, not an error)"
        )
        return VERDICT_THROTTLED, reasons
    if in_queue > 0 and in_progress == 0 and (ready + running + initializing) > 0:
        reasons.append(
            "jobs queued with workers up but none draining — workers loading "
            "model weights (cold start in progress)"
        )
        return VERDICT_COLD_STARTING, reasons
    if (idle + ready) > 0 and in_queue == 0:
        reasons.append("workers idle/ready, queue empty")
        return VERDICT_IDLE, reasons
    if unhealthy > 0:
        return VERDICT_DEGRADED, reasons
    if not reasons:
        reasons.append("no decisive signal from health counters")
    return VERDICT_UNKNOWN, reasons


def _endpoint_root(base_url: str) -> str:
    """Strip the trailing /openai/v1 to get the native /v2/<id> root."""
    return base_url.removesuffix("/openai/v1").rstrip("/")


async def check_endpoint(
    *,
    role: str,
    endpoint_id: str,
    api_key: str,
    base_url: str,
    do_probe: bool = True,
    poll_timeout_s: float = 240.0,
    poll_interval_s: float = 3.0,
) -> ServerlessHealthReport:
    """Fetch /health and (optionally) run a native /run probe to a terminal state."""

    root = _endpoint_root(base_url)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    probe_status: str | None = None
    delay_ms: int | None = None
    execution_ms: int | None = None
    probe_error: str | None = None

    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        health_resp = await client.get(f"{root}/health")
        health_resp.raise_for_status()
        health: dict[str, Any] = health_resp.json()

        if do_probe:
            payload = {
                "input": {
                    "messages": [{"role": "user", "content": "ping"}],
                    "sampling_params": {"max_tokens": 1, "temperature": 0},
                }
            }
            run = await client.post(f"{root}/run", json=payload)
            run.raise_for_status()
            job_id = run.json().get("id")
            if not job_id:
                probe_status = "FAILED"
                probe_error = f"/run returned no job id: {run.text[:200]}"
            else:
                loop = asyncio.get_event_loop()
                deadline = loop.time() + poll_timeout_s
                while loop.time() < deadline:
                    st = await client.get(f"{root}/status/{job_id}")
                    st.raise_for_status()
                    body = st.json()
                    probe_status = body.get("status")
                    if probe_status in {"COMPLETED", "FAILED"}:
                        delay_ms = body.get("delayTime")
                        execution_ms = body.get("executionTime")
                        if probe_status == "FAILED":
                            probe_error = str(body.get("error") or body.get("output"))[:300]
                        break
                    await asyncio.sleep(poll_interval_s)

    verdict, reasons = classify(
        health=health,
        probe_status=probe_status,
        delay_ms=delay_ms,
        execution_ms=execution_ms,
        probe_error=probe_error,
    )
    return ServerlessHealthReport(
        role=role,
        endpoint_id=endpoint_id,
        verdict=verdict,
        reasons=reasons,
        health=health,
        probe_status=probe_status,
        delay_ms=delay_ms,
        execution_ms=execution_ms,
        probe_error=probe_error,
    )


__all__ = [
    "ServerlessHealthReport",
    "check_endpoint",
    "classify",
    "COLD_START_DELAY_MS",
    "VERDICT_HEALTHY_WARM",
    "VERDICT_HEALTHY_COLD",
    "VERDICT_COLD_STARTING",
    "VERDICT_THROTTLED",
    "VERDICT_FAILING",
    "VERDICT_DEGRADED",
    "VERDICT_IDLE",
    "VERDICT_UNKNOWN",
]

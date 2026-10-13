"""cloud: RunPod serverless integration + budget governor + static workload router.

Public surface:
- `ServerlessClient` / `ServerlessResponse` — async httpx client for RunPod Serverless
- `classify` / health verdicts / `check_endpoint` — 8-verdict serverless health probe
- `CloudCostTracker` / `SpendReport` — advisory-lock atomic budget gate
- `CloudKillSwitch` / `KillReport` — emergency teardown (Tailscale + RunPod)
- `R2Staging` / `StagedInput` / `StagedOutputSlot` — presigned-URL artifact exchange
- `workload_location` / `WorkloadLocation` — static workload→location map
"""

from __future__ import annotations

from vo_rag.core.cloud.cost_tracker import CloudCostTracker, SpendReport
from vo_rag.core.cloud.kill_switch import CloudKillSwitch, KillReport
from vo_rag.core.cloud.r2_staging import R2Staging, StagedInput, StagedOutputSlot
from vo_rag.core.cloud.router import WorkloadLocation, workload_location
from vo_rag.core.cloud.serverless import ServerlessClient, ServerlessResponse
from vo_rag.core.cloud.serverless_health import (
    COLD_START_DELAY_MS,
    VERDICT_COLD_STARTING,
    VERDICT_DEGRADED,
    VERDICT_FAILING,
    VERDICT_HEALTHY_COLD,
    VERDICT_HEALTHY_WARM,
    VERDICT_IDLE,
    VERDICT_THROTTLED,
    VERDICT_UNKNOWN,
    ServerlessHealthReport,
    check_endpoint,
    classify,
)

__all__ = [
    "ServerlessClient",
    "ServerlessResponse",
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
    "CloudCostTracker",
    "SpendReport",
    "CloudKillSwitch",
    "KillReport",
    "R2Staging",
    "StagedInput",
    "StagedOutputSlot",
    "WorkloadLocation",
    "workload_location",
]

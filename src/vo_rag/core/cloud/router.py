"""Static workload→location router.

A fixed map, with no dynamic burst or sovereignty routing:
- text-gen / vlm / dsi   → RunPod serverless
- embedding              → local (RunPod for big batches)
- query-handling         → local
"""

from __future__ import annotations

from enum import StrEnum


class WorkloadLocation(StrEnum):
    """Compute location for a workload class."""

    RUNPOD_SERVERLESS = "runpod_serverless"
    LOCAL = "local"


_STATIC_MAP: dict[str, WorkloadLocation] = {
    "text-gen": WorkloadLocation.RUNPOD_SERVERLESS,
    "vlm": WorkloadLocation.RUNPOD_SERVERLESS,
    "dsi": WorkloadLocation.RUNPOD_SERVERLESS,
    "embedding": WorkloadLocation.LOCAL,
    "query-handling": WorkloadLocation.LOCAL,
}


def workload_location(workload: str) -> WorkloadLocation:
    """Return the canonical compute location for *workload*.

    Raises:
        ValueError: for unknown workload names.
    """
    loc = _STATIC_MAP.get(workload)
    if loc is None:
        raise ValueError(f"unknown workload: {workload!r}")
    return loc

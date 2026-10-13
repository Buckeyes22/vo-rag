"""Single source of truth for the worker roster.

The roster is declared once in pyproject.toml under
``[tool.vo_rag].workers`` as a list of inline tables. Each entry becomes a
strict ``WorkerSpec``; every workload is validated against the static
workload→location map in ``vo_rag.core.cloud.router`` (``misc`` is exempt
because it is accounting-only, not a real compute workload).

Mirrors ``vo_rag.core.config.pipelines`` — same loader shape, same duplicate
detection, same discover-the-pyproject-by-walking-up behaviour.
"""

from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path

from vo_rag.core.cloud.router import WorkloadLocation, workload_location
from vo_rag.core.models.common import VoBaseModel


class WorkerStatus(StrEnum):
    """Lifecycle status for a worker entry in the roster."""

    PLANNED = "planned"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class WorkerSpec(VoBaseModel):
    """A single deployable worker surface.

    Fields:
        name: registry key. Must be unique across the roster.
        workload: workload tag routed via ``workload_location()`` (or
            ``misc`` for exporter / accounting-only workers).
        location: the compute location the worker actually runs at. Must
            agree with ``workload_location(workload)`` for every workload
            except ``misc``, which is always local.
        entrypoint: how the worker is launched — a Python ``module:attr``
            (e.g. ``vo_rag.workers.embedding.server:app``) or a
            worker-vllm hub string for RunPod serverless deployments.
        deploy_artifacts: filesystem paths (Dockerfiles, templates,
            compose fragments) that materialize the worker in CI/staging.
        gpu_required: True when the worker would crash without a GPU
            (e.g. local BGE-M3, RunPod vLLM endpoints). Cloud-cost
            accounting workers are CPU-only.
        test_level: which test layers cover this worker (e.g.
            ``unit+gpu-integration``, ``contract+live``,
            ``unit+db-integration``).
        status: lifecycle state.
    """

    name: str
    workload: str
    location: WorkloadLocation
    entrypoint: str
    deploy_artifacts: list[str]
    gpu_required: bool
    test_level: str
    status: WorkerStatus


def _find_pyproject() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("pyproject.toml not found above vo_rag/workers/registry.py")


def load_worker_registry(pyproject_path: Path | None = None) -> dict[str, WorkerSpec]:
    """Load the worker roster from pyproject.toml.

    Args:
        pyproject_path: explicit override. When ``None``, walks up from this
            file looking for the nearest ``pyproject.toml``.

    Returns:
        Mapping of worker name → ``WorkerSpec``.

    Raises:
        FileNotFoundError: pyproject.toml missing.
        ValueError: duplicate worker name in roster, or a non-``misc``
            workload is not registered in the static compute router, or
            the registry-declared location disagrees with the router.
    """
    path = pyproject_path or _find_pyproject()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    raw = data.get("tool", {}).get("vo_rag", {}).get("workers", [])
    registry: dict[str, WorkerSpec] = {}
    for entry in raw:
        spec = WorkerSpec(**entry)
        if spec.name in registry:
            raise ValueError(f"duplicate worker in roster: {spec.name!r}")
        # Validate every real-compute workload against the static router. The
        # ``misc`` bucket is exporter/accounting-only — not a real compute
        # workload — so it is intentionally exempt from router lookups.
        if spec.workload != "misc":
            router_loc = workload_location(spec.workload)
            if router_loc != spec.location:
                raise ValueError(
                    f"worker {spec.name!r} declares location={spec.location.value!r} "
                    f"for workload={spec.workload!r}, but workload_location() says "
                    f"{router_loc.value!r}"
                )
        registry[spec.name] = spec
    return registry

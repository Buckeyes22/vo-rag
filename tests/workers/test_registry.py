"""Contract tests for src/vo_rag/workers/registry.py.

The worker registry is the single source of truth for deployable worker
surfaces. Five entries are expected:

    embedding          (local BGE-M3)            — workload=embedding
    runpod_text_gen    (RunPod serverless vLLM)  — workload=text-gen
    runpod_vlm         (RunPod serverless vLLM)  — workload=vlm
    runpod_dsi         (RunPod serverless vLLM)  — workload=dsi
    cloud_cost_exporter (local accounting)        — workload=misc

Each entry is validated against the static workload→location map from
``vo_rag.core.cloud.router.workload_location`` (with ``misc`` exempt because
it is accounting-only, not a real compute workload).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vo_rag.core.cloud.router import WorkloadLocation, workload_location
from vo_rag.workers.registry import (
    WorkerSpec,
    WorkerStatus,
    load_worker_registry,
)

EXPECTED_WORKERS: tuple[str, ...] = (
    "embedding",
    "runpod_text_gen",
    "runpod_vlm",
    "runpod_dsi",
    "cloud_cost_exporter",
)

# Worker name -> (workload, location) as declared in pyproject.toml [tool.vo_rag.workers].
EXPECTED_WORKLOAD_MAP: dict[str, tuple[str, WorkloadLocation]] = {
    "embedding": ("embedding", WorkloadLocation.LOCAL),
    "runpod_text_gen": ("text-gen", WorkloadLocation.RUNPOD_SERVERLESS),
    "runpod_vlm": ("vlm", WorkloadLocation.RUNPOD_SERVERLESS),
    "runpod_dsi": ("dsi", WorkloadLocation.RUNPOD_SERVERLESS),
    "cloud_cost_exporter": ("misc", WorkloadLocation.LOCAL),
}


def test_registry_loads_exactly_five_entries() -> None:
    reg = load_worker_registry()
    assert set(reg) == set(EXPECTED_WORKERS)
    assert len(reg) == 5
    for name, spec in reg.items():
        assert isinstance(spec, WorkerSpec), f"{name} must be a WorkerSpec"


def test_registry_entries_expose_required_fields() -> None:
    reg = load_worker_registry()
    for name, spec in reg.items():
        # Every entry surfaces the deployment surface.
        assert spec.entrypoint, f"{name} missing entrypoint"
        assert spec.deploy_artifacts, f"{name} missing deploy_artifacts"
        assert isinstance(spec.gpu_required, bool), f"{name} gpu_required must be bool"
        assert spec.test_level, f"{name} missing test_level"
        assert isinstance(spec.status, WorkerStatus), f"{name} status must be WorkerStatus"


def test_registry_workload_matches_static_router_for_real_compute() -> None:
    """For non-misc workloads, workload_location() must agree with the registry.

    ``misc`` is the exporter/accounting-only bucket — it is intentionally
    exempt from the static compute router and stays local.
    """
    reg = load_worker_registry()
    for name, spec in reg.items():
        if spec.workload == "misc":
            assert spec.location is WorkloadLocation.LOCAL
            continue
        assert workload_location(spec.workload) == spec.location, (
            f"workload_location({spec.workload!r}) disagrees with "
            f"registry location for worker {name!r}"
        )


def test_registry_spec_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        WorkerSpec(  # type: ignore[call-arg]
            name="bogus",
            workload="embedding",
            location=WorkloadLocation.LOCAL,
            entrypoint="x:app",
            deploy_artifacts=["Dockerfile"],
            gpu_required=True,
            test_level="unit",
            status=WorkerStatus.PLANNED,
            not_a_real_field="nope",
        )


def test_registry_expected_workload_and_location_per_entry() -> None:
    reg = load_worker_registry()
    for name, (workload, location) in EXPECTED_WORKLOAD_MAP.items():
        spec = reg[name]
        assert spec.workload == workload, f"{name} workload"
        assert spec.location is location, f"{name} location"


def test_registry_entrypoints_are_the_expected_modules() -> None:
    reg = load_worker_registry()
    assert reg["embedding"].entrypoint == "vo_rag.workers.embedding.server:app"
    assert reg["cloud_cost_exporter"].entrypoint == "services.cloud_cost_exporter.app:app"
    # RunPod serverless workers are deployed via the worker-vllm hub.
    for name in ("runpod_text_gen", "runpod_vlm", "runpod_dsi"):
        assert reg[name].entrypoint == "worker-vllm hub", f"{name} entrypoint"


def test_registry_deploy_artifacts_reference_templates() -> None:
    reg = load_worker_registry()
    assert "workers/embedding/Dockerfile" in reg["embedding"].deploy_artifacts
    assert "workers/embedding/Dockerfile.local" in reg["embedding"].deploy_artifacts
    assert "workers/runpod_vllm/templates/text-gen.json" in reg["runpod_text_gen"].deploy_artifacts
    assert "workers/runpod_vllm/templates/vlm.json" in reg["runpod_vlm"].deploy_artifacts
    assert "workers/runpod_vllm/templates/dsi.json" in reg["runpod_dsi"].deploy_artifacts
    assert "services/cloud_cost_exporter/Dockerfile" in reg["cloud_cost_exporter"].deploy_artifacts


def test_registry_gpu_required_matches_workload() -> None:
    reg = load_worker_registry()
    assert reg["embedding"].gpu_required is True
    assert reg["runpod_text_gen"].gpu_required is True
    assert reg["runpod_vlm"].gpu_required is True
    assert reg["runpod_dsi"].gpu_required is True
    assert reg["cloud_cost_exporter"].gpu_required is False


def test_registry_duplicate_name_raises(tmp_path) -> None:
    bad = tmp_path / "pyproject.toml"
    bad.write_text(
        "[tool.vo_rag]\n"
        "workers = [\n"
        '  { name = "dup", workload = "embedding", '
        'location = "local", entrypoint = "x:a", deploy_artifacts = ["Dockerfile"], '
        'gpu_required = true, test_level = "unit", status = "active" },\n'
        '  { name = "dup", workload = "vlm", '
        'location = "runpod_serverless", entrypoint = "x:b", deploy_artifacts = ["Dockerfile"], '
        'gpu_required = true, test_level = "unit", status = "active" },\n'
        "]\n"
    )
    with pytest.raises(ValueError, match="duplicate worker in roster"):
        load_worker_registry(bad)

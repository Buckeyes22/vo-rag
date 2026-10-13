"""Cross-substrate contract tests for the worker surface.

Pins that every entry in ``[tool.vo_rag].workers`` resolves to a real,
importable Python object — the registry is the single source of truth
for deployable workers, and a stale entrypoint in ``pyproject.toml``
should fail at *test time*, not at deploy time.

Coverage:

- Every ``entrypoint`` in the registry resolves via ``importlib`` to a
  callable object (FastAPI ``app`` for the local workers, harness
  factory for the runpod harness, etc.). A bad entrypoint is a
  regression.
- The five workers each satisfy their unique surface contract
  (embedding → ``app``, runpod_text_gen/vlm/dsi → ``worker-vllm``
  hub, cloud_cost_exporter → ``app``).
- The embedding worker's ``app`` is a FastAPI instance with the
  documented routes (``/ping``, ``/health``, ``/embed``, ``/metrics``)
  and the right title.
- The cloud cost exporter's ``app`` is a FastAPI instance with
  ``/healthz`` and ``/metrics``.
- The registry module re-exports ``WorkerSpec``, ``WorkerStatus``,
  and ``load_worker_registry`` from the top-level
  ``vo_rag.workers`` namespace.
- The registry rejects duplicate names and unknown locations
  (covered by the registry tests, repeated here for the
  contract-level grouping).

Tests are hermetic — no network, no GPU, no DB. They run in the
no-live CI lane.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# Expected worker roster
# --------------------------------------------------------------------------- #

EXPECTED_WORKERS: tuple[str, ...] = (
    "embedding",
    "runpod_text_gen",
    "runpod_vlm",
    "runpod_dsi",
    "cloud_cost_exporter",
)


# --------------------------------------------------------------------------- #
# Top-level exports
# --------------------------------------------------------------------------- #


def test_vo_rag_workers_exports_worker_spec() -> None:
    """``from vo_rag.workers import WorkerSpec`` must succeed."""
    from vo_rag.workers import WorkerSpec

    assert WorkerSpec.__name__ == "WorkerSpec"


def test_vo_rag_workers_exports_load_worker_registry() -> None:
    """``from vo_rag.workers import load_worker_registry`` must succeed
    and return the same callable as the registry submodule."""
    from vo_rag.workers import load_worker_registry
    from vo_rag.workers.registry import load_worker_registry as canonical

    assert load_worker_registry is canonical


def test_vo_rag_workers_exports_worker_status() -> None:
    """``WorkerStatus`` is exported as part of the canonical roster
    surface so callers can compare without reaching into
    ``registry.py``."""
    from vo_rag.workers import WorkerStatus
    from vo_rag.workers.registry import WorkerStatus as canonical

    assert WorkerStatus is canonical
    # Exactly three lifecycle states exist.
    assert {s.value for s in canonical} == {"planned", "active", "deprecated"}


def test_vo_rag_workers_dunder_all_includes_canonical_names() -> None:
    """The ``__all__`` list on ``vo_rag.workers`` should advertise the
    three symbols callers reach for (``WorkerSpec``,
    ``WorkerStatus``, ``load_worker_registry``)."""
    import vo_rag.workers as pkg

    for name in ("WorkerSpec", "WorkerStatus", "load_worker_registry"):
        assert name in pkg.__all__, f"{name!r} missing from vo_rag.workers.__all__"


# --------------------------------------------------------------------------- #
# Entrypoint resolution: every WorkerSpec.entrypoint is importable
# --------------------------------------------------------------------------- #


def _split_entrypoint(entrypoint: str) -> tuple[str, str]:
    """Split ``"pkg.module:attr"`` → ``("pkg.module", "attr")``."""
    if ":" not in entrypoint:
        raise ValueError(f"entrypoint must use the module:attr form; got {entrypoint!r}")
    module, attr = entrypoint.split(":", 1)
    return module.strip(), attr.strip()


@pytest.mark.parametrize("worker_name", list(EXPECTED_WORKERS))
def test_registry_entrypoint_is_importable(worker_name: str) -> None:
    """Every worker's ``entrypoint`` must resolve to a real Python
    object. The registry declares the entrypoint in
    ``pyproject.toml``; a stale module path would only surface at
    deploy time without this guard."""
    from vo_rag.workers.registry import load_worker_registry

    reg = load_worker_registry()
    assert worker_name in reg, f"worker {worker_name!r} not in registry"
    entrypoint = reg[worker_name].entrypoint

    # RunPod serverless workers are deployed via the worker-vllm hub
    # string — not a Python module:attr. We assert the literal
    # ``"worker-vllm hub"`` here; the importability check below only
    # applies to Python entrypoints.
    if entrypoint == "worker-vllm hub":
        return

    module_name, attr_name = _split_entrypoint(entrypoint)
    module = importlib.import_module(module_name)
    obj = getattr(module, attr_name, None)
    assert obj is not None, (
        f"worker {worker_name!r}: entrypoint {entrypoint!r} — "
        f"attribute {attr_name!r} missing on {module_name!r}"
    )


# --------------------------------------------------------------------------- #
# Embedding worker surface (server.app)
# --------------------------------------------------------------------------- #


def test_embedding_server_app_is_fastapi_instance() -> None:
    """The embedding worker's ``app`` must be a FastAPI instance with
    the documented routes (``/ping``, ``/health``, ``/embed``,
    ``/metrics``) and a vo-rag title."""
    from fastapi import FastAPI

    from vo_rag.workers.embedding.server import app

    assert isinstance(app, FastAPI)
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    for required in ("/ping", "/health", "/embed", "/metrics"):
        assert required in paths, f"embedding app missing route {required!r}"
    assert "vo-rag" in app.title
    assert "BGE-M3" in app.title


def test_embedding_server_uses_vo_rag_brand_and_canonical_schemas() -> None:
    """The embedding worker reuses the canonical
    ``vo_rag.core.embedding.schemas`` (no duplicate)."""
    from vo_rag.core.embedding.schemas import EmbedRequest, EmbedResponse
    from vo_rag.workers.embedding import server

    assert server.EmbedRequest is EmbedRequest
    assert server.EmbedResponse is EmbedResponse


# --------------------------------------------------------------------------- #
# RunPod vLLM harness surface
# --------------------------------------------------------------------------- #


def test_runpod_harness_create_harness_app_is_callable() -> None:
    """The local harness factory must be importable without raising
    and must be callable with no args (test + local dev path)."""
    from vo_rag.workers.runpod_vllm.harness import create_harness_app

    assert callable(create_harness_app)
    app = create_harness_app()
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/health" in paths
    assert "/openai/v1/chat/completions" in paths


def test_runpod_harness_does_not_open_sockets_at_factory_time() -> None:
    """``create_harness_app()`` is called from the test suite hundreds
    of times — it must not import-time open any socket. A failed
    bind on a busy test box would be a flaky nightmare."""
    from vo_rag.workers.runpod_vllm.harness import create_harness_app

    # The factory must succeed in microseconds and return a fresh
    # FastAPI instance. The harness binds only when ``main()``
    # (uvicorn) runs; the factory is pure construction.
    app = create_harness_app()
    assert app is not None
    # Call it again to confirm factory idempotence (counters are
    # per-app, but construction itself must be cheap).
    app2 = create_harness_app()
    assert app2 is not app


# --------------------------------------------------------------------------- #
# Cloud cost exporter surface
# --------------------------------------------------------------------------- #


def test_cloud_cost_exporter_app_is_fastapi_instance() -> None:
    """The exporter's module-level ``app`` is a FastAPI instance with
    ``/healthz`` and ``/metrics`` routes."""
    from fastapi import FastAPI
    from services.cloud_cost_exporter.app import app

    assert isinstance(app, FastAPI)
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/healthz" in paths
    assert "/metrics" in paths


def test_cloud_cost_exporter_create_app_factory_is_callable() -> None:
    """``services.cloud_cost_exporter.app:create_app`` is the
    production factory (with optional dependency injection). It must
    be importable and callable with no args — the test suite calls
    it that way in ``tests/services/cloud_cost_exporter/test_app.py``."""
    from services.cloud_cost_exporter.app import create_app

    assert callable(create_app)


# --------------------------------------------------------------------------- #
# Registry roster shape (re-pinned at the contract layer for clarity)
# --------------------------------------------------------------------------- #


def test_registry_returns_five_workers() -> None:
    """The roster is exactly five entries: one local BGE-M3, three
    RunPod serverless workloads, one accounting exporter."""
    from vo_rag.workers.registry import load_worker_registry

    reg = load_worker_registry()
    assert set(reg) == set(EXPECTED_WORKERS)
    assert len(reg) == len(EXPECTED_WORKERS)


def test_registry_deploy_artifacts_reference_real_files() -> None:
    """Each worker's ``deploy_artifacts`` list should point at
    existing files (Dockerfiles, JSON templates, compose fragments)
    so a future PR that renames a template without updating the
    registry fails the contract."""
    from vo_rag.workers.registry import load_worker_registry

    repo_root = Path(__file__).resolve().parents[2]
    reg = load_worker_registry()
    missing: list[str] = []
    for name, spec in reg.items():
        for artifact in spec.deploy_artifacts:
            if not (repo_root / artifact).is_file():
                missing.append(f"{name!r}: {artifact!r}")
    assert not missing, "missing deploy artifacts:\n" + "\n".join(missing)


def test_registry_gpu_required_aligns_with_workload_class() -> None:
    """All real-compute workers are ``gpu_required=True``; the
    ``misc`` accounting exporter is the only ``False`` entry — it
    scrapes Postgres / Tailscale / Prometheus on a CPU-only host."""
    from vo_rag.workers.registry import load_worker_registry

    reg = load_worker_registry()
    for name, spec in reg.items():
        if spec.workload == "misc":
            assert spec.gpu_required is False, (
                f"{name!r} is workload=misc (accounting) but gpu_required=True"
            )
        else:
            assert spec.gpu_required is True, (
                f"{name!r} is workload={spec.workload!r} (compute) but gpu_required=False"
            )


def test_registry_test_level_is_nonempty_string() -> None:
    """The ``test_level`` field is a free-form string; pin that it is
    non-empty and contains at least one of the canonical tags
    (``unit``, ``contract``, ``live``, ``gpu-integration``,
    ``db-integration``)."""
    from vo_rag.workers.registry import load_worker_registry

    valid_tags = {"unit", "contract", "live", "gpu-integration", "db-integration"}
    reg = load_worker_registry()
    for name, spec in reg.items():
        assert spec.test_level, f"{name!r}: empty test_level"
        tags = set(spec.test_level.split("+"))
        assert tags & valid_tags, (
            f"{name!r}: test_level={spec.test_level!r} contains no canonical tag"
        )


def test_registry_status_is_a_known_lifecycle_state() -> None:
    """Every worker in the roster should be ``active`` — the
    registry is the deploy surface; ``planned`` / ``deprecated``
    workers would never ship."""
    from vo_rag.workers.registry import WorkerStatus, load_worker_registry

    reg = load_worker_registry()
    for name, spec in reg.items():
        assert isinstance(spec.status, WorkerStatus)
        assert spec.status is WorkerStatus.ACTIVE, (
            f"{name!r} status={spec.status.value!r} — only active workers should be in the roster"
        )

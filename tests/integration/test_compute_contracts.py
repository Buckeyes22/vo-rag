"""Cross-substrate integration contract tests.

End-to-end contract tests that pin the *whole* compute substrate
together. The unit tests in ``tests/workers/`` and
``tests/services/cloud_cost_exporter`` cover each piece in isolation;
this file is the glue. It exercises:

1. ``load_worker_registry()`` entries line up with importable
   entrypoints (Python module:attr resolves; RunPod workers are
   ``worker-vllm hub``).
2. ``ServerlessClient`` can drive the local FastAPI harness for
   ``text-gen`` / ``vlm`` / ``dsi`` through the workload→role
   mapping in ``vo_rag.workers.runpod_vllm.contracts`` (no real
   RunPod, no real GPU, no network socket).
3. The API's admin ``compute_placement`` map agrees with
   ``workload_location()`` for the five compute workloads
   (``text-gen`` / ``vlm`` / ``dsi`` / ``embedding`` /
   ``query-handling``) — a future drift would be a silent
   regression for operators.
4. The embedding worker's wire format round-trips through the
   in-process ``EmbeddingClient`` fake, then through
   ``downsample_colbert_vectors()`` /
   ``fallback_colbert_vectors()`` without schema drift.
5. The cloud cost exporter's per-workload metric labels match the
   workload names declared in the worker registry — a future PR
   that renames a workload (e.g. ``text-gen`` → ``chat``) without
   updating the exporter would otherwise publish a metric set the
   dashboards no longer parse.

These tests are NOT marked ``@pytest.mark.integration`` (the
live-infra marker) — they live in ``tests/integration/`` for
organisational reasons and run in the default no-live CI lane.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

from vo_rag.core.cloud.cost_tracker import CloudCostTracker, SpendReport

# --------------------------------------------------------------------------- #
# 1. Registry entries line up with importable entrypoints
# --------------------------------------------------------------------------- #


EXPECTED_WORKERS: tuple[str, ...] = (
    "embedding",
    "runpod_text_gen",
    "runpod_vlm",
    "runpod_dsi",
    "cloud_cost_exporter",
)


def _split_entrypoint(entrypoint: str) -> tuple[str, str]:
    if ":" not in entrypoint:
        raise ValueError(f"entrypoint must use the module:attr form; got {entrypoint!r}")
    module, attr = entrypoint.split(":", 1)
    return module.strip(), attr.strip()


@pytest.mark.parametrize("worker_name", list(EXPECTED_WORKERS))
def test_registry_entrypoint_resolves(worker_name: str) -> None:
    """Each entry's ``entrypoint`` either resolves to a real Python
    object (Python entrypoint) or is the literal ``worker-vllm hub``
    string (RunPod hub-deployed worker). Either way, the registry
    must be consistent with what is actually on disk."""
    from vo_rag.workers.registry import load_worker_registry

    reg = load_worker_registry()
    spec = reg[worker_name]
    if spec.entrypoint == "worker-vllm hub":
        # RunPod serverless workers are deployed from the official
        # worker-vllm hub, not from a Python module. The registry
        # carries the deploy template under ``deploy_artifacts``
        # — the contract is that the template file exists.
        for artifact in spec.deploy_artifacts:
            assert (Path(__file__).resolve().parents[2] / artifact).is_file(), (
                f"worker {worker_name!r}: deploy artifact {artifact!r} missing"
            )
        return
    module_name, attr_name = _split_entrypoint(spec.entrypoint)
    module = importlib.import_module(module_name)
    obj = getattr(module, attr_name, None)
    assert obj is not None, (
        f"worker {worker_name!r}: entrypoint {spec.entrypoint!r} — "
        f"{attr_name!r} missing on {module_name!r}"
    )


def test_registry_only_declares_known_compute_workloads() -> None:
    """The roster's workloads are exactly the compute workload allowlist
    plus ``misc`` (accounting). A future PR that adds a new workload
    class here would also need a router entry — we pin the contract
    that adding one without the other is impossible to merge."""
    from vo_rag.workers.registry import load_worker_registry

    reg = load_worker_registry()
    declared = {spec.workload for spec in reg.values()}
    assert declared == {"embedding", "text-gen", "vlm", "dsi", "misc"}, (
        f"unexpected workloads in registry: {declared!r}"
    )


# --------------------------------------------------------------------------- #
# 2. ServerlessClient drives the local harness via the role mapping
# --------------------------------------------------------------------------- #


def _make_settings_double() -> Any:
    """Build a stand-in for ``ServerlessSettings`` that satisfies the
    protocol ``vo_rag.workers.runpod_vllm.contracts`` reads."""
    base_url_template = "https://api.runpod.ai/v2/{endpoint_id}/openai/v1"

    def _base_url(role: str) -> str:
        endpoint_id = {
            "vlm": "vlm-ep",
            "text": "text-ep",
            "dsi": "dsi-ep",
            "exp": "exp-ep",
        }[role]
        return base_url_template.format(endpoint_id=endpoint_id)

    def _model_name(role: str) -> str:
        return {
            "vlm": "qwen3-vl-32b",
            "text": "qwen3-6-27b",
            "dsi": "qwen3-dsi-7b",
            "exp": "qwen3-exp-9b",
        }[role]

    return SimpleNamespace(
        api_key="contract-test-key",  # pragma: allowlist secret
        vlm_endpoint_id="vlm-ep",
        vlm_model_name="qwen3-vl-32b",
        text_endpoint_id="text-ep",
        text_model_name="qwen3-6-27b",
        dsi_endpoint_id="dsi-ep",
        dsi_model_name="qwen3-dsi-7b",
        request_timeout_s=600,
        base_url=_base_url,
        model_name=_model_name,
    )


@pytest.mark.parametrize(
    "workload",
    ["text-gen", "vlm", "dsi"],
)
@pytest.mark.asyncio
async def test_serverless_client_drives_local_harness_via_role_mapping(
    workload: str,
) -> None:
    """The RunPod serverless surface goes through ONE seam: a
    ``ServerlessClient`` (from ``vo_rag.core.cloud.serverless``)
    driven by a ``client_for_workload(workload)`` factory. This test
    confirms the factory wires the harness-bound URL and the right
    role-bound model name, and that the in-process FastAPI harness
    serves an OpenAI-shaped response the client can parse.

    We bind the ``ServerlessClient``'s private ``httpx.AsyncClient``
    to an ASGI transport pointed at a fresh harness, so the whole
    flow is in-process — no RunPod bill, no real GPU, no socket.
    """
    from vo_rag.workers.runpod_vllm.contracts import client_for_workload
    from vo_rag.workers.runpod_vllm.harness import create_harness_app

    app = create_harness_app(
        response_text=f"ok-{workload}",
        model=f"harness-{workload}",
    )
    settings = _make_settings_double()
    client = client_for_workload(workload, settings=settings)

    # Override the underlying httpx client with an ASGI transport
    # so the test never opens a socket. The harness's route is
    # ``/openai/v1/chat/completions`` — same path the real
    # ServerlessClient would hit at api.runpod.ai, so we just
    # rebind the base URL to the local harness. Headers, model
    # name, and timeout are preserved from the real client.
    client._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://harness/openai/v1",
        headers=client._client.headers,
        timeout=10,
    )
    try:
        resp = await client.chat_completion(
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=4,
        )
    finally:
        await client.aclose()

    assert resp.content == f"ok-{workload}", (
        f"workload={workload!r} returned wrong content via the harness"
    )
    assert resp.model == f"harness-{workload}"
    assert resp.input_tokens >= 1
    assert resp.output_tokens >= 1
    assert resp.finish_reason == "stop"


# --------------------------------------------------------------------------- #
# 3. Admin compute_placement matches workload_location()
# --------------------------------------------------------------------------- #


def test_admin_compute_placement_matches_workload_location() -> None:
    """The API's admin/status route returns a static
    ``compute_placement`` map for the five compute workloads. This
    map MUST agree with ``workload_location()`` (the static router) —
    otherwise the admin console lies to the operator.

    We exercise the helper directly (no live API) because
    ``_compute_placement`` is a pure function over the static map.
    """
    from vo_rag.api.routes.admin import _compute_placement
    from vo_rag.core.cloud.router import WorkloadLocation, workload_location

    placement = _compute_placement()
    assert set(placement) == {
        "text-gen",
        "vlm",
        "dsi",
        "embedding",
        "query-handling",
    }, f"admin compute_placement drifted: {set(placement)!r}"

    for workload, expected in (
        ("text-gen", WorkloadLocation.RUNPOD_SERVERLESS),
        ("vlm", WorkloadLocation.RUNPOD_SERVERLESS),
        ("dsi", WorkloadLocation.RUNPOD_SERVERLESS),
        ("embedding", WorkloadLocation.LOCAL),
        ("query-handling", WorkloadLocation.LOCAL),
    ):
        # Both the helper and the router must agree.
        assert workload_location(workload) is expected
        # The admin output must match the router (the operator
        # console is the public contract).
        assert placement[workload] == expected.value, (
            f"admin compute_placement[{workload!r}] = {placement[workload]!r}, "
            f"router says {expected.value!r}"
        )


def test_registry_declarations_agree_with_admin_compute_placement() -> None:
    """The worker registry's ``workload`` / ``location`` fields MUST
    agree with what the admin console publishes. A worker declared
    as ``workload=text-gen`` cannot sit in a registry row that the
    admin says runs locally — the dashboards would be wrong."""
    from vo_rag.api.routes.admin import _compute_placement
    from vo_rag.workers.registry import load_worker_registry

    placement = _compute_placement()
    reg = load_worker_registry()

    for name, spec in reg.items():
        if spec.workload == "misc":
            # Accounting-only — exempt from the admin allowlist.
            continue
        assert spec.workload in placement, (
            f"worker {name!r}: workload {spec.workload!r} not in admin "
            f"compute_placement allowlist {set(placement)!r}"
        )
        assert placement[spec.workload] == spec.location.value, (
            f"worker {name!r} declares location={spec.location.value!r} for "
            f"workload={spec.workload!r}, admin says {placement[spec.workload]!r}"
        )


# --------------------------------------------------------------------------- #
# 4. Embedding worker wire format survives EmbeddingClient.embed + point_vectors
# --------------------------------------------------------------------------- #


def test_embedding_wire_format_round_trip_to_point_vectors() -> None:
    """The BGE-M3 wire format must survive: (a) a fake server response
    being consumed by ``EmbeddingClient.embed()`` (via respx), and
    (b) the resulting dense/sparse/colbert fields flowing into
    ``sparse_payload_to_int_map`` / ``downsample_colbert_vectors`` /
    ``fallback_colbert_vectors`` without schema drift.

    The embedding worker is the *only* writer of VORAG vectors — if
    the wire format changes under it, every Qdrant upsert in
    vo-rag breaks. Pin the contract."""
    from vo_rag.core.embedding.point_vectors import (
        downsample_colbert_vectors,
        fallback_colbert_vectors,
        sparse_payload_to_int_map,
    )
    from vo_rag.core.embedding.schemas import EmbedResponse

    # The fake server is the exact JSON the embedding worker
    # serialises from ``BgeM3Worker.embed`` (see
    # ``src/vo_rag/workers/embedding/model.py``). The keys here
    # are the worker contract; if any of them change, this test
    # fails before a Qdrant upsert would.
    fake_response: dict[str, Any] = {
        "dense": [[0.1] * 1024, [0.2] * 1024],
        "sparse": [
            {"indices": [1, 5, 9], "values": [0.5, 0.3, 0.1]},
            {"indices": [2, 6], "values": [0.7, 0.2]},
        ],
        "colbert": [
            [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]],  # 3 token-vecs
            [[0.07, 0.08], [0.09, 0.10], [0.11, 0.12]],
        ],
        "model": "BAAI/bge-m3",
        "dim": 1024,
        "latency_ms": 12.5,
    }

    # 4a. The same JSON must validate against the canonical
    #     ``EmbedResponse`` Pydantic model. Pydantic coerces
    #     sparse ``{"indices": ..., "values": ...}`` to
    #     ``SparseVec`` instances.
    parsed = EmbedResponse.model_validate(fake_response)
    assert parsed.model == "BAAI/bge-m3"
    assert parsed.dim == 1024
    assert parsed.dense is not None and len(parsed.dense) == 2
    assert parsed.sparse is not None and len(parsed.sparse) == 2

    # 4b. The sparse payload survives ``sparse_payload_to_int_map``
    #     — the Qdrant upsert path consumes it as ``{token_id: w}``.
    assert sparse_payload_to_int_map(fake_response["sparse"][0]) == {
        1: 0.5,
        5: 0.3,
        9: 0.1,
    }

    # 4c. ColBERT vectors survive ``downsample_colbert_vectors``
    #     with the default 192 cap — three token-vectors stays as-is.
    assert downsample_colbert_vectors(fake_response["colbert"][0]) == [
        [0.01, 0.02],
        [0.03, 0.04],
        [0.05, 0.06],
    ]

    # 4d. The empty-ColBERT fallback is the dense vector.
    assert fallback_colbert_vectors(fake_response["dense"][0]) == [fake_response["dense"][0]]


@pytest.mark.asyncio
async def test_embedding_client_consumes_worker_shape() -> None:
    """The ``EmbeddingClient.embed()`` happy path must parse the
    same worker JSON the contract test above pins. We use respx to
    intercept the POST so no real network call is made — the
    contract is the request/response shape, not the HTTP wire."""
    import respx

    from vo_rag.core.embedding.client import EmbeddingClient

    fake_response = {
        "dense": [[0.0, 1.0], [1.0, 0.0]],
        "sparse": [
            {"indices": [1], "values": [0.5]},
            {"indices": [2], "values": [0.7]},
        ],
        "colbert": [[[0.0, 1.0]], [[1.0, 0.0]]],
        "model": "BAAI/bge-m3",
        "dim": 2,
        "latency_ms": 5.0,
    }

    with respx.mock(base_url="http://embed.local") as mock:
        route = mock.post("/embed").respond(200, json=fake_response)
        client = EmbeddingClient("http://embed.local", retry_backoff_s=0.0, retry_attempts=1)
        try:
            result = await client.embed(
                ["alpha", "beta"],
                return_dense=True,
                return_sparse=True,
                return_colbert=True,
            )
        finally:
            await client.aclose()
        assert route.called, "embedding client did not POST /embed"

    # The vector fields remain the worker's raw shape. The client also
    # normalizes optional runtime/token diagnostics per input, leaving
    # absent worker metadata explicitly unknown rather than fabricating it.
    for field in ("dense", "sparse", "colbert", "model", "dim", "latency_ms"):
        assert result[field] == fake_response[field]
    assert result["runtime_identity"] is None
    assert result["runtime_identity_coverage"] == "none"
    assert result["runtime_identity_per_text"] == [None, None]
    assert result["input_token_counts_per_text"] == [None, None]
    assert result["input_truncated_per_text"] == [None, None]


# --------------------------------------------------------------------------- #
# 5. Cost exporter metric workload labels match the registry workload names
# --------------------------------------------------------------------------- #


def test_cost_exporter_workload_labels_match_registry() -> None:
    """The cloud cost exporter publishes
    ``vo_rag_cloud_spend_month_usd{workload=...}`` samples built
    from ``SpendReport.by_workload``. The ``workload`` label values
    must come from the same set the worker registry uses — a
    mismatch would silently lose the ability to break down spend by
    workload in Prometheus.

    We do not stand up the FastAPI app (no DB, no Tailscale); we
    build a ``SpendReport`` whose ``by_workload`` mirrors what a
    production Postgres query would return, then drive
    ``_refresh(app)`` and parse /metrics."""
    from prometheus_client.parser import text_string_to_metric_families
    from services.cloud_cost_exporter.app import _refresh, create_app

    from vo_rag.core.cloud.cost_tracker import SpendReport
    from vo_rag.workers.registry import load_worker_registry

    # Workload names the registry actually declares (skip ``misc`` —
    # the exporter's own workload, not a real compute workload).
    reg = load_worker_registry()
    registry_workloads = {s.workload for s in reg.values() if s.workload != "misc"}
    assert {"text-gen", "vlm", "dsi"}.issubset(registry_workloads)

    # The exporter sanitises labels for Prometheus (dashes → underscores
    # via ``_sanitize_label``); build the sanitised set so the label
    # comparison below is correct.
    import re as _re

    sanitized_registry = {_re.sub(r"[^a-zA-Z0-9_]", "_", w) for w in registry_workloads}

    # A SpendReport whose by_workload uses the same workload names.
    report = SpendReport(
        month="2026-06",
        total_spend_usd=10.0,
        budget_usd=50.0,
        budget_remaining_usd=40.0,
        budget_pct_used=20.0,
        by_provider={"runpod": 10.0},
        by_workload={"text-gen": 5.0, "vlm": 3.0, "dsi": 2.0},
        by_gpu_type={"A100": 10.0},
        jobs_this_month=3,
    )

    tracker = _StubTracker(report)
    pool = _StubPool()

    from typing import cast

    from vo_rag.core.cloud.cost_tracker import CloudCostTracker

    app = create_app(
        tracker=cast(CloudCostTracker, tracker),
        pool=cast("Any", pool),
        tailscale=None,
        poll_interval_s=3600,
    )
    import contextlib

    from fastapi.testclient import TestClient

    with TestClient(app), contextlib.suppress(Exception):
        asyncio.run(_refresh(app))

    # Parse /metrics — find the spend gauge samples.
    with TestClient(app) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    families = list(text_string_to_metric_families(resp.text))
    spend_family = next((f for f in families if f.name == "vo_rag_cloud_spend_month_usd"), None)
    assert spend_family is not None, "spend gauge missing from /metrics"

    workload_labels = {
        sample.labels["workload"]
        for sample in spend_family.samples
        if sample.labels.get("workload")
    }
    # Every workload label the exporter published must be a workload
    # name the registry knows about (or ``""`` for the per-provider /
    # per-gpu aggregates, which are NOT workload breakdowns). The
    # label is sanitised for Prometheus — dashes become underscores —
    # so we compare against the sanitised registry set.
    for label in workload_labels:
        if label == "":
            continue  # by-provider / by-gpu-type aggregate label
        assert label in sanitized_registry, (
            f"cost exporter published workload label {label!r} not declared "
            f"in the worker registry (sanitised={sanitized_registry!r})"
        )

    # The text-gen / vlm / dsi samples from the fake report must
    # surface with their sanitised labels — the operator dashboard
    # relies on the by-workload breakdown being labelled exactly
    # ``text_gen`` / ``vlm`` / ``dsi`` (dashes sanitised).
    by_workload = {
        sample.labels["workload"]: sample.value
        for sample in spend_family.samples
        if sample.labels.get("workload")
    }
    assert by_workload.get("text_gen") == pytest.approx(5.0)
    assert by_workload.get("vlm") == pytest.approx(3.0)
    assert by_workload.get("dsi") == pytest.approx(2.0)


def test_cost_exporter_unknown_workload_is_silently_labeled() -> None:
    """If the DB ever returns a workload name the registry does not
    know about, the exporter must NOT crash. The metric is best-
    effort; the contract is that the export endpoint stays up so
    dashboards keep refreshing."""
    from fastapi.testclient import TestClient
    from prometheus_client.parser import text_string_to_metric_families
    from services.cloud_cost_exporter.app import _refresh, create_app

    from vo_rag.core.cloud.cost_tracker import SpendReport

    report = SpendReport(
        month="2026-06",
        total_spend_usd=1.0,
        budget_usd=50.0,
        budget_remaining_usd=49.0,
        budget_pct_used=2.0,
        by_provider={},
        by_workload={"new-future-workload": 1.0},  # unknown to the registry
        by_gpu_type={},
        jobs_this_month=1,
    )

    tracker = _StubTracker(report)
    pool = _StubPool()

    app = create_app(
        tracker=cast(CloudCostTracker, tracker),
        pool=cast("Any", pool),
        tailscale=None,
        poll_interval_s=3600,
    )

    with TestClient(app), contextlib.suppress(Exception):
        asyncio.run(_refresh(app))

    with TestClient(app) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    # The unknown workload surfaces verbatim (sanitised) in the
    # metric — Prometheus does not care about the registry, and the
    # operator would rather see "I have a spend row nobody can map"
    # than a silent 500.
    families = list(text_string_to_metric_families(resp.text))
    spend_family = next((f for f in families if f.name == "vo_rag_cloud_spend_month_usd"), None)
    assert spend_family is not None
    assert any(
        s.labels.get("workload") == "new_future_workload"  # sanitised
        for s in spend_family.samples
    )


# --------------------------------------------------------------------------- #
# 6. Pyproject has the right worker roster shape (defence in depth)
# --------------------------------------------------------------------------- #


class _StubTracker:
    """Minimal ``CloudCostTracker`` double for the cost-exporter tests.

    Defined at module level so mypy treats it as a module-scoped
    type (its methods structurally match
    ``vo_rag.core.cloud.cost_tracker.CloudCostTracker``). The
    constructor takes a pre-built ``SpendReport``; tests that want
    to drive ``_refresh`` against a custom workload breakdown just
    build the report and hand it in.
    """

    def __init__(self, r: SpendReport | None = None) -> None:
        self._r = r or SpendReport(
            month="2026-06",
            total_spend_usd=0.0,
            budget_usd=50.0,
            budget_remaining_usd=50.0,
            budget_pct_used=0.0,
            by_provider={},
            by_workload={},
            by_gpu_type={},
            jobs_this_month=0,
        )

    async def get_spend_report(self) -> SpendReport:
        return self._r


class _StubConn:
    async def __aenter__(self) -> _StubConn:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def fetchval(self, query: str, *args: Any) -> int:  # noqa: ARG002
        return 0


class _StubPool:
    def acquire(self) -> _StubConn:
        return _StubConn()

    async def close(self) -> None:
        return None


def test_pyproject_worker_roster_is_canonical() -> None:
    """``[tool.vo_rag].workers`` is the single source of truth for
    the deployable worker surface. A drift between this roster and
    the on-disk modules (or the registry's runtime view) is a
    regression — we pin the roster here at the integration layer."""
    import tomllib

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    raw = data.get("tool", {}).get("vo_rag", {}).get("workers", [])
    names = sorted(e["name"] for e in raw)
    assert names == sorted(EXPECTED_WORKERS), (
        f"pyproject.toml [tool.vo_rag.workers] roster drifted: {names!r}"
    )


# --------------------------------------------------------------------------- #
# 7. Python import smoke
# --------------------------------------------------------------------------- #


def test_python_imports_clean_for_compute_surfaces() -> None:
    """All four compute import surfaces must resolve in-process
    without raising. A failed import here is a regression
    independent of the registry — a stale stub or a missing
    ``__init__`` re-export would trip it.

    We deliberately do NOT call any of the imported objects — the
    contract is that *import* is hermetic. Call-site tests are
    above. We also do NOT purge ``sys.modules`` here: doing so
    would invalidate the imports other tests in this run already
    captured, breaking test isolation. The cold-import behaviour
    is verified separately by the ``bash scripts/ci.sh`` invocation
    which runs each pytest collection in a fresh process.
    """
    # Embedding server — must NOT pull torch / FlagEmbedding at import.
    import vo_rag.workers.embedding.server as embed_server  # noqa: F401

    assert embed_server.app is not None
    # The two heavy modules we explicitly do NOT want on the import
    # path. ``FlagEmbedding`` is the BGE-M3 wheel; ``torch`` is the
    # CUDA stack. Both are heavy; both are imported lazily by
    # ``BgeM3Worker.__init__`` so the worker module itself is
    # importable in CPU-only test runs.
    assert "FlagEmbedding" not in sys.modules
    assert "torch" not in sys.modules

    # RunPod harness — must NOT open a socket (factory only
    # constructs the FastAPI app; uvicorn binds in ``main()``).
    from vo_rag.workers.runpod_vllm.harness import create_harness_app  # noqa: F401

    app = create_harness_app()
    assert app is not None

    # Cost exporter — must NOT open a DB connection (the factory
    # only constructs; the lifespan does the connect).
    from services.cloud_cost_exporter.app import create_app  # noqa: F401

    exporter = create_app()
    assert exporter is not None

    # Top-level exports.
    from vo_rag.workers import WorkerSpec, load_worker_registry  # noqa: F401

    reg = load_worker_registry()
    assert set(reg) == set(EXPECTED_WORKERS)

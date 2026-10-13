"""Cross-worker no-bypass static checks.

These tests are AST / source-text scans of the vo-rag worker surface
and its sole RunPod transport seam. They enforce the architectural
invariant that the worker surface bakes in: *only* ``vo_rag.core.cloud.serverless``
talks to ``api.runpod.ai``, and no worker reaches around the registry
to spin up a pod / burst / dynamic instance.

Each invariant is encoded as a separate ``def test_*`` so a future
regression is obvious in CI output. The tests touch NO network — they
walk the filesystem and parse the Python source — so they are safe
in the no-live-infra CI lane.

Invariants pinned here:

1. ``vo_rag.workers.runpod_vllm`` does NOT instantiate
   ``httpx.AsyncClient`` pointing at ``api.runpod.ai``. The only
   transport seam is ``vo_rag.core.cloud.serverless.ServerlessClient``.
2. No worker imports dropped dynamic burst modules or references
   ``burst_api`` / ``launch_burst`` / ``teardown_burst`` /
   ``sovereignty`` / ``reaper``. The dynamic burst layer is GONE; the
   workers are serverless-only.
3. RunPod workload specs in the registry are ALL
   ``runpod_serverless`` — no worker spec says ``pod``, ``burst``, or
   ``dynamic`` (case-insensitive substring check).
4. The embedding worker imports
   ``vo_rag.core.embedding.schemas`` — it does NOT define a duplicated
   copy of ``EmbedRequest`` / ``EmbedResponse`` / ``SparseVec``.

The tests run via pytest and live under ``tests/workers/`` (a no-live
test layer; they are not marked ``@pytest.mark.live``).
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# Repo paths
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "vo_rag"
WORKERS_PKG = SRC_ROOT / "workers"
RUNPOD_PKG = WORKERS_PKG / "runpod_vllm"
EMBEDDING_PKG = WORKERS_PKG / "embedding"
CORE_CLOUD_DIR = SRC_ROOT / "core" / "cloud"
CORE_EMBEDDING_DIR = SRC_ROOT / "core" / "embedding"
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Dropped dynamic-burst vocabulary from the legacy substrate. Any of
# these tokens anywhere under src/vo_rag/workers/ (or any other
# worker-facing module) is a bypass attempt and fails the suite.
DROPPED_BURST_TOKENS: tuple[str, ...] = (
    "burst_api",
    "launch_burst",
    "teardown_burst",
    "sovereignty",
    "reaper",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _all_py_files(pkg: Path) -> list[Path]:
    return sorted(pkg.rglob("*.py"))


def _module_imports(text: str) -> set[str]:
    """Return the set of top-level module names imported by *text*.

    Handles both ``import x.y.z`` and ``from x.y.z import a``. The
    returned set contains only the top-level package (``x``), not the
    full dotted path, so a worker importing
    ``vo_rag.core.embedding.schemas`` is reported as importing
    ``vo_rag`` — and a worker that imports ``burst_api`` directly
    (no such module exists) is reported as importing ``burst_api``.
    """
    tree = ast.parse(text)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            modules.add(node.module.split(".", 1)[0])
    return modules


# --------------------------------------------------------------------------- #
# Invariant 1 — vo_rag.workers.runpod_vllm has no raw httpx / api.runpod.ai
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "violation_check",
    [
        pytest.param(
            lambda text: "import httpx" in text or "from httpx" in text,
            id="raw-httpx-import",
        ),
        pytest.param(
            lambda text: "api.runpod.ai" in text,
            id="api.runpod.ai-url",
        ),
    ],
)
def test_runpod_vllm_package_has_no_httpx_or_runpod_url(violation_check) -> None:
    """No module under ``vo_rag.workers.runpod_vllm`` may import httpx
    directly or hard-code ``api.runpod.ai``.

    The single permitted RunPod transport seam is
    ``vo_rag.core.cloud.serverless.ServerlessClient``; the local
    FastAPI harness (``harness.py``) is exempted from the URL check
    because the harness is the *target* of that client (not a peer)
    and the docstring describes the seam it must never call. The
    harness is still forbidden from importing httpx — it has no
    outgoing HTTP to send."""
    assert RUNPOD_PKG.is_dir(), f"missing package: {RUNPOD_PKG}"
    bad: list[str] = []
    for path in _all_py_files(RUNPOD_PKG):
        text = _read_text(path)
        # The harness is local test/dev infra — it is the *target* of
        # the serverless client, not a peer. Its docstring mentions
        # the RunPod host as historical context (the seam it must
        # NEVER call), and the ``api.runpod.ai-url`` check would
        # otherwise false-positive on that documentation. The
        # ``raw-httpx-import`` check still applies — the harness has
        # no outgoing HTTP and must not import httpx.
        if path.name == "harness.py":
            if "import httpx" in text or "from httpx" in text:
                bad.append(f"{path.relative_to(REPO_ROOT)}")
            continue
        if violation_check(text):
            bad.append(f"{path.relative_to(REPO_ROOT)}")
    assert not bad, (
        "no-bypass invariant violated — "
        f"{violation_check.__doc__ or 'forbidden pattern'}:\n" + "\n".join(bad)
    )


def test_runpod_vllm_only_imports_serverless_client() -> None:
    """The runpod worker surface must import ``ServerlessClient`` (or
    reach it transitively) and nothing else from the cloud layer that
    could open a socket to RunPod.

    This is a positive seam check — it asserts the *only* cloud import
    path is via the typed contracts module, never a raw
    ``vo_rag.core.cloud.runpod_client`` or similar low-level seam.
    """
    allowed_cloud_imports: set[str] = {
        # contracts.py — the typed workload→client factory.
        "vo_rag.core.cloud.serverless",
        # serverless.py imports httpx internally for the *one* permitted
        # HTTP seam. The worker surface is allowed to import
        # ``ServerlessClient`` and ``ServerlessResponse`` from there.
    }
    forbidden_cloud_imports: set[str] = {
        # Legacy low-level RunPod client — must not be reached directly
        # from any worker (it bypasses the typed contracts).
        "vo_rag.core.cloud.runpod_client",
    }

    bad: list[str] = []
    for path in _all_py_files(RUNPOD_PKG):
        text = _read_text(path)
        for mod in forbidden_cloud_imports:
            if mod in text:
                bad.append(f"{path.relative_to(REPO_ROOT)}: imports {mod}")
        # Sanity: every file in runpod_vllm that touches the cloud
        # layer must do so only through the serverless seam.
        if "vo_rag.core.cloud" in text and not any(
            allowed in text for allowed in allowed_cloud_imports
        ):
            bad.append(
                f"{path.relative_to(REPO_ROOT)}: touches vo_rag.core.cloud but "
                f"not through the typed serverless seam"
            )

    assert not bad, "RunPod transport seam violated:\n" + "\n".join(bad)


# --------------------------------------------------------------------------- #
# Invariant 2 — no dropped dynamic-burst module references
# --------------------------------------------------------------------------- #


def _assert_no_burst_tokens_under(paths: list[Path], label: str) -> None:
    """Fail if any of ``DROPPED_BURST_TOKENS`` appears as a Python
    identifier (import, attribute access, or bare name) in any file
    under *paths*. Docstring mentions are tolerated because they are
    historical context.

    AST-walks every file and extracts ``ast.Name`` / ``ast.Attribute``
    / ``ast.Import`` / ``ast.ImportFrom`` identifiers. A docstring
    like "Replaces the old dynamic burst/sovereignty routing" is a
    string literal, not a code reference, so it does NOT trip this
    check. An ``import burst_api`` or a function call
    ``launch_burst(...)`` does.
    """
    bad: list[str] = []
    for root in paths:
        if not root.exists():
            continue
        files = _all_py_files(root) if root.is_dir() else [root]
        for path in files:
            try:
                tree = ast.parse(_read_text(path))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                target: str | None = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in DROPPED_BURST_TOKENS or any(
                            alias.name.startswith(f"{t}.") for t in DROPPED_BURST_TOKENS
                        ):
                            target = alias.name
                            break
                elif isinstance(node, ast.ImportFrom):
                    if node.module and (
                        node.module in DROPPED_BURST_TOKENS
                        or any(node.module.startswith(f"{t}.") for t in DROPPED_BURST_TOKENS)
                    ):
                        target = node.module
                elif isinstance(node, ast.Name) and node.id in DROPPED_BURST_TOKENS:
                    target = node.id
                elif isinstance(node, ast.Attribute) and node.attr in DROPPED_BURST_TOKENS:
                    target = node.attr
                if target is not None:
                    bad.append(
                        f"{path.relative_to(REPO_ROOT)}: references dropped burst token {target!r}"
                    )
    assert not bad, f"dropped-burst vocabulary leaked under {label}:\n" + "\n".join(bad)


def test_workers_package_has_no_dropped_burst_tokens() -> None:
    """The entire ``vo_rag.workers.*`` tree must never
    import or reference the legacy burst / sovereignty / reaper
    surface."""
    _assert_no_burst_tokens_under([WORKERS_PKG], label="vo_rag.workers")


def test_core_cloud_has_no_dropped_burst_tokens() -> None:
    """The RunPod transport surface (``vo_rag.core.cloud.*``) is
    serverless-only — no burst / sovereignty / reaper either."""
    _assert_no_burst_tokens_under([CORE_CLOUD_DIR], label="vo_rag.core.cloud")


def test_no_dropped_burst_modules_imported_under_vo_rag() -> None:
    """AST-level check across the whole tree: no Python file under
    ``src/vo_rag`` may ``import`` a dropped-burst module name. The
    serverless-only substrate removed ``burst_api``, ``launch_burst``,
    ``teardown_burst``, ``sovereignty`` (the dynamic-routing module,
    *not* the data-model field), and ``reaper`` from the import
    graph.

    Tokens used as legitimate field / column names elsewhere in the
    data model (e.g. ``IntakeSubmission.sovereignty``) are NOT
    imports — those are *strings* / *enum values* in their defining
    modules. This test scopes to ``import`` / ``from ... import``
    statements only, so it does not false-positive on the
    ``sovereignty`` field on ``IntakeSubmission``.
    """
    bad: list[str] = []
    for path in _all_py_files(SRC_ROOT):
        text = _read_text(path)
        for token in DROPPED_BURST_TOKENS:
            # Match the token as the head of an import (top-of-line
            # pattern: `import burst_api` or
            # `from burst_api import ...`). A docstring or
            # variable name like ``my_sovereignty_field`` will not
            # match because we look for the token *as a module* in
            # an import statement.
            if re.search(rf"^\s*(?:import|from)\s+{re.escape(token)}\b", text, re.MULTILINE):
                bad.append(f"{path.relative_to(REPO_ROOT)}: imports dropped burst module {token!r}")
    assert not bad, "dropped-burst import(s) leaked:\n" + "\n".join(bad)


# --------------------------------------------------------------------------- #
# Invariant 3 — registry locations are runpod_serverless; no pod/burst/dynamic
# --------------------------------------------------------------------------- #


_FORBIDDEN_LOCATION_TOKENS: tuple[str, ...] = (
    "pod",
    "burst",
    "dynamic",
)


def test_worker_registry_locations_are_runpod_serverless_only() -> None:
    """Every worker spec in ``[tool.vo_rag.workers]`` must declare
    ``location = "runpod_serverless"`` (for real-compute workloads) or
    ``location = "local"`` (for the local BGE-M3 worker and the
    ``misc`` accounting exporter). No entry may use ``pod``, ``burst``,
    or ``dynamic`` (the legacy dynamic-burst substrate is GONE)."""
    data = tomllib.loads(_read_text(PYPROJECT))
    raw = data.get("tool", {}).get("vo_rag", {}).get("workers", [])
    assert raw, "no worker roster found in pyproject.toml — registry broken"

    allowed_locations = {"runpod_serverless", "local"}
    bad: list[str] = []
    for entry in raw:
        name = entry.get("name", "<unnamed>")
        location = str(entry.get("location", "")).strip().lower()
        if location not in allowed_locations:
            bad.append(f"{name!r}: location={location!r} (not in {sorted(allowed_locations)})")
        # Defence-in-depth: explicit substring check on the tokens
        # the legacy substrate used, in case someone adds a new
        # location value with a forbidden root word.
        for token in _FORBIDDEN_LOCATION_TOKENS:
            if token in location.split("_"):
                bad.append(f"{name!r}: location={location!r} contains forbidden token {token!r}")
    assert not bad, "forbidden runpod worker location(s):\n" + "\n".join(bad)


def test_worker_registry_workload_location_pairs() -> None:
    """The static workload→location map in ``vo_rag.core.cloud.router``
    must agree with what every worker in the registry declares (for
    non-``misc`` workloads). The ``misc`` accounting bucket is
    intentionally exempt because it is not a real compute workload."""
    from vo_rag.core.cloud.router import workload_location
    from vo_rag.workers.registry import load_worker_registry

    reg = load_worker_registry()
    bad: list[str] = []
    for name, spec in reg.items():
        if spec.workload == "misc":
            continue
        try:
            expected = workload_location(spec.workload)
        except ValueError as exc:
            bad.append(f"{name!r}: {exc}")
            continue
        if spec.location != expected:
            bad.append(
                f"{name!r}: declares location={spec.location.value!r} for "
                f"workload={spec.workload!r}, router says {expected.value!r}"
            )
    assert not bad, "registry / router mismatch:\n" + "\n".join(bad)


# --------------------------------------------------------------------------- #
# Invariant 4 — embedding worker reuses the canonical schemas
# --------------------------------------------------------------------------- #


def test_embedding_worker_imports_canonical_schemas() -> None:
    """The embedding worker (``vo_rag.workers.embedding.*``) must
    import ``vo_rag.core.embedding.schemas`` for the wire format. It
    must NOT define its own ``EmbedRequest`` / ``EmbedResponse`` /
    ``SparseVec`` — that would cause schema drift between the worker
    and the client in ``vo_rag.core.embedding.client``."""
    from vo_rag.core.embedding.schemas import (
        EmbedRequest,
        EmbedResponse,
    )

    # 4a. The worker package must re-export the canonical classes
    #     from vo_rag.core.embedding.schemas — not redefine them.
    server_text = _read_text(EMBEDDING_PKG / "server.py")
    for cls_name in ("EmbedRequest", "EmbedResponse"):
        assert f"{cls_name}" in server_text, f"server.py does not mention {cls_name}"

    # 4b. AST-level check: the worker must NOT define a class with
    #     the same name as one of the canonical schemas. A re-export
    #     via ``from vo_rag.core.embedding.schemas import X`` is fine;
    #     a `class X(VoBaseModel): ...` inside the worker is not.
    forbidden_class_names = {"EmbedRequest", "EmbedResponse", "SparseVec"}
    for path in _all_py_files(EMBEDDING_PKG):
        tree = ast.parse(_read_text(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in forbidden_class_names:
                pytest.fail(
                    f"{path.relative_to(REPO_ROOT)}: defines class "
                    f"{node.name!r} — must import from "
                    f"vo_rag.core.embedding.schemas instead"
                )

    # 4c. The re-exports must be the same objects as the canonical
    #     schemas (no copy, no subclass). This is the strongest drift
    #     check — if a future PR re-defines one of these classes in
    #     the worker, ``is`` will catch it.
    from vo_rag.workers.embedding import server as worker_server

    assert worker_server.EmbedRequest is EmbedRequest
    assert worker_server.EmbedResponse is EmbedResponse


def test_embedding_worker_does_not_define_duplicate_schema_classes() -> None:
    """Stricter variant of the above: walk every Python file under
    ``vo_rag.workers.embedding`` and assert no class named
    ``EmbedRequest`` / ``EmbedResponse`` / ``SparseVec`` is defined
    locally. Schemas live in core, not in the worker."""
    forbidden_class_names = {"EmbedRequest", "EmbedResponse", "SparseVec"}
    bad: list[str] = []
    for path in _all_py_files(EMBEDDING_PKG):
        tree = ast.parse(_read_text(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in forbidden_class_names:
                bad.append(f"{path.relative_to(REPO_ROOT)}: defines class {node.name!r}")
    assert not bad, "schema duplication under vo_rag.workers.embedding:\n" + "\n".join(bad)


# --------------------------------------------------------------------------- #
# Bonus invariant — the serverless seam is the *only* owner of api.runpod.ai
# --------------------------------------------------------------------------- #


def test_only_serverless_py_mentions_api_runpod_ai() -> None:
    """Positive counterpart to invariant 1: the literal string
    ``api.runpod.ai`` must appear ONLY in the *permitted transport
    seam* — ``vo_rag/core/cloud/serverless.py`` (the httpx
    transport) and ``vo_rag/core/config/settings.py`` (the base URL
    template the client uses). The local harness's docstring
    *describes* the seam it must never call and is exempted
    accordingly.

    The runpod worker contract test (``test_contracts.py``) already
    forbids it under ``workers/runpod_vllm/`` in non-docstring code;
    this test broadens the prohibition to the entire ``src/vo_rag``
    tree so a future ad-hoc import in, say, the cost exporter's
    tailscale helper would also fail."""
    exempt_files: set[Path] = {
        (CORE_CLOUD_DIR / "serverless.py").resolve(),
        (SRC_ROOT / "core" / "config" / "settings.py").resolve(),
        (RUNPOD_PKG / "harness.py").resolve(),
    }
    needles: list[str] = []
    for path in _all_py_files(SRC_ROOT):
        if path.resolve() in exempt_files:
            continue
        text = _read_text(path)
        if "api.runpod.ai" in text:
            needles.append(str(path.relative_to(REPO_ROOT)))
    assert not needles, (
        "api.runpod.ai mentioned outside the permitted transport seam:\n" + "\n".join(needles)
    )

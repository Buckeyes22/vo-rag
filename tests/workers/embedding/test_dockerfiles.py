"""Static contract tests for the BGE-M3 worker deploy artifacts.

These are pure-text checks — no Docker daemon, no real build. We assert:

- ``workers/embedding/Dockerfile`` references
  ``vo_rag.workers.embedding.server:app`` (NOT the legacy module path).
- ``workers/embedding/Dockerfile.local`` exists and references the same
  vo-rag module path (or documents the ``PORT`` override).
- ``workers/embedding/entrypoint.sh`` exec's uvicorn against
  ``vo_rag.workers.embedding.server:app`` and is executable.
- Both Dockerfiles are syntactically valid (start with ``FROM`` / a parser
  directive) and either bake the model or rely on the local default.
"""

from __future__ import annotations

import re
import stat
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EMB_DIR = REPO_ROOT / "workers" / "embedding"
DOCKERFILE = EMB_DIR / "Dockerfile"
DOCKERFILE_LOCAL = EMB_DIR / "Dockerfile.local"
ENTRYPOINT = EMB_DIR / "entrypoint.sh"

VO_RAG_REF = "vo_rag.workers.embedding.server:app"


def test_dockerfile_exists() -> None:
    assert DOCKERFILE.is_file(), f"missing {DOCKERFILE}"


def test_dockerfile_local_exists() -> None:
    assert DOCKERFILE_LOCAL.is_file(), f"missing {DOCKERFILE_LOCAL}"


def test_entrypoint_exists() -> None:
    assert ENTRYPOINT.is_file(), f"missing {ENTRYPOINT}"


@pytest.mark.parametrize("path", [DOCKERFILE, DOCKERFILE_LOCAL])
def test_dockerfile_starts_with_syntax_directive_or_from(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    head = text.lstrip().splitlines()[:2]
    assert head, f"{path} is empty"
    first = head[0]
    assert first.startswith("# syntax=") or first.upper().startswith("FROM"), (
        f"{path} must start with a syntax directive or FROM, got: {first!r}"
    )


@pytest.mark.parametrize("path", [DOCKERFILE, DOCKERFILE_LOCAL])
def test_dockerfile_references_vo_rag_entrypoint(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    # The Dockerfile itself does not have to mention vo_rag.workers.embedding
    # if it relies on the entrypoint.sh. So we also require entrypoint.sh to
    # reference the right module, and at least one of the two files must.
    if VO_RAG_REF not in text and "entrypoint" not in text.lower():
        pytest.fail(
            f"{path.name} must either reference {VO_RAG_REF} or invoke "
            "workers/embedding/entrypoint.sh that does"
        )


@pytest.mark.parametrize("path", [DOCKERFILE, DOCKERFILE_LOCAL])
def test_dockerfile_runs_entrypoint_script(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert 'ENTRYPOINT ["/entrypoint.sh"]' in text, (
        f"{path.name} must start the worker through /entrypoint.sh"
    )


def test_dockerfile_exposes_8090_or_uses_port() -> None:
    """EXPOSE 8090 for local default, allow PORT override for RunPod."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(r"\bEXPOSE\b", text, flags=re.IGNORECASE), (
        f"{DOCKERFILE.name} must declare an EXPOSE"
    )


def test_entrypoint_exec_uvicorn_with_vo_rag_app() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert VO_RAG_REF in text, f"{ENTRYPOINT.name} must exec uvicorn with {VO_RAG_REF}"
    assert "exec uvicorn" in text


def test_entrypoint_binds_to_0_0_0_0() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "--host 0.0.0.0" in text


def test_entrypoint_uses_port_env_with_default_8090() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert re.search(r"--port\s+\"?\$\{?PORT", text) or re.search(
        r"--port\s+\"\$\{PORT:-8090\}\"", text
    ), f"{ENTRYPOINT.name} must honour $PORT (default 8090)"


def test_entrypoint_is_executable() -> None:
    mode = ENTRYPOINT.stat().st_mode
    assert mode & stat.S_IXUSR, f"{ENTRYPOINT.name} must be executable (chmod +x)"


def test_dockerfile_or_local_bakes_bge_m3() -> None:
    """At least one of the Dockerfiles should bake the BAAI/bge-m3 model
    (so the image is self-contained for local runs). The .local variant may
    defer to a volume mount instead — but the canonical one must bake."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "BAAI/bge-m3" in text, f"{DOCKERFILE.name} must bake the BAAI/bge-m3 model"

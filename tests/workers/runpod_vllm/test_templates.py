"""Tests for the RunPod endpoint JSON templates under workers/runpod_vllm/templates/.

Each template declares workload, serverless role, model name env
placeholder, expected base URL shape, min/max worker placeholders, a health
check command, and an explicit "use worker-vllm, not plain vllm-openai" note.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

TEMPLATES_DIR = Path(__file__).resolve().parents[3] / "workers" / "runpod_vllm" / "templates"


def _load(name: str) -> Any:
    path = TEMPLATES_DIR / name
    assert path.is_file(), f"missing template: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Per-workload template presence + minimum shape
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "filename,workload,role",
    [
        ("text-gen.json", "text-gen", "text"),
        ("vlm.json", "vlm", "vlm"),
        ("dsi.json", "dsi", "dsi"),
    ],
)
def test_template_minimum_shape(filename: str, workload: str, role: str) -> None:
    body = _load(filename)
    assert body["workload"] == workload, f"{filename}: workload mismatch"
    assert body["serverless_role"] == role, f"{filename}: role mismatch"
    # Model name env placeholder — must use ${...} so deployment can inject.
    model_env = body["model_name_env"]
    assert isinstance(model_env, str) and model_env.startswith("${")
    assert "}" in model_env, f"{filename}: model_name_env must be a ${{VAR:default}}"
    # Expected base URL shape — the runpod URL pattern is parameterized.
    base_url = body["base_url_shape"]
    assert "api.runpod.ai/v2/{endpoint_id}/openai/v1" in base_url, (
        f"{filename}: base_url_shape must parameterize endpoint_id"
    )
    # Min/max workers placeholders exist and are sensible.
    scaling = body["scaling"]
    assert isinstance(scaling["min_workers"], int) and scaling["min_workers"] >= 0
    assert (
        isinstance(scaling["max_workers"], int) and scaling["max_workers"] >= scaling["min_workers"]
    )
    # Health check command — must invoke vo-rag or a one-liner.
    health = body["health_check"]
    assert isinstance(health, str) and health.strip()
    assert (
        "vo-rag" in health or "urllib.request" in health or "httpx" in health or "curl" in health
    ), f"{filename}: health_check must use vo-rag, curl, httpx, or urllib.request"


# --------------------------------------------------------------------------- #
# Every template must use the official worker-vllm hub, not plain vllm/vllm-openai
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "filename",
    ["text-gen.json", "vlm.json", "dsi.json"],
)
def test_template_uses_official_worker_vllm_hub(filename: str) -> None:
    body = _load(filename)
    image = body.get("image", "")
    # Allowed: worker-vllm hub, runpod/worker-vllm, or any reference that
    # includes the official worker-vllm (not the bare vllm/vllm-openai image).
    assert "worker-vllm" in image, f"{filename}: must use official worker-vllm hub, got {image!r}"
    # Disallow plain "vllm/vllm-openai" (the older, less-featureful image).
    assert "vllm-openai" not in image or "worker-vllm" in image, (
        f"{filename}: must not declare plain vllm-openai image, got {image!r}"
    )


# --------------------------------------------------------------------------- #
# Notes: deploy-phase items (runpodctl, endpoint IDs, FlashBoot, Grafana) are
# out of scope for the committed templates.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "filename",
    ["text-gen.json", "vlm.json", "dsi.json"],
)
def test_template_documents_deploy_phase_as_out_of_scope(filename: str) -> None:
    body = _load(filename)
    notes = body.get("deploy_phase_notes", "")
    assert isinstance(notes, str) and notes.strip()
    # The note must mention at least one of the deploy-phase items so a
    # reader knows where to look.
    assert any(
        kw in notes.lower()
        for kw in ("runpodctl", "flashboot", "grafana", "endpoint id", "endpoint_id")
    ), f"{filename}: deploy_phase_notes must point at deploy-phase tooling"


# --------------------------------------------------------------------------- #
# README presence
# --------------------------------------------------------------------------- #


def test_runpod_vllm_readme_exists() -> None:
    readme = Path(__file__).resolve().parents[3] / "workers" / "runpod_vllm" / "README.md"
    assert readme.is_file(), f"missing README: {readme}"
    text = readme.read_text(encoding="utf-8")
    # README must explain what the templates cover and what stays in the
    # deploy phase.
    assert "worker-vllm" in text, "README must mention official worker-vllm hub"
    assert "runpodctl" in text or "FlashBoot" in text, (
        "README must call out deploy-phase items (runpodctl / FlashBoot / Grafana)"
    )
    for kw in ("text-gen", "vlm", "dsi"):
        assert kw in text, f"README must mention workload {kw}"

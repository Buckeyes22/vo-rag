"""Single source of truth for the pipeline roster.

The roster is declared once in pyproject.toml under [tool.vo_rag].pipelines.
Everything else (CLI, docs homes, eval gate, future dispatch) derives from here.
"""

from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel


class PipelineStatus(StrEnum):
    PLANNED = "planned"
    NEXT = "next"
    ACTIVE = "active"


class PipelineSpec(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    name: str
    status: PipelineStatus
    summary: str


def _find_pyproject() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("pyproject.toml not found above core/config/pipelines.py")


def load_pipeline_registry(pyproject_path: Path | None = None) -> dict[str, PipelineSpec]:
    path = pyproject_path or _find_pyproject()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    raw = data.get("tool", {}).get("vo_rag", {}).get("pipelines", [])
    registry: dict[str, PipelineSpec] = {}
    for entry in raw:
        spec = PipelineSpec(**entry)
        if spec.name in registry:
            raise ValueError(f"duplicate pipeline in roster: {spec.name}")
        registry[spec.name] = spec
    return registry

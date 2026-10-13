"""Collect honest quality warnings without changing ingestion availability."""

from __future__ import annotations

import json
from typing import Any

__all__ = ["collect_quality_warnings", "quality_warning"]


def quality_warning(stage: str, code: str, details: str) -> dict[str, Any]:
    """Describe a quality problem without claiming a successful measurement."""
    return {"stage": stage, "code": code, "severity": "warning", "details": details}


def collect_quality_warnings(ctx: Any) -> list[dict[str, Any]]:
    """Collect unique parse, passage, and late-stage warnings for one document."""
    sources = [ctx.envelope.metadata, ctx.parse_meta]
    if ctx.parsed is not None:
        sources.append(ctx.parsed.metadata)
    sources.extend(chunk.metadata for chunk in ctx.chunks)
    sources.append({"quality_warnings": ctx.quality_warnings})
    warnings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for metadata in sources:
        entries = metadata.get("quality_warnings", [])
        if not isinstance(entries, list):
            continue
        for warning in entries:
            if not isinstance(warning, dict):
                continue
            key = json.dumps(warning, sort_keys=True, default=str)
            if key not in seen:
                warnings.append(dict(warning))
                seen.add(key)
    return warnings

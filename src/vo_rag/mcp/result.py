"""Build native MCP tool results from typed domain payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel


def to_call_tool_result(
    payload: BaseModel | Mapping[str, Any],
    text: str,
    *,
    error: bool = False,
) -> CallToolResult:
    """Return a native MCP result with one domain payload.

    Args:
        payload: Typed model or already serialized domain payload.
        text: Human-readable tool summary.
        error: Whether the outcome is a tool error. Negative domain outcomes
            that are valid data should leave this false.
    """
    if isinstance(payload, BaseModel):
        structured = payload.model_dump(mode="json", by_alias=True)
    else:
        structured = dict(payload)
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent=structured,
        isError=error,
    )


__all__ = ["to_call_tool_result"]

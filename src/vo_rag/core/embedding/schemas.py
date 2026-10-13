"""HTTP wire-format models for the BGE-M3 embedding server.

The 1024-dim / 256 texts / 8192 token limits are part of the contract with
the worker — they must match exactly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    model_config = {"extra": "forbid"}

    texts: list[str] = Field(min_length=1, max_length=256)
    return_dense: bool = True
    return_sparse: bool = True
    return_colbert: bool = False
    max_length: int = Field(default=8192, ge=1, le=8192)


class SparseVec(BaseModel):
    indices: list[int]
    values: list[float]


class EmbedResponse(BaseModel):
    dense: list[list[float]] | None = None
    sparse: list[SparseVec] | None = None
    colbert: list[list[list[float]]] | None = None
    model: str
    dim: int = 1024
    latency_ms: float
    runtime_identity: dict[str, str | None] | None = None
    input_token_counts: list[int] | None = None
    input_truncated: list[bool] | None = None

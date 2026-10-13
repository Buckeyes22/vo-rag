"""Pydantic mirror of the downstream consumer `vceRowSchema v2` contract.

This is the **test-side** mirror of the contract downstream consumer's TypeScript
codebase parses with `zod`. Keeping a Python copy lets this repo's CI
gate catch contract drift without depending on downstream consumer's repo (which
lives outside this worktree).

When the contract version bumps in :data:`vo_rag.pipelines.vce.types.VCE_CONTRACT_VERSION`,
this mirror must bump too — and the test that asserts round-trip stability
will fail if the two diverge.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VceRowSchemaContractVersion: Literal["vce-episode-export@2"] = "vce-episode-export@2"


class VceRowAnchorSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_ts_seconds: float = Field(ge=0.0)
    vlm_description: str
    region: str | None = None
    frame_path: str | None = None


class VceRowSchema(BaseModel):
    """Mirror of downstream consumer's `vceRowSchema` at vce-episode-export@2."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=20, max_length=32)
    video_id: str
    start_ts: float = Field(ge=0.0)
    end_ts: float = Field(gt=0.0)
    observation: str
    evidence: Literal["inferred_from_audio", "explicit_numeric", "explicit_qualitative"]
    transcript_excerpt: str = ""
    transcript_source: Literal["whisperx", "platform_captions", "human"] | None = None
    anchors: list[VceRowAnchorSchema] = Field(default_factory=list)


__all__ = ["VceRowSchema", "VceRowAnchorSchema", "VceRowSchemaContractVersion"]

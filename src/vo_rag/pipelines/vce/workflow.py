"""VCE pipeline workflow — concrete :class:`PipelineBase` subclass.

downstream consumer does not consume envelopes; the VCE export is a batch CLI that
runs through every video and writes a JSON file. The pipeline shape here
is therefore *one* :class:`VcePipelineOutput` per video, written through
``index_output`` so the per-video coaching_record row lands in
``knowledge_artifacts`` (migration 026 enables this).

The class deliberately does NOT embed the stage DAG — the runner
(:class:`vo_rag.pipelines.vce.runner.run_vce_batch`) owns the DAG. The
pipeline here is the per-video terminal shape: the runner builds a
:class:`VceContext`, runs all stages, then calls
:meth:`VcePipeline.run_for_video` to get the
:class:`VcePipelineOutput` that ``PipelineBase.index_output`` consumes.
"""

from __future__ import annotations

import os
import uuid

import structlog

from vo_rag.pipelines.base import PipelineBase
from vo_rag.pipelines.vce.types import VceContext, VcePipelineOutput

log = structlog.get_logger(__name__)


# Target collection recorded on the knowledge_artifacts row. VCE
# does NOT actually upsert to Qdrant — downstream consumer does not consume Qdrant
# — but the column is non-null in the contract and ``PipelineBase.index_output``
# refuses to proceed without a value.
VCE_TARGET_COLLECTION = os.environ.get(
    "VORAG_VCE_TARGET_COLLECTION",
    "vorag_vce_v1",
)


class VcePipeline(PipelineBase):
    """Per-video pipeline that turns a finished :class:`VceContext` into a
    :class:`VcePipelineOutput` and writes it through ``index_output``.

    Args:
        target_collection: Qdrant collection to record on the artifact row
            (no upsert happens — VCE does not consume Qdrant).
    """

    target_collection: str = VCE_TARGET_COLLECTION

    def __init__(self, target_collection: str | None = None) -> None:
        super().__init__()
        if target_collection is not None:
            self.target_collection = target_collection

    async def run_for_video(self, ctx: VceContext) -> VcePipelineOutput:
        """Finalise one video's :class:`VceContext` into the output shape.

        The runner has already populated ``ctx.rows`` (the structured-extract
        output), ``ctx.superseded`` (rows whose interval changed), and
        ``ctx.stage_status`` (per-stage success/failure). This method
        composes the per-video terminal shape.
        """
        envelope_id = self._envelope_id_for(ctx)
        artifact_id = self._artifact_id_for(ctx)

        title = ctx.title or f"vce:{ctx.video_id}"
        export_path = f"vorag-vce-derived/exports/{ctx.run_id}/vce-episode-export.json"
        rows_json_path = f"vorag-vce-derived/exports/{ctx.run_id}/{ctx.video_id}.transcript.json"
        transcripts_dir = "vorag-vce-derived/transcripts"

        output = VcePipelineOutput(
            video_id=ctx.video_id,
            envelope_id=envelope_id,
            artifact_id=artifact_id,
            title=title,
            export_path=export_path,
            rows_json_path=rows_json_path,
            transcripts_dir=transcripts_dir,
            rows=list(ctx.rows),
            superseded=list(ctx.superseded),
            baseline_preserved_ids=list(ctx.baseline_preserved_ids),
            stage_status=dict(ctx.stage_status),
            cost_usd=ctx.cost_usd,
        )
        log.info(
            "vce_pipeline_run_for_video_done",
            run_id=ctx.run_id,
            video_id=ctx.video_id,
            row_count=len(output.rows),
            superseded=len(output.superseded),
            stage_states={
                stage: status.get("state") for stage, status in output.stage_status.items()
            },
        )
        return output

    async def index_for_video(self, ctx: VceContext) -> VcePipelineOutput:
        """Run :meth:`run_for_video` and write through ``PipelineBase.index_output``."""
        output = await self.run_for_video(ctx)
        await self.index_output(output.to_pipeline_output())
        return output

    def _envelope_id_for(self, ctx: VceContext) -> str:
        """The envelope id recorded on the per-video artifact row.

        VCE doesn't have envelopes in the book_rag sense; we mint a stable
        ULID per (run, video) so the artifact row is uniquely attributable.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"vce:{ctx.run_id}:{ctx.video_id}"))

    def _artifact_id_for(self, ctx: VceContext) -> str:
        """The artifact id — a deterministic 26-char hex derived from (run, video)."""
        from vo_rag.core.idempotency import deterministic_hex

        return deterministic_hex(
            f"vce-artifact:{ctx.run_id}:{ctx.video_id}",
            length=26,
        ).upper()


__all__ = ["VCE_TARGET_COLLECTION", "VcePipeline", "VcePipelineOutput"]

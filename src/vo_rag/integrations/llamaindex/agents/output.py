"""Fold LlamaIndex agent answers into vo-rag pipeline/artifact shapes."""

from __future__ import annotations

from collections.abc import Awaitable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from vo_rag.core.cloud.router import WorkloadLocation, workload_location
from vo_rag.core.idempotency import deterministic_hex
from vo_rag.core.models.artifact import (
    ArtifactType,
    KnowledgeArtifact,
    LicenseClass,
    ProvenanceEntry,
    TrustLevel,
)
from vo_rag.core.models.common import VoBaseModel
from vo_rag.core.models.envelope import Priority, Sovereignty, TargetPipeline
from vo_rag.pipelines.base import PipelineOutput


@runtime_checkable
class CostLaunchGate(Protocol):
    """Cloud-cost gate shape required before a serverless agent LLM call."""

    def try_launch(self, **kwargs: Any) -> Awaitable[bool] | bool: ...


class AgentOutputFold(VoBaseModel):
    """The folded output plus the in-memory artifact carrying trust."""

    pipeline_output: PipelineOutput
    artifact: KnowledgeArtifact
    llm_location: WorkloadLocation


def assert_agent_llm_substrate(
    *,
    workload: str = "text-gen",
    sovereignty: Sovereignty | str = Sovereignty.CLOUD_OK,
    cost_tracker: CostLaunchGate | None = None,
) -> WorkloadLocation:
    """Assert the agent LLM workload is routed through the vo-rag substrate.

    This helper does not launch compute or reserve budget; it refuses
    combinations that bypass the static router, sovereignty fence, or required
    serverless launch-gate dependency shape.
    """
    resolved_sovereignty = Sovereignty(sovereignty)
    location = workload_location(workload)

    if (
        resolved_sovereignty == Sovereignty.HOMELAB_ONLY
        and location == WorkloadLocation.RUNPOD_SERVERLESS
    ):
        raise PermissionError(
            "homelab_only work may not resolve an agent LLM workload to serverless"
        )
    if location == WorkloadLocation.RUNPOD_SERVERLESS:
        if cost_tracker is None:
            raise RuntimeError("serverless agent LLM workloads require CloudCostTracker.try_launch")
        if not isinstance(cost_tracker, CostLaunchGate):
            raise TypeError("cost_tracker must expose try_launch(...)")
    return location


def resolve_agent_output_trust_level(
    *,
    should_quarantine: bool | None = None,
    corpus_only: bool = False,
) -> TrustLevel:
    """Resolve artifact trust from the threaded quarantine signal."""
    if should_quarantine is False and corpus_only:
        return TrustLevel.TRUSTED
    return TrustLevel.QUARANTINED


def fold_agent_output(
    *,
    envelope_id: str,
    question: str,
    final_answer: object,
    evidence: Iterable[Mapping[str, Any]] = (),
    should_quarantine: bool | None = None,
    corpus_only: bool = False,
    artifact_id_basis: str | None = None,
    title: str | None = None,
    pipeline: str = TargetPipeline.RESEARCH_AGENT.value,
    artifact_type: str = ArtifactType.RESEARCH.value,
    tags: Sequence[str] = (),
    priority: str = Priority.MEDIUM.value,
    license_class: LicenseClass = LicenseClass.UNKNOWN,
    qdrant_collection: str | None = None,
    workload: str = "text-gen",
    sovereignty: Sovereignty | str = Sovereignty.CLOUD_OK,
    cost_tracker: CostLaunchGate | None = None,
) -> AgentOutputFold:
    """Build a ``PipelineOutput`` and trusted/quarantined ``KnowledgeArtifact``.

    ``PipelineOutput`` remains the work-step contract. The paired
    ``KnowledgeArtifact`` is built in memory here because the trust level lives
    on the artifact model, not on ``PipelineOutput``.
    """
    llm_location = assert_agent_llm_substrate(
        workload=workload,
        sovereignty=sovereignty,
        cost_tracker=cost_tracker,
    )
    evidence_items = [dict(item) for item in evidence]
    basis = artifact_id_basis or question
    provenance = _source_provenance_from_evidence(evidence_items)
    output = PipelineOutput(
        envelope_id=envelope_id,
        pipeline=pipeline,
        artifact_type=artifact_type,
        title=title or question[:120] or pipeline,
        content={
            "question": question,
            "answer": str(final_answer),
            "evidence": evidence_items,
            "artifact_id_basis": basis,
        },
        tags=list(tags),
        priority=priority,
        license_class=license_class,
        source_provenance=provenance,
    )
    artifact = _artifact_from_output(
        output,
        trust_level=resolve_agent_output_trust_level(
            should_quarantine=should_quarantine,
            corpus_only=corpus_only,
        ),
        qdrant_collection=qdrant_collection,
    )
    return AgentOutputFold(
        pipeline_output=output,
        artifact=artifact,
        llm_location=llm_location,
    )


def _artifact_from_output(
    output: PipelineOutput,
    *,
    trust_level: TrustLevel,
    qdrant_collection: str | None,
) -> KnowledgeArtifact:
    now = datetime.now(UTC)
    return KnowledgeArtifact(
        id=_artifact_id(output),
        envelope_id=output.envelope_id,
        pipeline=TargetPipeline(output.pipeline),
        artifact_type=ArtifactType(output.artifact_type),
        title=output.title,
        content=output.content,
        tags=output.tags,
        priority=Priority(output.priority),
        license_class=output.license_class,
        trust_level=trust_level,
        qdrant_collection=qdrant_collection,
        qdrant_point_ids=_point_ids_from_content(output.content),
        source_provenance=[
            ProvenanceEntry.model_validate(entry) for entry in output.source_provenance
        ],
        created_at=now,
        updated_at=now,
    )


def _artifact_id(output: PipelineOutput) -> str:
    artifact_basis = str(
        output.content.get("artifact_id_basis")
        or output.content.get("source_id")
        or output.envelope_id
    )
    return deterministic_hex(
        artifact_basis,
        output.pipeline,
        output.artifact_type,
        length=26,
    ).upper()


def _point_ids_from_content(content: Mapping[str, Any]) -> list[str]:
    point_ids = content.get("qdrant_point_ids")
    if not isinstance(point_ids, list):
        return []
    return [str(value) for value in point_ids]


def _source_provenance_from_evidence(
    evidence: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    provenance: list[dict[str, Any]] = []
    for item in evidence:
        source_id = item.get("source_id")
        source_type = item.get("source_type")
        if source_id is None or source_type is None:
            continue
        entry = ProvenanceEntry(
            source_id=str(source_id),
            source_title=_optional_str(item.get("source_title")),
            source_type=str(source_type),
            contribution=float(item.get("contribution", 0.0)),
            page_refs=_page_refs(item.get("page_refs")),
        )
        provenance.append(entry.model_dump(mode="json", exclude_none=True))
    return provenance


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _page_refs(value: object) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    refs: list[int] = []
    for raw_ref in value:
        if isinstance(raw_ref, int) and not isinstance(raw_ref, bool):
            refs.append(raw_ref)
        elif isinstance(raw_ref, str) and raw_ref.isdecimal():
            refs.append(int(raw_ref))
    return refs


__all__ = [
    "AgentOutputFold",
    "CostLaunchGate",
    "assert_agent_llm_substrate",
    "fold_agent_output",
    "resolve_agent_output_trust_level",
]

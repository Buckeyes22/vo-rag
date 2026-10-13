"""Trust and substrate contracts for folding agent output into artifacts."""

from __future__ import annotations

from typing import Any

import pytest

from vo_rag.core.cloud.router import WorkloadLocation
from vo_rag.core.models.artifact import ArtifactType, LicenseClass, TrustLevel
from vo_rag.core.models.envelope import Priority, Sovereignty, TargetPipeline
from vo_rag.integrations.llamaindex.agents import (
    assert_agent_llm_substrate,
    fold_agent_output,
)


class _FakeCostTracker:
    def __init__(self) -> None:
        self.launch_calls = 0

    async def try_launch(self, **kwargs: Any) -> bool:
        self.launch_calls += 1
        raise AssertionError("output folding must not reserve budget or launch compute")


def _fold(**overrides: Any) -> Any:
    base: dict[str, Any] = {
        "envelope_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "question": "Which sources support the agent answer?",
        "final_answer": "The agent answer cites the internal corpus.",
        "evidence": [
            {
                "source_id": "book-1",
                "source_title": "Internal Book",
                "source_type": "book",
                "contribution": 0.75,
                "page_refs": [12, 13],
            },
            {
                "source_id": "graph-7",
                "source_type": "graph_edge",
                "contribution": 0.25,
                "page_refs": [44],
            },
        ],
        "cost_tracker": _FakeCostTracker(),
    }
    base.update(overrides)
    return fold_agent_output(**base)


def test_should_quarantine_signal_produces_quarantined_artifact() -> None:
    folded = _fold(should_quarantine=True, corpus_only=True)

    assert folded.artifact.trust_level is not TrustLevel.TRUSTED
    assert folded.artifact.trust_level is TrustLevel.QUARANTINED
    assert folded.pipeline_output.content["artifact_id_basis"] == (
        "Which sources support the agent answer?"
    )
    assert folded.pipeline_output.source_provenance == [
        {
            "source_id": "book-1",
            "source_title": "Internal Book",
            "source_type": "book",
            "contribution": 0.75,
            "page_refs": [12, 13],
        },
        {
            "source_id": "graph-7",
            "source_type": "graph_edge",
            "contribution": 0.25,
            "page_refs": [44],
        },
    ]
    assert folded.artifact.source_provenance[0].source_id == "book-1"
    assert folded.artifact.source_provenance[0].page_refs == [12, 13]


def test_absent_quarantine_signal_defaults_to_quarantined_artifact() -> None:
    folded = _fold()

    assert folded.artifact.trust_level is not TrustLevel.TRUSTED
    assert folded.artifact.trust_level is TrustLevel.QUARANTINED


def test_explicit_clean_corpus_only_run_may_be_trusted() -> None:
    folded = _fold(should_quarantine=False, corpus_only=True)

    assert folded.artifact.trust_level is TrustLevel.TRUSTED
    assert folded.artifact.pipeline is TargetPipeline.RESEARCH_AGENT
    assert folded.artifact.artifact_type is ArtifactType.RESEARCH
    assert folded.artifact.priority is Priority.MEDIUM
    assert folded.artifact.license_class is LicenseClass.UNKNOWN


def test_same_artifact_id_basis_yields_stable_artifact_id() -> None:
    first = _fold(
        question="First rendering of the question",
        artifact_id_basis="stable-agent-question",
        final_answer="first answer text",
    )
    second = _fold(
        question="Second rendering of the question",
        artifact_id_basis="stable-agent-question",
        final_answer="second answer text",
    )

    assert first.pipeline_output.content["artifact_id_basis"] == "stable-agent-question"
    assert second.pipeline_output.content["artifact_id_basis"] == "stable-agent-question"
    assert first.artifact.id == second.artifact.id


def test_homelab_only_serverless_workload_is_refused_before_any_call() -> None:
    with pytest.raises(PermissionError, match="homelab_only"):
        assert_agent_llm_substrate(
            workload="text-gen",
            sovereignty=Sovereignty.HOMELAB_ONLY,
            cost_tracker=_FakeCostTracker(),
        )


def test_serverless_substrate_check_does_not_reserve_budget() -> None:
    cost_tracker = _FakeCostTracker()

    location = assert_agent_llm_substrate(
        workload="text-gen",
        sovereignty=Sovereignty.CLOUD_OK,
        cost_tracker=cost_tracker,
    )

    assert location is WorkloadLocation.RUNPOD_SERVERLESS
    assert cost_tracker.launch_calls == 0

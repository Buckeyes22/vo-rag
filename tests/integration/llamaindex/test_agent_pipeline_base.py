"""Contract tests for ``AgentPipelineBase``, the base class for agent-driven pipelines."""

from __future__ import annotations

import sys
from collections.abc import Awaitable
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.llms.mock import MockLLM
from llama_index.core.memory import Memory

from vo_rag.core.llm.client import LLMClient
from vo_rag.core.models.artifact import ArtifactType, TrustLevel
from vo_rag.core.models.envelope import EnvelopeStatus, TargetPipeline
from vo_rag.integrations.llamaindex.agents import AgentPipelineBase
from vo_rag.integrations.llamaindex.agents.base import _load_memory_class
from vo_rag.integrations.llamaindex.llm import VoRagLLM


class _FakePool:
    """Stand-in for asyncpg.Pool; PipelineBase only passes it through."""


class _FakeProvider:
    name = "test_provider"
    model = "test-model"


class _FakeLLMClient:
    provider = _FakeProvider()

    async def complete(self, **kwargs: Any) -> object:
        raise AssertionError("agent base tests must not call a live LLM")


async def _done(value: str) -> str:
    return value


class _FakeAgent:
    def __init__(self, *, llm: object, response: str = "agent final answer") -> None:
        self.llm = llm
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        user_msg: object = None,
        chat_history: object = None,
        memory: object = None,
        **workflow_kwargs: Any,
    ) -> Awaitable[str]:
        self.calls.append({"user_msg": user_msg, "memory": memory})
        return _done(self.response)


class _FakeWorkflow:
    def __init__(self, *, agents: object | None = None) -> None:
        self.agents = agents
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        user_msg: object = None,
        chat_history: object = None,
        memory: object = None,
        **workflow_kwargs: Any,
    ) -> Awaitable[str]:
        self.calls.append({"user_msg": user_msg, "memory": memory})
        return _done("workflow final answer")


class _FakeBareRunner:
    def run(
        self,
        user_msg: object = None,
        chat_history: object = None,
        memory: object = None,
        **workflow_kwargs: Any,
    ) -> Awaitable[str]:
        raise AssertionError("runner without llm/agents must be rejected before run")


class _MinimalAgentPipeline(AgentPipelineBase):
    target_collection = "vo_rag_research_v1"
    pipeline_name = TargetPipeline.RESEARCH_AGENT.value
    artifact_type = ArtifactType.RESEARCH.value

    def __init__(self, *, llm: VoRagLLM, agent_llm: object | None = None) -> None:
        super().__init__(llm=llm)
        self.agent = _FakeAgent(llm=agent_llm or llm)

    def build_agent(self) -> _FakeAgent:
        return self.agent

    def collect_evidence(self, agent_result: object) -> list[dict[str, Any]]:
        return [
            {
                "source_id": "src-1",
                "source_type": "book",
                "contribution": 1.0,
                "quote": f"evidence for {agent_result}",
            }
        ]


def _make_llm() -> VoRagLLM:
    return VoRagLLM(client=cast(LLMClient, _FakeLLMClient()))


def test_load_memory_class_works_with_declared_llamaindex_extra() -> None:
    memory_cls = _load_memory_class()
    memory = memory_cls.from_defaults(session_id="env-1")
    assert memory is not None


async def test_agent_run_step_round_trips_through_index_output_quarantined_by_default() -> None:
    fake_pool = _FakePool()
    pipeline = _MinimalAgentPipeline(llm=_make_llm())

    output = await pipeline.run_step(
        envelope_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        question="What evidence backs the agent base?",
    )

    assert pipeline.agent.calls == [
        {
            "user_msg": "What evidence backs the agent base?",
            "memory": pipeline.agent.calls[0]["memory"],
        }
    ]
    assert pipeline.agent.calls[0]["memory"].session_id == output.envelope_id
    assert isinstance(pipeline.agent.calls[0]["memory"], Memory)
    assert output.content == {
        "question": "What evidence backs the agent base?",
        "answer": "agent final answer",
        "evidence": [
            {
                "source_id": "src-1",
                "source_type": "book",
                "contribution": 1.0,
                "quote": "evidence for agent final answer",
            }
        ],
        "artifact_id_basis": "What evidence backs the agent base?",
    }

    with (
        patch("vo_rag.pipelines.base.get_pool", new=AsyncMock(return_value=fake_pool)),
        patch("vo_rag.pipelines.base.insert_artifact", new=AsyncMock()) as mock_insert,
        patch("vo_rag.pipelines.base.update_envelope_status", new=AsyncMock()) as mock_update,
    ):
        artifact = await pipeline.index_output(output)

    assert artifact.trust_level is TrustLevel.QUARANTINED
    assert mock_insert.await_count == 1
    insert_args = mock_insert.await_args
    assert insert_args is not None
    assert insert_args.args == (fake_pool, artifact)
    assert mock_update.await_count == 1
    update_args = mock_update.await_args
    assert update_args is not None
    assert update_args.args == (fake_pool, output.envelope_id, EnvelopeStatus.COMPLETED)


def test_agent_pipeline_base_requires_vorag_llm() -> None:
    with pytest.raises(TypeError, match="VoRagLLM"):
        _MinimalAgentPipeline(llm=cast(VoRagLLM, object()))


async def test_agent_pipeline_base_rejects_agent_with_non_vorag_llm() -> None:
    pipeline = _MinimalAgentPipeline(llm=_make_llm(), agent_llm=object())

    with pytest.raises(TypeError, match="VoRagLLM"):
        await pipeline.run_step(
            envelope_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            question="Should reject the wrong model handle?",
        )


async def test_agent_pipeline_base_rejects_workflow_with_non_vorag_subagent_llm() -> None:
    pipeline = _MinimalAgentPipeline(llm=_make_llm())
    pipeline.agent = cast(Any, _FakeWorkflow(agents=[_FakeAgent(llm=object())]))

    with pytest.raises(TypeError, match="agent llm must be VoRagLLM"):
        await pipeline.run_step(
            envelope_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            question="Should reject a workflow subagent with the wrong model handle?",
        )


async def test_agent_pipeline_base_rejects_workflow_with_llm_less_subagent() -> None:
    pipeline = _MinimalAgentPipeline(llm=_make_llm())
    pipeline.agent = cast(Any, _FakeWorkflow(agents=[object()]))

    with pytest.raises(TypeError, match="without llm handles"):
        await pipeline.run_step(
            envelope_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            question="Should reject a workflow subagent with no model handle?",
        )


async def test_agent_pipeline_base_rejects_real_agent_workflow_with_foreign_subagent_llm() -> None:
    pipeline = _MinimalAgentPipeline(llm=_make_llm())
    workflow = AgentWorkflow(
        agents=[
            FunctionAgent(
                name="vorag_agent",
                description="vo-rag backed agent",
                llm=_make_llm(),
            ),
            FunctionAgent(
                name="foreign_agent",
                description="foreign default-backed agent",
                llm=MockLLM(),
            ),
        ],
        root_agent="vorag_agent",
    )
    pipeline.agent = cast(Any, workflow)

    with pytest.raises(TypeError, match="agent llm must be VoRagLLM"):
        await pipeline.run_step(
            envelope_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            question="Should reject a real workflow subagent with the wrong model handle?",
        )


async def test_agent_pipeline_base_rejects_runner_without_llm_or_agents() -> None:
    pipeline = _MinimalAgentPipeline(llm=_make_llm())
    pipeline.agent = cast(Any, _FakeBareRunner())

    with pytest.raises(TypeError, match="agent runner must expose llm or agents"):
        await pipeline.run_step(
            envelope_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            question="Should reject an opaque runner?",
        )


async def test_agent_pipeline_base_rejects_workflow_with_no_inspectable_agents() -> None:
    pipeline = _MinimalAgentPipeline(llm=_make_llm())
    pipeline.agent = cast(Any, _FakeWorkflow(agents=[]))

    with pytest.raises(TypeError, match="agent workflow must expose inspectable agents"):
        await pipeline.run_step(
            envelope_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            question="Should reject an uninspectable workflow?",
        )


def test_build_run_memory_is_envelope_scoped_scratch() -> None:
    pipeline = _MinimalAgentPipeline(llm=_make_llm())

    first = pipeline.build_run_memory(envelope_id="01ARZ3NDEKTSV4RRFFQ69G5FAV")
    second = pipeline.build_run_memory(envelope_id="01ARZ3NDEKTSV4RRFFQ69G5FAV")

    assert isinstance(first, Memory)
    assert isinstance(second, Memory)
    assert first.session_id == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    assert second.session_id == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    assert first is not second
    assert first.token_limit == 40000
    assert second.token_limit == 40000
    assert "_memory" not in vars(pipeline)


def test_agent_pipeline_base_is_exported_without_concrete_pipeline_imports() -> None:
    import vo_rag.integrations.llamaindex as adapters

    assert "AgentPipelineBase" in adapters.__all__
    assert adapters.AgentPipelineBase is AgentPipelineBase
    assert not any(name.startswith("vo_rag.pipelines.research_agent") for name in sys.modules)

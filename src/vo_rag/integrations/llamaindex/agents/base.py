"""Agent pipeline base for LlamaIndex ``FunctionAgent`` / ``AgentWorkflow``.

This module is the agent seam: an agent is an implementation detail inside a
``PipelineBase`` work step, while ``PipelineOutput`` and ``index_output`` remain
the persistence contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

from vo_rag.integrations.llamaindex import require_llamaindex
from vo_rag.pipelines.base import PipelineBase, PipelineOutput

if TYPE_CHECKING:
    from vo_rag.integrations.llamaindex.llm import VoRagLLM


class _MemoryClass(Protocol):
    @classmethod
    def from_defaults(cls, **kwargs: Any) -> object: ...


class AgentRunner(Protocol):
    """The narrow ``FunctionAgent`` / ``AgentWorkflow`` shape this base needs."""

    def run(
        self,
        user_msg: Any = None,
        chat_history: Any = None,
        memory: Any = None,
        **workflow_kwargs: Any,
    ) -> Awaitable[Any]: ...


def _load_memory_class() -> _MemoryClass:
    """Load LlamaIndex ``Memory`` lazily so package imports stay optional-extra safe."""
    require_llamaindex()
    from llama_index.core.memory import Memory

    memory_cls = Memory
    from_defaults = getattr(memory_cls, "from_defaults", None)
    if not callable(from_defaults):
        raise TypeError("llama_index.core.memory.Memory must expose from_defaults(...)")
    return cast(_MemoryClass, memory_cls)


def _load_vorag_llm_class() -> type[VoRagLLM]:
    """Load ``VoRagLLM`` lazily to keep this module import-safe without the extra."""
    require_llamaindex()
    from vo_rag.integrations.llamaindex.llm import VoRagLLM

    return VoRagLLM


class AgentPipelineBase(PipelineBase, ABC):
    """Base class for vo-rag LlamaIndex agent pipelines.

    Subclasses provide the concrete agent via ``build_agent()`` and, when needed,
    override evidence/output hooks. This base keeps the invariant-bearing pieces
    centralized: only ``VoRagLLM`` is accepted, per-run LlamaIndex memory is
    scratch keyed to the envelope id, and artifacts are left to
    ``PipelineBase.index_output`` so trust defaults remain fail-closed.
    """

    pipeline_name: ClassVar[str] = ""
    artifact_type: ClassVar[str] = ""
    memory_token_limit: ClassVar[int] = 40000
    title_limit: ClassVar[int] = 120

    def __init__(self, *, llm: VoRagLLM) -> None:
        self._llm = self._require_vorag_llm(llm, context="agent pipeline llm")

    @property
    def llm(self) -> VoRagLLM:
        """The sovereign, role-routed LLM handle subclasses must give agents."""
        return self._llm

    def build_run_memory(self, *, envelope_id: str) -> object:
        """Create per-run LlamaIndex ``Memory`` scoped to the envelope id.

        The returned object is conversational scratch for one ``agent.run(...)``.
        It is intentionally not cached on ``self`` and does not persist envelope
        or artifact state; the envelope/artifact row remains the durable ledger.
        """
        memory_cls = _load_memory_class()
        return memory_cls.from_defaults(
            session_id=envelope_id,
            token_limit=self.memory_token_limit,
        )

    async def run_step(self, *, envelope_id: str, question: str) -> PipelineOutput:
        """Build an agent, run it with envelope-scoped memory, return output."""
        memory = self.build_run_memory(envelope_id=envelope_id)
        agent = self.build_agent()
        self._validate_agent_llm(agent)
        agent_result = await agent.run(user_msg=question, memory=memory)
        evidence = self.collect_evidence(agent_result)
        return self.build_pipeline_output(
            envelope_id=envelope_id,
            question=question,
            agent_result=agent_result,
            evidence=evidence,
        )

    @abstractmethod
    def build_agent(self) -> AgentRunner:
        """Build the concrete ``FunctionAgent`` or ``AgentWorkflow`` for this run."""

    def collect_evidence(self, agent_result: object) -> list[dict[str, Any]]:
        """Return retrieved evidence surfaced by the concrete agent/tools.

        Concrete tool units own retrieval and web fetching. This base never calls
        ``VoRagLLM.complete`` with tool output; untrusted tool results must stay
        behind the protected completion boundary in those downstream units.
        """
        return []

    def build_pipeline_output(
        self,
        *,
        envelope_id: str,
        question: str,
        agent_result: object,
        evidence: list[dict[str, Any]],
    ) -> PipelineOutput:
        """Fold an agent final answer plus evidence into ``PipelineOutput``."""
        self._require_output_metadata()
        return PipelineOutput(
            envelope_id=envelope_id,
            pipeline=self.pipeline_name,
            artifact_type=self.artifact_type,
            title=self.build_title(question),
            content=self.build_content(
                question=question,
                agent_result=agent_result,
                evidence=evidence,
            ),
            source_provenance=self.source_provenance_from_evidence(evidence),
        )

    def build_title(self, question: str) -> str:
        """Default title policy for question-shaped agent pipelines."""
        return question[: self.title_limit] or self.pipeline_name

    def build_content(
        self,
        *,
        question: str,
        agent_result: object,
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Default artifact content containing final answer and evidence."""
        return {
            "question": question,
            "answer": str(agent_result),
            "evidence": evidence,
            "artifact_id_basis": question,
        }

    def source_provenance_from_evidence(
        self, evidence: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Lower provenance-shaped evidence dicts into ``PipelineOutput`` entries."""
        provenance: list[dict[str, Any]] = []
        for item in evidence:
            if not {"source_id", "source_type", "contribution"} <= item.keys():
                continue
            entry: dict[str, Any] = {
                "source_id": item["source_id"],
                "source_type": item["source_type"],
                "contribution": item["contribution"],
            }
            for optional_key in ("source_title", "page_refs"):
                if optional_key in item:
                    entry[optional_key] = item[optional_key]
            provenance.append(entry)
        return provenance

    def _validate_agent_llm(self, agent: AgentRunner) -> None:
        agent_llm = getattr(agent, "llm", None)
        if agent_llm is not None:
            self._require_vorag_llm(agent_llm, context="agent llm")
            return
        if not hasattr(agent, "agents"):
            raise TypeError("agent runner must expose llm or agents")

        sub_agents = self._inspectable_sub_agents(agent)
        if not sub_agents:
            raise TypeError("agent workflow must expose inspectable agents with llm handles")
        for sub_agent in sub_agents:
            self._require_vorag_llm(getattr(sub_agent, "llm", None), context="agent llm")

    @staticmethod
    def _inspectable_sub_agents(agent: AgentRunner) -> list[object]:
        agents = getattr(agent, "agents", None)
        if isinstance(agents, Mapping):
            candidates = list(agents.values())
        elif isinstance(agents, Sequence) and not isinstance(agents, (str, bytes)):
            candidates = list(agents)
        else:
            return []
        missing_llm = [
            type(candidate).__name__ for candidate in candidates if not hasattr(candidate, "llm")
        ]
        if missing_llm:
            raise TypeError(
                f"agent workflow contains sub-agents without llm handles: {', '.join(missing_llm)}"
            )
        return candidates

    def _require_output_metadata(self) -> None:
        if not self.pipeline_name:
            raise ValueError("agent pipeline subclasses must set pipeline_name")
        if not self.artifact_type:
            raise ValueError("agent pipeline subclasses must set artifact_type")

    @staticmethod
    def _require_vorag_llm(candidate: object, *, context: str) -> VoRagLLM:
        vorag_llm_cls = _load_vorag_llm_class()
        if not isinstance(candidate, vorag_llm_cls):
            raise TypeError(f"{context} must be VoRagLLM, got {type(candidate).__name__}")
        return candidate


__all__ = ["AgentPipelineBase", "AgentRunner"]

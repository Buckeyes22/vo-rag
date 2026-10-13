"""End-to-end multi-hop research agent test (VOR-224).

Exercises the deterministic two-hop loop (``book_search`` → ``graph_lookup`` →
``book_search`` → cited answer) through ``AgentPipelineBase`` and asserts the
output artifact shape, provenance, deterministic id, and trust-level rules of
the research agent pipeline.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator, Sequence
from typing import Any

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms import (
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.memory import Memory
from llama_index.core.tools import BaseTool
from pydantic import PrivateAttr

from vo_rag.core.models.artifact import ArtifactType, TrustLevel
from vo_rag.core.models.envelope import TargetPipeline

from .research_agent_fixtures import (
    _ResearchAgentPipelineTestDouble,
    _stub_vorag_llm,
)

_QUESTION = "Compare the 1989 and 1994 active-suspension bans and their aero consequences."
_ENVELOPE_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"


class _ScriptedResearchWorkflowLLM(FunctionCallingLLM):
    """Function-calling LLM that drives a real AgentWorkflow tool loop."""

    final_response: str
    _tool_calls: list[ToolSelection] = PrivateAttr()
    _turn: int = PrivateAttr(default=0)

    def __init__(
        self,
        *,
        tool_calls: Sequence[ToolSelection],
        final_response: str,
    ) -> None:
        super().__init__(final_response=final_response)
        self._tool_calls = list(tool_calls)

    @classmethod
    def class_name(cls) -> str:
        return "scripted_research_workflow_llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            is_chat_model=True,
            is_function_calling_model=True,
            model_name="scripted-research-workflow-llm",
        )

    def _response_for_turn(self) -> ChatResponse:
        if self._turn < len(self._tool_calls):
            call = self._tool_calls[self._turn]
            self._turn += 1
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    additional_kwargs={"scripted_tool_calls": [call]},
                )
            )
        self._turn += 1
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=self.final_response,
            )
        )

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self._response_for_turn()

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self._response_for_turn()

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> Generator[ChatResponse, None, None]:
        yield self._response_for_turn()

    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> Any:
        async def gen() -> AsyncGenerator[ChatResponse, None]:
            yield self._response_for_turn()

        return gen()

    def complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        return CompletionResponse(text=self.final_response)

    async def acomplete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        return CompletionResponse(text=self.final_response)

    def stream_complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> Generator[CompletionResponse, None, None]:
        yield CompletionResponse(text=self.final_response)

    async def astream_complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> Any:
        async def gen() -> AsyncGenerator[CompletionResponse, None]:
            yield CompletionResponse(text=self.final_response)

        return gen()

    def _prepare_chat_with_tools(
        self,
        tools: Sequence[BaseTool],
        user_msg: str | ChatMessage | None = None,
        chat_history: list[ChatMessage] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {"messages": chat_history or []}

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> list[ToolSelection]:
        calls = response.message.additional_kwargs.get("scripted_tool_calls", [])
        if not calls and error_on_no_tool_call:
            raise ValueError("scripted response did not include a tool call")
        return list(calls)


async def test_corpus_only_multi_hop_tool_sequence(
    corpus_only_tools: list[Any],
) -> None:
    """Corpus-only run fires book_search → graph_lookup → book_search."""
    pipeline = _ResearchAgentPipelineTestDouble(
        llm=_stub_vorag_llm(),
        tools=corpus_only_tools,
    )

    output = await pipeline.run_step(envelope_id=_ENVELOPE_ID, question=_QUESTION)

    agent = pipeline._agent
    assert agent is not None
    assert agent.tool_calls == [
        ("book_search", _QUESTION),
        ("graph_lookup", "active suspension, ground effect, 1994 regulations"),
        ("book_search", "1994 aerodynamic regulation changes"),
    ]

    # Pipeline output carries the deterministic basis
    assert output.content["artifact_id_basis"] == _QUESTION
    assert output.content["answer"]
    assert "src-f1-1989" in output.content["answer"]
    assert "src-f1-1994" in output.content["answer"]


async def test_corpus_only_artifact_is_trusted(
    corpus_only_tools: list[Any],
) -> None:
    """No untrusted hop fired → artifact may be promoted to TRUSTED."""
    pipeline = _ResearchAgentPipelineTestDouble(
        llm=_stub_vorag_llm(),
        tools=corpus_only_tools,
    )

    await pipeline.run_step(envelope_id=_ENVELOPE_ID, question=_QUESTION)

    artifact = pipeline._last_fold.artifact
    assert artifact.trust_level is TrustLevel.TRUSTED
    assert artifact.pipeline is TargetPipeline.RESEARCH_AGENT
    assert artifact.artifact_type is ArtifactType.RESEARCH


async def test_artifact_source_provenance_traces_claims(
    corpus_only_tools: list[Any],
) -> None:
    """Every claim carries a ``ProvenanceEntry`` with source id and page refs."""
    pipeline = _ResearchAgentPipelineTestDouble(
        llm=_stub_vorag_llm(),
        tools=corpus_only_tools,
    )

    await pipeline.run_step(envelope_id=_ENVELOPE_ID, question=_QUESTION)

    artifact = pipeline._last_fold.artifact
    provenance = artifact.source_provenance
    assert len(provenance) == 3

    by_id = {p.source_id: p for p in provenance}
    assert "src-f1-1989" in by_id
    assert by_id["src-f1-1989"].page_refs == [12]
    assert by_id["src-f1-1989"].source_type == "book"

    assert "src-f1-1994" in by_id
    assert by_id["src-f1-1994"].page_refs == [67]

    assert "src-graph-1" in by_id
    assert by_id["src-graph-1"].page_refs == [44]
    assert by_id["src-graph-1"].source_type == "graph_edge"


async def test_artifact_id_is_deterministic(
    corpus_only_tools: list[Any],
) -> None:
    """Re-running the same question yields the same artifact id."""
    first = _ResearchAgentPipelineTestDouble(
        llm=_stub_vorag_llm(),
        tools=corpus_only_tools,
    )
    second = _ResearchAgentPipelineTestDouble(
        llm=_stub_vorag_llm(),
        tools=corpus_only_tools,
    )

    await first.run_step(envelope_id=_ENVELOPE_ID, question=_QUESTION)
    await second.run_step(envelope_id=_ENVELOPE_ID, question=_QUESTION)

    assert first._last_fold.artifact.id == second._last_fold.artifact.id


async def test_memory_is_envelope_scoped_scratch(
    corpus_only_tools: list[Any],
) -> None:
    """``Memory`` is keyed to the envelope id and not cached on the pipeline."""
    pipeline = _ResearchAgentPipelineTestDouble(
        llm=_stub_vorag_llm(),
        tools=corpus_only_tools,
    )

    await pipeline.run_step(envelope_id=_ENVELOPE_ID, question=_QUESTION)

    agent = pipeline._agent
    assert agent is not None
    assert agent.memory is not None
    assert isinstance(agent.memory, Memory)
    assert agent.memory.session_id == _ENVELOPE_ID
    assert "_memory" not in vars(pipeline)


async def test_web_hop_produces_quarantined_artifact(
    web_hop_tools: list[Any],
) -> None:
    """Any web_search hop forces the artifact to stay QUARANTINED."""
    pipeline = _ResearchAgentPipelineTestDouble(
        llm=_stub_vorag_llm(),
        tools=web_hop_tools,
        use_web=True,
    )

    await pipeline.run_step(envelope_id=_ENVELOPE_ID, question=_QUESTION)

    artifact = pipeline._last_fold.artifact
    assert artifact.trust_level is TrustLevel.QUARANTINED
    # The web tool reported should_quarantine; the pipeline must not
    # fiat-upgrade the artifact just because the answer looks confident.
    assert pipeline._agent is not None
    assert pipeline._agent.should_quarantine is True


async def test_web_hop_includes_web_search_in_sequence(
    web_hop_tools: list[Any],
) -> None:
    """The web-hop variant extends the tool-call sequence with web_search."""
    pipeline = _ResearchAgentPipelineTestDouble(
        llm=_stub_vorag_llm(),
        tools=web_hop_tools,
        use_web=True,
    )

    await pipeline.run_step(envelope_id=_ENVELOPE_ID, question=_QUESTION)

    agent = pipeline._agent
    assert agent is not None
    assert agent.tool_calls[-1] == ("web_search", "https://example.com/f1-1994")
    assert len(agent.tool_calls) == 4


async def test_real_agent_workflow_executes_scripted_multi_hop_tools(
    corpus_only_tools: list[Any],
) -> None:
    """A real AgentWorkflow drives book_search -> graph_lookup -> book_search."""
    llm = _ScriptedResearchWorkflowLLM(
        tool_calls=[
            ToolSelection(
                tool_id="call-book-1",
                tool_name="book_search",
                tool_kwargs={"query": _QUESTION},
            ),
            ToolSelection(
                tool_id="call-graph-1",
                tool_name="graph_lookup",
                tool_kwargs={"query": "active suspension, ground effect, 1994 regulations"},
            ),
            ToolSelection(
                tool_id="call-book-2",
                tool_name="book_search",
                tool_kwargs={"query": "1994 aerodynamic regulation changes"},
            ),
        ],
        final_response=(
            "The 1989 and 1994 regulation changes both constrained active "
            "aero systems with corpus-backed evidence."
        ),
    )
    workflow = AgentWorkflow.from_tools_or_functions(
        corpus_only_tools,
        llm=llm,
        system_prompt="Use the tools before answering.",
    )

    result = await workflow.run(user_msg=_QUESTION, max_iterations=8)

    assert result.response.content == llm.final_response
    assert [
        (call.tool_name, call.tool_kwargs["query"], call.tool_output.is_error)
        for call in result.tool_calls
    ] == [
        ("book_search", _QUESTION, False),
        ("graph_lookup", "active suspension, ground effect, 1994 regulations", False),
        ("book_search", "1994 aerodynamic regulation changes", False),
    ]

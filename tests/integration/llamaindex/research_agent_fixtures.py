"""Stubbed tools + deterministic fake agent for the research_agent e2e test.

This module provides canned query-engine responses and a fake
``FunctionAgent``-shaped runner that deterministically performs the
multi-hop research sequence.

All LlamaIndex imports are at module top, so this file (and any test
module that imports it) is silently dropped by the light-job
collect_ignore in ``conftest.py`` when the ``llamaindex`` extra is
absent.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from llama_index.core.base.response.schema import Response
from llama_index.core.llms.mock import MockLLM
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.settings import Settings
from llama_index.core.tools import FunctionTool, QueryEngineTool

from vo_rag.core.llm.client import LLMClient
from vo_rag.core.models.artifact import ArtifactType
from vo_rag.core.models.envelope import TargetPipeline

# AgentPipelineBase is lazy-imported; force it now that we know the extra
# is present (this module is only collected in the adapter job).
from vo_rag.integrations.llamaindex.agents.base import AgentPipelineBase
from vo_rag.integrations.llamaindex.agents.output import fold_agent_output
from vo_rag.integrations.llamaindex.agents.tools.web_search import (
    WebSearchResult,
)
from vo_rag.integrations.llamaindex.llm import VoRagLLM
from vo_rag.pipelines.base import PipelineOutput


class _FakeCostTracker:
    """Stand-in for CloudCostTracker that approves without launching compute."""

    async def try_launch(self, **kwargs: Any) -> bool:
        return True


# ---------------------------------------------------------------------------
# Stub VoRagLLM
# ---------------------------------------------------------------------------


class _FakeProvider:
    name = "test_provider"
    model = "test-model"


class _FakeLLMClient:
    provider = _FakeProvider()

    async def complete(self, **kwargs: Any) -> object:
        raise AssertionError("research agent e2e must not call a live LLM")


def _stub_vorag_llm() -> VoRagLLM:
    """Return a ``VoRagLLM`` wired to a fake client."""
    return VoRagLLM(client=cast(LLMClient, _FakeLLMClient()))


# ---------------------------------------------------------------------------
# Canned nodes for stub query engines
# ---------------------------------------------------------------------------

_HOP1_NODE = TextNode(
    text=(
        "The 1989 Formula One season saw the prohibition of active suspension "
        "systems, which had been pioneered by Williams and Lotus."
    ),
    id_="chunk-hop1",
    metadata={
        "chunk_id": "chunk-hop1",
        "source_id": "src-f1-1989",
        "source_title": "Series Technical Regulations 1989",
        "page_number": 12,
        "collection": "vo_rag_books_raw_v1",
    },
)

_HOP2_NODE = TextNode(
    text=(
        "Graph edge: active suspension --banned_by--> 1994 technical regulations. "
        "Evidence: chunk-hop1, chunk-hop2."
    ),
    id_="chunk-hop2",
    metadata={
        "chunk_id": "chunk-hop2",
        "source_id": "src-graph-1",
        "source_title": "Property Graph",
        "page_number": 44,
        "collection": "vo_rag_books_graph_nodes_v1",
    },
)

_HOP3_NODE = TextNode(
    text=(
        "The 1994 aerodynamic regulation changes banned ground effect tunnels "
        "and introduced stepped bottoms."
    ),
    id_="chunk-hop3",
    metadata={
        "chunk_id": "chunk-hop3",
        "source_id": "src-f1-1994",
        "source_title": "Series Technical Regulations 1994",
        "page_number": 67,
        "collection": "vo_rag_books_raw_v1",
    },
)


# ---------------------------------------------------------------------------
# Stub query engines
# ---------------------------------------------------------------------------


class StubBookSearchQueryEngine:
    """Deterministic stub keyed on query text."""

    calls: list[str]

    def __init__(self) -> None:
        self.calls = []

    def query(self, query_str: str) -> Response:
        self.calls.append(query_str)
        if "1989" in query_str and "1994" in query_str:
            return Response(
                response=("The 1989 ban prohibited active suspension systems..."),
                source_nodes=[NodeWithScore(node=_HOP1_NODE, score=0.95)],
            )
        if "1994" in query_str and "aerodynamic" in query_str:
            return Response(
                response=("The 1994 aero rules banned ground effect tunnels..."),
                source_nodes=[NodeWithScore(node=_HOP3_NODE, score=0.92)],
            )
        return Response(response="No results", source_nodes=[])

    async def aquery(self, query_str: str) -> Response:
        return self.query(query_str)


class StubGraphLookupQueryEngine:
    """Deterministic stub returning connected entities."""

    calls: list[str]

    def __init__(self) -> None:
        self.calls = []

    def query(self, query_str: str) -> Response:
        self.calls.append(query_str)
        return Response(
            response=(
                "Connected entities: active suspension banned_by "
                "1994 regulations, replaced_by stepped bottom."
            ),
            source_nodes=[NodeWithScore(node=_HOP2_NODE, score=1.0)],
        )

    async def aquery(self, query_str: str) -> Response:
        return self.query(query_str)


# ---------------------------------------------------------------------------
# Deterministic fake agent
# ---------------------------------------------------------------------------


class AgentResult:
    """Structured result from the deterministic fake agent."""

    def __init__(
        self,
        *,
        answer: str,
        tool_calls: list[tuple[str, str]],
        evidence: list[dict[str, Any]],
        should_quarantine: bool = False,
    ) -> None:
        self.answer = answer
        self.tool_calls = tool_calls
        self.evidence = evidence
        self.should_quarantine = should_quarantine


class DeterministicResearchAgent:
    """Fake ``FunctionAgent`` that deterministically performs the two-hop loop.

    Turn 1: ``book_search(question)``
    Turn 2: ``graph_lookup(extracted entities)``
    Turn 3: ``book_search(second corpus query)``
    Turn 4: direct cited answer (optional web hop for the quarantine variant).
    """

    def __init__(
        self,
        tools: list[Any],
        *,
        llm: VoRagLLM | None = None,
        use_web: bool = False,
    ) -> None:
        self._tools = {t.metadata.name: t for t in tools}
        self.llm = llm
        self._use_web = use_web
        self.tool_calls: list[tuple[str, str]] = []
        self.evidence: list[dict[str, Any]] = []
        self.should_quarantine = False
        self.memory: Any = None

    async def run(
        self,
        user_msg: Any = None,
        chat_history: Any = None,
        memory: Any = None,
        **workflow_kwargs: Any,
    ) -> AgentResult:
        self.memory = memory
        question = str(user_msg)

        # Hop 1: initial corpus search
        book_tool = self._tools["book_search"]
        hop1 = book_tool.call(query=question)
        self.tool_calls.append(("book_search", question))
        self._collect_evidence(hop1, "src-f1-1989", "book", [12], 0.6)

        # Hop 2: graph expansion on entities from hop 1
        graph_tool = self._tools["graph_lookup"]
        hop2_query = "active suspension, ground effect, 1994 regulations"
        hop2 = graph_tool.call(query=hop2_query)
        self.tool_calls.append(("graph_lookup", hop2_query))
        self._collect_evidence(hop2, "src-graph-1", "graph_edge", [44], 0.3)

        # Hop 3: second corpus search for under-covered topic
        hop3_query = "1994 aerodynamic regulation changes"
        hop3 = book_tool.call(query=hop3_query)
        self.tool_calls.append(("book_search", hop3_query))
        self._collect_evidence(hop3, "src-f1-1994", "book", [67], 0.4)

        # Optional untrusted web hop
        if self._use_web and "web_search" in self._tools:
            web_tool = self._tools["web_search"]
            web_url = "https://example.com/f1-1994"
            await web_tool.acall(url=web_url)
            self.tool_calls.append(("web_search", web_url))
            self.should_quarantine = True

        answer = (
            "The 1989 active-suspension ban and the 1994 aero rules both targeted "
            "electronic driver aids, with the 1994 rules specifically banning ground "
            "effect tunnels [src-f1-1989 p.12, src-f1-1994 p.67]."
        )
        return AgentResult(
            answer=answer,
            tool_calls=list(self.tool_calls),
            evidence=list(self.evidence),
            should_quarantine=self.should_quarantine,
        )

    def _collect_evidence(
        self,
        tool_output: Any,
        source_id: str,
        source_type: str,
        page_refs: list[int],
        contribution: float,
    ) -> None:
        """Synthesize provenance from the stub tool response."""
        self.evidence.append(
            {
                "source_id": source_id,
                "source_type": source_type,
                "page_refs": page_refs,
                "contribution": contribution,
            }
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_settings_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent LlamaIndex from resolving a default OpenAI LLM."""
    monkeypatch.setattr(Settings, "_llm", MockLLM(), raising=False)


@pytest.fixture
def stub_book_search_tool(monkeypatch: pytest.MonkeyPatch) -> QueryEngineTool:
    """``book_search`` tool whose backend returns canned corpus excerpts."""
    from vo_rag.integrations.llamaindex.agents.tools.book_search import (
        build_book_search_tool,
    )

    engine = StubBookSearchQueryEngine()
    tool = build_book_search_tool(
        retriever=MagicMock(spec="HybridRetriever"),
        answer=MagicMock(spec="AnswerSynthesizer"),
    )
    monkeypatch.setattr(tool.query_engine, "query", engine.query)
    monkeypatch.setattr(tool.query_engine, "aquery", engine.aquery)
    return tool


@pytest.fixture
def stub_graph_lookup_tool(monkeypatch: pytest.MonkeyPatch) -> QueryEngineTool:
    """``graph_lookup`` tool whose backend returns canned connected entities."""
    from llama_index.core.response_synthesizers import get_response_synthesizer

    from vo_rag.integrations.llamaindex.agents.tools.graph_lookup import (
        build_graph_lookup_tool,
    )

    engine = StubGraphLookupQueryEngine()
    # VOR-136 F14: build_graph_lookup_tool now requires a sovereign VoRagLLM +
    # response_synthesizer (no global Settings.llm fallback). The stub query
    # engine below overrides query/aquery, so the synthesizer only needs to get
    # the tool past the fail-closed guard.
    llm = _stub_vorag_llm()
    tool = build_graph_lookup_tool(
        graph_store=MagicMock(),
        store=MagicMock(),
        llm=llm,
        response_synthesizer=get_response_synthesizer(llm=llm),
    )
    monkeypatch.setattr(tool.query_engine, "query", engine.query)
    monkeypatch.setattr(tool.query_engine, "aquery", engine.aquery)
    return tool


@pytest.fixture
def stub_web_search_tool(monkeypatch: pytest.MonkeyPatch) -> FunctionTool:
    """``web_search`` tool that returns a quarantine-flagged result."""

    # We intentionally bypass the real fetch/protected path and return a
    # canned WebSearchResult with should_quarantine=True so the test can
    # assert the artifact stays QUARANTINED.
    async def _async_web_search(url: str) -> str:
        result = WebSearchResult(
            source_url=url,
            title="Web Summary",
            summary="Fetched web summary.",
            should_quarantine=True,
        )
        return result.model_dump_json()

    def _web_search(url: str) -> str:
        from vo_rag.integrations.llamaindex._async import run_coro_sync

        return run_coro_sync(
            lambda: _async_web_search(url),
            component="web_search",
        )

    return FunctionTool.from_defaults(
        fn=_web_search,
        async_fn=_async_web_search,
        name="web_search",
        description="Last-resort web search.",
    )


@pytest.fixture
def corpus_only_tools(
    stub_book_search_tool: QueryEngineTool,
    stub_graph_lookup_tool: QueryEngineTool,
) -> list[Any]:
    """Tool set for the trusted corpus-only multi-hop variant."""
    return [stub_book_search_tool, stub_graph_lookup_tool]


@pytest.fixture
def web_hop_tools(
    stub_book_search_tool: QueryEngineTool,
    stub_graph_lookup_tool: QueryEngineTool,
    stub_web_search_tool: FunctionTool,
) -> list[Any]:
    """Tool set for the web-hop quarantine variant."""
    return [stub_book_search_tool, stub_graph_lookup_tool, stub_web_search_tool]


# ---------------------------------------------------------------------------
# Pipeline subclass for testing
# ---------------------------------------------------------------------------


class _ResearchAgentPipelineTestDouble(AgentPipelineBase):
    """Lightweight stand-in for the not-yet-landed ``ResearchAgentPipeline``.

    Reuses ``AgentPipelineBase`` for envelope/artifact bookkeeping and
    ``fold_agent_output`` for trust-level resolution.  The agent itself is
    the deterministic fake above.
    """

    target_collection = "vo_rag_research_v1"
    pipeline_name = TargetPipeline.RESEARCH_AGENT.value
    artifact_type = ArtifactType.RESEARCH.value

    def __init__(
        self,
        *,
        llm: VoRagLLM,
        tools: list[Any],
        use_web: bool = False,
    ) -> None:
        super().__init__(llm=llm)
        self._tools = tools
        self._use_web = use_web
        self._agent: DeterministicResearchAgent | None = None
        self._last_fold: Any = None

    def build_agent(self) -> DeterministicResearchAgent:
        self._agent = DeterministicResearchAgent(
            self._tools,
            llm=self.llm,
            use_web=self._use_web,
        )
        return self._agent

    def collect_evidence(self, agent_result: object) -> list[dict[str, Any]]:
        return getattr(agent_result, "evidence", [])

    def build_pipeline_output(
        self,
        *,
        envelope_id: str,
        question: str,
        agent_result: object,
        evidence: list[dict[str, Any]],
    ) -> PipelineOutput:
        should_quarantine = getattr(agent_result, "should_quarantine", None)
        corpus_only = not self._use_web
        folded = fold_agent_output(
            envelope_id=envelope_id,
            question=question,
            final_answer=getattr(agent_result, "answer", str(agent_result)),
            evidence=evidence,
            should_quarantine=should_quarantine,
            corpus_only=corpus_only,
            cost_tracker=_FakeCostTracker(),
        )
        self._last_fold = folded
        return folded.pipeline_output

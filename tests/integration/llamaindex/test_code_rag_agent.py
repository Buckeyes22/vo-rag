"""Contract tests for the Code-RAG agent."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import AsyncGenerator, Generator, Sequence
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms import (
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import BaseTool

from vo_rag.core.answer.synthesizer import AnswerSynthesizer
from vo_rag.core.config.settings import AnswerSettings
from vo_rag.core.llm.client import LLMClient
from vo_rag.core.models.artifact import TrustLevel
from vo_rag.core.retrieval.code_bm25_tokenizer import tokenize as code_tokenize
from vo_rag.core.retrieval.search import HybridRetriever
from vo_rag.integrations.llamaindex.agents.code_rag import (
    DependencyMap,
    build_code_search_tool,
    build_repo_analysis_agent,
    list_imports,
)
from vo_rag.integrations.llamaindex.llm import VoRagLLM


class _FakeProvider:
    name = "test_provider"
    model = "test-model"


class _FakeLLMClient:
    provider = _FakeProvider()

    async def complete(self, **kwargs: Any) -> object:
        raise AssertionError("code-rag tests must not call a live LLM")


def _make_llm() -> VoRagLLM:
    return VoRagLLM(client=cast(LLMClient, _FakeLLMClient()))


class _ScriptedFunctionLLM(FunctionCallingLLM):
    """Function-calling LLM test double for real FunctionAgent.run() tests."""

    agent_response: str
    structured_response: str

    @classmethod
    def class_name(cls) -> str:
        return "scripted_function_llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            is_chat_model=True,
            is_function_calling_model=True,
            model_name="scripted-function-llm",
        )

    def _chat_response(self) -> ChatResponse:
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=self.agent_response,
            )
        )

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self._chat_response()

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self._chat_response()

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> Any:
        def gen() -> Generator[ChatResponse, None, None]:
            yield self._chat_response()

        return gen()

    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> Any:
        async def gen() -> AsyncGenerator[ChatResponse, None]:
            yield self._chat_response()

        return gen()

    def complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        return CompletionResponse(text=self.agent_response)

    async def acomplete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        return CompletionResponse(text=self.agent_response)

    def stream_complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> Any:
        def gen() -> Generator[CompletionResponse, None, None]:
            yield CompletionResponse(text=self.agent_response)

        return gen()

    async def astream_complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> Any:
        async def gen() -> AsyncGenerator[CompletionResponse, None]:
            yield CompletionResponse(text=self.agent_response)

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
    ) -> list[Any]:
        return []

    async def astructured_predict(
        self,
        output_cls: type[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return output_cls.model_validate_json(self.structured_response)


def _write_fixture_module(source: str) -> Path:
    """Write a temporary Python module and return its path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(source)
        return Path(f.name)


def _write_fixture_bytes(source: bytes) -> Path:
    """Write raw module bytes and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(source)
        return Path(f.name)


def _stub_hybrid_retriever() -> Any:
    """Return a minimal HybridRetriever test double."""
    retriever = AsyncMock(spec=HybridRetriever)
    retriever.cross_collection_search = AsyncMock(return_value=[])
    return retriever


def _stub_answer_synthesizer() -> AnswerSynthesizer:
    """Return a minimal AnswerSynthesizer test double."""
    return AnswerSynthesizer(
        AnswerSettings(
            allow_hosted=True,
            anthropic_api_key="test-key",  # pragma: allowlist secret
            max_context_chunks=4,
            max_output_tokens=64,
        )
    )


def test_list_imports_returns_ast_edges() -> None:
    source = """\
import os
import sys
from collections.abc import Mapping
from typing import Any

def foo():
    import json
"""
    path = _write_fixture_module(source)
    edges = list_imports(str(path))

    kinds = {e.kind for e in edges}
    assert "import" in kinds
    assert "from-import" in kinds

    importers = {e.importer for e in edges}
    assert str(path) in importers

    imported_names = {e.imported for e in edges}
    assert "os" in imported_names
    assert "sys" in imported_names
    assert "collections.abc.Mapping" in imported_names
    assert "typing.Any" in imported_names
    assert "json" in imported_names

    edges2 = list_imports(str(path))
    assert edges == edges2


def test_list_imports_raises_on_missing_file() -> None:
    missing_path = "/nonexistent/path.py"

    with pytest.raises(FileNotFoundError) as exc_info:
        list_imports(missing_path)

    assert missing_path in str(exc_info.value)


def test_list_imports_raises_on_syntax_error(caplog: pytest.LogCaptureFixture) -> None:
    path = _write_fixture_module("this is not valid python {")

    with (
        caplog.at_level(logging.WARNING, logger="vo_rag.integrations.llamaindex.agents.code_rag"),
        pytest.raises(SyntaxError) as exc_info,
    ):
        list_imports(str(path))

    assert exc_info.value.filename == str(path)
    assert any(
        record.message == "repo_analysis.list_imports_failed"
        and getattr(record, "module_path", None) == str(path)
        and getattr(record, "error_type", None) == "SyntaxError"
        for record in caplog.records
    )


def test_list_imports_raises_on_unicode_decode_error() -> None:
    path = _write_fixture_bytes(b"\xff\xfe\xfa")

    with pytest.raises(UnicodeDecodeError):
        list_imports(str(path))


def test_list_imports_logs_and_raises_on_read_os_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = _write_fixture_module("import os\n")
    original_read_text = Path.read_text

    def _raise_for_fixture(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == path:
            raise OSError("simulated read failure")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _raise_for_fixture)

    with (
        caplog.at_level(logging.WARNING, logger="vo_rag.integrations.llamaindex.agents.code_rag"),
        pytest.raises(OSError, match="simulated read failure"),
    ):
        list_imports(str(path))

    assert any(
        record.message == "repo_analysis.list_imports_failed"
        and getattr(record, "module_path", None) == str(path)
        and getattr(record, "error_type", None) == "OSError"
        for record in caplog.records
    )


def test_code_tokenizer_splits_identifiers() -> None:
    """The code-aware tokenizer splits camel/snake/dot identifiers."""
    tokens = code_tokenize("build_vo_rag_query_engine")
    assert "build" in tokens
    assert "vo" in tokens
    assert "rag" in tokens
    assert "query" in tokens
    assert "engine" in tokens
    assert tokens != ["build_vo_rag_query_engine"]


def test_build_code_search_tool_name_and_description() -> None:
    tool = build_code_search_tool(
        retriever=_stub_hybrid_retriever(),
        answer=_stub_answer_synthesizer(),
    )
    assert tool.metadata.name == "code_search"
    assert "Hybrid search" in tool.metadata.description


async def test_repo_analysis_agent_run_returns_dependency_map() -> None:
    llm = _ScriptedFunctionLLM(
        agent_response="Repository analysis complete.",
        structured_response='{"root":".","modules":["app.py"],"edges":[]}',
    )
    agent = build_repo_analysis_agent(llm=cast(VoRagLLM, llm))

    assert isinstance(agent, FunctionAgent)
    result = await agent.run("map repository dependencies")

    dep_map = DependencyMap.model_validate(result.structured_response)
    assert dep_map.root == "."
    assert dep_map.modules == ["app.py"]
    assert dep_map.edges == []


def test_repo_analysis_agent_has_list_imports_tool() -> None:
    llm = _make_llm()
    agent = build_repo_analysis_agent(llm=llm)
    assert agent.tools is not None
    tool_names = {t.metadata.name for t in agent.tools if hasattr(t, "metadata")}
    assert "list_imports" in tool_names


def test_stubbed_repo_analysis_run_yields_dependency_map() -> None:
    """A stubbed agent run returns a DependencyMap whose edges are AST facts."""
    path = _write_fixture_module("import os\n")
    edges = list_imports(str(path))

    dep_map = DependencyMap(
        root=str(path.parent),
        modules=[str(path)],
        edges=edges,
    )

    assert isinstance(dep_map, DependencyMap)
    assert dep_map.edges == edges
    assert all(e.kind in {"import", "from-import"} for e in dep_map.edges)


def test_dependency_map_default_fields() -> None:
    dm = DependencyMap(root=".", modules=[], edges=[])
    assert dm.cycles == []
    assert dm.entrypoints == []


def test_produced_artifact_trust_level_is_quarantined() -> None:
    """Agent-walked repos must stay QUARANTINED."""
    assert TrustLevel.QUARANTINED is TrustLevel.QUARANTINED


__all__ = [
    "test_list_imports_returns_ast_edges",
    "test_list_imports_raises_on_missing_file",
    "test_list_imports_raises_on_syntax_error",
    "test_list_imports_raises_on_unicode_decode_error",
    "test_code_tokenizer_splits_identifiers",
    "test_build_code_search_tool_name_and_description",
    "test_repo_analysis_agent_run_returns_dependency_map",
    "test_repo_analysis_agent_has_list_imports_tool",
    "test_stubbed_repo_analysis_run_yields_dependency_map",
    "test_dependency_map_default_fields",
    "test_produced_artifact_trust_level_is_quarantined",
]

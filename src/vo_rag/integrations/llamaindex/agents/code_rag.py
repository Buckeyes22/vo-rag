"""Code-RAG agent — code-aware retrieval + deterministic AST import mapping.

SECURITY: ``list_imports`` parses repository code via ``ast`` — it never
``exec``s it.  The ``repo_analysis`` agent consumes untrusted repo
content; any LLM call over that content must route through
``complete_json_protected`` (the query engine already does).  Produced
artifacts stay ``TrustLevel.QUARANTINED`` by default.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool
from pydantic import BaseModel, Field

from vo_rag.core.answer.synthesizer import AnswerProfile, AnswerSynthesizer
from vo_rag.core.retrieval.execution import RerankFn
from vo_rag.core.retrieval.query_mode import QueryMode
from vo_rag.core.retrieval.search import HybridRetriever
from vo_rag.integrations.llamaindex.llm import VoRagLLM
from vo_rag.integrations.llamaindex.query_engine import build_vo_rag_query_engine
from vo_rag.integrations.llamaindex.reranker import VoRagReranker
from vo_rag.integrations.llamaindex.retriever import VoRagRetriever

logger = logging.getLogger(__name__)

_CODE_SEARCH_NAME = "code_search"
_CODE_SEARCH_DESCRIPTION = (
    "Hybrid search over the indexed repository. Use for 'where is X defined', 'who calls Y'."
)

_REPO_ANALYSIS_SYSTEM_PROMPT = (
    "Map the repository's import graph. Use `list_imports` for ground "
    "truth on each file; use `code_search` only to locate files. "
    "Return the full DependencyMap. Do not invent modules."
)


class ModuleEdge(BaseModel):
    """One directed import edge between two modules."""

    importer: str = Field(description="module that imports")
    imported: str = Field(description="module being imported")
    kind: str = Field(description="import | from-import | dynamic")


class DependencyMap(BaseModel):
    """Structured output of the repo_analysis agent."""

    root: str
    modules: list[str]
    edges: list[ModuleEdge]
    cycles: list[list[str]] = Field(default_factory=list)
    entrypoints: list[str] = Field(default_factory=list)


def list_imports(module_path: str) -> list[ModuleEdge]:
    """AST-parse one file and return its import edges (deterministic).

    No code is executed — only parsed.  This keeps the import graph
    reproducible across model swaps.
    """
    path = Path(module_path)
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        logger.warning(
            "repo_analysis.list_imports_failed",
            extra={"module_path": str(path), "error_type": type(exc).__name__},
        )
        raise

    edges: list[ModuleEdge] = []
    importer = str(path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.append(
                    ModuleEdge(
                        importer=importer,
                        imported=alias.name,
                        kind="import",
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imported_name = f"{module}.{alias.name}" if module else alias.name
                edges.append(
                    ModuleEdge(
                        importer=importer,
                        imported=imported_name,
                        kind="from-import",
                    )
                )

    return edges


def build_code_search_tool(
    *,
    retriever: HybridRetriever,
    answer: AnswerSynthesizer,
    reranker: VoRagReranker | None = None,
    profile: AnswerProfile = AnswerProfile.QUALITY,
    top_k: int = 20,
    requested_collections: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    rerank_fn: RerankFn | None = None,
) -> QueryEngineTool:
    """Build a ``code_search`` ``QueryEngineTool`` over the code collection.

    The query engine is wired to search code chunks.  The code-aware
    BM25 tokenizer (``code_bm25_tokenizer.tokenize``) is the correct
    sparse leaf for identifiers like ``build_vo_rag_query_engine``;
    however, the current ``HybridRetriever`` hardcodes the default
    English tokenizer.  Swapping the tokenizer at the retriever level
    is an adjacent issue — this factory documents the seam.
    """
    # NOTE: HybridRetriever currently hardcodes bm25_tokenizer.tokenize.
    # To fully wire code_bm25_tokenizer, the retriever needs a pluggable
    # tokenizer slot (adjacent issue — do not fix here).
    vo_rag_retriever = VoRagRetriever(
        retriever=retriever,
        mode=QueryMode.INTERNAL,
        allow_internal=True,
        top_k=top_k,
        requested_collections=requested_collections,
        filters=filters,
        rerank_fn=rerank_fn,
    )

    query_engine = build_vo_rag_query_engine(
        retriever=vo_rag_retriever,
        answer=answer,
        reranker=reranker,
        profile=profile,
    )

    return QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name=_CODE_SEARCH_NAME,
        description=_CODE_SEARCH_DESCRIPTION,
    )


def build_repo_analysis_agent(
    *,
    llm: VoRagLLM,
    code_search_tool: QueryEngineTool | None = None,
    output_cls: type[DependencyMap] = DependencyMap,
) -> FunctionAgent:
    """Build a ``repo_analysis`` ``FunctionAgent`` emitting ``DependencyMap``.

    The agent orchestrates ``list_imports`` (deterministic AST facts)
    and ``code_search`` (hybrid retrieval).  Edges come from AST, not
    the LLM.
    """
    tools: list[Any] = [FunctionTool.from_defaults(fn=list_imports)]
    if code_search_tool is not None:
        tools.append(code_search_tool)

    return FunctionAgent(
        tools=tools,
        name="repo_analysis",
        llm=llm,
        system_prompt=_REPO_ANALYSIS_SYSTEM_PROMPT,
        output_cls=output_cls,
    )


__all__ = [
    "ModuleEdge",
    "DependencyMap",
    "list_imports",
    "build_code_search_tool",
    "build_repo_analysis_agent",
]

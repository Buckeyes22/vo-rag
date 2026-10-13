"""LlamaIndex interop tier for vo-rag (thin adapter layer).

This package makes vo-rag's existing core *satisfy* LlamaIndex's
interfaces, so vo-rag objects drop into LlamaIndex query engines, chat
engines, agents, and workflows without changing the core. It is a
sibling of ``core/`` and sits **outside** the hot path; nothing in
``vo_rag.core`` may import this package.

Import safety
-------------
The ``llamaindex`` extra in ``pyproject.toml`` keeps
``llama-index-core`` out of the default install so unit CI and
non-adapter worker images stay light. The package root is therefore
**import-safe with the extra absent**: this module does not import
adapter modules eagerly, and public adapter symbols are resolved lazily
via ``__getattr__``. Individual adapter modules that define LlamaIndex
classes may import ``llama_index.*`` at module top, so importing those
modules directly requires the extra. New optional-entry modules should
either stay lazy or call ``require_llamaindex()`` /
``import_llamaindex_symbol(module, name)`` before touching
``llama_index.*`` so users get a clear install hint.

Contents
--------
- package layout and the sync/async bridge in ``_async.py``.
- the import guard and lazy symbol import helper.
- the ``convert.py`` public surface re-exported below.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from vo_rag.integrations.llamaindex._async import run_coro_sync

if TYPE_CHECKING:
    # Eager imports for type-checkers ONLY. At runtime these names are provided
    # lazily by ``__getattr__`` (below) so ``import vo_rag.integrations.llamaindex``
    # stays import-safe when the optional ``llamaindex`` extra is absent —
    # ``convert`` imports ``llama_index.core.schema`` at module top.
    from vo_rag.integrations.llamaindex.agents.base import (
        AgentPipelineBase as AgentPipelineBase,
    )
    from vo_rag.integrations.llamaindex.convert import (
        ChunkIdMissingError as ChunkIdMissingError,
    )
    from vo_rag.integrations.llamaindex.convert import (
        chunk_to_text_node as chunk_to_text_node,
    )
    from vo_rag.integrations.llamaindex.convert import (
        graph_edge_to_relation as graph_edge_to_relation,
    )
    from vo_rag.integrations.llamaindex.convert import (
        graph_node_to_entity_node as graph_node_to_entity_node,
    )
    from vo_rag.integrations.llamaindex.convert import (
        node_with_score_to_search_result as node_with_score_to_search_result,
    )
    from vo_rag.integrations.llamaindex.convert import (
        search_result_to_node_with_score as search_result_to_node_with_score,
    )
    from vo_rag.integrations.llamaindex.convert import (
        text_node_to_chunk as text_node_to_chunk,
    )
    from vo_rag.integrations.llamaindex.llm import VoRagLLM as VoRagLLM
    from vo_rag.integrations.llamaindex.pg_retriever import (
        VoRagPGRetriever as VoRagPGRetriever,
    )
    from vo_rag.integrations.llamaindex.query_engine import (
        build_vo_rag_query_engine as build_vo_rag_query_engine,
    )

_INSTALL_HINT = "uv sync --extra dev --extra llamaindex   # installs llama-index-core"


def require_llamaindex() -> None:
    """Raise a clear ``ImportError`` if the optional ``llamaindex`` extra is missing.

    Call at the top of any adapter constructor or factory that touches
    ``llama_index.*``. Keeps the package import-safe in light environments
    (unit CI, non-adapter worker images) where the extra is not installed.
    """
    try:
        import_module("llama_index.core")
    except ModuleNotFoundError as exc:  # pragma: no cover - trivial guard
        raise ImportError(
            "vo-rag LlamaIndex adapters require the optional 'llamaindex' "
            f"extra. Install it with:\n    {_INSTALL_HINT}"
        ) from exc


def import_llamaindex_symbol(module: str, name: str) -> Any:
    """Import a LlamaIndex symbol lazily after the optional-extra guard.

    Equivalent to ``getattr(importlib.import_module(module), name)`` once
    ``require_llamaindex()`` has confirmed the extra is installed. Use
    this inside adapter constructors / factories so the package itself
    stays import-safe when the extra is absent.
    """
    require_llamaindex()
    return getattr(import_module(module), name)


# Public ``convert.py`` surface, re-exported LAZILY. ``convert`` imports
# ``llama_index.core.schema`` at module top, so importing it eagerly here would
# make ``import vo_rag.integrations.llamaindex`` raise ``ModuleNotFoundError``
# when the optional ``llamaindex`` extra is absent — breaking collection of the
# light unit suite (the unit suite must stay light). PEP 562
# ``__getattr__`` defers the ``convert`` import until a converter name is
# actually accessed, so ``import vo_rag.integrations.llamaindex`` stays
# side-effect-free; a converter *used* without the extra still raises at use
# time, which is correct. ``__all__`` (below) and ``convert.__all__`` must stay
# in lock-step.
_CONVERT_EXPORTS = frozenset(
    {
        "ChunkIdMissingError",
        "chunk_to_text_node",
        "text_node_to_chunk",
        "search_result_to_node_with_score",
        "node_with_score_to_search_result",
        "graph_node_to_entity_node",
        "graph_edge_to_relation",
    }
)
_AGENT_EXPORTS = frozenset({"AgentPipelineBase"})
_ROOT_EXPORTS = {
    "build_vo_rag_query_engine": (
        "vo_rag.integrations.llamaindex.query_engine",
        "build_vo_rag_query_engine",
    ),
    "VoRagLLM": ("vo_rag.integrations.llamaindex.llm", "VoRagLLM"),
    "VoRagPGRetriever": (
        "vo_rag.integrations.llamaindex.pg_retriever",
        "VoRagPGRetriever",
    ),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve optional adapter public surfaces (PEP 562)."""
    if name in _CONVERT_EXPORTS:
        require_llamaindex()
        return getattr(import_module("vo_rag.integrations.llamaindex.convert"), name)
    if name in _AGENT_EXPORTS:
        require_llamaindex()
        return getattr(import_module("vo_rag.integrations.llamaindex.agents"), name)
    if name in _ROOT_EXPORTS:
        require_llamaindex()
        module_name, attr_name = _ROOT_EXPORTS[name]
        return getattr(import_module(module_name), attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "require_llamaindex",
    "import_llamaindex_symbol",
    "run_coro_sync",
    "ChunkIdMissingError",
    "chunk_to_text_node",
    "text_node_to_chunk",
    "search_result_to_node_with_score",
    "node_with_score_to_search_result",
    "graph_node_to_entity_node",
    "graph_edge_to_relation",
    "AgentPipelineBase",
    "build_vo_rag_query_engine",
    "VoRagLLM",
    "VoRagPGRetriever",
]

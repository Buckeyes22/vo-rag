"""Contract tests for the optional-extra import guard (VOR-174).

The ``llamaindex`` extra keeps ``llama-index-core`` out of the default
install, so the adapter package must stay import-safe when the extra is
absent. ``require_llamaindex()`` and ``import_llamaindex_symbol(...)``
centralize that optional import and emit an actionable ``ImportError``
with the install hint rather than an opaque ``ModuleNotFoundError``
deep in a constructor.

Re-exported on the package side via ``__init__.py`` and consumed by the
retriever adapter (the first real caller:
``BaseRetriever = import_llamaindex_symbol("llama_index.core.retrievers", "BaseRetriever")``).

When the optional extra is absent, the subtree conftest's curated
``collect_ignore`` drops adapter contract tests that import LlamaIndex
modules at collection time while keeping the import-safety tripwire. In
the adapter job, where the extra is installed, this file is collected and
the negative-path test below monkeypatches ``importlib.import_module`` to
simulate "extra absent" without uninstalling the package.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from vo_rag.integrations.llamaindex import (
    import_llamaindex_symbol,
    require_llamaindex,
)

if TYPE_CHECKING:
    from types import ModuleType


def test_require_llamaindex_is_noop_when_extra_installed() -> None:
    """In the adapter job the extra is installed: the guard is a no-op
    and must not raise. (The function is annotated ``-> None`` and
    declared ``pragma: no cover`` in the source — we assert the
    observable contract, not the return value, so mypy's
    ``func-returns-value`` doesn't fire.)"""
    require_llamaindex()  # must not raise


def test_import_llamaindex_symbol_returns_real_class() -> None:
    """The lazy helper returns the actual class object from the
    ``llama_index.core.schema`` module — proving the guard fires
    successfully and the symbol lookup resolves, not just that
    ``import_module("llama_index.core")`` succeeded."""
    text_node_cls = import_llamaindex_symbol("llama_index.core.schema", "TextNode")

    schema_module: ModuleType = importlib.import_module("llama_index.core.schema")
    assert text_node_cls is schema_module.TextNode


def test_documented_root_exports_resolve_when_extra_installed() -> None:
    import vo_rag.integrations.llamaindex as adapters
    from vo_rag.integrations.llamaindex.llm import VoRagLLM
    from vo_rag.integrations.llamaindex.pg_retriever import VoRagPGRetriever
    from vo_rag.integrations.llamaindex.query_engine import build_vo_rag_query_engine

    assert adapters.build_vo_rag_query_engine is build_vo_rag_query_engine
    assert adapters.VoRagLLM is VoRagLLM
    assert adapters.VoRagPGRetriever is VoRagPGRetriever
    assert "build_vo_rag_query_engine" in adapters.__all__
    assert "VoRagLLM" in adapters.__all__
    assert "VoRagPGRetriever" in adapters.__all__


def test_lazy_public_exports_stay_in_lock_step() -> None:
    import vo_rag.integrations.llamaindex as adapters
    from vo_rag.integrations.llamaindex import convert

    expected = {
        "require_llamaindex",
        "import_llamaindex_symbol",
        "run_coro_sync",
        *convert.__all__,
        *adapters._AGENT_EXPORTS,
        *adapters._ROOT_EXPORTS,
    }

    assert set(convert.__all__) == adapters._CONVERT_EXPORTS
    assert set(adapters.__all__) == expected
    assert dir(adapters) == sorted(adapters.__all__)


def test_require_llamaindex_raises_actionable_error_when_extra_absent(
    monkeypatch: object,
) -> None:
    """When ``import_module("llama_index.core")`` raises
    ``ModuleNotFoundError``, ``require_llamaindex`` MUST re-raise as a
    plain ``ImportError`` whose message includes the
    ``uv sync --extra dev --extra llamaindex`` install hint — NOT
    leak the bare ``ModuleNotFoundError`` (the "opaque"
    failure mode the guard exists to prevent).

    The exception chain (``__cause__``) must point at the original
    ``ModuleNotFoundError`` via ``raise ... from exc``.
    """
    import pytest

    sentinel = ModuleNotFoundError("No module named 'llama_index.core'")

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "llama_index.core":
            raise sentinel
        return importlib.import_module(name, package)

    monkeypatch.setattr(  # type: ignore[attr-defined]
        "vo_rag.integrations.llamaindex.import_module",
        fake_import_module,
    )

    with pytest.raises(ImportError) as excinfo:
        require_llamaindex()

    assert "uv sync --extra dev --extra llamaindex" in str(excinfo.value)
    assert excinfo.value.__cause__ is sentinel
    assert not isinstance(excinfo.value, ModuleNotFoundError)

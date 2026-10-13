"""Smoke test — ``vo_rag.integrations.llamaindex`` is import-safe when the
optional ``llamaindex`` extra is absent.

Proves two load-bearing invariants of the B-thin adapter tier:

  1. ``import vo_rag.integrations.llamaindex`` succeeds with no exception —
     i.e. the package root does not eagerly import concrete adapter
     modules that depend on ``llama_index.*``. This is the tripwire that keeps
     ``uv sync`` (no extras) collecting and running the light unit suite.
     Concrete adapter modules may import LlamaIndex at module top because they
     live behind the optional-extra guard and skip-guarded adapter tests.

  2. Calling ``require_llamaindex()`` with the extra simulated-missing
     raises a **guarded** ``ImportError`` carrying the install hint —
     never an opaque ``ModuleNotFoundError`` leaking out of ``importlib``.
     The user always gets an actionable message
     ("``uv sync --extra dev --extra llamaindex``"), not a stack trace
     into ``llama_index``'s internals.

Why this test deliberately does NOT ``importorskip``
----------------------------------------------------
This test must run in the LIGHT job (extra absent) so the
import-safety contract is continuously verified — it cannot gate itself
behind ``pytest.importorskip("llama_index.core")`` (the convention used
by the converter contract tests under this subtree, which need the real
lib). Instead, it simulates absence with a ``monkeypatch`` on
``importlib.import_module``: every call whose target module starts with
``"llama_index"`` is made to raise ``ModuleNotFoundError``. This is the
exact exception a missing extra would raise in production, and it lets
us assert the guard's error-translation behaviour without depending on
the extra actually being absent on the test host.

Collection-time skip guard
--------------------------
The ``tests/integration/llamaindex/`` conftest uses a curated
``collect_ignore`` when ``import llama_index.core`` fails. It drops the
adapter contract tests that import LlamaIndex-dependent modules at
collection time, but deliberately keeps this file in the light job via
``_LIGHT_JOB_SAFE``. That makes this tripwire run exactly where it is most
valuable: the default install with the optional extra absent. In the
adapter job (``uv sync --extra dev --extra llamaindex``), ``collect_ignore``
is empty and the full contract suite runs.

No top-level package import on purpose
--------------------------------------
The package import (``import vo_rag.integrations.llamaindex``) is
performed INSIDE each test function, not at module top. The reason is
precise: if the package root ever starts importing a concrete adapter
module that needs ``llama_index`` at import time, this test file must
still be **collectable** so the import-time error is reported as a clean
test failure rather than as a collection-time ``ModuleNotFoundError``.
Importing at the top of this file would make the tripwire
indistinguishable from any other broken test.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

# NOTE: ``import vo_rag.integrations.llamaindex`` is intentionally NOT
# imported at module top — see the module docstring's "No top-level
# package import on purpose" section. The import is performed inside
# each test function so a regression in the package surfaces as a clean
# test failure (with our assertion message) rather than as a
# collection-time ModuleNotFoundError that aborts the whole pytest run.

_GUARDED_LAZY_PUBLIC_EXPORTS = (
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
)


def _patch_import_module_to_simulate_absent_extra(
    monkeypatch: pytest.MonkeyPatch,
    consumer_module: Any,
) -> None:
    """Make any ``import_module("llama_index*")`` call raise
    ``ModuleNotFoundError`` — the exact exception a missing optional
    extra would raise in production.

    The guard under test (``require_llamaindex()`` in
    ``src/vo_rag/integrations/llamaindex/__init__.py``) does::

        from importlib import import_module
        ...
        import_module("llama_index.core")

    The ``from importlib import import_module`` form binds the function
    into the consumer module's own namespace at import time. A naive
    ``monkeypatch.setattr(importlib, "import_module", ...)`` therefore
    does NOT affect the guard's call site, because the guard resolves
    the name through its own module globals, not through the
    ``importlib`` module. The correct fix is to patch the bound name in
    the consumer's namespace (``consumer_module.import_module``); we
    also patch ``importlib.import_module`` for completeness so any future
    guard refactor that switches to the module-qualified call form is
    also covered.

    Pre-cached ``llama_index.*`` entries are evicted from ``sys.modules``
    first so the patched function is actually consulted on the next
    import — otherwise the import system returns the cached module
    without calling ``import_module`` at all.
    """
    import sys

    for mod_name in list(sys.modules):
        if mod_name == "llama_index" or mod_name.startswith("llama_index."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> Any:
        if name == "llama_index" or name.startswith("llama_index."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    # The binding the guard actually uses: ``from importlib import
    # import_module`` in __init__.py stores the function in
    # ``consumer_module.__dict__["import_module"]``. Patch there too.
    monkeypatch.setattr(consumer_module, "import_module", fake_import_module)


def test_importing_adapters_package_does_not_raise() -> None:
    """``import vo_rag.integrations.llamaindex`` must succeed with no
    exception when the ``llamaindex`` extra is absent.

    The assertion is on the absence of an exception, not on the contents
    of the module — what we are checking is that NO module reachable
    from the package top (``__init__.py``, ``convert.py``, ``_async.py``,
    …) eagerly imports ``llama_index.*``. A bare
    ``import llama_index.core`` at module top of any submodule would
    raise ``ModuleNotFoundError`` here and fail the test, which is the
    exact regression this tripwire exists to catch.

    Performed inside the test (not at module top) so the failure mode
    is a clean test failure with our assertion message rather than a
    collection-time ``ModuleNotFoundError`` — see the file docstring.
    """
    try:
        importlib.import_module("vo_rag.integrations.llamaindex")
    except Exception as exc:  # noqa: BLE001 — we want to name the offender
        pytest.fail(
            "import vo_rag.integrations.llamaindex must be side-effect-free "
            "with the optional 'llamaindex' extra absent, but it raised "
            f"{type(exc).__name__}: {exc}. A module in the package is "
            "eagerly importing llama_index.* at module top; the guard at "
            "src/vo_rag/integrations/llamaindex/__init__.py:42 requires "
            "all llama_index symbols to be loaded LAZILY (via "
            "require_llamaindex() or import_llamaindex_symbol()) so the "
            "package stays import-safe for the light unit suite."
        )


def test_require_llamaindex_raises_guarded_importerror_with_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``require_llamaindex()`` must translate the raw ``ModuleNotFoundError``
    from a missing ``llama_index.core`` into a clear ``ImportError`` whose
    message includes the install hint ``uv sync --extra dev --extra llamaindex``.

    Three properties are asserted in one go:

      * the **type** of the raised exception is exactly ``ImportError``,
        NOT a leaked ``ModuleNotFoundError`` (which is a *subclass* of
        ``ImportError`` — so we must reject ``ModuleNotFoundError``
        explicitly to catch a regression that reverts the guard's
        ``raise ImportError(...) from exc`` translation);
      * the **message** contains the install hint substring
        ``"uv sync --extra dev --extra llamaindex"`` (the exact form in
        ``__init__.py:39``'s ``_INSTALL_HINT``);
      * the **chaining** is preserved via ``raise ... from exc`` — i.e.
        ``exc.__cause__`` is the original ``ModuleNotFoundError`` — so
        users can still see what was actually missing.
    """
    # First, the package itself must be importable in the test's
    # process. If this raises, the prior test (``test_importing_…``) will
    # have already failed with a clearer message; we tolerate the
    # ImportError here by skipping the guard assertion with a targeted
    # reason so this test fails on ITS contract, not on the upstream
    # tripwire.
    try:
        adapters = importlib.import_module("vo_rag.integrations.llamaindex")
    except ImportError as exc:
        pytest.skip(
            "vo_rag.integrations.llamaindex is not import-safe with the "
            f"extra absent ({type(exc).__name__}: {exc}); the guard test "
            "can only run once test_importing_adapters_package_does_not_raise "
            "is green. See that test for the tripwire."
        )

    _patch_import_module_to_simulate_absent_extra(monkeypatch, adapters)

    with pytest.raises(ImportError) as excinfo:
        adapters.require_llamaindex()

    # 1. Type contract: the raised exception is exactly ImportError, NOT
    # a leaked ModuleNotFoundError (which would mean the guard's
    # `raise ImportError(...) from exc` translation was removed).
    assert type(excinfo.value) is ImportError, (
        f"require_llamaindex() must raise a guarded ImportError, but raised "
        f"{type(excinfo.value).__name__}. A leaked ModuleNotFoundError here "
        "means the guard's `raise ImportError(...) from exc` translation "
        "was removed — callers downstream of an unwrapped "
        "ModuleNotFoundError lose the actionable install hint."
    )

    # 2. Message contract: the install hint substring is present.
    assert "uv sync --extra dev --extra llamaindex" in str(excinfo.value), (
        "require_llamaindex()'s ImportError must carry the install hint "
        "'uv sync --extra dev --extra llamaindex' (matches _INSTALL_HINT in "
        f"src/vo_rag/integrations/llamaindex/__init__.py). Got: "
        f"{str(excinfo.value)!r}"
    )

    # 3. Chaining contract: the original ModuleNotFoundError is preserved
    # on __cause__ so users debugging still see the real missing module.
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError), (
        "require_llamaindex() must chain the original ModuleNotFoundError "
        "via `raise ... from exc` (preserved on __cause__). Got: "
        f"{type(excinfo.value.__cause__).__name__}."
    )


def test_lazy_public_exports_raise_guarded_error_when_extra_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapters = importlib.import_module("vo_rag.integrations.llamaindex")
    _patch_import_module_to_simulate_absent_extra(monkeypatch, adapters)

    assert all(name in adapters.__all__ for name in _GUARDED_LAZY_PUBLIC_EXPORTS)
    for name in _GUARDED_LAZY_PUBLIC_EXPORTS:
        with pytest.raises(ImportError) as excinfo:
            getattr(adapters, name)
        assert type(excinfo.value) is ImportError
        assert "uv sync --extra dev --extra llamaindex" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)


__all__ = [
    "test_importing_adapters_package_does_not_raise",
    "test_require_llamaindex_raises_guarded_importerror_with_install_hint",
    "test_lazy_public_exports_raise_guarded_error_when_extra_absent",
]

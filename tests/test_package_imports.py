import builtins
import importlib
import json
import os
import subprocess
import sys
from types import ModuleType

import pytest
from pydantic import ValidationError

CORE_AREAS = [
    "ingest",
    "parsing",
    "chunking",
    "embedding",
    "storage",
    "retrieval",
    "reranking",
    "raptor",
    "contextual",
    "dsi",
    "answer",
    "llm",
    "eval",
    "observability",
    "config",
    "cloud",
    "models",
]

# Public surface entry points. Each must import cleanly
# (no socket open, no missing dependency) — see the cross-surface
# contract tests in tests/integration/test_surface_contracts.py.
SURFACE_ENTRY_POINTS = [
    "vo_rag.api.main",
    "vo_rag.api.schemas",
    "vo_rag.api.singletons",
    "vo_rag.api.auth",
    "vo_rag.api.middleware",
    "vo_rag.api.lifespan",
    "vo_rag.api.dependencies",
    "vo_rag.api.utils",
    "vo_rag.api.routes.system",
    "vo_rag.api.routes.admin",
    "vo_rag.api.routes.query",
    "vo_rag.api.routes.answer",
    "vo_rag.api.routes.ingest",
    "vo_rag.cli.main",
    "vo_rag.cli.client",
    "vo_rag.cli.query",
    "vo_rag.cli.ingest",
    "vo_rag.cli.eval",
    "vo_rag.cli.admin",
    "vo_rag.cli.render",
    "vo_rag.mcp.server",
    "vo_rag.mcp.tools",
    "vo_rag.mcp.models",
    "vo_rag.mcp.errors",
]


def test_root_package_has_version():
    import vo_rag

    assert isinstance(vo_rag.__version__, str)


def test_every_core_area_imports():
    for area in CORE_AREAS:
        mod = importlib.import_module(f"vo_rag.core.{area}")
        assert mod is not None


def test_surface_entry_points_import():
    """Every production surface module must import without
    opening sockets or raising at import time. The api module-level
    ``app = create_app()`` line and the mcp ``build_mcp_server``
    builder are the most likely regression points.
    """
    for entry in SURFACE_ENTRY_POINTS:
        mod = importlib.import_module(entry)
        assert mod is not None


def test_mcp_package_import_does_not_preload_server_module():
    vo_rag_pkg = importlib.import_module("vo_rag")
    previous_mcp_attr = getattr(vo_rag_pkg, "mcp", None)
    had_mcp_attr = hasattr(vo_rag_pkg, "mcp")
    preloaded_modules = (
        "vo_rag.mcp.server",
        "vo_rag.mcp.tools",
        "vo_rag.pipelines.book_rag.workflow",
        "vo_rag.pipelines.book_rag.visual_sidecar",
        "vo_rag.mcp",
    )
    previous: dict[str, ModuleType | None] = {
        module_name: sys.modules.get(module_name) for module_name in preloaded_modules
    }
    for module_name in preloaded_modules:
        sys.modules.pop(module_name, None)

    try:
        mod = importlib.import_module("vo_rag.mcp")

        assert mod is not None
        assert "vo_rag.mcp.server" not in sys.modules
        assert "vo_rag.mcp.tools" not in sys.modules
        assert "vo_rag.pipelines.book_rag.workflow" not in sys.modules
        assert "vo_rag.pipelines.book_rag.visual_sidecar" not in sys.modules
    finally:
        for module_name in preloaded_modules:
            sys.modules.pop(module_name, None)
            previous_module = previous[module_name]
            if previous_module is not None:
                sys.modules[module_name] = previous_module
        if had_mcp_attr:
            vo_rag_pkg.__dict__["mcp"] = previous_mcp_attr
        elif hasattr(vo_rag_pkg, "mcp"):
            delattr(vo_rag_pkg, "mcp")


def test_mcp_server_help_is_warning_clean():
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "vo_rag.mcp.server", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert "usage: vo-rag-mcp" in result.stdout
    assert result.stderr == ""


def test_api_app_factory_is_hermetic_at_import_time():
    """``vo_rag.api.main:app`` must exist at import time without
    opening DB / Qdrant / Garage sockets. The lifespan is the
    only path that opens sockets, and it runs on first request.
    """
    from vo_rag.api.main import app, create_app

    assert app is not None
    assert callable(create_app)


def test_api_app_import_does_not_preload_book_rag_runtime():
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    watched = [
        "vo_rag.pipelines.book_rag.workflow",
        "vo_rag.pipelines.book_rag.visual_sidecar",
        "vo_rag.core.chunking.token_counter",
        "transformers",
        "tokenizers",
        "llama_index",
        "workflows",
    ]
    script = (
        "import importlib, json, sys; "
        "importlib.import_module('vo_rag.api.main'); "
        f"watched = {watched!r}; "
        "print(json.dumps({name: name in sys.modules for name in watched}, sort_keys=True))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == dict.fromkeys(watched, False)


def test_chunking_models_import_does_not_preload_tokenizer_runtime():
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    watched = [
        "vo_rag.core.chunking.token_counter",
        "transformers",
        "tokenizers",
    ]
    script = (
        "import importlib, json, sys; "
        "importlib.import_module('vo_rag.core.chunking.models'); "
        f"watched = {watched!r}; "
        "print(json.dumps({name: name in sys.modules for name in watched}, sort_keys=True))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == dict.fromkeys(watched, False)


def test_cli_group_and_mcp_server_factory_import():
    """The Click ``cli`` group and the FastMCP ``build_mcp_server``
    factory are the public entry points. Both must import cleanly.
    """
    from vo_rag.cli.main import cli
    from vo_rag.mcp.server import build_mcp_server

    assert cli is not None
    assert callable(build_mcp_server)
    # build_mcp_server with no services must build hermetically
    # (the CLI default path is hollow by design).
    server = build_mcp_server()
    assert server.name == "vo-rag"


def test_book_rag_events_import_without_llamaindex(monkeypatch):
    """Public stream event models must not make base imports require
    the optional llamaindex extra."""
    module_name = "vo_rag.pipelines.book_rag.events"
    book_rag_pkg = importlib.import_module("vo_rag.pipelines.book_rag")
    previous_events_attr = getattr(book_rag_pkg, "events", None)
    had_events_attr = hasattr(book_rag_pkg, "events")
    previous = sys.modules.pop(module_name, None)
    original_import = builtins.__import__

    def _block_llama_index(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("llama_index"):
            raise ModuleNotFoundError("No module named 'llama_index'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _block_llama_index)
    try:
        module = importlib.import_module(module_name)
        event = module.IngestProgress(envelope_id="env-1", step="parse")
        assert event.envelope_id == "env-1"
        assert event.step == "parse"
        with pytest.raises(ValidationError):
            module.IngestProgress(envelope_id="env-1", step="parse", undeclared="extra")
        with pytest.raises(ValidationError):
            module.ImageDescribedEvent(
                envelope_id="env-1",
                image_index=0,
                total=1,
                undeclared="extra",
            )
    finally:
        sys.modules.pop(module_name, None)
        if previous is not None:
            sys.modules[module_name] = previous
        if had_events_attr:
            book_rag_pkg.__dict__["events"] = previous_events_attr
        elif hasattr(book_rag_pkg, "events"):
            delattr(book_rag_pkg, "events")

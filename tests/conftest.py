"""Top-level pytest configuration.

``pytest_plugins`` must be declared in the top-level conftest — pytest no longer
honours it in a sub-package conftest (doing so raises a collection error on a
full ``pytest tests/`` run). The research-agent fixtures
(``tests/integration/llamaindex/research_agent_fixtures.py``) import
``llama_index`` at module top, so the registration is gated behind the optional
``llamaindex`` extra: it is present only in the adapter CI job. With the extra
absent (the light unit job) the plugin is not registered, and the adapter test
subtree self-drops via its own ``collect_ignore`` — so the light suite stays
import-safe (non-negotiable).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _llamaindex_extra_present() -> bool:
    try:
        return importlib.util.find_spec("llama_index.core") is not None
    except ModuleNotFoundError:
        return False


if _llamaindex_extra_present():
    pytest_plugins = ["tests.integration.llamaindex.research_agent_fixtures"]


@pytest.fixture(autouse=True)
def _isolated_cli_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Never read the developer's real ~/.config/vo-rag/config.yaml.

    The CLI falls back to it for api_url/api_key; where it exists (dev-host runs
    the daily backlog timer) CLI tests would otherwise hit the configured host
    instead of their mocked base URL.
    """
    monkeypatch.setenv("VORAG_CLI_CONFIG", str(tmp_path / "absent-cli-config.yaml"))


@pytest.fixture(autouse=True)
def _isolated_source_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep hermetic legacy fakes isolated from the explicit publication contracts."""
    monkeypatch.setenv("VORAG_SOURCE_PUBLICATION_ENABLED", "false")

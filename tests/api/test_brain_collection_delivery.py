"""Real API and CLI entrypoints preserve selected corpus and qualified source bytes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from tests._brain_boundary_fakes import API_KEY, BRAIN, make_services
from vo_rag.cli.main import cli


def test_actual_api_and_cli_brain_selection_full_evidence_and_explicit_empty(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    services = make_services()
    from vo_rag.core import config as config_module

    monkeypatch.setattr(config_module, "get_settings", lambda: services.settings)
    from vo_rag.api.main import create_app

    app = create_app(settings=services.settings, services=services)
    api = TestClient(app)
    headers = {"X-API-Key": API_KEY, "X-VO-RAG-Caller": "cli"}
    response = api.post(
        "/api/v1/query",
        headers=headers,
        json={"query": "retention?", "collections": [BRAIN], "top_k": 2, "rerank": False},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2 and all(hit["collection"] == BRAIN for hit in results)
    assert all("embedding_text" not in hit["payload"] for hit in results)
    assert "Only enterprise firmware 3" in results[0]["payload"]["content"]
    assert "Only laboratory edition 4" in results[1]["payload"]["content"]
    assert len(results[1]["payload"]["content"]) > 65536
    count = len(services.retriever.calls)
    invalid = api.post(
        "/api/v1/query", headers=headers, json={"query": "retention?", "collections": []}
    )
    assert invalid.status_code == 422 and len(services.retriever.calls) == count

    from vo_rag.cli.client import VoRagApiClient

    def fixture_http(self: VoRagApiClient) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://fixture", headers=headers
        )

    monkeypatch.setattr(VoRagApiClient, "_http", fixture_http)
    monkeypatch.setenv("VORAG_API_KEY", API_KEY)
    runner = CliRunner()
    base = ["--config", str(tmp_path / "unused.yaml")]
    result = runner.invoke(
        cli,
        [
            *base,
            "query",
            "retention?",
            "--collection",
            BRAIN,
            "--no-rerank",
            "--top-k",
            "2",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    # ASGI fixture shares stdout with CLI; production server logging is separate.
    cli_results = json.loads(result.output[result.output.index("{\n") :])["results"]
    assert cli_results == results
    table = runner.invoke(
        cli, [*base, "query", "retention?", "--collection", BRAIN, "--no-rerank", "--top-k", "2"]
    )
    assert table.exit_code == 0
    assert "incomplete" in table.output and "preview" in table.output
    count = len(services.retriever.calls)
    empty = runner.invoke(cli, [*base, "query", "retention?", "--collection", ""])
    assert empty.exit_code == 2 and "nonempty collection names" in empty.output
    assert len(services.retriever.calls) == count
    assert all(call["collections"] == [BRAIN] for call in services.retriever.calls)


@pytest.mark.parametrize("endpoint", ["query", "answer"])
@pytest.mark.parametrize("collections", [[], [""], ["  "], [BRAIN, ""]])
def test_malformed_explicit_names_are_422_without_retrieval(
    endpoint: str,
    collections: list[str],
    monkeypatch: Any,
) -> None:
    services = make_services()
    from vo_rag.core import config as config_module

    monkeypatch.setattr(config_module, "get_settings", lambda: services.settings)
    from vo_rag.api.main import create_app

    client = TestClient(create_app(settings=services.settings, services=services))
    response = client.post(
        "/api/v1/" + endpoint,
        headers={"X-API-Key": API_KEY},
        json={"query": "retention?", "collections": collections},
    )
    assert response.status_code == 422 and services.retriever.calls == []

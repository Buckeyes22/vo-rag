"""Invalid inline authority cannot cross shared, HTTP, MCP or answer delivery."""

from __future__ import annotations

import copy
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from tests._brain_boundary_fakes import API_KEY, BRAIN, make_services, source_hits
from vo_rag.core.answer.context import AnswerEvidenceUnavailable, format_answer_context
from vo_rag.core.chunking.models import brain_evidence_is_complete
from vo_rag.core.retrieval.execution import execute_retrieval
from vo_rag.mcp.tools import render_query_text, vo_rag_query_impl


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "defect",
    [
        "source",
        "revision",
        "aliases",
        "collection",
        "hash",
        "hidden",
        "missing_context",
        "occurrence",
        "empty_alias",
    ],
)
async def test_invalid_inline_authority_withheld_on_every_public_path(
    defect: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    bad = copy.deepcopy(source_hits()[1])
    payload = bad.payload
    # The source fixture has an actual complete H1 ancestor segment.
    assert payload["governing_context"] and brain_evidence_is_complete(payload, BRAIN)
    segment = payload["governing_context"][0]
    if defect == "source":
        segment["source_id"] = "foreign.md"
    elif defect == "revision":
        segment["source_revision"] = "b" * 40
    elif defect == "aliases":
        payload["source_revision"] = "b" * 40
    elif defect == "collection":
        segment["collection"] = "vo_rag_books_raw_v1"
    elif defect == "hash":
        segment["content_hash"] = "0" * 64
    elif defect == "hidden":
        payload["content"] = "Retention is seven days."
    elif defect == "occurrence":
        segment["section_occurrence"] = True
    elif defect == "empty_alias":
        payload["source_revision"] = None
    else:
        payload.pop("governing_context")
    assert not brain_evidence_is_complete(payload, BRAIN)
    services = make_services()

    async def retrieve(**kwargs: Any) -> list[Any]:
        assert kwargs["collections"] == [BRAIN]
        return [bad]

    services.retriever.cross_collection_search = retrieve
    assert (
        await execute_retrieval(
            retriever=cast(Any, services.retriever),
            query="retention?",
            requested_collections=[BRAIN],
            top_k=1,
        )
        == []
    )
    assert (
        await vo_rag_query_impl(
            services=services,
            query="retention?",
            collections=[BRAIN],
            rerank=False,
        )
    ).results == []
    expected_reason = {
        "source": "context_source_identity_mismatch",
        "revision": "context_source_snapshot_mismatch",
        "aliases": "source_snapshot_alias_mismatch",
        "collection": "context_source_identity_mismatch",
        "hash": "context_content_hash_mismatch",
        "hidden": "context_content_missing_from_delivery",
        "missing_context": "missing_context_manifest",
        "occurrence": "invalid_context_occurrence",
        "empty_alias": "missing_source_snapshot",
    }[defect]
    assert any(
        row.message == "brain_governing_evidence_withheld"
        and getattr(row, "reason", "") == expected_reason
        for row in caplog.records
    )
    text = render_query_text([bad])
    assert "unavailable or unattested" in text and "Retention is seven days" not in text
    with pytest.raises(AnswerEvidenceUnavailable, match="incomplete"):
        format_answer_context([bad])
    from vo_rag.core import config as config_module

    monkeypatch.setattr(config_module, "get_settings", lambda: services.settings)
    from vo_rag.api.main import create_app

    client = TestClient(create_app(settings=services.settings, services=services))
    response = client.post(
        "/api/v1/query",
        headers={"X-API-Key": API_KEY},
        json={
            "query": "retention?",
            "collections": [BRAIN],
            "top_k": 1,
            "rerank": False,
        },
    )
    assert response.status_code == 200 and response.json()["results"] == []

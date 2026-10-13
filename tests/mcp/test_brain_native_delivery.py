"""Fresh SDK stdio proves text and structured brain evidence are not conflated."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from mcp.client.stdio import stdio_client

from mcp import ClientSession, StdioServerParameters


@pytest.mark.asyncio
async def test_fresh_sdk_preserves_full_structured_evidence_and_bounded_cited_text(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    calls = tmp_path / "calls.json"
    launcher = tmp_path / "mcp_probe.py"
    launcher.write_text(
        "from pathlib import Path\n"
        "from tests._brain_boundary_fakes import make_services\n"
        "from vo_rag.mcp.server import build_mcp_server\n"
        "build_mcp_server(services=make_services(Path("
        + repr(str(calls))
        + "))).run(transport='stdio')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + str(root / "src")
    parameters = StdioServerParameters(command=sys.executable, args=[str(launcher)], env=env)
    async with stdio_client(parameters) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        listed = await session.list_tools()
        query_schema = next(
            tool.outputSchema for tool in listed.tools if tool.name == "vo_rag_query"
        )
        assert query_schema is not None and "results" in query_schema["properties"]
        result = await session.call_tool(
            "vo_rag_query",
            {
                "query": "retention?",
                "collections": ["vo_rag_brain_v1"],
                "top_k": 2,
                "rerank": False,
            },
        )
        assert not result.isError
        structured = result.structuredContent
        assert isinstance(structured, dict) and "results" in structured
        assert len(result.content) == 1 and result.content[0].type == "text"
        text = result.content[0].text
        rows = structured["results"]
        assert len(rows) == 2 and all(row["collection"] == "vo_rag_brain_v1" for row in rows)
        assert "Only laboratory edition 4" in rows[1]["payload"]["content"]
        assert "Only enterprise firmware 3" in text
        assert "Only laboratory edition 4" not in text
        assert "Complete evidence withheld" in text and len(text) <= 65536
        assert "wiki/retention.md@" + "a" * 40 in text
        assert "source_origin: upstream" in text
        before = calls.read_bytes()
        empty = await session.call_tool(
            "vo_rag_query", {"query": "retention?", "collections": [], "rerank": False}
        )
        assert empty.isError and calls.read_bytes() == before
    recorded = json.loads(calls.read_text())
    assert len(recorded) == 2
    assert all(call["collections"] == ["vo_rag_brain_v1"] for call in recorded)
    assert [call["top_k"] for call in recorded] == [2, 100]


@pytest.mark.parametrize("version", [2, None, "absent"])
def test_unsupported_context_text_never_becomes_legacy_prefix(version: object) -> None:
    from vo_rag.core.retrieval.search import SearchResult
    from vo_rag.mcp.tools import render_query_text

    hit = SearchResult(
        id="p",
        score=1.0,
        collection="vo_rag_brain_v1",
        payload={
            "content": "Bare value. " * 100 + "Only laboratory edition.",
            "governing_context_version": version,
            "governing_context_complete": True,
        },
    )
    if version == "absent":
        hit.payload.pop("governing_context_version")
        hit.payload.pop("governing_context_complete")
    text = render_query_text([hit])
    assert "governing evidence unavailable" in text and "Bare value" not in text

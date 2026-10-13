"""MCP lifespan and service ownership contracts using in-memory SDK streams."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import anyio
import pytest
from mcp.shared.exceptions import McpError
from mcp.shared.session import ProgressFnT
from mcp.types import CancelledNotification, CancelledNotificationParams, ClientNotification
from tests.mcp._fakes import FakeRetriever, make_fake_services, make_search_result

from mcp import ClientSession


@pytest.mark.asyncio
async def test_stdio_main_entrypoint_builds_and_closes_runtime_on_server_loop(
    tmp_path: Path,
) -> None:
    from mcp.client.stdio import stdio_client

    from mcp import StdioServerParameters

    root = Path(__file__).resolve().parents[2]
    events = tmp_path / "runtime-events.jsonl"
    launcher = tmp_path / "stdio_entrypoint.py"
    launcher.write_text(
        "import asyncio, json\n"
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "import vo_rag.mcp.server as entrypoint\n"
        "from tests.mcp._fakes import FakeRetriever, make_fake_services, make_search_result\n"
        f"events = Path({str(events)!r})\n"
        "real_build = entrypoint.build_mcp_server\n"
        "async def build(_settings):\n"
        "    services = make_fake_services(\n"
        "        retriever=FakeRetriever(scripted=[make_search_result()])\n"
        "    )\n"
        "    services.settings = SimpleNamespace(version='test')\n"
        "    event = {'event': 'build', 'loop': id(asyncio.get_running_loop())}\n"
        "    events.write_text(json.dumps(event) + '\\n')\n"
        "    return services\n"
        "async def close(_services):\n"
        "    with events.open('a') as stream:\n"
        "        event = {'event': 'close', 'loop': id(asyncio.get_running_loop())}\n"
        "        stream.write(json.dumps(event) + '\\n')\n"
        "def build_for_entrypoint(**kwargs):\n"
        "    return real_build(runtime_builder=build, close_services=close, **kwargs)\n"
        "entrypoint.build_mcp_server = build_for_entrypoint\n"
        "entrypoint.main(transport='stdio')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(root), str(root / "src")))
    parameters = StdioServerParameters(command=sys.executable, args=[str(launcher)], env=env)

    async with stdio_client(parameters) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("vo_rag_query", {"query": "runtime fixture"})
        assert result.isError is False
        assert result.structuredContent is not None
        assert result.structuredContent["results"][0]["id"] == "point-1"

    lifecycle = [json.loads(line) for line in events.read_text().splitlines()]
    assert [event["event"] for event in lifecycle] == ["build", "close"]
    assert lifecycle[0]["loop"] == lifecycle[1]["loop"]


@pytest.mark.asyncio
async def test_sdk_lifespan_initializes_once_persists_intake_and_closes_owned_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.core.models.envelope import IntakeSource
    from vo_rag.mcp.server import build_mcp_server

    saved: dict[str, Any] = {}
    progress: list[tuple[float, float | None, str | None]] = []
    cancellation_started = asyncio.Event()
    cancellation_ids: set[str] = set()
    runtime_services: list[Any] = []
    created_on: list[tuple[int, int]] = []
    closed_on: list[tuple[int, int]] = []
    build_count = 0

    async def insert(_pool: Any, envelope: Any) -> None:
        saved[str(envelope.id)] = envelope

    async def lookup(_pool: Any, envelope_id: str) -> Any:
        return saved.get(envelope_id)

    async def update(_pool: Any, envelope_id: str, status: Any) -> None:
        saved[envelope_id].status = status

    async def fail(_pool: Any, envelope_id: str, error: str) -> None:
        saved[envelope_id].status = "failed"
        saved[envelope_id].error = error

    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.insert_envelope", insert)
    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.get_envelope", lookup)
    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.update_envelope_status", update)
    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.fail_envelope", fail)

    async def fake_stream(envelope: Any, **_kwargs: Any):
        from vo_rag.core.models.envelope import EnvelopeStatus
        from vo_rag.pipelines.book_rag.events import IngestProgress
        from vo_rag.pipelines.book_rag.workflow import PipelineOutput

        yield IngestProgress(envelope_id=str(envelope.id), step="parse", detail="fixture")
        if str(envelope.id) in cancellation_ids:
            cancellation_started.set()
            await asyncio.Event().wait()
        yield PipelineOutput(
            source_id=str(envelope.id),
            status=EnvelopeStatus.COMPLETED,
            qdrant_point_ids=[],
            raw_point_ids=[],
            sterile_point_ids=[],
            smoke_decision=None,
            chunk_count=0,
        )

    async def ignore_progress(_progress: float, _total: float | None, _message: str | None) -> None:
        return None

    async def collect_progress(current: float, total: float | None, message: str | None) -> None:
        progress.append((current, total, message))

    monkeypatch.setattr("vo_rag.mcp.tools.stream_book_rag_ingest", fake_stream)

    async def build(_settings: Any) -> Any:
        nonlocal build_count
        build_count += 1
        created_on.append((id(asyncio.get_running_loop()), id(asyncio.current_task())))
        services = make_fake_services(retriever=FakeRetriever(scripted=[make_search_result()]))
        services.db_pool = object()
        services.settings = SimpleNamespace(version="test")
        runtime_services.append(services)
        return services

    async def close(_services: Any) -> None:
        closed_on.append((id(asyncio.get_running_loop()), id(asyncio.current_task())))

    server = build_mcp_server(runtime_builder=build, close_services=close)
    client_to_server_send, client_to_server_receive = anyio.create_memory_object_stream(0)
    server_to_client_send, server_to_client_receive = anyio.create_memory_object_stream(0)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(
            server._mcp_server.run,
            client_to_server_receive,
            server_to_client_send,
            server._mcp_server.create_initialization_options(),
        )
        async with ClientSession(server_to_client_receive, client_to_server_send) as session:
            await session.initialize()
            query = await session.call_tool("vo_rag_query", {"query": "trail braking"})
            assert not query.isError
            assert query.structuredContent is not None
            assert query.structuredContent["results"][0]["id"] == "point-1"

            accepted = await session.call_tool(
                "vo_rag_ingest",
                {"source_type": IntakeSource.TEXT_PROMPT.value, "content": "test intake"},
            )
            assert not accepted.isError
            intake = accepted.structuredContent
            assert isinstance(intake, dict) and intake["dispatch_status"] == "submitted"
            envelope_id = intake["envelope_id"]

            status = await session.call_tool("vo_rag_status", {"envelope_id": envelope_id})
            assert not status.isError
            assert status.structuredContent is not None
            assert status.structuredContent["envelope"]["envelope_id"] == envelope_id

            missing = await session.call_tool(
                "vo_rag_status", {"envelope_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV"}
            )
            assert missing.isError
            assert missing.structuredContent is not None
            assert missing.structuredContent["ok"] is False
            assert missing.structuredContent["code"] == "envelope_not_found"
            assert len(missing.content) == 1 and missing.content[0].type == "text"

            runtime_services[0].db_pool = None
            unavailable_intake = await session.call_tool(
                "vo_rag_ingest",
                {"source_type": IntakeSource.TEXT_PROMPT.value, "content": "unavailable store"},
            )
            assert unavailable_intake.isError
            assert unavailable_intake.structuredContent is not None
            assert unavailable_intake.structuredContent["code"] == "intake_unavailable"
            assert "database unavailable" not in unavailable_intake.structuredContent["error"]

            unavailable_status = await session.call_tool(
                "vo_rag_status", {"envelope_id": envelope_id}
            )
            assert unavailable_status.isError
            assert unavailable_status.structuredContent is not None
            assert unavailable_status.structuredContent["code"] == "envelope_status_unavailable"
            assert "database unavailable" not in unavailable_status.structuredContent["error"]
            runtime_services[0].db_pool = object()

            streamed = await session.call_tool(
                "vo_rag_ingest_stream",
                {"envelope_id": envelope_id},
                progress_callback=cast(ProgressFnT, collect_progress),
            )
            assert not streamed.isError
            assert streamed.structuredContent is not None
            assert streamed.structuredContent["dispatch_status"] == "streamed"
            assert progress == [(1, 8, "parse: fixture")]

            cancel_target = await session.call_tool(
                "vo_rag_ingest",
                {"source_type": IntakeSource.TEXT_PROMPT.value, "content": "cancel test"},
            )
            assert not cancel_target.isError and cancel_target.structuredContent is not None
            cancelled_envelope_id = cancel_target.structuredContent["envelope_id"]
            cancellation_ids.add(cancelled_envelope_id)
            request_id = session._request_id
            pending_stream = asyncio.create_task(
                session.call_tool(
                    "vo_rag_ingest_stream",
                    {"envelope_id": cancelled_envelope_id},
                    progress_callback=cast(ProgressFnT, ignore_progress),
                )
            )
            await asyncio.wait_for(cancellation_started.wait(), timeout=2)
            await session.send_notification(
                ClientNotification(
                    root=CancelledNotification(
                        params=CancelledNotificationParams(
                            requestId=request_id, reason="fixture cancellation"
                        )
                    )
                )
            )
            with pytest.raises(McpError):
                await asyncio.wait_for(pending_stream, timeout=2)
            assert saved[cancelled_envelope_id].status == "failed"

        task_group.cancel_scope.cancel()

    assert build_count == 1
    assert len(saved) == 2
    assert len(created_on) == len(closed_on) == 1
    assert created_on[0][0] == closed_on[0][0]


@pytest.mark.asyncio
async def test_inspection_mode_builds_tools_without_runtime_startup() -> None:
    from vo_rag.mcp.server import build_mcp_server

    called = False

    async def build(_settings: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("inspection must not initialize runtime services")

    server = build_mcp_server(inspection_only=True, runtime_builder=build)
    assert "vo_rag_query" in server._tool_manager._tools
    assert called is False


@pytest.mark.asyncio
async def test_injected_services_are_not_closed_by_server_lifespan() -> None:
    from vo_rag.mcp.server import build_mcp_server

    called = False

    async def close(_services: Any) -> None:
        nonlocal called
        called = True

    build_mcp_server(services=make_fake_services(), close_services=close)
    assert called is False


@pytest.mark.asyncio
async def test_runtime_closes_builder_services_if_callback_wiring_fails() -> None:
    from vo_rag.mcp.runtime import build_runtime

    class SlottedServices:
        __slots__ = ("settings",)

        def __init__(self) -> None:
            self.settings = SimpleNamespace()

    services = SlottedServices()
    closed: list[Any] = []

    async def build(_settings: Any) -> Any:
        return services

    async def close(value: Any) -> None:
        closed.append(value)

    with pytest.raises(AttributeError):
        await build_runtime(
            settings=services.settings,
            service_builder=build,
            close_services=close,
        )
    assert closed == [services]


@pytest.mark.asyncio
async def test_runtime_status_reports_partial_capability_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.mcp.runtime import build_runtime

    monkeypatch.delenv("VORAG_API_URL", raising=False)
    monkeypatch.delenv("VORAG_API_KEY", raising=False)
    services = make_fake_services()
    services.db_pool = None
    services.retriever = None
    services.garage = None
    services.embedding_client = None

    async def build(_settings: Any) -> Any:
        return services

    async def close(_services: Any) -> None:
        return None

    runtime = await build_runtime(
        settings=services.settings,
        service_builder=build,
        close_services=close,
    )
    status = await services.status_provider()
    assert status["availability"] == "configured_dependencies"
    assert status["capabilities"]["retrieval"]["ready"] is False
    assert status["capabilities"]["intake"]["ready"] is False
    assert status["capabilities"]["workflow_streaming"]["ready"] is False
    assert "acquisition" not in status["capabilities"]
    assert "not a live health probe" in status["warnings"][0]
    await runtime.close(runtime.services)


@pytest.mark.asyncio
async def test_stream_cancellation_marks_envelope_failed_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from vo_rag.mcp.tools import vo_rag_ingest_stream_impl

    envelope_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    events: list[tuple[str, str]] = []

    async def lookup(_pool: Any, _envelope_id: str) -> Any:
        return SimpleNamespace(id=envelope_id, source_type="text_prompt")

    async def fail(_pool: Any, identifier: str, error: str) -> None:
        events.append((identifier, error))

    async def cancelled_stream(*_args: Any, **_kwargs: Any):
        raise asyncio.CancelledError
        yield

    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.get_envelope", lookup)
    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.fail_envelope", fail)
    monkeypatch.setattr("vo_rag.mcp.tools.stream_book_rag_ingest", cancelled_stream)
    services = make_fake_services()
    services.db_pool = object()

    with pytest.raises(asyncio.CancelledError):
        await vo_rag_ingest_stream_impl(
            services=services,
            envelope_id=envelope_id,
            ctx=SimpleNamespace(report_progress=lambda *_args, **_kwargs: None),
        )
    assert events == [(envelope_id, "CancelledError")]

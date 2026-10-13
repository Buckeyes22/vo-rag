"""SSE ingest stream route cancellation regressions."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from starlette.requests import Request
from starlette.types import Scope

from vo_rag.api.routes import ingest
from vo_rag.core.models.envelope import EnvelopeStatus
from vo_rag.pipelines.book_rag.events import IngestProgress
from vo_rag.pipelines.book_rag.workflow import PipelineOutput


def _fake_request(services: object) -> Request:
    app = SimpleNamespace(state=SimpleNamespace(surface_services=services))
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/v1/intake/env-1/stream",
        "raw_path": b"/api/v1/intake/env-1/stream",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "app": app,
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_api_stream_proxy_threads_runner_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.pipelines.book_rag import runner as runner_mod

    envelope = object()
    settings = object()
    pg_pool = object()
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def fake_runner_stream(*args: Any, **kwargs: Any) -> AsyncIterator[IngestProgress]:
        calls.append((args, kwargs))
        yield IngestProgress(envelope_id="env-1", step="parse", detail="done")

    monkeypatch.setattr(runner_mod, "stream_book_rag_ingest", fake_runner_stream)

    events = [
        ev
        async for ev in ingest.stream_book_rag_ingest(
            envelope,
            settings=settings,
            pg_pool=pg_pool,
        )
    ]

    assert len(events) == 1
    assert calls == [((envelope,), {"settings": settings, "pg_pool": pg_pool})]


@pytest.mark.asyncio
async def test_stream_disconnect_does_not_cancel_ingest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    cancelled = False

    async def fake_stream(*args: Any, **kwargs: Any) -> AsyncIterator[IngestProgress]:
        nonlocal cancelled
        started.set()
        try:
            await release.wait()
            yield IngestProgress(envelope_id="env-1", step="parse", detail="done")
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            finished.set()

    async def fake_get_envelope(pool: object, envelope_id: str) -> object:
        assert pool is not None
        assert envelope_id == "env-1"
        return SimpleNamespace(id=envelope_id)

    monkeypatch.setattr(ingest, "stream_book_rag_ingest", fake_stream)
    monkeypatch.setattr(ingest, "get_envelope", fake_get_envelope)

    services = SimpleNamespace(
        db_pool=object(),
        db_pool_unavailable_reason="",
        settings=SimpleNamespace(),
    )

    response = await ingest.stream_intake("env-1", _fake_request(services))
    body_iterator = cast(AsyncIterator[object], response.body_iterator)
    task: asyncio.Future[object] = asyncio.ensure_future(anext(body_iterator))

    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    try:
        assert cancelled is False
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), timeout=1.0)


@pytest.mark.asyncio
async def test_stream_disconnect_persists_background_ingest_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    failed = asyncio.Event()
    failures: list[tuple[object, str, str]] = []
    errors: list[dict[str, Any]] = []

    async def fake_stream(*args: Any, **kwargs: Any) -> AsyncIterator[IngestProgress]:
        started.set()
        await release.wait()
        raise RuntimeError("background ingest failed")
        yield  # pragma: no cover

    async def fake_get_envelope(pool: object, envelope_id: str) -> object:
        assert pool is not None
        assert envelope_id == "env-1"
        return SimpleNamespace(id=envelope_id)

    async def fake_fail_envelope(pool: object, envelope_id: str, error: str) -> None:
        failures.append((pool, envelope_id, error))
        failed.set()

    def fake_log_error(event: str, **kwargs: Any) -> None:
        errors.append({"event": event, **kwargs})

    monkeypatch.setattr(ingest, "stream_book_rag_ingest", fake_stream)
    monkeypatch.setattr(ingest, "get_envelope", fake_get_envelope)
    monkeypatch.setattr(ingest, "fail_envelope", fake_fail_envelope, raising=False)
    monkeypatch.setattr(ingest.log, "error", fake_log_error)

    services = SimpleNamespace(
        db_pool=object(),
        db_pool_unavailable_reason="",
        settings=SimpleNamespace(),
    )

    response = await ingest.stream_intake("env-1", _fake_request(services))
    body_iterator = cast(AsyncIterator[object], response.body_iterator)
    task: asyncio.Future[object] = asyncio.ensure_future(anext(body_iterator))

    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    release.set()
    await asyncio.wait_for(failed.wait(), timeout=1.0)

    assert failures == [(services.db_pool, "env-1", "background ingest failed")]
    assert errors == [
        {
            "event": "book_rag_sse_ingest_failed",
            "envelope_id": "env-1",
            "error": "background ingest failed",
            "exc_info": True,
        }
    ]


@pytest.mark.asyncio
async def test_stream_missing_terminal_output_fails_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failures: list[tuple[object, str, str]] = []

    async def fake_stream(*args: Any, **kwargs: Any) -> AsyncIterator[IngestProgress]:
        yield IngestProgress(envelope_id="env-1", step="parse", detail="done")

    async def fake_get_envelope(pool: object, envelope_id: str) -> object:
        assert pool is not None
        assert envelope_id == "env-1"
        return SimpleNamespace(id=envelope_id)

    async def fake_fail_envelope(pool: object, envelope_id: str, error: str) -> None:
        failures.append((pool, envelope_id, error))

    monkeypatch.setattr(ingest, "stream_book_rag_ingest", fake_stream)
    monkeypatch.setattr(ingest, "get_envelope", fake_get_envelope)
    monkeypatch.setattr(ingest, "fail_envelope", fake_fail_envelope, raising=False)

    services = SimpleNamespace(
        db_pool=object(),
        db_pool_unavailable_reason="",
        settings=SimpleNamespace(),
    )

    response = await ingest.stream_intake("env-1", _fake_request(services))
    body_iterator = cast(AsyncIterator[object], response.body_iterator)

    first = cast(dict[str, str], await asyncio.wait_for(anext(body_iterator), timeout=1.0))
    assert first == {
        "event": "parse",
        "data": '{"envelope_id":"env-1","step":"parse","detail":"done","pct":null}',
    }
    with pytest.raises(RuntimeError, match="ingest stream completed without output"):
        await asyncio.wait_for(anext(body_iterator), timeout=1.0)

    assert failures == [(services.db_pool, "env-1", "ingest stream completed without output")]


@pytest.mark.asyncio
async def test_stream_cancelled_workflow_fails_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failures: list[tuple[object, str, str]] = []

    async def fake_stream(*args: Any, **kwargs: Any) -> AsyncIterator[IngestProgress]:
        yield IngestProgress(envelope_id="env-1", step="parse", detail="started")
        raise asyncio.CancelledError("workflow cancelled")

    async def fake_get_envelope(pool: object, envelope_id: str) -> object:
        assert pool is not None
        assert envelope_id == "env-1"
        return SimpleNamespace(id=envelope_id)

    async def fake_fail_envelope(pool: object, envelope_id: str, error: str) -> None:
        failures.append((pool, envelope_id, error))

    monkeypatch.setattr(ingest, "stream_book_rag_ingest", fake_stream)
    monkeypatch.setattr(ingest, "get_envelope", fake_get_envelope)
    monkeypatch.setattr(ingest, "fail_envelope", fake_fail_envelope, raising=False)

    services = SimpleNamespace(
        db_pool=object(),
        db_pool_unavailable_reason="",
        settings=SimpleNamespace(),
    )

    response = await ingest.stream_intake("env-1", _fake_request(services))
    body_iterator = cast(AsyncIterator[object], response.body_iterator)

    first = cast(dict[str, str], await asyncio.wait_for(anext(body_iterator), timeout=1.0))
    assert first["event"] == "parse"
    with pytest.raises(asyncio.CancelledError, match="workflow cancelled"):
        await asyncio.wait_for(anext(body_iterator), timeout=1.0)

    assert failures == [(services.db_pool, "env-1", "workflow cancelled")]


@pytest.mark.asyncio
async def test_stream_failure_surfaces_when_fail_envelope_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_stream(*args: Any, **kwargs: Any) -> AsyncIterator[IngestProgress]:
        yield IngestProgress(envelope_id="env-1", step="parse", detail="started")
        raise RuntimeError("pipeline failed")

    async def fake_get_envelope(pool: object, envelope_id: str) -> object:
        assert pool is not None
        assert envelope_id == "env-1"
        return SimpleNamespace(id=envelope_id)

    async def fake_fail_envelope(pool: object, envelope_id: str, error: str) -> None:
        assert error == "pipeline failed"
        raise RuntimeError("persist failed")

    monkeypatch.setattr(ingest, "stream_book_rag_ingest", fake_stream)
    monkeypatch.setattr(ingest, "get_envelope", fake_get_envelope)
    monkeypatch.setattr(ingest, "fail_envelope", fake_fail_envelope, raising=False)

    services = SimpleNamespace(
        db_pool=object(),
        db_pool_unavailable_reason="",
        settings=SimpleNamespace(),
    )

    response = await ingest.stream_intake("env-1", _fake_request(services))
    body_iterator = cast(AsyncIterator[object], response.body_iterator)

    first = cast(dict[str, str], await asyncio.wait_for(anext(body_iterator), timeout=1.0))
    assert first["event"] == "parse"
    with pytest.raises(RuntimeError, match="pipeline failed"):
        await asyncio.wait_for(anext(body_iterator), timeout=1.0)


@pytest.mark.asyncio
async def test_stream_disconnect_persists_background_ingest_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    completed = asyncio.Event()
    updates: list[tuple[object, str, EnvelopeStatus]] = []

    async def fake_stream(*args: Any, **kwargs: Any) -> AsyncIterator[PipelineOutput]:
        started.set()
        await release.wait()
        yield PipelineOutput(
            source_id="source-1",
            status=EnvelopeStatus.COMPLETED,
            qdrant_point_ids=[],
            raw_point_ids=[],
            sterile_point_ids=[],
            smoke_decision=None,
            chunk_count=0,
        )

    async def fake_get_envelope(pool: object, envelope_id: str) -> object:
        assert pool is not None
        assert envelope_id == "env-1"
        return SimpleNamespace(id=envelope_id)

    async def fake_update_envelope_status(
        pool: object,
        envelope_id: str,
        status: EnvelopeStatus,
    ) -> None:
        updates.append((pool, envelope_id, status))
        completed.set()

    monkeypatch.setattr(ingest, "stream_book_rag_ingest", fake_stream)
    monkeypatch.setattr(ingest, "get_envelope", fake_get_envelope)
    monkeypatch.setattr(ingest, "update_envelope_status", fake_update_envelope_status)

    services = SimpleNamespace(
        db_pool=object(),
        db_pool_unavailable_reason="",
        settings=SimpleNamespace(),
    )

    response = await ingest.stream_intake("env-1", _fake_request(services))
    body_iterator = cast(AsyncIterator[object], response.body_iterator)
    task: asyncio.Future[object] = asyncio.ensure_future(anext(body_iterator))

    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    release.set()
    await asyncio.wait_for(completed.wait(), timeout=1.0)

    assert updates == [(services.db_pool, "env-1", EnvelopeStatus.COMPLETED)]


@pytest.mark.asyncio
async def test_stream_success_status_persist_failure_does_not_fail_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failures: list[tuple[object, str, str]] = []
    errors: list[dict[str, Any]] = []

    async def fake_stream(*args: Any, **kwargs: Any) -> AsyncIterator[PipelineOutput]:
        yield PipelineOutput(
            source_id="source-1",
            status=EnvelopeStatus.COMPLETED,
            qdrant_point_ids=[],
            raw_point_ids=[],
            sterile_point_ids=[],
            smoke_decision=None,
            chunk_count=0,
            quality_warnings=[{"stage": "parse", "code": "low_page_coverage"}],
        )

    async def fake_get_envelope(pool: object, envelope_id: str) -> object:
        assert pool is not None
        assert envelope_id == "env-1"
        return SimpleNamespace(id=envelope_id)

    async def fake_update_envelope_status(
        pool: object,
        envelope_id: str,
        status: EnvelopeStatus,
    ) -> None:
        raise RuntimeError("success persist failed")

    async def fake_fail_envelope(pool: object, envelope_id: str, error: str) -> None:
        failures.append((pool, envelope_id, error))

    def fake_log_error(event: str, **kwargs: Any) -> None:
        errors.append({"event": event, **kwargs})

    monkeypatch.setattr(ingest, "stream_book_rag_ingest", fake_stream)
    monkeypatch.setattr(ingest, "get_envelope", fake_get_envelope)
    monkeypatch.setattr(ingest, "update_envelope_status", fake_update_envelope_status)
    monkeypatch.setattr(ingest, "fail_envelope", fake_fail_envelope, raising=False)
    monkeypatch.setattr(ingest.log, "error", fake_log_error)

    services = SimpleNamespace(
        db_pool=object(),
        db_pool_unavailable_reason="",
        settings=SimpleNamespace(),
    )

    response = await ingest.stream_intake("env-1", _fake_request(services))
    body_iterator = cast(AsyncIterator[object], response.body_iterator)

    item = cast(dict[str, str], await asyncio.wait_for(anext(body_iterator), timeout=1.0))
    assert item["event"] == "done"
    body = json.loads(item["data"])
    assert body["status"] == EnvelopeStatus.COMPLETED.value
    assert body["quality_warnings"] == [{"stage": "parse", "code": "low_page_coverage"}]

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(body_iterator), timeout=1.0)

    assert failures == []
    assert errors == [
        {
            "event": "book_rag_sse_success_persist_failed",
            "envelope_id": "env-1",
            "status": EnvelopeStatus.COMPLETED.value,
            "error": "success persist failed",
            "exc_info": True,
        }
    ]

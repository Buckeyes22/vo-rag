from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pytest

from tests.api.conftest import make_services
from vo_rag.api import operational_sampler as sampler
from vo_rag.core.observability.metrics import (
    intake_envelopes,
    last_completion_age_seconds,
    oldest_claimable_age_seconds,
)


class _Connection:
    def __init__(self, row: dict[str, Any] | None = None, error: Exception | None = None):
        self.row = row
        self.error = error

    async def fetchrow(self, _query: str) -> dict[str, Any] | None:
        if self.error is not None:
            raise self.error
        return self.row


class _Pool:
    def __init__(self, connection: _Connection):
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):  # noqa: ANN201
        yield self.connection


async def _ok(_services: Any, _settings: Any = None) -> dict[str, bool]:
    return {"ok": True}


@pytest.mark.asyncio
async def test_sampler_uses_frozen_clock_and_exports_queue_ages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.api.routes import system

    frozen = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)
    observed: list[datetime] = []
    monkeypatch.setattr(system, "_db_pool_health", _ok)
    monkeypatch.setattr(system, "_qdrant_health", _ok)
    monkeypatch.setattr(system, "_embedding_health", lambda _settings: _ok(None))
    monkeypatch.setattr(system, "_reranker_health", _ok)

    def dispatcher_health(*, now: datetime) -> dict[str, object]:
        observed.append(now)
        return {"ok": True, "age_seconds": 4.5}

    monkeypatch.setattr(sampler, "dispatcher_health", dispatcher_health)
    pool = _Pool(
        _Connection(
            {
                "received_count": 3,
                "processing_count": 1,
                "oldest_claimable_age_seconds": 42.0,
                "last_completion_age_seconds": 12.0,
            }
        )
    )
    result = await sampler.sample_operational_metrics(
        make_services(db_pool=pool), object(), now=frozen
    )
    assert observed == [frozen]
    assert result == {"query": True, "intake": True}
    assert intake_envelopes.labels(status="received")._value.get() == 3  # noqa: SLF001
    assert oldest_claimable_age_seconds._value.get() == 42  # noqa: SLF001
    assert last_completion_age_seconds._value.get() == 12  # noqa: SLF001


@pytest.mark.asyncio
async def test_sampler_fails_closed_when_queue_query_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.api.routes import system

    monkeypatch.setattr(system, "_db_pool_health", _ok)
    monkeypatch.setattr(system, "_qdrant_health", _ok)
    monkeypatch.setattr(system, "_embedding_health", lambda _settings: _ok(None))
    monkeypatch.setattr(system, "_reranker_health", _ok)
    monkeypatch.setattr(
        sampler,
        "dispatcher_health",
        lambda *, now: {"ok": True, "age_seconds": 1.0},
    )
    result = await sampler.sample_operational_metrics(
        make_services(db_pool=_Pool(_Connection(error=RuntimeError("db unavailable")))),
        object(),
    )
    assert result == {"query": False, "intake": False}
    assert intake_envelopes.labels(status="received")._value.get() == -1  # noqa: SLF001

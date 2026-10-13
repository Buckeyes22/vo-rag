from __future__ import annotations

from collections.abc import Coroutine
from typing import Never

import pytest

from vo_rag.integrations.llamaindex._async import run_coro_sync


def test_run_coro_sync_runs_factory_without_active_loop() -> None:
    async def _value() -> str:
        return "ok"

    assert run_coro_sync(lambda: _value(), component="test") == "ok"


@pytest.mark.asyncio
async def test_run_coro_sync_fails_loud_inside_running_loop_without_creating_coroutine() -> None:
    created = False

    async def _value() -> Never:
        raise AssertionError("should not run")

    def _factory() -> Coroutine[object, object, Never]:
        nonlocal created
        created = True
        return _value()

    with pytest.raises(RuntimeError, match="Use the adapter's async API instead"):
        run_coro_sync(_factory, component="test.sync")

    assert created is False

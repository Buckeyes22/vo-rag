"""Sync/async bridge for LlamaIndex adapter hooks.

Several LlamaIndex interfaces still expose synchronous hooks while the
vo-rag seams they wrap are async-native. The single helper below is
used by every adapter that has to call an async vo-rag function from a
sync LlamaIndex API.

The bridge is deliberately fail-loud: if it is called from inside a
running event loop it raises ``RuntimeError`` rather than nesting
``asyncio.run(...)``. Adapters running under an async host (FastAPI,
Workflows, notebooks, ...) must use the adapter's async API directly
(``aretrieve`` / ``_aretrieve`` / ``asynthesize`` / ...) so cancellation,
tracing, and request-scoped resources stay predictable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine


def run_coro_sync[T](
    factory: Callable[[], Coroutine[object, object, T]],
    *,
    component: str,
) -> T:
    """Run a coroutine from a sync LlamaIndex hook only when no loop is active.

    ``factory`` is intentionally lazy: it prevents creating an un-awaited
    coroutine when the caller is already inside an async host. FastAPI,
    Workflows, notebooks, and other async hosts must call the adapter's
    async API (``aretrieve``, ``_aretrieve``, ``asynthesize``, etc.)
    directly.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    raise RuntimeError(
        f"{component} cannot use its synchronous LlamaIndex API inside a running "
        "event loop. Use the adapter's async API instead."
    )

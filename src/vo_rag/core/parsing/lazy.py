"""Defer parser backend construction until a format actually needs it (VOR-1405).

``build_parser_router`` constructed Docling and EPUB (plus, before VOR-1427
dropped them, two undeployed OCR-adapter clients) up front, then
``ParserRouter`` dispatched by format and typically used one of them.
Measured on localhost 2026-08-14 that eager build cost **9.45s per
document** (import of the docling stack plus construction), against 0.1ms to
actually parse a Markdown file — so a text-format ingest spent essentially all
of its time building a parser it never called.

Wrapping each backend here keeps ``ParserRouter`` and every call site
unchanged: attribute access builds on demand, and the result is memoised.

``aclose`` is deliberately a **real** attribute rather than something reached
through ``__getattr__``. ``runner._close_quietly`` looks a closer up with
``getattr(obj, "aclose")``, so a pure ``__getattr__`` proxy would construct the
backend purely in order to close it and hand the whole cost back at teardown.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["LazyParser"]


class LazyParser:
    """Build ``build()`` on first attribute access, then delegate to it."""

    __slots__ = ("_backend", "_build")

    def __init__(self, build: Callable[[], Any]) -> None:
        self._build = build
        self._backend: Any | None = None

    @property
    def is_built(self) -> bool:
        """Whether the backend has been constructed yet."""
        return self._backend is not None

    def unwrap(self) -> Any:
        """Return the backend, constructing it if necessary."""
        if self._backend is None:
            self._backend = self._build()
        return self._backend

    async def aclose(self) -> None:
        """Close the backend, but only if it was ever built."""
        backend = self._backend
        if backend is None:
            return
        close = getattr(backend, "aclose", None)
        if close is None:
            return
        result = close()
        if hasattr(result, "__await__"):
            await result

    def __getattr__(self, name: str) -> Any:
        # Only reached for names that are not slots/class attributes, so
        # is_built / unwrap / aclose never trigger construction.
        return getattr(self.unwrap(), name)

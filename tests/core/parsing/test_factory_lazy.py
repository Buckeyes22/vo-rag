"""build_parser_router must not construct backends it may never use (VOR-1405).

Eager construction cost 9.45s per document on localhost (2026-08-14), paid
by every ingest regardless of format. ParserRouter dispatches by format, so a
Markdown/TXT ingest never touches Docling and an EPUB ingest never touches
Docling.
"""

from __future__ import annotations

from types import SimpleNamespace

from vo_rag.core.parsing.factory import build_parser_router
from vo_rag.core.parsing.lazy import LazyParser

_BACKENDS = ("docling", "epub")


def test_no_backend_is_constructed_when_the_router_is_built() -> None:
    router = build_parser_router(SimpleNamespace(), SimpleNamespace())

    for name in _BACKENDS:
        backend = getattr(router, name)
        assert isinstance(backend, LazyParser), f"{name} should be wrapped"
        assert backend.is_built is False, f"{name} must not be constructed eagerly"

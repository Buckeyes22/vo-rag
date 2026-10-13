"""LazyParser — defer parser construction until a format actually needs it (VOR-1405).

``build_parser_router`` used to construct Docling and EPUB (plus, before
VOR-1427 dropped them, two undeployed OCR-adapter clients) eagerly. Measured
on localhost 2026-08-14 that cost **9.45s per document**, and
``ParserRouter`` then dispatches by format — so a Markdown or TXT ingest
paid all of it for a Docling parser it never called.

The subtle requirement is the close path: ``runner._close_quietly`` does
``getattr(obj, "aclose")``. A naive ``__getattr__`` proxy would construct the
parser purely in order to close it, which would reintroduce the whole cost at
teardown. ``aclose`` must therefore be a real attribute.
"""

from __future__ import annotations

from typing import Any

import pytest

from vo_rag.core.parsing.lazy import LazyParser


class _Backend:
    def __init__(self) -> None:
        self.closed = False

    async def parse(self, path: str) -> str:
        return f"parsed:{path}"

    async def aclose(self) -> None:
        self.closed = True


def test_construction_is_deferred_until_first_attribute_access() -> None:
    builds: list[int] = []

    def build() -> _Backend:
        builds.append(1)
        return _Backend()

    lazy = LazyParser(build)

    assert builds == [], "must not build at wrap time"
    assert lazy.is_built is False

    _ = lazy.parse

    assert builds == [1]
    assert lazy.is_built is True


def test_backend_is_built_only_once() -> None:
    builds: list[int] = []

    def build() -> _Backend:
        builds.append(1)
        return _Backend()

    lazy = LazyParser(build)

    _ = lazy.parse
    _ = lazy.parse
    _ = lazy.parse

    assert builds == [1]


@pytest.mark.asyncio
async def test_attribute_access_delegates_to_the_real_backend() -> None:
    lazy = LazyParser(_Backend)

    assert await lazy.parse("book.pdf") == "parsed:book.pdf"


@pytest.mark.asyncio
async def test_aclose_on_an_unbuilt_parser_does_not_construct_it() -> None:
    """The whole point: tearing down a router must not build what it never used."""
    builds: list[int] = []

    def build() -> _Backend:
        builds.append(1)
        return _Backend()

    lazy = LazyParser(build)

    await lazy.aclose()

    assert builds == [], "aclose must not trigger construction"
    assert lazy.is_built is False


@pytest.mark.asyncio
async def test_aclose_delegates_when_the_parser_was_built() -> None:
    backend = _Backend()
    lazy = LazyParser(lambda: backend)

    _ = lazy.parse  # force construction
    await lazy.aclose()

    assert backend.closed is True


@pytest.mark.asyncio
async def test_aclose_is_safe_when_the_backend_has_no_aclose() -> None:
    class _NoClose:
        value = 1

    lazy = LazyParser(_NoClose)
    _ = lazy.value

    await lazy.aclose()  # must not raise


def test_getattr_on_a_missing_attribute_still_raises() -> None:
    lazy = LazyParser(_Backend)

    with pytest.raises(AttributeError):
        _ = lazy.does_not_exist


def test_unwrap_exposes_the_backend_for_callers_that_need_the_real_object() -> None:
    backend = _Backend()
    lazy = LazyParser(lambda: backend)

    assert lazy.unwrap() is backend
    assert lazy.is_built is True


def test_build_errors_propagate_and_do_not_cache_a_broken_backend() -> None:
    calls: list[int] = []

    def build() -> Any:
        calls.append(1)
        raise RuntimeError("backend unavailable")

    lazy = LazyParser(build)

    with pytest.raises(RuntimeError, match="backend unavailable"):
        _ = lazy.parse
    with pytest.raises(RuntimeError, match="backend unavailable"):
        _ = lazy.parse

    assert calls == [1, 1], "a failed build must be retried, not cached"
    assert lazy.is_built is False

"""Parser-router factory wiring."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from vo_rag.core.parsing.router import ParserRouter


class _FakeDoclingParser:
    def __init__(self, figures: Any, *, do_ocr: bool = True) -> None:
        self.figures = figures
        self.do_ocr = do_ocr


class _FakeEpubParser:
    def __init__(self, figures: Any) -> None:
        self.figures = figures


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        garage=SimpleNamespace(buckets=SimpleNamespace(artifacts="settings-artifacts")),
    )


def test_build_parser_router_uses_layout_only_docling_and_configured_parsers(
    monkeypatch,
) -> None:
    import vo_rag.core.parsing.parsers.docling as docling_mod
    import vo_rag.core.parsing.parsers.epub as epub_mod
    from vo_rag.core.parsing.factory import build_parser_router

    monkeypatch.setattr(docling_mod, "DoclingParser", _FakeDoclingParser)
    monkeypatch.setattr(epub_mod, "EpubParser", _FakeEpubParser)

    figures = object()

    router = build_parser_router(_settings(), figures)

    assert isinstance(router, ParserRouter)
    # VOR-1405: backends are LazyParser-wrapped and built on first use, so each
    # is unwrapped to a local before assertions. Unwrapping is also what proves
    # construction still wires the settings/env config through correctly.
    docling = router.docling.unwrap()  # type: ignore[attr-defined]
    epub = router.epub.unwrap()  # type: ignore[attr-defined]

    assert isinstance(docling, _FakeDoclingParser)
    assert docling.figures is figures
    assert docling.do_ocr is False
    assert isinstance(epub, _FakeEpubParser)
    assert epub.figures is figures


def test_build_parser_router_can_pin_the_docling_device(monkeypatch) -> None:
    """``rechunk-preview --reparse`` pins docling to CUDA through the factory
    itself, so its re-parse is the ingest's router."""
    import vo_rag.core.parsing.parsers.docling as docling_mod
    from vo_rag.core.parsing.factory import build_parser_router

    seen: dict[str, Any] = {}

    class _Recording:
        def __init__(self, figures: Any, **kwargs: Any) -> None:
            seen.update(kwargs, figures=figures)

    monkeypatch.setattr(docling_mod, "DoclingParser", _Recording)
    figures = object()
    build_parser_router(_settings(), figures, docling_device="cuda").docling.unwrap()  # type: ignore[attr-defined]
    assert seen == {"figures": figures, "do_ocr": False, "device": "cuda"}


# ---------------------------------------------------------------------------
# VOR-1427: the router gets a Garage client + bucket so it can read
# the OCR farm's cached ``ocr/<sha256>/docling.json``.
# ---------------------------------------------------------------------------


def test_build_parser_router_passes_garage_bucket_and_figures_from_settings() -> None:
    from vo_rag.core.parsing.factory import build_parser_router
    from vo_rag.core.storage.garage import GarageClient

    settings = _settings()
    settings.garage = SimpleNamespace(
        endpoint="http://garage:3900",
        access_key="ak",
        secret_key="sk",  # pragma: allowlist secret
        region="garage",
        buckets=SimpleNamespace(artifacts="settings-artifacts"),
    )
    figures = object()

    router = build_parser_router(settings, figures)

    assert isinstance(router.garage, GarageClient)
    assert router.ocr_bucket == "settings-artifacts"
    assert router.figures is figures


def test_build_parser_router_without_garage_credentials_has_no_cache() -> None:
    from vo_rag.core.parsing.factory import build_parser_router

    router = build_parser_router(_settings(), object())

    assert router.garage is None
    assert router.ocr_bucket is None


def test_build_parser_router_uses_explicit_garage() -> None:
    from vo_rag.core.parsing.factory import build_parser_router

    garage = object()
    figures = object()
    router = build_parser_router(_settings(), figures, garage=garage, bucket="b")

    assert router.garage is garage
    assert router.ocr_bucket == "b"
    assert router.figures is figures

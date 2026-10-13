"""Parser-router factory for production ingest workers.

The router stays dependency-injected for tests; this module is the production
construction seam for the book_rag parser wiring, built from the vo-rag
parser set.
"""

from __future__ import annotations

from typing import Any, cast

from vo_rag.core.parsing.lazy import LazyParser
from vo_rag.core.parsing.router import ParserRouter


def build_parser_router(
    settings: Any,
    figures: Any,
    *,
    garage: Any | None = None,
    bucket: str | None = None,
    docling_device: str | None = None,
) -> ParserRouter:
    """Construct the book_rag parser router from ``settings`` and env overrides.

    The router also gets a Garage client and the artifacts bucket so scanned
    PDFs / image-only EPUBs can use the OCR farm's cached
    ``ocr/<sha256>/docling.json`` (VOR-1427). ``garage`` defaults to a
    client built from ``settings.garage`` when its credentials are set (the
    same rule as ``api.singletons``); without one the cache is off.

    Native PDFs use Docling layout extraction with OCR disabled; scanned PDFs
    use the OCR farm's cache when one exists, else docling's own GPU OCR
    (``DoclingParser.parse_scanned``). The vendor-SDK forward path and its
    accelerated-OCR challenger fallback that used to sit here were never
    deployed and are gone (VOR-1427).

    ``docling_device`` pins docling's device ("cuda"); ``None`` keeps the
    worker default (``VORAG_DOCLING_DEVICE``, else auto). ``corpus
    rechunk-preview --reparse`` pins "cuda" so its re-parse is this exact
    router, never a CPU run (VOR-1452).

    Backends are wrapped in :class:`LazyParser` and built on first use. The
    router dispatches by format, so eager construction charged every ingest for
    parsers it would never call -- 9.45s per document, measured 2026-08-14,
    against 0.1ms to parse a Markdown file (VOR-1405). Configuration is still
    resolved here so a bad setting fails at build time, not mid-parse.
    """

    # Imports live inside the builders: importing the docling stack is itself a
    # large part of the cost, so deferring construction without deferring the
    # import would save very little.
    def _build_docling() -> Any:
        from vo_rag.core.parsing.parsers.docling import DoclingParser

        if docling_device is None:
            return DoclingParser(figures, do_ocr=False)
        return DoclingParser(figures, do_ocr=False, device=docling_device)

    def _build_epub() -> Any:
        from vo_rag.core.parsing.parsers.epub import EpubParser

        return EpubParser(figures)

    garage_client = garage if garage is not None else _garage_from_settings(settings)
    return ParserRouter(
        docling=cast("Any", LazyParser(_build_docling)),
        epub=cast("Any", LazyParser(_build_epub)),
        garage=garage_client,
        bucket=(bucket or _artifacts_bucket(settings)) if garage_client is not None else None,
        figures=figures,
    )


def _garage_from_settings(settings: Any) -> Any | None:
    garage_settings = getattr(settings, "garage", None)
    endpoint = getattr(garage_settings, "endpoint", "")
    access_key = getattr(garage_settings, "access_key", "")
    secret_key = getattr(garage_settings, "secret_key", "")
    if not (endpoint and access_key and secret_key):
        return None
    from vo_rag.core.storage.garage import GarageClient

    return GarageClient(
        endpoint_url=str(endpoint),
        access_key=str(access_key),
        secret_key=str(secret_key),
        region=str(getattr(garage_settings, "region", "garage") or "garage"),
    )


def _artifacts_bucket(settings: Any) -> str:
    buckets = getattr(getattr(settings, "garage", None), "buckets", None)
    return str(getattr(buckets, "artifacts", "vorag-artifacts"))


__all__ = ["build_parser_router"]

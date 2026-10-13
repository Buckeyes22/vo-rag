"""Smoke test for the parsers subpackage.

Verifies that the dropped dead parsers (olmocr, mineru) are not present in
the new layout, and that the active parsers package is importable.
"""

from __future__ import annotations


def test_parsers_package_imports() -> None:
    from vo_rag.core.parsing import parsers

    assert parsers is not None


def test_dead_olmocr_parser_dropped() -> None:
    """olmOCR is SUPERSEDED by docling; do not re-add."""
    import importlib

    try:
        importlib.import_module("vo_rag.core.parsing.parsers.olmocr")
    except ModuleNotFoundError:
        return
    raise AssertionError("vo_rag.core.parsing.parsers.olmocr must not be present")


def test_dead_mineru_parser_dropped() -> None:
    """MinerU is DEPRECATED in favor of docling; do not re-add."""
    import importlib

    try:
        importlib.import_module("vo_rag.core.parsing.parsers.mineru")
    except ModuleNotFoundError:
        return
    raise AssertionError("vo_rag.core.parsing.parsers.mineru must not be present")


def test_dead_glm_ocr_parser_dropped() -> None:
    """The vendor-SDK OCR client was never deployed; do not re-add (VOR-1427)."""
    import importlib

    try:
        importlib.import_module("vo_rag.core.parsing.parsers.glm_ocr")
    except ModuleNotFoundError:
        return
    raise AssertionError("vo_rag.core.parsing.parsers.glm_ocr must not be present")


def test_dead_paddleocr_vl_parser_dropped() -> None:
    """The accelerated-OCR challenger client was never deployed; do not
    re-add (VOR-1427)."""
    import importlib

    try:
        importlib.import_module("vo_rag.core.parsing.parsers.paddleocr_vl")
    except ModuleNotFoundError:
        return
    raise AssertionError("vo_rag.core.parsing.parsers.paddleocr_vl must not be present")


def test_active_parsers_present() -> None:
    """docling and epub are the active parsers."""
    import importlib

    for name in ("docling", "epub", "docling_mapping", "epub_markdown"):
        mod = importlib.import_module(f"vo_rag.core.parsing.parsers.{name}")
        assert mod is not None

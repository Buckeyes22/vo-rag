"""Tests for vo_rag.core.parsing.router.

The router takes no mineru parser (deprecated), no vendor-SDK OCR forward
path and no accelerated-OCR challenger fallback (never deployed). All
PDF dispatch tests use mocked parsers so no live GPU server is needed; the
fitz-backed _make_pdf helper requires PyMuPDF (added as a dev dep) and
exercises the real `_pdf_kind` heuristic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import vo_rag.core.parsing.router as router_module
from tests.core.parsing._router_fakes import (
    FakeDocling as _FakeDocling,
)
from tests.core.parsing._router_fakes import (
    FakeEbookParser as _FakeEbookParser,
)
from vo_rag.core.parsing.router import ParserRouter


def _router() -> ParserRouter:
    return ParserRouter(
        docling=_FakeDocling(),
        epub=_FakeEbookParser("epub"),
    )


def _make_pdf(path: str, text: str) -> None:
    import fitz  # type: ignore[import-untyped]

    doc = fitz.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def _make_multipage_pdf(path: str, page_texts: list[str]) -> None:
    """One page per entry; an empty string yields a text-less (image-like) page."""
    import fitz

    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


@pytest.fixture(autouse=True)
def _stub_magic(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_from_file(path: str, mime: bool = True) -> str:
        _ = (path, mime)
        return "application/octet-stream"

    monkeypatch.setattr(router_module.magic, "from_file", fake_from_file)  # type: ignore[attr-defined]


def test_detect_format_by_extension(tmp_path: Path) -> None:
    router = _router()
    assert router._detect_format(str(tmp_path / "x.docx"), None) == "docx"
    assert router._detect_format(str(tmp_path / "x.epub"), None) == "epub"
    assert router._detect_format(str(tmp_path / "x.mobi"), None) == "mobi"
    assert router._detect_format(str(tmp_path / "x.html"), None) == "html"
    assert router._detect_format(str(tmp_path / "x.txt"), None) == "text"


def test_detect_format_by_provided_mime(tmp_path: Path) -> None:
    router = _router()
    assert router._detect_format(str(tmp_path / "x.bin"), "application/pdf") == "pdf"
    assert router._detect_format(str(tmp_path / "x.bin"), "text/plain") == "text"
    assert (
        router._detect_format(str(tmp_path / "x.bin"), "application/x-mobipocket-ebook") == "mobi"
    )


def test_detect_format_magic_priority(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_from_file(path: str, mime: bool = True) -> str:
        _ = (path, mime)
        return "application/pdf"

    monkeypatch.setattr(router_module.magic, "from_file", fake_from_file)  # type: ignore[attr-defined]
    router = _router()
    assert (
        router._detect_format(
            str(tmp_path / "x.docx"),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        == "pdf"
    )


def test_detect_format_unsupported_raises(tmp_path: Path) -> None:
    router = _router()
    with pytest.raises(ValueError):
        router._detect_format(str(tmp_path / "x.zzz"), None)


@pytest.mark.asyncio
async def test_docling_dispatch_for_docx(tmp_path: Path) -> None:
    router = _router()
    path = tmp_path / "x.docx"
    path.write_bytes(b"ignored")
    doc = await router.parse(str(path), file_hash="h", mime_type=None)
    assert doc.parser_used == "docling"
    assert router.docling.called_fmt == "docx"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_epub_dispatch(tmp_path: Path) -> None:
    router = _router()
    path = tmp_path / "x.epub"
    path.write_bytes(b"ignored")
    doc = await router.parse(str(path), file_hash="h", mime_type=None)
    assert doc.parser_used == "epub"
    assert router.epub.called is True  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_mobi_dispatches_to_ebook_parser(tmp_path: Path) -> None:
    router = _router()
    path = tmp_path / "x.mobi"
    path.write_bytes(b"ignored")
    doc = await router.parse(str(path), file_hash="h", mime_type=None)
    assert doc.parser_used == "mobi"
    assert router.epub.called is True  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_text_dispatch_builds_parsed_document(tmp_path: Path) -> None:
    router = _router()
    path = tmp_path / "notes.txt"
    path.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")

    doc = await router.parse(str(path), file_hash="hash-1", mime_type=None)

    assert doc.parser_used == "text"
    assert doc.document_id == "hash-1"
    assert doc.source_id == "hash-1"
    assert doc.title == "notes"
    assert [element.text for element in doc.elements] == [
        "First paragraph.",
        "Second paragraph.",
    ]
    assert doc.page_qa[0].text_element_count == 2


@pytest.mark.asyncio
async def test_pdf_native_routes_to_docling(tmp_path: Path) -> None:
    pdf = tmp_path / "native.pdf"
    _make_pdf(str(pdf), "the system model data figure equation " * 40)
    router = _router()

    await router.parse(str(pdf), file_hash="hash")

    assert router.docling.called is True  # type: ignore[attr-defined]
    assert router.docling.called_fmt == "pdf"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_pdf_garbled_text_layer_routes_to_docling_ocr(tmp_path: Path) -> None:
    pdf = tmp_path / "garbled.pdf"
    _make_pdf(str(pdf), "xqzk wbfp zztq vkxn " * 200)
    router = _router()
    doc = await router.parse(str(pdf), file_hash="hash")
    assert doc.parser_used == "docling-ocr"
    assert router.docling.scanned_called is True  # type: ignore[attr-defined]
    assert router.docling.called is False  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_pdf_real_text_layer_routes_to_docling(tmp_path: Path) -> None:
    pdf = tmp_path / "real.pdf"
    _make_pdf(
        str(pdf),
        "the car rotates under braking and the driver controls the slip angle " * 30,
    )
    router = _router()
    await router.parse(str(pdf), file_hash="hash")
    assert router.docling.called is True  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_pdf_scanned_routes_to_docling_ocr(tmp_path: Path) -> None:
    pdf = tmp_path / "scanned.pdf"
    _make_pdf(str(pdf), "")
    router = _router()

    doc = await router.parse(str(pdf), file_hash="hash")

    assert doc.parser_used == "docling-ocr"
    assert router.docling.scanned_called is True  # type: ignore[attr-defined]
    assert router.docling.called is False  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_pdf_text_native_with_image_front_matter_routes_to_docling(
    tmp_path: Path,
) -> None:
    """A text-native book whose first pages are image-only front matter (cover,
    plates, title pages) must route to docling, not OCR. The first-pages-only
    probe misclassified these as scanned; spread-sampling the whole document
    fixes it."""
    pdf = tmp_path / "front_matter.pdf"
    body = "the car rotates under braking and the driver controls the slip angle " * 30
    _make_multipage_pdf(str(pdf), [""] * 8 + [body] * 8)
    router = _router()

    await router.parse(str(pdf), file_hash="hash")

    assert router.docling.called is True  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_pdf_fully_textless_multipage_still_routes_to_docling_ocr(tmp_path: Path) -> None:
    """A true scan (no extractable text on any page) stays "scanned" under
    spread-sampling — the fix must not over-correct genuine image-only PDFs."""
    pdf = tmp_path / "true_scan.pdf"
    _make_multipage_pdf(str(pdf), [""] * 30)
    router = _router()

    doc = await router.parse(str(pdf), file_hash="hash")

    assert doc.parser_used == "docling-ocr"
    assert router.docling.scanned_called is True  # type: ignore[attr-defined]
    assert router.docling.called is False  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_pdf_dispatches_to_docling_ocr_forward_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Scanned PDFs go to docling's GPU OCR -- the only scanned-PDF forward
    path in this deployment (VOR-1427)."""
    router = _router()

    async def fake_pdf_kind(self: ParserRouter, path: str) -> str:
        _ = (self, path)
        return "scanned"

    monkeypatch.setattr(ParserRouter, "_pdf_kind", fake_pdf_kind)

    path = tmp_path / "x.pdf"
    path.write_bytes(b"%PDF-1.4\n")
    doc = await router.parse(str(path), file_hash="h", mime_type="application/pdf")

    assert doc.parser_used == "docling-ocr"
    assert router.docling.scanned_called is True  # type: ignore[attr-defined]
    assert router.docling.scanned_file_hash == "h"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_unsupported_format_raises(tmp_path: Path) -> None:
    router = _router()
    path = tmp_path / "x.zzz"
    path.write_bytes(b"")
    with pytest.raises(ValueError):
        await router.parse(str(path), file_hash="h", mime_type=None)


def test_router_signature_drops_mineru_glm_and_paddle() -> None:
    """The router's forward path no longer takes a mineru, glm, or paddle
    parser -- the vendor-SDK forward path and its accelerated-OCR challenger
    fallback were never deployed and are gone (VOR-1427)."""
    import inspect

    sig = inspect.signature(ParserRouter.__init__)
    assert "mineru" not in sig.parameters
    assert "glm" not in sig.parameters
    assert "paddle" not in sig.parameters
    assert "allow_accelerated_paddle_fallback" not in sig.parameters
    # Confirm the surviving parsers
    assert "docling" in sig.parameters
    assert "epub" in sig.parameters

"""Parse-step gates on pre-parsed native PDFs (VOR-1427).

A native PDF served from ``ocr/<sha256>/docling.json`` (``meta.pipeline ==
"native"``, parsed off-box) must pass through exactly the gates a live native
docling parse passes through in ``do_parse_document``. Every case runs twice
through a real ``ParserRouter`` -- once from the cache, once live -- on the
same DoclingDocument stand-in, and both must end the same way.

The upload is a real PDF built with fitz (blank pages, as many as the case
asks for), so the parse step's page-count reconcile genuinely runs and is
compared on both paths.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fitz  # type: ignore[import-untyped]
import pytest

import vo_rag.core.parsing.router as router_module
from tests.core.parsing._router_fakes import FakeParser
from vo_rag.core.parsing.contract import ParsedDocument
from vo_rag.core.parsing.parsers.docling_mapping import docling_to_parsed_document
from vo_rag.core.parsing.router import ParserRouter
from vo_rag.core.storage.figures import FigureStore
from vo_rag.pipelines.book_rag.steps.parse import do_parse_document

BUCKET = "vorag-artifacts"
UPLOAD_KEY = "uploads/native-book.pdf"
RUNPOD_META = {"pipeline": "native", "host": "runpod", "docling": "2.97.0", "backend": "pypdfium"}
PATHS = ["pre-parsed", "live"]


class _Store:
    """Fake Garage shared by the parse step and the router's cache lookup."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = dict(objects)
        self.puts: list[str] = []

    async def object_metadata(self, bucket: str, key: str) -> Any:
        assert bucket == BUCKET
        return SimpleNamespace(size=len(self.objects[key])) if key in self.objects else None

    async def get_bytes(self, bucket: str, key: str) -> bytes:
        assert bucket == BUCKET
        return self.objects[key]

    async def put_bytes(self, bucket: str, key: str, data: bytes, content_type: str) -> None:
        _ = content_type
        assert bucket == BUCKET
        self.objects[key] = data
        self.puts.append(key)


class _LiveDocling:
    """Live native docling parse: maps the same stand-in the way DoclingParser.parse does."""

    def __init__(self, document: Any, figures: FigureStore, *, allowed: bool) -> None:
        self.document = document
        self.figures = figures
        self.allowed = allowed
        self.called = False

    async def parse(
        self, file_path: str, fmt: str, *, file_hash: str | None = None
    ) -> ParsedDocument:
        # On the pre-parsed path a live parse would make every gate case pass
        # for the wrong reason, so it fails the test outright instead.
        if not self.allowed:
            raise AssertionError("a pre-parsed native PDF must not be parsed live")
        self.called = True
        return docling_to_parsed_document(
            self.document, self.figures, file_path, fmt, src_hash=file_hash or ""
        )

    async def parse_scanned(self, file_path: str, *, file_hash: str) -> ParsedDocument:
        raise AssertionError("a native PDF must never be OCR'd")


def _pdf_bytes(page_count: int) -> bytes:
    """A real PDF with ``page_count`` blank pages (fitz can open and count it)."""
    pdf = fitz.open()
    try:
        for _ in range(page_count):
            pdf.new_page(width=612, height=792)
        return bytes(pdf.tobytes())
    finally:
        pdf.close()


def _document(page_count: int, texts: dict[int, str]) -> Any:
    return SimpleNamespace(
        name="native-book",
        pages={n: SimpleNamespace(page_no=n) for n in range(1, page_count + 1)},
        texts=[
            SimpleNamespace(text=text, label="text", prov=[SimpleNamespace(page_no=page)])
            for page, text in texts.items()
        ],
        tables=[],
        pictures=[],
    )


@pytest.fixture(autouse=True)
def _native_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_from_file(path: str, mime: bool = True) -> str:
        _ = (path, mime)
        return "application/pdf"

    async def native(self: ParserRouter, path: str) -> str:
        _ = (self, path)
        return "native"

    monkeypatch.setattr(router_module.magic, "from_file", fake_from_file)  # type: ignore[attr-defined]
    monkeypatch.setattr(ParserRouter, "_pdf_kind", native)


async def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path: str,
    document: Any,
    *,
    pdf_pages: int | None = None,
) -> tuple[dict[str, Any], _Store, _LiveDocling]:
    """Parse a real ``pdf_pages``-page PDF (default: as many as ``document``)."""
    monkeypatch.setattr(router_module, "_load_docling_json", lambda data: document)
    pdf = _pdf_bytes(len(document.pages) if pdf_pages is None else pdf_pages)
    file_hash = hashlib.sha256(pdf).hexdigest()
    objects = {UPLOAD_KEY: pdf}
    if path == "pre-parsed":
        objects[f"ocr/{file_hash}/docling.json"] = b'{"schema_name": "DoclingDocument"}'
        objects[f"ocr/{file_hash}/meta.json"] = json.dumps(RUNPOD_META).encode()
    store = _Store(objects)
    figures = FigureStore(tmp_path / "figures")
    live = _LiveDocling(document, figures, allowed=path == "live")
    router = ParserRouter(
        docling=live,
        epub=FakeParser("epub"),
        garage=store,
        bucket=BUCKET,
        figures=figures,
    )
    result = await do_parse_document(
        envelope_id="env-native",
        garage_key=UPLOAD_KEY,
        source_id="src-native",
        garage=store,  # type: ignore[arg-type]
        bucket=BUCKET,
        parser_router=router,
    )
    return result, store, live


@pytest.mark.parametrize("path", PATHS)
async def test_good_native_book_passes_the_same_gates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    document = _document(2, {1: "Chapter one opens here.", 2: "Chapter two follows on."})

    result, store, live = await _run(monkeypatch, tmp_path, path, document)

    assert live.called is (path == "live")
    assert result["parser_used"] == "docling"
    assert result["forward_path"]["ocr_stack"] == "docling-layout-pdf"
    assert result["forward_path"]["parser_used"] == "docling"
    persisted = json.loads(store.objects["parsed/env-native/parsed.json"])
    assert persisted["document_id"] == hashlib.sha256(store.objects[UPLOAD_KEY]).hexdigest()
    assert persisted["metadata"]["parse_quality_gate"]["quarantined_page_count"] == 0
    assert persisted["metadata"]["book_rag_forward_path"]["ocr_stack"] == "docling-layout-pdf"
    # fitz opened the upload and its page count agreed with the parse.
    assert "pdf_page_count_reconcile_error" not in persisted["metadata"]
    assert result["total_pages"] == 2
    if path == "pre-parsed":
        assert persisted["metadata"]["pre_parsed"] == "native"
        assert persisted["metadata"]["ocr_host"] == "runpod"
    else:
        assert "pre_parsed" not in persisted["metadata"]


@pytest.mark.parametrize("path", PATHS)
async def test_no_text_gate_applies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    with pytest.raises(RuntimeError, match="docling PDF parse produced no text"):
        await _run(monkeypatch, tmp_path, path, _document(3, {}))


@pytest.mark.parametrize("path", PATHS)
async def test_page_coverage_gate_applies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    document = _document(12, {1: "Only the first page carries any extracted text."})

    result, store, _ = await _run(monkeypatch, tmp_path, path, document)
    persisted = json.loads(store.objects["parsed/env-native/parsed.json"])
    assert result["quality_warnings"][0]["code"] == "low_page_coverage"
    assert persisted["elements"][0]["text"] == "Only the first page carries any extracted text."


@pytest.mark.parametrize("path", PATHS)
async def test_content_corruption_gate_applies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    glyph_soup = "YHUWLFDO FF] WDQJHQWLDO VXEVFULSW\x03V \x11\x03)LJXUH " * 20
    document = _document(2, {1: glyph_soup, 2: glyph_soup})

    with pytest.raises(RuntimeError, match="content-corruption gate"):
        await _run(monkeypatch, tmp_path, path, document)


@pytest.mark.parametrize("path", PATHS)
async def test_pages_missing_from_the_parse_are_padded_before_the_coverage_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    """The PDF has 12 pages but the parse only knows page 1: the page-count
    reconcile pads it to 12, so the coverage gate sees 1/12 and fires."""
    document = _document(1, {1: "Only the first page carries any extracted text."})

    result, _, _ = await _run(monkeypatch, tmp_path, path, document, pdf_pages=12)
    assert result["quality_warnings"][0]["code"] in {"low_page_coverage", "page_quality_findings"}
    assert any(w["code"] == "low_page_coverage" for w in result["quality_warnings"])

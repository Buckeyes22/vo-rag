"""A native PDF the OCR farm has OCR'd is served from the farm cache (VOR-1427).

Production case: a 34-page practical guide.
``_pdf_kind`` rightly calls it native (text on 83% of sampled pages,
dictionary-hit ratio 0.23), yet its live docling parse failed "covered content on
only 0/34 pages", so it went to the farm backlog.
Parsing it live again after the farm OCR'd it would fail again and it could
never ingest. An explicit ``"pipeline": "ocr"`` label in the farm's
``meta.json`` is trusted: the cache is used as ``docling-ocr``, and the parse
step applies the docling-ocr gates to it exactly as for a scanned PDF.

Here ``_pdf_kind`` is not stubbed: the upload is a real fitz-built 34-page
PDF with a text layer, and the test first checks it classifies native.
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
from vo_rag.core.parsing.router import ParserRouter
from vo_rag.core.storage.figures import FigureStore
from vo_rag.pipelines.book_rag.steps.parse import do_parse_document

BUCKET = "vorag-artifacts"
UPLOAD_KEY = "uploads/a-practical-guide-to-rowing.pdf"
PAGES = 34
FARM_META = {"pipeline": "ocr", "host": "lin"}
PROSE = (
    "A rowing crew is a team that steadily completes a course on the\n"
    "water. It uses a coxswain to steer the boat and decide when the\n"
    "stroke rate should rise, and it can shift pace to hold the line.\n"
)


class _Store:
    """Fake Garage shared by the parse step and the router's cache lookup."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = dict(objects)

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


class _NoLiveDocling:
    """The live parse already failed for this book; calling it again fails the test."""

    async def parse(
        self, file_path: str, fmt: str, *, file_hash: str | None = None
    ) -> ParsedDocument:
        raise AssertionError("an OCR-labelled cached book must not be parsed live again")

    async def parse_scanned(self, file_path: str, *, file_hash: str) -> ParsedDocument:
        raise AssertionError("the farm already OCR'd this book")


def _native_pdf() -> bytes:
    """34 pages; a cover and 5 other pages have no text, the rest carry prose."""
    textless = {0, 1, 7, 15, 22, 33}
    pdf = fitz.open()
    try:
        for n in range(PAGES):
            page = pdf.new_page(width=612, height=792)
            if n not in textless:
                page.insert_text((72, 72), PROSE, fontsize=10)
        return bytes(pdf.tobytes())
    finally:
        pdf.close()


def _ocr_document(texts: dict[int, str]) -> Any:
    return SimpleNamespace(
        name="farm-ocr",
        pages={n: SimpleNamespace(page_no=n) for n in range(1, PAGES + 1)},
        texts=[
            SimpleNamespace(text=text, label="text", prov=[SimpleNamespace(page_no=page)])
            for page, text in texts.items()
        ],
        tables=[],
        pictures=[],
    )


@pytest.fixture(autouse=True)
def _pdf_magic(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_from_file(path: str, mime: bool = True) -> str:
        _ = (path, mime)
        return "application/pdf"

    monkeypatch.setattr(router_module.magic, "from_file", fake_from_file)  # type: ignore[attr-defined]


async def _run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, document: Any
) -> tuple[dict[str, Any], _Store]:
    monkeypatch.setattr(router_module, "_load_docling_json", lambda data: document)
    pdf = _native_pdf()
    file_hash = hashlib.sha256(pdf).hexdigest()
    store = _Store(
        {
            UPLOAD_KEY: pdf,
            f"ocr/{file_hash}/docling.json": b'{"schema_name": "DoclingDocument"}',
            f"ocr/{file_hash}/meta.json": json.dumps(FARM_META).encode(),
        }
    )
    router = ParserRouter(
        docling=_NoLiveDocling(),
        epub=FakeParser("epub"),
        garage=store,
        bucket=BUCKET,
        figures=FigureStore(tmp_path / "figures"),
    )
    local = tmp_path / "upload.pdf"
    local.write_bytes(pdf)
    assert await router._pdf_kind(str(local)) == "native"
    result = await do_parse_document(
        envelope_id="01KVPE4CZ10XN8PY7J2MJE0F5Y",
        garage_key=UPLOAD_KEY,
        source_id="src-agents",
        garage=store,  # type: ignore[arg-type]
        bucket=BUCKET,
        parser_router=router,
    )
    return result, store


async def test_native_book_with_farm_ocr_cache_ingests_as_docling_ocr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    document = _ocr_document({n: f"OCR text of page {n} about agents." for n in range(1, 35)})

    result, store = await _run(monkeypatch, tmp_path, document)

    assert result["parser_used"] == "docling-ocr"
    assert result["forward_path"]["ocr_stack"] == "docling-rapidocr-torch-cuda"
    assert result["forward_path"]["parser_used"] == "docling-ocr"
    assert result["total_pages"] == PAGES
    persisted = json.loads(store.objects["parsed/01KVPE4CZ10XN8PY7J2MJE0F5Y/parsed.json"])
    assert persisted["metadata"]["ocr_source"] == "farm"
    assert persisted["metadata"]["ocr_host"] == "lin"
    assert "pre_parsed" not in persisted["metadata"]
    assert persisted["metadata"]["book_rag_forward_path"]["ocr_stack"] == (
        "docling-rapidocr-torch-cuda"
    )


async def test_docling_ocr_coverage_gate_applies_to_the_cached_book(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    document = _ocr_document({1: "Only the cover was read by the OCR farm."})

    result, _ = await _run(monkeypatch, tmp_path, document)
    assert any(w["code"] == "low_page_coverage" for w in result["quality_warnings"])


async def test_docling_ocr_no_text_gate_applies_to_the_cached_book(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="docling PDF parse produced no text"):
        await _run(monkeypatch, tmp_path, _ocr_document({}))

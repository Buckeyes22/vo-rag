"""Absurdly large PDF pages are rescaled before docling opens them (VOR-1434).

The real regression file had a 13,400 x 9,100pt page that docling could not
parse. ``DoclingParser`` rescales any page whose longest side exceeds
``OVERSIZE_PAGE_PT`` (5,000pt) to at most
``RESCALED_PAGE_MAX_PT`` (1,200pt) with PyMuPDF ``show_pdf_page`` into a temp
copy; page count and order are preserved and the original is never modified.

docling is not installed in the light CI suite, so the converter is a fake
that opens whatever path it is handed with PyMuPDF and records what it saw.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import fitz  # type: ignore[import-untyped, unused-ignore]
import pytest

from vo_rag.core.parsing import preflight
from vo_rag.core.parsing.parsers import docling as docling_module
from vo_rag.core.parsing.parsers.docling import (
    OVERSIZE_PAGE_PT,
    RESCALED_PAGE_MAX_PT,
    DoclingParser,
    rescale_oversize_pages,
)


def _build_pdf(path: Path, sizes: list[tuple[float, float]]) -> Path:
    doc = fitz.open()
    for index, (width, height) in enumerate(sizes):
        page = doc.new_page(width=width, height=height)
        page.insert_text((72, 144), f"Page marker {index}", fontsize=24)
    doc.save(path)
    doc.close()
    return path


def _sizes(path: str | Path) -> list[tuple[float, float]]:
    with fitz.open(path) as doc:
        return [(page.rect.width, page.rect.height) for page in doc]


def _texts(path: str | Path) -> list[str]:
    with fitz.open(path) as doc:
        return [page.get_text("text") for page in doc]


def test_threshold_is_shared_with_preflight() -> None:
    assert OVERSIZE_PAGE_PT == 5_000.0
    assert RESCALED_PAGE_MAX_PT == 1_200.0
    assert preflight.OVERSIZE_PAGE_PT is docling_module.OVERSIZE_PAGE_PT
    # One measure of "longest side" for both preflight and the parser.
    assert vars(preflight)["_longest_side"] is docling_module._longest_side


def test_rescale_shrinks_only_oversize_pages(tmp_path: Path) -> None:
    src = _build_pdf(tmp_path / "plate.pdf", [(400, 500), (13_400, 9_100), (400, 500)])
    before = src.read_bytes()
    workdir = tmp_path / "work"
    workdir.mkdir()

    out = rescale_oversize_pages(src, workdir)

    assert out is not None and out != src
    assert out.parent == workdir and out.name == src.name
    assert src.read_bytes() == before  # original never modified
    sizes = _sizes(out)
    assert len(sizes) == 3
    assert sizes[0] == (400, 500) and sizes[2] == (400, 500)
    width, height = sizes[1]
    assert max(width, height) <= RESCALED_PAGE_MAX_PT
    assert width / height == pytest.approx(13_400 / 9_100, rel=1e-3)
    texts = _texts(out)
    assert [f"Page marker {i}" in t for i, t in enumerate(texts)] == [True, True, True]


def test_rescale_is_noop_at_the_threshold(tmp_path: Path) -> None:
    src = _build_pdf(tmp_path / "ok.pdf", [(OVERSIZE_PAGE_PT, 500)])
    assert rescale_oversize_pages(src, tmp_path) is None


def test_rescale_leaves_unopenable_files_to_docling(tmp_path: Path) -> None:
    src = tmp_path / "broken.pdf"
    src.write_bytes(b"not a pdf")
    assert rescale_oversize_pages(src, tmp_path) is None


class _FakeConverter:
    def __init__(self) -> None:
        self.seen: list[tuple[str, list[tuple[float, float]]]] = []

    def convert(self, file_path: str) -> Any:
        self.seen.append((file_path, _sizes(file_path)))
        return type("Result", (), {"document": object()})()


def _parser(monkeypatch: pytest.MonkeyPatch) -> tuple[DoclingParser, _FakeConverter, list[Any]]:
    converter = _FakeConverter()
    parser = object.__new__(DoclingParser)
    parser._converter = converter
    parser._scanned_converter = converter
    parser._device = "auto"
    parser._pdf_backend = "default"
    parser._figures = object()  # type: ignore[assignment]
    mapped: list[Any] = []

    def _map(doc: Any, figures: Any, source_path: str, fmt: Any, **kwargs: Any) -> Any:
        mapped.append((source_path, fmt, kwargs))
        return {"source_path": source_path}

    monkeypatch.setattr(docling_module, "docling_to_parsed_document", _map)
    monkeypatch.setattr(docling_module, "_release_cuda_cache", lambda: None)
    return parser, converter, mapped


def test_native_parse_of_oversize_page_pdf_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _build_pdf(tmp_path / "plate.pdf", [(13_400, 9_100), (400, 500)])
    before = src.read_bytes()
    parser, converter, mapped = _parser(monkeypatch)

    result = asyncio.run(parser.parse(str(src), "pdf"))

    assert result == {"source_path": str(src)}
    [(seen_path, seen_sizes)] = converter.seen
    assert seen_path != str(src)
    assert len(seen_sizes) == 2  # page count preserved
    assert all(max(w, h) <= OVERSIZE_PAGE_PT for w, h in seen_sizes)
    assert not Path(seen_path).exists()  # temp copy cleaned up
    assert src.read_bytes() == before
    # identity stays the original file's, not the temp copy's
    assert mapped[0][2]["src_hash"] == hashlib.sha256(before).hexdigest()


def test_scanned_parse_of_oversize_page_pdf_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _build_pdf(tmp_path / "plate.pdf", [(400, 500), (9_100, 13_400)])
    parser, converter, mapped = _parser(monkeypatch)

    asyncio.run(parser.parse_scanned(str(src), file_hash="abc"))

    [(seen_path, seen_sizes)] = converter.seen
    assert len(seen_sizes) == 2
    assert all(max(w, h) <= OVERSIZE_PAGE_PT for w, h in seen_sizes)
    assert not Path(seen_path).exists()
    assert mapped[0][0] == str(src)
    assert mapped[0][2]["src_hash"] == "abc"


def test_normal_pdf_is_handed_to_docling_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _build_pdf(tmp_path / "book.pdf", [(400, 500), (400, 500)])
    parser, converter, _ = _parser(monkeypatch)

    asyncio.run(parser.parse(str(src), "pdf", file_hash="h"))

    assert converter.seen[0][0] == str(src)


@pytest.mark.parametrize(
    ("owner", "method"),
    [(fitz.Document, "insert_pdf"), (fitz.Page, "show_pdf_page"), (fitz.Document, "save")],
)
def test_pymupdf_failure_mid_rewrite_hands_docling_the_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner: Any, method: str
) -> None:
    """A PDF PyMuPDF opens but can't walk or rewrite must not fail before
    docling gets a chance: the rescale falls back to the original path."""
    src = _build_pdf(tmp_path / "plate.pdf", [(13_400, 9_100)])

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError(f"{method} failed")

    monkeypatch.setattr(owner, method, _boom)
    parser, converter, _ = _parser(monkeypatch)

    asyncio.run(parser.parse(str(src), "pdf", file_hash="h"))

    assert converter.seen[0][0] == str(src)

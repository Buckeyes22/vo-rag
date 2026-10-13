"""Tests for content fingerprints and same-content dedup (VOR-1428)."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import fitz  # type: ignore[import-untyped]
import pytest

from vo_rag.core.parsing.fingerprint import content_fingerprint, duplicate_groups

_WORDS = ["tolerance", "allowance", "knurling", "spindle", "thread", "gauge", "bearing", "torque"]


def _page_text(book: str, page: int) -> str:
    return f"{book} page {page} " + " ".join(_WORDS[(page + i) % len(_WORDS)] for i in range(120))


def _pdf(path: Path, book: str, pages: int = 20, *, title: str = "") -> Path:
    doc = fitz.open()
    for n in range(pages):
        doc.new_page().insert_textbox(fitz.Rect(36, 36, 560, 800), _page_text(book, n), fontsize=8)
    doc.set_metadata({"title": title})  # metadata differs between re-saves; text does not
    doc.save(path)
    return path


def _epub(path: Path, chapters: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr(
            "META-INF/container.xml",
            '<container><rootfiles><rootfile full-path="OEBPS/book.opf"/></rootfiles></container>',
        )
        items = "".join(f'<item id="c{i}" href="c{i}.xhtml"/>' for i in range(len(chapters)))
        spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(chapters)))
        z.writestr(
            "OEBPS/book.opf",
            f"<package><manifest>{items}</manifest><spine>{spine}</spine></package>",
        )
        for i, text in enumerate(chapters):
            z.writestr(f"OEBPS/c{i}.xhtml", f"<html><body><p>{text}</p></body></html>")
    return path


def test_resaved_pdf_matches_and_a_different_book_does_not(tmp_path: Path) -> None:
    a = content_fingerprint(_pdf(tmp_path / "a.pdf", "workshops handbook", title="one"))
    b = content_fingerprint(_pdf(tmp_path / "b.pdf", "workshops handbook", title="two"))
    other = content_fingerprint(_pdf(tmp_path / "c.pdf", "wind tunnels"))

    assert (tmp_path / "a.pdf").read_bytes() != (tmp_path / "b.pdf").read_bytes()
    assert a is not None and a.startswith("pdf:20:")
    assert a == b
    assert other != a


def test_page_count_is_part_of_the_pdf_fingerprint(tmp_path: Path) -> None:
    assert content_fingerprint(_pdf(tmp_path / "a.pdf", "x", pages=20)) != content_fingerprint(
        _pdf(tmp_path / "b.pdf", "x", pages=21)
    )


def test_epub_fingerprint_follows_spine_text(tmp_path: Path) -> None:
    chapters = [_page_text("ch", n) for n in range(5)]
    a = content_fingerprint(_epub(tmp_path / "a.epub", chapters))
    b = content_fingerprint(_epub(tmp_path / "b.epub", chapters))
    c = content_fingerprint(_epub(tmp_path / "c.epub", chapters[::-1]))

    assert a is not None and a.startswith("epub:")
    assert a == b
    assert c != a


@pytest.mark.parametrize("name", ["scan.pdf", "broken.pdf", "book.mobi"])
def test_unidentifiable_files_get_no_fingerprint(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    if name == "scan.pdf":
        doc = fitz.open()
        for _ in range(10):
            doc.new_page()  # image-only scans have no text layer
        doc.save(path)
    else:
        path.write_bytes(b"%PDF-1.4 not really")
    assert content_fingerprint(path) is None


def test_duplicate_groups_only_reports_shared_fingerprints() -> None:
    assert duplicate_groups({"s1": "fp", "s2": "fp", "s3": "other", "s4": ""}) == {
        "fp": ["s1", "s2"]
    }


def test_epub_fingerprint_never_decompresses_a_zip_bomb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preflight fingerprints a file before its
    container check, so the fingerprint must refuse a bomb from the central
    directory alone, without reading a member."""
    from tests.fixtures.epub.build_preflight_epub_fixtures import build_zip_bomb_epub

    bomb = build_zip_bomb_epub(tmp_path / "bomb.epub")
    reads: list[str] = []
    real_read = zipfile.ZipFile.read

    def _tracking_read(self: zipfile.ZipFile, name: Any, pwd: Any = None) -> bytes:
        reads.append(str(name))
        return real_read(self, name, pwd)

    monkeypatch.setattr(zipfile.ZipFile, "read", _tracking_read)
    assert content_fingerprint(bomb) is None
    assert reads == []

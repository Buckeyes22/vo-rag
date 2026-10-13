"""Tests for vo_rag.core.parsing.parsers.epub helper functions.

Verifies the
pure helpers (TOC tree normalization, synthetic page QA, MOBI conversion
shim, FigureStore image write helper) without requiring an ebooklib install.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from vo_rag.core.parsing.contract import ElementType, ParsedElement
from vo_rag.core.parsing.parsers import epub as epub_module
from vo_rag.core.parsing.parsers.epub import (
    _convert_mobi_to_epub,
    _ebook_format,
    _extract_embedded_images,
    _synthetic_page_qa,
    _toc_to_heading_tree,
)
from vo_rag.core.storage.figures import FigureStore


@dataclass
class _TocEntry:
    title: str
    href: str


def test_toc_depth_normalized_to_root_one() -> None:
    toc = [
        (_TocEntry(title="Chapter 1", href="chap1.xhtml"), [_TocEntry("Section 1.1", "s1.xhtml")]),
        _TocEntry(title="Chapter 2", href="chap2.xhtml"),
    ]
    tree = _toc_to_heading_tree(toc)
    assert tree
    assert min(item["depth"] for item in tree) == 1
    assert tree[0]["depth"] == 1
    assert any(item["depth"] == 2 for item in tree)


def test_synthetic_page_qa_has_full_confidence() -> None:
    grp = [
        ParsedElement(
            element_id="e",
            element_type=ElementType.NARRATIVE_TEXT,
            text="x",
            page_number=1,
        )
    ]
    qa = _synthetic_page_qa(1, grp)
    assert qa.ocr_confidence == 1.0
    assert qa.parser_used == "epub"


def test_synthetic_page_qa_can_record_mobi_parser() -> None:
    qa = _synthetic_page_qa(1, [], parser_used="mobi")
    assert qa.parser_used == "mobi"


def test_ebook_format_detects_mobi_extension(tmp_path: Path) -> None:
    assert _ebook_format(str(tmp_path / "book.mobi")) == "mobi"
    assert _ebook_format(str(tmp_path / "book.epub")) == "epub"


def test_convert_mobi_to_epub_invokes_calibre_converter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "book.mobi"
    target = tmp_path / "book.epub"
    source.write_bytes(b"mobi")
    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        check: bool,  # noqa: ARG001
        capture_output: bool,  # noqa: ARG001
        env: dict[str, str],
        text: bool,  # noqa: ARG001
        timeout: int,  # noqa: ARG001
    ) -> subprocess.CompletedProcess[str]:
        assert env["PYTHONNOUSERSITE"] == "1"
        assert "PYTHONPATH" not in env
        calls.append(command)
        Path(command[2]).write_bytes(b"epub")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(epub_module.shutil, "which", lambda name: f"/usr/bin/{name}")  # type: ignore[attr-defined]
    monkeypatch.setattr(epub_module.subprocess, "run", fake_run)  # type: ignore[attr-defined]

    _convert_mobi_to_epub(source, target)

    assert calls == [["/usr/bin/ebook-convert", str(source), str(target)]]


def test_convert_mobi_to_epub_fails_when_converter_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "book.mobi"
    target = tmp_path / "book.epub"
    source.write_bytes(b"mobi")
    monkeypatch.setattr(epub_module.shutil, "which", lambda name: None)  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="ebook-convert"):
        _convert_mobi_to_epub(source, target)


def test_extract_embedded_images_writes_to_figure_store(tmp_path: Path) -> None:
    src_hash = "ab" * 32

    class _ImageItem:
        media_type = "image/png"
        file_name = "image.png"

        def get_content(self) -> bytes:
            return b"image-bytes"

    class _Book:
        def get_items(self) -> list[_ImageItem]:
            return [_ImageItem()]

    images = _extract_embedded_images(_Book(), FigureStore(tmp_path), src_hash)

    assert len(images) == 1
    assert images[0].ref == f"{src_hash}/p001_img001.png"
    assert images[0].mime_type == "image/png"
    assert (tmp_path / images[0].ref).is_file()

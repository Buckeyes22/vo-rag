"""One narrative element per EPUB paragraph, not per 72-column line (VOR-1437).

Pandoc hard-wraps markdown at 72 columns by default and ``markdown_to_elements``
makes one element per line, so every paragraph was split into several
elements. On The Science of Car Racing (01M302PPABV8ZH1WFKSSB57MN5), 134 page
images plus a copyright page wrapped into 50 line elements kept the image
share at 72.8% and the book off the OCR path.
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Any

import pytest

from vo_rag.core.parsing.contract import ElementType


def _require_pandoc_stack() -> Any:
    epub = pytest.importorskip("ebooklib.epub")
    pytest.importorskip("pypandoc")
    if shutil.which("pandoc") is None:
        pytest.skip("pandoc binary not installed")
    return epub


def _jpeg() -> bytes:
    pil_image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    pil_image.new("RGB", (8, 8), color=(20, 20, 20)).save(buf, format="JPEG")
    return buf.getvalue()


def _write_book(epub: Any, path: Path, chapters: list[tuple[str, str]], images: int = 0) -> Path:
    book = epub.EpubBook()
    book.set_identifier(f"vorag-test-{path.stem}")
    book.set_title(path.stem)
    book.set_language("en")
    for i in range(images):
        book.add_item(
            epub.EpubItem(
                uid=f"page{i}",
                file_name=f"images/p{i}.jpg",
                media_type="image/jpeg",
                content=_jpeg(),
            )
        )
    items = []
    for i, (title, body) in enumerate(chapters):
        item = epub.EpubHtml(title=title, file_name=f"text/c{i}.xhtml", lang="en")
        item.content = f"<html><body>{body}</body></html>"
        book.add_item(item)
        items.append(item)
    book.toc = tuple(items)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *items]
    epub.write_epub(str(path), book)
    return path


def _elements(path: Path) -> list[Any]:
    from vo_rag.core.parsing.parsers.epub import _read_epub_and_convert
    from vo_rag.core.parsing.parsers.epub_markdown import markdown_to_elements

    _, _, toc_tree, markdown = _read_epub_and_convert(str(path), str(path), source_format="epub")
    return markdown_to_elements(markdown, toc_tree)


def test_long_paragraph_is_one_narrative_element(tmp_path: Path) -> None:
    epub = _require_pandoc_stack()
    paragraph = "A long paragraph of real prose that pandoc used to hard-wrap. " * 12
    path = _write_book(epub, tmp_path / "prose.epub", [("One", f"<h1>One</h1><p>{paragraph}</p>")])

    narrative = [e for e in _elements(path) if e.element_type == ElementType.NARRATIVE_TEXT]

    assert len(narrative) == 1
    assert narrative[0].text == paragraph.strip()


def test_page_scan_with_wrapped_copyright_page_is_image_only(tmp_path: Path) -> None:
    epub = _require_pandoc_stack()
    from vo_rag.core.parsing.parsers.epub import is_image_only

    copyright_page = "".join(
        f"<p>Practical Tuning is published by Northwind Press, 100 Example Street, Springfield, "
        f"Exampleland 00001. Copyright line {i} of the imprint page for this title.</p>"
        for i in range(10)
    )
    pages = [(f"Page {i}", f'<img src="../images/p{i}.jpg" alt=""/>') for i in range(50)]
    path = _write_book(
        epub, tmp_path / "scan.epub", [("Copyright", copyright_page), *pages], images=50
    )

    elements = _elements(path)
    types = [e.element_type for e in elements]

    assert types.count(ElementType.IMAGE) == 50
    assert types.count(ElementType.NARRATIVE_TEXT) == 10
    assert is_image_only(elements) is True

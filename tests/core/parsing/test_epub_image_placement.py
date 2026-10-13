"""EPUB images attach to the IMAGE element that shows them (VOR-1435).

Before the fix every ``ExtractedImage`` had ``element_index=0``, so each
figure composite took the book's root section and front-matter context
(Linear Algebra canary: 9,983 composites, all ``[Linear Algebra]``).
"""

from __future__ import annotations

import asyncio
import io
import shutil
from pathlib import Path

import pytest

from vo_rag.core.parsing.contract import ElementType, ParsedElement
from vo_rag.core.parsing.parsers.epub import _image_element_positions
from vo_rag.core.storage.figures import FigureStore


def _element(index: int, element_type: ElementType, text: str) -> ParsedElement:
    return ParsedElement(element_id=f"e{index}", element_type=element_type, text=text)


def test_image_positions_key_by_decoded_path_and_name_first_occurrence() -> None:
    elements = [
        _element(0, ElementType.TITLE, "Chapter 1"),
        _element(1, ElementType.IMAGE, '![a](images/fig%200.png "Figure 1")'),
        _element(2, ElementType.NARRATIVE_TEXT, "![not an image element](x.png)"),
        _element(3, ElementType.IMAGE, "![again](images/fig%200.png)"),
        _element(4, ElementType.IMAGE, "![b](../Images/Other.JPG)"),
    ]

    by_path, by_name = _image_element_positions(elements)

    assert by_path == {"images/fig 0.png": 1, "images/other.jpg": 4}
    assert by_name == {"fig 0.png": 1, "other.jpg": 4}


def _png(color: tuple[int, int, int]) -> bytes:
    pil_image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    pil_image.new("RGB", (4, 4), color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_each_image_is_placed_inside_its_own_chapter(tmp_path: Path) -> None:
    epub = pytest.importorskip("ebooklib.epub")
    pytest.importorskip("pypandoc")
    if shutil.which("pandoc") is None:
        pytest.skip("pandoc binary not installed")

    from vo_rag.core.parsing.parsers.epub import EpubParser

    book = epub.EpubBook()
    book.set_identifier("vorag-test-image-placement")
    book.set_title("Placement")
    book.set_language("en")
    chapters = []
    for i in range(2):
        book.add_item(
            epub.EpubItem(
                uid=f"img{i}",
                file_name=f"images/fig {i}.png",
                media_type="image/png",
                content=_png((i * 90, 10, 10)),
            )
        )
        chapter = epub.EpubHtml(title=f"Chapter {i + 1}", file_name=f"text/ch{i}.xhtml", lang="en")
        prose = "Real narrative prose sits next to the figure. " * 10
        chapter.content = (
            f"<html><body><h1>Chapter {i + 1}</h1><p>{prose}</p>"
            f'<img src="../images/fig%20{i}.png" alt="figure {i}"/><p>{prose}</p></body></html>'
        )
        book.add_item(chapter)
        chapters.append(chapter)
    book.toc = tuple(chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *chapters]
    path = tmp_path / "placement.epub"
    epub.write_epub(str(path), book)

    doc = asyncio.run(EpubParser(FigureStore(tmp_path / "figs")).parse(str(path)))

    titles = [i for i, e in enumerate(doc.elements) if e.element_type == ElementType.TITLE]
    chapter_starts = [i for i in titles if doc.elements[i].text.startswith("Chapter")]
    assert len(chapter_starts) == 2 and len(doc.images) == 2
    first, second = sorted(doc.images, key=lambda image: image.element_index)
    assert chapter_starts[0] < first.element_index < chapter_starts[1]
    assert chapter_starts[1] < second.element_index
    for image in doc.images:
        element = doc.elements[image.element_index]
        assert element.element_type == ElementType.IMAGE
        assert image.page_number == element.page_number
    assert doc.elements[first.element_index].section_title == "Chapter 1"
    assert doc.elements[second.element_index].section_title == "Chapter 2"


def test_mathml_altimg_fallbacks_and_unused_images_are_not_figures(tmp_path: Path) -> None:
    """VOR-1436: De Gruyter's Linear Algebra ships 9,961 MathML ``altimg``
    PNG fallbacks and 22 real figures; every fallback became a context-free
    "figure". Only images the text shows are extracted."""
    epub = pytest.importorskip("ebooklib.epub")
    pytest.importorskip("pypandoc")
    if shutil.which("pandoc") is None:
        pytest.skip("pandoc binary not installed")

    from vo_rag.core.parsing.parsers.epub import EpubParser

    book = epub.EpubBook()
    book.set_identifier("vorag-test-mathml-altimg")
    book.set_title("Algebra")
    book.set_language("en")
    for name, color in (
        ("fig.png", (200, 0, 0)),
        ("eq1.png", (0, 200, 0)),
        ("eq2.png", (0, 0, 200)),
    ):
        book.add_item(
            epub.EpubItem(
                uid=name, file_name=f"images/{name}", media_type="image/png", content=_png(color)
            )
        )
    book.add_item(
        epub.EpubItem(
            uid="unused",
            file_name="images/unused.png",
            media_type="image/png",
            content=_png((9, 9, 9)),
        )
    )
    math = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML" altimg="../images/{0}">'
        "<mi>x</mi><mo>+</mo><mn>{1}</mn></math>"
    )
    chapter = epub.EpubHtml(title="Chapter 1", file_name="text/ch1.xhtml", lang="en")
    chapter.content = (
        "<html><body><h1>Chapter 1</h1>"
        f"<p>Solve {math.format('eq1.png', 1)} and then {math.format('eq2.png', 2)} for x.</p>"
        '<p>See the figure.</p><img src="../images/fig.png" alt="the figure"/></body></html>'
    )
    book.add_item(chapter)
    book.toc = (chapter,)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]
    path = tmp_path / "algebra.epub"
    epub.write_epub(str(path), book)

    doc = asyncio.run(EpubParser(FigureStore(tmp_path / "figs")).parse(str(path)))

    assert len(doc.images) == 1
    (image,) = doc.images
    assert doc.elements[image.element_index].element_type == ElementType.IMAGE
    assert "fig.png" in doc.elements[image.element_index].text

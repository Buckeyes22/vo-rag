"""EPUB tables GFM cannot express keep their text (VOR-1427).

With raw_html off, pandoc's gfm writer replaces any table whose cells hold
block content (paragraphs, lists, headers, nested tables) or a line break
with the literal line ``[TABLE]`` and drops its text. One consumer guide lays out
every page as nested tables, so the whole book parsed to 276 page TITLEs plus 277 ``[TABLE]``
lines. ``epub_anchors.lua``
now unwraps layout tables into their blocks and flattens data-table cells to
one line; ``markdown_to_elements`` drops any ``[TABLE]`` line that remains.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from vo_rag.core.parsing.contract import ElementType
from vo_rag.core.parsing.parsers.epub_markdown import markdown_to_elements


def _require_pandoc_stack() -> Any:
    epub = pytest.importorskip("ebooklib.epub")
    pytest.importorskip("pypandoc")
    if shutil.which("pandoc") is None:
        pytest.skip("pandoc binary not installed")
    return epub


def _write_book(
    epub: Any,
    path: Path,
    files: list[tuple[str, str]],
    toc: list[tuple[str, str]],
) -> Path:
    """``files``: (file name, body html); ``toc``: (href, title)."""
    book = epub.EpubBook()
    book.set_identifier(f"vorag-test-{path.stem}")
    book.set_title(path.stem)
    book.set_language("en")
    items = []
    for name, body in files:
        item = epub.EpubHtml(title=name, file_name=f"text/{name}", lang="en")
        item.content = f"<html><body>{body}</body></html>"
        book.add_item(item)
        items.append(item)
    book.toc = tuple(
        epub.Link(f"text/{href}", title, f"toc{i}") for i, (href, title) in enumerate(toc)
    )
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *items]
    epub.write_epub(str(path), book)
    return path


def _parse(path: Path) -> tuple[list[Any], str]:
    from vo_rag.core.parsing.parsers.epub import _read_epub_and_convert

    _, _, toc_tree, markdown = _read_epub_and_convert(str(path), str(path), source_format="epub")
    return markdown_to_elements(markdown, toc_tree), markdown


def _typed(elements: list[Any], kind: ElementType) -> list[str]:
    return [e.text for e in elements if e.element_type == kind]


def _no_table_placeholder(elements: list[Any], markdown: str) -> None:
    assert "[TABLE]" not in markdown
    assert not any(e.text.strip() == "[TABLE]" for e in elements)


# ---------------------------------------------------------------------------
# markdown_to_elements defense in depth (no pandoc needed)
# ---------------------------------------------------------------------------


def test_table_placeholder_lines_never_become_narrative() -> None:
    md = "\n\n".join(["Before.", "[TABLE]", "  [TABLE]  ", "After [TABLE] text."])
    elements = markdown_to_elements(md, [])
    assert [(e.element_type, e.text) for e in elements] == [
        (ElementType.NARRATIVE_TEXT, "Before."),
        (ElementType.NARRATIVE_TEXT, "After [TABLE] text."),
    ]


# ---------------------------------------------------------------------------
# Through pandoc + the Lua filter
# ---------------------------------------------------------------------------


def test_layout_table_with_heading_and_paragraphs_is_unwrapped(tmp_path: Path) -> None:
    epub = _require_pandoc_stack()
    page = (
        '<table border="0" width="100%"><tr><td>'
        "<h2>Page Heading</h2><p>First paragraph of the page.</p>"
        "<ul><li>A listed point.</li></ul>"
        "</td></tr><tr><td><p>Second row paragraph.</p><p>Third paragraph.</p></td></tr>"
        "</table>"
    )
    elements, markdown = _parse(
        _write_book(epub, tmp_path / "layout.epub", [("p1.xhtml", page)], [])
    )

    _no_table_placeholder(elements, markdown)
    assert "Page Heading" in _typed(elements, ElementType.TITLE)
    narrative = _typed(elements, ElementType.NARRATIVE_TEXT)
    assert narrative == [
        "First paragraph of the page.",
        "Second row paragraph.",
        "Third paragraph.",
    ]
    assert _typed(elements, ElementType.LIST_ITEM) == ["A listed point."]
    assert not _typed(elements, ElementType.TABLE)


def test_spanning_data_table_with_inline_cells_stays_one_table(tmp_path: Path) -> None:
    epub = _require_pandoc_stack()
    table = (
        "<table>"
        '<tr><th colspan="2">Engine</th><th>Power</th></tr>'
        '<tr><td rowspan="2">V8</td><td>Bore<br/>Stroke</td><td><div>450 hp</div></td></tr>'
        "<tr><td>Cams</td><td>500 hp</td></tr>"
        "</table>"
    )
    elements, markdown = _parse(_write_book(epub, tmp_path / "span.epub", [("t.xhtml", table)], []))

    _no_table_placeholder(elements, markdown)
    tables = _typed(elements, ElementType.TABLE)
    assert len(tables) == 1
    for text in ("Engine", "Power", "V8", "Bore", "Stroke", "450 hp", "Cams", "500 hp"):
        assert text in tables[0]
    assert not _typed(elements, ElementType.NARRATIVE_TEXT)


def test_page_layout_with_nested_spacer_tables_becomes_plain_text(tmp_path: Path) -> None:
    """The consumer-guide shape: a page table whose rows hold spacer-column
    tables with one text cell each. The nested tables are simple, but were
    never written before; they must not become one padded pipe table per
    paragraph."""
    epub = _require_pandoc_stack()
    spacer = "<td>&#160;</td>"
    inner = (
        "<table><tr>" + spacer * 2 + "<td><p>{text}</p></td>" + spacer * 2 + "</tr>"
        "<tr>" + spacer * 5 + "</tr></table>"
    )
    page = (
        '<table border="0" width="100%">'
        f"<tr><td>{inner.format(text='Handicapping starts with the form.')}</td></tr>"
        f"<tr><td>{inner.format(text='Speed figures come next.')}</td></tr>"
        "</table>"
    )
    elements, markdown = _parse(
        _write_book(epub, tmp_path / "page.epub", [("page_1.xml.xhtml", page)], [])
    )

    _no_table_placeholder(elements, markdown)
    assert not _typed(elements, ElementType.TABLE)
    assert _typed(elements, ElementType.NARRATIVE_TEXT) == [
        "Handicapping starts with the form.",
        "Speed figures come next.",
    ]


def test_nested_table_inside_a_cell_keeps_its_text(tmp_path: Path) -> None:
    epub = _require_pandoc_stack()
    table = (
        "<table><tr><td><p>Outer text.</p>"
        "<table><tr><td>Inner A</td><td>Inner B</td></tr></table>"
        "</td><td><p>Right column.</p></td></tr></table>"
    )
    elements, markdown = _parse(
        _write_book(epub, tmp_path / "nested.epub", [("n.xhtml", table)], [])
    )

    _no_table_placeholder(elements, markdown)
    joined = "\n".join(e.text for e in elements)
    for text in ("Outer text.", "Inner A", "Inner B", "Right column."):
        assert text in joined
    assert joined.index("Outer text.") < joined.index("Inner A") < joined.index("Right column.")


def test_simple_table_is_unchanged(tmp_path: Path) -> None:
    epub = _require_pandoc_stack()
    table = (
        "<table><tr><th>Gear</th><th>Ratio</th></tr>"
        "<tr><td>First</td><td>2.66</td></tr>"
        '<tr><td colspan="2">Final drive</td></tr></table>'
        "<table><tr><td>Only cell</td></tr></table>"
    )
    elements, _ = _parse(_write_book(epub, tmp_path / "simple.epub", [("s.xhtml", table)], []))

    assert _typed(elements, ElementType.TABLE) == [
        "| Gear        | Ratio |\n"
        "|-------------|-------|\n"
        "| First       | 2.66  |\n"
        "| Final drive |       |",
        "|           |\n|-----------|\n| Only cell |",
    ]


def test_anchor_inside_layout_table_cell_places_toc_title(tmp_path: Path) -> None:
    epub = _require_pandoc_stack()
    files = [
        ("front.xhtml", "<p>Front matter.</p>"),
        (
            "c1.xhtml",
            "<table><tr><td>"
            '<p><a id="s1"></a>S E C T I O N &#160; O N E</p><p>Section one body.</p>'
            '<p><span id="s2"></span>S E C T I O N &#160; T W O</p><p>Section two body.</p>'
            "</td></tr></table>",
        ),
    ]
    toc = [("c1.xhtml#s1", "Section One"), ("c1.xhtml#s2", "Section Two")]
    elements, markdown = _parse(_write_book(epub, tmp_path / "anchors.epub", files, toc))

    _no_table_placeholder(elements, markdown)
    assert [(e.text, e.section_depth) for e in elements if e.element_type == ElementType.TITLE] == [
        ("Section One", 1),
        ("Section Two", 1),
    ]
    by_text = {e.text: e for e in elements}
    assert by_text["Section one body."].section_title == "Section One"
    assert by_text["Section two body."].section_title == "Section Two"


# ---------------------------------------------------------------------------
# Placeholder logging, nested markup, spanning cells and table ids
# ---------------------------------------------------------------------------


def test_placeholder_drop_is_logged_with_a_count() -> None:
    from structlog.testing import capture_logs

    md = "\n\n".join(["Before.", "[TABLE]", "Middle.", "[TABLE]"])
    with capture_logs() as logs:
        markdown_to_elements(md, [])
    dropped = [e for e in logs if e["event"] == "epub_table_placeholder_dropped"]
    assert len(dropped) == 1
    assert dropped[0]["count"] == 2
    assert dropped[0]["log_level"] == "warning"

    with capture_logs() as logs:
        markdown_to_elements("Just text.", [])
    assert not [e for e in logs if e["event"] == "epub_table_placeholder_dropped"]


def test_line_break_nested_in_inline_markup_keeps_the_table(tmp_path: Path) -> None:
    """The gfm writer rejects a LineBreak at any depth, not just top level."""
    epub = _require_pandoc_stack()
    table = (
        "<table><tr><th>Name</th><th>Note</th></tr>"
        "<tr><td><b>Alpha<br/>Beta</b></td><td>first</td></tr>"
        "<tr><td>Gamma</td><td><i>second<br/>line</i></td></tr></table>"
    )
    elements, markdown = _parse(_write_book(epub, tmp_path / "br.epub", [("b.xhtml", table)], []))

    _no_table_placeholder(elements, markdown)
    tables = _typed(elements, ElementType.TABLE)
    assert len(tables) == 1
    for text in ("Alpha", "Beta", "Gamma", "first", "second", "line"):
        assert text in tables[0]


def test_spanning_first_column_with_block_cells_stays_one_table(tmp_path: Path) -> None:
    """Rows shortened by a rowspan still show two filled grid columns."""
    epub = _require_pandoc_stack()
    table = (
        "<table>"
        '<tr><td rowspan="3">Ferrari</td><td><div>Ascari</div></td></tr>'
        "<tr><td><div>Farina</div></td></tr>"
        "<tr><td><div>Hill</div></td></tr>"
        "</table>"
    )
    elements, markdown = _parse(_write_book(epub, tmp_path / "rs.epub", [("r.xhtml", table)], []))

    _no_table_placeholder(elements, markdown)
    tables = _typed(elements, ElementType.TABLE)
    assert len(tables) == 1
    for text in ("Ferrari", "Ascari", "Farina", "Hill"):
        assert text in tables[0]
    assert not _typed(elements, ElementType.NARRATIVE_TEXT)


def test_ids_on_a_kept_table_and_its_cells_place_toc_titles(tmp_path: Path) -> None:
    epub = _require_pandoc_stack()
    files = [
        ("front.xhtml", "<p>Front matter.</p>"),
        (
            "c1.xhtml",
            "<p>Intro one.</p>"
            '<table id="t1"><tr><th>Gear</th><th>Ratio</th></tr>'
            "<tr><td>First</td><td>2.66</td></tr></table>"
            "<p>After table one.</p>"
            "<table><tr><th>Lap</th><th>Time</th></tr>"
            '<tr><td id="lap1">One</td><td>1:32</td></tr></table>',
        ),
    ]
    toc = [("c1.xhtml#t1", "Gear Ratios"), ("c1.xhtml#lap1", "Lap Times")]
    elements, markdown = _parse(_write_book(epub, tmp_path / "tid.epub", files, toc))

    _no_table_placeholder(elements, markdown)
    assert [(e.text, e.section_depth) for e in elements if e.element_type == ElementType.TITLE] == [
        ("Gear Ratios", 1),
        ("Lap Times", 1),
    ]
    tables = [e for e in elements if e.element_type == ElementType.TABLE]
    assert [t.section_title for t in tables] == ["Gear Ratios", "Lap Times"]
    assert tables[0].text.startswith("| Gear")
    by_text = {e.text: e for e in elements}
    assert by_text["After table one."].section_title == "Gear Ratios"

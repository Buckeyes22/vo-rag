"""EPUB TOC anchors survive pandoc and place chapter TITLEs (VOR-1427).

GFM has no identifier syntax, so pandoc dropped the ``[]{#chapter01.xhtml}``
span at each spine-file start and every source id. When chapter titles are
styled ``<p>`` paragraphs whose text differs from the TOC ("C H A P T E R 1"),
title matching missed and whole books collapsed into one front-matter section
(a book with 74 TOC entries got 6 TITLEs, and thousands of elements sat
under 'Acknowledgments'). The ``epub_anchors.lua`` filter now writes each id as a
``⟦vorag-anchor:<id>⟧`` line that ``markdown_to_elements`` turns into the TOC
entry's TITLE.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from vo_rag.core.parsing.contract import ElementType
from vo_rag.core.parsing.parsers.epub_markdown import (
    markdown_to_elements,
    strip_anchor_sentinels,
)


def _titles(elements: list[Any]) -> list[tuple[str, int]]:
    return [(e.text, e.section_depth) for e in elements if e.element_type == ElementType.TITLE]


def _anchor(anchor_id: str) -> str:
    return f"⟦vorag-anchor:{anchor_id}⟧"


# ---------------------------------------------------------------------------
# markdown_to_elements with sentinel lines (no pandoc needed)
# ---------------------------------------------------------------------------

_STYLED_TOC = [
    {"depth": 1, "title": "Chapter One: Growth", "href": "Text/c1.xhtml"},
    {"depth": 2, "title": "Betting Fractions", "href": "Text/c1.xhtml#s1"},
    {"depth": 1, "title": "Chapter Two: Risk", "href": "Text/c2.xhtml"},
]


def test_file_start_and_fragment_anchors_become_toc_titles() -> None:
    md = "\n\n".join(
        [
            _anchor("front.xhtml"),
            "Acknowledgments go here.",
            _anchor("c1.xhtml"),
            "C H A P T E R 1",
            "Body of chapter one.",
            _anchor("c1.xhtml#s1"),
            "Fraction text.",
            _anchor("c2.xhtml"),
            "C H A P T E R 2",
            "Body of chapter two.",
        ]
    )
    elements = markdown_to_elements(md, _STYLED_TOC)

    assert _titles(elements) == [
        ("Chapter One: Growth", 1),
        ("Betting Fractions", 2),
        ("Chapter Two: Risk", 1),
    ]
    by_text = {e.text: e for e in elements}
    assert by_text["Acknowledgments go here."].section_title == ""
    assert by_text["Body of chapter one."].section_title == "Chapter One: Growth"
    assert by_text["Fraction text."].section_title == "Betting Fractions"
    assert by_text["Body of chapter two."].section_title == "Chapter Two: Risk"
    assert not any("vorag-anchor" in e.text for e in elements)


def test_unmatched_anchors_are_dropped_silently() -> None:
    md = "\n\n".join([_anchor("c9.xhtml#x"), _anchor("nav.xhtml"), "Just text."])
    elements = markdown_to_elements(md, _STYLED_TOC)
    assert [e.text for e in elements] == ["Just text."]
    assert markdown_to_elements(md, []) == markdown_to_elements("Just text.", [])


def test_percent_encoded_ids_and_pandoc_311_underscore_ids_match() -> None:
    """pandoc >= 3.2 writes file-scoped ids as ``c1.xhtml_s1``; the filter
    percent-encodes everything outside [A-Za-z0-9.-#:/]."""
    md = "\n\n".join(
        [
            _anchor("c1.xhtml"),
            "Body one.",
            _anchor("c1.xhtml%5Fs1"),
            "Fraction text.",
        ]
    )
    assert _titles(markdown_to_elements(md, _STYLED_TOC)) == [
        ("Chapter One: Growth", 1),
        ("Betting Fractions", 2),
    ]


def test_anchor_then_matching_heading_line_is_one_title() -> None:
    md = "\n\n".join(
        [
            _anchor("c1.xhtml"),
            "# **Chapter One: Growth**",
            "Body one.",
            _anchor("c2.xhtml"),
            "Chapter Two: Risk",
            "Body two.",
        ]
    )
    assert _titles(markdown_to_elements(md, _STYLED_TOC)) == [
        ("Chapter One: Growth", 1),
        ("Chapter Two: Risk", 1),
    ]


def test_text_match_then_anchor_for_same_entry_is_one_title() -> None:
    md = "\n\n".join(
        [
            "Chapter One: Growth",  # plain line consumes the entry first
            _anchor("c1.xhtml"),
            "Body one.",
        ]
    )
    elements = markdown_to_elements(md, _STYLED_TOC)
    assert _titles(elements) == [("Chapter One: Growth", 1)]
    assert elements[-1].section_title == "Chapter One: Growth"


def test_anchors_inside_the_toc_page_are_skipped() -> None:
    md = "\n\n".join(
        [
            _anchor("toc.xhtml"),
            "[Chapter One: Growth](#c1.xhtml)",
            _anchor("c1.xhtml"),  # a stray id on the TOC page itself
            "[Betting Fractions](#c1.xhtml#s1)",
            "[Chapter Two: Risk](#c2.xhtml)",
            _anchor("c1.xhtml"),
            "Body one.",
        ]
    )
    elements = markdown_to_elements(md, _STYLED_TOC)
    assert _titles(elements) == [("Chapter One: Growth", 1)]
    assert elements[-1].section_title == "Chapter One: Growth"


def test_file_start_falls_back_to_first_fragment_entry_without_its_own_anchor() -> None:
    """calibre split files: TOC href ``index_split_003.html#p26`` whose id was on
    a ``<p>`` (pandoc drops those) is placed at the file start."""
    toc = [
        {"depth": 1, "title": "Chapter 1", "href": "index_split_003.html#p26"},
        {"depth": 2, "title": "Section 1.1", "href": "index_split_003.html#p40"},
        {"depth": 1, "title": "Chapter 2", "href": "index_split_004.html#p90"},
    ]
    md = "\n\n".join(
        [
            _anchor("index_split_003.html"),
            "C H A P T E R 1",
            "Body one.",
            _anchor("index_split_003.html#p40"),
            "Section text.",
            _anchor("index_split_004.html"),
            "Front of file four.",
            _anchor("index_split_004.html#p90"),
            "Body two.",
        ]
    )
    elements = markdown_to_elements(md, toc)
    assert _titles(elements) == [("Chapter 1", 1), ("Section 1.1", 2), ("Chapter 2", 1)]
    by_text = {e.text: e for e in elements}
    # p90 has its own anchor, so the file start does not take it early.
    assert by_text["Front of file four."].section_title == "Section 1.1"
    assert by_text["Body two."].section_title == "Chapter 2"


def test_cjk_book_with_backlinked_headings_gets_one_title_per_entry() -> None:
    """de9420be5179: body headings are ``# [title](#content.xhtml#tocN)``
    back-links after a BOM line, and titles normalized to their digits
    ("项目2 …" and "任务2 …" both became "2")."""
    toc = [
        {"depth": 1, "title": "目录", "href": "contents/content.xhtml"},
        {"depth": 1, "title": "项目1 汽车维修", "href": "contents/Chapter_3.xhtml"},
        {"depth": 1, "title": "项目2 汽车分类", "href": "contents/Chapter_4_1.xhtml"},
        {"depth": 2, "title": "任务2 汽车VIN码", "href": "contents/Chapter_4_2.xhtml"},
    ]
    md = "\n\n".join(
        [
            _anchor("content.xhtml"),
            "\ufeff",
            "目录",
            _anchor("content.xhtml#toc5"),
            "[项目1 汽车维修](#Chapter_3.xhtml)",
            "[项目2 汽车分类](#Chapter_4_1.xhtml)",
            "[任务2 汽车VIN码](#Chapter_4_2.xhtml)",
            _anchor("Chapter%5F3.xhtml"),
            "\ufeff",
            "# [项目1 汽车维修](#content.xhtml#toc5)",
            "正文一。",
            "2",
            _anchor("Chapter%5F4%5F1.xhtml"),
            "\ufeff",
            "# [项目2 汽车分类](#content.xhtml#toc6)",
            "正文二。",
            _anchor("Chapter%5F4%5F2.xhtml"),
            "## [任务2 汽车VIN码](#content.xhtml#toc8)",
            "正文三。",
        ]
    )
    elements = markdown_to_elements(md, toc)
    assert _titles(elements) == [
        ("目录", 1),
        ("项目1 汽车维修", 1),
        ("项目2 汽车分类", 1),
        ("任务2 汽车VIN码", 2),
    ]
    by_text = {e.text: e for e in elements}
    assert by_text["2"].section_title == "项目1 汽车维修"
    assert by_text["正文三。"].section_title == "任务2 汽车VIN码"
    assert "\ufeff" not in by_text


def test_strip_anchor_sentinels_removes_only_sentinel_lines() -> None:
    md = "\n\n".join([_anchor("c1.xhtml"), "Body one.", _anchor("c1.xhtml#s1"), "Two."])
    assert strip_anchor_sentinels(md) == "Body one.\n\nTwo."


# ---------------------------------------------------------------------------
# Through pandoc + the Lua filter
# ---------------------------------------------------------------------------


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


def test_styled_paragraph_chapters_become_titles(tmp_path: Path) -> None:
    epub = _require_pandoc_stack()
    files = [
        ("front.xhtml", "<p>Acknowledgments to many people.</p>"),
        (
            "c1.xhtml",
            '<p class="h1">C H A P T E R 1</p><p class="h1">Growth</p><p>Body one.</p>',
        ),
        ("c2.xhtml", '<p class="h1">C H A P T E R 2</p><p>Body two.</p>'),
    ]
    toc = [("c1.xhtml", "Chapter 1: Growth"), ("c2.xhtml", "Chapter 2: Risk")]
    elements, markdown = _parse(_write_book(epub, tmp_path / "styled.epub", files, toc))

    assert _titles(elements) == [("Chapter 1: Growth", 1), ("Chapter 2: Risk", 1)]
    by_text = {e.text: e for e in elements}
    assert by_text["Body one."].section_title == "Chapter 1: Growth"
    assert by_text["Body two."].section_title == "Chapter 2: Risk"
    assert by_text["Acknowledgments to many people."].section_title == ""
    assert not any("vorag-anchor" in e.text for e in elements)
    assert "vorag-anchor" in markdown


def test_calibre_split_file_fragment_anchors_place_titles(tmp_path: Path) -> None:
    epub = _require_pandoc_stack()
    files = [
        (
            "index_split_000.xhtml",
            "<p>Front matter.</p>"
            '<p class="c"><a id="p26"></a><b>C H A P T E R 1</b></p><p>Body one.</p>'
            '<div id="p40"><p class="c">1.1 Odds</p></div><p>Odds text.</p>'
            '<p><span id="p55"></span>C H A P T E R 2</p><p>Body two.</p>',
        ),
    ]
    toc = [
        ("index_split_000.xhtml#p26", "Chapter 1 Basics"),
        ("index_split_000.xhtml#p40", "Odds and Ends"),
        ("index_split_000.xhtml#p55", "Chapter 2 More"),
    ]
    elements, _ = _parse(_write_book(epub, tmp_path / "split.epub", files, toc))

    by_text = {e.text: e for e in elements}
    assert by_text["Front matter."].section_title == ""
    assert by_text["Body one."].section_title == "Chapter 1 Basics"
    assert by_text["Odds text."].section_title == "Odds and Ends"
    assert by_text["Body two."].section_title == "Chapter 2 More"
    titles = _titles(elements)
    assert [t for t, _ in titles] == ["Chapter 1 Basics", "Odds and Ends", "Chapter 2 More"]


def test_toc_targets_on_paragraph_ids_are_kept(tmp_path: Path) -> None:
    """pandoc drops ids on ``<p>`` (and ``<img>``); calibre PDF conversions
    put TOC targets there (c8865c4a9058: ``<p id="page_17">``)."""
    epub = _require_pandoc_stack()
    files = [
        (
            "index_split_001.xhtml",
            '<p id="page_11" class="c1">Preface</p><p>Preface text.</p>'
            '<p class="c1" id="page_17"><img src="p17.png" alt=""/></p><p>1</p>'
            "<p>Intro text.</p>"
            '<p><img id="page_29" src="p29.png" alt=""/></p><p>Market text.</p>'
            '<p id="not_a_target">Untouched.</p>',
        ),
    ]
    toc = [
        ("index_split_001.xhtml#page_11", "Preface ix"),
        ("index_split_001.xhtml#page_17", "1. Introduction 1"),
        ("index_split_001.xhtml#page_29", "2. Market Perspectives 13"),
    ]
    elements, markdown = _parse(_write_book(epub, tmp_path / "pids.epub", files, toc))

    by_text = {e.text: e for e in elements}
    assert by_text["Preface text."].section_title == "Preface ix"
    assert by_text["Intro text."].section_title == "1. Introduction 1"
    assert by_text["Market text."].section_title == "2. Market Perspectives 13"
    assert [t for t, _ in _titles(elements)] == [
        "Preface ix",
        "1. Introduction 1",
        "2. Market Perspectives 13",
    ]
    assert "not%5Fa%5Ftarget" not in markdown and "not_a_target" not in markdown


def test_toc_page_links_and_real_headings_give_one_title_each(tmp_path: Path) -> None:
    epub = _require_pandoc_stack()
    files = [
        (
            "contents.xhtml",
            "<p>Contents</p>"
            '<p><a id="t1" href="c1.xhtml">Chapter One</a></p>'
            '<p><a id="t2" href="c2.xhtml">Chapter Two</a></p>',
        ),
        ("c1.xhtml", '<h1 id="h">Chapter One</h1><p>Body one.</p>'),
        ("c2.xhtml", "<h1>Chapter Two</h1><p>Body two.</p>"),
    ]
    toc = [("c1.xhtml", "Chapter One"), ("c2.xhtml", "Chapter Two")]
    elements, _ = _parse(_write_book(epub, tmp_path / "tocpage.epub", files, toc))

    assert _titles(elements) == [("Chapter One", 1), ("Chapter Two", 1)]
    by_text = {e.text: e for e in elements}
    assert by_text["Body one."].section_title == "Chapter One"
    assert by_text["Body two."].section_title == "Chapter Two"


def test_parsed_document_markdown_has_no_sentinels(tmp_path: Path) -> None:
    epub = _require_pandoc_stack()
    from vo_rag.core.parsing.parsers.epub import EpubParser
    from vo_rag.core.storage.figures import FigureStore

    files = [("c1.xhtml", '<p class="h1">ONE</p><p>Body one.</p>')]
    path = _write_book(epub, tmp_path / "md.epub", files, [("c1.xhtml", "Chapter One")])

    parsed = EpubParser(FigureStore(tmp_path / "figs"))._parse_sync(str(path))
    assert "vorag-anchor" not in parsed.markdown
    assert "Body one." in parsed.markdown
    assert [e.text for e in parsed.elements if e.element_type == ElementType.TITLE] == [
        "Chapter One"
    ]


# ---------------------------------------------------------------------------
# Label lines, images and backlinks around anchors
# ---------------------------------------------------------------------------

_INTRO_TOC = [
    {"depth": 1, "title": "Introduction", "href": "Text/c1.xhtml"},
    {"depth": 1, "title": "Growth", "href": "Text/c2.xhtml"},
]


def _kinds(elements: list[Any]) -> list[str]:
    return [str(e.element_type.value) for e in elements]


@pytest.mark.parametrize(
    "heading",
    ["# Introduction", "# [Introduction](#toc.xhtml#t1)"],
)
def test_label_line_between_anchor_and_heading_is_still_one_title(heading: str) -> None:
    md = "\n\n".join([_anchor("c1.xhtml"), "CHAPTER 1", heading, "Body one."])
    elements = markdown_to_elements(md, _INTRO_TOC)
    assert _titles(elements) == [("Introduction", 1)]
    assert [e.text for e in elements if e.element_type != ElementType.TITLE] == [
        "CHAPTER 1",
        "Body one.",
    ]
    assert all(e.section_title == "Introduction" for e in elements)


def test_image_between_anchor_and_heading_is_still_one_title() -> None:
    md = "\n\n".join(
        [_anchor("c1.xhtml"), "![](images/ornament.jpg)", "# Introduction", "Body one."]
    )
    elements = markdown_to_elements(md, _INTRO_TOC)
    assert _titles(elements) == [("Introduction", 1)]


def test_backlink_to_an_anchor_consumed_entry_after_another_title_is_not_a_title() -> None:
    md = "\n\n".join(
        [
            _anchor("c1.xhtml"),
            "Body one.",
            _anchor("c2.xhtml"),
            "Body two.",
            "[Introduction](#toc.xhtml#t1)",  # stray back-link, entry already taken
        ]
    )
    assert _titles(markdown_to_elements(md, _INTRO_TOC)) == [("Introduction", 1), ("Growth", 1)]


def test_chapter_label_and_h1_book_without_anchors_is_unchanged() -> None:
    md = "\n\n".join(["CHAPTER 1", "# Introduction", "Body one.", "# Growth", "Body two."])
    elements = markdown_to_elements(md, _INTRO_TOC)
    assert _titles(elements) == [("Introduction", 1), ("Growth", 1)]
    assert elements[0].text == "CHAPTER 1"


def test_chapter_label_and_h1_book_through_pandoc(tmp_path: Path) -> None:
    epub = _require_pandoc_stack()
    files = [
        ("c1.xhtml", "<p>CHAPTER 1</p><h1>Introduction</h1><p>Body one.</p>"),
        ("c2.xhtml", "<p>CHAPTER 2</p><h1>Growth</h1><p>Body two.</p>"),
    ]
    toc = [("c1.xhtml", "Introduction"), ("c2.xhtml", "Growth")]
    elements, _ = _parse(_write_book(epub, tmp_path / "labels.epub", files, toc))
    assert _titles(elements) == [("Introduction", 1), ("Growth", 1)]
    by_text = {e.text: e for e in elements}
    assert by_text["CHAPTER 2"].section_title == "Growth"


def test_book_text_that_contains_a_sentinel_is_not_an_anchor() -> None:
    line = f"Quoted {_anchor('c1.xhtml')} inside a sentence."
    md = "\n\n".join([line, f"{_anchor('c1.xhtml')} trailing text", "Body."])
    elements = markdown_to_elements(md, _INTRO_TOC)
    assert _titles(elements) == []
    assert [e.text for e in elements] == [line, f"{_anchor('c1.xhtml')} trailing text", "Body."]


def test_caption_heading_per_image_book_is_not_image_only() -> None:
    from vo_rag.core.parsing.parsers.epub import is_image_only

    pages = [f"## Plate {i}: the rear wing\n\n![](images/p{i}.jpg)" for i in range(30)]
    assert is_image_only(markdown_to_elements("\n\n".join(pages), [])) is False


def test_anchor_titles_do_not_rescue_a_page_scan_from_image_only() -> None:
    from vo_rag.core.parsing.parsers.epub import is_image_only

    toc = [{"depth": 1, "title": f"Page {i}", "href": f"p{i}.xhtml"} for i in range(30)]
    pages = [f"{_anchor(f'p{i}.xhtml')}\n\n![](images/p{i}.jpg)" for i in range(30)]
    elements = markdown_to_elements("\n\n".join(pages), toc)
    assert len(_titles(elements)) == 30
    assert is_image_only(elements) is True


def _zip(path: Path, members: dict[str, str]) -> Path:
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        for name, text in members.items():
            zf.writestr(name, text)
    return path


def test_toc_target_rewrite_is_linear_on_adversarial_markup(tmp_path: Path) -> None:
    import time

    from vo_rag.core.parsing.parsers.epub import _toc_target_id_rewrites

    unclosed_ids = "<p" + ' id="a"' * 30_000  # ~210 KB, no closing '>'
    long_name = "<" + "a" * 200_000
    path = _zip(
        tmp_path / "bomb.epub",
        {"OEBPS/content.opf": "<package/>", "OEBPS/ch.xhtml": unclosed_ids + long_name},
    )
    start = time.perf_counter()
    _toc_target_id_rewrites(path, [{"href": "ch.xhtml#a"}])
    assert time.perf_counter() - start < 1.0


def test_toc_target_rewrite_uses_full_archive_paths_and_skips_span_invalid_tags(
    tmp_path: Path,
) -> None:
    from vo_rag.core.parsing.parsers.epub import _toc_target_id_rewrites

    path = _zip(
        tmp_path / "paths.epub",
        {
            "OEBPS/content.opf": "<package/>",
            "OEBPS/a/ch.xhtml": '<p id="p1">A</p>',
            "OEBPS/b/ch.xhtml": '<p class="x" id="p1">B</p>',
            "OEBPS/c/tables.xhtml": (
                '<body id="bd"><table id="t1"><tbody id="tb"><tr id="r1"><td>x</td></tr>'
                '</tbody></table><ul id="u1"><li>y</li></ul></body>'
            ),
        },
    )
    toc = [
        {"href": "b/ch.xhtml#p1"},
        *({"href": f"c/tables.xhtml#{i}"} for i in ("bd", "t1", "tb", "r1", "u1")),
    ]
    rewrites = _toc_target_id_rewrites(path, toc)
    assert set(rewrites) == {"OEBPS/b/ch.xhtml"}
    assert rewrites["OEBPS/b/ch.xhtml"] == b'<p class="x"><span id="p1"></span>B</p>'


def test_sentinel_right_after_a_pipe_table_ends_the_table_first() -> None:
    """VOR-1438 groups pipe rows into one TABLE; an anchor line closes it
    before the chapter TITLE, so the table stays in the previous section."""
    md = "\n".join(
        [
            "| Year | Bank |",
            "|------|------|",
            "| 1956 | 100 |",
            _anchor("c1.xhtml"),
            "",
            "Body one.",
        ]
    )
    elements = markdown_to_elements(md, _INTRO_TOC)
    assert [e.element_type for e in elements] == [
        ElementType.TABLE,
        ElementType.TITLE,
        ElementType.NARRATIVE_TEXT,
    ]
    assert elements[0].text.splitlines()[-1] == "| 1956 | 100 |"
    assert elements[0].section_title == ""
    assert elements[2].section_title == "Introduction"

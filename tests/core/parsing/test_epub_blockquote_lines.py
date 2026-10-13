"""Contents pages and body text inside GFM blockquotes (VOR-1427).

d076890761f1 (a calibre ``index_split_*.html`` book, file-level TOC hrefs)
prints its contents page as links to ``filepos`` fragments, sub-entries nested
in blockquotes (``> > [9.1 ...](#index_split_013.html#filepos968757)``). The
chapter lines linked into the chapter's own file were read as back-link
headings, so every chapter TITLE landed on the contents page, the file-start
anchors found their entries taken, and a closing "Table of Contents" page
re-emitted all nine: 5,193 of 5,377 elements sat under chapter 9.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from vo_rag.core.parsing.contract import ElementType
from vo_rag.core.parsing.parsers.epub_markdown import markdown_to_elements


def _titles(elements: list[Any]) -> list[tuple[str, int]]:
    return [(e.text, e.section_depth) for e in elements if e.element_type == ElementType.TITLE]


def _anchor(anchor_id: str) -> str:
    return f"⟦vorag-anchor:{anchor_id}⟧"


_FILE_TOC = [
    {"depth": 1, "title": "Front Matter", "href": "index_split_000.html"},
    {"depth": 1, "title": "1. Introduction", "href": "index_split_005.html"},
    {"depth": 1, "title": "2. Tires", "href": "index_split_006.html"},
]


def test_contents_links_into_the_entry_file_are_not_headings() -> None:
    md = "\n\n".join(
        [
            _anchor("index%5Fsplit%5F000.html"),
            "Front matter text.",
            _anchor("index%5Fsplit%5F004.html"),
            "Contents",
            "[1 Introduction](#index_split_005.html#filepos42442)",
            "> > [1.1 System and System Dynamics](#index_split_005.html#filepos44916)",
            "> > > [1.1.1 Research Contents](#index_split_005.html#filepos59028)",
            "[2 Tires](#index_split_006.html#filepos113761)",
            "> > [2.1 Construction of Tires](#index_split_006.html#filepos115401)",
            _anchor("index%5Fsplit%5F005.html"),
            "1. Introduction",
            "Body of chapter one.",
            "> > College of Energy and Power Engineering",
            _anchor("index%5Fsplit%5F006.html"),
            "2. Tires",
            "Body of chapter two.",
            _anchor("index%5Fsplit%5F014.html"),
            "Table of Contents",
            "[Front Matter](#index_split_000.html#filepos173)",
            "[1. Introduction](#index_split_005.html#filepos42442)",
            "[2. Tires](#index_split_006.html#filepos113761)",
        ]
    )
    elements = markdown_to_elements(md, _FILE_TOC)

    assert _titles(elements) == [("Front Matter", 1), ("1. Introduction", 1), ("2. Tires", 1)]
    by_text = {e.text: e for e in elements}
    assert by_text["Body of chapter one."].section_title == "1. Introduction"
    assert by_text["College of Energy and Power Engineering"].section_title == "1. Introduction"
    assert by_text["Body of chapter two."].section_title == "2. Tires"
    assert by_text["Table of Contents"].section_title == "2. Tires"
    # The contents page lines stay narrative, without their quote markers.
    assert by_text["[2.1 Construction of Tires](#index_split_006.html#filepos115401)"]
    assert not any(e.text.startswith(">") for e in elements)


def test_entries_sharing_an_href_all_open_at_its_anchor() -> None:
    """3cedb67ba6d0 lists chapters 43 and 44 at one href; its contents page
    links 43 into the same file. The text after the anchor is the last one's."""
    toc = [
        {"depth": 2, "title": "42. Golf", "href": "index_split_007.html#p414"},
        {"depth": 2, "title": "43. e-Sports", "href": "index_split_007.html#p432"},
        {"depth": 2, "title": "44. Gambling 101", "href": "index_split_007.html#p432"},
    ]
    md = "\n\n".join(
        [
            "[42. Golf](#index_split_007.html#p414)",
            "[43. e-Sports](#index_split_007.html#p421)",
            "[44.](#index_split_007.html#p432) [Gambling 101](#index_split_007.html#p432)",
            _anchor("index%5Fsplit%5F007.html#p414"),
            "Golf text.",
            _anchor("index%5Fsplit%5F007.html#p432"),
            "Gambling text.",
        ]
    )
    elements = markdown_to_elements(md, toc)
    assert _titles(elements) == [("42. Golf", 2), ("43. e-Sports", 2), ("44. Gambling 101", 2)]
    by_text = {e.text: e for e in elements}
    assert by_text["Golf text."].section_title == "42. Golf"
    assert by_text["Gambling text."].section_title == "44. Gambling 101"


_FRAGMENT_TOC = [
    {"depth": 1, "title": "Chapter One: Growth", "href": "Text/c1.xhtml#c1"},
    {"depth": 2, "title": "Betting Fractions", "href": "Text/c1.xhtml#s1"},
    {"depth": 1, "title": "Chapter Two: Risk", "href": "Text/c2.xhtml#c2"},
]


def test_blockquoted_contents_page_and_body_anchors_place_titles_in_the_body() -> None:
    md = "\n\n".join(
        [
            "> [Chapter One: Growth](#c1.xhtml#c1)",
            "> > [Betting Fractions](#c1.xhtml#s1)",
            "> Betting Fractions",  # an unlinked entry of the quoted contents page
            "> [Chapter Two: Risk](#c2.xhtml#c2)",
            "> " + _anchor("c1.xhtml#c1"),
            "> C H A P T E R 1",
            "Body of chapter one.",
            "> > " + _anchor("c1.xhtml#s1"),
            "Fraction text.",
            "> " + _anchor("c2.xhtml#c2"),
            "Body of chapter two.",
        ]
    )
    elements = markdown_to_elements(md, _FRAGMENT_TOC)

    assert _titles(elements) == [
        ("Chapter One: Growth", 1),
        ("Betting Fractions", 2),
        ("Chapter Two: Risk", 1),
    ]
    by_text = {e.text: e for e in elements}
    assert by_text["Betting Fractions"].element_type == ElementType.TITLE
    assert by_text["Body of chapter one."].section_title == "Chapter One: Growth"
    assert by_text["Fraction text."].section_title == "Betting Fractions"
    assert by_text["Body of chapter two."].section_title == "Chapter Two: Risk"
    assert not any("vorag-anchor" in e.text for e in elements)


def test_blockquoted_narrative_heading_and_list_lose_their_markers() -> None:
    md = "\n".join(
        [
            "> ## Sidebar Title",
            ">",
            "> Quoted paragraph one.",
            ">",
            "> > Nested quote.",
            "> - quoted item",
        ]
    )
    elements = markdown_to_elements(md, [])
    assert [(e.element_type, e.text) for e in elements] == [
        (ElementType.TITLE, "Sidebar Title"),
        (ElementType.NARRATIVE_TEXT, "Quoted paragraph one."),
        (ElementType.NARRATIVE_TEXT, "Nested quote."),
        (ElementType.LIST_ITEM, "quoted item"),
    ]


def test_blockquoted_pipe_table_is_one_table() -> None:
    md = "\n".join(["Before.", "", "> | a | b |", "> |---|---|", "> | 1 | 2 |", "", "After."])
    elements = markdown_to_elements(md, [])
    tables = [e for e in elements if e.element_type == ElementType.TABLE]
    assert [t.text for t in tables] == ["| a | b |\n|---|---|\n| 1 | 2 |"]
    assert [e.text for e in elements] == ["Before.", tables[0].text, "After."]


def test_blockquoted_code_fence_is_one_code_block_without_markers() -> None:
    md = "\n".join(["> ``` python", "> x = 1", ">", ">     y = 2", "> ```", "", "After."])
    elements = markdown_to_elements(md, [])
    assert [(e.element_type, e.text) for e in elements] == [
        (ElementType.CODE_BLOCK, "x = 1\n\n    y = 2"),
        (ElementType.NARRATIVE_TEXT, "After."),
    ]


def test_tilde_fenced_code_shields_heading_and_table_syntax() -> None:
    md = "\n".join(
        [
            "~~~python",
            "# This is source code, not a chapter title",
            "| this | is code |",
            "|---|---|",
            "~~~",
            "",
            "After the code block.",
        ]
    )
    elements = markdown_to_elements(md, [])

    assert [(e.element_type, e.text) for e in elements] == [
        (
            ElementType.CODE_BLOCK,
            "# This is source code, not a chapter title\n| this | is code |\n|---|---|",
        ),
        (ElementType.NARRATIVE_TEXT, "After the code block."),
    ]


def test_code_fence_closes_only_on_matching_character_and_sufficient_run() -> None:
    md = "\n".join(
        [
            "````python",
            "~~~",
            "```",
            "# still code",
            "`````",
            "After the code block.",
        ]
    )
    elements = markdown_to_elements(md, [])

    assert [(e.element_type, e.text) for e in elements] == [
        (ElementType.CODE_BLOCK, "~~~\n```\n# still code"),
        (ElementType.NARRATIVE_TEXT, "After the code block."),
    ]


def test_blockquoted_tilde_fence_is_one_code_block_without_markers() -> None:
    md = "\n".join(["> ~~~python", "> # code", "> | a | b |", "> ~~~", "", "After."])
    elements = markdown_to_elements(md, [])

    assert [(e.element_type, e.text) for e in elements] == [
        (ElementType.CODE_BLOCK, "# code\n| a | b |"),
        (ElementType.NARRATIVE_TEXT, "After."),
    ]


def test_quote_markers_inside_an_unquoted_code_block_are_code() -> None:
    md = "\n".join(["```", "> summary(x)", ">", ">>> 1 + 1", "```", "", "After."])
    elements = markdown_to_elements(md, [])
    assert [(e.element_type, e.text) for e in elements] == [
        (ElementType.CODE_BLOCK, "> summary(x)\n>\n>>> 1 + 1"),
        (ElementType.NARRATIVE_TEXT, "After."),
    ]


# ---------------------------------------------------------------------------
# Through pandoc + the Lua filter
# ---------------------------------------------------------------------------


def _write_book(epub: Any, path: Path, files: list[tuple[str, str]], toc: list[tuple[str, str]]):
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


def test_pandoc_blockquoted_contents_page_and_heading(tmp_path: Path) -> None:
    epub = pytest.importorskip("ebooklib.epub")
    pytest.importorskip("pypandoc")
    if shutil.which("pandoc") is None:
        pytest.skip("pandoc binary not installed")
    from vo_rag.core.parsing.parsers.epub import _read_epub_and_convert

    contents = (
        "<p>Contents</p>"
        '<p><a href="c1.xhtml#p1">1 Growth</a></p>'
        '<blockquote><blockquote><p><a href="c1.xhtml#p2">1.1 Odds</a></p>'
        '<blockquote><p><a href="c1.xhtml#p3">1.1.1 Edges</a></p></blockquote>'
        "</blockquote></blockquote>"
        '<p><a href="c2.xhtml#p4">2 Risk</a></p>'
    )
    files = [
        ("front.xhtml", "<p>Front matter.</p>"),
        ("contents.xhtml", contents),
        (
            "c1.xhtml",
            '<blockquote><p class="h1">1. Growth</p></blockquote>'
            "<p>Body one.</p>"
            "<blockquote><p>Quoted sidebar text.</p></blockquote>",
        ),
        ("c2.xhtml", "<blockquote><h2>2. Risk</h2></blockquote><p>Body two.</p>"),
    ]
    toc = [("front.xhtml", "Front"), ("c1.xhtml", "1. Growth"), ("c2.xhtml", "2. Risk")]
    path = _write_book(epub, tmp_path / "quoted.epub", files, toc)
    _, _, toc_tree, markdown = _read_epub_and_convert(str(path), str(path), source_format="epub")
    assert "> > [1.1 Odds]" in markdown
    elements = markdown_to_elements(markdown, toc_tree)

    assert _titles(elements) == [("Front", 1), ("1. Growth", 1), ("2. Risk", 1)]
    by_text = {e.text: e for e in elements}
    assert by_text["Contents"].section_title == "Front"
    assert by_text["Body one."].section_title == "1. Growth"
    assert by_text["Quoted sidebar text."].section_title == "1. Growth"
    assert by_text["Body two."].section_title == "2. Risk"
    assert not any(e.text.startswith(">") for e in elements)


# ---------------------------------------------------------------------------
# File-level entries, code-block quotes and self-linked headings
# ---------------------------------------------------------------------------


def test_file_level_entries_sharing_one_file_open_one_at_a_time() -> None:
    """One/Two/Three all at ``book.html``: the file-start anchor opens only
    the first; the others come from their own title lines."""
    toc = [
        {"depth": 1, "title": "One", "href": "Text/book.html"},
        {"depth": 1, "title": "Two", "href": "Text/book.html"},
        {"depth": 1, "title": "Three", "href": "Text/book.html"},
    ]
    md = "\n\n".join(
        [_anchor("book.html"), "One", "Text one.", "Two", "Text two.", "Three", "Text three."]
    )
    elements = markdown_to_elements(md, toc)
    assert _titles(elements) == [("One", 1), ("Two", 1), ("Three", 1)]
    by_text = {e.text: e for e in elements}
    assert by_text["Text one."].section_title == "One"
    assert by_text["Text two."].section_title == "Two"
    assert by_text["Text three."].section_title == "Three"


def test_quote_markers_of_an_indented_code_block_are_code() -> None:
    """pandoc's gfm writer indents class-less <pre> by four spaces."""
    md = "\n".join(["Before.", "", "    >>> 1 + 1", "    > summary(x)", "", "After."])
    texts = [e.text for e in markdown_to_elements(md, [])]
    assert ">>> 1 + 1" in texts
    assert "> summary(x)" in texts


_SELF_TOC = [
    {"depth": 1, "title": "1 Introduction", "href": "Text/c1.xhtml"},
    {"depth": 1, "title": "2 Methods", "href": "Text/c2.xhtml"},
]


def test_self_linked_body_heading_outside_a_contents_cluster_is_a_title() -> None:
    md = "\n\n".join(
        [
            "Front text.",
            "[1 Introduction](#c1.xhtml#h1)",
            "Body one.",
            *[f"Filler {i}." for i in range(70)],
            "[2 Methods](#c2.xhtml#h2)",
            "Body two.",
        ]
    )
    elements = markdown_to_elements(md, _SELF_TOC)
    assert _titles(elements) == [("1 Introduction", 1), ("2 Methods", 1)]
    by_text = {e.text: e for e in elements}
    assert by_text["Body one."].section_title == "1 Introduction"
    assert by_text["Body two."].section_title == "2 Methods"


def test_same_file_links_in_a_contents_cluster_are_contents_lines() -> None:
    md = "\n\n".join(
        [
            "Contents",
            "[1 Introduction](#c1.xhtml#p10)",
            "[2 Methods](#c2.xhtml#p20)",
            _anchor("c1.xhtml"),
            "Body one.",
            _anchor("c2.xhtml"),
            "Body two.",
        ]
    )
    elements = markdown_to_elements(md, _SELF_TOC)
    assert _titles(elements) == [("1 Introduction", 1), ("2 Methods", 1)]
    by_text = {e.text: e for e in elements}
    assert by_text["[1 Introduction](#c1.xhtml#p10)"].section_title == ""
    assert by_text["Body one."].section_title == "1 Introduction"
    assert by_text["Body two."].section_title == "2 Methods"


def test_self_linked_hash_heading_is_titled_with_its_label() -> None:
    md = "\n\n".join(["# [**1 Introduction**](#c1.xhtml#x)", "Body one."])
    elements = markdown_to_elements(md, _SELF_TOC)
    assert _titles(elements) == [("1 Introduction", 1)]


# ---------------------------------------------------------------------------
# Contents clusters
# ---------------------------------------------------------------------------

_CH_TOC = [
    {"depth": 1, "title": "Preface", "href": "Text/pre.xhtml#p0"},
    {"depth": 1, "title": "Chapter One: Growth", "href": "Text/c1.xhtml#c1"},
    {"depth": 1, "title": "Chapter Two: Risk", "href": "Text/c2.xhtml#c2"},
]


def test_chapter_anchors_between_two_nearby_contents_pages_become_titles() -> None:
    """A brief and a detailed contents page with short front matter between:
    the TOC entry anchors in the gap open their chapters."""
    md = "\n\n".join(
        [
            "Brief Contents",
            "[Preface](#pre.xhtml#p0)",
            "[Chapter One: Growth](#c1.xhtml#c1)",
            "[Chapter Two: Risk](#c2.xhtml#c2)",
            _anchor("pre.xhtml#p0"),
            "PREFACE",
            "A short preface.",
            "Contents",
            "[Preface](#pre.xhtml#p0)",
            "[Chapter One: Growth](#c1.xhtml#c1)",
            "[Chapter Two: Risk](#c2.xhtml#c2)",
            _anchor("c1.xhtml#c1"),
            "Body one.",
            _anchor("c2.xhtml#c2"),
            "Body two.",
        ]
    )
    elements = markdown_to_elements(md, _CH_TOC)
    assert _titles(elements) == [
        ("Preface", 1),
        ("Chapter One: Growth", 1),
        ("Chapter Two: Risk", 1),
    ]
    by_text = {e.text: e for e in elements}
    assert by_text["A short preface."].section_title == "Preface"
    assert by_text["Body one."].section_title == "Chapter One: Growth"
    assert by_text["Body two."].section_title == "Chapter Two: Risk"


def test_one_contents_page_with_short_separator_lines_stays_one_cluster() -> None:
    md = "\n\n".join(
        [
            _anchor("index%5Fsplit%5F004.html"),
            "Contents",
            "[1 Introduction](#index_split_005.html#filepos42442)",
            "(1)",
            "—",
            _anchor("index%5Fsplit%5F004.html#calibre%5Fpb%5F4"),
            "1. Introduction",  # an unlinked repeat inside the page
            "vi Contents",
            "[2 Tires](#index_split_006.html#filepos113761)",
            _anchor("index%5Fsplit%5F005.html"),
            "1. Introduction",
            "Body of chapter one.",
            *[f"Chapter one line {i}." for i in range(70)],
            _anchor("index%5Fsplit%5F006.html"),
            "2. Tires",
            "Body of chapter two.",
        ]
    )
    elements = markdown_to_elements(md, _FILE_TOC)
    assert _titles(elements) == [("1. Introduction", 1), ("2. Tires", 1)]
    by_text = {e.text: e for e in elements}
    assert by_text["[1 Introduction](#index_split_005.html#filepos42442)"].section_title == ""
    assert by_text["vi Contents"].section_title == ""
    assert by_text["Body of chapter one."].section_title == "1. Introduction"
    assert by_text["Body of chapter two."].section_title == "2. Tires"

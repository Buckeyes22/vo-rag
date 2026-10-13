"""Tests for vo_rag.core.parsing.parsers.epub_markdown.

Verifies the
two real defects the function fixes:
1. pandoc renders GFM bold around styled chapter titles ("## **ADJUSTABLE
   DAMPERS**"); the ** poisons downstream heading filters.
2. EPUBs whose chapter titles are <p>/<div> (not <h1>-<h6>) produce ZERO
   '#' lines from pandoc — the author's structure survives only in the
   EPUB TOC, which the chunker needs to recover.
"""

from __future__ import annotations

from vo_rag.core.parsing.contract import ElementType
from vo_rag.core.parsing.parsers.epub import _split_pages
from vo_rag.core.parsing.parsers.epub_markdown import markdown_to_elements


def _titles(elements):
    return [e for e in elements if e.element_type == ElementType.TITLE]


def test_strips_emphasis_markers_from_hash_heading() -> None:
    md = "## **ADJUSTABLE DAMPERS**\n\nSome body text about dampers."
    titles = _titles(markdown_to_elements(md, []))
    assert len(titles) == 1
    assert titles[0].text == "ADJUSTABLE DAMPERS"


def test_strips_italic_and_code_markers() -> None:
    md = "### *The System Fluid*\n\nbody\n\n# `Glossary`\n\nbody"
    texts = [t.text for t in _titles(markdown_to_elements(md, []))]
    assert "The System Fluid" in texts
    assert "Glossary" in texts


def test_recovers_toc_headings_when_markdown_has_no_hash_headings() -> None:
    toc = [
        {"depth": 1, "title": "Why Bet on Sports", "href": "c1"},
        {"depth": 1, "title": "The Basics", "href": "c2"},
    ]
    md = "\n".join(
        [
            "Front blurb.",
            "Why Bet on Sports",
            "Reasons to bet.",
            "The Basics",
            "Fundamentals here.",
        ]
    )
    titles = {(t.section_depth, t.text) for t in _titles(markdown_to_elements(md, toc))}
    assert (1, "Why Bet on Sports") in titles
    assert (1, "The Basics") in titles


def test_toc_recovery_preserves_author_depth() -> None:
    toc = [
        {"depth": 1, "title": "Suspension", "href": "c1"},
        {"depth": 2, "title": "Dampers", "href": "c1#d"},
    ]
    md = "Suspension\n\nintro\n\nDampers\n\ndetail"
    depths = {t.text: t.section_depth for t in _titles(markdown_to_elements(md, toc))}
    assert depths.get("Suspension") == 1
    assert depths.get("Dampers") == 2


def test_plain_line_not_in_toc_stays_narrative() -> None:
    toc = [{"depth": 1, "title": "Why Bet on Sports", "href": "c1"}]
    md = "Some ordinary paragraph that is not a chapter."
    els = markdown_to_elements(md, toc)
    assert _titles(els) == []
    assert any(e.element_type == ElementType.NARRATIVE_TEXT for e in els)


def test_existing_hash_headings_still_emit_titles() -> None:
    md = "# Introduction\n\nText.\n\n## Details\n\nMore."
    titles = {(t.section_depth, t.text) for t in _titles(markdown_to_elements(md, []))}
    assert (1, "Introduction") in titles
    assert (2, "Details") in titles


def test_toc_title_matched_only_once() -> None:
    """A recurring running-header line must not spam the heading tree."""
    toc = [{"depth": 1, "title": "Baseball", "href": "c1"}]
    md = "Baseball\n\nbody\n\nBaseball\n\nmore body"
    titles = _titles(markdown_to_elements(md, toc))
    assert sum(1 for t in titles if t.text == "Baseball") == 1


def test_hash_heading_wins_over_toc_for_same_title() -> None:
    toc = [{"depth": 1, "title": "Baseball", "href": "c1"}]
    md = "# Baseball\n\nbody about baseball"
    titles = _titles(markdown_to_elements(md, toc))
    assert sum(1 for t in titles if t.text == "Baseball") == 1


def test_hash_heading_uses_authored_toc_depth() -> None:
    toc = [
        {"depth": 1, "title": "Chapter One", "href": "one.xhtml"},
        {"depth": 1, "title": "Chapter Two", "href": "two.xhtml"},
    ]
    elements = markdown_to_elements(
        "## **Chapter One**\nBody one\n## **Chapter Two**\nBody two",
        toc,
    )

    assert [(title.text, title.section_depth) for title in _titles(elements)] == [
        ("Chapter One", 1),
        ("Chapter Two", 1),
    ]
    pages = _split_pages(elements)
    assert len(pages) == 2
    assert {element.page_number for element in pages[0]} == {1}
    assert {element.page_number for element in pages[1]} == {2}


def test_split_pages_uses_shallowest_heading_when_epub_has_no_h1() -> None:
    elements = markdown_to_elements(
        "## Chapter One\nBody one\n### Table detail\n| A | B |\n## Chapter Two\nBody two",
        [],
    )

    pages = _split_pages(elements)

    assert len(pages) == 2
    assert [element.text for element in pages[0]] == [
        "Chapter One",
        "Body one",
        "Table detail",
        "| A | B |",
    ]
    assert [element.text for element in pages[1]] == ["Chapter Two", "Body two"]


# VOR-1427: EPUBs (a machine-learning textbook, for one) whose body headings are
# links back to the TOC page. The TOC page itself links forward to the TOC
# entries' own hrefs.
_BACKLINK_TOC = [
    {"depth": 1, "title": "1     Introduction", "href": "Text/chapter001.xhtml"},
    {"depth": 3, "title": "1.2.1     Classification", "href": "Text/chapter001.xhtml#Sec_1.2.1"},
    {"depth": 1, "title": "2     Probability", "href": "Text/chapter002.xhtml"},
]


def test_toc_backlink_body_headings_become_titles() -> None:
    md = "\n\n".join(
        [
            "[1     Introduction](#chapter001.xhtml)",
            "[1.2.1     Classification](#chapter001.xhtml#Sec_1.2.1)",
            "[**2     Probability**](#chapter002.xhtml)",
            "[**1 Introduction**](#toc.xhtml#chap-1)",
            "Body of chapter one.",
            "[**1.2.1 Classification**](#toc.xhtml#RSec_1.2.1)",
            "Body of the classification section.",
            "[**2 Probability**](#toc.xhtml#chap-2)",
            "Body of chapter two.",
        ]
    )
    elements = markdown_to_elements(md, _BACKLINK_TOC)
    assert [(t.text, t.section_depth) for t in _titles(elements)] == [
        ("1 Introduction", 1),
        ("1.2.1 Classification", 3),
        ("2 Probability", 1),
    ]
    # The TOC page's forward links stay narrative (the chunker rejects them).
    narrative = [e.text for e in elements if e.element_type == ElementType.NARRATIVE_TEXT]
    assert "[1     Introduction](#chapter001.xhtml)" in narrative
    body = next(e for e in elements if e.text == "Body of chapter two.")
    assert body.section_title == "2 Probability"


def test_backlink_heading_emitted_even_after_a_plain_toc_line_consumed_its_entry() -> None:
    md = "\n\n".join(
        [
            "1.2.1     Classification",  # plain TOC-page line consumes the entry
            "[**1.2.1 Classification**](#toc.xhtml#RSec_1.2.1)",
            "Body of the classification section.",
        ]
    )
    titles = _titles(markdown_to_elements(md, _BACKLINK_TOC))
    assert [t.text for t in titles] == ["1.2.1 Classification"] * 2


def test_links_stay_narrative_without_toc_hrefs_or_toc_match() -> None:
    no_href_toc = [{"depth": 1, "title": "1     Introduction", "href": ""}]
    md = "[**1 Introduction**](#toc.xhtml#chap-1)"
    assert _titles(markdown_to_elements(md, no_href_toc)) == []
    unknown = "[**9 Not In The TOC**](#toc.xhtml#chap-9)"
    assert _titles(markdown_to_elements(unknown, _BACKLINK_TOC)) == []
    external = "[1     Introduction](https://example.com/intro)"
    assert _titles(markdown_to_elements(external, _BACKLINK_TOC)) == []


def test_plain_toc_lines_inside_the_toc_page_are_not_headings() -> None:
    """Unlinked entries of the TOC page itself must not open sections."""
    md = "\n\n".join(
        [
            "[1     Introduction](#chapter001.xhtml)",
            "1.2.1     Classification",  # unlinked TOC-page entry
            "[**2     Probability**](#chapter002.xhtml)",
            "Preface text follows the table of contents.",
            "[**1 Introduction**](#toc.xhtml#chap-1)",
            "Body of chapter one.",
            "[**1.2.1 Classification**](#toc.xhtml#RSec_1.2.1)",
            "Body of the classification section.",
        ]
    )
    elements = markdown_to_elements(md, _BACKLINK_TOC)
    assert [t.text for t in _titles(elements)] == [
        "1 Introduction",
        "1.2.1 Classification",
    ]
    preface = next(e for e in elements if e.text.startswith("Preface"))
    assert preface.section_title == ""


def test_wrapped_toc_link_fragments_count_as_toc_page_lines() -> None:
    md = "\n\n".join(
        [
            "[1.2.1     Classification of",
            "things](#chapter001.xhtml#Sec_1.2.1)",
            "2     Probability",  # unlinked entry between wrapped TOC links
            "[1     Introduction](#chapter001.xhtml)",
        ]
    )
    assert _titles(markdown_to_elements(md, _BACKLINK_TOC)) == []


def test_plain_heading_far_from_toc_links_is_still_recovered() -> None:
    filler = [f"Body line {i} of the front matter." for i in range(80)]
    md = "\n\n".join(["[1     Introduction](#chapter001.xhtml)", *filler, "2     Probability"])
    assert [t.text for t in _titles(markdown_to_elements(md, _BACKLINK_TOC))] == ["2 Probability"]


def test_backlink_uses_the_depth_of_the_toc_entry_it_consumes() -> None:
    """Repeated TOC titles ("Exercises") sit at different depths."""
    toc = [
        {"depth": 1, "title": "Exercises", "href": "Text/part1.xhtml#ex"},
        {"depth": 3, "title": "Exercises", "href": "Text/chapter002.xhtml#ex"},
    ]
    md = "\n\n".join(
        [
            "[Exercises](#part1.xhtml#ex)",
            "[Exercises](#chapter002.xhtml#ex)",
            "[**Exercises**](#toc.xhtml#R1)",
            "Part-level exercises.",
            "[**Exercises**](#toc.xhtml#R2)",
            "Chapter-level exercises.",
            "[**Exercises**](#toc.xhtml#R3)",
            "A third, unlisted exercises heading.",
        ]
    )
    titles = _titles(markdown_to_elements(md, toc))
    assert [(t.text, t.section_depth) for t in titles] == [
        ("Exercises", 1),
        ("Exercises", 3),
        ("Exercises", 1),  # queue exhausted: first TOC occurrence
    ]


# ---------------------------------------------------------------------------
# VOR-1436: markup-only lines (e.g. the <svg> wrapper pandoc leaves around a
# fixed-layout page image) are not narrative text.
# ---------------------------------------------------------------------------

_SVG_OPEN = (
    '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
    'version="1.1" width="100%" height="100%" viewbox="0 0 1200 1600">'
)


def test_markup_only_lines_are_dropped() -> None:
    markdown = "\n\n".join(
        [
            _SVG_OPEN,
            "</svg>",
            '<div class="page">',
            "<!-- page 3 -->",
            '<svg xmlns="http://www.w3.org/2000/svg"',
            "Real text with <b>inline</b> markup stays.",
            "<https://example.com/stays>",
        ]
    )

    texts = [e.text for e in markdown_to_elements(markdown, [])]

    assert texts == [
        "Real text with <b>inline</b> markup stays.",
        "<https://example.com/stays>",
    ]


def test_svg_wrapped_page_scan_is_image_only() -> None:
    """The Science of Car Racing shape: page images, SVG wrapper lines and a
    copyright page. Before the fix the wrappers counted as narrative text and
    pushed the image share under 80%."""
    from vo_rag.core.parsing.parsers.epub import is_image_only

    pages = [f"![](images/p{i}.jpg)\n\n{_SVG_OPEN}\n\n</svg>" for i in range(134)]
    copyright_page = "Copyright 2014 Northwind Press. All rights reserved. " * 3
    elements = markdown_to_elements("\n\n".join([copyright_page, *pages]), [])

    assert [e.element_type for e in elements].count(ElementType.NARRATIVE_TEXT) == 1
    assert is_image_only(elements) is True


def test_pipe_table_rows_form_one_table_element() -> None:
    # VOR-1438: every pipe row became its own TABLE element, shattering real
    # tables into one-row chunks (one book produced 480 row elements).
    md = "Before.\n\n| Year | Wins |\n|------|------|\n| 1998 | 12 |\n| 1999 | 9 |\n\nAfter."
    elements = markdown_to_elements(md, [])
    tables = [e for e in elements if e.element_type == ElementType.TABLE]
    assert len(tables) == 1
    assert tables[0].text == "| Year | Wins |\n|------|------|\n| 1998 | 12 |\n| 1999 | 9 |"
    assert [e.text for e in elements if e.element_type == ElementType.NARRATIVE_TEXT] == [
        "Before.",
        "After.",
    ]


def test_blank_spacer_table_emits_nothing() -> None:
    # VOR-1438: calibre spacer tables (`<td/>`) render as `|     |` /
    # `|-----|` / `|     |`; one algorithms book had 108 such blank chunks.
    md = "Intro text.\n\n|     |\n|-----|\n|     |\n\nMore text."
    elements = markdown_to_elements(md, [])
    assert not [e for e in elements if e.element_type == ElementType.TABLE]
    assert [e.text for e in elements] == ["Intro text.", "More text."]


def test_adjacent_tables_split_on_blank_line_and_keep_their_section() -> None:
    md = "## Results\n\n| a |\n|---|\n| 1 |\n\n| b |\n|---|\n| 2 |\n"
    tables = [e for e in markdown_to_elements(md, []) if e.element_type == ElementType.TABLE]
    assert [t.text for t in tables] == ["| a |\n|---|\n| 1 |", "| b |\n|---|\n| 2 |"]
    assert {t.section_title for t in tables} == {"Results"}


def test_table_at_end_of_document_is_flushed() -> None:
    tables = [
        e
        for e in markdown_to_elements("Text.\n\n| x | y |\n|---|---|\n| 1 | 2 |", [])
        if e.element_type == ElementType.TABLE
    ]
    assert [t.text for t in tables] == ["| x | y |\n|---|---|\n| 1 | 2 |"]

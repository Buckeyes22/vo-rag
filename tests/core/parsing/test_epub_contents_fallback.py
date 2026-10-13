"""Chapters for EPUBs whose navigation TOC has at most one entry (VOR-1427).

Calibre often writes a one-entry NCX ("Start" -> titlepage.xhtml), so the TOC
anchor recovery had nothing to match and whole books collapsed into one or two
sections. Two fallbacks, both off whenever the real TOC has two or more
entries:

A. A linked in-book contents page (epub.py): the first spine file with at
   least 5 links to spine files becomes the TOC, and the anchor path places
   the chapters.
B. An unlinked printed contents list (epub_markdown.py): entries after a
   "Contents" line, or a leading numbered run, are matched to later body lines
   after normalizing numbering, "Chapter" words and page numbers.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from vo_rag.core.parsing.contract import ElementType
from vo_rag.core.parsing.parsers.epub_markdown import contents_key, markdown_to_elements

_START_TOC = [{"depth": 1, "title": "Start", "href": "titlepage.xhtml"}]
_PROSE = (
    "This is a long line of ordinary narrative prose that clearly is not an entry "
    "of any printed table of contents in the book."
)


def _titles(elements: list[Any]) -> list[tuple[str, int]]:
    return [(e.text, e.section_depth) for e in elements if e.element_type == ElementType.TITLE]


def _md(*lines: str) -> str:
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("contents_line", "body_line"),
    [
        ("Chapter 1 – The New Age of AI Agents", "1 – The New Age of AI Agents"),
        ("1.- Introducción.", "1.Introducción."),
        ("*2* Getting to know the lingo", "**Getting to know the lingo**"),
        ("1.  Introduction to Generative AI Agents", "1\\. Introduction to Generative AI Agents"),
        ("Part II — Building Intelligent Foundations", "Building Intelligent Foundations"),
        ("Capítulo 3. Tipos de apuestas ........ 17", "3.Tipos de apuestas"),
        ("A. Appendix of Tables", "Appendix of Tables"),
        ("- Understanding AI Agents", "1.1 Understanding AI Agents"),
        ("Kapitel 4: Grundlagen 23", "## 4 Grundlagen"),
    ],
)
def test_contents_key_ignores_numbering_labels_and_page_numbers(
    contents_line: str, body_line: str
) -> None:
    assert contents_key(contents_line)
    assert contents_key(contents_line) == contents_key(body_line)


def test_contents_key_keeps_words_that_only_look_like_labels() -> None:
    assert contents_key("A Betting Strategy For Roulette") == "a betting strategy for roulette"
    assert contents_key("I want to thank") == "i want to thank"


# ---------------------------------------------------------------------------
# Group B: printed contents list (no pandoc needed)
# ---------------------------------------------------------------------------


def test_heading_anchored_contents_block_places_chapters() -> None:
    md = _md(
        "Book Title",
        "Table of Content",
        "Preface",
        "Chapter 1 – The New Age of AI Agents",
        "Chapter 2 – How LLMs Think: The Transformer and Beyond",
        "Chapter 3 – RAG: The Backbone of Truthful Agents",
        "Why This Book Exists",
        _PROSE,
        "1 – The New Age of AI Agents",
        "Body one.",
        "2 – How LLMs Think: The Transformer and Beyond",
        "Body two.",
        "3 – RAG: The Backbone of Truthful Agents",
        "Body three.",
    )
    elements = markdown_to_elements(md, _START_TOC)

    assert _titles(elements) == [
        ("Chapter 1 – The New Age of AI Agents", 1),
        ("Chapter 2 – How LLMs Think: The Transformer and Beyond", 1),
        ("Chapter 3 – RAG: The Backbone of Truthful Agents", 1),
    ]
    by_text = {e.text: e for e in elements}
    assert by_text["Body one."].section_title == "Chapter 1 – The New Age of AI Agents"
    assert (
        by_text["Body three."].section_title == "Chapter 3 – RAG: The Backbone of Truthful Agents"
    )
    # The contents block itself stays narrative text.
    assert by_text["Why This Book Exists"].element_type == ElementType.NARRATIVE_TEXT
    assert by_text[_PROSE].section_title == ""


def test_spanish_indice_block_with_page_footers() -> None:
    md = _md(
        "Índice",
        "1.- Introducción.",
        "\\- Que son las apuestas deportivas.",
        "\\- Vivir de las apuestas.",
        "\\-",  # a bullet whose text pandoc put on the next line
        "Donde apostar.",
        "2.- Glosario para novatos.",
        "pág. 5",
        "3.- Tipos de apuestas.",
        "pág. 6",
        "1.Introducción.",
        _PROSE,
        "pág. 7",
        "2.Glosario para novatos.",
        "Texto del glosario.",
        "pág. 8",
        "3.Tipos de apuestas.",
        "Texto de tipos.",
        "pág. 9",
    )
    elements = markdown_to_elements(md, _START_TOC)
    assert [t for t, _ in _titles(elements)] == [
        "1.- Introducción.",
        "2.- Glosario para novatos.",
        "3.- Tipos de apuestas.",
    ]
    assert {e.text: e for e in elements}[
        "Texto de tipos."
    ].section_title == "3.- Tipos de apuestas."


def test_emphasized_numbers_and_bold_body_headings() -> None:
    md = _md(
        "Contents",
        "Photo credits Acknowledgments",
        "*1* Getting acquainted",
        "*2* Getting to know the lingo",
        "*3* Getting comfortable",
        "------------------------------------------------------------------------",
        "**Acknowledgments**",
        _PROSE,
        "**Getting acquainted**",
        "Body one.",
        "**Getting to know the lingo**",
        "Body two.",
        "## Getting comfortable",
        "Body three.",
    )
    assert [t for t, _ in _titles(markdown_to_elements(md, _START_TOC))] == [
        "1 Getting acquainted",
        "2 Getting to know the lingo",
        "3 Getting comfortable",
    ]


def test_leading_numbered_run_without_contents_heading() -> None:
    md = _md(
        "![](cover.jpg)",
        "Generative AI Agents: A Practical Guide",
        "Part 1: Foundations",
        "1.  Introduction to Generative AI Agents",
        "    - Understanding AI Agents",
        "    - Applications and Use Cases",
        "2.  Core Concepts in Generative AI",
        "3.  Multi-Agent Systems and Coordination",
        "------------------------------------------------------------------------",
        "Part 2: Designing Agents",
        "4.  Building Blocks of Generative AI Agents",
        "5.  Frameworks and Tools for AI Agent Development",
        "1\\. Introduction to Generative AI Agents",
        "1.1 Understanding AI Agents",
        "An AI agent is a system that perceives its environment and acts on it.",
        "2\\. Core Concepts in Generative AI",
        "Body two.",
        "3\\. Multi-Agent Systems and Coordination",
        "Body three.",
        "4\\. Building Blocks of Generative AI Agents",
        "5\\. Frameworks and Tools for AI Agent Development",
        "Body five.",
    )
    assert _titles(markdown_to_elements(md, _START_TOC)) == [
        ("1. Introduction to Generative AI Agents", 1),
        ("Understanding AI Agents", 2),
        ("2. Core Concepts in Generative AI", 1),
        ("3. Multi-Agent Systems and Coordination", 1),
        ("4. Building Blocks of Generative AI Agents", 1),
        ("5. Frameworks and Tools for AI Agent Development", 1),
    ]


def test_roulette_numbered_betting_lines_give_no_titles() -> None:
    """de65e13d4196: short numbered wagers restart at 1 and repeat through the
    booklet; there is no contents heading, so nothing may become a TITLE."""
    wagers = [
        "1. Place 18-units on numbers 1 thru 18.",
        "2. Place 6-units on numbers 19 thru 24.",
        "3. Place 6-units on numbers 25 thru 30.",
    ]
    lines = ["The Rule of 2/3rds", "A Betting Strategy For Roulette."]
    for n in range(30, 36):
        lines += [f"Play {n} Numbers", *wagers, "4. Place 1-unit on number 31."]
        lines += ["When any one of these numbers hit, the payout is for 36-units."]
        lines += ["This multiple wager is expressed in the following manner:"]
    md = _md(*lines)
    elements = markdown_to_elements(md, _START_TOC)
    assert _titles(elements) == []
    assert elements == markdown_to_elements(md, [])


def test_running_header_in_contents_block_is_not_matched() -> None:
    """0007eb2807d1: a running header repeats on every page; it must never
    become a TITLE even when the contents block lists it."""
    md = _md(
        "Contents",
        "Timer's Guide",
        "Getting Started",
        "Timing the Race",
        "Scoring Results",
        _PROSE,
        "Timer's Guide",
        "Getting Started",
        "Body one.",
        "Timer's Guide",
        "Timing the Race",
        "Body two.",
        "Timer's Guide",
        "Scoring Results",
        "Body three.",
        "Timer's Guide",
    )
    assert [t for t, _ in _titles(markdown_to_elements(md, _START_TOC))] == [
        "Getting Started",
        "Timing the Race",
        "Scoring Results",
    ]


def test_too_few_body_matches_gives_no_titles() -> None:
    md = _md(
        "Contents",
        "Timer's Guide",
        "Getting Started",
        "Timing the Race",
        _PROSE,
        "Timer's Guide",
        "Getting Started",
        "Body one.",
        "Timer's Guide",
        "Timing the Race",
        "Timer's Guide",
    )
    elements = markdown_to_elements(md, _START_TOC)
    assert _titles(elements) == []
    assert [e.text for e in elements] == [e.text for e in markdown_to_elements(md, [])]


def test_lines_before_the_contents_block_are_not_matched() -> None:
    md = _md(
        "1 – Alpha Topic",
        "Contents",
        "Chapter 1 – Alpha Topic",
        "Chapter 2 – Beta Topic",
        "Chapter 3 – Gamma Topic",
        _PROSE,
        "1 – Alpha Topic",
        "2 – Beta Topic",
        "3 – Gamma Topic",
    )
    elements = markdown_to_elements(md, [])
    assert elements[0].element_type == ElementType.NARRATIVE_TEXT
    assert len(_titles(elements)) == 3


def test_printed_contents_ignored_when_toc_has_two_entries() -> None:
    toc = [
        {"depth": 1, "title": "Front", "href": "front.xhtml"},
        {"depth": 1, "title": "Back", "href": "back.xhtml"},
    ]
    md = _md(
        "Contents",
        "Chapter 1 – Alpha Topic",
        "Chapter 2 – Beta Topic",
        "Chapter 3 – Gamma Topic",
        _PROSE,
        "1 – Alpha Topic",
        "2 – Beta Topic",
        "3 – Gamma Topic",
    )
    assert _titles(markdown_to_elements(md, toc)) == []


# ---------------------------------------------------------------------------
# Group A: linked in-book contents page (through ebooklib and pandoc)
# ---------------------------------------------------------------------------

_CHAPTERS = ["Alpha Ideas", "Beta Methods", "Gamma Results", "Delta Tools", "Epsilon Notes"]


def _write_book(epub: Any, path: Path, toc_count: int) -> Path:
    """Titlepage, a linked contents page, five chapters whose titles are styled
    paragraphs that differ from the link text, plus a fragment target."""
    book = epub.EpubBook()
    book.set_identifier(f"vorag-test-{path.stem}")
    book.set_title(path.stem)
    book.set_language("en")
    links = "".join(
        f'<p><a href="c{i}.xhtml">{title}</a></p>' for i, title in enumerate(_CHAPTERS, 1)
    )
    links += '<p><a href="c5.xhtml#extra">Zeta Appendix</a></p><p><a href="c1.xhtml">dup</a></p>'
    files = [
        ("titlepage.xhtml", "<p>The Book</p>"),
        ("contents.xhtml", f"<p>Table of Contents</p>{links}"),
    ]
    for i, _ in enumerate(_CHAPTERS, 1):
        body = f'<p class="h">C H A P T E R {i}</p><p>Body {i}.</p>'
        if i == 5:
            body += '<p id="extra" class="h">APPENDIX</p><p>Appendix body.</p>'
        files.append((f"c{i}.xhtml", body))
    items = []
    for name, body in files:
        item = epub.EpubHtml(title=name, file_name=f"text/{name}", lang="en")
        item.content = f"<html><body>{body}</body></html>"
        book.add_item(item)
        items.append(item)
    toc_targets = [("titlepage.xhtml", "Start"), ("c3.xhtml", "Gamma Results")][:toc_count]
    book.toc = tuple(
        epub.Link(f"text/{href}", title, f"toc{i}") for i, (href, title) in enumerate(toc_targets)
    )
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *items]
    epub.write_epub(str(path), book)
    return path


def test_contents_page_links_become_the_toc(tmp_path: Path) -> None:
    epub = pytest.importorskip("ebooklib.epub")
    from vo_rag.core.parsing.parsers.epub import _contents_page_toc

    book = epub.read_epub(str(_write_book(epub, tmp_path / "a.epub", 1)))
    assert _contents_page_toc(book) == [
        *({"depth": 1, "title": t, "href": f"text/c{i}.xhtml"} for i, t in enumerate(_CHAPTERS, 1)),
        {"depth": 1, "title": "Zeta Appendix", "href": "text/c5.xhtml#extra"},
    ]


def _require_pandoc_stack() -> Any:
    epub = pytest.importorskip("ebooklib.epub")
    pytest.importorskip("pypandoc")
    if shutil.which("pandoc") is None:
        pytest.skip("pandoc binary not installed")
    return epub


def _parse(path: Path) -> list[Any]:
    from vo_rag.core.parsing.parsers.epub import _read_epub_and_convert

    _, _, toc_tree, markdown = _read_epub_and_convert(str(path), str(path), source_format="epub")
    return markdown_to_elements(markdown, toc_tree)


def test_one_entry_toc_with_linked_contents_page_gets_chapter_titles(tmp_path: Path) -> None:
    epub = _require_pandoc_stack()
    elements = _parse(_write_book(epub, tmp_path / "linked.epub", 1))

    assert [t for t, _ in _titles(elements)] == ["Start", *_CHAPTERS, "Zeta Appendix"]
    by_text = {e.text: e for e in elements}
    assert by_text["Body 2."].section_title == "Beta Methods"
    assert by_text["Appendix body."].section_title == "Zeta Appendix"


def test_linked_contents_page_ignored_when_toc_has_two_entries(tmp_path: Path) -> None:
    epub = _require_pandoc_stack()
    elements = _parse(_write_book(epub, tmp_path / "real.epub", 2))
    assert [t for t, _ in _titles(elements)] == ["Start", "Gamma Results"]


# ---------------------------------------------------------------------------
# Key-takeaway lists, leading runs, index pages and charsets
# ---------------------------------------------------------------------------


def test_key_takeaways_list_after_prose_is_not_a_contents_run() -> None:
    """A chapter-1 "Key takeaways" list whose items reappear later split a
    book into fake chapters: a leading run must precede the first prose."""
    items = [
        f"{n}. Takeaway number {word}"
        for n, word in enumerate(["one", "two", "three", "four", "five"], 1)
    ]
    long_prose = "Opening prose of chapter one. " * 10
    md = _md(
        "Book Title",
        "Chapter One",
        long_prose,
        "Key takeaways:",
        *items,
        long_prose,
        *(line for item in items for line in (item.split(". ", 1)[1], "More text.")),
    )
    elements = markdown_to_elements(md, _START_TOC)
    assert _titles(elements) == []


def test_leading_run_matched_out_of_order_is_not_a_contents_run() -> None:
    """A real contents list is matched in increasing document order (one
    skip allowed); a numbered list whose items recur shuffled is not one."""
    names = ["Apples", "Bananas", "Cherries", "Dates", "Elderberries", "Figs"]
    md = _md(
        "Fruit Notes",
        *(f"{n}. {name}" for n, name in enumerate(names, 1)),
        _PROSE,
        *(
            line
            for name in ["Figs", "Dates", "Bananas", "Elderberries", "Apples", "Cherries"]
            for line in (name, "Some text.")
        ),
    )
    assert _titles(markdown_to_elements(md, _START_TOC)) == []


def test_leading_run_allows_one_out_of_order_match() -> None:
    names = ["Apples", "Bananas", "Cherries", "Dates", "Elderberries"]
    md = _md(
        *(f"{n}. {name}" for n, name in enumerate(names, 1)),
        _PROSE,
        *(
            line
            for name in ["Apples", "Cherries", "Bananas", "Dates", "Elderberries"]
            for line in (name, "Some text.")
        ),
    )
    assert len(_titles(markdown_to_elements(md, _START_TOC))) == 5


def test_long_dot_leader_line_is_linear() -> None:
    import time

    md = _md(
        "Contents",
        "Chapter 1 – Alpha Topic",
        "Chapter 2 – Beta Topic",
        "Chapter 3 – Gamma Topic",
        _PROSE,
        "1 – Alpha Topic",
        "Chapter 9 " + "." * 40_000 + "x",
        "2 – Beta Topic",
        "_" * 40_000 + "y",
        "3 – Gamma Topic",
    )
    started = time.perf_counter()
    elements = markdown_to_elements(md, [])
    assert time.perf_counter() - started < 0.5
    assert len(_titles(elements)) == 3
    started = time.perf_counter()
    assert contents_key("Title " + "." * 40_000 + "x") == ""
    assert time.perf_counter() - started < 0.5


@pytest.mark.parametrize(
    ("line", "key"),
    [
        ("X-Men Origins", "x men origins"),
        ("E-commerce Basics", "e commerce basics"),
        ("B) Methods", "methods"),
        ("IV — Results", "results"),
    ],
)
def test_single_letter_numbering_needs_a_dot_or_parenthesis(line: str, key: str) -> None:
    assert contents_key(line) == key


def test_printed_contents_engages_with_several_nodes_but_one_href() -> None:
    toc: list[dict[str, object]] = [
        {"depth": 1, "title": "Part One", "href": ""},
        {"depth": 1, "title": "Part Two", "href": ""},
        {"depth": 1, "title": "Start", "href": "titlepage.xhtml"},
    ]
    md = _md(
        "Contents",
        "Chapter 1 – Alpha Topic",
        "Chapter 2 – Beta Topic",
        "Chapter 3 – Gamma Topic",
        _PROSE,
        "1 – Alpha Topic",
        "2 – Beta Topic",
        "3 – Gamma Topic",
    )
    assert len(_titles(markdown_to_elements(md, toc))) == 3


def _xhtml(body: str, *, title: str = "", encoding: str = "utf-8") -> bytes:
    head = f"<title>{title}</title>" if title else ""
    return (
        f'<?xml version="1.0" encoding="{encoding}"?>\n'
        f'<html xmlns="http://www.w3.org/1999/xhtml"><head>{head}</head>'
        f"<body>{body}</body></html>"
    ).encode(encoding)


def _raw_book(epub: Any, path: Path, pages: list[tuple[str, bytes]], chapters: int = 12) -> Any:
    """A book with a one-entry TOC, ``chapters`` chapter files and ``pages``
    (name, raw XHTML bytes) placed before them in the spine; read back."""
    book = epub.EpubBook()
    book.set_identifier(f"vorag-test-{path.stem}")
    book.set_title(path.stem)
    book.set_language("en")
    chapter_pages = [
        (f"c{i}.xhtml", _xhtml(f"<p>C H A P T E R {i}</p><p>Body {i}.</p>"))
        for i in range(1, chapters + 1)
    ]
    items = []
    for n, (name, data) in enumerate(
        [("titlepage.xhtml", _xhtml("<p>Book</p>")), *pages, *chapter_pages]
    ):
        item = epub.EpubItem(
            uid=f"p{n}",
            file_name=f"text/{name}",
            media_type="application/xhtml+xml",
            content=data,
        )
        book.add_item(item)
        items.append(item)
    book.toc = (epub.Link("text/titlepage.xhtml", "Start", "start"),)
    book.add_item(epub.EpubNcx())
    book.spine = items
    epub.write_epub(str(path), book)
    return epub.read_epub(str(path))


def _chapter_links(count: int = 6) -> str:
    return "".join(f'<p><a href="c{i}.xhtml">Chapter Name {i}</a></p>' for i in range(1, count + 1))


def test_index_page_gives_no_fallback_toc(tmp_path: Path) -> None:
    epub = pytest.importorskip("ebooklib.epub")
    from vo_rag.core.parsing.parsers.epub import _contents_page_toc

    links = "".join(
        f'<p>term {j}, <a href="c{j % 12 + 1}.xhtml#idx{j}">term {j}</a></p>' for j in range(300)
    )
    book = _raw_book(epub, tmp_path / "index.epub", [("index.xhtml", _xhtml(links, title="Index"))])
    assert _contents_page_toc(book) == []


@pytest.mark.parametrize(
    ("page", "why"),
    [
        (
            _xhtml(
                "".join(f'<p><a href="c{j % 2 + 1}.xhtml#n{j}">note {j}</a></p>' for j in range(40))
            ),
            "fragments into few files",
        ),
        (
            _xhtml(
                "".join(
                    f'<p><a href="c{j % 40 + 1}.xhtml#s{j}">sec {j}</a></p>' for j in range(160)
                )
            ),
            "over 150 entries",
        ),
        (_xhtml(f"<h1>Notes</h1>{_chapter_links()}"), "notes heading"),
        (_xhtml(_chapter_links(), title="Bibliography"), "bibliography title"),
    ],
)
def test_index_like_pages_are_rejected(tmp_path: Path, page: bytes, why: str) -> None:
    epub = pytest.importorskip("ebooklib.epub")
    from vo_rag.core.parsing.parsers.epub import _contents_page_toc

    book = _raw_book(epub, tmp_path / "rej.epub", [("x.xhtml", page)], chapters=40)
    assert _contents_page_toc(book) == [], why


def test_contents_page_preferred_over_index_and_unlabelled_pages(tmp_path: Path) -> None:
    epub = pytest.importorskip("ebooklib.epub")
    from vo_rag.core.parsing.parsers.epub import _contents_page_toc

    index = "".join(f'<p><a href="c{j % 12 + 1}.xhtml#i{j}">word {j}</a></p>' for j in range(20))
    figures = "".join(f'<p><a href="c{i}.xhtml#f{i}">Figure {i}</a></p>' for i in range(1, 7))
    pages = [
        ("index.xhtml", _xhtml(index, title="Index")),
        ("figures.xhtml", _xhtml(figures)),
        ("toc.xhtml", _xhtml(f"<h1>Table of Contents</h1>{_chapter_links()}")),
    ]
    toc = _contents_page_toc(_raw_book(epub, tmp_path / "both.epub", pages))
    assert [e["title"] for e in toc] == [f"Chapter Name {i}" for i in range(1, 7)]


@pytest.mark.parametrize("declared", ["xml", "meta"])
def test_contents_page_declared_charset_is_honoured(tmp_path: Path, declared: str) -> None:
    epub = pytest.importorskip("ebooklib.epub")
    from vo_rag.core.parsing.parsers.epub import _contents_page_toc

    links = "".join(f'<p><a href="c{i}.xhtml">Café Chapter {i}</a></p>' for i in range(1, 7))
    if declared == "xml":
        page = _xhtml(links, encoding="windows-1252")
    else:
        page = (
            '<html><head><meta http-equiv="Content-Type" content="text/html; charset=cp1252"/>'
            f"</head><body>{links}</body></html>"
        ).encode("cp1252")
    toc = _contents_page_toc(_raw_book(epub, tmp_path / "cp.epub", [("toc.xhtml", page)]))
    assert [e["title"] for e in toc] == [f"Café Chapter {i}" for i in range(1, 7)]


def test_contents_page_fallback_gates_on_href_entries(tmp_path: Path) -> None:
    epub = pytest.importorskip("ebooklib.epub")
    from vo_rag.core.parsing.parsers.epub import _with_contents_page_fallback

    book = _raw_book(epub, tmp_path / "gate.epub", [("toc.xhtml", _xhtml(_chapter_links()))])
    one_href = [
        {"depth": 1, "title": "Part One", "href": ""},
        {"depth": 1, "title": "Part Two", "href": ""},
        {"depth": 1, "title": "Start", "href": "text/titlepage.xhtml"},
    ]
    assert len(_with_contents_page_fallback(book, one_href)) == 3 + 6
    two_hrefs = [*one_href, {"depth": 1, "title": "C1", "href": "text/c1.xhtml"}]
    assert _with_contents_page_fallback(book, two_hrefs) == two_hrefs

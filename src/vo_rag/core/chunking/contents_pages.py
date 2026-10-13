"""Page-level contents-page detection (VOR-1452).

A scanned book's contents page reaches the chunker as TITLEs ("12 The
shot-put 250") and short lines, which used to open one section per line.
Contents pages are found here, page by page, so the chunker can file each one
whole as contents body under a "Contents" label (``HierarchicalChunker``): on a
contents page no element opens a section.

Context decides first. A page is in a contents context when a
Contents heading sits on it or on the page before, or the page before is a
contents page (a run). Only a page in context can be a contents page by its
lines; a page with no Contents heading near it is one only when docling marked
an element on it as ``document_index`` (and that mark is trusted only in the
front part or in context). A list of rising numbers anywhere
else -- a volume progression, warm-up minutes, a record progression -- is
never contents.

In context the check on the lines is light (``_is_entry_page``):

- entries -- short lines ending in a page number ("12 The shot-put 250",
  "Preface ..... vii"; not a one-word label such as "Chapter 1"), table rows
  included -- number at least CONTENTS_MIN_ENTRIES and make up at least
  CONTENTS_MIN_SHARE of the page's short elements;
- their arabic page numbers never fall in more than 1 - CONTENTS_MIN_RISING of
  the steps (non-decreasing: a chapter and its first section share a page
  number) and stay within the book;
- blurbs and question titles are allowed: the page is prose only when its
  sentences outnumber its entries;
- a chapter heading that is not itself an entry ("Chapter 1 Foundations" with
  no page number) makes the page a body page: the chapter starts here.

Thresholds were tuned on a corpus run: a typical contents page scores about
0.44-0.67 entries; back-of-book indexes
are never in a contents context; an edition-history page ("Reprinted 1974")
ends in years past the book's length; results lists end in times ("10.2"),
which are not page numbers, or in falling points.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Final

from vo_rag.core.chunking.front_matter import front_matter_kind
from vo_rag.core.chunking.heading_filter import is_labelled_heading
from vo_rag.core.parsing.contract import ElementType, ParsedDocument, ParsedElement

CONTENTS_MIN_ENTRIES: Final[int] = 3
CONTENTS_MIN_SHARE: Final[float] = 0.4
CONTENTS_MIN_RISING: Final[float] = 0.8
#: The front part of a book: the first max(CONTENTS_FRONT_PAGES,
#: CONTENTS_FRONT_SHARE * total_pages) pages.
CONTENTS_FRONT_PAGES: Final[int] = 20
CONTENTS_FRONT_SHARE: Final[float] = 0.15
#: A contents entry is a short line; longer elements are not judged.
CONTENTS_ENTRY_MAX_CHARS: Final[int] = 120
#: docling's label for a table of contents (``DocItemLabel.DOCUMENT_INDEX``).
DOCUMENT_INDEX_LABEL: Final[str] = "document_index"

_JUDGED_TYPES: Final[frozenset[ElementType]] = frozenset(
    {ElementType.TITLE, ElementType.NARRATIVE_TEXT, ElementType.LIST_ITEM, ElementType.TABLE}
)
# Lower-case Roman folios only: "Part II" / "Chapter IV" are titles.
_ROMAN: Final[str] = r"(?:x{0,3})(?:ix|iv|v?i{1,3}|v|x)"
# The page number follows whitespace or a dot leader; "10.2" and "10:2" (a
# time, a decimal) never end in one.
_ENTRY_RE: Final[re.Pattern[str]] = re.compile(
    rf"^(?P<title>.*?[^\W\d_].*?)(?:(?P<leader>\s*(?:\.\s*){{2,}}|\s*[…_]+\s*)|\s+)"
    rf"(?P<folio>\d{{1,4}}|{_ROMAN})$"
)
_LEADING_NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"^\d+(?:\.\d+)*\.?\s+")
_NUMBER_ONLY_RE: Final[re.Pattern[str]] = re.compile(rf"^(?:\d{{1,4}}|{_ROMAN})$", re.I)


#: A dot leader of any length counts as a short one when a line is measured.
_LONG_LEADER_RE: Final[re.Pattern[str]] = re.compile(r"(?:\.\s*){4,}")


def _measured(text: str) -> str:
    """``text`` with long dot leaders shortened: "Audience ......(140 dots) 1"
    is a short contents line, not a long one."""
    return _LONG_LEADER_RE.sub(" .... ", text)


def _folio(text: str) -> str | None:
    """The page number a contents line ends in, if ``text`` is one."""
    text = " ".join(_measured(text).split())
    if len(text) > CONTENTS_ENTRY_MAX_CHARS:
        return None
    match = _ENTRY_RE.match(text)
    if match is None:
        return None
    # A one-word label and its number ("Chapter 1", "Session 12", "Week 3") is
    # not a contents line; a dot leader makes one of any title ("Preface ....
    # vii").
    title = _LEADING_NUMBER_RE.sub("", match.group("title"))
    words = [word for word in title.split() if any(ch.isalpha() for ch in word)]
    if match.group("leader") is None and len(words) < 2:
        return None
    return match.group("folio")


# A sentence: at least this many words, ending in terminal punctuation. Contents
# lines end in a page number; a blurb under an entry is a sentence, so a page
# is prose only when its sentences outnumber its entries (a "?" inside a topic
# list ("Who are the hurdlers?") is not a break).
_PROSE_MIN_WORDS: Final[int] = 6
_SENTENCE_END_RE: Final[re.Pattern[str]] = re.compile(r"[.!?][\"')\]]*$")
_SENTENCE_BREAK_RE: Final[re.Pattern[str]] = re.compile(r"(?<=[a-z]{3})[.!]\s+[A-Z]")


def _is_prose(element: ParsedElement) -> bool:
    if element.element_type is ElementType.TABLE:
        return False
    text = " ".join(element.text.split())
    if len(text.split()) < _PROSE_MIN_WORDS:
        return False
    return bool(_SENTENCE_END_RE.search(text) or _SENTENCE_BREAK_RE.search(text))


def _is_entry_page(elements: list[ParsedElement], max_folio: int) -> bool:
    """The light entry check for a page in a contents context (module docstring)."""
    lines = [
        line
        for element in elements
        for line in _entry_lines(element)
        if not _CONTENTS_ROW_RE.match(line)  # the table's own "Contents" row
    ]
    short = [
        line
        for line in lines
        if len(_measured(line)) <= CONTENTS_ENTRY_MAX_CHARS and not _NUMBER_ONLY_RE.match(line)
    ]
    folios = [folio for line in short if (folio := _folio(line)) is not None]
    if len(folios) < CONTENTS_MIN_ENTRIES or len(folios) < CONTENTS_MIN_SHARE * len(short):
        return False
    prose = sum(1 for element in elements if _is_prose(element))
    if prose >= len(folios):
        return False  # a body page with a list in it, not a list with blurbs
    if any(
        element.element_type is ElementType.TITLE
        and is_labelled_heading(element.text)
        and _folio(" ".join(element.text.split())) is None
        for element in elements
    ):
        return False  # "Chapter 1 Foundations" with no page number: a chapter opens here
    numbers = [int(folio) for folio in folios if folio.isdigit()]
    if any(number > max_folio for number in numbers):
        return False
    if len(numbers) >= 2:
        rising = sum(later >= earlier for earlier, later in zip(numbers, numbers[1:], strict=False))
        if rising < CONTENTS_MIN_RISING * (len(numbers) - 1):
            return False
    return True


#: A table row that heads a contents table ("Table of Contents ..... 2").
_CONTENTS_ROW_RE: Final[re.Pattern[str]] = re.compile(r"^(?:table of )?contents\b", re.I)
#: A paragraph this long on a contents page is not an entry's blurb.
CONTENTS_BLURB_MAX_WORDS: Final[int] = 40


#: An annotated entry opens with its page or chapter ("Page 11 Speed
#: Training - This opening chapter ...", "4 The challenge method ...").
_ENTRY_START_RE: Final[re.Pattern[str]] = re.compile(r"^(?:page\s*)?\d|^(?:chapter|part)\b", re.I)


def is_body_paragraph(element: ParsedElement) -> bool:
    """A prose paragraph too long to be a contents blurb and not opening with a
    page or chapter number: a foreword or a prologue beside the contents."""
    text = " ".join(element.text.split())
    return (
        element.element_type is ElementType.NARRATIVE_TEXT
        and _is_prose(element)
        and len(text.split()) >= CONTENTS_BLURB_MAX_WORDS
        and not _ENTRY_START_RE.match(text)
    )


#: A contents row is short; longer table lines are not judged (bounds _ENTRY_RE).
_LEADER_ROW_MAX_CHARS: Final[int] = 400


def _leader_rows(element: ParsedElement) -> int:
    """Rows of a table that end in a page number after a dot leader."""
    if element.element_type is not ElementType.TABLE:
        return 0
    return sum(
        1
        for line in _entry_lines(element)
        if len(line) <= _LEADER_ROW_MAX_CHARS
        and (match := _ENTRY_RE.match(line)) is not None
        and match.group("leader")
    )


def _entry_lines(element: ParsedElement) -> list[str]:
    """The candidate entry lines of an element: its text, or a table's rows."""
    if element.element_type is ElementType.TABLE:
        lines: list[str] = []
        for row in element.text.splitlines():
            cells = [" ".join(cell.split()) for cell in row.strip().strip("|").split("|")]
            # docling repeats a spanned cell in every column it spans: once.
            kept = [
                cell for index, cell in enumerate(cells) if index == 0 or cell != cells[index - 1]
            ]
            line = " ".join(cell for cell in kept if cell)
            if line and not set(line) <= set("-: "):
                lines.append(line)
        return lines
    return [" ".join(element.text.split())]


def contents_pages(doc: ParsedDocument) -> frozenset[int]:
    """The pages of ``doc`` that are contents pages (see the module docstring)."""
    by_page: defaultdict[int, list[ParsedElement]] = defaultdict(list)
    marked: set[int] = set()
    headed: set[int] = set()
    #: Pages with a dot-leader table ("About the Tutorial ...... 1"): contents
    #: context in the front part, where a contents table sits.
    leader_pages: set[int] = set()
    for element in doc.elements:
        page = element.page_number
        if page is None or not element.text.strip():
            continue
        if element.metadata.get("docling_label") == DOCUMENT_INDEX_LABEL:
            marked.add(page)
        if element.element_type not in _JUDGED_TYPES:
            continue
        by_page[page].append(element)
        if front_matter_kind(" ".join(element.text.split())) == "toc" or (
            element.element_type is ElementType.TABLE
            and any(_CONTENTS_ROW_RE.match(line) for line in _entry_lines(element)[:3])
        ):
            headed.add(page)
        elif _leader_rows(element) >= CONTENTS_MIN_ENTRIES:
            leader_pages.add(page)

    total = doc.total_pages or max([*by_page, *marked], default=0)
    front = max(CONTENTS_FRONT_PAGES, int(CONTENTS_FRONT_SHARE * total))
    headed |= {page for page in leader_pages if page <= front}
    max_folio = total + max(20, total // 5)
    found: set[int] = set()
    previous: int | None = None
    for page in sorted({*by_page, *marked}):
        in_context = page in headed or (
            previous is not None and (previous in headed or previous in found)
        )
        if (page in marked and (in_context or page <= front)) or (
            in_context and page in by_page and _is_entry_page(by_page[page], max_folio)
        ):
            found.add(page)
        previous = page
    return frozenset(found)


def is_entry_table(element: ParsedElement) -> bool:
    """A table whose rows are contents entries ("Hurdling .... 45")."""
    if element.element_type is not ElementType.TABLE:
        return False
    rows = [
        line for line in _entry_lines(element) if len(_measured(line)) <= CONTENTS_ENTRY_MAX_CHARS
    ]
    folios = [folio for line in rows if (folio := _folio(line)) is not None]
    return len(folios) >= CONTENTS_MIN_ENTRIES and len(folios) >= CONTENTS_MIN_SHARE * len(rows)


def contents_section_openers(doc: ParsedDocument, pages: frozenset[int]) -> frozenset[str]:
    """``element_id`` of the headings on contents pages that open a section of
    their own: a TITLE with no page number, not the Contents heading,
    whose next text on the page is prose ("PREFACE" then its first paragraph,
    on a booklet page that also holds the contents). Its section is not
    contents; a title followed by more entries ("one The Distances") is a
    contents line."""
    openers: set[str] = set()
    elements = [
        element
        for element in doc.elements
        if element.page_number in pages
        and element.element_type is not ElementType.IMAGE
        and any(ch.isalpha() for ch in element.text)
    ]
    for element, following in zip(elements, elements[1:], strict=False):
        if (
            element.element_type is ElementType.TITLE
            and following.page_number == element.page_number
            and _folio(" ".join(element.text.split())) is None
            and front_matter_kind(" ".join(element.text.split())) != "toc"
            and _is_prose(following)
        ):
            openers.add(element.element_id)
    return frozenset(openers)


__all__ = [
    "CONTENTS_FRONT_PAGES",
    "CONTENTS_FRONT_SHARE",
    "CONTENTS_MIN_ENTRIES",
    "CONTENTS_MIN_RISING",
    "CONTENTS_MIN_SHARE",
    "DOCUMENT_INDEX_LABEL",
    "contents_pages",
    "contents_section_openers",
    "is_body_paragraph",
    "is_entry_table",
]

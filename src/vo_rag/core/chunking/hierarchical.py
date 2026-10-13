from __future__ import annotations

import bisect
import os
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum

import structlog

from vo_rag.core.chunking.artifact_filters import (
    detect_running_lines,
    normalize_running_line,
    strip_page_furniture_text,
)
from vo_rag.core.chunking.chapter_path import derive_chapter_path
from vo_rag.core.chunking.composite_emitter import (
    CompositeChunk,
    CompositeEmitter,
    FigurePlacement,
    section_front_matter_kind,
)
from vo_rag.core.chunking.contents_pages import (
    contents_pages,
    contents_section_openers,
    is_body_paragraph,
    is_entry_table,
)
from vo_rag.core.chunking.counters import TokenCounterFn, naive_counter
from vo_rag.core.chunking.front_matter import front_matter_kind
from vo_rag.core.chunking.heading_filter import (
    is_back_matter_heading,
    is_degenerate_heading,
    is_entry_fragment_title,
    is_entry_list_heading,
    is_front_matter_line,
    is_index_entry_title,
    is_index_heading,
    is_labelled_heading,
    is_lead_in_line,
    is_section_numbered,
    is_structural_heading,
    reads_as_citations,
    reads_as_index,
    reads_as_reference_list,
)
from vo_rag.core.chunking.heading_tracker import (
    REFLOWABLE_PARSERS,
    HeadingTracker,
    TitleOnlyFallback,
)
from vo_rag.core.chunking.ids import _chunk_id
from vo_rag.core.chunking.models import DropReason, PageDrop, RetrievalChunk
from vo_rag.core.chunking.reading_order import in_reading_order
from vo_rag.core.chunking.section_packer import (
    APPENDED_TEXT_ELEMENT_TYPES,
    TEXT_BEARING_ELEMENT_TYPES,
    ProseSplitter,
    SectionPacker,
)
from vo_rag.core.chunking.short_text import ChunkScope, ShortTextMerger, SmallChunkMerger
from vo_rag.core.chunking.table_linearizer import TableLinearizer
from vo_rag.core.chunking.token_ceiling import TokenCeilingBackstop
from vo_rag.core.chunking.token_counter import default_token_counter
from vo_rag.core.parsing.contract import ElementType, ParsedDocument, ParsedElement
from vo_rag.core.parsing.text_normalization import normalize_extracted_text
from vo_rag.pipelines.book_rag.ocr_quality import is_word

log = structlog.get_logger(__name__)

__all__ = [
    "ElementRoute",
    "HierarchicalChunker",
    "ProseSplitter",
    "RetrievalChunk",
    "RoutingContext",
    "_chunk_id",
    "route_element",
]

# A genuine run of stacked standalone headings ("PART TWO" / "Sprints" /
# "The start") is a handful of lines. A longer run of short title-case lines is
# list content — a back-matter index, a results roster — and must stay body
# text: gluing it into the section path made a 3,125-char path that exhausted
# the token budget and chopped a book into ~297k one-character chunks (VOR-1416).
MAX_PENDING_HEADING_RUN = 4

# Running heads the parser typed as TITLE (VOR-1427). A real
# heading appears once, even when its words recur in its running head
# ("10.2. MEAN FIELD VI"), and must never be skipped with it. A TITLE is a
# running head only when it sits at a page edge (first or last text-bearing
# element of its page), is not the potential opener of its run, and either
# - carries an edge page number that follows the page ("412 MEAN FIELD VI",
#   "214 | Chapter 8: X", "3.1 MEAN FIELD 43"): its text recurs on
#   RUNNING_TITLE_MIN_PAGES+ pages with one printed-minus-physical offset
#   (PAGE_OFFSET_DRIFT either way), and it is not an enumerated label; or
# - carries a pipe-attached edge number ("203 | X", "The Setup | 203") that
#   is a page number (it sits within PAGE_OFFSET_DRIFT of the page plus the
#   print offset of the nearest page-numbered region: the offset of the
#   nearest page on which some other key's number follows that key's
#   dominant offset, an equidistant tie going to the longer adjacent run of
#   same-offset pages (a run survives gaps up to ANCHOR_RUN_MAX_PAGE_GAP: a
#   plate section or a part-title spread does not split a region), so
#   chapters printed at +200 and +203 each judge their own pages and a
#   distant region never decides between them; the page
#   itself when the book has no such region) and its text matches a real
#   heading earlier in its run: that layout is only ever a running head, so a
#   short section needs no page minimum. The same holds, in a book that has
#   such a region, for an unpiped number that follows the region's offset
#   when the title repeats an earlier member of its run in another surface
#   form ("METHODS 5" after "4 Methods"; "Lecture 5" after "Lecture 4"
#   shares the form and is the next lecture, and so is "LECTURE 5": two
#   label-shaped titles counting on by one are a numbered sequence, since a
#   head for "Lecture 4" would read "LECTURE 4 5"). A piped number that does not
#   follow the page ("Revenue | 2023", a caption with a year) is not a
#   running head, and captions never define a region: a key with no dominant
#   offset of its own never anchors one, nor does a key whose observations
#   inside its winning offset window share one constant number; or
# - repeats its exact text at page edges on RUNNING_TITLE_MIN_PAGES+ pages,
#   or is an unpiped constant-number head ("CHAPTER 3", "INDEX 999") opening
#   RUNNING_TITLE_MIN_PAGES+ consecutive pages; its run's first page stays
#   the possible opener.
# An enumerated label is one word followed by its number ("Step 3", "Chapter
# 2") whose family, in page order, counts from 1 and either restarts at 1 or
# steps by one ("Step 1..5" in two chapters is one family). Running heads
# never start at 1 because chapter openers carry no head, so "INTRODUCTION
# 2..15" and verso "2, 4, 6.." are head families, not labels (case-independent).
# A run is one title text (section numbers, edge numbers and punctuation
# removed, so "Chapter 8. X" and "Chapter 8: X" share a run) on pages at most
# RUN_MAX_PAGE_GAP apart. Running heads are left off a chapter's opening
# page, so a run's first title is its potential opener -- unless the run
# already has a title whose number does not follow the page (the real opener,
# "3 Variational Inference" beside "41 VARIATIONAL INFERENCE"). A first title
# that itself follows the page is an opener only when numbered 1 (chapter 1
# is the one chapter whose number can equal its printed page: "1
# Introduction" beside "INTRODUCTION 2") or when its surface form -- which
# side its number is on plus its case-exact text -- is unlike every other
# member of a run of three or more ("2 Methods" beside "3 METHODS", "12
# Results" beside "RESULTS 13"), or unlike the single other member of a run
# of two in its case-exact text ("16 Exercises" beside "EXERCISES 17" in a
# two-page chapter). "2 INDEX" starting a head family with no
# opener of its own shares its form with "3 INDEX" and is a running head, as
# is the first of outer-edge heads that alternate sides ("2 METHODS",
# "METHODS 3", "4 METHODS"), a pair of them cut off from their family by
# gaps ("INTRODUCTION 11", "12 INTRODUCTION"), and a first head that shares
# the run's case-exact text and differs only by the side of its number when
# that side is its page's outer edge (left on even printed pages, right on
# odd, the printed page being the physical page plus the nearest region's
# print offset: "2 METHODS" on printed page 2 beside "METHODS 3", "METHODS
# 5"). A pipe-attached number never gets the exemption.
RUNNING_TITLE_MIN_PAGES = 3
PAGE_OFFSET_DRIFT = 1
RUN_MAX_PAGE_GAP = 3
# A page-numbered region survives a plate section, a part-title spread or heads
# caught only every fourth page; anchors of one offset this far apart are one
# run when no other-offset anchor lies between them.
ANCHOR_RUN_MAX_PAGE_GAP = 10
_LEADING_EDGE_NUMBER_RE = re.compile(r"^(\d+)(?:\s*\|\s*|\s+)")
_TRAILING_EDGE_NUMBER_RE = re.compile(r"(?:\s*\|\s*|\s+)(\d+)$")
_LEADING_SECTION_NUMBER_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)+")
_ENUMERATED_LABEL_RE = re.compile(r"^([^\W\d_]+) (\d+)$")
_PUNCTUATION_RE = re.compile(r"[^\w\s]+")

# OCR-variant running heads (VOR-1452): a page header the OCR spells
# differently on each page ("TRACK AND FIELD: FIELD EVENTS" / "... HELD
# EVENTS" / "TRACK AND HELD: HELD EVENTS" / "... FIELD EUENTS"). Titles at a
# page edge are folded (letters only, case and spacing folded) and grouped
# when the folded texts are OCR_HEAD_SIMILARITY alike (difflib ratio) AND have
# the same number of words, each word pair OCR_HEAD_WORD_SIMILARITY alike.
# The measured variants score 0.89-0.96 overall and 0.67 ("field"/"held") or
# more per word; distinct headings that look alike fail the per-word test
# ("Exercise A"/"B": "a"/"b" 0.0; "Warm Up"/"Warm Down Drills": 0.8 overall
# but "up"/"down" 0.0; "Jumping"/"Throwing Events": 0.69 overall). A group is a
# running head only when it spans RUNNING_TITLE_MIN_PAGES+ pages AND holds at
# least two distinct spellings: an author's real heading repeated on several
# pages ("Jumping Events") is spelled the same every time, so OCR garbling is
# the evidence. Titles with digits are left to the page-number rules above
# ("Step 3"/"Step 4" are labels), and so are folded texts under
# OCR_HEAD_MIN_CHARS. Every member except one on page 1 (the title page) is
# skipped, so a paragraph continues across the page.
#
# Two real headings can look like OCR spellings of one
# another ("Block Start"/"Block Starts" on pp. 10/21/31, "Sprint Drills"/
# "Sprint Skills"). So a group's pages must form a run like a running head's:
# at least RUNNING_TITLE_MIN_PAGES pages, each at most OCR_HEAD_RUN_MAX_PAGE_GAP
# after the last (a verso-only head skips a page, and a chapter opener one
# more: one book's head sits on pp. 12, 14, 16, 20, 24, ...). Only the members
# in such runs are skipped. Within a run, the group also needs evidence that
# it is one text: some spelling differs from the representative in a word that
# is not a dictionary word (the OCR garble, "EUENTS"), or the representative
# spelling is itself on RUNNING_TITLE_MIN_PAGES+ pages of the run -- a known
# running head, whose page-edge variants ("HELD EVENTS") are skipped with it
# Two real headings alternating ("Drills"/"Skills") have
# neither.
#
# Grouping compares each spelling with every group's
# representative, O(distinct edge titles^2) with difflib. Groups are bucketed
# by word count, a pair whose lengths already rule out OCR_HEAD_SIMILARITY is
# never scored, difflib's cheap upper bound (``quick_ratio``, letter
# multisets) runs before ``ratio()``, and per document at most
# OCR_HEAD_MAX_QUICK_COMPARISONS bounds and OCR_HEAD_MAX_COMPARISONS full
# scores are computed (well under 1 s together). Real headings rarely share a letter
# multiset, so a real book needs a small fraction of either; past them,
# spellings stay ungrouped -- the heading is kept, the pre-VOR-1452 behaviour.
OCR_HEAD_SIMILARITY = 0.8
OCR_HEAD_WORD_SIMILARITY = 0.6
OCR_HEAD_MIN_CHARS = 8
OCR_HEAD_RUN_MAX_PAGE_GAP = 5
OCR_HEAD_MAX_QUICK_COMPARISONS = 100_000
OCR_HEAD_MAX_COMPARISONS = 5_000
_NON_LETTER_RE = re.compile(r"[^a-z]+")


class ElementRoute(StrEnum):
    """Which branch of ``HierarchicalChunker.chunk`` an element takes.

    Shared with ``scripts/replay_headings.py`` so the heading replay fixtures
    record exactly the routing the chunker applies.
    """

    SKIP_HEADER_FOOTER = "skip_header_footer"
    SKIP_REPEATED = "skip_repeated"
    UPDATE_TITLE = "update_title"
    TITLE_REJECTED = "title_rejected"
    SKIP_QUARANTINED = "skip_quarantined"
    TABLE = "table"
    LIST_ITEM = "list_item"
    PENDING_HEADING = "pending_heading"
    BODY = "body"
    IGNORED = "ignored"


@dataclass(frozen=True)
class RoutingContext:
    """Document-level inputs to ``route_element``."""

    repeated_short_text: frozenset[str]
    quarantined: frozenset[int]
    #: ``element_id`` of every TITLE that is a running head (see above).
    running_head_titles: frozenset[str] = frozenset()
    #: ``element_id`` of the first and last lettered element of each page: only
    #: there is repeated short text page furniture. ``None``
    #: (a hand-built context) leaves the position unchecked.
    page_edge_ids: frozenset[str] | None = None
    #: Words of the document's body text (VOR-1454): a heading none of whose
    #: words the book uses and no dictionary knows is garbled OCR. ``None``
    #: (a hand-built context) skips the check.
    vocabulary: frozenset[str] | None = None
    #: ``element_id`` of every TITLE that is a clipped piece of another title
    #: ("eriodization of Strength" beside "Periodization of Strength"; VOR-1454).
    clipped_titles: frozenset[str] = frozenset()
    #: Whether the document sets headings in lower case ("lower-body plyos");
    #: when it does not, a TITLE opening in lower case is a piece of a line
    #: ("the power transferred to the implement", "delivery"; VOR-1454).
    lower_case_headings: bool = True
    #: Pages of front matter (``_front_matter_pages``): an imprint line there
    #: is no heading.
    front_pages: int = 0
    #: ``element_id`` of TITLEs between two tables: a table's own row or
    #: column header split off by the parser ("Dates", "Comments"; VOR-1454).
    table_cell_titles: frozenset[str] = frozenset()

    @classmethod
    def for_document(cls, doc: ParsedDocument) -> RoutingContext:
        repeated_short_text = frozenset(
            SectionPacker.repeated_short_page_text(doc.elements, doc.total_pages)
        )
        # The EPUB/MOBI parser's "pages" are chapters, each opened by its title:
        # "Chapter N" on page N + constant is the chapter, never a running head.
        running_head_titles = (
            frozenset()
            if doc.parser_used in REFLOWABLE_PARSERS
            else _running_head_titles(doc.elements, repeated_short_text)
        )
        # Born-digital ebooks carry no OCR garble to judge.
        vocabulary = (
            None if doc.parser_used in REFLOWABLE_PARSERS else _body_vocabulary(doc.elements)
        )
        return cls(
            repeated_short_text=repeated_short_text,
            quarantined=frozenset(doc.quarantined_pages),
            running_head_titles=running_head_titles,
            page_edge_ids=_lettered_page_edge_ids(doc.elements),
            vocabulary=vocabulary,
            clipped_titles=_clipped_titles(doc.elements, vocabulary or frozenset()),
            table_cell_titles=_table_cell_titles(doc.elements),
            front_pages=_front_matter_pages(doc),
            lower_case_headings=(
                doc.parser_used in REFLOWABLE_PARSERS or _lower_case_headings(doc.elements)
            ),
        )


#: A title text recurring at least this often, on at least this many pages, may
#: be a generic subheading.
GENERIC_SUBHEADING_MIN_OCCURRENCES = 5


def _heading_text_key(text: str) -> str:
    return " ".join(normalize_extracted_text(text).split()).casefold().rstrip(":")


def generic_subheadings(doc: ParsedDocument, routing: RoutingContext) -> frozenset[str]:
    """Title texts that head an item's part, not a section.

    A drill book heads the tips of every drill with the same words ("COACHING
    TIPS" 205 times, "Variation", "Key point"); the parser often ranks them
    above the drill (a short all-caps line is depth 1), so each one closed the
    drill and its chapter and the next drill nested under it. A title text is
    such a generic subheading when it recurs GENERIC_SUBHEADING_MIN_OCCURRENCES+
    times on as many pages and appears at least twice within one chapter (the
    span between two top-level titles that do not recur). A per-chapter
    heading ("Summary", "Introduction") recurs once per chapter and stays a
    section. Structural and back-matter headings never are generic.
    """
    titles: list[tuple[str, int, int | None]] = []
    for element in doc.elements:
        if element.element_type is not ElementType.TITLE:
            continue
        text = normalize_extracted_text(element.text)
        if route_element(element, text, routing) is not ElementRoute.UPDATE_TITLE:
            continue
        titles.append(
            (_heading_text_key(text), max(1, element.section_depth or 1), element.page_number)
        )
    counts = Counter(key for key, _, _ in titles)
    pages: defaultdict[str, set[int | None]] = defaultdict(set)
    for key, _, page in titles:
        pages[key].add(page)
    recurring = {
        key
        for key, count in counts.items()
        if count >= GENERIC_SUBHEADING_MIN_OCCURRENCES
        and len(pages[key]) >= GENERIC_SUBHEADING_MIN_OCCURRENCES
        and not is_structural_heading(key)
        and not is_back_matter_heading(key)
    }
    if not recurring:
        return frozenset()
    top = min((depth for key, depth, _ in titles if key not in recurring), default=1)
    in_chapter: Counter[str] = Counter()
    generic: set[str] = set()
    for key, depth, _ in titles:
        if key not in recurring:
            if depth <= top:
                in_chapter.clear()
            continue
        in_chapter[key] += 1
        if in_chapter[key] >= 2:
            generic.add(key)
    return frozenset(generic)


def _generic_depth(
    text: str, depth: int, open_headings: dict[int, str], generic: frozenset[str]
) -> int:
    """A generic subheading's depth: one below the deepest open heading that is
    not itself generic, so "COACHING TIPS" heads the tips of the drill before
    it (Chapter > Drill > COACHING TIPS) instead of closing the drill."""
    if _heading_text_key(text) not in generic:
        return depth
    owners = [
        level
        for level, heading in open_headings.items()
        if _heading_text_key(heading) not in generic
    ]
    return max(owners) + 1 if owners else depth


#: Words of each line in a run of standalone lines read as a diagram's labels.
LABEL_RUN_MAX_WORDS = 2


def _label_run(lines: list[str]) -> bool:
    """Two or more one- or two-word standalone lines in a row ("Carry" / "Run" /
    "Withdrawal" / "Cross-Step"): the labels of a figure, not nested sections
    (VOR-1454). A structural line ("PART TWO") heads what follows."""
    return len(lines) >= 2 and all(
        len(line.split()) <= LABEL_RUN_MAX_WORDS and not is_structural_heading(line)
        for line in lines
    )


def _reads_as_words(text: str) -> bool:
    """At least half the lettered tokens of ``text`` are real words: a standalone
    line may label its section ("Monday:", "Order Forms"); OCR noise ("Vy.datt
    LGR") stays text only."""
    tokens = [token for token in _NON_LETTER_RE.split(text.casefold()) if len(token) >= 2]
    return bool(tokens) and 2 * sum(1 for token in tokens if is_word(token)) >= len(tokens)


# Front matter (VOR-1454): the pages before the book's body -- cover, library
# stamps, title, half-title, series and copyright pages. A heading there (a
# stamp, the book's title, an author's name, ranked depth 1 because it is set
# in capitals) never parents a heading on a later page
# (``HeadingTracker.front_pages``). A page of prose holds a paragraph of
# FRONT_PROSE_MIN_WORDS+ words with two sentence stops that is not a
# copyright or cataloging notice; the body starts at the first page of prose
# with another within FRONT_PROSE_RUN_GAP pages. A document whose first page
# starts the body (an article, a chapter extract) has no front matter, nor does
# one whose body does not start within FRONT_MATTER_MAX_PAGES pages (nothing
# to tell it by).
FRONT_PROSE_MIN_WORDS = 20
FRONT_MATTER_MAX_PAGES = 20
FRONT_PROSE_RUN_GAP = 2
_SENTENCE_STOP_RE = re.compile(r"(?<=[a-z]{2})[.!?](?:\s|$)")
_NOTICE_RE = re.compile(
    r"rights reserved|\bisbn\b|library of congress|catalog(?:u)?ing|printed in|published by"
    r"|copyright|\u00a9|no part of this",
    re.IGNORECASE,
)


def _is_prose(text: str) -> bool:
    return (
        len(text.split()) >= FRONT_PROSE_MIN_WORDS
        and len(_SENTENCE_STOP_RE.findall(text)) >= 2
        and not _NOTICE_RE.search(text)
    )


def _front_matter_pages(doc: ParsedDocument) -> int:
    """Pages before the document's body (see above): the body starts at the
    first page of prose followed by another within FRONT_PROSE_RUN_GAP pages --
    a lone blurb or a praise page inside the cover is not the body."""
    if doc.parser_used in REFLOWABLE_PARSERS:
        return 0  # an ebook's "pages" are its chapters
    limit = FRONT_MATTER_MAX_PAGES + FRONT_PROSE_RUN_GAP
    prose = sorted(
        {
            element.page_number
            for element in doc.elements
            if element.page_number is not None
            and element.page_number <= limit
            and element.element_type is ElementType.NARRATIVE_TEXT
            and _is_prose(element.text)
        }
    )
    for page, following in zip(prose, prose[1:], strict=False):
        if page > FRONT_MATTER_MAX_PAGES:
            break
        if following - page <= FRONT_PROSE_RUN_GAP:
            return page - 1
    return 0  # no body found where it should start: nothing to tell


_LETTER_RUN_RE = re.compile(r"[^\W\d_]+")
#: Share of a document's words that are English dictionary words above which
#: its vocabulary is filtered for OCR garble (``_body_vocabulary``).
VOCABULARY_ENGLISH_SHARE = 0.5
#: An element vouches for its words when it has this many longer words and
#: this share of them are dictionary words: prose, not an OCR-garbled table.
VOCABULARY_MIN_WORDS = 8
VOCABULARY_ELEMENT_SHARE = 0.7
#: Element types whose words make up the body vocabulary.
#: Running text only: tables and captions are where OCR garble collects.
_VOCABULARY_TYPES: frozenset[ElementType] = frozenset(
    {ElementType.NARRATIVE_TEXT, ElementType.LIST_ITEM}
)


def _body_vocabulary(elements: Sequence[ParsedElement]) -> frozenset[str] | None:
    """Case-folded words (letter runs) of the document's readable body text,
    or ``None`` for a text not in English.

    An element counts when it is prose: VOCABULARY_MIN_WORDS+ longer letter
    runs (3+ letters), VOCABULARY_ELEMENT_SHARE of them dictionary words: an
    OCR-garbled table ("Arenuer ALVC / LIOOI") would
    otherwise vouch for the same garble typed as a heading. In a text in
    another language (under VOCABULARY_ENGLISH_SHARE of its runs are English
    words) the dictionary cannot tell garble from words, so none is judged.
    """
    per_element: list[tuple[list[str], int, int]] = []
    for element in elements:
        if element.element_type not in _VOCABULARY_TYPES:
            continue
        tokens = _LETTER_RUN_RE.findall(element.text.casefold())
        long_tokens = [token for token in tokens if len(token) >= 3]
        known = sum(1 for token in long_tokens if is_word(token))
        per_element.append((tokens, known, len(long_tokens)))
    total_known = sum(known for _, known, _ in per_element)
    total_long = sum(count for _, _, count in per_element)
    if total_long == 0 or total_known < VOCABULARY_ENGLISH_SHARE * total_long:
        return None
    return frozenset(
        token
        for tokens, known, count in per_element
        if count >= VOCABULARY_MIN_WORDS and known >= VOCABULARY_ELEMENT_SHARE * count
        for token in tokens
    )


def _supported(token: str, vocabulary: frozenset[str]) -> bool:
    return token in vocabulary or is_word(token)


def _garbled(text: str, vocabulary: frozenset[str]) -> bool:
    """Most words of a heading are neither the book's words nor dictionary
    words: OCR garble ("RW ALVG", "SUPERTRA/NJNC", "TEESE RINGS VART",
    "D 1SCUS"; VOR-1454). The book's own vocabulary keeps its names, terms and
    other languages; single letters are not judged."""
    tokens = [token for token in _LETTER_RUN_RE.findall(text.casefold()) if len(token) >= 2]
    if not tokens:
        return False
    supported = sum(1 for token in tokens if _supported(token, vocabulary))
    return 2 * supported < len(tokens)


def _ends_inside_word(other: str, piece: str) -> bool:
    """``other`` ends with ``piece`` and a letter comes just before it."""
    return len(other) > len(piece) and other.endswith(piece) and other[-len(piece) - 1].isalpha()


def _starts_inside_word(other: str, piece: str) -> bool:
    """``other`` starts with ``piece`` and a letter comes just after it."""
    return len(other) > len(piece) and other.startswith(piece) and other[len(piece)].isalpha()


def _clipped_titles(
    elements: Sequence[ParsedElement], vocabulary: frozenset[str]
) -> frozenset[str]:
    """``element_id`` of TITLEs that are a clipped piece of a heading (VOR-1454).

    A running head or title cut at the page edge loses its first letters
    ("eriodization of Strength", "hulsport", "aching Techniques") or its last
    ones ("Periodization of Stren"). A title is clipped when it starts in lower
    case and its first word is not a word, or when another title of the
    document continues it inside a word: ends with it after a letter (the
    lower-case start), or starts with it before a letter (the last word not a
    word).
    """
    titles = [
        (element.element_id, " ".join(normalize_extracted_text(element.text).split()))
        for element in elements
        if element.element_type is ElementType.TITLE
    ]
    by_last: defaultdict[str, set[str]] = defaultdict(set)
    by_first: defaultdict[str, set[str]] = defaultdict(set)
    for _, text in titles:
        words = text.casefold().split()
        if words:
            by_last[words[-1]].add(text.casefold())
            by_first[words[0]].add(text.casefold())
    clipped: set[str] = set()
    for element_id, text in titles:
        folded = text.casefold()
        words = folded.split()
        if len(folded) < 3 or not words:
            continue
        first_alpha = next((ch for ch in text if ch.isalpha()), "")
        if first_alpha.islower():
            first = _LETTER_RUN_RE.findall(words[0])
            if first and not _supported(first[0], vocabulary):
                clipped.add(element_id)
                continue
            if any(_ends_inside_word(other, folded) for other in by_last[words[-1]]):
                clipped.add(element_id)
                continue
        last = _LETTER_RUN_RE.findall(words[-1])
        if (
            last
            and not _supported(last[-1], vocabulary)
            and any(_starts_inside_word(other, folded) for other in by_first[words[0]])
        ):
            clipped.add(element_id)
    return frozenset(clipped)


#: Pages a standalone line must recur on to be a running line, not a heading.
REPEATED_LINE_MIN_PAGES = 3
#: Longest line (words) judged by ``_repeated_lines``.
REPEATED_LINE_MAX_WORDS = 8


def _line_key(text: str) -> str:
    return " ".join(_LETTER_RUN_RE.findall(text.casefold()))


def _repeated_lines(elements: Sequence[ParsedElement]) -> frozenset[str]:
    """Keys of short standalone text lines found on REPEATED_LINE_MIN_PAGES+
    pages: a running head the parser typed as text ("Periodization" on 140
    pages), never a heading of the text below it (VOR-1454)."""
    pages: defaultdict[str, set[int]] = defaultdict(set)
    for element in elements:
        if element.page_number is None or element.element_type not in _EDGE_TEXT_TYPES:
            continue
        if len(element.text.split()) > REPEATED_LINE_MAX_WORDS:
            continue
        key = _line_key(element.text)
        if key:
            pages[key].add(element.page_number)
    return frozenset(key for key, seen in pages.items() if len(seen) >= REPEATED_LINE_MIN_PAGES)


#: Share of a heading's letters in upper case from which it is a capitalised
#: heading (``_case_depth``).
CAPS_HEADING_SHARE = 0.85
CAPS_HEADING_MIN_LETTERS = 6


def _case_depth(text: str, depth: int) -> int:
    """Depth 1 for a capitalised heading the parser ranked 2 (VOR-1454).

    The PDF parser ranks a heading 1 when all its letters are capitals, else 2
    (``docling_mapping._infer_heading_level``), so a unit or a stray
    lower-case letter ("THE 4x 100m. RELAY EXCHANGE") sank a chapter title
    below the chapter before it. A heading whose letters are nearly all
    capitals is ranked with them; section-numbered headings keep their rank.
    """
    if depth != 2 or is_section_numbered(text):
        return depth
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < CAPS_HEADING_MIN_LETTERS:
        return depth
    upper = sum(1 for ch in letters if ch.isupper())
    return 1 if upper >= CAPS_HEADING_SHARE * len(letters) else depth


#: Capitalised headings a book needs before their placement is judged, and the
#: share of them opening a page under which they are inline headings.
INLINE_CAPS_MIN_HEADINGS = 20
INLINE_CAPS_OPENER_SHARE = 0.3


#: A book whose capitalised headings are at most this share of its mixed-case
#: ones (and at most RARE_CAPS_MAX_HEADINGS) sets its headings in mixed case.
RARE_CAPS_SHARE = 0.05
RARE_CAPS_MAX_HEADINGS = 5
RARE_CAPS_MIN_MIXED = 40


def _rare_capitals(elements: Sequence[ParsedElement], routing: RoutingContext) -> bool:
    """Whether capitalised headings are rare exceptions in a mixed-case book."""
    caps = mixed = 0
    for element in elements:
        if element.element_type is not ElementType.TITLE:
            continue
        text = normalize_extracted_text(element.text)
        if is_structural_heading(text) or is_section_numbered(text):
            continue
        if route_element(element, text, routing) is not ElementRoute.UPDATE_TITLE:
            continue
        if _case_depth(text, 2) == 1:
            caps += 1
        else:
            mixed += 1
    return (
        mixed >= RARE_CAPS_MIN_MIXED
        and caps <= RARE_CAPS_MAX_HEADINGS
        and caps <= RARE_CAPS_SHARE * mixed
    )


def _inline_capitals(
    elements: Sequence[ParsedElement], routing: RoutingContext, openers: frozenset[str]
) -> bool:
    """Whether the book sets run-in item headings in capitals (VOR-1454).

    The parser ranks every all-capitals heading above every mixed-case one.
    In a book whose capitalised headings mostly sit mid-page -- drill names
    ("WALL CATCH DRILL"), exercise and photo titles ("SNATCH TECHNIQUE") --
    they are items, and the mixed-case chapter sections after them are not
    theirs. Chapters open pages; items do not.
    """
    caps: list[bool] = []
    for element in elements:
        if element.element_type is not ElementType.TITLE:
            continue
        text = normalize_extracted_text(element.text)
        if is_structural_heading(text) or is_section_numbered(text):
            continue
        if route_element(element, text, routing) is not ElementRoute.UPDATE_TITLE:
            continue
        if _case_depth(text, 2) == 1:
            caps.append(element.element_id in openers)
    return len(caps) >= INLINE_CAPS_MIN_HEADINGS and sum(caps) < INLINE_CAPS_OPENER_SHARE * len(
        caps
    )


#: Letters an element needs to count as a page's content for
#: ``_page_opening_titles`` (a stray OCR mark "ia" does not).
_OPENER_MIN_LETTERS = 4


def _page_opening_titles(
    elements: Sequence[ParsedElement], routing: RoutingContext
) -> frozenset[str]:
    """``element_id`` of TITLEs with nothing but headings before them on their
    page (page furniture and stray OCR marks aside; VOR-1454)."""
    openers: set[str] = set()
    closed: set[int] = set()
    for element in elements:
        page = element.page_number
        if page is None or page in closed:
            continue
        if element.element_type not in TEXT_BEARING_ELEMENT_TYPES and (
            element.element_type is not ElementType.TABLE
        ):
            continue
        if sum(1 for ch in element.text if ch.isalpha()) < _OPENER_MIN_LETTERS:
            continue
        text = normalize_extracted_text(element.text)
        route = route_element(element, text, routing)
        if route in _SKIPPED_ROUTES:
            continue
        if element.element_type is ElementType.TITLE:
            openers.add(element.element_id)
        else:
            closed.add(page)
    return frozenset(openers)


#: Share of a document's TITLEs opening in lower case from which lower case is
#: its heading style.
LOWER_CASE_HEADING_SHARE = 0.3
#: Longest TITLE (words) counted for the heading style: longer ones are mostly
#: sentences typed as titles, whatever the style.
LOWER_CASE_STYLE_MAX_WORDS = 6


def _lower_case_headings(elements: Sequence[ParsedElement]) -> bool:
    """Whether the document sets its headings in lower case (VOR-1454)."""
    starts = [
        first
        for element in elements
        if element.element_type is ElementType.TITLE
        and len(element.text.split()) <= LOWER_CASE_STYLE_MAX_WORDS
        and is_degenerate_heading(element.text) is None
        and (first := next((ch for ch in element.text if ch.isalpha()), ""))
    ]
    lower = sum(1 for first in starts if first.islower())
    return bool(starts) and lower >= LOWER_CASE_HEADING_SHARE * len(starts)


def _table_cell_titles(elements: Sequence[ParsedElement]) -> frozenset[str]:
    """``element_id`` of TITLEs with a table right before and right after
    them (page furniture and other TITLEs aside): cells of a table the parser
    split, never a section (VOR-1454)."""
    cells: set[str] = set()
    pending: list[str] = []
    after_table = False
    for element in elements:
        kind = element.element_type
        if kind is ElementType.TITLE:
            if after_table:
                pending.append(element.element_id)
            continue
        if kind in {ElementType.HEADER, ElementType.FOOTER, ElementType.PAGE_BREAK}:
            continue
        if kind is not ElementType.TABLE and not element.text.strip():
            continue
        if kind is ElementType.TABLE:
            cells.update(pending)
        pending = []
        after_table = kind is ElementType.TABLE
    return frozenset(cells)


def _table_titles(elements: Sequence[ParsedElement]) -> frozenset[str]:
    """``element_id`` of TITLEs whose next content element is a TABLE: the
    table's title ("UNITS OF MOMENTUM" / "MASS IN POUNDS" over a chart),
    other TITLEs between them aside (VOR-1454). A structural heading ("Chapter
    3 Training Plans") before a table is still the chapter's."""
    heads: set[str] = set()
    pending: list[str] = []
    texts = {e.element_id: e.text for e in elements if e.element_type is ElementType.TITLE}
    for element in elements:
        if element.element_type is ElementType.TITLE:
            pending.append(element.element_id)
            continue
        if element.element_type in {ElementType.HEADER, ElementType.FOOTER, ElementType.PAGE_BREAK}:
            continue
        if not element.text.strip() and element.element_type is not ElementType.TABLE:
            continue
        if element.element_type is ElementType.TABLE:
            heads.update(
                element_id
                for element_id in pending
                if not is_structural_heading(normalize_extracted_text(texts[element_id]))
            )
        pending = []
    return frozenset(heads)


def _page_edge_running_lines(
    doc: ParsedDocument, routing: RoutingContext, packer: SectionPacker
) -> set[str]:
    """The running lines to strip from text: those found at page edges, less
    any whose words also stand as a whole element inside a page:
    "Discus Throw" heading a section mid-page is the book's text, while its
    copies at the page edges are skipped as running heads by routing. A line
    carrying a page number stays furniture wherever it stands."""
    running = detect_running_lines(packer.page_text_for_running_line_detection(doc))
    if not running or routing.page_edge_ids is None:
        return running
    mid_page = {
        key
        for element in doc.elements
        if element.element_id not in routing.page_edge_ids
        and element.element_type in _EDGE_TEXT_TYPES
        and (key := normalize_running_line(element.text)) in running
        and "#" not in key  # a page marker ("Page 47", "<?dp n=74 ?>") is furniture anywhere
    }
    return running - mid_page


def _lettered_page_edge_ids(elements: list[ParsedElement]) -> frozenset[str]:
    """``element_id`` of the first and the last element of each page that has
    letters, tables and pictures aside: where a running head or footer sits
    (a page number beside it has no letters)."""
    first: dict[int, str] = {}
    last: dict[int, str] = {}
    for element in elements:
        if (
            element.page_number is None
            or element.element_type not in _EDGE_TEXT_TYPES
            or not any(ch.isalpha() for ch in element.text)
        ):
            continue
        first.setdefault(element.page_number, element.element_id)
        last[element.page_number] = element.element_id
    return frozenset(first.values()) | frozenset(last.values())


_EDGE_TEXT_TYPES: frozenset[ElementType] = frozenset(
    {
        ElementType.NARRATIVE_TEXT,
        ElementType.TITLE,
        ElementType.LIST_ITEM,
        ElementType.FIGURE_CAPTION,
        ElementType.HEADER,
        ElementType.FOOTER,
    }
)


@dataclass(frozen=True)
class _TablePosition:
    """The section a table sits in: its label and its merge scope."""

    section_path: str
    section_title: str
    epoch: int
    headings: tuple[tuple[int, str, int], ...]


def _title_key(text: str) -> str:
    return " ".join(text.split()).casefold()


def _split_edge_number(text: str) -> tuple[str, int | None, bool]:
    """``(key without an edge number, the number, whether a pipe attaches it)``;
    dotted section numbers ("10.2 ...") are not edge numbers."""
    collapsed = " ".join(text.split())
    leading = _LEADING_EDGE_NUMBER_RE.match(collapsed)
    if leading:
        return collapsed[leading.end() :].casefold(), int(leading.group(1)), "|" in leading[0]
    trailing = _TRAILING_EDGE_NUMBER_RE.search(collapsed)
    if trailing:
        return collapsed[: trailing.start()].casefold(), int(trailing.group(1)), "|" in trailing[0]
    return collapsed.casefold(), None, False


def _run_key(text: str) -> str:
    """Title text without edge page numbers, section numbering and punctuation."""
    key, _, _ = _split_edge_number(text)
    return " ".join(_PUNCTUATION_RE.sub(" ", _LEADING_SECTION_NUMBER_RE.sub("", key)).split())


def _surface_form(text: str) -> tuple[str, str]:
    """``(side of the edge number, case-exact text without it or punctuation)``:
    "2 Methods" -> ("leading", "Methods"), "RESULTS 13" -> ("trailing", "RESULTS")."""
    collapsed = " ".join(text.split())
    leading = _LEADING_EDGE_NUMBER_RE.match(collapsed)
    if leading:
        side, rest = "leading", collapsed[leading.end() :]
    else:
        trailing = _TRAILING_EDGE_NUMBER_RE.search(collapsed)
        if trailing:
            side, rest = "trailing", collapsed[: trailing.start()]
        else:
            side, rest = "none", collapsed
    return side, " ".join(_PUNCTUATION_RE.sub(" ", rest).split())


def _enumerated_label_families(titles: list[tuple[int, int, str]]) -> set[str]:
    """Words whose "Word N" titles, in page order, count from 1 and at each
    step either restart at 1 or advance by one ("Step 1..5" in two chapters).
    ``titles`` are ``(page, order, text)``."""
    numbers: defaultdict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for page, order, text in titles:
        match = _ENUMERATED_LABEL_RE.match(" ".join(text.split()))
        if match:
            numbers[match.group(1).casefold()].append((page, order, int(match.group(2))))
    families: set[str] = set()
    for word, observed in numbers.items():
        sequence = [number for _, _, number in sorted(observed)]
        if sequence[0] == 1 and all(
            number in (1, previous + 1)
            for previous, number in zip(sequence, sequence[1:], strict=False)
        ):
            families.add(word)
    return families


def _dominant_offset(observed: list[tuple[int, int]]) -> int | None:
    """The page offset shared (within PAGE_OFFSET_DRIFT) by the most distinct
    pages, if that is RUNNING_TITLE_MIN_PAGES or more. Linear in the number
    of observations: pages are grouped per offset once, then each offset is
    combined with its neighbours."""
    pages_by_offset: defaultdict[int, set[int]] = defaultdict(set)
    for offset, page in observed:
        pages_by_offset[offset].add(page)
    best: tuple[int, int] | None = None  # (page count, -offset)
    for candidate in sorted(pages_by_offset):
        pages: set[int] = set()
        for delta in range(-PAGE_OFFSET_DRIFT, PAGE_OFFSET_DRIFT + 1):
            pages |= pages_by_offset.get(candidate + delta, set())
        score = (len(pages), -candidate)
        if best is None or score > best:
            best = score
    if best is None or best[0] < RUNNING_TITLE_MIN_PAGES:
        return None
    return -best[1]


def _key_offset(observed: list[tuple[int, int]]) -> int | None:
    """The key's dominant offset, unless the winning window holds one constant
    number. A page number varies with the page: a key whose observations
    inside the window share one number over several pages ("Revenue | 2023"
    over a continued table on three consecutive pages, whose offsets
    2020/2019/2018 sit within drift of each other) is not carrying one and gets
    no offset, whatever the key shows elsewhere in the book."""
    offset = _dominant_offset(observed)
    if offset is None:
        return None
    window = [(o, page) for o, page in observed if abs(o - offset) <= PAGE_OFFSET_DRIFT]
    if len({page for _, page in window}) > 1 and len({o + page for o, page in window}) == 1:
        return None
    return offset


def _anchor_run_sizes(anchors: list[tuple[int, int]]) -> list[int]:
    """For each of the page-sorted ``anchors`` (``(page, offset)``), the size of
    its run: the longest sequence of anchors around it that share its offset,
    with no other-offset anchor between them and no gap wider than
    ANCHOR_RUN_MAX_PAGE_GAP pages."""
    sizes = [0] * len(anchors)
    start = 0
    for index in range(1, len(anchors) + 1):
        if (
            index == len(anchors)
            or anchors[index][1] != anchors[start][1]
            or anchors[index][0] - anchors[index - 1][0] > ANCHOR_RUN_MAX_PAGE_GAP
        ):
            sizes[start:index] = [index - start] * (index - start)
            start = index
    return sizes


def _nearest_offset(
    page: int, anchors: list[tuple[int, int]], run_sizes: Sequence[int] | None = None
) -> int:
    """The print offset of the page-numbered region nearest ``page``: among the
    ``anchors`` (``(page, offset)``, sorted by page) at the smallest page
    distance, the offset whose adjacent run is the longest (``run_sizes``, see
    _anchor_run_sizes, computed when not given), then the lowest. A region
    far from ``page`` never decides a tie between the ones beside it. 0 when
    the book has no anchors."""
    if not anchors:
        return 0
    if run_sizes is None:
        run_sizes = _anchor_run_sizes(anchors)
    after = bisect.bisect_left(anchors, page, key=lambda anchor: anchor[0])
    before = after - 1
    distance = min(
        abs(anchors[index][0] - page) for index in (before, after) if 0 <= index < len(anchors)
    )
    nearest: list[int] = []
    while before >= 0 and page - anchors[before][0] == distance:
        nearest.append(before)
        before -= 1
    while after < len(anchors) and anchors[after][0] - page == distance:
        nearest.append(after)
        after += 1
    return -max((run_sizes[index], -anchors[index][1]) for index in nearest)[1]


def _on_outer_edge(side: str, printed: int) -> bool:
    """Outer-edge page numbers sit on the left of even (verso) pages and on the
    right of odd (recto) pages; recto and verso follow the printed page number,
    which an odd print offset inverts relative to the physical page."""
    return (side == "leading") == (printed % 2 == 0)


def _page_edge_element_ids(elements: list[ParsedElement]) -> tuple[set[str], set[str]]:
    """``element_id`` of the first, and of the last, text-bearing element of
    each page."""
    first: dict[int, str] = {}
    last: dict[int, str] = {}
    for element in elements:
        if (
            element.page_number is None
            or element.element_type not in TEXT_BEARING_ELEMENT_TYPES
            or not element.text.strip()
        ):
            continue
        first.setdefault(element.page_number, element.element_id)
        last[element.page_number] = element.element_id
    return set(first.values()), set(last.values())


def _longest_consecutive(pages: set[int]) -> int:
    """Length of the longest run of consecutive page numbers in ``pages``."""
    best = 0
    for page in pages:
        if page - 1 not in pages:
            length = 1
            while page + length in pages:
                length += 1
            best = max(best, length)
    return best


@dataclass(frozen=True)
class _TitleObservation:
    page: int
    order: int  # position in the document's title sequence
    element_id: str
    number: int | None  # edge page number, if any
    piped: bool  # the edge number is attached with a pipe
    tracks: bool  # the edge number follows the page at the key's dominant offset
    paged: bool  # key has no dominant offset, but the number follows the book's print offset
    label: bool  # an enumerated label ("Step 3") in a family counting from 1
    shaped: bool  # label-shaped, "Word N", whatever its family does
    repeats: bool  # exact text repeated at page edges on RUNNING_TITLE_MIN_PAGES+ pages
    form: tuple[str, str]  # number side + case-exact text (see _surface_form)
    printed: int  # the page as printed: its own number when it tracks, else the region's

    @property
    def position(self) -> tuple[int, int]:
        return (self.page, self.order)


def _split_runs(observations: list[_TitleObservation]) -> list[list[_TitleObservation]]:
    runs: list[list[_TitleObservation]] = []
    for observation in sorted(observations, key=lambda o: o.position):
        if runs and observation.page - runs[-1][-1].page <= RUN_MAX_PAGE_GAP:
            runs[-1].append(observation)
        else:
            runs.append([observation])
    return runs


def _numbered_sequence(earlier: _TitleObservation, later: _TitleObservation) -> bool:
    """Whether two label-shaped titles count on from one another ("Step 4",
    "STEP 5"): a numbered sequence, not a heading and its running head. A head
    for "Step 4" would read "STEP 4 5", never "STEP 5"."""
    return (
        earlier.shaped
        and later.shaped
        and earlier.number is not None
        and later.number == earlier.number + 1
    )


def _opener_form(opener: _TitleObservation, others: list[_TitleObservation]) -> bool:
    """Whether ``opener``'s surface form marks it as unlike every one of
    ``others``: a shared form is a running head, and so is a form that shares
    the case-exact text and differs only by the number's side when that side
    is the outer edge of the opener's printed page. Against a single other
    member a side-only difference is no evidence at all (outer-edge heads
    alternate sides), so a run of two needs the case-exact text to differ; a
    run of one has nothing to differ from."""
    side, text = opener.form
    for other in others:
        if other.form == opener.form:
            return False
        if other.form[1] == text and (len(others) < 2 or _on_outer_edge(side, opener.printed)):
            return False
    return bool(others)


def _fold_ocr_title(text: str) -> str:
    return " ".join(_NON_LETTER_RE.sub(" ", text.casefold()).split())


def _ocr_variants(first: str, second: str, matcher: SequenceMatcher[str] | None = None) -> bool:
    """Whether two folded titles are OCR spellings of one text (see above).

    ``matcher``, when given, already holds ``first`` as its second sequence
    (difflib caches that side), so a caller comparing many titles with one
    representative reuses it.
    """
    first_words, second_words = first.split(), second.split()
    if len(first_words) != len(second_words):
        return False
    shorter, longer = sorted((len(first), len(second)))
    if 2 * shorter < OCR_HEAD_SIMILARITY * (shorter + longer):
        return False  # ratio() <= 2*shorter/(shorter+longer): cannot reach it
    if matcher is None:
        matcher = SequenceMatcher(None, second, first)
    else:
        matcher.set_seq1(second)
    if (
        matcher.real_quick_ratio() < OCR_HEAD_SIMILARITY
        or matcher.quick_ratio() < OCR_HEAD_SIMILARITY
        or matcher.ratio() < OCR_HEAD_SIMILARITY
    ):
        return False
    return all(
        a == b or SequenceMatcher(None, a, b).ratio() >= OCR_HEAD_WORD_SIMILARITY
        for a, b in zip(first_words, second_words, strict=True)
    )


#: OCR confusions, each folded to one spelling: "field" and "held", "rnodern"
#: and "modern", "clrill" and "drill" read alike once folded.
_OCR_CONFUSIONS: tuple[tuple[str, str], ...] = (
    ("fi", "h"),
    ("rn", "m"),
    ("cl", "d"),
    ("1", "l"),
    ("i", "l"),
    ("0", "o"),
)


def _ocr_fold(word: str) -> str:
    for seen, meant in _OCR_CONFUSIONS:
        word = word.replace(seen, meant)
    return word


def _ocr_evidence(spellings: list[str]) -> tuple[bool, bool]:
    """``(garbled, substituted)`` for the spellings of a family seen in one run.

    ``garbled``: some spelling differs from the first in a word that is not a
    real word ("euents"): the OCR evidence. ``substituted``: some spelling
    differs in a real word that is an OCR confusion of the other ("field" /
    "held": ``_OCR_CONFUSIONS``). Two real words that differ otherwise are two
    headings ("Hurdle Drills" / "Hurdle Skills"), and a trailing
    "s" ("Coaching Points"/"Coaching Point") is neither: real headings recur
    in both numbers.
    """
    representative = spellings[0].split()
    garbled = substituted = False
    for spelling in spellings[1:]:
        for a, b in zip(representative, spelling.split(), strict=True):
            if a == b or a.rstrip("s") == b.rstrip("s"):
                continue
            if not (is_word(a) and is_word(b)):
                garbled = True
            elif _ocr_fold(a) == _ocr_fold(b):
                substituted = True
    return garbled, substituted


def _ocr_variant_running_heads(
    titles: list[tuple[ParsedElement, str]], edge_ids: set[str]
) -> set[str]:
    """``element_id`` of page-edge TITLEs in an OCR-variant running-head family."""
    members: defaultdict[str, list[ParsedElement]] = defaultdict(list)
    for element, text in titles:
        if element.element_id not in edge_ids or any(ch.isdigit() for ch in text):
            continue
        folded = _fold_ocr_title(text)
        if len(folded) >= OCR_HEAD_MIN_CHARS:
            members[folded].append(element)
    # Most frequent spelling first: it is the family's representative.
    by_words: defaultdict[int, list[tuple[list[str], SequenceMatcher[str]]]] = defaultdict(list)
    families: list[list[str]] = []
    quick_budget, full_budget = OCR_HEAD_MAX_QUICK_COMPARISONS, OCR_HEAD_MAX_COMPARISONS
    for folded in sorted(members, key=lambda key: (-len(members[key]), key)):
        bucket = by_words[len(folded.split())]
        joined = False
        for family, matcher in bucket:
            if quick_budget <= 0 or full_budget <= 0:
                break
            shorter, longer = sorted((len(family[0]), len(folded)))
            if 2 * shorter < OCR_HEAD_SIMILARITY * (shorter + longer):
                continue
            quick_budget -= 1
            matcher.set_seq1(folded)
            if matcher.quick_ratio() < OCR_HEAD_SIMILARITY:
                continue
            full_budget -= 1
            if _ocr_variants(family[0], folded, matcher):
                family.append(folded)
                joined = True
                break
        if not joined:
            family = [folded]
            families.append(family)
            bucket.append((family, SequenceMatcher(None, "", folded)))
    running: set[str] = set()
    for family in families:
        if len(family) < 2:
            continue
        observations = sorted(
            (element for folded in family for element in members[folded]),
            key=lambda element: element.page_number or 0,
        )
        runs: list[list[ParsedElement]] = []
        for element in observations:
            page = element.page_number or 0
            if runs and page - (runs[-1][-1].page_number or 0) <= OCR_HEAD_RUN_MAX_PAGE_GAP:
                runs[-1].append(element)
            else:
                runs.append([element])
        representative = {id(element) for element in members[family[0]]}
        for run in runs:
            run_pages = {element.page_number for element in run}
            if len(run_pages) < RUNNING_TITLE_MIN_PAGES:
                continue
            # The evidence is the spellings in this run, not in the family: a
            # garble on p. 80 says nothing about the real headings on pp. 10-16.
            run_ids = {id(element) for element in run}
            garbled, substituted = _ocr_evidence(
                [f for f in family if any(id(e) in run_ids for e in members[f])]
            )
            known_head = (
                substituted
                and len({e.page_number for e in run if id(e) in representative})
                >= RUNNING_TITLE_MIN_PAGES
            )
            if not (garbled or known_head):
                continue
            # Without a garbled word the run's first page may be the real
            # heading (the exact-repeat rule exempts it too).
            opener_page = run[0].page_number
            running.update(
                element.element_id
                for element in run
                if element.page_number != 1 and (garbled or element.page_number != opener_page)
            )
    return running


# Book-level running heads (VOR-1454): a head carrying the book's or a part's
# title ("STRENGTHCRAFT" on dozens of pages of a book of that name) recurs at
# page edges all through the book. The exact-repeat rule above keeps each run's first page as
# its possible opener, and a head that skips pages starts a new run every few
# pages, so such a head was kept as a depth-1 heading dozens of times and
# parented every mixed-case heading of the book. A title text is a book-level
# head when its page-edge occurrences span BOOK_HEAD_MIN_PAGES+ pages in
# BOOK_HEAD_MIN_RUNS+ runs and nearly all its occurrences sit at a page edge (a
# real heading repeated per chapter, "Summary", mostly sits mid-page). Every
# page-edge occurrence but one on the title page is skipped, openers included,
# and so are page-edge OCR spellings of it ("STRENGTHCRAFC").
BOOK_HEAD_MIN_PAGES = 6
BOOK_HEAD_MIN_RUNS = 3
BOOK_HEAD_EDGE_SHARE = 0.8
BOOK_HEAD_VARIANT_SIMILARITY = 0.75
BOOK_HEAD_MAX_LENGTH_DIFFERENCE = 3


def _book_head_variant(head: str, folded: str) -> bool:
    """An OCR spelling of a book-level head, word breaks aside: the garble
    splits and joins words ("STRENGTHCRAfC" for "STRENGTHCRAFT")."""
    if _ocr_variants(head, folded):
        return True
    first, second = head.replace(" ", ""), folded.replace(" ", "")
    if abs(len(first) - len(second)) > BOOK_HEAD_MAX_LENGTH_DIFFERENCE:
        return False
    return SequenceMatcher(None, first, second).ratio() >= BOOK_HEAD_VARIANT_SIMILARITY


def _book_running_heads(titles: list[tuple[ParsedElement, str]], edge_ids: set[str]) -> set[str]:
    """``element_id`` of page-edge TITLEs that are a book-level running head."""
    members: defaultdict[str, list[ParsedElement]] = defaultdict(list)
    for element, text in titles:
        if any(ch.isdigit() for ch in text) or is_structural_heading(text):
            continue
        if is_back_matter_heading(text):
            continue
        folded = _fold_ocr_title(text)
        if len(folded) >= OCR_HEAD_MIN_CHARS:
            members[folded].append(element)
    heads: list[str] = []
    for folded, elements in members.items():
        edge = [element for element in elements if element.element_id in edge_ids]
        if len(edge) < BOOK_HEAD_EDGE_SHARE * len(elements):
            continue
        pages = sorted({element.page_number or 0 for element in edge})
        if len(pages) < BOOK_HEAD_MIN_PAGES:
            continue
        gaps = zip(pages, pages[1:], strict=False)
        runs = 1 + sum(1 for a, b in gaps if b - a > RUN_MAX_PAGE_GAP)
        if runs >= BOOK_HEAD_MIN_RUNS:
            heads.append(folded)
    if not heads:
        return set()
    running: set[str] = set()
    for folded, elements in members.items():
        if folded not in heads and not any(_book_head_variant(head, folded) for head in heads):
            continue
        running.update(
            element.element_id
            for element in elements
            if element.element_id in edge_ids and element.page_number != 1
        )
    return running


def _running_head_titles(
    elements: list[ParsedElement], repeated_short_text: frozenset[str]
) -> frozenset[str]:
    titles = [
        (element, normalize_extracted_text(element.text))
        for element in elements
        if element.element_type is ElementType.TITLE and element.page_number is not None
    ]
    if not titles:
        return frozenset()
    first_ids, last_ids = _page_edge_element_ids(elements)
    edge_ids = first_ids | last_ids

    offsets: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    exact_pages: defaultdict[str, set[int]] = defaultdict(set)
    first_pages: defaultdict[str, set[int]] = defaultdict(set)  # exact text opens the page
    for element, text in titles:
        page = element.page_number or 0
        exact_pages[_title_key(text)].add(page)
        if element.element_id in first_ids:
            first_pages[_title_key(text)].add(page)
        key, number, _ = _split_edge_number(text)
        if number is not None:
            offsets[key].append((number - page, page))
    dominant = {key: _key_offset(observed) for key, observed in offsets.items()}
    label_families = _enumerated_label_families(
        [(element.page_number or 0, order, text) for order, (element, text) in enumerate(titles)]
    )
    # The book's print offset anchors: every (page, offset) at which a title
    # demonstrably carries a page number, i.e. follows its key's dominant
    # offset. A label ("Step 3") is not a page, and a key with no dominant
    # offset of its own ("Revenue | 2023", a caption with a year) never
    # anchors, so stray captions cannot pool into an offset of their own.
    anchors = sorted(
        (page, offset)
        for key, observed in offsets.items()
        if key not in label_families and (key_offset := dominant[key]) is not None
        for offset, page in observed
        if abs(offset - key_offset) <= PAGE_OFFSET_DRIFT
    )
    run_sizes = _anchor_run_sizes(anchors)

    page_numbered: defaultdict[str, list[_TitleObservation]] = defaultdict(list)
    exact: defaultdict[str, list[_TitleObservation]] = defaultdict(list)
    for order, (element, text) in enumerate(titles):
        page = element.page_number or 0
        key, number, piped = _split_edge_number(text)
        offset = dominant.get(key)
        label_match = _ENUMERATED_LABEL_RE.match(" ".join(text.split()))
        exact_key = _title_key(text)
        normalized = SectionPacker.normalize_line_for_signal(text).lower()
        tracks = paged = False
        printed = page
        if number is not None and offset is not None:
            tracks = abs(number - page - offset) <= PAGE_OFFSET_DRIFT
            if tracks:
                printed = number  # a page-number head prints its own number
        elif number is not None:
            region = _nearest_offset(page, anchors, run_sizes)
            paged = abs(number - page - region) <= PAGE_OFFSET_DRIFT
            printed = page + region
        observation = _TitleObservation(
            page=page,
            order=order,
            element_id=element.element_id,
            number=number,
            piped=piped,
            tracks=tracks,
            paged=paged,
            label=label_match is not None and label_match.group(1).casefold() in label_families,
            shaped=label_match is not None,
            repeats=(
                len(exact_pages[exact_key]) >= RUNNING_TITLE_MIN_PAGES
                and (
                    (bool(normalized) and normalized in repeated_short_text)
                    # An unpiped head with a constant number ("CHAPTER 3") that
                    # opens RUNNING_TITLE_MIN_PAGES+ consecutive pages. The exact
                    # text carries the number, so it is constant by definition.
                    or (
                        number is not None
                        and not piped
                        and _longest_consecutive(first_pages[exact_key]) >= RUNNING_TITLE_MIN_PAGES
                    )
                )
            ),
            form=_surface_form(text),
            printed=printed,
        )
        page_numbered[_run_key(text)].append(observation)
        exact[exact_key].append(observation)

    running: set[str] = set()
    for observations in page_numbered.values():
        for run in _split_runs(observations):
            # Labels are kept regardless, so they neither count as running heads
            # nor stand in as the run's real opener.
            has_real_opener = any(not o.tracks and not o.label for o in run)
            opener = run[0]
            # Chapter 1 on printed page 1, or an opener whose surface form no
            # other member of the run shares ("2 Methods" beside "3 METHODS";
            # see _opener_form for what a run of two or one can show, and for
            # the outer-edge side rule).
            opener_exempt = (
                opener.tracks
                and not has_real_opener
                and not opener.piped
                and (opener.number == 1 or _opener_form(opener, run[1:]))
            )
            for observation in run:
                if observation.element_id not in edge_ids or observation.label:
                    continue
                if observation.tracks:
                    if observation is opener and opener_exempt:
                        continue
                    running.add(observation.element_id)
                elif not observation.paged:
                    continue
                elif observation.piped:
                    if any(o.number is None and o.position < observation.position for o in run):
                        running.add(observation.element_id)  # short section, real heading before
                elif anchors and any(
                    o.position < observation.position
                    and not o.label
                    and o.form != observation.form
                    and not _numbered_sequence(o, observation)
                    for o in run
                ):
                    # Short chapter, unpiped: "METHODS 5" after "4 Methods" repeats
                    # an earlier member of its run in another surface form ("STEP
                    # 5" after "Step 4" counts on from it instead). Only with
                    # page-number evidence: with no anchors the region offset is
                    # a fallback, not a finding.
                    running.add(observation.element_id)
    for observations in exact.values():
        for run in _split_runs(observations):
            opener_page = run[0].page
            for observation in run:
                # Page 1 is the title page; a run's first page may be its opener.
                if (
                    observation.repeats
                    and observation.page not in (1, opener_page)
                    and observation.element_id in edge_ids
                ):
                    running.add(observation.element_id)
    running |= _ocr_variant_running_heads(titles, edge_ids)
    running |= _book_running_heads(titles, edge_ids)
    return frozenset(running)


#: A chapter's number on a line of its own: "4 --", "12 \u2014".
_CHAPTER_NUMBER_LINE_RE = re.compile(r"^\s*\d{1,2}\s*[\u2014\u2013]\s*$")

#: A line opened by a bullet glyph: a list item, whatever the parser typed.
_BULLET_LINE_RE = re.compile(r"^\s*[■□▪▫●○◦•‣►▶➤✓✔◆◇·]\s*\S")

#: Element types whose text is body text for ``body_input_texts``.
_BODY_INPUT_TYPES: frozenset[ElementType] = frozenset(
    {ElementType.NARRATIVE_TEXT, ElementType.LIST_ITEM}
)
#: Routes by which the chunker skips an element on purpose (page furniture).
_SKIPPED_ROUTES: frozenset[ElementRoute] = frozenset(
    {
        ElementRoute.SKIP_HEADER_FOOTER,
        ElementRoute.SKIP_REPEATED,
        ElementRoute.SKIP_QUARANTINED,
    }
)
#: ``is_degenerate_heading`` reasons that mean the TITLE carries no text worth
#: keeping (a page number, a folio, a watermark): only these are discarded.
_NOISE_TITLE_REASONS: frozenset[str] = frozenset(
    {
        "empty",
        "numeric_only",
        "roman_only",
        "no_word_chars",
        "too_short",
        "page_marker_only",
        "watermark_or_drm",
        "publisher_slogan",
        "library_stamp",
        "bare_label",
        "masthead",
    }
)


def _junk_title(element: ParsedElement, text: str, context: RoutingContext) -> bool:
    """A TITLE whose text is not a heading of this book (VOR-1454): a title- or
    copyright-page line, a clipped piece of another title, or garbled OCR.
    Rejected, it stays text of the section it sits in."""
    if is_front_matter_line(text, element.page_number, context.front_pages):
        return True
    if element.element_id in context.clipped_titles:
        return True
    if element.element_id in context.table_cell_titles:
        return True
    first_alpha = next((ch for ch in text if ch.isalpha()), "")
    if first_alpha.islower() and not context.lower_case_headings and not is_section_numbered(text):
        return True
    # A section-numbered heading is structure; its terms may be the book's
    # own coinage, used nowhere else ("9.2.1 BP for polytrees").
    return (
        context.vocabulary is not None
        and not is_labelled_heading(text)
        and not is_section_numbered(text)
        and _garbled(text, context.vocabulary)
    )


def route_element(
    element: ParsedElement,
    element_text: str,
    context: RoutingContext,
) -> ElementRoute:
    """Classify ``element`` (its text already ``normalize_extracted_text``-ed)."""
    element_type = element.element_type
    if element_type in {ElementType.HEADER, ElementType.FOOTER}:
        return ElementRoute.SKIP_HEADER_FOOTER
    normalized = SectionPacker.normalize_line_for_signal(element_text).lower()
    if element_type is ElementType.TITLE:
        if element.element_id in context.running_head_titles:
            return ElementRoute.SKIP_REPEATED
    elif (
        normalized
        and normalized in context.repeated_short_text
        and (context.page_edge_ids is None or element.element_id in context.page_edge_ids)
    ):
        # Repeated short text at a page edge is a running head or footer; the
        # same words mid-page are the book's text.
        return ElementRoute.SKIP_REPEATED
    if element_type is ElementType.TITLE:
        if is_degenerate_heading(element_text) is None and not _junk_title(
            element, element_text, context
        ):
            return ElementRoute.UPDATE_TITLE
        return ElementRoute.TITLE_REJECTED
    if element.page_number in context.quarantined:
        return ElementRoute.SKIP_QUARANTINED
    if element_type is ElementType.TABLE:
        return ElementRoute.TABLE
    if element_type is ElementType.LIST_ITEM:
        return ElementRoute.LIST_ITEM
    if element_type in APPENDED_TEXT_ELEMENT_TYPES:
        if element_type is ElementType.NARRATIVE_TEXT and SectionPacker.is_standalone_heading(
            element_text
        ):
            return ElementRoute.PENDING_HEADING
        return ElementRoute.BODY
    return ElementRoute.IGNORED


#: Page-reference lists an index table holds ("126, 128, 130", "39-41, 18").
_PAGE_REF_LISTS_RE = re.compile(r"(?:\b\d{1,4}(?:[-\u2013]\d{1,4})?,\s+){2,}\d{1,4}\b")
_PUBLICATION_DETAILS_RE = re.compile(
    r"(?:^|\n)\s*(?:copyright\b|library of congress catalog|"
    r"british library catalog)|catalog(?:u)?ing[- ]in[- ]publication",
    re.IGNORECASE,
)


def _label_unheaded_back_matter(
    chunks: list[RetrievalChunk],
    doc_title: str,
    count: Callable[[str], int],
    *,
    index_pages: frozenset[int] = frozenset(),
) -> None:
    """Label an index or a reference list the parser gave no heading (VOR-1454).

    Its entries otherwise took the last heading before them ("Chapter 30 >
    Summary" over index lines, "15.6 Summary" over a chapter's references). A
    text chunk that reads as index entries is filed under "Index"; one that
    reads as a reference list replaces its leaf with "References" (the
    chapter it closes is kept). Strong index evidence can replace an inherited
    back-matter heading. Publication details retain their actual title section.
    """
    title = doc_title or "Document"
    publication_paths = {
        chunk.section_path
        for chunk in chunks
        if chunk.chunk_type == "text" and _PUBLICATION_DETAILS_RE.search(chunk.content)
    }
    for chunk in chunks:
        if chunk.chunk_type not in {"text", "table"}:
            continue  # an index laid out in columns may parse as a table
        # One bracket each side: a heading may end in its own "[74]".
        inner = chunk.section_path[1:-1] if chunk.section_path.startswith("[") else ""
        components = inner.split(" > ")[1:]
        if components and is_index_heading(components[-1]):
            continue
        source_pages = ShortTextMerger.chunk_pages(chunk)
        if (source_pages and source_pages <= index_pages) or (
            reads_as_index(chunk.content, strict=chunk.chunk_type != "text")
            and (chunk.chunk_type == "text" or _PAGE_REF_LISTS_RE.search(chunk.content))
        ):
            components = ["Index"]
        elif components and is_back_matter_heading(components[-1]):
            continue
        elif (
            chunk.chunk_type == "text"
            and chunk.section_path not in publication_paths
            and reads_as_reference_list(chunk.content)
        ):
            components = [*components[:-1], "References"]
        else:
            continue
        old_prefix = f"{chunk.section_path}\n\n"
        body = (
            chunk.embedding_text[len(old_prefix) :]
            if chunk.embedding_text.startswith(old_prefix)
            else chunk.content
        )
        chunk.section_path = f"[{title} > {' > '.join(components)}]"
        chunk.section_title = components[-1]
        chunk.chapter_path = derive_chapter_path(chunk.section_path, title)
        chunk.embedding_text = f"{chunk.section_path}\n\n{body}"
        chunk.token_count = count(chunk.embedding_text)
        chunk.metadata["front_matter_kind"] = "index" if components == ["Index"] else "bibliography"


def _numeric_folio_element_ids(doc: ParsedDocument) -> frozenset[str]:
    """Locate plausible numeric folios only at actual parsed page edges."""
    by_page: dict[int, list[ParsedElement]] = defaultdict(list)
    for element in doc.elements:
        if (
            element.page_number is not None
            and element.element_type is not ElementType.IMAGE
            and element.text.strip()
        ):
            by_page[element.page_number].append(element)
    found: set[str] = set()
    for page, elements in by_page.items():
        for position in {0, len(elements) - 1}:
            element = elements[position]
            if element.element_type is ElementType.CODE_BLOCK or not element.text.strip().isdigit():
                continue
            neighbors = (
                elements[max(0, position - 1) : position] + elements[position + 1 : position + 2]
            )
            if any(SectionPacker.has_mathematical_value(e.text) for e in neighbors):
                continue
            if not strip_page_furniture_text(element.text, [page]):
                found.add(element.element_id)
    return frozenset(found)


def _preserve_numeric_element(element: ParsedElement, folios: frozenset[str]) -> bool:
    """Code numbers and non-folio scalar elements remain source data."""
    return element.element_type is ElementType.CODE_BLOCK or (
        element.text.strip().isdigit() and element.element_id not in folios
    )


class HierarchicalChunker:
    def __init__(
        self,
        *,
        target: int = 500,
        max_: int = 600,
        min_: int = 100,
        overlap: int = 50,
        counter: TokenCounterFn | None = None,
        table_linearizer: object | Callable[[ParsedElement], str] | None = None,
        prose_splitter: ProseSplitter | None = None,
    ) -> None:
        self.target = target
        self.max = max_
        self.min = min_
        self.overlap = overlap
        self.count = counter or self._default_counter()
        self.linearize_table = table_linearizer or TableLinearizer(
            max_tokens=max_,
            counter=self.count,
        )
        self.packer = SectionPacker(
            counter=self.count,
            target=target,
            max_tokens=max_,
            overlap=overlap,
            table_linearizer=self.linearize_table,
            prose_splitter=prose_splitter,
        )
        self.emitter = CompositeEmitter(
            counter=self.count,
            max_tokens=max_,
            overlap=overlap,
            table_linearizer=self.linearize_table,
        )
        self.short_text = ShortTextMerger(counter=self.count)
        # ``min_`` is the floor below which a text/table chunk is merged into a
        # same-section neighbour (VOR-1452); 0 disables the merge.
        self.small_chunks = SmallChunkMerger(
            counter=self.count,
            min_tokens=min_,
            target=target,
            max_tokens=max_,
        )
        self.token_ceiling = TokenCeilingBackstop(
            counter=self.count,
            max_tokens=max_,
            overlap=overlap,
        )
        self.titleonly_fallback = TitleOnlyFallback(counter=self.count)
        self.fallback_min_titleonly_chunks = self.titleonly_fallback.fallback_min_titleonly_chunks
        self.fallback_section_pages = self.titleonly_fallback.fallback_section_pages
        self.fallback_chapter_pages = self.titleonly_fallback.fallback_chapter_pages
        self.last_page_drop_ledger: list[PageDrop] = []
        self.last_drop_summary: dict[DropReason, int] = {}

    @staticmethod
    def _default_counter() -> TokenCounterFn:
        mode = os.environ.get("VORAG_CHUNK_TOKEN_COUNTER", "naive").strip().lower()
        if mode == "tokenizer":
            try:
                return default_token_counter()
            except Exception:
                if os.environ.get("VORAG_CHUNK_TOKEN_COUNTER_FAIL_SOFT") == "1":
                    log.warning("vo_rag.chunking.token_counter_fallback", mode=mode)
                    return naive_counter
                raise
        return naive_counter

    def chunk(
        self,
        doc: ParsedDocument,
        composites: list[CompositeChunk],
    ) -> list[RetrievalChunk]:
        self.last_page_drop_ledger = []
        self.last_drop_summary = {}
        # Page order first: a legacy parse lists its tables after
        # the book's last text, so each would take the last heading of the book.
        # ``new_position`` re-keys composites to where their element now sits.
        doc, new_position = in_reading_order(doc, REFLOWABLE_PARSERS)
        routing = RoutingContext.for_document(doc)
        generic = generic_subheadings(doc, routing)
        reflowable = doc.parser_used in REFLOWABLE_PARSERS
        repeated_lines = _repeated_lines(doc.elements)
        openers = frozenset() if reflowable else _page_opening_titles(doc.elements, routing)
        rare_capitals = not reflowable and _rare_capitals(doc.elements, routing)
        table_heads = _table_titles(doc.elements)
        quarantined = routing.quarantined
        pages_with_text_elements = self.packer.pages_with_text_elements(doc.elements)
        drop_ledger: list[PageDrop] = []
        drop_counts: Counter[DropReason] = Counter()
        running_lines = _page_edge_running_lines(doc, routing, self.packer)
        numeric_folios = _numeric_folio_element_ids(doc)
        index_pages = frozenset(
            e.page_number
            for e in doc.elements
            if e.element_type in {ElementType.HEADER, ElementType.TITLE}
            and e.page_number is not None
            and is_index_heading(e.text)
            and not any(character.isdigit() for character in e.text)
        )
        # Emission stays keyed on the composite's own position (clamped, so an
        # out-of-range index still emits at the end rather than vanishing); the
        # label comes from its page-order anchor (FigurePlacement), so relabelling
        # a figure never moves a text chunk boundary (VOR-1452).
        by_index: dict[int, list[CompositeChunk]] = defaultdict(list)
        by_anchor: dict[int, list[CompositeChunk]] = defaultdict(list)
        placement = (
            FigurePlacement(doc.elements) if doc.parser_used not in REFLOWABLE_PARSERS else None
        )
        for composite in composites:
            position = new_position[min(max(composite.element_index, 0), len(doc.elements))]
            by_index[position].append(composite)
            anchor = (
                placement.anchor(position, composite.page_number)
                if placement is not None
                else position
            )
            if anchor != position:
                by_anchor[anchor].append(composite)
        # Figure label per composite: taken where it sits in page order.
        figure_labels: dict[int, tuple[str, str, str | None]] = {}

        chunks: list[RetrievalChunk] = []
        tracker = HeadingTracker(
            doc_title=doc.title,
            front_pages=routing.front_pages,
            inline_caps=not reflowable and _inline_capitals(doc.elements, routing, openers),
        )
        current_parts: list[str] = []
        current_pages: list[int] = []
        current_section_path = tracker.section_path()
        current_section_title = tracker.section_title()
        # Heading dropped by the tracker's staleness guard for the current text.
        current_stale_heading: str | None = None
        # Section epoch (see ChunkScope): bumped at every accepted depth-1/2
        # heading and every tracker path change; ``current_epoch`` is the one
        # the gathered text belongs to.
        section_epoch = 0
        current_epoch = 0
        chunk_scopes: dict[int, ChunkScope] = {}
        # One serial per accepted heading (ChunkScope.headings): the open path
        # as (depth, text, serial), so sibling sections are told apart even when
        # their texts repeat ("Week 1 > Monday" vs "Week 2 > Monday").
        heading_serial = 0
        heading_serials: dict[int, int] = {}
        current_headings: tuple[tuple[int, str, int], ...] = ()

        def open_headings() -> tuple[tuple[int, str, int], ...]:
            nonlocal heading_serial
            for depth in tracker.headings:
                if depth not in heading_serials:
                    heading_serial += 1
                    heading_serials[depth] = heading_serial
            return tuple(
                (depth, tracker.headings[depth], heading_serials[depth])
                for depth in sorted(tracker.headings)
            )

        next_overlap_prefix = ""
        pending_heading_parts: list[str] = []
        pending_heading_pages: list[int | None] = []
        # Headings replaced before any text came under them: their
        # words lead the next text instead of vanishing ("Cycle 4: Four Weeks"
        # straight before "Week 1"). ``last_title`` is the latest accepted one.
        orphan_parts: list[str] = []
        orphan_pages: list[int] = []
        # Accepted headings with no text under them yet: (depth, text, page).
        titles_since_body: list[tuple[int, str, int | None]] = []
        pending_composites: list[CompositeChunk] = []
        # Tables met mid-sentence, emitted after the text around them (as figures
        # are) but labelled where they sit.
        pending_tables: list[tuple[ParsedElement, _TablePosition]] = []

        def record_drop(page_number: int | None, reason: DropReason) -> None:
            if page_number is None:
                return
            drop_ledger.append(PageDrop(page_number=page_number, reason=reason))
            drop_counts[reason] += 1
            log.info(
                "book_rag.chunk.page_dropped",
                source_id=doc.source_id,
                page=page_number,
                reason=reason,
            )

        def record_drop_for_pages(pages: list[int], reason: DropReason) -> None:
            for page in sorted(set(pages)):
                record_drop(page, reason)

        def queue_short_text_candidate(content: str, pages: list[int]) -> None:
            content = content.strip()
            if not content:
                record_drop_for_pages(pages, "empty_after_normalize")
                return
            metadata: dict[str, object] = {}
            fm_kind = section_front_matter_kind(current_section_path)
            if fm_kind:
                metadata["front_matter_kind"] = fm_kind
            if current_stale_heading:
                metadata["stale_heading"] = current_stale_heading
            # Emitted in place, flagged as short text: SmallChunkMerger folds it
            # into a same-section neighbour or keeps it -- never drops it
            # (VOR-1452; the old page-based ShortTextMerger path dropped short
            # text whose page another chunk already covered).
            unique_pages = sorted(set(pages))
            embedding_text = f"{current_section_path}\n\n{content}"
            chunks.append(
                RetrievalChunk(
                    chunk_id=_chunk_id(doc.source_id, content, "text"),
                    chunk_type="text",
                    document_id=doc.document_id,
                    source_id=doc.source_id,
                    page_number=unique_pages[0] if unique_pages else None,
                    page_range=self.short_text.page_range(unique_pages),
                    section_path=current_section_path,
                    section_title=current_section_title,
                    chapter_path=derive_chapter_path(current_section_path, doc.title),
                    content=content,
                    embedding_text=embedding_text,
                    token_count=self.count(embedding_text),
                    metadata=metadata,
                )
            )
            chunk_scopes[id(chunks[-1])] = ChunkScope(
                epoch=current_epoch,
                headings=current_headings,
                short_text=True,
            )

        def emit_pending_composites() -> None:
            while pending_composites:
                emit_composite(pending_composites.pop(0))
            while pending_tables:
                emit_table(*pending_tables.pop(0))

        def flush_text() -> None:
            nonlocal current_parts, current_pages, next_overlap_prefix
            nonlocal current_section_path, current_section_title

            if not current_parts:
                emit_pending_composites()
                return

            body = "\n\n".join(part for part in current_parts if part.strip()).strip()
            body = self.packer.clean_chunk_body(body, current_pages, running_lines)
            if not body:
                record_drop_for_pages(current_pages, "empty_after_normalize")
                current_parts = []
                current_pages = []
                emit_pending_composites()
                return
            low_signal_reason = self.packer.low_signal_text_reason(
                body,
                pages=current_pages,
                doc_title=doc.title,
                doc_total_pages=doc.total_pages,
            )
            if low_signal_reason is not None:
                if low_signal_reason == "below_min_words":
                    queue_short_text_candidate(body, current_pages)
                else:
                    record_drop_for_pages(current_pages, low_signal_reason)
                current_parts = []
                current_pages = []
                next_overlap_prefix = ""
                emit_pending_composites()
                return

            content = body
            prefix = ""
            if next_overlap_prefix:
                prefix = self.packer.clean_chunk_body(
                    next_overlap_prefix,
                    current_pages,
                    running_lines,
                )
                content = f"{prefix}\n\n{body}" if prefix else body
            content = self.packer.clean_chunk_body(content, current_pages, running_lines)
            if not content:
                record_drop_for_pages(current_pages, "empty_after_normalize")
                current_parts = []
                current_pages = []
                next_overlap_prefix = ""
                emit_pending_composites()
                return
            low_signal_reason = self.packer.low_signal_text_reason(
                content,
                pages=current_pages,
                doc_title=doc.title,
                doc_total_pages=doc.total_pages,
            )
            if low_signal_reason is not None:
                if low_signal_reason == "below_min_words":
                    # ``body``: the overlap prefix is a copy of the previous chunk.
                    queue_short_text_candidate(body, current_pages)
                else:
                    record_drop_for_pages(current_pages, low_signal_reason)
                current_parts = []
                current_pages = []
                next_overlap_prefix = ""
                emit_pending_composites()
                return

            page_number = current_pages[0] if current_pages else None
            page_range = self.short_text.page_range(current_pages)
            fm_kind = section_front_matter_kind(current_section_path)
            chunk_metadata: dict[str, object] = {"front_matter_kind": fm_kind} if fm_kind else {}
            if current_stale_heading:
                chunk_metadata["stale_heading"] = current_stale_heading
            for part_index, part in enumerate(
                self.packer.split_text(content, current_section_path)
            ):
                embedding_text = f"{current_section_path}\n\n{part}"
                chunks.append(
                    RetrievalChunk(
                        chunk_id=_chunk_id(doc.source_id, part, "text"),
                        chunk_type="text",
                        document_id=doc.document_id,
                        source_id=doc.source_id,
                        page_number=page_number,
                        page_range=page_range,
                        section_path=current_section_path,
                        section_title=current_section_title,
                        chapter_path=derive_chapter_path(current_section_path, doc.title),
                        content=part,
                        embedding_text=embedding_text,
                        token_count=self.count(embedding_text),
                        metadata=dict(chunk_metadata),
                    )
                )
                chunk_scopes[id(chunks[-1])] = ChunkScope(
                    epoch=current_epoch,
                    overlap_prefix=prefix if part_index == 0 else "",
                    headings=current_headings,
                )
            next_overlap_prefix = self.packer.tail_for_overlap(chunks[-1].content)
            current_parts = []
            current_pages = []
            emit_pending_composites()

        def take_pending_headings() -> tuple[list[str], list[int]]:
            heading_parts = self.packer.filtered_pending_headings(pending_heading_parts)
            kept = set(heading_parts)
            heading_pages = [
                page
                for raw, page in zip(pending_heading_parts, pending_heading_pages, strict=True)
                if page is not None and self.packer.rendered_pending_heading(raw) in kept
            ]
            pending_heading_parts.clear()
            pending_heading_pages.clear()
            return heading_parts, heading_pages

        def observe_body(text: str, page_number: int | None, *, entry: bool = False) -> None:
            """Let the tracker expire a terminal back-matter heading or a stale
            heading (VOR-1452). On a change, text gathered so far -- including
            queued pending headings such as title-case index entries -- is
            emitted under the old path before the new path applies."""
            nonlocal next_overlap_prefix, section_epoch, current_epoch, current_headings
            nonlocal current_section_path, current_section_title, current_stale_heading
            before = (
                (tracker.section_path(), tracker.section_title(), tracker.stale_heading)
                if pending_heading_parts
                else None
            )
            before_epoch = section_epoch
            before_headings = open_headings()
            if not tracker.observe_body(text, page_number, entry=entry):
                return
            section_epoch += 1
            # The tracker restored or cleared its path: those are new sections.
            heading_serials.clear()
            flush_text()
            if before is not None:
                heading_parts, heading_pages = take_pending_headings()
                if heading_parts:
                    current_section_path, current_section_title, current_stale_heading = before
                    current_epoch = before_epoch
                    current_headings = before_headings
                    current_parts.extend(heading_parts)
                    current_pages.extend(heading_pages)
                    flush_text()
            next_overlap_prefix = ""

        def append_text(text: str, page_number: int | None) -> None:
            nonlocal current_section_path, current_section_title, current_stale_heading
            nonlocal current_epoch, current_headings
            text = self.packer.clean_text_for_append(
                text, page_number, running_lines, preserve_numeric_data=True
            )
            if not text.strip():
                record_drop(page_number, "empty_after_normalize")
                return
            titles_since_body.clear()
            if orphan_parts:
                text = "\n".join([*orphan_parts, text])
                if page_number is None and orphan_pages:
                    page_number = orphan_pages[0]
                orphan_parts.clear()
                orphan_pages.clear()

            if not current_parts:
                heading_parts, heading_pages = take_pending_headings()
                path_headings = (
                    [
                        part
                        for part in heading_parts
                        if _reads_as_words(part)
                        and not is_front_matter_line(part, None)
                        and _line_key(part) not in repeated_lines
                    ]
                    if len(heading_parts) <= MAX_PENDING_HEADING_RUN
                    and not _label_run(heading_parts)
                    else []
                )
                current_section_path = tracker.section_path(path_headings)
                current_section_title = tracker.section_title(path_headings)
                current_stale_heading = tracker.stale_heading
                current_epoch = section_epoch
                current_headings = open_headings()
                current_parts.extend(heading_parts)
                # Kept pending headings are body text too; keep their pages so a
                # heading-led flush is never emitted without a page number.
                current_pages.extend(heading_pages)

            candidate = "\n\n".join(current_parts + [text]) if current_parts else text
            if (
                current_parts
                and self.packer.embedding_token_count(current_section_path, candidate) > self.max
            ):
                current_body = "\n\n".join(current_parts)
                candidate_tokens = self.packer.embedding_token_count(
                    current_section_path,
                    candidate,
                )
                if (
                    self.packer.ends_at_retrieval_boundary(current_body)
                    or candidate_tokens > self.packer.hard_token_ceiling()
                ):
                    flush_text()

            current_parts.append(text)
            if page_number is not None:
                current_pages.append(page_number)

            maybe = "\n\n".join(current_parts)
            maybe_tokens = self.count(maybe)
            if maybe_tokens >= self.target and (
                self.packer.ends_at_retrieval_boundary(maybe)
                or maybe_tokens >= self.packer.hard_token_ceiling()
            ):
                flush_text()

        def in_index_region() -> bool:
            """A live index heading (the terminal-heading rule decides its end)."""
            terminal = tracker.terminal_heading
            return terminal is not None and is_index_heading(terminal)

        def in_reference_region() -> bool:
            """A live entry list other than an index ("References", "Notes")."""
            terminal = tracker.terminal_heading
            return (
                terminal is not None
                and is_entry_list_heading(terminal)
                and not is_index_heading(terminal)
            )

        def label_figure(composite: CompositeChunk) -> None:
            """Record the label text at this position would get (VOR-1452). A
            figure beside a reference list is the chapter's."""
            outside = tracker.outside_terminal() if in_reference_region() else None
            if outside is not None:
                figure_labels.setdefault(id(composite), (*outside, tracker.stale_heading))
            figure_labels.setdefault(
                id(composite), tracker.figure_attribution(composite.page_number)
            )

        def emit_composite(composite: CompositeChunk) -> None:
            nonlocal next_overlap_prefix
            if composite.page_number in quarantined:
                record_drop(composite.page_number, "qa_quarantined")
                return
            # The section the chunker was in where the figure sits, as for the
            # text around it (terminal back matter and staleness included); a
            # pending figure keeps the label of its position, not of the flush.
            section_path, section_title, stale_heading = figure_labels.get(
                id(composite), tracker.figure_attribution(composite.page_number)
            )
            chunk = self.emitter.emit_composite(
                composite,
                document_id=doc.document_id,
                source_id=doc.source_id,
                section_path=section_path,
                section_title=section_title or composite.section_title,
                doc_title=doc.title,
            )
            if stale_heading:
                chunk.metadata["stale_heading"] = stale_heading
            chunks.append(chunk)
            next_overlap_prefix = ""

        def table_position(text: str = "") -> _TablePosition:
            """The section a table takes where it sits. A data
            table laid out after a reference list is the chapter's, not the
            list's: under a live reference-list heading, a table that does not
            read as citations takes the path from before that heading."""
            outside = tracker.outside_terminal() if in_reference_region() else None
            if outside is not None and not reads_as_citations(text):
                path, title = outside
                return _TablePosition(
                    section_path=path, section_title=title, epoch=section_epoch, headings=()
                )
            return _TablePosition(
                section_path=tracker.section_path(),
                section_title=tracker.section_title(),
                epoch=section_epoch,
                headings=open_headings(),
            )

        def emit_table(element: ParsedElement, position: _TablePosition | None = None) -> None:
            nonlocal next_overlap_prefix
            titles_since_body.clear()
            where = position or table_position(element.text)
            table_chunks = self.emitter.emit_table(
                element,
                document_id=doc.document_id,
                source_id=doc.source_id,
                section_path=where.section_path,
                section_title=where.section_title,
                doc_title=doc.title,
            )
            if not table_chunks:
                record_drop(element.page_number, "empty_after_normalize")
                return
            visual = element.metadata.get("table_visual")
            if (
                isinstance(visual, dict)
                and visual.get("status") == "source_bound"
                and isinstance(visual.get("image_ref"), str)
            ):
                # Original rows and mathematical typesetting accompany every
                # text part, including tables split at the token ceiling.
                for table_chunk in table_chunks:
                    table_chunk.has_image = True
                    table_chunk.image_id = "tbl-" + str(visual["sha256"])
                    table_chunk.image_ref = str(visual["image_ref"])
                    table_chunk.metadata.update(
                        table_visual=dict(visual),
                        table_source_element_id=element.element_id,
                        figure_kind="table",
                        symbolic_transcription_verified=False,
                    )
            chunks.extend(table_chunks)
            for table_chunk in table_chunks:
                chunk_scopes[id(table_chunk)] = ChunkScope(
                    epoch=where.epoch,
                    headings=where.headings,
                )
            next_overlap_prefix = ""

        # Contents pages: each run of them is
        # one "Contents" section; on them no element opens a section.
        contents = contents_pages(doc)
        # A heading on a contents page followed by prose opens its own section
        # for the rest of that page ("PREFACE" on a booklet's contents page).
        contents_openers = contents_section_openers(doc, contents)
        closed_contents_pages: set[int] = set()
        contents_where: _TablePosition | None = None
        in_contents = False

        def set_contents(opening: bool, page: int | None) -> None:
            nonlocal in_contents, next_overlap_prefix, section_epoch, contents_where
            # A contents run opens or ends: a new section either way.
            flush_text()
            next_overlap_prefix = ""
            if opening:
                tracker.open_contents(page)
            else:
                tracker.close_contents()
            in_contents = opening
            section_epoch += 1
            heading_serials.clear()
            if opening:
                contents_where = table_position()

        # A chapter's number set on a line of its own ("4 --"), waiting for
        # the chapter's title (VOR-1454).
        after_chapter_number = False
        index = 0
        while index < len(doc.elements):
            page_now = doc.elements[index].page_number
            if (
                page_now is not None
                and (page_now in contents and page_now not in closed_contents_pages) != in_contents
            ):
                set_contents(not in_contents, page_now)
            if in_contents and (
                doc.elements[index].element_id in contents_openers
                or is_body_paragraph(doc.elements[index])
            ):
                if page_now is not None:
                    closed_contents_pages.add(page_now)
                set_contents(False, page_now)
            for composite in by_anchor.get(index, []):
                label_figure(composite)
            for composite in by_index.get(index, []):
                label_figure(composite)
                current_body = "\n\n".join(current_parts)
                if (
                    current_parts
                    and not self.packer.ends_at_retrieval_boundary(current_body)
                    and self.count(current_body) < self.packer.hard_token_ceiling()
                ):
                    pending_composites.append(composite)
                else:
                    flush_text()
                    emit_composite(composite)

            element = doc.elements[index]
            element_text = normalize_extracted_text(element.text)
            route = route_element(element, element_text, routing)
            if (
                not in_contents
                and is_body_paragraph(element)
                and any(
                    front_matter_kind(heading) == "toc" for heading in tracker.headings.values()
                )
            ):
                # A foreword or prologue after a Contents heading is not contents.
                flush_text()
                next_overlap_prefix = ""
                if tracker.close_contents_heading():
                    section_epoch += 1
                    heading_serials.clear()

            equation_visual = element.metadata.get("equation_visual")
            if (
                element.element_type is ElementType.EQUATION
                and element.page_number not in quarantined
                and isinstance(equation_visual, dict)
                and equation_visual.get("status") == "source_bound"
                and isinstance(equation_visual.get("image_ref"), str)
            ):
                # A flattened text layer cannot express every mathematical
                # relationship. The original crop is the retrievable primary
                # representation; original characters remain searchable.
                flush_text()
                observe_body(element_text, element.page_number)
                path = tracker.section_path()
                title = tracker.section_title()
                reference = str(equation_visual["image_ref"])
                original_text = str(element.metadata.get("docling_orig") or element_text)
                for part in self.packer.split_text(original_text, path):
                    embedded = f"{path}\n\n{part}"
                    chunks.append(
                        RetrievalChunk(
                            chunk_id=_chunk_id(
                                doc.source_id,
                                element.element_id + reference + part,
                                "figure_composite",
                            ),
                            chunk_type="figure_composite",
                            document_id=doc.document_id,
                            source_id=doc.source_id,
                            page_number=element.page_number,
                            page_range=str(element.page_number) if element.page_number else None,
                            section_path=path,
                            section_title=title,
                            chapter_path=derive_chapter_path(path, doc.title),
                            content=part,
                            embedding_text=embedded,
                            token_count=self.count(embedded),
                            has_image=True,
                            image_id="eq-" + str(equation_visual["sha256"]),
                            image_ref=reference,
                            metadata={
                                "equation_visual": dict(equation_visual),
                                "equation_source_element_id": element.element_id,
                                "figure_kind": "equation",
                                "symbolic_transcription_verified": False,
                            },
                        )
                    )
                next_overlap_prefix = ""
                index += 1
                continue

            if route in {ElementRoute.SKIP_HEADER_FOOTER, ElementRoute.SKIP_REPEATED}:
                if (
                    element.page_number in index_pages
                    and is_index_heading(element_text)
                    and not any(character.isdigit() for character in element_text)
                    and not in_index_region()
                ):
                    # A genuine index page header restores the source section
                    # before short entries can merge into the preceding answers.
                    flush_text()
                    if tracker.update_title(element_text, depth=1, page=element.page_number):
                        heading_serials.clear()
                        section_epoch += 1
                    next_overlap_prefix = ""
                # A running head that repeats the open heading keeps it fresh.
                tracker.reconfirm(element_text, element.page_number)
                index += 1
                continue

            if in_contents and route in {
                ElementRoute.UPDATE_TITLE,
                ElementRoute.TITLE_REJECTED,
                ElementRoute.PENDING_HEADING,
            }:
                route = ElementRoute.BODY  # a contents line, never a section

            if (
                route is ElementRoute.TITLE_REJECTED
                and element.page_number not in quarantined
                and is_degenerate_heading(element_text) not in _NOISE_TITLE_REASONS
            ):
                # Not a heading, but text ("2.50 4x 400", an equation, an OCR'd
                # line): body text of the open section, judged like any other
                # (a rejected TITLE must not vanish).
                route = ElementRoute.BODY

            if (
                route is ElementRoute.TITLE_REJECTED
                and element.page_number not in quarantined
                and (
                    self.packer.has_mathematical_value(element_text)
                    or (
                        element_text.strip().isdigit()
                        and _preserve_numeric_element(element, numeric_folios)
                    )
                )
            ):
                # A short formula or interior scalar can be rejected as a
                # heading but remains actual source data. Evidenced folios
                # still pass through the normal page-furniture removal.
                route = ElementRoute.BODY

            if (
                route is ElementRoute.BODY
                and element.element_type is ElementType.TITLE
                and len(element_text.split()) >= 5
                and element_text.split()[-1].lower() in {"in", "of", "for", "to", "with"}
                and index + 1 < len(doc.elements)
            ):
                following = doc.elements[index + 1]
                if (
                    following.element_type is ElementType.TITLE
                    and following.page_number == element.page_number
                    and route_element(following, normalize_extracted_text(following.text), routing)
                    is ElementRoute.UPDATE_TITLE
                ):
                    # A split printed title belongs to the next title's body,
                    # rather than the section that ended before this page.
                    flush_text()
                    next_overlap_prefix = ""
                    pending_heading_parts.append(element_text.strip())
                    pending_heading_pages.append(element.page_number)
                    index += 1
                    continue

            if route is ElementRoute.UPDATE_TITLE:
                words = element_text.split()
                numbered = bool(words and words[0].isdigit())
                numbered_value = (
                    numbered
                    and len(words) > 2
                    and re.fullmatch(r"\d+\.\d+", words[1])
                    and words[2][:1].islower()
                )
                numbered_reference = (
                    numbered and len(words) >= 4 and words[-2] == "page" and words[-1].isdigit()
                )
                if (
                    numbered_value and not is_structural_heading(element_text)
                ) or numbered_reference:
                    # A numbered answer or page reference typed as TITLE is
                    # source data. Keep it in the body before title transitions.
                    route = ElementRoute.BODY

            if route is ElementRoute.UPDATE_TITLE and _BULLET_LINE_RE.match(element_text):
                # A bulleted line the parser typed as TITLE ("■ to raise the body
                # temperature") is a list item, never a heading.
                route = ElementRoute.BODY

            if route is ElementRoute.UPDATE_TITLE and is_lead_in_line(element_text):
                # A sentence lead-in ("The constants are:") the parser typed as
                # TITLE: body text of the open section.
                route = ElementRoute.BODY

            if route is ElementRoute.UPDATE_TITLE and (
                (
                    (in_index_region() or element.page_number in index_pages)
                    and is_index_entry_title(element_text)
                )
                or (in_reference_region() and is_entry_fragment_title(element_text))
            ):
                # An index entry the parser typed as TITLE ("Ankle Hops"): a body
                # line of the index, never a section of its own (VOR-1452). The
                # tracker's terminal rule still decides where the index ends.
                entry_text = self.packer.clean_text_for_append(
                    element_text, element.page_number, running_lines
                )
                if entry_text.strip():
                    observe_body(entry_text, element.page_number, entry=True)
                    append_text(entry_text, element.page_number)
                index += 1
                continue

            if route in {ElementRoute.UPDATE_TITLE, ElementRoute.TITLE_REJECTED}:
                if route is ElementRoute.TITLE_REJECTED and _CHAPTER_NUMBER_LINE_RE.match(
                    element_text
                ):
                    after_chapter_number = True
                    index += 1
                    continue
                depth = max(1, element.section_depth or 1)
                if not reflowable:
                    depth = _case_depth(element_text, depth)
                    if rare_capitals and depth == 1 and not is_structural_heading(element_text):
                        # A lone capitalised heading ("DOMS", "RICE") among
                        # mixed-case ones is an acronym, not a chapter (VOR-1454).
                        depth = 2
                if route is ElementRoute.UPDATE_TITLE and generic:
                    depth = _generic_depth(element_text, depth, tracker.headings, generic)
                if self.packer.should_flush_for_title(depth):
                    flush_text()
                    next_overlap_prefix = ""
                if route is ElementRoute.UPDATE_TITLE and pending_heading_parts:
                    # Standalone lines before a heading ("CIP AC", "CHAPTER 5") lead
                    # the text under it; they never sit below it in its path.
                    for part, part_page in zip(
                        pending_heading_parts, pending_heading_pages, strict=True
                    ):
                        orphan_parts.append(" ".join(part.split()))
                        if part_page is not None:
                            orphan_pages.append(part_page)
                    pending_heading_parts.clear()
                    pending_heading_pages.clear()
                if route is ElementRoute.UPDATE_TITLE and tracker.update_title(
                    element_text,
                    depth=depth,
                    page=element.page_number,
                    opens_page=reflowable or element.element_id in openers,
                    heads_table=element.element_id in table_heads,
                    chapter=after_chapter_number,
                ):
                    for stale_depth in [key for key in heading_serials if key >= depth]:
                        del heading_serials[stale_depth]
                    if depth <= 2:
                        section_epoch += 1
                    for held in list(titles_since_body):
                        if tracker.headings.get(held[0]) != held[1]:
                            # Replaced before any text came under it: keep its words.
                            titles_since_body.remove(held)
                            orphan_parts.append(" ".join(held[1].split()))
                            if held[2] is not None:
                                orphan_pages.append(held[2])
                    if depth > 3 and current_parts:
                        # A deep heading does not close the text gathered so far,
                        # so the chunk keeps the path it started with: the heading
                        # is a run-in line of it, not lost.
                        current_parts.append(" ".join(element_text.split()))
                        if element.page_number is not None:
                            current_pages.append(element.page_number)
                    else:
                        titles_since_body.append((depth, element_text, element.page_number))
                # A depth-3 heading still closes the gathered text, but that
                # boundary is soft: SmallChunkMerger may rejoin a small leaf with
                # its neighbour inside the same depth-2 section (VOR-1452).
                if depth == 3:
                    flush_text()
                if route is ElementRoute.UPDATE_TITLE:
                    after_chapter_number = False
                index += 1
                continue

            if route is ElementRoute.SKIP_QUARANTINED:
                if element.element_type in TEXT_BEARING_ELEMENT_TYPES:
                    record_drop(element.page_number, "qa_quarantined")
                index += 1
                continue

            if (
                route is ElementRoute.TABLE
                and element.page_number in contents
                and contents_where is not None
                and is_entry_table(element)
            ):
                # The contents table of a page that also holds another section.
                flush_text()
                emit_table(element, contents_where)
                index += 1
                continue

            if route is ElementRoute.TABLE:
                observe_body(element_text, element.page_number)
                current_body = "\n\n".join(current_parts)
                if (
                    current_parts
                    and not self.packer.ends_at_retrieval_boundary(current_body)
                    and self.count(current_body) < self.packer.hard_token_ceiling()
                ):
                    # Mid-sentence (a sentence running on past the table, often
                    # across a page break): the text is not cut here.
                    pending_tables.append((element, table_position(element.text)))
                else:
                    flush_text()
                    emit_table(element)
                index += 1
                continue

            if route is ElementRoute.LIST_ITEM:
                run: list[str] = []
                run_pages: list[int] = []
                next_index = index
                while (
                    next_index < len(doc.elements)
                    and doc.elements[next_index].element_type is ElementType.LIST_ITEM
                ):
                    row = doc.elements[next_index]
                    if row.page_number not in quarantined:
                        row_text = self.packer.clean_text_for_append(
                            row.text,
                            row.page_number,
                            running_lines,
                            preserve_numeric_data=_preserve_numeric_element(row, numeric_folios),
                        )
                        if row_text.strip():
                            # List rows are entry-shaped: they keep back matter alive.
                            observe_body(row_text, row.page_number, entry=True)
                            run.append(row_text)
                            if row.page_number is not None:
                                run_pages.append(row.page_number)
                        else:
                            record_drop(row.page_number, "empty_after_normalize")
                    else:
                        record_drop(row.page_number, "qa_quarantined")
                    next_index += 1
                if run:
                    append_text("\n".join(run), run_pages[0] if run_pages else None)
                    for page in run_pages[1:]:
                        current_pages.append(page)
                index = next_index
                continue

            if route in {ElementRoute.PENDING_HEADING, ElementRoute.BODY}:
                cleaned_element_text = self.packer.clean_text_for_append(
                    element_text,
                    element.page_number,
                    running_lines,
                    preserve_numeric_data=_preserve_numeric_element(element, numeric_folios),
                )
                if not cleaned_element_text.strip():
                    record_drop(element.page_number, "empty_after_normalize")
                    index += 1
                    continue
                observe_body(cleaned_element_text, element.page_number)
                if route is ElementRoute.PENDING_HEADING and (
                    in_index_region() or in_reference_region()
                ):
                    # Inside an index a title-case line ("Ankle Hops") is an
                    # entry, never a heading for what follows (VOR-1452): it is
                    # index body text, subject to the index filters like the rest.
                    route = ElementRoute.BODY
                if route is ElementRoute.PENDING_HEADING:
                    flush_text()
                    next_overlap_prefix = ""
                    pending_heading_parts.append(element_text.strip())
                    pending_heading_pages.append(element.page_number)
                    index += 1
                    continue
                low_signal_reason = (
                    self.packer.low_signal_element_text_reason(
                        cleaned_element_text,
                        pages=[element.page_number] if element.page_number is not None else None,
                        doc_title=doc.title,
                        doc_total_pages=doc.total_pages,
                    )
                    if element.element_type is ElementType.NARRATIVE_TEXT
                    else None
                )
                if low_signal_reason is not None:
                    current_body = "\n\n".join(current_parts)
                    if not current_parts or self.packer.ends_at_retrieval_boundary(current_body):
                        if low_signal_reason == "below_min_words":
                            append_text(cleaned_element_text, element.page_number)
                        else:
                            record_drop(element.page_number, low_signal_reason)
                        index += 1
                        continue
                append_text(cleaned_element_text, element.page_number)

            index += 1

        for composite in by_anchor.get(len(doc.elements), []):
            label_figure(composite)
        for composite in by_index.get(len(doc.elements), []):
            label_figure(composite)
            flush_text()
            emit_composite(composite)

        if orphan_parts:
            # Headings with no text after them at the end of the book.
            leftover, leftover_page = "\n".join(orphan_parts), orphan_pages[:1]
            orphan_parts.clear()
            orphan_pages.clear()
            append_text(leftover, leftover_page[0] if leftover_page else None)
        flush_text()
        chunks = self.small_chunks.merge(chunks, chunk_scopes)
        _label_unheaded_back_matter(chunks, doc.title, self.count, index_pages=index_pages)
        chunks = self._page_section_titleonly_fallback(chunks, doc)
        covered_pages = self.short_text.covered_pages(chunks)
        for page_number in range(1, doc.total_pages + 1):
            if page_number in quarantined:
                continue
            if page_number not in pages_with_text_elements and page_number not in covered_pages:
                record_drop(page_number, "no_text_extracted")
        self.last_page_drop_ledger = list(drop_ledger)
        self.last_drop_summary = {reason: count for reason, count in drop_counts.items() if count}
        log.info(
            "book_rag.chunk.drop_summary",
            source_id=doc.source_id,
            dropped_by_reason=self.last_drop_summary,
        )
        chunks = self.token_ceiling.enforce(chunks)
        mathematical_pages = {
            element.page_number: visual
            for element in doc.elements
            if isinstance((visual := element.metadata.get("source_page_visual")), dict)
            and visual.get("status") == "source_bound"
            and isinstance(visual.get("image_ref"), str)
        }
        original_tables = {
            element.element_id: visual
            for element in doc.elements
            if element.element_type is ElementType.TABLE
            and isinstance((visual := element.metadata.get("table_visual")), dict)
            and visual.get("status") == "source_bound"
            and isinstance(visual.get("image_ref"), str)
        }
        for chunk in chunks:
            table_ids = [
                chunk.metadata.get("element_id"),
                chunk.metadata.get("table_source_element_id"),
            ]
            merged_ids = chunk.metadata.get("merged_table_element_ids")
            if isinstance(merged_ids, list):
                table_ids.extend(merged_ids)
            source_tables = [
                {"source_element_id": identity, **original_tables[identity]}
                for identity in dict.fromkeys(
                    identity for identity in table_ids if isinstance(identity, str)
                )
                if identity in original_tables
            ]
            if source_tables:
                chunk.metadata["source_table_visuals"] = source_tables
                primary_table = next(
                    (v for v in source_tables if v.get("physical_page") == chunk.page_number), None
                )
                if not chunk.has_image and primary_table is not None:
                    chunk.has_image = True
                    chunk.image_id = "tbl-" + str(primary_table["sha256"])
                    chunk.image_ref = str(primary_table["image_ref"])
            originals = [
                dict(mathematical_pages[page])
                for page in sorted(self._chunk_pages(chunk))
                if page in mathematical_pages
            ]
            if originals:
                chunk.metadata["source_page_visuals"] = originals
                primary_page = next(
                    (v for v in originals if v.get("physical_page") == chunk.page_number), None
                )
                if not chunk.has_image and primary_page is not None:
                    chunk.has_image = True
                    chunk.image_id = "pg-" + str(primary_page["sha256"])
                    chunk.image_ref = str(primary_page["image_ref"])
        return chunks

    def body_input_texts(self, doc: ParsedDocument) -> list[str]:
        """The body text ``chunk`` is given to keep, element by element.

        Narrative text and list items, cleaned exactly as ``chunk`` cleans them,
        leaving out what the chunker skips on purpose as page furniture: page
        headers and footers, running heads and repeated short page text, page
        numbers, running lines and distribution watermarks stripped from the
        text, and quarantined pages.
        ``rechunk-preview`` measures ``input_cov`` against these: every sentence
        in them must reach a chunk (the OCR gate's drops aside).
        """
        doc, _ = in_reading_order(doc, REFLOWABLE_PARSERS)
        routing = RoutingContext.for_document(doc)
        running_lines = _page_edge_running_lines(doc, routing, self.packer)
        numeric_folios = _numeric_folio_element_ids(doc)
        texts: list[str] = []
        for element in doc.elements:
            if element.element_type not in _BODY_INPUT_TYPES:
                continue
            text = normalize_extracted_text(element.text)
            if route_element(element, text, routing) in _SKIPPED_ROUTES:
                continue
            cleaned = self.packer.clean_text_for_append(
                text,
                element.page_number,
                running_lines,
                preserve_numeric_data=_preserve_numeric_element(element, numeric_folios),
            )
            # The chunk body is cleaned once more (``flush_text``): mirror it,
            # watermark lines and all, so input and output compare like for like.
            pages = [element.page_number] if element.page_number is not None else []
            cleaned = self.packer.clean_chunk_body(cleaned, pages, running_lines)
            if cleaned.strip():
                texts.append(cleaned)
        return texts

    def _page_section_titleonly_fallback(
        self,
        chunks: list[RetrievalChunk],
        doc: ParsedDocument,
    ) -> list[RetrievalChunk]:
        return self.titleonly_fallback.apply(chunks, doc)

    @staticmethod
    def _covered_pages(chunks: list[RetrievalChunk]) -> set[int]:
        return ShortTextMerger.covered_pages(chunks)

    @staticmethod
    def _chunk_pages(chunk: RetrievalChunk) -> set[int]:
        return ShortTextMerger.chunk_pages(chunk)

    @staticmethod
    def _page_range(pages: list[int]) -> str | None:
        return ShortTextMerger.page_range(pages)

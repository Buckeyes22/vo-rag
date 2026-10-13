from __future__ import annotations

import os
import re
from collections.abc import Callable

from vo_rag.core.chunking.chapter_path import derive_chapter_path
from vo_rag.core.chunking.front_matter import front_matter_kind
from vo_rag.core.chunking.heading_filter import (
    clean_heading_text,
    has_trailing_folio,
    is_back_matter_heading,
    is_chapter_heading,
    is_degenerate_heading,
    is_entry_list_heading,
    is_front_page_imprint,
    is_imprint_heading,
    is_part_heading,
    is_section_numbered,
    is_structural_heading,
    looks_like_back_matter_entry,
)
from vo_rag.core.chunking.models import RetrievalChunk
from vo_rag.core.chunking.section_packer import SectionPacker
from vo_rag.core.parsing.contract import ParsedDocument

#: ``ParsedDocument.parser_used`` values of reflowable ebooks (``EPUB_FORMAT`` /
#: ``MOBI_FORMAT`` in ``core/parsing/parsers/epub.py``), whose page numbers are
#: synthetic chapter indexes rather than print pages.
REFLOWABLE_PARSERS: frozenset[str] = frozenset({"epub", "mobi"})

#: Staleness guard (VOR-1452): a heading that nothing has reconfirmed for more
#: than this many pages while body text keeps coming is no longer trusted, and
#: the body falls back to the document title (``stale_heading`` in metadata).
#: "Reconfirmed" means a new accepted TITLE at any depth, or a skipped running
#: head / page header that repeats a held heading. 25 pages is longer than a
#: typical nonfiction chapter with no subheading at all (textbook chapters run
#: 10-30 pages and are subdivided far more often than every 25), and running
#: heads refresh the longer ones; a heading that silently spans 25+ pages of a
#: scanned book is almost always a missed next title -- the audited
#: "Bibliography"/"INDEX" labels rode onto hundreds of pages that way.
STALE_HEADING_PAGES = 25

#: Pages an ordinary (non-structural) heading may parent headings over without a
#: running head repeating it (VOR-1454, ``HeadingTracker._detached``). Longer than
#: a section and its subsections run in these books; a chapter set in capitals
#: that runs longer is repeated by its running heads.
PARENT_SPAN_PAGES = 20

#: The section label of a run of contents pages (``contents_pages``).
CONTENTS_LABEL = "Contents"

_KEY_DROP_RE = re.compile(r"[\W\d_]+")


#: Front/back-matter kinds whose heading never parents a later heading. "About
#: the author" is left out: a contributor's name heading belongs under it.
_NON_PARENT_KINDS: frozenset[str] = frozenset(
    {"toc", "copyright", "acknowledgements", "list_of_symbols", "index", "bibliography", "glossary"}
)


def _is_matter_heading(heading: str) -> bool:
    """A front/back-matter or imprint heading ("References", "Contents",
    "Copyright © 2003 by ..."): never restored when back matter expires
    (a dedication would otherwise be filed under the copyright line before it)."""
    lowered = " ".join(heading.split()).lower()
    return (
        is_back_matter_heading(heading)
        or front_matter_kind(heading) is not None
        or "©" in heading
        or lowered.startswith("copyright")
    )


#: Front-matter labels that head the title and copyright pages (VOR-1454):
#: "Title Page > Second Edition", "Front Matter > Editors" filed a book's
#: opening under its title page.
_FRONT_LABEL_RE = re.compile(
    r"^(?:title page|half[- ]title(?: page)?|front ?matter|dedication|epigraph|frontispiece"
    r"|(?:photo |picture |illustration )?credits"
    r"|series page|also by .{1,60}|praise for .{1,60}|copyright(?: page| notice)?|imprint)$"
)


_WEEKDAY_RE = re.compile(
    r"^(?:mon|tues|wednes|thurs|fri|satur|sun)day\b|^(?:mon|tue|wed|thu|fri|sat|sun)\.?(?:\W|$)",
    re.IGNORECASE,
)
_SERIES_LABEL_RE = re.compile(
    r"^(day|week|session|workout|phase|stage|step|month|lesson|period|level|cycle|microcycle"
    r"|mesocycle|variant|exercise|drill|circuit)\s*#?\s*(?:\d+|[ivx]+|[a-e]|one|two|three|four|five"
    r"|six|seven|eight|nine|ten)\b",
    re.IGNORECASE,
)


#: Series whose members head an outline (a step's drills), not a single item.
_OUTLINE_SERIES = frozenset(
    {"step", "phase", "stage", "lesson", "week", "day", "session", "level", "cycle", "period"}
)


def _series(heading: str) -> str | None:
    """The series a heading is a member of: "weekday" for "Wednesday:", "week"
    for "Week 2: Exploration" -- members of one series are siblings."""
    stripped = " ".join(heading.split()).lstrip("'\"(")
    if _WEEKDAY_RE.match(stripped):
        return "weekday"
    label = _SERIES_LABEL_RE.match(stripped)
    return label.group(1).lower() if label else None


#: Sections that open a book and sit inside nothing (VOR-1454): "Part III ...
#: > Preface" filed a preface under a contents line.
_OPENING_SECTION_RE = re.compile(
    r"^(?:preface|foreword|prologue|(?:publisher'?s?|author'?s?|editor'?s?|translator'?s?)"
    r" (?:note|preface|foreword))$"
)


def _capitalised(heading: str) -> bool:
    letters = [ch for ch in heading if ch.isalpha()]
    return len(letters) >= 4 and sum(ch.isupper() for ch in letters) >= 0.85 * len(letters)


def _never_parents(heading: str) -> bool:
    if front_matter_kind(heading) in _NON_PARENT_KINDS:
        return True
    normalized = " ".join(heading.lower().split()).strip(" .:;,")
    # An imprint line ("Human Kinetics") labels its own page, never a chapter.
    return bool(_FRONT_LABEL_RE.match(normalized)) or is_imprint_heading(heading)


def _heading_key(text: str) -> str:
    return " ".join(_KEY_DROP_RE.sub(" ", text.lower()).split())


class HeadingTracker:
    """The open heading path while the chunker walks a document.

    Terminal back-matter headings (VOR-1452, ``is_back_matter_heading``): a
    heading such as "References", "INDEX" or "About the contributors" labels its
    own content but is terminal -- once content that is not back matter arrives
    on a later page, it expires. The rule, applied by ``observe_body`` to every
    text-bearing element:

    - on the heading's own page everything keeps the label (a bio under "About
      the authors", the first citations);
    - on a later page, content that reads as entries (index lines "word, 12,
      45", citations, numbered notes -- ``looks_like_back_matter_entry`` -- or a
      list) keeps the label and moves the terminal page forward, as does a page
      header/running head equal to the heading;
    - any other content on a later page expires the heading: the tracker
      restores the heading path that was open just before the back-matter
      heading appeared (the chapter/section a "References" closed), or the
      document title when there was none. Restoring the pre-terminal path rather
      than the terminal heading's ancestors matters when the parser inferred
      levels unevenly: a leftover level-1 heading from chapters earlier would
      otherwise resurface. The chunker flushes on expiry so earlier entries keep
      their label;
    - a real heading nested under the terminal one on a later page (an all-caps
      "ABOUT THE CONTRIBUTORS" inferred at level 1, then a mixed-case chapter at
      level 2) likewise expires it first instead of nesting under it.

    This is conservative: a back-matter section that runs onto a later page as
    prose (a two-page acknowledgements) loses its label there and gets the
    ancestors/document title -- never a wrong label.
    """

    def __init__(self, *, doc_title: str, front_pages: int = 0, inline_caps: bool = False) -> None:
        self.doc_title = doc_title or "Document"
        #: Headings on pages ``<= front_pages`` (the title, half-title and
        #: copyright pages) never parent a heading on a later page (VOR-1454):
        #: a stamp or the book's title there, ranked depth 1, held every
        #: chapter of the book below it. 0 turns the rule off (reflowable
        #: ebooks, whose "pages" are chapters).
        self.front_pages = front_pages
        #: The book sets run-in item headings in capitals
        #: (``hierarchical._inline_capitals``): a capitalised heading parents
        #: only headings on its own page (VOR-1454).
        self.inline_caps = inline_caps
        self._front_headings: set[str] = set()
        #: Where each heading text was last opened: ``(page, opened its page or
        #: is structural, heads a table)`` (``_detached``).
        self._placed: dict[str, tuple[int | None, bool, bool]] = {}
        #: Last page a running head repeated each held heading on (``reconfirm``).
        self._reconfirmed: dict[str, int] = {}
        self._headings: dict[int, str] = {}
        self._terminal_depth: int | None = None
        self._terminal_page: int | None = None
        #: Heading path open just before the live terminal heading appeared.
        self._pre_terminal: dict[int, str] = {}
        self._confirmed_page: int | None = None
        #: The heading path open before a run of contents pages.
        self._before_contents: dict[int, str] = {}
        #: The deepest heading dropped by the staleness guard, until the next title.
        self.stale_heading: str | None = None
        #: An entry list ("References") a prose back-matter heading
        #: ("Acknowledgements") replaced at its own level: ``(depth, heading)``,
        #: and whether that heading's note text has been seen.
        self._interrupted_list: tuple[int, str] | None = None
        self._note_seen = False

    @property
    def headings(self) -> dict[int, str]:
        return self._headings

    @property
    def terminal_heading(self) -> str | None:
        """The live terminal back-matter heading, if any."""
        if self._terminal_depth is None:
            return None
        return self._headings.get(self._terminal_depth)

    def _clear_terminal(self) -> None:
        self._terminal_depth = None
        self._terminal_page = None
        self._pre_terminal = {}

    def _expire_terminal(self) -> None:
        if self._terminal_depth is not None:
            self._headings = {
                depth: heading
                for depth, heading in self._pre_terminal.items()
                if not _is_matter_heading(heading)
            }
        self._clear_terminal()

    def _terminal_is_later(self, page: int | None) -> bool:
        return (
            self._terminal_depth is not None
            and page is not None
            and self._terminal_page is not None
            and page > self._terminal_page
        )

    def update_title(
        self,
        text: str,
        *,
        depth: int,
        page: int | None = None,
        opens_page: bool = True,
        heads_table: bool = False,
        chapter: bool = False,
    ) -> bool:
        """Open ``text`` at ``depth``; False when it is no heading.

        ``opens_page``: nothing but headings comes before it on its page.
        ``heads_table``: the next content after it is a table -- it is the
        table's title. Both only matter for what the heading may parent
        (VOR-1454, ``_detached``). ``chapter``: a chapter's number line ("4
        --") came just before it, so it is the chapter's title.
        """
        if is_degenerate_heading(text) is not None:
            return False
        live_terminal = self._terminal_depth
        terminal = (is_back_matter_heading(text) or is_front_page_imprint(text, page)) and (
            live_terminal is not None or not self._reads_as_contents_line(text)
        )
        if (
            not terminal
            and live_terminal is not None
            and depth > live_terminal
            and self._terminal_is_later(page)
        ):
            self._expire_terminal()
            live_terminal = None
        inherited: dict[int, str] | None = None
        replaced = self._headings.get(live_terminal) if live_terminal is not None else None
        self._interrupted_list = None
        if (
            terminal
            and replaced is not None
            and live_terminal == depth
            and is_entry_list_heading(replaced)
            and not is_entry_list_heading(text)
        ):
            # "Acknowledgements" between two columns of references: its note is
            # its own, the references after it are still the list's.
            self._interrupted_list = (depth, replaced)
            self._note_seen = False
        if live_terminal is not None and depth <= live_terminal:
            # Replaced at its own level or above: that back-matter section is
            # closed. Another back-matter heading ("Titles in This Series" then
            # "INDEX") continues the back matter and keeps the path from before it.
            inherited = self._pre_terminal if terminal else None
            self._clear_terminal()
            live_terminal = None
        if terminal and live_terminal is None:
            self._pre_terminal = inherited if inherited is not None else dict(self._headings)
        self._headings[depth] = text
        for stale_depth in [key for key in list(self._headings.keys()) if key > depth]:
            del self._headings[stale_depth]
        # A front- or back-matter heading ("CONTENTS", "Acknowledgements",
        # "Index") never parents a later heading: a book whose
        # chapters sit one level below its contents heading filed every chapter
        # under "CONTENTS". The later heading closes it. Nor does a heading of
        # the book's first pages parent one on a later page (VOR-1454).
        front = page is not None and page <= self.front_pages
        if front:
            self._front_headings.add(text)
        for upper in [key for key in self._headings if key < depth]:
            held = self._headings[upper]
            if (
                _never_parents(held)
                or self._detached(held, text, page, opens_page, front)
                # A chapter ("2 — Predictive Testing of Athletes", "Chapter 5")
                # is never inside an ordinary heading: a table's title or a
                # drill set in capitals held the chapters after it (VOR-1454).
                # ... nor inside an appendix or another chapter: only a part holds one.
                or ((chapter or is_chapter_heading(text)) and not is_part_heading(held))
                # "Wednesday" then "Friday", "Week 1" then "Week 2": siblings,
                # whatever depth the parser gave them (VOR-1454).
                or (_series(held) is not None and _series(held) == _series(text))
                # "About the author" parents a contributor's name, never a
                # numbered section ("2.00 OVERVIEW OF EVENTS").
                or (
                    is_back_matter_heading(held)
                    and (
                        is_section_numbered(text)
                        # "ABOUT THE AUTHOR" then "Acknowledgements": siblings.
                        or is_back_matter_heading(text)
                        or front_matter_kind(text) is not None
                    )
                )
                # A preface or foreword opens the book, inside no part.
                or bool(_OPENING_SECTION_RE.match(" ".join(text.lower().split()).strip(" .:")))
            ):
                del self._headings[upper]
                if upper == self._terminal_depth and not terminal:
                    self._clear_terminal()
        if terminal:
            self._terminal_depth = depth if live_terminal is None else live_terminal
            self._terminal_page = page
        self._placed[text] = (page, opens_page or is_structural_heading(text), heads_table)
        if page is not None:
            self._reconfirmed[text] = page
        self._confirmed_page = page
        self.stale_heading = None
        return True

    def _detached(
        self, held: str, text: str, page: int | None, opens_page: bool, front: bool
    ) -> bool:
        """Whether the open heading ``held`` must not parent ``text`` (VOR-1454).

        - a heading of the front matter never parents one after it (a stamp or
          the book's title set in capitals held every chapter); a structural
          one ("Part I: Setting the Stage") still heads its part;
        - a table's title ("UNITS OF MOMENTUM", "MASS IN POUNDS" over a chart)
          never parents: the next heading is not in the table;
        - a heading set mid-page (not opening its page) never parents a heading
          that opens a later page or is a chapter ("4. Sprints"): the parser
          ranks lines by case alone, so a capitalised line inside a figure
          ("MATCHES TO COME.") or a drill name ("MULTISWINGS") outranked every
          mixed-case chapter title after it. A heading opening a page stays a
          chapter's peer, so it still parents its sections.
        """
        if page is None:
            return False
        if not front and held in self._front_headings and not is_structural_heading(held):
            return True
        placed = self._placed.get(held)
        if placed is None:
            return False
        held_page, held_opens, held_table = placed
        if held_table:
            return True
        if (
            self.inline_caps
            and held_page is not None
            and page > held_page
            and _capitalised(held)
            and not is_structural_heading(held)
            and not is_section_numbered(held)
            # "STEP 3: HAMMER TURNS" heads the drills of its step.
            and _series(held) not in _OUTLINE_SERIES
        ):
            return True
        # An ordinary heading's span as a parent: past PARENT_SPAN_PAGES with
        # no running head repeating it, a heading on a later page is no longer
        # inside it (a sidebar title "SIMPLIFIED NOTATION ..." held the rest of
        # its chapter, VOR-1454). A chapter or part keeps its span.
        seen = self._reconfirmed.get(held, held_page)
        if seen is not None and page - seen > PARENT_SPAN_PAGES and not is_structural_heading(held):
            return True
        return (
            not held_opens
            and held_page is not None
            and page > held_page
            and (opens_page or is_structural_heading(text))
        )

    def open_contents(self, page: int | None) -> None:
        """A run of contents pages starts: label it "Contents".

        The open path is kept aside and restored by ``close_contents``: the
        contents are a section of their own, never nested under the heading
        before them, nor a parent of the one after them.
        """
        self._before_contents = dict(self._headings)
        self._clear_terminal()
        self._headings = {1: CONTENTS_LABEL}
        self._confirmed_page = page
        self.stale_heading = None

    def close_contents_heading(self) -> bool:
        """Drop an open Contents heading: the prose after it (a foreword
        on the contents page) is not contents. True when one was open."""
        toc = [
            depth
            for depth, heading in self._headings.items()
            if front_matter_kind(heading) == "toc"
        ]
        for depth in toc:
            for deeper in [key for key in self._headings if key >= depth]:
                self._headings.pop(deeper, None)
        return bool(toc)

    def close_contents(self) -> None:
        """The contents run ended: restore the path open before it, less any
        Contents heading in it (the run was that section)."""
        self._headings = {
            depth: heading
            for depth, heading in self._before_contents.items()
            if front_matter_kind(heading) != "toc"
        }
        self._before_contents = {}

    def _reads_as_contents_line(self, text: str) -> bool:
        """A back-matter title that is a contents-page line, not the section.

        "Index 245" typed as TITLE on the contents page: the
        trailing folio or an open Contents heading marks it as a TOC line. Made
        terminal, it would turn the next real heading ("Preface") into an index
        entry and file the text after it under the contents page. Only a first
        back-matter heading is judged: while back matter is live, a running head
        "INDEX 999" still continues it.
        """
        if has_trailing_folio(text):
            return True
        return any(front_matter_kind(heading) == "toc" for heading in self._headings.values())

    def reconfirm(self, text: str, page: int | None) -> None:
        """A skipped page header / running head: refresh a heading it repeats."""
        if page is None or not self._headings:
            return
        key = _heading_key(text)
        if not key:
            return
        for depth, heading in self._headings.items():
            heading_key = _heading_key(heading)
            if len(heading_key) < 4 or f" {heading_key} " not in f" {key} ":
                continue
            self._confirmed_page = max(self._confirmed_page or page, page)
            self._reconfirmed[heading] = max(self._reconfirmed.get(heading) or page, page)
            if depth == self._terminal_depth and heading_key == key:
                self._terminal_page = max(self._terminal_page or page, page)

    def observe_body(self, text: str, page: int | None, *, entry: bool = False) -> bool:
        """Apply the terminal-heading and staleness rules to content on ``page``.

        ``entry`` marks content that is list-shaped by type (list items). Returns
        True when the open heading path changed, so the caller must flush the
        text gathered under the old path first.
        """
        if page is None:
            return False
        changed = False
        if self._interrupted_list is not None:
            if entry or looks_like_back_matter_entry(text):
                depth, heading = self._interrupted_list
                self._interrupted_list = None
                if self._note_seen and self._terminal_depth == depth:
                    # The list resumes after the note: its heading is back.
                    self._headings[depth] = heading
                    for deeper in [key for key in self._headings if key > depth]:
                        del self._headings[deeper]
                    self._terminal_page = page
                    self._confirmed_page = page
                    return True
            else:
                self._note_seen = True
        if self._terminal_depth is not None and self._terminal_page is None:
            self._terminal_page = page
        if self._terminal_is_later(page):
            if entry or looks_like_back_matter_entry(text):
                self._terminal_page = page
                self._confirmed_page = page
            else:
                self._expire_terminal()
                changed = True
        if self._headings:
            if self._confirmed_page is None:
                self._confirmed_page = page
            elif page - self._confirmed_page > STALE_HEADING_PAGES:
                self.stale_heading = self._headings[max(self._headings)]
                self._headings.clear()
                self._clear_terminal()
                changed = True
        return changed

    def outside_terminal(self) -> tuple[str, str] | None:
        """``(section_path, section_title)`` of the path open before the live
        terminal heading, or ``None`` when there is no terminal heading (a data
        table or figure laid out after a reference list is the chapter's)."""
        if self._terminal_depth is None:
            return None
        headings = {
            depth: heading
            for depth, heading in self._pre_terminal.items()
            if not is_back_matter_heading(heading)
        }
        return self._format_path(headings), self._format_title(headings)

    def figure_attribution(self, page: int | None) -> tuple[str, str, str | None]:
        """``(section_path, section_title, stale_heading)`` for a figure on ``page``.

        A figure carries no text, so it never moves the tracker; it gets the
        label the next body text on its page would get (VOR-1452). The terminal
        and staleness rules of ``observe_body`` are applied to a copy: a figure on
        a later page than a live back-matter heading takes the path that body
        text there restores, and a figure more than ``STALE_HEADING_PAGES`` past
        the last reconfirmation falls back to the document title. On the
        back-matter heading's own page the figure keeps its label, as text does.
        """
        headings = dict(self._headings)
        stale_heading = self.stale_heading
        if page is not None:
            if self._terminal_is_later(page):
                headings = {
                    depth: heading
                    for depth, heading in self._pre_terminal.items()
                    if not is_back_matter_heading(heading)
                }
            if (
                headings
                and self._confirmed_page is not None
                and page - self._confirmed_page > STALE_HEADING_PAGES
            ):
                stale_heading = headings[max(headings)]
                headings = {}
        return (
            self._format_path(headings),
            self._format_title(headings),
            stale_heading,
        )

    def section_title(self, pending_headings: list[str] | None = None) -> str:
        if pending_headings:
            for candidate in reversed(pending_headings):
                if is_degenerate_heading(candidate) is None and clean_heading_text(candidate):
                    return clean_heading_text(candidate)
        return self._format_title(self._headings)

    def _format_title(self, headings: dict[int, str]) -> str:
        for depth in sorted(headings.keys(), reverse=True):
            candidate = headings[depth]
            if is_degenerate_heading(candidate) is None and clean_heading_text(candidate):
                return clean_heading_text(candidate)
        return self.doc_title

    def section_path(self, pending_headings: list[str] | None = None) -> str:
        ordered = [
            self._headings[depth]
            for depth in sorted(self._headings.keys())
            if is_degenerate_heading(self._headings[depth]) is None
        ]
        if pending_headings and any(_capitalised(heading) for heading in pending_headings):
            # A standalone capitalised line ("WEIGHT LIFTING") ranks with the
            # capitalised headings: the open mixed-case ones ("Hurdler's
            # Stretch") are its siblings, not its parents (VOR-1454).
            while (
                ordered and not _capitalised(ordered[-1]) and not is_structural_heading(ordered[-1])
            ):
                ordered.pop()
        if pending_headings and not (
            ordered and (_never_parents(ordered[-1]) or is_back_matter_heading(ordered[-1]))
        ):
            # Standalone lines under a copyright, contents or back-matter heading
            # are its text (credits, contents lines, a series list), never
            # sections below it (VOR-1454).
            ordered.extend(
                heading for heading in pending_headings if is_degenerate_heading(heading) is None
            )
        return self._path_of(ordered)

    def _format_path(self, headings: dict[int, str]) -> str:
        return self._path_of(
            [
                headings[depth]
                for depth in sorted(headings.keys())
                if is_degenerate_heading(headings[depth]) is None
            ]
        )

    def _path_of(self, headings: list[str]) -> str:
        # Path components as shown (VOR-1454): OCR debris cut, empties dropped.
        ordered = [cleaned for heading in headings if (cleaned := clean_heading_text(heading))]
        if ordered:
            return f"[{self.doc_title} > {' > '.join(ordered)}]"
        return f"[{self.doc_title}]"

    def chapter_path(self, pending_headings: list[str] | None = None) -> str:
        return derive_chapter_path(self.section_path(pending_headings), self.doc_title)


class TitleOnlyFallback:
    def __init__(
        self,
        *,
        counter: Callable[[str], int],
        fallback_min_titleonly_chunks: int | None = None,
        fallback_section_pages: int | None = None,
        fallback_chapter_pages: int | None = None,
    ) -> None:
        self.count = counter
        self.fallback_min_titleonly_chunks = (
            fallback_min_titleonly_chunks
            if fallback_min_titleonly_chunks is not None
            else int(os.environ.get("VORAG_BOOK_RAG_FALLBACK_MIN_TITLEONLY_CHUNKS", "10"))
        )
        self.fallback_section_pages = (
            fallback_section_pages
            if fallback_section_pages is not None
            else int(os.environ.get("VORAG_BOOK_RAG_FALLBACK_SECTION_PAGES", "10"))
        )
        self.fallback_chapter_pages = (
            fallback_chapter_pages
            if fallback_chapter_pages is not None
            else int(os.environ.get("VORAG_BOOK_RAG_FALLBACK_CHAPTER_PAGES", "50"))
        )

    def apply(
        self,
        chunks: list[RetrievalChunk],
        doc: ParsedDocument,
    ) -> list[RetrievalChunk]:
        pages = {chunk.page_number for chunk in chunks if chunk.page_number is not None}
        if len(pages) <= 1 or doc.parser_used in REFLOWABLE_PARSERS:
            # Unpaginated source (EPUB/HTML): page buckets would stamp every chunk "pp. 1-10".
            # The EPUB parser's "pages" are chapters, not print pages, so they get none either.
            return chunks
        title = doc.title or "Document"
        title_only = f"[{title}]"
        candidates = [
            chunk
            for chunk in chunks
            if chunk.section_path == title_only and chunk.page_number is not None
        ]
        if len(candidates) <= self.fallback_min_titleonly_chunks:
            return chunks
        section_pages = max(1, self.fallback_section_pages)
        chapter_pages = max(section_pages, self.fallback_chapter_pages)
        for chunk in candidates:
            page = chunk.page_number or 1
            section_bucket = (page - 1) // section_pages
            chapter_bucket = (page - 1) // chapter_pages
            section_start = section_bucket * section_pages + 1
            chapter_start = chapter_bucket * chapter_pages + 1
            section_end = section_start + section_pages - 1
            chapter_end = chapter_start + chapter_pages - 1
            # Swap only the path prefix: a figure's embedding_text is a wider
            # window than its content (VOR-1452); for text the result is the
            # same as rebuilding from content.
            old_prefix = f"{chunk.section_path}\n\n"
            body = (
                chunk.embedding_text[len(old_prefix) :]
                if chunk.embedding_text.startswith(old_prefix)
                else chunk.content
            )
            chunk.section_path = f"[{title} > pp. {section_start}-{section_end}]"
            chunk.chapter_path = f"[{title} > pp. {chapter_start}-{chapter_end}]"
            chunk.section_title = f"pp. {section_start}-{section_start + section_pages - 1}"
            chunk.embedding_text = f"{chunk.section_path}\n\n{body}"
            chunk.token_count = self.count(chunk.embedding_text)
        return chunks


def filtered_pending_headings(
    packer: SectionPacker,
    headings: list[str],
) -> list[str]:
    return packer.filtered_pending_headings(headings)

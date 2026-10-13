"""VOR-1452: junk section labels on body text.

Audit of a track-and-field corpus: body text filed under "References",
"Index", "About the contributors", "Titles in This Series", imprint lines.
Scanned back matter rarely produces the next TITLE, so
the tracker kept the back-matter heading for everything after it. These
fixtures rebuild the audited shapes as ``ParsedDocument``s (no docling needed).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vo_rag.core.chunking.heading_filter import (
    is_back_matter_heading,
    is_degenerate_heading,
    looks_like_back_matter_entry,
)
from vo_rag.core.chunking.heading_tracker import STALE_HEADING_PAGES, HeadingTracker
from vo_rag.core.chunking.hierarchical import HierarchicalChunker
from vo_rag.core.chunking.models import RetrievalChunk
from vo_rag.core.parsing.contract import ElementType, PageQA, ParsedDocument, ParsedElement

TITLE = "Track and Field Coaching"
POLE_VAULT = (
    "The vaulter plants the pole in the box while the take-off foot is directly under the "
    "top hand, then swings the trail leg long and fast to load the pole before the rock back."
)
HIGH_JUMP = (
    "In the flop the athlete runs a curved approach, leans away from the bar on the last "
    "strides and drives the knee up at take-off so that the body rotates over the bar."
)
BIO = (
    "Jane Doe coached the national pole vault squad for twelve years and now lectures on "
    "biomechanics at a university, where she studies the energy exchange in the vault."
)
CITATIONS = (
    "Avery, J. (1998). Mechanics of rotation in the layout dive. Archives of Motion "
    "Science and Training, 12, 37-44.",
    "Ellis, J. G. (1991). The design of invented sports fields. 4th ed. Springfield, "
    "ST: Northwind Books.",
    "Marlow, N. P. (2000). Energy loss in the bar clearance take-off. Applied "
    "Mechanics Letters, 3, 205-218.",
)


def _el(
    i: int,
    kind: ElementType,
    text: str,
    page: int,
    depth: int = 1,
) -> ParsedElement:
    return ParsedElement(
        element_id=f"e{i}",
        element_type=kind,
        text=text,
        page_number=page,
        section_depth=depth,
        section_title="",
    )


def _doc(elements: list[ParsedElement], total_pages: int) -> ParsedDocument:
    return ParsedDocument(
        document_id="d",
        source_id="s",
        title=TITLE,
        total_pages=total_pages,
        elements=elements,
        images=[],
        page_qa=[
            PageQA(page_number=p, ocr_confidence=1.0, parser_used="docling")
            for p in range(1, total_pages + 1)
        ],
        markdown="",
        heading_tree=[],
        parser_used="docling",
    )


class _Builder:
    def __init__(self) -> None:
        self.elements: list[ParsedElement] = []

    def add(self, kind: ElementType, text: str, page: int, depth: int = 1) -> None:
        self.elements.append(_el(len(self.elements), kind, text, page, depth))

    def title(self, text: str, page: int, depth: int = 1) -> None:
        self.add(ElementType.TITLE, text, page, depth)

    def body(self, text: str, page: int) -> None:
        self.add(ElementType.NARRATIVE_TEXT, text, page)


def _chunks(doc: ParsedDocument) -> list[RetrievalChunk]:
    return HierarchicalChunker().chunk(doc, composites=[])


def _containing(chunks: list[RetrievalChunk], needle: str) -> list[RetrievalChunk]:
    found = [c for c in chunks if needle in c.content]
    assert found, needle
    return found


# --- predicates ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "References",
        "REFERENCES",
        "Reference List",
        "Bibliography",
        "Selected Bibliography",
        "INDEX",
        "Index",
        "Subject Index",
        "About the Author",
        "About the authors",
        "ABOUT THE CONTRIBUTORS",
        "About the Editors",
        "Titles in This Series",
        "Other Titles in This Series",
        "Acknowledgments",
        "ACKNOWLEDGEMENTS",
        "Notes",
        "Endnotes",
        "Further Reading",
        "Suggested Readings",
        "Glossary",
        "Copyright",
        "Library of Congress Cataloging-in-Publication Data",
        "Prentice Hall International",
        "Human Kinetics",
        "Oxford University Press",
        "Tafnews Press, Inc.",
        "Crowood Publishing",
        "References:",
        "INDEX 999",
    ],
)
def test_back_matter_headings_are_recognised(text: str) -> None:
    assert is_back_matter_heading(text)


@pytest.mark.parametrize(
    "text",
    [
        "Chapter 1 Introduction",
        "High Jump Technique",
        "The Reference Point",
        "Notes on the Pole Vault",
        "Coaching Notes",
        "About the Hammer Throw",
        "Indexing Training Load",
        "Going International",
        "Press-Ups and Pull-Ups",
        # Exercise headings ending in an imprint suffix word.
        "Bench Press",
        "Leg Press",
        "Push Press",
        "Military press",
        "The Overhead Press",
        "Full-court press",
        "Speed Limited",
        "Titles",
        # Numbered in-chapter sections of a structured book are not terminal.
        "38.9 Further Reading",
        "12.5 References",
    ],
)
def test_real_headings_are_not_back_matter(text: str) -> None:
    assert not is_back_matter_heading(text)


def test_back_matter_headings_are_still_meaningful_headings() -> None:
    # They label their own section; they are terminal, not degenerate.
    for text in ("Bibliography", "References", "INDEX", "About the contributors"):
        assert is_degenerate_heading(text) is None


@pytest.mark.parametrize(
    "text",
    [
        *CITATIONS,
        "Aldridge, H. M. 42",
        "aerobic capacity, 12, 45-47",
        "hurdles 88, 102",
        "[12] Calder C. Strength assessment with the Calder test. Rome: Regional Federation, 1999.",
        "3. See Northrop (1999), pp. 45-50.",
        "Smith, J., & Jones, K. (2004). Speed. Journal of Sports Sciences, 22, 1-10.",
        "Aerobic: requiring oxygen for the release of energy.",
        "[Qui+16] M. Quill et al. ScaleNet: a system for large-scale machine learning. 2016.",
        "Isometrics - hold training that uses a fixed joint angle under load.",
    ],
)
def test_reference_and_index_entries_look_like_back_matter(text: str) -> None:
    assert looks_like_back_matter_entry(text)


@pytest.mark.parametrize("text", [POLE_VAULT, HIGH_JUMP, BIO, "The Approach Run"])
def test_body_text_does_not_look_like_back_matter(text: str) -> None:
    assert not looks_like_back_matter_entry(text)


# --- terminal back-matter headings in the chunker -------------------------------


def test_body_after_about_the_contributors_on_a_later_page_drops_the_label() -> None:
    """Audited: pole-vault technique filed under ABOUT THE CONTRIBUTORS."""
    b = _Builder()
    b.title("ABOUT THE CONTRIBUTORS", 5)
    b.body(BIO, 5)
    for page in range(6, 10):
        b.body(f"{POLE_VAULT} Drill {page}.", page)
    chunks = _chunks(_doc(b.elements, 12))

    for chunk in _containing(chunks, "Jane Doe coached"):
        assert "ABOUT THE CONTRIBUTORS" in chunk.section_path
    for chunk in _containing(chunks, "The vaulter plants the pole"):
        assert "CONTRIBUTORS" not in chunk.section_path, chunk.section_path
        assert chunk.section_title != "ABOUT THE CONTRIBUTORS"


def test_references_keep_their_label_and_later_body_returns_to_the_chapter() -> None:
    """Audited: high-jump technique filed under Bibliography/References."""
    b = _Builder()
    b.title("Chapter 4 The High Jump", 30)
    for page in range(30, 36):
        b.body(f"{HIGH_JUMP} Phase {page}.", page)
    b.title("References", 36, depth=2)
    b.body(CITATIONS[0], 36)
    b.body(CITATIONS[1], 37)
    b.body(CITATIONS[2], 37)
    for page in range(38, 41):
        b.body(f"{HIGH_JUMP} Drill {page}.", page)
    chunks = _chunks(_doc(b.elements, 45))

    for needle in ("Avery, J.", "Ellis, J. G.", "Marlow, N. P."):
        for chunk in _containing(chunks, needle):
            assert chunk.section_title == "References", chunk.section_path
    late = [c for c in chunks if (c.page_number or 0) >= 38]
    assert late
    for chunk in late:
        assert "References" not in chunk.section_path, chunk.section_path
        assert chunk.section_path == f"[{TITLE} > Chapter 4 The High Jump]"


def test_body_after_index_on_a_later_page_drops_the_label() -> None:
    """Audited: shot-put technique filed under INDEX / Titles in This Series."""
    b = _Builder()
    b.title("Titles in This Series", 3)
    b.add(ElementType.LIST_ITEM, "Sprinting and Hurdling", 3)
    b.add(ElementType.LIST_ITEM, "The Throws", 3)
    b.title("INDEX", 8)
    b.body("aerobic capacity, 12, 45-47", 8)
    b.body("hurdles 88, 102", 9)
    for page in range(10, 13):
        b.body(f"The shot is held at the base of the fingers against the neck, page {page}.", page)
    chunks = _chunks(_doc(b.elements, 14))

    for chunk in _containing(chunks, "The shot is held"):
        assert "INDEX" not in chunk.section_path, chunk.section_path
        assert "Titles in This Series" not in chunk.section_path, chunk.section_path
    for chunk in _containing(chunks, "hurdles 88, 102"):
        assert "INDEX" in chunk.section_path


def test_real_heading_nested_under_stale_back_matter_drops_the_back_matter() -> None:
    """A mixed-case chapter (level 2) after an all-caps back-matter title (level 1)."""
    b = _Builder()
    b.title("ABOUT THE CONTRIBUTORS", 5)
    b.body(BIO, 5)
    b.title("Pole Vault Technique", 6, depth=2)
    for page in range(6, 9):
        b.body(f"{POLE_VAULT} Drill {page}.", page)
    chunks = _chunks(_doc(b.elements, 10))

    for chunk in _containing(chunks, "The vaulter plants the pole"):
        assert chunk.section_path == f"[{TITLE} > Pole Vault Technique]", chunk.section_path


# --- running headers ------------------------------------------------------------


def test_docling_running_header_never_becomes_a_section_heading(tmp_path) -> None:
    """A chapter's running head on a few pages of a long book is not caught by
    repeated-text detection; mapped to HEADER it is skipped."""
    from vo_rag.core.parsing.parsers.docling_mapping import docling_to_parsed_document
    from vo_rag.core.storage.figures import FigureStore

    def item(text: str, label: str, page: int) -> object:
        return SimpleNamespace(text=text, label=label, prov=[SimpleNamespace(page_no=page)])

    texts: list[object] = [item("Chapter 1 Sprinting", "section_header", 1)]
    for page in range(1, 201):
        if 10 <= page <= 13:
            texts.append(item("Sprint Drills Explained", "page_header", page))
        texts.append(item(f"{HIGH_JUMP} Page {page}.", "text", page))
    fake = SimpleNamespace(
        texts=texts,
        tables=[],
        pictures=[],
        pages=[SimpleNamespace(page_no=p) for p in range(1, 201)],
    )
    doc = docling_to_parsed_document(
        fake, FigureStore(tmp_path), "b.pdf", "pdf", src_hash="ab" * 32
    )
    chunks = _chunks(doc)

    assert chunks
    assert all("Sprint Drills Explained" not in c.section_path for c in chunks)
    assert all("Sprint Drills Explained" not in c.content for c in chunks)


# --- staleness guard ------------------------------------------------------------


def _long_unheaded_chapter(pages: int, *, running_head: bool) -> ParsedDocument:
    b = _Builder()
    b.title("Chapter 1 Endurance", 1)
    for page in range(1, pages + 1):
        if running_head and page > 1:
            b.add(ElementType.HEADER, f"CHAPTER 1 ENDURANCE {page}", page)
        b.body(f"{HIGH_JUMP} Page {page}.", page)
    return _doc(b.elements, pages)


def test_stale_heading_falls_back_to_the_document_title() -> None:
    chunks = _chunks(_long_unheaded_chapter(80, running_head=False))

    fresh = [c for c in chunks if (c.page_number or 0) <= 1 + STALE_HEADING_PAGES - 2]
    stale = [c for c in chunks if (c.page_number or 0) > 1 + STALE_HEADING_PAGES + 1]
    assert fresh and stale
    assert all("Chapter 1 Endurance" in c.section_path for c in fresh)
    for chunk in stale:
        assert "Chapter 1 Endurance" not in chunk.section_path, chunk.section_path
        assert chunk.metadata.get("stale_heading") == "Chapter 1 Endurance"
    assert all("stale_heading" not in c.metadata for c in fresh)


def test_running_head_matching_the_heading_keeps_it_fresh() -> None:
    chunks = _chunks(_long_unheaded_chapter(80, running_head=True))

    assert chunks
    assert all("Chapter 1 Endurance" in c.section_path for c in chunks)
    assert all("stale_heading" not in c.metadata for c in chunks)


def test_tracker_new_heading_clears_staleness() -> None:
    tracker = HeadingTracker(doc_title=TITLE)
    tracker.update_title("Chapter 1 Endurance", depth=1, page=1)
    assert tracker.observe_body(HIGH_JUMP, 1 + STALE_HEADING_PAGES + 1)
    assert tracker.section_path() == f"[{TITLE}]"
    assert tracker.stale_heading == "Chapter 1 Endurance"
    tracker.update_title("Chapter 2 Speed", depth=1, page=40)
    assert tracker.stale_heading is None
    assert tracker.section_path() == f"[{TITLE} > Chapter 2 Speed]"


# --- normal books unchanged -----------------------------------------------------


def test_normal_multi_chapter_book_keeps_chapter_labels() -> None:
    b = _Builder()
    chapters = ["Chapter 1 Sprints", "Chapter 2 Hurdles", "Chapter 3 Jumps", "Chapter 4 Throws"]
    page = 1
    expected: dict[int, str] = {}
    for chapter in chapters:
        b.title(chapter, page)
        for sub in range(3):
            b.title(f"Training Phase {sub + 1}", page, depth=2)
            for _ in range(7):
                expected[page] = chapter
                b.body(f"{HIGH_JUMP} {chapter} page {page}.", page)
                page += 1
    # A 24-page chapter with no subheadings stays labelled (under the staleness limit).
    b.title("Chapter 5 Relays", page)
    for _ in range(STALE_HEADING_PAGES - 1):
        expected[page] = "Chapter 5 Relays"
        b.body(f"{HIGH_JUMP} relays page {page}.", page)
        page += 1
    chunks = _chunks(_doc(b.elements, page))

    assert chunks
    for chunk in chunks:
        assert chunk.page_number is not None
        assert expected[chunk.page_number] in chunk.section_path, (
            chunk.page_number,
            chunk.section_path,
        )
        assert "stale_heading" not in chunk.metadata


# --- a contents-page "Index NNN" line ------------------

PREFACE = (
    "This book grew out of twenty years of coaching sprinters and hurdlers at club "
    "level, and it is written for the coach who has to plan a whole season alone."
)


@pytest.mark.parametrize("toc_depth", [1, 2])
def test_contents_page_index_line_does_not_swallow_the_next_heading(toc_depth: int) -> None:
    """The line "Index 245" on the contents page is a TOC line, not the
    index. It must not become a live terminal index heading that turns the next
    real heading ("Preface") into an index entry and files the preface under the
    contents page with ``front_matter_kind="toc"``."""
    b = _Builder()
    b.title("Contents", 3)
    for line in ("Preface ix", "1 Sprinting 1", "2 Hurdling 45", "Index 245"):
        b.title(line, 3, depth=toc_depth)
    b.title("Preface", 4)
    for page in (4, 5):
        b.body(f"{PREFACE} {HIGH_JUMP} {POLE_VAULT} Preface page {page}.", page)
    b.title("Sprinting", 6)
    b.body(f"{POLE_VAULT} {HIGH_JUMP} {POLE_VAULT} Sprint drill.", 6)
    chunks = _chunks(_doc(b.elements, 8))

    for chunk in _containing(chunks, "This book grew out of"):
        assert chunk.section_path == f"[{TITLE} > Preface]", chunk.section_path
        assert chunk.metadata.get("front_matter_kind") != "toc", chunk.metadata
    for chunk in _containing(chunks, "Sprint drill"):
        assert chunk.section_path == f"[{TITLE} > Sprinting]", chunk.section_path


def test_contents_page_back_matter_line_is_not_terminal_but_the_real_one_is() -> None:
    """The contents-page guard is narrow: a folio-carrying back-matter line, or one under an
    open Contents section, is not terminal; the real "Index" at the back of the
    book still is, and a running head "INDEX 999" still continues a live index."""
    toc = HeadingTracker(doc_title=TITLE)
    toc.update_title("Contents", depth=1, page=3)
    toc.update_title("Index", depth=2, page=3)
    assert toc.terminal_heading is None
    folio = HeadingTracker(doc_title=TITLE)
    folio.update_title("1 Sprinting 1", depth=1, page=3)
    folio.update_title("Index 245", depth=1, page=3)
    assert folio.terminal_heading is None

    real = HeadingTracker(doc_title=TITLE)
    real.update_title("Chapter 9 Relays", depth=1, page=40)
    real.update_title("Index", depth=1, page=50)
    assert real.terminal_heading == "Index"
    real.update_title("INDEX 999", depth=1, page=51)
    assert real.terminal_heading == "INDEX 999"


# --- "Bench Press" is an exercise, not an imprint ------

LIFT = (
    "Lower the bar under control to the lower chest, keep the shoulder blades pinched "
    "together and drive the bar up and slightly back over the shoulders."
)


def test_exercise_heading_ending_in_press_keeps_its_label_on_later_pages() -> None:
    """The heading "Bench Press" is not a terminal imprint heading: if it were, its
    second page would lose the label and "Common Faults" would lose its parent."""
    b = _Builder()
    b.title("Chapter 5 Strength Training", 10)
    b.body(f"{LIFT} Chapter intro.", 10)
    b.title("Bench Press", 11, depth=2)
    b.body(f"{LIFT} Set-up.", 11)
    b.body(f"{LIFT} Second page of the lift.", 12)
    b.title("Common Faults", 12, depth=3)
    b.body(f"{LIFT} Faults.", 12)
    chunks = _chunks(_doc(b.elements, 14))

    for needle in ("Set-up.", "Second page of the lift."):
        for chunk in _containing(chunks, needle):
            assert chunk.section_path == f"[{TITLE} > Chapter 5 Strength Training > Bench Press]"
    for chunk in _containing(chunks, "Faults."):
        assert "Bench Press" in chunk.section_path, chunk.section_path


def test_press_imprint_on_the_title_pages_is_still_terminal() -> None:
    """The ambiguous "... Press" suffix still marks an imprint line on the
    title/copyright pages (first pages of the book)."""
    tracker = HeadingTracker(doc_title=TITLE)
    tracker.update_title("Tafnews Press", depth=1, page=2)
    assert tracker.terminal_heading == "Tafnews Press"
    later = HeadingTracker(doc_title=TITLE)
    later.update_title("Bench Press", depth=2, page=11)
    assert later.terminal_heading is None


# --- program lines are not glossary entries ------------

PROGRAM_LINES = (
    "Monday: acceleration work of six by thirty metres from a crouch start, walk back "
    "recovery, then three by sixty metres at full speed with eight minutes between runs.",
    "Squat: 3x5 at 80%",
    "Day 1: 4x60m @95%",
    "Wednesday: tempo runs over the grass, ten by one hundred metres at seventy percent.",
)


@pytest.mark.parametrize("line", PROGRAM_LINES)
def test_training_program_lines_are_not_glossary_entries(line: str) -> None:
    assert not looks_like_back_matter_entry(line)


@pytest.mark.parametrize(
    "line",
    [
        "Aerobic: requiring oxygen",
        "Isometrics - hold training that uses a fixed joint angle under load on the muscle",
        "Anaerobic threshold: the intensity above which lactate accumulates in the blood",
    ],
)
def test_glossary_definitions_are_still_entries(line: str) -> None:
    assert looks_like_back_matter_entry(line)


def test_notes_heading_expires_on_a_weekly_program() -> None:
    """A "Notes" heading followed by two pages of a weekly program must not keep
    its label on every page (13-16 [.. > Week 3 Speed Program > Notes])."""
    b = _Builder()
    b.title("Week 3 Speed Program", 13)
    b.body(f"{HIGH_JUMP} Week three builds on the base.", 13)
    b.title("Notes", 13, depth=2)
    b.body("Keep all sprint work on a soft surface this week.", 13)
    for page in (14, 15, 16):
        b.body("\n".join(PROGRAM_LINES), page)
    chunks = _chunks(_doc(b.elements, 18))

    late = [c for c in chunks if (c.page_number or 0) >= 14]
    assert late
    for chunk in late:
        assert chunk.section_path == f"[{TITLE} > Week 3 Speed Program]", chunk.section_path

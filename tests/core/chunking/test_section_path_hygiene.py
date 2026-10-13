"""VOR-1454: junk headings never reach a chunk's section path.

The track-and-field tagger reads every ancestor of ``section_path``; a spot
check of 40 live chunks after the VOR-1452/1453 re-ingest found ~14 paths with
a junk ancestor (library stamps, bylines, sentences, axis labels, OCR garble,
front matter parenting the book). Depths follow ``docling_mapping``: short
ALL-CAPS headings are depth 1, mixed-case headings depth 2.
"""

from __future__ import annotations

import pytest

from vo_rag.core.chunking.heading_filter import (
    clean_heading_text,
    is_degenerate_heading,
    is_front_matter_line,
    junk_heading_reason,
    path_component_junk_reason,
    reads_as_index,
    reads_as_reference_list,
)
from vo_rag.core.chunking.heading_tracker import HeadingTracker
from vo_rag.core.chunking.hierarchical import HierarchicalChunker, _label_unheaded_back_matter
from vo_rag.core.chunking.models import RetrievalChunk
from vo_rag.core.parsing.contract import ElementType, PageQA, ParsedDocument, ParsedElement

T, N, L, TB, IMG = (
    ElementType.TITLE,
    ElementType.NARRATIVE_TEXT,
    ElementType.LIST_ITEM,
    ElementType.TABLE,
    ElementType.IMAGE,
)
Row = tuple[int, ElementType, int, str]

PROSE = (
    "Aerobic volume is built gradually across the season, and the coach raises it only "
    "when the athlete has absorbed the load of the weeks before it. The sprinter keeps "
    "the hips tall and the trunk relaxed through the drive phase of every repetition."
)


def _doc(rows: list[Row], pages: int, title: str = "Sprint Training") -> ParsedDocument:
    return ParsedDocument(
        document_id="d",
        source_id="s",
        title=title,
        total_pages=pages,
        elements=[
            ParsedElement(
                element_id=f"e{index}",
                element_type=kind,
                text=text,
                page_number=page,
                section_depth=depth,
                metadata={"markdown": text} if kind is TB else {},
            )
            for index, (page, kind, depth, text) in enumerate(rows)
        ],
        images=[],
        page_qa=[
            PageQA(page_number=p, ocr_confidence=0.8, parser_used="docling")
            for p in range(1, pages + 1)
        ],
        markdown="",
        heading_tree=[],
        parser_used="docling",
    )


def _chunks(doc: ParsedDocument) -> list[RetrievalChunk]:
    return HierarchicalChunker().chunk(doc, composites=[])


def _with(chunks: list[RetrievalChunk], text: str) -> RetrievalChunk:
    found = [c for c in chunks if text in c.content]
    assert found, f"{text!r} is in no chunk"
    return found[0]


def _prose(page: int, depth: int = 2, tag: str = "", repeat: int = 1) -> Row:
    return (page, N, depth, f"{' '.join([PROSE] * repeat)} {tag or f'Page {page}.'}")


# --- text classes ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("ACPL ITEM DISCARDED", "library_stamp"),
        ("PROPERTY OF", "library_stamp"),
        ("WEEK LOAN", "library_stamp"),
        ("Boston Public library", "library_stamp"),
        ("DR SAM ELLIS", "byline"),
        ("K. VELDRIMOR", "byline"),
        ("WILLIAM J. ALDRIDGE", "byline"),
        ("by Professor Lena Marlow Whitfield", "byline"),
        ("By Jordan Avery, Ukraine", "byline"),
        ("Sam Ellis, PhD", "byline"),
        ("Ted McAvery Head Track Coach", "byline"),
        ("Alameda, CA", "place_line"),
        ("In college in Ontario, she competed in the sprints and", "sentence"),
        ("Even though Sir Isaac Newton", "sentence"),
        ("Strength Endurance is usually defined as:", "sentence"),
        ("41, 77, 93). Gains in repeated sprint performance are likely related to", "sentence"),
        ("Further,", "sentence"),
        ("6-7 meters", "measurement"),
        ("40-60 seconds anaerobic 10-15% grade", "measurement"),
        ("MASS IN POUNDS", "axis_label"),
        ("Illustration y", "caption_label"),
        ("(Figure 16)", "caption_label"),
        ("Picture IV", "caption_label"),
        ("DATE MM,", "form_field"),
        ("440-Yard Dash NAME", "form_field"),
        ("ATHLETE", "form_field"),
        ("Kentish Post 14 August, 1754", "dateline"),
        ("CHAPTER", "bare_label"),
        ("ATHLETICS COACH", "masthead"),
        ("Remember:", "callout"),
        ("OF THE JAVELIN", "continuation"),
        ("-Meter Hurdles (Off-Season)", "continuation"),
        ("IWiTHDRAV WN", "library_stamp"),
        ("Jon\r\nDetermination of power dissipated to friction (P f )", "insufficient_words"),
        ("APPENDIX BODY FAT ESTIMATION TABLES (continued)", "continued"),
        ("AP Photo/Pat Marlow", "credit"),
        ("Multilateral Development · Slower performance improvement", "bullet_list"),
        ("1. Introduction 2. Retrospective analysis", "bullet_list"),
        ("Two Training Sessions Per Day 10:00 a.m. to Second session", "clock_time"),
        ("242 Basketball", "running_head"),
        ("Inertia, 60", "contents_line"),
    ],
)
def test_junk_heading_classes(text: str, reason: str) -> None:
    assert is_degenerate_heading(text) == reason


@pytest.mark.parametrize(
    "text",
    [
        "By the Numbers",
        "BY THE NUMBERS",
        "400 Meters",
        "800 metres",
        "The Shot Put Routine",
        "Warm Up and Cool Down",
        "Name That Muscle",
        "Mark McAvery",
        "3 The Technique of Hurdling",
        "35.4.2 Field Study-based Evaluation: Does the Tool Help a Reader Finish a Job?",
        "What Exercises Are Included in the Second Group?",
        "Chapter 9: Strength and coordination training for runners",
        "part one: - jumping out of the fire",
        "I. INTRODUCTION",
        "E. CONDITIONING",
        "PRACTICAL QUESTIONS RELATING TO TRAINING ANSWERED BY DR. ELLIS",
        "Relays, Hurdles",
        "The Assistant Coach",
        "5 — 100 ­ and 110 ­ Meter Hurdles",
        "8 — 3000 to 10,000 Meters",
    ],
)
def test_real_headings_are_kept(text: str) -> None:
    assert is_degenerate_heading(text) is None, junk_heading_reason(text)


@pytest.mark.parametrize(
    ("text", "page", "expected"),
    [
        ("Library of Congress Cataloging-in-Publication Data", 50, True),
        ("Copyright 2003 by Jordan Avery", 6, True),
        ("ISBN 0-7000-1234-6 LC 80-12345", 8, True),
        ("Ellis, Samuel, 1930–", 5, True),
        ("NORTHWIND PRESS LTD", 3, True),
        ("Human Kinetics", 3, True),
        ("Human Kinetics", 300, False),
        ("Bench Press", 3, True),  # a bare "X Press" on the title page
        ("Bench Press", 40, False),
        ("Copyright", 4, False),  # the copyright page's own label
    ],
)
def test_front_matter_lines(text: str, page: int, expected: bool) -> None:
    assert is_front_matter_line(text, page) is expected


@pytest.mark.parametrize(
    ("text", "cleaned"),
    [
        ("COMMUNICATING \r\nYO G \r\nYOUR APPROACH", "COMMUNICATING YOUR APPROACH"),
        ('"REFERENCES', "REFERENCES"),
        ("~ BIOMECHANICAL ANALYSIS", "BIOMECHANICAL ANALYSIS"),
        ("__Microcycles for the Shot Put __", "Microcycles for the Shot Put"),
        ("CHAPTER 9 —mmecaimenmssssssesssnesem", "CHAPTER 9"),
        ("BASIC CONSIDERATIONS |", "BASIC CONSIDERATIONS"),
        ("COACHING PRINCIPLES COACHING PRINCIPLES", "COACHING PRINCIPLES"),
        ("3 The  Technique  of Hurdling", "3 The  Technique  of Hurdling"),
        ("TRACK AND FIELD\nFOR GIRLS", "TRACK AND FIELD FOR GIRLS"),
    ],
)
def test_clean_heading_text(text: str, cleaned: str) -> None:
    assert clean_heading_text(text) == cleaned


def test_path_component_junk_reason() -> None:
    assert path_component_junk_reason("WEEK LOAN") == "library_stamp"
    assert path_component_junk_reason("RW ALVG") == "garbled"
    assert path_component_junk_reason("Library of Congress Cataloging Data") == (
        "front_matter_line"
    )
    assert path_component_junk_reason("pp. 221-230") is None  # the fallback label
    assert path_component_junk_reason("Erste Doppelstunde") is None
    assert path_component_junk_reason("DOMS") is None
    assert path_component_junk_reason("Pole Vault Pit Selection") is None


def test_index_and_reference_detection() -> None:
    index = (
        "abrasion 45-47 acceleration 12, 18, 22 approach run 56 arm action 71 baton 88-90 "
        "bend running 23 blocks 14, 16 bounding 102 calf raise 64 circuit 75 cross country 91"
    )
    schedule = (
        "Basic Core x 10 each side Big circle x 10 each direction Over the top x 10 each side "
        "Rotate and step x 10 each side Chop and step x 10 each side Twist x 10 each way"
    )
    references = (
        "Marlow PJ, Quill RN, Hartwell TT. Results of staged tendon grafting. "
        "J Appl Mech Rehabil. 2004;35(3):353-7. Northrop JM, Calder JR. Fatigue "
        "behaviour of the lumbar spine. Ann Sport Eng Rev. 2004;12:96-105."
    )
    history = "In 1968 Alex Rivera jumped 8.90 m. In 1991 Pat Marlow broke it in Tokyo."
    spaced = (
        "116 motion laws 40 -42 motor neurons 85 , 96 motor units 22 muscle 12 , 14 nerve 101 "
        "overload 3 , 7 , 9 pain 44 power 51 -53 reps 66 rest 70"
    )
    assert reads_as_index(index)
    assert reads_as_index(spaced)
    assert not reads_as_index(schedule)
    assert reads_as_reference_list(references)
    assert not reads_as_reference_list(history)


# --- the chunker ----------------------------------------------------------------


def test_a_title_page_stamp_parents_nothing() -> None:
    """ "WEEK LOAN" (depth 1, title page) held every mixed-case chapter of a book."""
    rows: list[Row] = [
        (1, T, 1, "RUNNING"),
        (3, T, 1, "WEEK LOAN"),
        (3, T, 1, "BIOMECHANICS AND EXERCISE PHYSIOLOGY"),
        (3, N, 1, "Jordan Avery Sam Ellis"),
        (5, T, 2, "Coordination and Running"),
        _prose(5),
        _prose(6),
        (7, T, 2, "Errors during thrust"),
        _prose(7, tag="Thrust page."),
    ]
    chunks = _chunks(_doc(rows, 8))
    thrust = _with(chunks, "Thrust page.")
    assert thrust.section_path == "[Sprint Training > Errors during thrust]", thrust.section_path
    assert all("WEEK LOAN" not in c.section_path for c in chunks)


def test_a_cip_block_and_bylines_are_text_not_headings() -> None:
    rows: list[Row] = [
        (2, T, 1, "DISCUS"),
        (2, T, 2, "by Dr. Sam Ellis"),
        (4, T, 2, "Library of Congress Cataloging-in-Publication Data"),
        (4, N, 2, "Ellis, Sam. Discus. Includes index. ISBN 0-7000-5432-4"),
        (6, T, 1, "DR SAM ELLIS"),
        (6, T, 2, "Learning the Turn"),
        _prose(6, tag="Turn page."),
        _prose(7),
    ]
    chunks = _chunks(_doc(rows, 8))
    turn = _with(chunks, "Turn page.")
    assert turn.section_path == "[Sprint Training > Learning the Turn]", turn.section_path
    assert all("Cataloging" not in c.section_path for c in chunks)


def test_a_garbled_heading_is_text() -> None:
    rows: list[Row] = [
        (1, T, 1, "THE DISCUS"),
        _prose(1),
        (2, T, 1, "RW ALVG"),
        _prose(2, tag="Routine page."),
        (3, T, 2, "The Discus Routine"),
        _prose(3, tag="Discus routine page."),
    ]
    chunks = _chunks(_doc(rows, 3))
    assert all("ALVG" not in c.section_path for c in chunks)
    assert _with(chunks, "Discus routine page.").section_path == (
        "[Sprint Training > THE DISCUS > The Discus Routine]"
    )


def test_a_clipped_running_head_is_text() -> None:
    rows: list[Row] = [
        (1, T, 2, "Periodization of Strength"),
        _prose(1),
        (2, T, 2, "eriodization of Strength"),
        _prose(2, tag="Clipped page."),
        (3, T, 2, "Periodization of Agility and Power Training"),
        _prose(3, tag="Agility page."),
    ]
    chunks = _chunks(_doc(rows, 3))
    assert _with(chunks, "Clipped page.").section_path == (
        "[Sprint Training > Periodization of Strength]"
    )


def test_a_book_title_running_head_is_skipped_on_every_run() -> None:
    """ "STRENGTHCRAFT" opened page after page; every run's first page used to
    keep it as a depth-1 heading that parented the sections after it."""
    rows: list[Row] = []
    for page in range(2, 40, 5):
        rows += [(page, T, 1, "STRENGTHCRAFT"), _prose(page)]
    rows.insert(3, (2, T, 2, "Adaptation and Training"))
    rows.append((39, T, 2, "Strength Deficit"))
    rows.append(_prose(39, tag="Deficit page."))
    chunks = _chunks(_doc(rows, 40))
    assert all("STRENGTHCRAFT" not in c.section_path for c in chunks)
    assert _with(chunks, "Deficit page.").section_path == "[Sprint Training > Strength Deficit]"


def test_a_tables_title_parents_no_later_heading() -> None:
    rows: list[Row] = [
        (53, T, 1, "UNITS OF MOMENTUM"),
        (54, T, 1, "MASS IN POUNDS"),
        (54, TB, 1, "| 100 | 125 | 150 |\n|---|---|---|\n| 600 | 750 | 900 |"),
        _prose(54),
        (55, T, 2, "Conservation Of Rotary Momentum"),
        _prose(55, tag="Rotary page."),
    ]
    chunks = _chunks(_doc(rows, 56))
    rotary = _with(chunks, "Rotary page.")
    assert rotary.section_path == "[Sprint Training > Conservation Of Rotary Momentum]"


def test_a_chapter_is_never_inside_an_appendix_or_a_drill() -> None:
    rows: list[Row] = [
        (35, T, 1, "APPENDIX BODY FAT ESTIMATION TABLES"),
        _prose(35),
        (40, T, 2, "3 — 100 and 200 Meters"),
        _prose(40, tag="Sprint chapter page.", repeat=6),
        (41, T, 3, "Sprinting Mechanics"),
        _prose(41, tag="Mechanics page.", repeat=6),
    ]
    chunks = _chunks(_doc(rows, 42))
    assert _with(chunks, "Mechanics page.").section_path == (
        "[Sprint Training > 3 — 100 and 200 Meters > Sprinting Mechanics]"
    )


def test_a_mid_page_capitalised_line_does_not_parent_the_next_pages_section() -> None:
    """ "MATCHES TO COME." inside a reproduced poster held every mixed-case
    chapter after it."""
    rows: list[Row] = [
        (37, T, 2, "The World of the Professional"),
        _prose(37),
        _prose(38),
        (38, T, 1, "MATCHES TO COME."),
        (38, L, 1, "7. Barker and Jones to run 10 miles, 10 a side, at Brompton."),
        (43, T, 2, "Sprints"),
        _prose(43, tag="Sprints page."),
    ]
    chunks = _chunks(_doc(rows, 44))
    assert _with(chunks, "Sprints page.").section_path == "[Sprint Training > Sprints]"


def test_weekdays_are_siblings_whatever_their_depth() -> None:
    rows: list[Row] = [
        (1, T, 1, "PRE-SEASON TRAINING"),
        (1, T, 2, "Wednesday"),
        _prose(1, repeat=6),
        (1, T, 3, "Friday"),
        _prose(1, tag="Friday text.", repeat=6),
    ]
    chunks = _chunks(_doc(rows, 2))
    assert _with(chunks, "Friday text.").section_path == (
        "[Sprint Training > PRE-SEASON TRAINING > Friday]"
    )


def test_a_run_of_short_standalone_lines_is_a_figures_labels() -> None:
    rows: list[Row] = [
        (1, T, 1, "THE JAVELIN THROW"),
        _prose(1),
        (2, N, 2, "Carry"),
        (2, N, 2, "Run"),
        (2, N, 2, "Withdrawal"),
        _prose(2, tag="Figure page."),
    ]
    chunks = _chunks(_doc(rows, 3))
    assert _with(chunks, "Figure page.").section_path == "[Sprint Training > THE JAVELIN THROW]"


def test_an_unheaded_index_is_labelled_index() -> None:
    index = (
        "abrasion 45-47 acceleration 12, 18, 22 approach run 56 arm action 71 baton 88-90 "
        "bend running 23 blocks 14, 16 bounding 102 calf raise 64 circuit 75 cross country 91"
    )
    rows: list[Row] = [
        (1, T, 1, "CHAPTER 30 SUMMARY"),
        _prose(1),
        (40, N, 1, index),
    ]
    chunks = _chunks(_doc(rows, 41))
    labelled = _with(chunks, "abrasion 45-47")
    assert labelled.section_path == "[Sprint Training > Index]"
    assert labelled.metadata.get("front_matter_kind") == "index"


def test_actual_index_replaces_inherited_answers_heading() -> None:
    index = (
        "abrasion 45-47 acceleration 12, 18, 22 approach run 56 arm action 71 baton 88-90 "
        "bend running 23 blocks 14, 16 bounding 102 calf raise 64 circuit 75 cross country 91"
    )
    rows: list[Row] = [
        (1, T, 1, "ANSWERS TO PRACTICE EXERCISES"),
        (2, N, 1, index),
    ]
    chunks = _chunks(_doc(rows, 3))
    assert _with(chunks, "abrasion 45-47").section_path == "[Sprint Training > Index]"
    answers = _chunks(_doc([(1, T, 1, "ANSWERS TO PRACTICE EXERCISES"), _prose(1)], 1))
    _label_unheaded_back_matter(answers, "Sprint Training", lambda text: len(text.split()))
    assert "ANSWERS TO PRACTICE EXERCISES" in answers[0].section_path


def test_late_title_and_publication_details_are_not_relabelled_references() -> None:
    rows: list[Row] = [
        (1, T, 1, "CHAPTER 30 SUMMARY"),
        _prose(1),
        (40, T, 2, "Notes on the Use of Tables and Formulas in"),
        (40, T, 2, "Workshop Reference 27th Edition"),
        (40, N, 1, "BY JORDAN A. AVERY, SAM B. ELLIS, AND CASEY C. MARLOW"),
        (40, N, 1, "PAT J. NORTHWIND, EDITOR ROBIN QUILL, ASSOCIATE EDITOR"),
        (40, N, 1, "2004\nNORTHWIND PRESS INC. SPRINGFIELD"),
        (40, N, 1, "COPYRIGHT 1931, 1939, 1951, 1954, 2004 by Northwind Press Inc."),
        (40, N, 1, "Library of Congress Cataloging-in-Publication Data"),
        (41, N, 1, "Avery, Jordan Alan, 1887-1968\nEllis, Samuel Blake, 1879-1967"),
        (41, N, 1, "ISBN 0-7000-9876-3. This publication accompanies the technical handbook."),
    ]
    chunks = _chunks(_doc(rows, 42))
    publication = _with(chunks, "BY JORDAN A. AVERY")
    assert publication.section_path.endswith(" > Workshop Reference 27th Edition]")
    assert "Notes on the Use of Tables and Formulas in" in publication.content
    source_fields = "\n".join(c.content + "\n" + c.section_path for c in chunks)
    for text in ("Notes on the Use of Tables and Formulas in", "Workshop Reference 27th Edition"):
        assert text in source_fields
    assert "References" not in _with(chunks, "ISBN 0-7000-9876-3").section_path


def test_source_index_header_labels_short_continuations_only_on_its_page() -> None:
    rows: list[Row] = [
        (1, T, 1, "ANSWERS TO PRACTICE EXERCISES"),
        _prose(1, repeat=12),
        (2, ElementType.HEADER, 1, "INDEX"),
        (2, T, 2, "G ear"),
        (2, N, 1, "helical, 169-172\nspur, 173-177\nworm, 185-190"),
        (3, T, 1, "APPENDIX A FORMULAS"),
        _prose(3, tag="Appendix prose."),
    ]
    chunks = _chunks(_doc(rows, 3))
    assert _with(chunks, "helical, 169-172").section_path == "[Sprint Training > INDEX]"
    assert "G ear" in _with(chunks, "helical, 169-172").content
    assert "Index" not in _with(chunks, "Appendix prose.").section_path


def test_lower_case_title_in_a_capitalised_book_is_text() -> None:
    rows: list[Row] = []
    for page in range(1, 9):
        rows += [(page, T, 2, f"Section {chr(65 + page)} Training"), _prose(page)]
    rows += [(9, T, 2, "the power transferred to the implement"), _prose(9, tag="Implement.")]
    chunks = _chunks(_doc(rows, 10))
    assert "the power transferred" not in _with(chunks, "Implement.").section_path


def test_a_capitalised_standalone_line_is_a_sibling_of_mixed_case_headings() -> None:
    tracker = HeadingTracker(doc_title="Book")
    tracker.update_title("Chapter 4 Training", depth=1, page=10)
    tracker.update_title("Hurdler's Stretch", depth=2, page=12)
    assert tracker.section_path(["WEIGHT LIFTING"]) == (
        "[Book > Chapter 4 Training > WEIGHT LIFTING]"
    )
    assert tracker.section_path(["Toe Raises"]) == (
        "[Book > Chapter 4 Training > Hurdler's Stretch > Toe Raises]"
    )


def test_tracker_front_headings_never_parent_later_pages() -> None:
    tracker = HeadingTracker(doc_title="Book", front_pages=4)
    tracker.update_title("TRACK AND FIELD", depth=1, page=4)
    tracker.update_title("The Finish", depth=2, page=34)
    assert tracker.section_path() == "[Book > The Finish]"
    structural = HeadingTracker(doc_title="Book", front_pages=8)
    structural.update_title("Part I: Setting the Stage", depth=1, page=8)
    structural.update_title("Reason One", depth=2, page=20)
    assert structural.section_path() == "[Book > Part I: Setting the Stage > Reason One]"

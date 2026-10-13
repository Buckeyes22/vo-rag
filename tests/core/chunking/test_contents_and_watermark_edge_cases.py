"""Contents-page, numbered-step, running-head, lead-in, reading-order and
nothing-dropped edge cases.

Depths follow what ``docling_mapping._infer_heading_level`` emits: "Chapter N"
and short ALL-CAPS headings are depth 1; numbered ("1 Warm up") and mixed-case
headings are depth 2.
"""

from __future__ import annotations

import pytest

from vo_rag.core.chunking.contents_pages import contents_pages
from vo_rag.core.chunking.heading_filter import is_lead_in_line
from vo_rag.core.chunking.hierarchical import (
    ElementRoute,
    HierarchicalChunker,
    RoutingContext,
    route_element,
)
from vo_rag.core.chunking.models import RetrievalChunk
from vo_rag.core.chunking.reading_order import reading_order_permutation
from vo_rag.core.parsing.contract import ElementType, PageQA, ParsedDocument, ParsedElement

T, N, L, TB = (
    ElementType.TITLE,
    ElementType.NARRATIVE_TEXT,
    ElementType.LIST_ITEM,
    ElementType.TABLE,
)
Row = tuple[int, ElementType, int, str]

PROSE = (
    "Aerobic volume is built gradually across the season, and the coach raises it only "
    "when the athlete has absorbed the load of the weeks before it."
)
LONG = " ".join([PROSE] * 4)


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


def _prose_pages(pages: range, depth: int = 1) -> list[Row]:
    return [(p, N, depth, f"{PROSE} Page {p}.") for p in pages]


def _with(chunks: list[RetrievalChunk], text: str) -> RetrievalChunk:
    found = [c for c in chunks if text in c.content]
    assert found, f"{text!r} is in no chunk"
    return found[0]


# --- a page in a Contents context needs only a light entry check -----------


def _two_level_contents() -> list[Row]:
    """A chapter and its first section share a page number (5 of 7 steps rise)."""
    return [
        (5, T, 1, "Contents"),
        (5, T, 2, "1 The Hurdles 45"),
        (5, N, 2, "Hurdle History 45"),
        (5, N, 2, "Hurdle Technique 47"),
        (5, N, 2, "Hurdle Drills 52"),
        (5, T, 2, "2 The Sprints 60"),
        (5, N, 2, "Sprint History 60"),
        (5, N, 2, "Block Starts 63"),
        (5, N, 2, "Sprint Drills 70"),
    ]


def test_a_two_level_contents_page_with_shared_page_numbers_is_contents() -> None:
    rows = _two_level_contents() + [(7, N, 1, f"{PROSE} Untitled page seven.")]
    doc = _doc(rows, 200)
    assert contents_pages(doc) == frozenset({5})
    page_seven = _with(_chunks(doc), "Untitled page seven")
    assert "Contents" not in page_seven.section_path
    assert page_seven.metadata.get("front_matter_kind") != "toc"


def test_a_short_continuation_page_with_one_tie_continues_the_run() -> None:
    rows = _two_level_contents() + [
        (6, T, 2, "3 The Throws 80"),
        (6, N, 2, "Throwing History 80"),
        (6, N, 2, "The Shot Put 84"),
        (6, N, 2, "The Discus 90"),
    ]
    assert contents_pages(_doc(rows, 200)) == frozenset({5, 6})


def test_blurbs_and_split_page_numbers_on_a_headed_page_are_contents() -> None:
    rows: list[Row] = [
        (5, T, 1, "Contents"),
        (5, T, 2, "1 The Hurdles 45"),
        (5, N, 2, "How to clear the barrier without losing speed. Drills for every level."),
        (5, T, 2, "2 The Sprints 60"),
        (5, T, 2, "3 The Relays 75"),
        (5, T, 2, "4 How fast should a young sprinter train?"),
        (5, N, 2, "95"),
    ]
    assert contents_pages(_doc(rows, 200)) == frozenset({5})


def test_a_headed_page_whose_prose_outweighs_its_entries_is_not_contents() -> None:
    rows: list[Row] = [
        (5, T, 1, "Contents of the kit bag"),
        (5, N, 1, PROSE),
        (5, N, 1, PROSE + " Again."),
        (5, N, 1, PROSE + " Once more."),
        (5, L, 1, "Spikes 12"),
        (5, L, 1, "Tape 14"),
    ]
    assert contents_pages(_doc(rows, 200)) == frozenset()


def test_a_contents_table_is_an_entry_page() -> None:
    rows: list[Row] = [
        (5, T, 1, "Contents"),
        (
            5,
            TB,
            1,
            "| Chapter | Title | Page |\n|---|---|---|\n| 1 | Sprinting Technique | 5 |\n"
            "| 2 | Hurdling Technique | 30 |\n| 3 | Relay Racing | 52 |",
        ),
    ]
    assert contents_pages(_doc(rows, 120)) == frozenset({5})


def test_a_heading_with_prose_on_a_contents_page_opens_its_own_section() -> None:
    """A booklet's contents page that also holds the preface: the preface is
    not filed as contents; the contents table keeps the Contents label."""
    rows: list[Row] = [
        (3, T, 1, "CONTENTS"),
        (3, T, 1, "PREFACE"),
        (3, N, 1, "The author has coached the throws for thirty years. " + PROSE),
        (
            3,
            TB,
            1,
            "| Title | Page |\n|---|---|\n| Historical Development | 1 |\n"
            "| Basic Mechanics | 4 |\n| Training the Thrower | 12 |\n| Competition Rules | 20 |",
        ),
        *_prose_pages(range(4, 6)),
    ]
    doc = _doc(rows, 48)
    assert contents_pages(doc) == frozenset({3})
    chunks = _chunks(doc)
    preface = _with(chunks, "thirty years")
    assert preface.section_path.endswith("> PREFACE]"), preface.section_path
    assert preface.metadata.get("front_matter_kind") != "toc"
    table = _with(chunks, "Historical Development")
    assert table.section_path.endswith("> Contents]"), table.section_path


def test_a_body_page_after_a_contents_page_keeps_its_text_and_chapter() -> None:
    """A body page after a contents page keeps its text; the text must exist."""
    rows: list[Row] = [
        (3, T, 1, "Contents"),
        (3, T, 2, "1 Foundations of Sprinting 5"),
        (3, T, 2, "2 Building Training Volume 12"),
        (3, T, 2, "3 Speed and Power 30"),
        (4, T, 1, "Chapter 1 Foundations"),
        (4, N, 1, PROSE),
        (4, L, 1, "Week 1 Tempo runs 100"),
        (4, L, 1, "Week 2 Tempo runs 120"),
        (4, L, 1, "Week 3 Tempo runs 140"),
        *_prose_pages(range(5, 7)),
    ]
    doc = _doc(rows, 150)
    assert contents_pages(doc) == frozenset({3})
    week = _with(_chunks(doc), "Week 1 Tempo")
    assert week.section_path == "[Sprint Training > Chapter 1 Foundations]"


# --- numbered steps vs chapters, by the siblings -----------------------------


def test_numbered_steps_under_a_session_plan_merge_at_the_parser_depths() -> None:
    rows: list[Row] = [
        (5, T, 1, "Chapter 2 Sessions"),
        (5, N, 1, PROSE),
        (5, T, 2, "Session Plan"),
        (5, T, 2, "1 Warm up"),
        (5, N, 2, "Jog 800 m easy."),
        (5, T, 2, "2 Mobility"),
        (5, N, 2, "Hips and ankles, ten reps each."),
        (5, T, 2, "3 Acceleration"),
        (5, N, 2, "Six 30 m builds from blocks."),
        (5, T, 2, "4 Cool down"),
        (5, N, 2, "Walk 400 m and stretch."),
    ]
    steps = _with(_chunks(_doc(rows, 40)), "Jog 800 m easy")
    assert "Six 30 m builds" in steps.content
    assert "Walk 400 m" in steps.content


def test_a_numbered_chapter_beside_unnumbered_siblings_is_not_merged() -> None:
    rows: list[Row] = [
        (5, T, 1, "INTRODUCTION"),
        (5, T, 2, "1 The Sprint Start"),
        (5, N, 2, "Every race begins in the blocks."),
        (5, T, 2, "The Set Position"),
        (5, N, 2, "Hips rise just above the shoulders."),
        (5, T, 2, "The Drive Phase"),
        (5, N, 2, LONG),
    ]
    chunks = _chunks(_doc(rows, 40))
    start = _with(chunks, "Every race begins")
    assert "Hips rise" not in start.content
    assert start.section_path.endswith("> 1 The Sprint Start]"), start.section_path


def test_numbered_chapters_with_long_sections_stay_apart() -> None:
    rows: list[Row] = [
        (5, T, 1, "SPRINT TRAINING"),
        (5, T, 2, "1 Introduction"),
        (5, N, 2, LONG + " One."),
        (6, T, 2, "2 The Sprint Start"),
        (6, N, 2, LONG + " Two."),
        (7, T, 2, "3 Acceleration"),
        (7, N, 2, LONG + " Three."),
    ]
    chunks = _chunks(_doc(rows, 40))
    for title, marker in (("1 Introduction", "One."), ("2 The Sprint Start", "Two.")):
        chunk = _with(chunks, marker)
        assert chunk.section_path.endswith(f"> {title}]"), chunk.section_path


# --- running heads need page-edge evidence; real-word swaps are headings ------


def _routes(doc: ParsedDocument, prefix: str) -> dict[int, ElementRoute]:
    routing = RoutingContext.for_document(doc)
    return {
        e.page_number or 0: route_element(e, e.text, routing)
        for e in doc.elements
        if e.element_type is T and e.text.startswith(prefix)
    }


def test_a_real_word_swap_does_not_make_real_headings_running_heads() -> None:
    heads = {10: "Hurdle Drills", 13: "Hurdle Drills", 16: "Hurdle Drills", 19: "Hurdle Skills"}
    rows: list[Row] = [(9, T, 1, "Part One Hurdles")]
    for page in range(9, 21):
        if page in heads:
            rows.append((page, T, 2, heads[page]))
        rows.append((page, N, 2, f"{PROSE} Page {page}."))
    routes = _routes(_doc(rows, 40), "Hurdle")
    assert set(routes.values()) == {ElementRoute.UPDATE_TITLE}, routes


def test_an_ocr_confusion_run_keeps_its_first_page_as_the_heading() -> None:
    """ "Track and Field"/"Track and Held" at page tops: a running head, except
    the run's first page, which opens the section (the opener exemption)."""
    heads = {20: "Track and Field", 21: "Track and Held", 22: "Track and Field"}
    heads[23] = "Track and Field"
    rows: list[Row] = [(19, T, 1, "Chapter 3 Field Events")]
    for page in range(19, 25):
        if page in heads:
            rows.append((page, T, 2, heads[page]))
        rows.append((page, N, 2, f"{PROSE} Page {page}."))
    routes = _routes(_doc(rows, 40), "Track and")
    assert routes[20] is ElementRoute.UPDATE_TITLE, routes
    assert {routes[21], routes[22], routes[23]} == {ElementRoute.SKIP_REPEATED}, routes


def test_repeated_short_text_mid_page_is_kept() -> None:
    """ "Weekly Plan" closes every page (a page edge: a running footer); the same
    words inside a page head a workout there and are the book's text."""
    rows: list[Row] = []
    for page in range(10, 16):
        rows.append((page, N, 1, f"{PROSE} Page {page}."))
        if page == 12:
            rows.append((page, L, 1, "Weekly Plan"))
            rows.append((page, L, 1, "6 x 200 m at 90 percent."))
        rows.append((page, N, 1, "Weekly Plan"))
    doc = _doc(rows, 16)
    routing = RoutingContext.for_document(doc)
    mid = [e for e in doc.elements if e.page_number == 12 and e.text == "Weekly Plan"][0]
    edge = [e for e in doc.elements if e.page_number == 11 and e.text == "Weekly Plan"][0]
    assert route_element(edge, edge.text, routing) is ElementRoute.SKIP_REPEATED
    assert route_element(mid, mid.text, routing) is ElementRoute.LIST_ITEM


# --- lead-in lines ------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["For example:", "For instance:", "In summary:", "In other words:", "To summarise:"],
)
def test_a_connective_opener_is_a_lead_in(text: str) -> None:
    assert is_lead_in_line(text)


@pytest.mark.parametrize(
    "text",
    [
        "Points to note:",
        "Key points to note:",
        "Drills to include:",
        "Sprinting as a skill:",
        "Arm action as a counterbalance:",
    ],
)
def test_a_label_with_an_infinitive_or_as_is_a_heading(text: str) -> None:
    assert not is_lead_in_line(text)


def test_for_example_does_not_open_a_section() -> None:
    rows: list[Row] = [
        (5, T, 1, "Chapter 4 Technique"),
        (5, N, 1, PROSE),
        (5, T, 2, "For example:"),
        (5, L, 2, "Keep the hips tall through the drive phase."),
    ]
    chunk = _with(_chunks(_doc(rows, 40)), "hips tall")
    assert chunk.section_path == "[Sprint Training > Chapter 4 Technique]", chunk.section_path


# --- reading order -------------------------------------------------------------


def _stored_order_tables() -> list[Row]:
    """A legacy parse: the table of p. 8 listed after the book's last text."""
    return [
        (7, T, 1, "Chapter 1 Speed"),
        (7, N, 1, f"{PROSE} Page 7."),
        (8, N, 1, f"{PROSE} Page 8."),
        (9, T, 1, "Chapter 2 Power"),
        (9, N, 1, f"{PROSE} Page 9."),
        (30, T, 1, "Quick Order Form"),
        (30, N, 1, "Mail this form with your payment to the publisher."),
        (8, TB, 1, "| Test | Score |\n|---|---|\n| 40 yd | 4.5 |\n| 10 yd | 1.6 |"),
    ]


def test_a_legacy_table_takes_the_heading_of_its_own_page() -> None:
    doc = _doc(_stored_order_tables(), 30)
    assert reading_order_permutation(doc.elements) == [0, 1, 2, 7, 3, 4, 5, 6]
    table = _with(_chunks(doc), "40 yd")
    assert table.section_path == "[Sprint Training > Chapter 1 Speed]", table.section_path


def test_a_table_met_mid_sentence_does_not_cut_the_sentence() -> None:
    rows: list[Row] = [
        (7, T, 1, "Chapter 1 Speed"),
        (7, N, 1, f"{PROSE} The athlete who runs the 40 yd"),
        (7, TB, 1, "| Test | Score |\n|---|---|\n| 40 yd | 4.5 |"),
        (8, N, 1, "dash under five seconds has elite starting speed for the sport."),
    ]
    chunks = _chunks(_doc(rows, 30))
    text = " ".join(" ".join(c.content.split()) for c in chunks)
    assert "runs the 40 yd dash under five seconds" in text
    table = _with(chunks, "Score")
    assert table.section_path == "[Sprint Training > Chapter 1 Speed]"


# --- a note between references --------------------------------------------------


def test_references_after_an_acknowledgements_note_keep_their_label() -> None:
    rows: list[Row] = [
        (30, T, 1, "Chapter 1 Muscle"),
        (30, N, 1, LONG),
        (31, T, 2, "References"),
        (31, L, 2, "Northrop, G.R. (1993) Mapping of tissue response. J Appl Mech 74, 532-537."),
        (31, L, 2, "Calder, G.M. (1995) Reliability of measurements. Tissue Res 18, 593-600."),
        (31, T, 2, "Acknowledgements"),
        (31, N, 2, "This work was partially supported by a grant from the National Institutes."),
        (
            31,
            L,
            2,
            "Avery, R.M. (1988) Strength and its development. Sports Eng 6, 146-168.",
        ),
        (32, L, 2, "Ellis, E. (1957) Relation between size of cells. Lab Notes 126, 1345-1347."),
    ]
    chunks = _chunks(_doc(rows, 40))
    note = _with(chunks, "partially supported")
    assert note.section_path.endswith("> Acknowledgements]"), note.section_path
    for citation in ("Avery, R.M.", "Ellis, E."):
        chunk = _with(chunks, citation)
        assert chunk.section_path.endswith("> References]"), chunk.section_path
        assert "partially supported" not in chunk.content


# --- nothing the chunker is given is dropped -----------------------------------


def test_index_lines_are_kept_under_the_index() -> None:
    rows: list[Row] = [
        (60, T, 1, "Chapter 9 Throws"),
        (60, N, 1, LONG),
        (68, T, 1, "INDEX"),
        *[
            (68, N, 1, line)
            for line in ("Calisthenics / 22", "Finishing / 32", "Floating / 34", "Hurdles / 43-46")
        ],
    ]
    text = " ".join(c.content for c in _chunks(_doc(rows, 68)))
    for line in ("Calisthenics / 22", "Floating / 34", "Hurdles / 43-46"):
        assert line in text, line


def test_workout_lines_heavy_in_digits_are_kept() -> None:
    rows: list[Row] = [
        (124, T, 2, "January and February"),
        (124, N, 2, "Monday:"),
        (124, N, 2, "2  X  220,  550,  2  X  220  or  2  X  660"),
        (124, N, 2, "5  X  3000r7  X  2200r10 X  150"),
        (124, N, 2, PROSE),
    ]
    text = " ".join(c.content for c in _chunks(_doc(rows, 424)))
    assert "2 X 220, 550, 2 X 220 or 2 X 660" in " ".join(text.split())
    assert "5 X 3000r7" in " ".join(text.split())


def test_a_rejected_title_that_carries_text_is_body_text() -> None:
    rows: list[Row] = [
        (21, T, 1, "Chapter 2 Relays"),
        (21, N, 1, PROSE),
        (21, T, 2, "2.50 4x 400"),
        (21, N, 2, PROSE + " After the split."),
    ]
    text = " ".join(c.content for c in _chunks(_doc(rows, 40)))
    assert "2.50 4x 400" in text


def test_a_bulleted_line_typed_as_a_title_is_a_list_line() -> None:
    rows: list[Row] = [
        (48, T, 2, "STRENGTH"),
        (48, T, 5, "■ strength"),
        (48, T, 5, "■ speed"),
        (48, T, 5, "■ power"),
        (48, N, 5, PROSE),
    ]
    chunks = _chunks(_doc(rows, 60))
    text = " ".join(c.content for c in chunks)
    for bullet in ("■ strength", "■ speed", "■ power"):
        assert bullet in text, bullet
    assert all("■" not in c.section_path for c in chunks)


def test_a_heading_replaced_before_any_text_is_kept() -> None:
    rows: list[Row] = [
        (211, T, 2, "4.3 Derived Principles of Training"),
        (211, T, 3, "4.3.1 Types of workloads"),
        (211, T, 2, "Progressive workloads"),
        (211, N, 2, PROSE),
    ]
    chunk = _with(_chunks(_doc(rows, 300)), "Aerobic volume")
    assert "4.3 Derived Principles of Training" in chunk.content
    assert "4.3.1 Types of workloads" in chunk.content
    assert chunk.section_path.endswith("> Progressive workloads]"), chunk.section_path


def test_a_deep_heading_inside_a_chunk_is_a_run_in_line() -> None:
    rows: list[Row] = [
        (10, T, 2, "Safety"),
        (10, T, 4, "Mit Matten sichern"),
        (10, N, 4, "Landing areas are covered with mats. " + PROSE),
        (10, T, 4, "Noch ein wichtiger Hinweis:"),
        (10, N, 4, "Indoor shoes only in the hall."),
    ]
    text = " ".join(c.content + " " + c.section_path for c in _chunks(_doc(rows, 40)))
    for heading in ("Mit Matten sichern", "Noch ein wichtiger Hinweis"):
        assert heading in text, heading


# --- a generic subheading belongs to the item before it --------------------------


def _drills(generic: str = "COACHING TIPS") -> list[Row]:
    rows: list[Row] = [(20, T, 1, "HURDLES")]
    for number, drill in enumerate(["High-Knee Running", "Simulated Hurdling", "Lead-Leg Action"]):
        page = 20 + number
        rows += [
            (page, T, 2, drill),
            (page, N, 2, f"This drill builds the hurdle rhythm. {LONG}"),
            (page, T, 1, generic),
            (page, L, 1, f"Look forward on the {drill.lower()} and keep the arms swinging."),
        ]
    rows += [(40, T, 1, "RELAYS"), (40, N, 1, LONG)]
    for number, drill in enumerate(["Baton Pass", "Visual Exchange"]):
        page = 41 + number
        rows += [
            (page, T, 2, drill),
            (page, N, 2, f"This drill builds the exchange. {LONG}"),
            (page, T, 1, generic),
            (page, L, 1, f"Look forward on the {drill.lower()} and keep the arms swinging."),
        ]
    return rows


def test_recurring_coaching_tips_belong_to_their_drill() -> None:
    chunks = _chunks(_doc(_drills(), 60))
    for drill in ("High-Knee Running", "Simulated Hurdling", "Baton Pass"):
        tips = _with(chunks, f"on the {drill.lower()}")
        assert f"> {drill}" in tips.section_path, tips.section_path
    # The drills are not nested under a "COACHING TIPS" section.
    assert not [c for c in chunks if "COACHING TIPS >" in c.section_path]


def test_a_heading_that_recurs_once_per_chapter_stays_a_section() -> None:
    rows: list[Row] = []
    for chapter in range(1, 7):
        page = chapter * 10
        rows += [
            (page, T, 1, f"Chapter {chapter} Event {chapter}"),
            (page, T, 2, f"Technique of event {chapter}"),
            (page, N, 2, LONG),
            (page + 1, T, 2, "Summary"),
            (page + 1, N, 2, f"{PROSE} Summary of chapter {chapter}."),
        ]
    summary = _with(_chunks(_doc(rows, 80)), "Summary of chapter 3")
    assert summary.section_path == "[Sprint Training > Chapter 3 Event 3 > Summary]"


def test_only_the_watermark_sentence_goes_and_the_text_after_it_stays() -> None:
    rows: list[Row] = [
        (
            3,
            N,
            1,
            "Measured drag fell with the rear fins. This file was downloaded from "
            "example-site.org. The athlete then ran the test again at full speed.",
        ),
    ]
    text = " ".join(c.content for c in _chunks(_doc(rows, 40)))
    assert "Measured drag fell" in text
    assert "The athlete then ran the test again" in text
    assert "example-site.org" not in text


# --- Front/back matter never parents a later heading ---------------------------------


def test_a_contents_heading_does_not_parent_the_chapters_after_it() -> None:
    """The parser put the front "CONTENTS" at depth 1 and every chapter at depth
    2: every chapter of the book was filed under "CONTENTS"."""
    rows: list[Row] = [
        (4, T, 1, "CONTENTS"),
        (4, N, 1, "Preface vii"),
        (19, T, 2, "Sport Biomechanics"),
        (19, N, 2, LONG + " Biomechanics chapter."),
        (37, T, 2, "Fundamental Terms and Concepts"),
        (37, N, 2, LONG + " Terms chapter."),
    ]
    chunks = _chunks(_doc(rows, 300))
    for marker, title in (
        ("Biomechanics chapter.", "Sport Biomechanics"),
        ("Terms chapter.", "Fundamental Terms and Concepts"),
    ):
        chunk = _with(chunks, marker)
        assert chunk.section_path == f"[Sprint Training > {title}]", chunk.section_path


def test_a_citation_piece_typed_as_a_title_stays_in_the_reference_list() -> None:
    rows: list[Row] = [
        (40, T, 1, "Chapter 18"),
        (40, N, 1, LONG),
        (41, T, 2, "References"),
        (41, L, 2, "Northrop, G.R. (1993) Mapping of tissue response. J Appl Mech 74, 532-537."),
        (41, T, 3, "Brook & Harbor, Madison."),
        (
            41,
            N,
            3,
            "Avery, R.M. (1988) Strength and its development. Sports Eng 6, 146-168.",
        ),
        (41, N, 3, "Society,"),
        (41, N, 3, "Ellis, E. (1957) Relation between size of cells. Lab Notes 126, 1345-1347."),
    ]
    chunks = _chunks(_doc(rows, 60))
    for text in ("Brook & Harbor, Madison.", "Avery, R.M.", "Society,", "Ellis, E."):
        chunk = _with(chunks, text)
        assert chunk.section_path.endswith("> References]"), chunk.section_path


def test_a_parent_intro_joins_its_merged_tiny_sections() -> None:
    rows: list[Row] = [
        (25, T, 1, "THE SHOT"),
        (25, N, 1, LONG),
        (26, T, 1, "KEYS TO GOOD THROWING"),
        (26, N, 1, "Before taking a detailed look at the event, consider its fundamentals."),
        (26, T, 2, "Balance"),
        (26, N, 2, "Stay balanced over the right leg in the glide."),
        (26, T, 2, "Rhythm"),
        (26, N, 2, "The rhythm of the throw comes with practice."),
    ]
    chunks = _chunks(_doc(rows, 60))
    intro = _with(chunks, "consider its fundamentals")
    assert "Stay balanced" in intro.content and "rhythm of the throw" in intro.content
    assert intro.section_path == "[Sprint Training > KEYS TO GOOD THROWING]"


def test_a_data_table_after_a_reference_list_is_the_chapters() -> None:
    rows: list[Row] = [
        (78, T, 1, "HURDLE MECHANICS"),
        (78, N, 1, LONG),
        (78, T, 2, "REFERENCES"),
        (78, L, 2, "Quill, V. (1979). Notes on an Imaginary Track. Fictional Sport 17, 35."),
        (78, L, 2, "Marlow, K. (1972). Beschreibung einer erfundenen Laufbahn. Leipzig."),
        (
            78,
            TB,
            2,
            "| Segment | Braking | Driving |\n|---|---|---|\n| Trail leg | 1.2 | 2.4 |\n"
            "| Lead leg | 0.8 | 1.9 |",
        ),
    ]
    chunks = _chunks(_doc(rows, 90))
    table = _with(chunks, "Trail leg")
    assert table.section_path == "[Sprint Training > HURDLE MECHANICS]", table.section_path
    assert _with(chunks, "Quill, V.").section_path.endswith("> REFERENCES]")


def test_a_dedication_after_the_imprint_is_not_filed_under_the_copyright_line() -> None:
    rows: list[Row] = [
        (3, T, 1, "COACHING RUNNING & JUMPING SUCCESSFULLY"),
        (6, T, 2, "Copyright 2003 by Jordan Avery"),
        (6, N, 2, "All rights reserved. Except for use in a review, no part may be copied."),
        (6, T, 2, "Library of Congress Cataloging-in-Publication Data"),
        (6, N, 2, "796.42--dc21"),
        (7, N, 2, "To my parents, for their love, kindness, discipline and support in all things."),
    ]
    chunk = _with(_chunks(_doc(rows, 220)), "To my parents")
    assert "Copyright" not in chunk.section_path, chunk.section_path


def test_a_standalone_line_before_a_heading_leads_its_text_not_its_path() -> None:
    rows: list[Row] = [
        (6, T, 2, "Printing history"),
        (6, N, 2, PROSE),
        (6, N, 2, "CIP AC"),
        (8, T, 1, "ABOUT THE AUTHOR"),
        (8, N, 1, "Sam Ellis is the author of over 70 books for children and young adults."),
    ]
    chunk = _with(_chunks(_doc(rows, 60)), "Sam Ellis is the author")
    assert chunk.section_path == "[Sprint Training > ABOUT THE AUTHOR]", chunk.section_path
    assert "CIP AC" in chunk.content


def test_a_foreword_paragraph_on_a_contents_page_is_not_contents() -> None:
    rows: list[Row] = [
        (7, T, 1, "Contents"),
        (
            7,
            N,
            1,
            "I am delighted that Pat Marlow, former international athlete and now chief "
            "coach for the hurdles, has written this book. " + PROSE,
        ),
        (
            7,
            TB,
            1,
            "| Chapter | Title | Page |\n|---|---|---|\n| 1 | Hurdling History | 2 |\n"
            "| 2 | Hurdling Skill | 8 |\n| 3 | Hurdling Mobility | 49 |",
        ),
        *_prose_pages(range(8, 10)),
    ]
    doc = _doc(rows, 120)
    assert contents_pages(doc) == frozenset({7})
    chunks = _chunks(doc)
    foreword = _with(chunks, "I am delighted")
    assert "Contents" not in foreword.section_path, foreword.section_path
    assert _with(chunks, "Hurdling Skill").section_path.endswith("> Contents]")


def test_a_table_opening_with_a_contents_row_gives_its_page_a_contents_context() -> None:
    rows: list[Row] = [
        (2, T, 1, "Copyright & Disclaimer"),
        (
            2,
            N,
            1,
            "Copyright 2016 by Example Tutorials. All the content is the property of the owner.",
        ),
        (
            3,
            TB,
            1,
            "| Table of Contents | 2 |\n|---|---|\n| About the Tutorial | 1 |\n"
            "| History of the Discus | 4 |\n| Discus Throwing Technique | 7 |",
        ),
        *_prose_pages(range(4, 6)),
    ]
    doc = _doc(rows, 30)
    assert contents_pages(doc) == frozenset({3})
    assert _with(_chunks(doc), "History of the Discus").section_path.endswith("> Contents]")


def test_a_foreword_after_a_contents_heading_on_an_undetected_page_is_not_contents() -> None:
    rows: list[Row] = [
        (7, T, 1, "Contents"),
        (
            7,
            N,
            1,
            "I am delighted that Pat Marlow, former international athlete and now chief "
            "coach for the hurdles, has written this book. " + PROSE,
        ),
        *_prose_pages(range(8, 10)),
    ]
    doc = _doc(rows, 120)
    assert contents_pages(doc) == frozenset()
    foreword = _with(_chunks(doc), "I am delighted")
    assert "Contents" not in foreword.section_path, foreword.section_path


def test_a_dot_leader_table_in_the_front_part_is_a_contents_page() -> None:
    rows: list[Row] = [
        (2, T, 1, "Copyright & Disclaimer"),
        (
            2,
            N,
            1,
            "Copyright 2016 by Example Tutorials. All the content is the property of the owner.",
        ),
        (
            3,
            TB,
            1,
            "| About the Tutorial ...... | 1 |\n|---|---|\n| History of the Discus ...... | 4 |\n"
            "| Discus Throwing Technique ...... | 7 |\n| Discus Training Drills ...... | 12 |",
        ),
        *_prose_pages(range(4, 6)),
    ]
    doc = _doc(rows, 30)
    assert contents_pages(doc) == frozenset({3})
    assert _with(_chunks(doc), "History of the Discus").section_path.endswith("> Contents]")


def test_a_contents_table_with_spanned_cells_repeated_is_a_contents_page() -> None:
    """docling repeats a spanned cell in both columns; each copy alone is an
    entry ("About the Tutorial ....... 1"), the doubled row is too long to be."""
    leader = "." * 60
    entries = (("About the Tutorial", 1), ("Discus History", 4), ("Throwing Technique", 7))
    table = "\n".join(
        [f"| {entries[0][0]} {leader} {entries[0][1]} | {entries[0][0]} {leader} {entries[0][1]} |"]
        + ["|---|---|"]
        + [f"| {title} {leader} {page} | {title} {leader} {page} |" for title, page in entries[1:]]
    )
    rows: list[Row] = [
        (3, T, 1, "Table of Contents"),
        (3, TB, 1, table),
        *_prose_pages(range(4, 6)),
    ]
    assert contents_pages(_doc(rows, 30)) == frozenset({3})


def test_a_running_line_standing_mid_page_is_kept() -> None:
    """ "Discus Throw" heads every page (a running head); on p. 12 the same words
    head a section inside the page and are kept."""
    rows: list[Row] = []
    for page in range(10, 20):
        rows.append((page, N, 1, "Discus Throw"))
        rows.append((page, N, 1, f"{PROSE} Page {page}."))
        if page == 12:
            rows.append((page, N, 1, "Discus Throw"))
            rows.append((page, N, 1, "The discus is thrown from a circle of 2.5 m. " + PROSE))
    text = " ".join(c.content for c in _chunks(_doc(rows, 20)))
    assert text.count("Discus Throw") == 1, text.count("Discus Throw")


def test_an_ocr_noise_line_never_labels_the_text_after_it() -> None:
    rows: list[Row] = [
        (214, T, 1, "Quick Order Form"),
        (214, N, 1, "Speed Skills for Young Athletes by Dr. Casey B. Marlow. " + LONG),
        (216, N, 1, "Vy.datt LGR"),
        (216, N, 1, "Dr. Marlow is an internationally recognized authority. " + LONG),
    ]
    chunks = _chunks(_doc(rows, 220))
    chunk = _with(chunks, "internationally recognized authority")
    assert "Vy.datt" not in chunk.section_path, chunk.section_path
    assert "Vy.datt LGR" in " ".join(c.content for c in chunks)


def test_contents_rows_with_very_long_dot_leaders_are_entries() -> None:
    leader = "." * 140
    table = "\n".join(
        [f"| About the Tutorial {leader}1 |", "|---|"]
        + [
            f"| {title} {leader}{page} |"
            for title, page in (("Discus History", 4), ("Throw Technique", 7))
        ]
    )
    rows: list[Row] = [(3, T, 1, "Contents"), (3, TB, 1, table), *_prose_pages(range(4, 6))]
    assert contents_pages(_doc(rows, 30)) == frozenset({3})

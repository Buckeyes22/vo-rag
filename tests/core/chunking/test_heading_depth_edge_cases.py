"""Heading and contents-page edge cases: rising-number lists, document-index labels,
numbered steps, running-head evidence and colon-ended labels."""

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
from vo_rag.core.parsing.contract import ElementType, PageQA, ParsedDocument, ParsedElement

T, N, L = ElementType.TITLE, ElementType.NARRATIVE_TEXT, ElementType.LIST_ITEM
Row = tuple[int, ElementType, int, str]

PROSE = (
    "Aerobic volume is built gradually across the season, and the coach raises it only "
    "when the athlete has absorbed the load of the weeks before it."
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


# --- a front-part list of rising numbers is not a contents page ------------


def _volume() -> ParsedDocument:
    rows: list[Row] = [(11, T, 1, "Chapter 1 Foundations")]
    rows += _prose_pages(range(11, 12))
    rows += [
        (12, T, 1, "Chapter 2 Building Volume"),
        (12, N, 1, PROSE),
        (12, L, 1, "Week 1 Tempo runs 100"),
        (12, L, 1, "Week 2 Tempo runs 120"),
        (12, L, 1, "Week 3 Tempo runs 140"),
        (12, L, 1, "Week 4 Tempo runs 160"),
    ]
    rows += _prose_pages(range(13, 15))
    return _doc(rows, 150)


def test_a_volume_progression_is_not_a_contents_page() -> None:
    doc = _volume()
    assert contents_pages(doc) == frozenset()
    chunks = _chunks(doc)
    for chunk in chunks:
        if "Week 1 Tempo" in chunk.content or "Page 13." in chunk.content:
            assert chunk.section_path.endswith("> Chapter 2 Building Volume]"), chunk.section_path
            assert chunk.metadata.get("front_matter_kind") != "toc"


def test_warm_up_minutes_are_not_a_contents_page_and_are_kept() -> None:
    rows: list[Row] = [
        (9, T, 1, "The Warm-up"),
        (9, L, 1, "1 Easy jog 10"),
        (9, L, 1, "2 Dynamic drills 15"),
        (9, L, 1, "3 Build-up strides 20"),
        *_prose_pages(range(10, 12)),
    ]
    doc = _doc(rows, 120)
    assert contents_pages(doc) == frozenset()
    assert any("Easy jog" in chunk.content for chunk in _chunks(doc))


def test_a_high_jump_record_progression_is_not_a_contents_page() -> None:
    rows: list[Row] = [
        (14, T, 1, "World records"),
        (14, L, 1, "Alex Rivera USA 215"),
        (14, L, 1, "Pat Marlow USSR 216"),
        (14, L, 1, "Sam Ellis USA 223"),
        (14, L, 1, "Jordan Avery USSR 228"),
        *_prose_pages(range(15, 17)),
    ]
    assert contents_pages(_doc(rows, 260)) == frozenset()


def test_a_body_page_after_a_contents_page_is_not_contents() -> None:
    """A chapter heading, prose and a list that follow the contents run: the
    page before being contents is context, not proof."""
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
    for chunk in _chunks(doc):
        if "Week 1 Tempo" in chunk.content:
            assert chunk.section_path == "[Sprint Training > Chapter 1 Foundations]"


# --- a docling document_index label needs a contents context ---------------


def test_a_document_index_label_in_the_back_of_the_book_is_not_contents() -> None:
    rows: list[Row] = [
        (238, T, 1, "Chapter 9 Recovery"),
        (238, N, 1, PROSE),
        (240, N, 1, "Ankle hops 12, 45"),
        (240, N, 1, "Block starts 88"),
    ]
    doc = _doc(rows, 250)
    doc.elements[2].metadata["docling_label"] = "document_index"
    assert contents_pages(doc) == frozenset()


# --- sentence-case numbered steps merge like all-caps ones -------------------


def test_sentence_case_numbered_steps_of_a_session_merge() -> None:
    rows: list[Row] = [
        (5, T, 1, "Chapter 2 Sessions"),
        (5, N, 1, PROSE),
        (5, T, 1, "Session Plan"),
        (5, T, 2, "1 Warm up"),
        (5, N, 2, "Jog 800 m easy."),
        (5, T, 2, "2 Mobility"),
        (5, N, 2, "Hips and ankles, ten reps each."),
        (5, T, 2, "3 Acceleration"),
        (5, N, 2, "Six 30 m builds from blocks."),
        (5, T, 2, "4 Cool down"),
        (5, N, 2, "Walk 400 m and stretch."),
    ]
    chunks = _chunks(_doc(rows, 40))
    (steps,) = [c for c in chunks if "Jog 800 m easy" in c.content]
    assert "Six 30 m builds" in steps.content
    assert "Walk 400 m" in steps.content


# --- running-head evidence is per run, and a plural is none -------------


def _routes(doc: ParsedDocument, prefix: str) -> dict[int, ElementRoute]:
    routing = RoutingContext.for_document(doc)
    return {
        e.page_number or 0: route_element(e, e.text, routing)
        for e in doc.elements
        if e.element_type is T and e.text.startswith(prefix)
    }


def test_a_heading_that_recurs_in_singular_and_plural_stays_a_heading() -> None:
    heads = {20: "Coaching Points", 22: "Coaching Points", 24: "Coaching Points"}
    heads[26] = "Coaching Point"
    rows: list[Row] = [(19, T, 1, "Chapter 3 Hurdling")]
    for page in range(19, 28):
        if page in heads:
            rows.append((page, T, 2, heads[page]))
        rows.append((page, N, 2, f"{PROSE} Page {page}."))
    routes = _routes(_doc(rows, 40), "Coaching")
    assert set(routes.values()) == {ElementRoute.UPDATE_TITLE}, routes


def test_a_garble_far_away_does_not_turn_real_headings_into_running_heads() -> None:
    heads = {10: "Hurdle Drills", 13: "Hurdle Skills", 16: "Hurdle Drills", 80: "Hurdle Driils"}
    rows: list[Row] = [(9, T, 1, "Part One Hurdles")]
    for page in list(range(9, 17)) + list(range(78, 83)):
        if page in heads:
            rows.append((page, T, 2, heads[page]))
        rows.append((page, N, 2, f"{PROSE} Page {page}."))
    routes = _routes(_doc(rows, 90), "Hurdle")
    assert routes[10] is ElementRoute.UPDATE_TITLE, routes
    assert routes[13] is ElementRoute.UPDATE_TITLE, routes
    assert routes[16] is ElementRoute.UPDATE_TITLE, routes


# --- a sentence-case label ending in a colon is a heading --------------------


@pytest.mark.parametrize(
    "text", ["Key points:", "Coaching points:", "Warm-up routine:", "Common faults to avoid:"]
)
def test_a_label_ending_in_a_colon_is_not_a_lead_in(text: str) -> None:
    assert not is_lead_in_line(text)


@pytest.mark.parametrize(
    "text", ["The constants are:", "Details from:", "The steps include:", "The variables are:"]
)
def test_a_sentence_ending_in_a_colon_is_a_lead_in(text: str) -> None:
    assert is_lead_in_line(text)


def test_a_sentence_case_sub_heading_with_a_colon_opens_a_section() -> None:
    rows: list[Row] = [
        (5, T, 1, "Chapter 4 Technique"),
        (5, N, 1, PROSE),
        (5, T, 2, "Coaching points:"),
        (5, L, 2, "Keep the hips tall through the drive phase of every rep."),
        (5, L, 2, "Land the foot under the centre of mass on every ground contact."),
        (5, N, 2, PROSE + " The list above applies to every rep of the session."),
    ]
    chunks = _chunks(_doc(rows, 40))
    (holding,) = [c for c in chunks if "hips tall" in c.content]
    assert "Coaching points:" in holding.section_path

"""Running heads the OCR spells differently on each page.

A children's sports book (OCR): the page header "ARENA AND FIELD: FIELD
EVENTS" is stored as a TITLE on ~15 pages as "ARENA AND FIELD: HELD EVENTS",
"ARENA AND HELD: HELD EVENTS", "ARENA AND FIELD: FIELD EUENTS" ..., so
exact-repeat detection missed it and every page break split a paragraph
("straightens and your feet point forward. This helps carry you up and over
the rail." became a chunk of its own).
"""

from __future__ import annotations

from vo_rag.core.chunking.hierarchical import (
    ElementRoute,
    HierarchicalChunker,
    RoutingContext,
    route_element,
)
from vo_rag.core.parsing.contract import ElementType, PageQA, ParsedDocument, ParsedElement

HEAD_VARIANTS = [
    "ARENA AND FIELD: FIELD EVENTS",
    "ARENA AND FIELD: HELD EVENTS",
    "ARENA AND HELD: HELD EVENTS",
    "ARENA AND FIELD: FIELD EUENTS",
]
FILLER = (
    "Young athletes learn the event step by step, practising the approach, the take-off "
    "and the landing until each part feels natural and relaxed. "
)
SPLIT_HEAD = "As you clear the rail your back arches and your body"
SPLIT_TAIL = "straightens and your feet point forward. This helps carry you up and over the rail."


def _el(eid: str, kind: ElementType, text: str, page: int, depth: int = 0) -> ParsedElement:
    return ParsedElement(
        element_id=eid, element_type=kind, text=text, page_number=page, section_depth=depth
    )


def _doc(elements: list[ParsedElement], pages: int, title: str) -> ParsedDocument:
    return ParsedDocument(
        document_id="d",
        source_id="s",
        title=title,
        total_pages=pages,
        elements=elements,
        images=[],
        page_qa=[
            PageQA(page_number=p, ocr_confidence=0.8, parser_used="docling")
            for p in range(1, pages + 1)
        ],
        markdown="",
        heading_tree=[],
        parser_used="docling",
    )


def _arena_book() -> ParsedDocument:
    pages = 20
    elements = [_el("t1", ElementType.TITLE, "ARENA AND FIELD: FIELD EVENTS", 1, 1)]
    elements.append(_el("hj", ElementType.TITLE, "The High Jump", 2, 1))
    elements.append(_el("b2", ElementType.NARRATIVE_TEXT, FILLER * 2, 2))
    for page in range(3, 18):
        head = HEAD_VARIANTS[page % len(HEAD_VARIANTS)]
        elements.append(_el(f"h{page}", ElementType.TITLE, head, page, 1))
        if page == 9:
            elements.append(_el("split-a", ElementType.NARRATIVE_TEXT, f"{FILLER}{SPLIT_HEAD}", 9))
            continue
        if page == 10:
            elements.append(_el("split-b", ElementType.NARRATIVE_TEXT, SPLIT_TAIL, 10))
            continue
        elements.append(_el(f"b{page}", ElementType.NARRATIVE_TEXT, f"{FILLER} Page {page}.", page))
    return _doc(elements, pages, "Arena and field: field events")


def test_ocr_variant_running_heads_are_skipped() -> None:
    doc = _arena_book()
    routing = RoutingContext.for_document(doc)
    heads = [e for e in doc.elements if e.element_id[0] == "h" and e.element_id[1:].isdigit()]
    assert len(heads) == 15
    assert heads
    assert all(route_element(e, e.text, routing) is ElementRoute.SKIP_REPEATED for e in heads), [
        (e.page_number, e.text) for e in heads if e.element_id not in routing.running_head_titles
    ]


def test_paragraph_continues_across_an_ocr_variant_running_head() -> None:
    chunks = HierarchicalChunker().chunk(_arena_book(), composites=[])

    holding = [c for c in chunks if SPLIT_TAIL in c.content]
    assert len(holding) == 1
    assert f"{SPLIT_HEAD}\n\n{SPLIT_TAIL}" in holding[0].content or (
        f"{SPLIT_HEAD} {SPLIT_TAIL}" in holding[0].content
    )
    assert holding[0].section_path == "[Arena and field: field events > The High Jump]"
    for chunk in chunks:
        assert "EVENTS" not in chunk.section_path and "EUENTS" not in chunk.section_path


def _real_sections(titles: list[str]) -> ParsedDocument:
    pages = 20
    elements: list[ParsedElement] = []
    for page in range(1, pages + 1):
        if page in (4, 9, 14):
            title = titles[(page // 5) % len(titles)]
            elements.append(_el(f"s{page}", ElementType.TITLE, title, page, 1))
        elements.append(_el(f"b{page}", ElementType.NARRATIVE_TEXT, f"{FILLER} Page {page}.", page))
    return _doc(elements, pages, "Athletics for Schools")


def test_repeated_real_section_heading_is_not_skipped() -> None:
    doc = _real_sections(["Jumping Events"])
    routing = RoutingContext.for_document(doc)
    titles = [e for e in doc.elements if e.element_type is ElementType.TITLE]
    assert all(route_element(e, e.text, routing) is ElementRoute.UPDATE_TITLE for e in titles)
    chunks = HierarchicalChunker().chunk(doc, composites=[])
    assert any(c.section_title == "Jumping Events" for c in chunks)


def test_distinct_page_top_headings_that_look_alike_are_not_skipped() -> None:
    for family in (
        ["Exercise A", "Exercise B", "Exercise C"],
        ["Warm Up Drills", "Warm Down Drills", "Cool Down Drills"],
        ["Jumping Events", "Throwing Events", "Running Events"],
    ):
        doc = _real_sections(family)
        routing = RoutingContext.for_document(doc)
        titles = [e for e in doc.elements if e.element_type is ElementType.TITLE]
        assert all(
            route_element(e, e.text, routing) is ElementRoute.UPDATE_TITLE for e in titles
        ), family


# --- OCR-variant grouping -------------------------------------------------


def _parts_with_block_starts(spellings: tuple[str, str, str]) -> ParsedDocument:
    """Three spellings of one heading: "Block Starts" p10, "Block Start" p21, "Block
    Starts" p31, each a depth-2 heading opening its page under a Part."""
    elements: list[ParsedElement] = []
    parts = {9: "Part One Sprints", 20: "Part Two Hurdles", 30: "Part Three Relays"}
    heads = dict(zip((10, 21, 31), spellings, strict=True))
    for page in range(1, 41):
        if page in parts:
            elements.append(_el(f"p{page}", ElementType.TITLE, parts[page], page, 1))
        if page in heads:
            elements.append(_el(f"h{page}", ElementType.TITLE, heads[page], page, 2))
        elements.append(_el(f"b{page}", ElementType.NARRATIVE_TEXT, f"{FILLER} Page {page}.", page))
    return _doc(elements, 40, "Athletics for Schools")


def test_singular_plural_headings_far_apart_are_not_running_heads() -> None:
    doc = _parts_with_block_starts(("Block Starts", "Block Start", "Block Starts"))
    routing = RoutingContext.for_document(doc)
    heads = [e for e in doc.elements if e.element_id.startswith("h")]
    assert all(route_element(e, e.text, routing) is ElementRoute.UPDATE_TITLE for e in heads)
    chunks = HierarchicalChunker().chunk(doc, composites=[])
    by_page = {c.page_number: c.section_path for c in chunks}
    assert by_page[10] == "[Athletics for Schools > Part One Sprints > Block Starts]"
    assert by_page[21] == "[Athletics for Schools > Part Two Hurdles > Block Start]"
    assert by_page[31] == "[Athletics for Schools > Part Three Relays > Block Starts]"


def _consecutive_heads(heads: list[str]) -> ParsedDocument:
    elements: list[ParsedElement] = []
    for page, head in enumerate(heads, start=2):
        elements.append(_el(f"h{page}", ElementType.TITLE, head, page, 1))
        elements.append(_el(f"b{page}", ElementType.NARRATIVE_TEXT, f"{FILLER} Page {page}.", page))
    return _doc(elements, len(heads) + 2, "Athletics for Schools")


def test_real_word_substitutions_are_not_an_ocr_variant_family() -> None:
    """Different real headings: "Sprint Drills"/"Sprint Skills", "Men Events"/"Women
    Events", "Power Clean"/"Power Cleans" are different real headings even on
    consecutive pages: no spelling carries an OCR-garbled word."""
    for pair in (
        ("Sprint Drills", "Sprint Skills"),
        ("Hurdle Drills", "Hurdle Skills"),
        ("Men Events", "Women Events"),
        ("Power Clean", "Power Cleans"),
        ("Long Jump", "Long Jumps"),
    ):
        doc = _consecutive_heads([pair[0], pair[1], pair[0], pair[1]])
        routing = RoutingContext.for_document(doc)
        heads = [e for e in doc.elements if e.element_id.startswith("h")]
        assert all(route_element(e, e.text, routing) is ElementRoute.UPDATE_TITLE for e in heads), (
            pair
        )


def test_ocr_variant_grouping_is_bounded_on_thousands_of_edge_titles() -> None:
    """Grouping must not be O(distinct edge titles^2): pairwise difflib over
    3,000 titles takes minutes. A crafted many-page PDF must not stall it."""
    import random
    import time

    rng = random.Random(1452)
    letters = "abcdefghijklmnopqrstuvwxyz"

    def word() -> str:
        return "".join(rng.choice(letters) for _ in range(rng.randint(5, 9)))

    shuffled = list("abcdefghijklmnopq")

    def anagram() -> str:
        # Same letters every time: passes difflib's cheap bounds, fails ratio().
        rng.shuffle(shuffled)
        return f"{''.join(shuffled[:8])} {''.join(shuffled[8:])}"

    pages = 1500
    for make in (lambda: f"{word()} {word()}", anagram):
        elements: list[ParsedElement] = []
        for page in range(1, pages + 1):
            elements.append(_el(f"a{page}", ElementType.TITLE, make(), page, 1))
            elements.append(_el(f"z{page}", ElementType.TITLE, make(), page, 1))
        doc = _doc(elements, pages, "Crafted")
        # CPU time of this process: stable when the suite runs in parallel. The
        # quadratic version took 137 s; ~1 s here, ~2 s on a loaded CI runner, so
        # 10 s keeps a wide margin both ways.
        started = time.process_time()
        RoutingContext.for_document(doc)
        assert time.process_time() - started < 10.0


def test_garbled_spellings_far_apart_are_not_a_running_head() -> None:
    """Even with an OCR-garbled spelling
    ("EUENTS"), headings 10+ pages apart are not a running head -- a running
    head sits on (nearly) every page."""
    doc = _parts_with_block_starts(("Field Events", "Field Euents", "Field Events"))
    routing = RoutingContext.for_document(doc)
    heads = [e for e in doc.elements if e.element_id.startswith("h")]
    assert all(route_element(e, e.text, routing) is ElementRoute.UPDATE_TITLE for e in heads)

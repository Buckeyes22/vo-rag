"""Doubled-page detection and page splicing for pypdfium parses (VOR-1453).

The doubled page is modelled on a real defect (a biomechanics monograph,
page 53) where docling's pypdfium backend emitted each
line followed by a cropped copy of a neighbouring line while pypdfium2's own
page text was clean.
"""

from __future__ import annotations

from vo_rag.core.parsing.contract import (
    ElementType,
    ExtractedImage,
    PageQA,
    ParsedDocument,
    ParsedElement,
)
from vo_rag.core.parsing.parsers.pdf_page_rebuild import (
    coverage_shortfall,
    doubled_pages,
    missing_words,
    plain_text_elements,
    rebuild_ranges,
    splice_pages,
    text_layer_pages,
)

CLEAN_PAGE = (
    "good runs the rotation is between 1.5 to 11.5 revolutions per\r\n"
    "second. (b) This is the only mount with recorded positive as well\r\n"
    "as negative rotations about the long axis. For good runs the\r\n"
    "rotations have ranged from 4.0 degrees per second with the front\r\n"
    "of the rotor turning up to 7.2 degrees per second with the\r\n"
    "front of the rotor turning down. (c) There is more sporadic yaw\r\n"
    "of the rotor to the left or right than with the other two mounts.\r\n"
)

# What the pypdfium backend stored for it: lines interleaved with cropped
# copies of their neighbours.
DOUBLED_PAGE = (
    "good runs the rotation is between 1.5 to 11.5 revolutions per "
    "second. (b) This is the only mount with recorded positive as well good runs the rotation i "
    "second. (b) This is the only mount with recorded positive as well as negative rotations "
    "about the long axis. For good runs the s per second with the fron as negative "
    "rotations about rotations have ranged from 4.0 degrees per second with the front "
    "er second with the rotations have rang of the rotor turning up to 7.2 degrees per "
    "second with the ng down. (c) There is more sporadic of the rotor turning up to 7.2 "
    "degrees front of the rotor turning down. (c) There is more sporadic yaw with the "
    "other two mounts. front of the rotor turning d of the rotor to the left or right "
    "than with the other two mounts."
)

# A training program: the same lines really do repeat, in the text layer too.
PROGRAM_PAGE = "\n".join(
    [
        "Week 1 Monday: w-up B; ext T; 6 x 60m acceleration runs from blocks",
        "Week 1 Wednesday: w-up B; ext T; 6 x 60m acceleration runs from blocks",
        "Week 1 Friday: w-up B; ext T; 6 x 60m acceleration runs from blocks",
        "Week 2 Monday: w-up B; ext T; 6 x 60m acceleration runs from blocks",
        "Week 2 Wednesday: w-up B; ext T; 6 x 60m acceleration runs from blocks",
        "Week 2 Friday: w-up B; ext T; 6 x 60m acceleration runs from blocks",
    ]
)


def _element(
    text: str,
    page: int,
    element_type: ElementType = ElementType.NARRATIVE_TEXT,
    **metadata: object,
) -> ParsedElement:
    return ParsedElement(
        element_id=f"{page}-{text[:12]}",
        element_type=element_type,
        text=text,
        page_number=page,
        metadata=dict(metadata),
    )


def test_doubled_page_is_flagged() -> None:
    flagged = doubled_pages([_element(DOUBLED_PAGE, 1)], [CLEAN_PAGE])

    assert list(flagged) == [1]
    assert flagged[1].extra_words >= 20


def test_clean_page_is_not_flagged() -> None:
    clean = " ".join(CLEAN_PAGE.split())

    assert doubled_pages([_element(clean, 1)], [CLEAN_PAGE]) == {}


def test_legitimately_repetitive_page_is_not_flagged() -> None:
    elements = [_element(line, 1, ElementType.LIST_ITEM) for line in PROGRAM_PAGE.splitlines()]

    assert doubled_pages(elements, [PROGRAM_PAGE]) == {}


def test_table_with_repeated_cells_is_not_measured() -> None:
    table = "| X | X | X |\n|---|---|---|\n| 10.2 | 10.2 | 10.2 |" * 20

    assert doubled_pages([_element(table, 1, ElementType.TABLE)], ["X 10.2"]) == {}


def test_paragraph_continued_on_the_next_page_is_not_doubling() -> None:
    # docling files a paragraph that runs over a page break under its first
    # page; the second page's text is part of that page's reference.
    first, second = CLEAN_PAGE[:200], CLEAN_PAGE[200:]
    element = _element(" ".join(CLEAN_PAGE.split()), 1)

    assert doubled_pages([element], [first, second]) == {}


def test_column_interleaved_text_layer_is_not_doubling() -> None:
    # pypdfium2 reads two columns line by line; the parse has them in order,
    # with a phrase each column repeats.
    left = ["group A carried out isokinetic exercises", "at an angular velocity of 60"]
    right = ["group B carried out isokinetic exercises", "at an angular velocity of 300"]
    layer = "\n".join(f"{a}   {b}" for a, b in zip(left, right, strict=True))
    elements = [_element(" ".join(left), 1), _element(" ".join(right), 1)]

    assert doubled_pages(elements, [layer]) == {}


def test_text_absent_from_the_text_layer_is_not_doubling() -> None:
    element = _element(DOUBLED_PAGE, 1)

    assert doubled_pages([element], ["an unrelated page about the discus"]) == {}


def test_text_layer_pages_and_coverage_shortfall() -> None:
    texts = ["", "x" * 10, CLEAN_PAGE, CLEAN_PAGE]

    assert text_layer_pages(texts) == {3, 4}
    assert coverage_shortfall({3, 4}, {3, 4}, 4) is None
    assert coverage_shortfall({3}, {3, 4, 5}, 5) == "below_text_layer"
    assert coverage_shortfall({1}, set(range(1, 27)), 26) == "page_coverage_gate"


def test_rebuild_ranges_cover_items_straddling_a_page_break() -> None:
    # page 5 is doubled; a paragraph runs 4 -> 5 and another 5 -> 6 -> 7.
    assert rebuild_ranges({5}, [(4, 5), (5, 7)]) == [(4, 7)]
    assert rebuild_ranges({5, 9}, []) == [(5, 5), (9, 9)]
    assert rebuild_ranges({5, 6}, [(1, 2)]) == [(5, 6)]
    assert rebuild_ranges(range(1, 11), [], max_pages=4) == [(1, 4), (5, 8), (9, 10)]


def test_missing_words_finds_a_dropped_word() -> None:
    reference = [
        "Starting with the 1961 Harbor Games rowing has become a way of\nmaking statements."
    ]
    rebuilt = ["Starting with the 1961 Harbor Games rowing has become a way making statements."]

    assert missing_words(reference, rebuilt) == {"of": 1}
    assert missing_words(reference, [reference[0].replace("\n", " ")]) == {}


def test_missing_words_ignores_pdfium_reading_order() -> None:
    # pypdfium2 runs two columns (and a figure label) line by line; the
    # rebuilt text has them in reading order. Nothing is missing.
    reference = ["Fast twitch fibres   Slow twitch fibres\nanaerobic work   aerobic work Fig. 39"]
    rebuilt = ["Fast twitch fibres anaerobic work", "Slow twitch fibres aerobic work", "Fig. 39"]

    assert missing_words(reference, rebuilt) == {}


def test_missing_words_accepts_words_split_or_joined_differently() -> None:
    reference = ["an intelli\r\ngent performer needs self discipline"]
    rebuilt = ["An intelligent performer needs self-discipline"]

    assert missing_words(reference, rebuilt) == {}


def test_missing_words_joins_hyphenated_line_breaks() -> None:
    reference = ["the effect on desmin was demon\ufffe\r\nstrated, as well as up- wards"]
    rebuilt = ["The effect on desmin was demonstrated, as well as upwards"]

    assert missing_words(reference, rebuilt) == {}


def test_plain_text_elements() -> None:
    [element] = plain_text_elements(7, "the loss of desmin stain￾\r\ning in both muscles\r\n")

    assert element.page_number == 7
    assert element.element_type == ElementType.NARRATIVE_TEXT
    assert element.text == "the loss of desmin staining in both muscles"
    assert plain_text_elements(7, " 12 \r\n") == []


def _image(page: int, image_id: str) -> ParsedElement:
    return ParsedElement(
        element_id=f"img-{image_id}",
        element_type=ElementType.IMAGE,
        page_number=page,
        metadata={"image_id": image_id, "image_ref": f"{'a' * 64}/p{page:03d}_img000.png"},
    )


def _document() -> ParsedDocument:
    elements = [
        ParsedElement(
            element_id="t1",
            element_type=ElementType.TITLE,
            text="Chapter 1 Mounts",
            page_number=1,
            section_depth=1,
            section_title="Chapter 1 Mounts",
        ),
        _element("Page one text.", 1),
        _element("Page two doubled text doubled text.", 2),
        _image(2, "i" * 64),
        _element("Page two more doubled.", 2),
        _element("Page four text.", 4),
    ]
    images = [
        ExtractedImage(
            image_id="i" * 64,
            page_number=2,
            ref=f"{'a' * 64}/p002_img000.png",
            element_index=3,
        )
    ]
    return ParsedDocument(
        document_id="a" * 64,
        source_id="book.pdf",
        title="Book",
        total_pages=4,
        elements=elements,
        images=images,
        page_qa=[
            PageQA(page_number=page, ocr_confidence=1.0, parser_used="docling")
            for page in (1, 2, 4)
        ],
        markdown="",
        heading_tree=[],
        parser_used="docling",
    )


def test_splice_replaces_a_page_and_keeps_its_figure() -> None:
    rebuilt_title = ParsedElement(
        element_id="x",
        element_type=ElementType.TITLE,
        text="The best mount",
        page_number=2,
        section_depth=2,
    )
    parsed = splice_pages(
        _document(),
        {2: [_element("Page two clean.", 2), rebuilt_title, _element("After the heading.", 2)]},
    )

    assert [(e.page_number, e.element_type.value, e.text) for e in parsed.elements] == [
        (1, "title", "Chapter 1 Mounts"),
        (1, "narrative_text", "Page one text."),
        (2, "narrative_text", "Page two clean."),
        (2, "image", ""),
        (2, "title", "The best mount"),
        (2, "narrative_text", "After the heading."),
        (4, "narrative_text", "Page four text."),
    ]
    # The figure keeps its record, now pointing at its new element position.
    assert parsed.images[0].element_index == 3
    assert parsed.elements[3].metadata["image_id"] == "i" * 64
    # Heading context is recomputed through the new page.
    assert parsed.elements[2].section_title == "Chapter 1 Mounts"
    assert parsed.elements[5].section_title == "The best mount"
    assert parsed.elements[6].section_title == "The best mount"
    assert parsed.elements[6].section_depth == 2
    assert [node["title"] for node in parsed.heading_tree] == ["Chapter 1 Mounts"]
    assert parsed.heading_tree[0]["children"][0]["title"] == "The best mount"
    # Page QA counts follow the new page; ids stay unique.
    qa = {q.page_number: q for q in parsed.page_qa}
    assert qa[2].text_element_count == 3
    assert len({e.element_id for e in parsed.elements}) == len(parsed.elements)
    assert "Page two clean." in parsed.markdown
    assert "doubled" not in parsed.markdown


def test_splice_fills_a_page_docling_failed_on() -> None:
    parsed = splice_pages(_document(), {3: [_element("Page three recovered.", 3)]})

    assert [e.page_number for e in parsed.elements] == [1, 1, 2, 2, 2, 3, 4]
    assert {q.page_number for q in parsed.page_qa} == {1, 2, 3, 4}
    assert parsed.images[0].element_index == 3


def test_splice_without_replacements_is_a_no_op() -> None:
    document = _document()

    assert splice_pages(document, {}) is document

"""VOR-1452 task 2: short fragments and small tables merge into a same-section neighbour.

Audited examples: "Try to hold each stretch for 8 seconds." (a unit-bearing
one-sentence leaf section) and "OFF DI 30/40/50 x2 w/ 3' & 5'^s skD60 x4 w/ 90s^"
(one linearized training-plan row group) were each emitted as a whole chunk.
"""

from __future__ import annotations

import re

from vo_rag.core.chunking.composite_emitter import CompositeChunk
from vo_rag.core.chunking.hierarchical import HierarchicalChunker, RetrievalChunk
from vo_rag.core.parsing.contract import ElementType, PageQA, ParsedDocument, ParsedElement

STRETCH = "Try to hold each stretch for 8 seconds."
ROW_GROUP = "OFF DI 30/40/50 x2 w/ 3' & 5'^s skD60 x4 w/ 90s^"

_SENTENCES = [
    "The sprinter keeps the hips tall and the trunk relaxed through the drive phase.",
    "Coaches watch the shin angle at touchdown to judge how well force is applied.",
    "Early acceleration rewards patience, long pushes and a gradual rise of the torso.",
    "Athletes who rush upright lose horizontal force and plateau before the tenth stride.",
    "Recovery between repetitions should be long enough to keep every rep at full quality.",
    "A short warm-up of skips, bounds and strides prepares the tendons for fast contacts.",
]


def _prose(sentences: int, offset: int = 0) -> str:
    return " ".join(_SENTENCES[(offset + i) % len(_SENTENCES)] for i in range(sentences))


def _title(eid: str, text: str, depth: int, page: int) -> ParsedElement:
    return ParsedElement(
        element_id=eid,
        element_type=ElementType.TITLE,
        text=text,
        page_number=page,
        section_depth=depth,
    )


def _text(eid: str, text: str, page: int) -> ParsedElement:
    return ParsedElement(
        element_id=eid,
        element_type=ElementType.NARRATIVE_TEXT,
        text=text,
        page_number=page,
    )


def _table(eid: str, text: str, page: int) -> ParsedElement:
    return ParsedElement(
        element_id=eid,
        element_type=ElementType.TABLE,
        text=text,
        page_number=page,
    )


def _doc(elements: list[ParsedElement], total_pages: int = 6) -> ParsedDocument:
    return ParsedDocument(
        document_id="d",
        source_id="s",
        title="Sprint Book",
        total_pages=total_pages,
        elements=elements,
        images=[],
        page_qa=[
            PageQA(page_number=p, ocr_confidence=0.9, parser_used="p")
            for p in range(1, total_pages + 1)
        ],
        markdown="",
        heading_tree=[],
        parser_used="p",
        quarantined_pages=[],
    )


def _holding(chunks: list[RetrievalChunk], needle: str) -> list[RetrievalChunk]:
    return [chunk for chunk in chunks if needle in chunk.content]


def test_unit_bearing_one_sentence_leaf_merges_with_its_neighbour() -> None:
    elements = [
        _title("c1", "Chapter 1 Flexibility", 1, 1),
        _title("s1", "Stretching Routine", 2, 1),
        _title("l1", "Quad Stretch", 3, 1),
        _text("p1", _prose(6), 1),
        _title("l2", "Hamstring Stretch", 3, 2),
        _text("p2", STRETCH, 2),
        _title("s2", "Warm-up Drills", 2, 3),
        _text("p3", _prose(6, 2), 3),
    ]
    chunks = HierarchicalChunker().chunk(_doc(elements), composites=[])

    holding = _holding(chunks, STRETCH)
    assert len(holding) == 1
    merged = holding[0]
    assert merged.content.strip() != STRETCH
    assert _SENTENCES[0] in merged.content  # joined the Quad Stretch leaf
    assert "Warm-up Drills" not in merged.section_path  # never the next depth-2 section
    assert "Hamstring Stretch" in merged.content  # the leaf heading travels with it
    assert merged.page_range == "1-2"
    assert merged.chunk_type == "text"


def test_one_row_group_table_merges_into_the_preceding_text() -> None:
    elements = [
        _title("c1", "Chapter 3 Training Plans", 1, 1),
        _title("s1", "Week One", 2, 1),
        _text("p1", _prose(5), 1),
        _table("t1", ROW_GROUP, 1),
        _title("s2", "Week Two", 2, 2),
        _text("p2", _prose(5, 3), 2),
    ]
    chunks = HierarchicalChunker().chunk(_doc(elements), composites=[])

    holding = _holding(chunks, ROW_GROUP)
    assert len(holding) == 1
    merged = holding[0]
    assert _SENTENCES[0] in merged.content
    assert merged.chunk_type == "text"
    assert merged.metadata.get("merged_tables") == 1
    assert merged.metadata.get("merged_table_element_ids") == ["t1"]
    assert "Week One" in merged.section_path
    assert not [chunk for chunk in chunks if chunk.chunk_type == "table"]


def test_leading_small_table_merges_into_the_following_text() -> None:
    elements = [
        _title("c1", "Chapter 3 Training Plans", 1, 1),
        _title("s1", "Week One", 2, 1),
        _table("t1", ROW_GROUP, 1),
        _text("p1", _prose(5), 1),
    ]
    chunks = HierarchicalChunker().chunk(_doc(elements), composites=[])

    holding = _holding(chunks, ROW_GROUP)
    assert len(holding) == 1
    assert holding[0].content.index(ROW_GROUP) < holding[0].content.index(_SENTENCES[0])
    assert holding[0].metadata.get("merged_tables") == 1


def test_merge_never_crosses_a_depth_two_boundary() -> None:
    elements = [
        _title("c1", "Chapter 1 Flexibility", 1, 1),
        _title("s1", "Stretching Routine", 2, 1),
        _text("p1", _prose(8), 1),
        _title("s2", "Hold Times", 2, 2),
        _text("p2", STRETCH, 2),
        _title("s3", "Warm-up Drills", 2, 3),
        _text("p3", _prose(8, 2), 3),
    ]
    chunks = HierarchicalChunker().chunk(_doc(elements), composites=[])

    holding = _holding(chunks, STRETCH)
    assert len(holding) == 1
    assert holding[0].content.strip() == STRETCH  # kept, alone: no same-section neighbour
    assert "Hold Times" in holding[0].section_path


def test_merge_never_crosses_a_depth_one_boundary_with_a_repeated_heading() -> None:
    elements = [
        _title("c1", "Workout", 1, 1),
        _text("p1", _prose(8), 1),
        _title("c2", "Workout", 1, 2),
        _table("t1", ROW_GROUP, 2),
        _title("c3", "Workout", 1, 3),
        _text("p3", _prose(8, 2), 3),
    ]
    chunks = HierarchicalChunker().chunk(_doc(elements), composites=[])

    holding = _holding(chunks, ROW_GROUP)
    assert len(holding) == 1
    assert holding[0].content.strip() == ROW_GROUP
    assert holding[0].chunk_type == "table"


def test_merge_never_crosses_a_back_matter_boundary() -> None:
    elements = [
        _title("c1", "Chapter 9 Racing", 1, 1),
        _title("s1", "Race Day", 2, 1),
        _title("l1", "Warm-up", 3, 1),
        _text("p1", _prose(4), 1),
        _title("l2", "References", 3, 2),
        _text("p2", "Smith, J. (2001). Sprint mechanics. J Sports Sci 19: 12-20.", 2),
    ]
    chunks = HierarchicalChunker().chunk(_doc(elements), composites=[])

    holding = _holding(chunks, "Sprint mechanics")
    assert len(holding) == 1
    assert _SENTENCES[0] not in holding[0].content
    assert "References" in holding[0].section_path


def test_lone_fragment_with_no_neighbour_is_kept() -> None:
    lone_text = HierarchicalChunker().chunk(
        _doc([_title("c1", "Stretching", 1, 1), _text("p1", STRETCH, 1)], total_pages=1),
        composites=[],
    )
    assert [chunk.content for chunk in lone_text] == [STRETCH]

    lone_table = HierarchicalChunker().chunk(
        _doc([_title("c1", "Week One", 1, 1), _table("t1", ROW_GROUP, 1)], total_pages=1),
        composites=[],
    )
    assert [(chunk.chunk_type, chunk.content) for chunk in lone_table] == [("table", ROW_GROUP)]


def test_hard_ceiling_respected_after_merges() -> None:
    chunker = HierarchicalChunker()
    elements: list[ParsedElement] = [_title("c1", "Chapter 1", 1, 1), _title("s1", "Plans", 2, 1)]
    for index in range(6):
        page = index + 1
        elements.append(_title(f"l{index}", f"Session {index + 1}", 3, page))
        elements.append(_text(f"p{index}", _prose(22, index), page))  # ~ 1,800 chars, near target
        elements.append(_table(f"t{index}", ROW_GROUP, page))
        elements.append(_text(f"q{index}", STRETCH, page))
    chunks = chunker.chunk(_doc(elements), composites=[])

    assert chunks
    assert max(chunk.token_count for chunk in chunks) <= chunker.max
    for chunk in chunks:
        assert chunk.token_count == chunker.count(chunk.embedding_text)
        assert chunk.embedding_text == f"{chunk.section_path}\n\n{chunk.content}"


_WORD_RE = re.compile(r"[^\s]+")


def _words(chunks: list[RetrievalChunk], drop_lines: set[str]) -> list[str]:
    words: list[str] = []
    for chunk in chunks:
        for line in chunk.content.splitlines():
            if line.strip() in drop_lines:
                continue
            words.extend(_WORD_RE.findall(line))
    return words


def test_total_text_is_preserved_by_merging() -> None:
    elements = [
        _title("c1", "Chapter 1 Flexibility", 1, 1),
        _title("s1", "Stretching Routine", 2, 1),
        _title("l1", "Quad Stretch", 3, 1),
        _text("p1", _prose(6), 1),
        _table("t1", ROW_GROUP, 1),
        _title("l2", "Hamstring Stretch", 3, 2),
        _text("p2", STRETCH, 2),
        _title("l3", "Calf Stretch", 3, 2),
        _text("p3", "Hold 20 seconds per side.", 2),
        _title("s2", "Warm-up Drills", 2, 3),
        _table("t2", "A-skip 2x20m", 3),
        _text("p4", _prose(6, 2), 3),
        _title("c2", "Chapter 2 Speed", 1, 4),
        _text("p5", _prose(3, 4), 4),
    ]
    titles = {el.text for el in elements if el.element_type is ElementType.TITLE}
    doc = _doc(elements)

    unmerged = HierarchicalChunker(min_=0, overlap=0).chunk(doc, composites=[])
    merged = HierarchicalChunker(overlap=0).chunk(doc, composites=[])

    assert len(merged) < len(unmerged)
    assert _words(merged, titles) == _words(unmerged, titles)


# Tiny sibling sections (OCR'd books tag day blocks as top-level
# headings) and the short-text path, which must keep text on covered pages.

DAY = "Jog one easy lap. Do calisthenics for 15 minutes. Walk one lap to cool down."
SHORT = "Smooth hands settle chassis balance."


def test_diamond_weekday_sibling_sections_merge_into_one_chunk() -> None:
    elements = [
        _title("c1", "Beginners Program", 1, 1),
        _text("p1", _prose(8), 1),
        _title("d1", "Monday", 1, 2),
        _text("m", DAY, 2),
        _title("d2", "Tuesday", 1, 2),
        _text("t", DAY.replace("one easy lap", "two easy laps"), 2),
        _title("d3", "Wednesday", 1, 3),
        _text("w", DAY.replace("15 minutes", "20 minutes"), 3),
    ]
    chunks = HierarchicalChunker().chunk(_doc(elements), composites=[])

    days = _holding(chunks, "Jog")
    assert len(days) == 1
    merged = days[0]
    # The group is labelled with the siblings' common parent
    # (the document title for top-level days), not "Monday" -- Tuesday's and
    # Wednesday's text is not Monday's. Every day's title leads its own text.
    assert merged.section_path == "[Sprint Book]"
    assert merged.metadata.get("merged_sections") == ["Monday", "Tuesday", "Wednesday"]
    assert merged.content.startswith("Monday\n")
    assert "two easy laps" in merged.content and "20 minutes" in merged.content
    assert "\nTuesday\n" in merged.content and "\nWednesday\n" in merged.content
    assert merged.page_range == "2-3"
    # The big chapter before them is never merged into.
    assert not [
        chunk for chunk in chunks if _SENTENCES[0] in chunk.content and "Jog" in chunk.content
    ]


def test_sibling_merge_needs_the_same_parent_section() -> None:
    elements = [
        _title("w1", "Week 1", 1, 1),
        _title("w1m", "Monday", 2, 1),
        _text("a", DAY, 1),
        _title("w1t", "Tuesday", 2, 1),
        _text("b", DAY.replace("one easy lap", "two easy laps"), 1),
        _title("w2", "Week 2", 1, 2),
        _title("w2m", "Monday", 2, 2),
        _text("c", DAY.replace("one easy lap", "three easy laps"), 2),
        _title("w2t", "Tuesday", 2, 2),
        _text("d", DAY.replace("one easy lap", "four easy laps"), 2),
    ]
    chunks = HierarchicalChunker().chunk(_doc(elements), composites=[])

    days = _holding(chunks, "Jog")
    assert len(days) == 2
    # Depth-2 day siblings merge under their week, labelled
    # with the week (their common parent), not with Monday.
    assert days[0].section_path == "[Sprint Book > Week 1]"
    assert "two easy laps" in days[0].content and "three easy laps" not in days[0].content
    assert days[1].section_path == "[Sprint Book > Week 2]"
    assert "four easy laps" in days[1].content


def test_lone_tiny_top_level_section_never_merges_into_a_large_chapter() -> None:
    elements = [
        _title("c1", "Chapter 1 Starts", 1, 1),
        _text("p1", _prose(8), 1),
        _title("n", "Key Points", 1, 2),
        _text("k", DAY, 2),
        _title("c2", "Chapter 2 Curves", 1, 3),
        _text("p2", _prose(8, 3), 3),
    ]
    chunks = HierarchicalChunker().chunk(_doc(elements), composites=[])

    holding = _holding(chunks, "Jog")
    assert len(holding) == 1
    assert holding[0].content.strip() == DAY
    assert holding[0].section_path == "[Sprint Book > Key Points]"
    assert "merged_sections" not in holding[0].metadata


def test_sibling_merge_never_crosses_a_back_matter_boundary() -> None:
    elements = [
        _title("d1", "Monday", 1, 1),
        _text("m", DAY, 1),
        _title("r", "References", 1, 1),
        _text("ref", "Smith, J. (2001). Sprint mechanics. J Sports Sci 19: 12-20.", 1),
    ]
    chunks = HierarchicalChunker().chunk(_doc(elements, total_pages=1), composites=[])

    assert len(_holding(chunks, "Jog")) == 1
    assert "Sprint mechanics" not in _holding(chunks, "Jog")[0].content


def test_sibling_groups_stay_under_target() -> None:
    chunker = HierarchicalChunker()
    elements: list[ParsedElement] = []
    for index in range(30):
        elements.append(_title(f"d{index}", f"Session {index + 1}", 1, 1 + index // 6))
        elements.append(_text(f"t{index}", f"{DAY} Repeat {index + 1}.", 1 + index // 6))
    chunks = chunker.chunk(_doc(elements), composites=[])

    assert 1 < len(chunks) < 30
    assert all(chunk.token_count < chunker.target for chunk in chunks)
    assert sum(chunk.content.count("Jog one easy lap") for chunk in chunks) == 30


def test_short_text_on_a_covered_page_is_merged_never_dropped() -> None:
    elements = [
        _text("p1", _prose(6), 1),
        _title("c2", "Covered Short Note", 1, 2),
        _text("s", SHORT, 2),
        _table("t2", "Gear 1 ratio 2.92\nGear 2 ratio 1.88", 2),
    ]
    chunker = HierarchicalChunker()
    chunks = chunker.chunk(_doc(elements, total_pages=2), composites=[])

    holding = _holding(chunks, SHORT)
    assert len(holding) == 1
    assert "Gear 1 ratio 2.92" in holding[0].content  # same-section neighbour
    assert "Covered Short Note" in holding[0].section_path
    assert all(drop.reason != "below_min_words" for drop in chunker.last_page_drop_ledger)


def test_short_text_with_no_same_section_neighbour_is_kept() -> None:
    elements = [
        _title("c1", "Chapter 1", 1, 1),
        _text("p1", _prose(8), 1),
        _title("c2", "Chapter 2", 1, 1),
        _text("s", SHORT, 1),
        _title("c3", "Chapter 3", 1, 1),
        _text("p3", _prose(8, 2), 1),
    ]
    chunker = HierarchicalChunker()
    chunks = chunker.chunk(_doc(elements, total_pages=1), composites=[])

    holding = _holding(chunks, SHORT)
    assert [chunk.content for chunk in holding] == [SHORT]
    assert "Chapter 2" in holding[0].section_path
    assert chunker.last_page_drop_ledger == []


# Heading-only chunks and index entries
# that the parser typed as TITLE.

PART_HEADING = "**part one:**\n\n**jumping out of the fire**"
PART_HEADING_TEXT = "jumping out of the fire"


def _figure(index: int, page: int) -> CompositeChunk:
    return CompositeChunk(
        chunk_id=f"fig-{index}",
        section_title="",
        preceding_text="",
        vlm_description="A box jump.",
        following_text="",
        image_id=f"img-{index}",
        image_ref=f"img-{index}.png",
        embedding_text="Figure: an athlete jumps onto a box.",
        token_count=10,
        page_number=page,
        source_id="s",
        element_index=index,
    )


def test_heading_only_chunk_before_a_figure_prefixes_the_next_text() -> None:
    elements = [
        _title("c0", "Introduction", 1, 1),
        _text("p1", _prose(8), 1),
        _text("h", PART_HEADING, 2),
        _title("c1", "Chapter 1 Landing Mechanics", 1, 3),
        _text("p2", _prose(8, 3), 3),
    ]
    # A figure closes the introduction text, the part heading follows on its own
    # and a second figure sits between it and the chapter.
    chunks = HierarchicalChunker().chunk(_doc(elements), composites=[_figure(2, 2), _figure(3, 2)])

    holding = _holding(chunks, PART_HEADING_TEXT)
    assert len(holding) == 1
    merged = holding[0]
    assert merged.chunk_type == "text"
    assert merged.content.index(PART_HEADING_TEXT) < merged.content.index(_SENTENCES[3])
    assert "Chapter 1 Landing Mechanics" in merged.section_path
    assert merged.content.startswith("**part one:**")  # never appended to the section before
    assert merged.metadata.get("merged_headings") == ["**part one:** **jumping out of the fire**"]
    assert [chunk.chunk_type for chunk in chunks].count("figure_composite") == 2


def test_heading_only_chunk_at_the_end_merges_backward() -> None:
    elements = [
        _title("c0", "Programs", 1, 1),
        _text("p1", _prose(8), 1),
        _text("h", "**part two:**\n\n**the programs**", 2),
    ]
    chunks = HierarchicalChunker().chunk(_doc(elements, total_pages=2), composites=[])

    holding = _holding(chunks, "the programs**")
    assert len(holding) == 1
    assert _SENTENCES[0] in holding[0].content


def test_lone_heading_only_chunk_is_kept() -> None:
    chunks = HierarchicalChunker().chunk(
        _doc([_text("h", PART_HEADING, 1)], total_pages=1), composites=[]
    )
    assert len(_holding(chunks, PART_HEADING_TEXT)) == 1


def test_genuine_short_heading_with_body_text_is_unchanged() -> None:
    elements = [
        _title("c1", "Chapter 2 Drills", 1, 1),
        _title("s1", "Depth Jumps", 2, 1),
        _text("p1", _prose(8), 1),
        _text("h", "Box Height", 2),  # a standalone (pending) heading line
        _text("p2", _prose(8, 2), 2),
        _title("s2", "Ankle Hops", 2, 3),
        _text("p3", _prose(8, 4), 3),
    ]
    doc = _doc(elements)
    merged = HierarchicalChunker().chunk(doc, composites=[])
    unmerged = HierarchicalChunker(min_=0).chunk(doc, composites=[])

    assert [(c.section_path, c.content) for c in merged] == [
        (c.section_path, c.content) for c in unmerged
    ]
    assert [c.section_title for c in merged] == ["Depth Jumps", "Box Height", "Ankle Hops"]


def _index_book() -> ParsedDocument:
    """Drill-manual shape: every index entry is a TITLE; the lines beside them
    carry no page numbers (index pages with numbers are dropped as index
    fragments by the low-signal filter, as before)."""
    elements = [
        _title("c1", "Chapter 1 Jumps", 1, 1),
        _text("p1", _prose(12), 1),
        _text("p2", _prose(12, 2), 2),
        _title("ix", "Index", 1, 5),
        _title("e1", "Ankle Hops", 2, 5),
        _text("e1b", "Australian football", 5),
        _title("e2", "Back Toss", 2, 5),
        _text("e2b", "basketball drills", 5),
        _title("e3", "Bicycling", 2, 6),
        _title("e4", "Depth Jumps", 2, 6),
        _title("e5", "Dots", 2, 6),
        _title("e6", "Double-Leg Hops", 2, 6),
    ]
    return _doc(elements, total_pages=6)


def test_index_entries_typed_as_titles_are_index_body_lines() -> None:
    chunks = HierarchicalChunker().chunk(_index_book(), composites=[])

    index_chunks = [chunk for chunk in chunks if "Ankle Hops" in chunk.content]
    assert len(index_chunks) == 1
    index_chunk = index_chunks[0]
    assert index_chunk.section_path == "[Sprint Book > Index]"
    for entry in ("Australian football", "Back Toss", "Bicycling", "Dots", "Double-Leg Hops"):
        assert entry in index_chunk.content
    assert not [c for c in chunks if "Ankle Hops" in c.section_path or "Dots" in c.section_path]


def test_prose_after_an_index_still_ends_it() -> None:
    doc = _index_book()
    doc.elements.append(_text("after", _prose(8, 1), 7))
    doc.total_pages = 7
    chunks = HierarchicalChunker().chunk(doc, composites=[])

    after = [c for c in chunks if c.content.startswith(_SENTENCES[1])]
    assert after and all("Index" not in c.section_path for c in after)


# The index is one TITLE
# ("index") and one NARRATIVE_TEXT element per entry; title-case entries must
# not be promoted to headings.

_DRILL_ENTRIES = [
    "Ab Dolly",
    "Alternating Lateral Lunges",
    "Ankle Hops",
    "Australian football",
    "Back Toss",
    "basketball",
    "Bicycling",
    "Box Jumps",
    "Depth Jumps",
    "Dots",
    "Double-Leg Hops",
    "football",
    "Hurdle Hops",
    "jump rope",
    "Lateral Bounds",
    "Medicine Ball Chest Pass",
    "Power Skips",
    "soccer",
    "Split Squat Jumps",
    "Squat Jumps",
    "tennis",
    "Tuck Jumps",
    "volleyball",
]


def _drill_index_book() -> ParsedDocument:
    elements = [
        _title("c1", "Chapter 1 Jumps", 1, 1),
        _text("p1", _prose(12), 1),
        _text("p2", _prose(12, 2), 2),
        _title("ix", "index", 1, 7),
    ]
    elements.extend(_text(f"e{n}", entry, 7) for n, entry in enumerate(_DRILL_ENTRIES))
    return _doc(elements, total_pages=7)


def test_title_case_index_lines_are_index_body_never_headings() -> None:
    chunks = HierarchicalChunker().chunk(_drill_index_book(), composites=[])

    entries = set(_DRILL_ENTRIES)
    for chunk in chunks:
        path_parts = chunk.section_path.strip("[]").split(" > ")
        assert not entries & set(path_parts), chunk.section_path
    index_chunks = [c for c in chunks if c.section_path == "[Sprint Book > index]"]
    # A few normal-size chunks labelled Index (or dropped by the index filter).
    assert len(index_chunks) <= 2, [c.content for c in index_chunks]
    for chunk in index_chunks:
        assert "Ankle Hops" in chunk.content or len(chunk.content) >= 100


# --- pass 3 never crosses a depth-1/2 boundary ---------


def test_tiny_chapters_are_never_merged_across_the_chapter_boundary() -> None:
    """Two "Chapter N" sections of one short paragraph each must not become one
    chunk labelled with Chapter 1, or Chapter 2's text would carry
    Chapter 1's section and chapter path."""
    elements = [
        _title("c1", "Chapter 1 The Sprint Start", 1, 3),
        _text("p1", DAY, 3),
        _title("c2", "Chapter 2 Hurdling", 1, 4),
        _text("p2", DAY.replace("one easy lap", "two easy laps"), 4),
    ]
    chunks = HierarchicalChunker().chunk(_doc(elements), composites=[])

    for chunk in _holding(chunks, "two easy laps"):
        assert chunk.section_path == "[Sprint Book > Chapter 2 Hurdling]", chunk.section_path
        assert chunk.chapter_path == "[Sprint Book > Chapter 2 Hurdling]", chunk.chapter_path
        assert "merged_sections" not in chunk.metadata
    for chunk in _holding(chunks, "one easy lap"):
        assert chunk.section_path == "[Sprint Book > Chapter 1 The Sprint Start]"


def test_merged_top_level_siblings_carry_no_one_siblings_label() -> None:
    """A group of tiny depth-1 siblings is labelled with their
    common parent (the document title), never with the first sibling's heading,
    and every sibling's title leads its own text."""
    elements = [
        _title("d1", "Monday", 1, 2),
        _text("m", DAY, 2),
        _title("d2", "Tuesday", 1, 2),
        _text("t", DAY.replace("one easy lap", "two easy laps"), 2),
    ]
    chunks = HierarchicalChunker().chunk(_doc(elements), composites=[])

    (merged,) = _holding(chunks, "Jog")
    assert merged.section_path == "[Sprint Book]"
    assert merged.section_title == "Sprint Book"
    assert merged.chapter_path == "[Sprint Book]"
    assert merged.embedding_text.startswith("[Sprint Book]\n\n")
    assert merged.metadata.get("merged_sections") == ["Monday", "Tuesday"]
    assert merged.content.startswith("Monday\n")
    assert "\nTuesday\n" in merged.content


def test_overlap_copy_is_dropped_when_the_first_chunk_breaks_its_lines() -> None:
    # VOR-1453: the overlap tail "... during a game The whole time ..." came
    # from text the first chunk holds as "... during a game\n\nThe whole
    # time ..."; the copy was kept and the merged chunk held it twice.
    from vo_rag.core.chunking.models import RetrievalChunk
    from vo_rag.core.chunking.short_text import ChunkScope, SmallChunkMerger

    def chunk(content: str) -> RetrievalChunk:
        return RetrievalChunk(
            chunk_id=content[:8],
            chunk_type="text",
            document_id="doc",
            source_id="source",
            section_path="[Book > Weeks 14-11]",
            section_title="Weeks 14-11",
            content=content,
            embedding_text=content,
            token_count=len(content.split()),
        )

    tail = "the legs more specifically to how they function during a game"
    body = "The whole time you are moving down the track you change your method of travel."
    first = chunk(f"We address the capacity of {tail}\n\n{body}")
    overlap = f"{tail} {body}"
    second = chunk(f"{overlap}\n\nAs for your strength improvements.")
    merger = SmallChunkMerger(
        counter=lambda text: len(text.split()), min_tokens=5, target=200, max_tokens=400
    )
    scopes = {
        id(first): ChunkScope(epoch=1),
        id(second): ChunkScope(epoch=1, overlap_prefix=overlap),
    }

    merged = merger._merged_content(first, second, scopes)

    assert merged.count("The whole time") == 1
    assert merged.endswith("As for your strength improvements.")

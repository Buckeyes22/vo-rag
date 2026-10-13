"""Tests for boilerplate-chunk detection and the ingest-time filter
(VOR-1427).

``BOILERPLATE_CHUNKS`` ships with only the known seed entry -- the full
reviewed set comes from a production corpus scan. ``do_build_chunks``'s
filter is exercised here with the fixtures ``test_chunk_embed_step.py``
already established (a parsed doc with real narrative text plus one
boilerplate-only element).
"""

from __future__ import annotations

import pytest

from vo_rag.core.chunking.composite_emitter import CompositeChunk
from vo_rag.core.parsing.contract import ElementType, ParsedDocument, ParsedElement
from vo_rag.pipelines.book_rag.boilerplate import (
    BOILERPLATE_CHUNKS,
    is_boilerplate,
    normalize_chunk_text,
)
from vo_rag.pipelines.book_rag.steps.chunk_embed import do_build_chunks


def test_normalization() -> None:
    assert normalize_chunk_text("  This page   intentionally LEFT blank. ") == (
        "this page intentionally left blank"
    )


def test_normalization_strips_punctuation_and_collapses_whitespace() -> None:
    assert normalize_chunk_text("Copyright © 2020.\n\nAll rights reserved!!") == (
        "copyright 2020 all rights reserved"
    )


@pytest.mark.parametrize("entry", sorted(BOILERPLATE_CHUNKS))
def test_known_boilerplate_detected(entry: str) -> None:
    """Every reviewed entry in `BOILERPLATE_CHUNKS` -- not just the one seed
    entry the set ships with -- must round-trip through `is_boilerplate`.
    Parametrized over the frozenset itself so a row added later (from a
    production scan) is covered automatically without anyone remembering to
    add a case for it."""
    assert is_boilerplate(entry)


def test_boilerplate_prefix_with_real_text_is_kept() -> None:
    assert not is_boilerplate(
        "This page intentionally left blank. Chapter 3 begins with load transfer under braking."
    )


def test_empty_and_non_matching_text_is_not_boilerplate() -> None:
    assert not is_boilerplate("")
    assert not is_boilerplate("A perfectly ordinary sentence about motorcycles.")


# ---------------------------------------------------------------------------
# do_build_chunks integration -- mirrors tests/pipelines/book_rag/test_chunk_embed_step.py's
# `_doc`/`_el` fixtures (this file owns its own copies; it does not import
# them, so the two files stay independent).
# ---------------------------------------------------------------------------


def _doc(
    *, elements: list[ParsedElement], title: str = "Test Book", total_pages: int = 1
) -> ParsedDocument:
    return ParsedDocument(
        document_id="doc-1",
        source_id="src-1",
        title=title,
        total_pages=total_pages,
        elements=elements,
        images=[],
        page_qa=[],
        markdown="",
        heading_tree=[],
        parser_used="test",
    )


def _el(text: str, etype: ElementType = ElementType.NARRATIVE_TEXT, page: int = 1) -> ParsedElement:
    return ParsedElement(
        element_id=f"el-{text[:20]}",
        element_type=etype,
        text=text,
        page_number=page,
    )


def _composite(element_index: int, *, chunk_id: str = "composite-1") -> CompositeChunk:
    return CompositeChunk(
        chunk_id=chunk_id,
        section_title="Chapter 1",
        preceding_text="",
        vlm_description="A diagram of the widget.",
        following_text="",
        image_id="img-1",
        image_ref="ref-1",
        embedding_text="A diagram of the widget.",
        token_count=5,
        page_number=2,
        source_id="src-1",
        element_index=element_index,
    )


def test_build_chunks_drops_boilerplate_only_chunk() -> None:
    doc = _doc(
        elements=[
            _el("Chapter 1", ElementType.TITLE, page=1),
            _el(
                "This is real narrative text that should survive the filter intact.",
                ElementType.NARRATIVE_TEXT,
                page=1,
            ),
            _el("This page intentionally left blank", ElementType.NARRATIVE_TEXT, page=2),
        ]
    )

    chunks = do_build_chunks(doc)

    assert chunks
    # This must be a literal substring check, not
    # `is_boilerplate(content)`. The chunker's packer can merge the real
    # narrative-text element and the boilerplate element (short, adjacent
    # pages) into ONE chunk -- "This is real narrative text...\n\nThis page
    # intentionally left blank" -- whose content is NOT an exact match for
    # any BOILERPLATE_CHUNKS entry, so `is_boilerplate()` on the merged
    # content returns False regardless of whether the real filtering
    # happened. Such a check would pass even with filtering fully disabled,
    # proving nothing about the merge case this fixture actually produces.
    for chunk in chunks:
        assert "intentionally left blank" not in chunk.content.lower()
    assert any("real narrative text" in chunk.content for chunk in chunks)


def test_build_chunks_all_boilerplate_raises_zero_chunks() -> None:
    """A document that is ENTIRELY boilerplate has nothing left to chunk once
    filtered -- the existing zero-chunk guard (not a new one) is what fires."""
    doc = _doc(
        elements=[
            _el("This page intentionally left blank", ElementType.NARRATIVE_TEXT, page=1),
        ]
    )

    with pytest.raises(RuntimeError, match="zero chunks"):
        do_build_chunks(doc)


def test_build_chunks_zero_chunks_error_reports_both_element_counts() -> None:
    """The error must report BOTH the pre-filter
    element count and the post-filter count, not just the pre-filter one --
    otherwise a doc that was entirely boilerplate looks identical (in the
    error) to a doc that genuinely parsed to zero elements."""
    doc = _doc(
        elements=[
            _el("This page intentionally left blank", ElementType.NARRATIVE_TEXT, page=1),
        ]
    )

    with pytest.raises(RuntimeError, match=r"parsed_elements=1.*after_boilerplate_filter=0"):
        do_build_chunks(doc)


# ---------------------------------------------------------------------------
# CompositeChunk.element_index remapping --
# composites (figure/table composites from the VLM sidecar /
# composite_emitter.py) are computed upstream, against the ORIGINAL element
# list, before this filter ever runs. HierarchicalChunker.chunk keys
# composites by their position in the list it's actually given, so removing
# elements here without remapping silently shifts a composite away from the
# text it belongs next to.
# ---------------------------------------------------------------------------


def test_composite_lands_adjacent_to_its_element_without_boilerplate() -> None:
    """Control: no boilerplate in the document at all -- the composite must
    land immediately before the text chunk its element_index points at."""
    doc = _doc(
        elements=[
            _el("Chapter 1", ElementType.TITLE, page=1),
            _el("Narrative A follows the diagram closely.", page=2),
        ]
    )
    composite = _composite(element_index=1)

    chunks = do_build_chunks(doc, composites=[composite])

    composite_pos = next(i for i, c in enumerate(chunks) if c.chunk_type == "figure_composite")
    text_pos = next(i for i, c in enumerate(chunks) if "Narrative A" in (c.content or ""))
    assert composite_pos < text_pos


def test_composite_stays_adjacent_after_a_boilerplate_element_before_it_is_removed() -> None:
    """A boilerplate element sits between the title and
    the composite's target text. `element_index=2` was computed upstream
    against the ORIGINAL 3-element list (narrative_A was originally at index
    2) -- without remapping, removing the boilerplate element at index 1
    shifts narrative_A to index 1 in the filtered list, so the untouched
    `element_index=2` would now point PAST narrative_A (at end-of-document),
    landing the composite after everything instead of before its own text.
    """
    doc = _doc(
        elements=[
            _el("Chapter 1", ElementType.TITLE, page=1),
            _el("This page intentionally left blank", page=1),
            _el("Narrative A follows the diagram closely.", page=2),
        ]
    )
    composite = _composite(element_index=2)

    chunks = do_build_chunks(doc, composites=[composite])

    composite_pos = next(i for i, c in enumerate(chunks) if c.chunk_type == "figure_composite")
    text_pos = next(i for i, c in enumerate(chunks) if "Narrative A" in (c.content or ""))
    assert composite_pos < text_pos, (
        "composite was not adjacent to its associated text -- landed at "
        f"position {composite_pos} vs text at {text_pos}"
    )


def test_composite_attaches_to_next_kept_element_when_its_own_element_is_boilerplate() -> None:
    """Deliberate choice (documented in do_build_chunks): a composite whose
    OWN associated element is the one dropped as boilerplate attaches to the
    NEXT surviving element instead of being silently lost -- a figure/table
    composite still carries real content (an image, a table) that must not
    disappear just because the text element it was originally paired with
    turned out to be a blank-page/copyright notice."""
    doc = _doc(
        elements=[
            _el("Chapter 1", ElementType.TITLE, page=1),
            _el("This page intentionally left blank", page=1),
            _el("Narrative A follows the diagram closely.", page=2),
        ]
    )
    # element_index=1 pointed at the BOILERPLATE element itself (the second
    # element, originally at index 1) -- not at narrative_A.
    composite = _composite(element_index=1)

    chunks = do_build_chunks(doc, composites=[composite])

    assert any(c.chunk_type == "figure_composite" for c in chunks)
    composite_pos = next(i for i, c in enumerate(chunks) if c.chunk_type == "figure_composite")
    text_pos = next(i for i, c in enumerate(chunks) if "Narrative A" in (c.content or ""))
    assert composite_pos < text_pos


def test_composite_attaches_to_end_of_document_when_nothing_survives_after_it() -> None:
    """The other half of the deliberate choice: if a composite's element
    (dropped as boilerplate) was the LAST element and nothing survives after
    it, the composite still isn't lost -- it lands at the end of the
    document (the same end-of-document slot HierarchicalChunker already
    uses for element_index == len(elements))."""
    doc = _doc(
        elements=[
            _el("Chapter 1", ElementType.TITLE, page=1),
            _el("Narrative A opens the chapter.", page=1),
            _el("This page intentionally left blank", page=2),
        ]
    )
    # element_index=2 pointed at the trailing boilerplate element itself.
    composite = _composite(element_index=2)

    chunks = do_build_chunks(doc, composites=[composite])

    assert any(c.chunk_type == "figure_composite" for c in chunks)


# ---------------------------------------------------------------------------
# The element filter touches body text only, the stored element_index stays
# the ORIGINAL one, and the visual sidecar chunk path filters too.
# ---------------------------------------------------------------------------


@pytest.fixture
def notes_is_boilerplate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Short common texts like "Notes"/"Index" count as boilerplate."""
    from vo_rag.pipelines.book_rag import boilerplate as boilerplate_module

    monkeypatch.setattr(
        boilerplate_module,
        "BOILERPLATE_CHUNKS",
        boilerplate_module.BOILERPLATE_CHUNKS | {"notes", "index"},
    )


@pytest.mark.usefixtures("notes_is_boilerplate")
def test_title_in_the_boilerplate_set_is_kept_and_titles_its_section() -> None:
    doc = _doc(
        elements=[
            _el("Chapter 1", ElementType.TITLE, page=1),
            _el("Brake bias shifts forward under hard deceleration in every car.", page=1),
            _el("Notes", ElementType.TITLE, page=2),
            _el("1. The torque figures come from the manufacturer's service manual.", page=2),
        ],
        total_pages=2,
    )

    chunks = do_build_chunks(doc)

    notes = [c for c in chunks if "torque figures" in c.content]
    assert notes, [c.content for c in chunks]
    assert notes[0].section_title == "Notes"


@pytest.mark.usefixtures("notes_is_boilerplate")
def test_only_body_elements_are_filtered() -> None:
    from vo_rag.pipelines.book_rag.boilerplate import filter_boilerplate_elements

    kept_types = [ElementType.TITLE, ElementType.TABLE, ElementType.HEADER, ElementType.FOOTER]
    # Every type the chunker appends as body text, plus list items.
    body_types = [
        ElementType.NARRATIVE_TEXT,
        ElementType.LIST_ITEM,
        ElementType.FIGURE_CAPTION,
        ElementType.EQUATION,
        ElementType.CODE_BLOCK,
    ]
    doc = _doc(
        elements=[
            # Lower-case: not a standalone heading, so body text for the chunker.
            ParsedElement(element_id=f"e{i}", element_type=etype, text="index", page_number=1)
            for i, etype in enumerate(kept_types + body_types)
        ]
    )

    filtered, _ = filter_boilerplate_elements(doc, [])

    assert [e.element_type for e in filtered.elements] == kept_types


def test_filterable_types_are_the_chunkers_body_types() -> None:
    from vo_rag.core.chunking.section_packer import APPENDED_TEXT_ELEMENT_TYPES
    from vo_rag.pipelines.book_rag.boilerplate import FILTERABLE_ELEMENT_TYPES

    body_types = {*APPENDED_TEXT_ELEMENT_TYPES, ElementType.LIST_ITEM}
    assert body_types == FILTERABLE_ELEMENT_TYPES


@pytest.mark.usefixtures("notes_is_boilerplate")
def test_narrative_text_heading_in_the_boilerplate_set_is_kept_and_starts_its_section() -> None:
    # Parsers often emit "Notes"/"Index"/"Bibliography" as NARRATIVE_TEXT; the
    # chunker turns such a standalone heading into the section title.
    doc = _doc(
        elements=[
            _el("Chapter 1", ElementType.TITLE, page=1),
            _el("Brake bias shifts forward under hard deceleration in every car.", page=1),
            _el("Notes", ElementType.NARRATIVE_TEXT, page=2),
            _el("1. The torque figures come from the manufacturer's service manual.", page=2),
        ],
        total_pages=2,
    )

    chunks = do_build_chunks(doc)

    notes = [c for c in chunks if "torque figures" in c.content]
    assert notes, [c.content for c in chunks]
    assert notes[0].section_title == "Notes"


def test_stored_element_index_is_the_original_parsed_index() -> None:
    doc = _doc(
        elements=[
            _el("Chapter 1", ElementType.TITLE, page=1),
            _el("This page intentionally left blank", page=1),
            _el("Narrative A follows the diagram closely.", page=2),
        ]
    )

    chunks = do_build_chunks(doc, composites=[_composite(element_index=2)])

    [figure] = [c for c in chunks if c.chunk_type == "figure_composite"]
    # parsed.json's element 2 is the figure's text, whatever the chunker saw.
    assert figure.metadata["element_index"] == 2
    composite_pos = chunks.index(figure)
    text_pos = next(i for i, c in enumerate(chunks) if "Narrative A" in (c.content or ""))
    assert composite_pos < text_pos


async def test_upserted_payload_keeps_the_original_element_index() -> None:
    from vo_rag.pipelines.book_rag.steps.chunk_embed import do_upsert_raw

    doc = _doc(
        elements=[
            _el("Chapter 1", ElementType.TITLE, page=1),
            _el("This page intentionally left blank", page=1),
            _el("Narrative A follows the diagram closely.", page=2),
        ]
    )
    chunks = do_build_chunks(doc, composites=[_composite(element_index=2)])

    class _Store:
        points: list[object] = []

        async def upsert_vectors(self, collection: str, points: list[object]) -> None:
            self.points = points

    store = _Store()
    dense = [[0.01] * 1024 for _ in chunks]
    colbert = [[[0.01] * 1024] for _ in chunks]
    await do_upsert_raw(
        chunks,
        {"dense": dense, "sparse": [{} for _ in chunks], "colbert": colbert},
        store,
        "src-1",
    )
    payloads = [p.payload for p in store.points]  # type: ignore[attr-defined]
    [figure] = [p for p in payloads if p["chunk_type"] == "figure_composite"]
    assert figure["metadata"]["element_index"] == 2


async def test_visual_sidecar_chunks_drop_boilerplate(monkeypatch: pytest.MonkeyPatch) -> None:
    from vo_rag.pipelines.book_rag import visual_sidecar

    doc = _doc(
        elements=[
            _el("Chapter 1", ElementType.TITLE, page=1),
            _el("This is real narrative text that should survive the filter intact.", page=1),
            _el("This page intentionally left blank", page=2),
        ],
        total_pages=2,
    )
    objects = {"parsed/env-1/parsed.json": doc.model_dump_json().encode("utf-8")}

    class _Garage:
        async def get_bytes(self, bucket: str, key: str) -> bytes:
            return objects[key]

        async def put_bytes(
            self, bucket: str, key: str, data: bytes, content_type: str | None = None
        ) -> None:
            objects[key] = data

    monkeypatch.setattr(visual_sidecar, "_garage", lambda: _Garage())

    chunks = await visual_sidecar.build_visual_chunks(
        envelope_id="env-1",
        parsed_garage_key="parsed/env-1/parsed.json",
        descriptions_garage_key=None,
    )

    assert any("real narrative text" in c.content for c in chunks)
    for chunk in chunks:
        assert "intentionally left blank" not in chunk.content.lower()


@pytest.mark.parametrize("parser_used", ["epub", "mobi"])
async def test_ebook_chunk_payload_carries_the_ebook_parser(parser_used: str) -> None:
    # Only the PDF parse path builds a forward path; an EPUB/MOBI payload used
    # to store parser_used=null, so health couldn't tell it from a legacy PDF.
    from vo_rag.pipelines.book_rag.steps.chunk_embed import do_upsert_raw

    doc = _doc(
        elements=[
            _el("Chapter 1", ElementType.TITLE, page=1),
            _el("Brake bias shifts forward under hard deceleration in every car.", page=1),
        ]
    ).model_copy(update={"parser_used": parser_used})
    chunks = do_build_chunks(doc)

    class _Store:
        points: list[object] = []

        async def upsert_vectors(self, collection: str, points: list[object]) -> None:
            self.points = points

    store = _Store()
    dense = [[0.01] * 1024 for _ in chunks]
    colbert = [[[0.01] * 1024] for _ in chunks]
    await do_upsert_raw(
        chunks,
        {"dense": dense, "sparse": [{} for _ in chunks], "colbert": colbert},
        store,
        "src-1",
    )
    assert {p.payload["parser_used"] for p in store.points} == {parser_used}  # type: ignore[attr-defined]

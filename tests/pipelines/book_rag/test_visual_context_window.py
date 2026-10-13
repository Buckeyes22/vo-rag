"""Tests for the figure-describe context window (VOR-303).

`extract_context_window` feeds the VLM describe prompt. Historically it only
pulled section_title + the single element immediately before/after the image and
never populated the caption. These tests pin the fixed behavior: pull the figure
caption, and widen the prose window to multiple elements with a char budget.
"""

from __future__ import annotations

from vo_rag.core.parsing.contract import (
    ElementType,
    ExtractedImage,
    ParsedDocument,
    ParsedElement,
)
from vo_rag.pipelines.book_rag.visual_sidecar import (
    _build_describe_prompt,
    _context_for_image,
    extract_context_window,
    native_pdf_image_context_spend_rejection_reason,
)


def _el(
    text: str,
    *,
    etype: ElementType = ElementType.NARRATIVE_TEXT,
    page: int = 1,
    section_title: str = "",
) -> ParsedElement:
    return ParsedElement(
        element_id=f"e-{abs(hash(text)) % 99999}",
        element_type=etype,
        text=text,
        page_number=page,
        section_title=section_title,
    )


def test_extract_context_window_pulls_caption_from_figure_caption_element() -> None:
    els = [
        _el("Earlier paragraph far above."),
        _el("The figure below shows the lift curve."),
        _el("", etype=ElementType.IMAGE),
        _el(
            "Figure 4.2: Lift coefficient versus angle of attack.", etype=ElementType.FIGURE_CAPTION
        ),
        _el("As the figure shows, lift increases linearly with angle."),
        _el("Further discussion continues on the next line."),
    ]
    cw = extract_context_window(els, 2)
    assert cw.caption == "Figure 4.2: Lift coefficient versus angle of attack."
    assert "lift curve" in cw.preceding_text
    assert "lift increases linearly" in cw.following_text
    # The caption element must not be duplicated into the following prose.
    assert "Figure 4.2" not in cw.following_text


def test_extract_context_window_widens_to_multiple_prose_elements() -> None:
    els = [
        _el("First far paragraph."),
        _el("Second nearer paragraph."),
        _el("", etype=ElementType.IMAGE),
        _el("First following paragraph."),
        _el("Second following paragraph."),
    ]
    cw = extract_context_window(els, 2)
    # Both preceding prose elements included, in reading order (far before near).
    assert "First far paragraph." in cw.preceding_text
    assert "Second nearer paragraph." in cw.preceding_text
    assert cw.preceding_text.index("First far") < cw.preceding_text.index("Second nearer")
    assert "First following paragraph." in cw.following_text
    assert "Second following paragraph." in cw.following_text


def test_extract_context_window_skips_non_prose_and_repeated_text() -> None:
    els = [
        _el("Running header", etype=ElementType.HEADER),
        _el("Real preceding prose."),
        _el("", etype=ElementType.IMAGE),
        _el("Page 12", etype=ElementType.FOOTER),
        _el("Real following prose."),
        _el("Repeated boilerplate."),
    ]
    cw = extract_context_window(els, 2, skip_texts={"Repeated boilerplate."})
    assert "Real preceding prose." in cw.preceding_text
    assert "Running header" not in cw.preceding_text
    assert "Real following prose." in cw.following_text
    assert "Page 12" not in cw.following_text
    assert "Repeated boilerplate." not in cw.following_text


def test_extract_context_window_respects_char_budget() -> None:
    big = "X" * 5000
    els = [_el(big), _el(big), _el("", etype=ElementType.IMAGE), _el(big)]
    cw = extract_context_window(els, 2)
    # Budget-bounded per side (not the full 5000-char element).
    assert 0 < len(cw.preceding_text) <= 700
    assert 0 < len(cw.following_text) <= 700


def test_build_describe_prompt_prefers_image_caption() -> None:
    els = [
        _el("Some context prose."),
        _el("", etype=ElementType.IMAGE),
        _el("Figure 1: caption from a parsed element.", etype=ElementType.FIGURE_CAPTION),
    ]
    parsed = ParsedDocument(
        document_id="d",
        source_id="s",
        title="Book",
        total_pages=1,
        elements=els,
        images=[],
        page_qa=[],
        markdown="",
        heading_tree=[],
        parser_used="x",
    )
    image = ExtractedImage(
        image_id="i",
        page_number=1,
        ref="s/p001_img001.png",
        element_index=1,
        caption="Explicit image caption from parser.",
    )
    cw = extract_context_window(els, 1)
    prompt = _build_describe_prompt(image=image, image_role="figure", parsed=parsed, context=cw)
    # image.caption wins over the figure_caption element.
    assert "Explicit image caption from parser." in prompt
    assert "Figure 1: caption from a parsed element." not in prompt


def test_build_describe_prompt_strips_meta_brackets() -> None:
    """Citation / editorial brackets must not reach the VLM prompt — smaller VLMs
    parrot them into the description (observed on an 8B run). Real prose is kept."""
    els = [
        _el("As shown in the data [58], the efficiency rises."),
        _el("", etype=ElementType.IMAGE),
        _el("Fig. 4-38: Slot resistance [corrected: was 4-37].", etype=ElementType.FIGURE_CAPTION),
    ]
    parsed = ParsedDocument(
        document_id="d",
        source_id="s",
        title="Book",
        total_pages=1,
        elements=els,
        images=[],
        page_qa=[],
        markdown="",
        heading_tree=[],
        parser_used="x",
    )
    image = ExtractedImage(image_id="i", page_number=1, ref="s/p001_img001.png", element_index=1)
    cw = extract_context_window(els, 1)
    prompt = _build_describe_prompt(image=image, image_role="figure", parsed=parsed, context=cw)
    assert "[58]" not in prompt
    assert "corrected" not in prompt  # the editorial note is gone
    assert "the efficiency rises" in prompt  # real prose kept
    assert "Slot resistance" in prompt  # real caption kept


def test_build_describe_prompt_strips_raw_svg_context() -> None:
    els = [
        _el(
            'Useful setup prose before the image. <svg width="100"><path d="M0 0"/></svg>',
        ),
        _el("", etype=ElementType.IMAGE),
        _el("Figure 2: Cornering-line diagram.", etype=ElementType.FIGURE_CAPTION),
    ]
    parsed = ParsedDocument(
        document_id="d",
        source_id="s",
        title="Book",
        total_pages=1,
        elements=els,
        images=[],
        page_qa=[],
        markdown="",
        heading_tree=[],
        parser_used="x",
    )
    image = ExtractedImage(image_id="i", page_number=1, ref="s/p001_img001.png", element_index=1)
    cw = extract_context_window(els, 1)
    prompt = _build_describe_prompt(image=image, image_role="figure", parsed=parsed, context=cw)
    assert "<svg" not in prompt
    assert "<path" not in prompt
    assert "Useful setup prose" in prompt
    assert "Return JSON only" in prompt


def test_context_for_image_adds_bounded_page_text_and_human_title() -> None:
    els = [
        _el("Trail braking and apex timing", etype=ElementType.TITLE, page=7),
        _el(
            "The exercise compares earlier braking with a later release point before the apex.",
            page=7,
        ),
        *[_el(f"Running header {idx}", etype=ElementType.HEADER, page=7) for idx in range(13)],
        _el("", etype=ElementType.IMAGE, page=7, section_title="Module II"),
    ]
    parsed = ParsedDocument(
        document_id="d",
        source_id="s",
        title="ab" * 32,
        total_pages=240,
        elements=els,
        images=[],
        page_qa=[],
        markdown="",
        heading_tree=[],
        metadata={},
        parser_used="x",
    )
    image = ExtractedImage(image_id="i", page_number=7, ref="s/p007_img001.png", element_index=15)

    cw = _context_for_image(
        parsed,
        image.element_index,
        document_metadata={"title": "THE CIRCUIT DRIVING BOOK Volume 1 - Car Handling"},
    )
    prompt = _build_describe_prompt(image=image, image_role="figure", parsed=parsed, context=cw)

    assert cw.section_title == "Trail braking and apex timing"
    assert "earlier braking" in cw.page_text
    assert "Book title: THE CIRCUIT DRIVING BOOK Volume 1 - Car Handling" in prompt
    assert "Page text:" in prompt
    assert "earlier braking" in prompt
    assert "ab" * 32 not in prompt


def test_context_spend_gate_uses_page_text_for_front_matter_rights() -> None:
    els = [
        _el("Copyright 2026. All rights reserved. No part of this book may be reproduced.", page=1),
        *[_el(f"Running header {idx}", etype=ElementType.HEADER, page=1) for idx in range(13)],
        _el("", etype=ElementType.IMAGE, page=1),
    ]
    parsed = ParsedDocument(
        document_id="d",
        source_id="s",
        title="Book",
        total_pages=20,
        elements=els,
        images=[],
        page_qa=[],
        markdown="",
        heading_tree=[],
        parser_used="x",
    )
    image = ExtractedImage(image_id="i", page_number=1, ref="s/p001_img001.png", element_index=14)
    cw = _context_for_image(parsed, image.element_index)

    assert cw.preceding_text == ""
    assert "All rights reserved" in cw.page_text
    assert (
        native_pdf_image_context_spend_rejection_reason(image=image, context=cw)
        == "front_matter_nontechnical_image"
    )


# ---------------------------------------------------------------------------
# VOR-1427: figure composites were titled "Visual" (37 image-heavy EPUBs had
# ~77% of their points in that one section) and some carried no text at all.
# ---------------------------------------------------------------------------


def _doc(
    elements: list[ParsedElement], images: list[ExtractedImage] | None = None
) -> ParsedDocument:
    return ParsedDocument(
        document_id="d",
        source_id="vo-rag-corpus-sha256:" + "ab" * 32,
        title="Field Guide to Sprint Mechanics",
        total_pages=1,
        elements=elements,
        images=images or [],
        page_qa=[],
        markdown="",
        heading_tree=[],
        metadata={},
        parser_used="epub",
    )


def test_context_for_image_finds_a_chapter_heading_far_back() -> None:
    """Equation-heavy EPUBs put hundreds of image elements between a heading and a
    figure; the heading must still be found (it was capped at 80 elements)."""
    els = [_el("Chapter 4: Pot Odds", etype=ElementType.TITLE)]
    for idx in range(300):
        els.append(_el(f"step {idx} of the derivation", page=1))
        els.append(_el("", etype=ElementType.IMAGE))
    parsed = _doc(els)

    cw = _context_for_image(parsed, len(els) - 1)

    assert cw.section_title == "Chapter 4: Pot Odds"


def test_composite_section_falls_back_to_the_book_title_never_visual() -> None:
    from vo_rag.pipelines.book_rag.visual_sidecar import (
        CompositeChunkBuilder,
        ContextWindow,
        VLMDescription,
    )

    image = ExtractedImage(image_id="i", page_number=1, ref="s/p001_img001.png", element_index=0)
    vlm = VLMDescription(description="", model_used="", token_count=0, duration_ms=0, image_id="i")
    builder = CompositeChunkBuilder()

    titled = builder.build(
        source_id="s",
        context=ContextWindow(following_text="x", document_title="Field Guide to Sprint Mechanics"),
        vlm=vlm,
        image=image,
        page_number=1,
        element_index=0,
    )
    untitled = builder.build(
        source_id="s",
        context=ContextWindow(following_text="x"),
        vlm=vlm,
        image=image,
        page_number=1,
        element_index=0,
    )

    assert titled.section_title == "Field Guide to Sprint Mechanics"
    assert untitled.section_title == "Document"  # never "Visual", never empty
    # No builder-side "Section:" line any more: the chunker prefixes the
    # section path it gives the surrounding text (VOR-1452).
    assert "Section:" not in untitled.embedding_text
    assert untitled.embedding_text == "Following: x"


async def test_extraction_composites_skip_images_with_no_text_around_them() -> None:
    from vo_rag.pipelines.book_rag.visual_sidecar import build_extraction_composites

    els = [
        _el("Chapter 1", etype=ElementType.TITLE),
        _el("A worked example follows.", page=1),
        _el("", etype=ElementType.IMAGE),
    ]
    with_text = ExtractedImage(image_id="a", page_number=1, ref="s/a.png", element_index=2)
    bare_els = [_el("", etype=ElementType.IMAGE), _el("", etype=ElementType.IMAGE)]
    bare = ExtractedImage(image_id="b", page_number=1, ref="s/b.png", element_index=1)

    kept = await build_extraction_composites(_doc(els, [with_text]))
    dropped = await build_extraction_composites(_doc(bare_els, [bare]))

    assert [c.image_id for c in kept] == ["a"]
    assert dropped == []


async def test_extraction_composites_collapse_a_run_of_image_tiles() -> None:
    """VOR-1441: a PDF-to-EPUB dump split one chart into 2,409 tile images with
    no text between them (Mathletics, `index-526_*.png`); each tile became a
    figure chunk with identical text, swamping its chapter. A run of images
    with the same context is one figure; a later image with its own text or
    caption is still its own chunk."""
    from vo_rag.pipelines.book_rag.visual_sidecar import build_extraction_composites

    els = [
        _el("Chapter 53", etype=ElementType.TITLE),
        _el("The shot chart below shows every attempt.", page=1),
        _el("", etype=ElementType.IMAGE),
        _el("", etype=ElementType.IMAGE),
        _el("", etype=ElementType.IMAGE),
        _el("A separate table of results follows.", page=1),
        _el("", etype=ElementType.IMAGE),
    ]
    images = [
        ExtractedImage(image_id=f"t{i}", page_number=1, ref=f"s/t{i}.png", element_index=i)
        for i in (2, 3, 4)
    ] + [
        ExtractedImage(
            image_id="captioned",
            page_number=1,
            ref="s/c.png",
            element_index=4,
            caption="Figure 53.2",
        ),
        ExtractedImage(image_id="later", page_number=1, ref="s/l.png", element_index=6),
    ]

    composites = await build_extraction_composites(_doc(els, images))

    assert [c.image_id for c in composites] == ["t2", "captioned", "later"]


def test_page_context_never_carries_a_bare_label() -> None:
    from vo_rag.pipelines.book_rag.visual_sidecar import _page_text_context

    els = [_el("", etype=ElementType.IMAGE, page=12), _el("12", page=12)]

    text = _page_text_context(els, 0, skip_texts=set())

    # A lone printed page number is furniture: stripped before labelling, so no
    # "Later on page: 12" and no bare "Later on page:" either.
    assert "12" not in text
    assert "Later on page:" not in text

"""Figure chunks store the figure's own text, labelled like the text around them
(VOR-1452).

A ``figure_composite`` chunk's stored ``content`` must not be the whole
embedding block -- "Section: ... Preceding: ... Following: ..." -- otherwise
the prose around a figure is indexed twice and a tagger tags the borrowed
text. With the VLM off (production) the chunk still carries the figure's own
text. And in the stored (pre-VOR-1417) element order every picture sits after
the last text element, so all figures of a children's "field events" book
would be labelled with the book's last heading, "Titles in This Series"; the
chunker places them in page order instead.

These tests run the real path: ``build_extraction_composites`` (the VLM-off
builder) and ``chunk_without_boilerplate`` (the chunker the pipeline calls).
"""

from __future__ import annotations

from vo_rag.core.chunking.composite_emitter import CompositeChunk, CompositeEmitter
from vo_rag.core.chunking.hierarchical import HierarchicalChunker
from vo_rag.core.chunking.models import RetrievalChunk
from vo_rag.core.parsing.contract import (
    ElementType,
    ExtractedImage,
    PageQA,
    ParsedDocument,
    ParsedElement,
)
from vo_rag.pipelines.book_rag.boilerplate import chunk_without_boilerplate
from vo_rag.pipelines.book_rag.visual_sidecar import (
    PAGE_CONTEXT_EXCERPT_CHARS,
    build_extraction_composites,
)

TITLE = "Track and Field: Field Events"
HIGH_JUMP = (
    "In the flop the athlete runs a curved approach, leans away from the bar on the last "
    "strides and drives the knee up at take-off so that the body rotates over the bar."
)
SHOT_PUT = (
    "The glide starts with the athlete facing the back of the circle, low over the right "
    "leg, and ends with the hips driving the chest round before the arm strikes."
)
SERIES = (
    "Practical Coaching Series, Volume 2: Sprint Mechanics.",
    "Practical Coaching Series, Volume 3: Middle Distance.",
)


class _Book:
    """Elements plus images, built in reading order unless placed explicitly."""

    def __init__(self) -> None:
        self.elements: list[ParsedElement] = []
        self.images: list[ExtractedImage] = []

    def add(self, kind: ElementType, text: str, page: int, depth: int = 1) -> int:
        index = len(self.elements)
        self.elements.append(
            ParsedElement(
                element_id=f"e{index}",
                element_type=kind,
                text=text,
                page_number=page,
                section_depth=depth,
                # The parser's own heading stack: what the old builder used.
                section_title="",
            )
        )
        return index

    def title(self, text: str, page: int, depth: int = 1) -> None:
        self.add(ElementType.TITLE, text, page, depth)

    def body(self, text: str, page: int) -> None:
        self.add(ElementType.NARRATIVE_TEXT, text, page)

    def caption(self, text: str, page: int) -> None:
        self.add(ElementType.FIGURE_CAPTION, text, page)

    def image(self, page: int, *, name: str, caption: str | None = None) -> None:
        index = self.add(ElementType.IMAGE, "", page)
        self.images.append(
            ExtractedImage(
                image_id=f"img-{name}",
                page_number=page,
                image_index=len(self.images),
                ref=f"src/{name}.png",
                caption=caption,
                element_index=index,
            )
        )

    def doc(self, total_pages: int) -> ParsedDocument:
        return ParsedDocument(
            document_id="d",
            source_id="s",
            title=TITLE,
            total_pages=total_pages,
            elements=self.elements,
            images=self.images,
            page_qa=[
                PageQA(page_number=p, ocr_confidence=1.0, parser_used="docling")
                for p in range(1, total_pages + 1)
            ],
            markdown="",
            heading_tree=[],
            parser_used="docling",
        )


async def _chunks(doc: ParsedDocument) -> list[RetrievalChunk]:
    composites = await build_extraction_composites(doc)
    chunks, _ = chunk_without_boilerplate(HierarchicalChunker(), doc, composites)
    return chunks


def _figures(chunks: list[RetrievalChunk]) -> list[RetrievalChunk]:
    return [c for c in chunks if c.chunk_type == "figure_composite"]


def _chapter_with_figure() -> _Book:
    book = _Book()
    book.title("Chapter 3 The High Jump", 10)
    book.body(f"{HIGH_JUMP} Approach run on page 9 of the drills.", 9)
    for page in range(10, 13):
        book.body(f"{HIGH_JUMP} Page {page} notes on the curve.", page)
    book.body("Figure 3.1 shows the athlete at the moment of take-off.", 12)
    book.image(12, name="takeoff")
    book.caption("Figure 3.1 Take-off in the flop.", 12)
    book.body(f"{HIGH_JUMP} After take-off on page 12 the arms lead.", 12)
    for page in range(13, 15):
        book.body(f"{HIGH_JUMP} Page {page} notes on the bar clearance.", page)
    return book


async def test_figure_content_has_no_neighbour_blocks() -> None:
    chunks = await _chunks(_chapter_with_figure().doc(20))
    [figure] = _figures(chunks)

    assert "Preceding:" not in figure.content
    assert "Following:" not in figure.content
    assert "Section:" not in figure.content
    assert figure.content.startswith("Caption: Figure 3.1 Take-off in the flop.")
    assert "Page context:" in figure.content
    assert "moment of take-off" in figure.content
    assert figure.image_ref == "src/takeoff.png"
    assert figure.image_id == "img-takeoff"
    assert figure.has_image is True


async def test_figure_embedding_text_keeps_the_neighbour_window() -> None:
    chunks = await _chunks(_chapter_with_figure().doc(20))
    [figure] = _figures(chunks)

    assert figure.embedding_text.startswith(f"[{TITLE} > Chapter 3 The High Jump]")
    assert "Preceding:" in figure.embedding_text
    assert "Following:" in figure.embedding_text
    assert "Caption: Figure 3.1 Take-off in the flop." in figure.embedding_text
    assert "the arms lead" in figure.embedding_text
    assert figure.content != figure.embedding_text
    assert figure.token_count >= len(figure.embedding_text) // 4 - 1


async def test_figure_page_context_is_bounded_and_nearest_the_figure() -> None:
    book = _Book()
    book.title("Chapter 5 The Shot Put", 1)
    # A long page: the excerpt must keep the sentences next to the figure.
    far = " ".join(f"Far sentence {n} about the circle and the stop board." for n in range(30))
    book.body(far, 3)
    book.body("The next picture shows the power position.", 3)
    book.image(3, name="power", caption="Figure 5.2 The power position.")
    book.body("Here the left arm is long and the chin is over the right knee.", 3)
    book.body(" ".join(f"Later sentence {n} on the release angle." for n in range(30)), 3)
    book.body(f"{SHOT_PUT} Page 4 notes.", 4)
    chunks = await _chunks(book.doc(6))
    [figure] = _figures(chunks)

    context = figure.content.split("Page context: ", 1)[1]
    assert len(context) <= PAGE_CONTEXT_EXCERPT_CHARS
    assert "shows the power position" in context
    assert "left arm is long" in context
    assert "Far sentence 0 " not in context
    assert figure.content.startswith("Caption: Figure 5.2 The power position.")
    # The full page window still feeds the embedding.
    assert "Far sentence 29" in figure.embedding_text


async def test_figure_with_no_caption_and_no_page_text_keeps_a_minimal_chunk() -> None:
    """Empty-figure policy: keep the chunk, never with empty content. The
    figure gallery (``/api/v1/artifacts/{id}/images``) and query hits only know
    a figure through a chunk's ``image_ref``; the neighbour window still gives
    the chunk its recall through ``embedding_text``."""
    book = _Book()
    book.title("Chapter 3 The High Jump", 1)
    book.body(f"{HIGH_JUMP} Page 1.", 1)
    book.image(2, name="lonely")  # nothing else on page 2, no caption
    book.body(f"{HIGH_JUMP} Page 3.", 3)
    chunks = await _chunks(book.doc(4))
    [figure] = _figures(chunks)

    assert figure.content == "Figure on page 2 (Chapter 3 The High Jump)"
    assert figure.image_ref == "src/lonely.png"
    assert "Preceding:" in figure.embedding_text


def test_emitter_never_emits_empty_figure_content() -> None:
    composite = CompositeChunk(
        chunk_id="c",
        section_title="Builder Section",
        preceding_text="neighbour text before",
        vlm_description="",
        following_text="neighbour text after",
        image_id="img-1",
        image_ref="src/1.png",
        embedding_text="Preceding: neighbour text before\n\nFollowing: neighbour text after",
        token_count=10,
        page_number=7,
        source_id="s",
        element_index=3,
    )
    chunk = CompositeEmitter(counter=lambda text: len(text.split())).emit_composite(
        composite,
        document_id="d",
        source_id="s",
        section_path="[Book > Throws]",
        section_title="Throws",
        doc_title="Book",
    )
    assert chunk.content == "Figure on page 7 (Throws)"
    assert chunk.embedding_text.startswith("[Book > Throws]\n\n")
    assert "neighbour text before" in chunk.embedding_text


def test_emitter_content_carries_the_vlm_description() -> None:
    composite = CompositeChunk(
        chunk_id="c",
        section_title="",
        preceding_text="before",
        vlm_description="A hurdler clearing the barrier with the lead leg extended.",
        following_text="after",
        caption="Figure 2 Lead leg.",
        page_context="The lead leg attacks the hurdle.",
        image_id="img-1",
        image_ref="src/1.png",
        embedding_text="Caption: Figure 2 Lead leg.\n\nPreceding: before\n\nFollowing: after",
        token_count=0,
        page_number=2,
        source_id="s",
        element_index=1,
    )
    chunk = CompositeEmitter(counter=lambda text: len(text.split())).emit_composite(
        composite,
        document_id="d",
        source_id="s",
        section_path="[Book > Hurdles]",
        section_title="Hurdles",
        doc_title="Book",
    )
    assert chunk.content == (
        "Caption: Figure 2 Lead leg.\n\n"
        "Page context: The lead leg attacks the hurdle.\n\n"
        "Visual: A hurdler clearing the barrier with the lead leg extended."
    )
    assert "Preceding: before" in chunk.embedding_text
    assert "Following: after" in chunk.embedding_text


async def test_figure_inside_a_normal_chapter_keeps_the_chapter_label() -> None:
    chunks = await _chunks(_chapter_with_figure().doc(20))
    [figure] = _figures(chunks)
    text_on_page = [c for c in chunks if c.chunk_type == "text" and c.page_number == 12]

    assert figure.section_path == f"[{TITLE} > Chapter 3 The High Jump]"
    assert figure.section_title == "Chapter 3 The High Jump"
    assert text_on_page and all(c.section_path == figure.section_path for c in text_on_page)


async def test_figure_after_back_matter_on_a_later_page_takes_the_body_section() -> None:
    """References close on page 11; page 12 opens with a figure before any
    prose. The prose after it returns to the chapter, and so must the figure."""
    book = _Book()
    book.title("Chapter 4 The Shot Put", 5)
    for page in range(5, 11):
        book.body(f"{SHOT_PUT} Page {page}.", page)
    book.title("References", 11, depth=2)
    book.body(
        "Ellis, J. G. (1991). A practical guide to rotational throwing drills. 4th ed. "
        "Springfield, ST: Northwind Books.",
        11,
    )
    book.image(12, name="glide", caption="Figure 4.3 The glide.")
    for page in range(12, 15):
        book.body(f"{SHOT_PUT} Drill on page {page}.", page)
    chunks = await _chunks(book.doc(16))
    [figure] = _figures(chunks)

    assert "References" not in figure.section_path, figure.section_path
    assert figure.section_path == f"[{TITLE} > Chapter 4 The Shot Put]"
    later_text = [c for c in chunks if c.chunk_type == "text" and (c.page_number or 0) >= 12]
    assert later_text and all(c.section_path == figure.section_path for c in later_text)


def _stored_order_book() -> _Book:
    """Children's-book-shaped: pictures appended after the last text element (the
    stored pre-VOR-1417 order), and the book ends with "Titles in This Series"."""
    book = _Book()
    book.title("The High Jump", 2)
    for page in range(2, 9):
        book.body(f"{HIGH_JUMP} Page {page}.", page)
    book.title("The Shot Put", 9)
    for page in range(9, 15):
        book.body(f"{SHOT_PUT} Page {page}.", page)
    book.title("Titles in This Series", 15)
    for line in SERIES:
        book.body(line, 15)
    book.image(4, name="flop")
    book.image(11, name="glide")
    return book


async def test_stored_order_figures_are_labelled_by_their_page_not_the_last_heading() -> None:
    chunks = await _chunks(_stored_order_book().doc(16))
    figures = {c.image_id: c for c in _figures(chunks)}

    assert set(figures) == {"img-flop", "img-glide"}
    for figure in figures.values():
        assert "Titles in This Series" not in figure.section_path
        assert "Titles in This Series" not in figure.embedding_text
    assert figures["img-flop"].section_path == f"[{TITLE} > The High Jump]"
    assert figures["img-glide"].section_path == f"[{TITLE} > The Shot Put]"
    # Its neighbour window is its own page's prose, not the back of the book.
    assert "Page 4." in figures["img-flop"].embedding_text
    assert "Athletics Coaching Series" not in figures["img-flop"].embedding_text


def _page_order_book() -> _Book:
    """``_stored_order_book`` with each picture at the end of its page's text."""
    book = _Book()
    book.title("The High Jump", 2)
    for page in range(2, 9):
        book.body(f"{HIGH_JUMP} Page {page}.", page)
        if page == 4:
            book.image(4, name="flop")
    book.title("The Shot Put", 9)
    for page in range(9, 15):
        book.body(f"{SHOT_PUT} Page {page}.", page)
        if page == 11:
            book.image(11, name="glide")
    book.title("Titles in This Series", 15)
    for line in SERIES:
        book.body(line, 15)
    return book


async def test_stored_order_figures_chunk_as_in_page_order() -> None:
    """A parse stored before VOR-1417 (pictures appended after the last text)
    is chunked in page order (VOR-1452): the same text chunks and
    the same figure labels as the book with each picture on its own page."""
    stored = await _chunks(_stored_order_book().doc(16))
    ordered = await _chunks(_page_order_book().doc(16))

    def text(chunks: list[RetrievalChunk]) -> list[tuple[str, str]]:
        return [(c.section_path, c.content) for c in chunks if c.chunk_type == "text"]

    def figures(chunks: list[RetrievalChunk]) -> dict[str | None, str]:
        return {c.image_id: c.section_path for c in _figures(chunks)}

    assert text(stored) == text(ordered)
    assert figures(stored) == figures(ordered)
    assert [c.chunk_type for c in stored] == [c.chunk_type for c in ordered]


def test_page_context_excerpt_never_exceeds_its_bound() -> None:
    from vo_rag.pipelines.book_rag.visual_sidecar import page_context_excerpt

    one_long = "word " * 300
    for page_text in (
        f"Earlier on page: {one_long.strip()}.",
        f"Later on page: {one_long.strip()}.",
        "Earlier on page: A. " + "B " * 500 + "\nLater on page: " + "C " * 500,
        "no labels at all. " * 60,
    ):
        excerpt = page_context_excerpt(page_text)
        assert excerpt
        assert len(excerpt) <= PAGE_CONTEXT_EXCERPT_CHARS, len(excerpt)


def test_page_context_excerpt_takes_the_sentences_next_to_the_figure() -> None:
    from vo_rag.pipelines.book_rag.visual_sidecar import page_context_excerpt

    earlier = " ".join(f"Before {n}." for n in range(100))
    later = " ".join(f"After {n}." for n in range(100))
    excerpt = page_context_excerpt(f"Earlier on page: {earlier}\nLater on page: {later}")

    assert "Before 99." in excerpt and "After 0." in excerpt
    assert "Before 0." not in excerpt and "After 99." not in excerpt
    assert excerpt.index("Before 99.") < excerpt.index("After 0.")  # reading order


def test_stale_figure_falls_back_like_the_text_around_it() -> None:
    from vo_rag.core.chunking.heading_tracker import STALE_HEADING_PAGES, HeadingTracker

    tracker = HeadingTracker(doc_title="Book")
    tracker.update_title("Chapter 1 Sprinting", depth=1, page=1)
    assert tracker.figure_attribution(5) == (
        "[Book > Chapter 1 Sprinting]",
        "Chapter 1 Sprinting",
        None,
    )
    far = 1 + STALE_HEADING_PAGES + 1
    assert tracker.figure_attribution(far) == ("[Book]", "Book", "Chapter 1 Sprinting")
    # Peeking never moves the tracker.
    assert tracker.section_path() == "[Book > Chapter 1 Sprinting]"


async def test_page_bucket_fallback_keeps_the_figure_window() -> None:
    """A book with no headings gets "pp. N-M" labels (TitleOnlyFallback), which
    rewrites embedding_text; a figure must keep its neighbour window there."""
    book = _Book()
    for page in range(1, 61):
        for part in range(3):
            book.body(f"{HIGH_JUMP} Unheaded page {page}, part {part}.", page)
            if (page, part) == (12, 0):
                book.image(12, name="unheaded", caption="Figure 9 The scissors.")
    chunks = await _chunks(book.doc(60))
    [figure] = _figures(chunks)

    assert figure.section_path == f"[{TITLE} > pp. 11-20]"
    assert figure.embedding_text.startswith(f"[{TITLE} > pp. 11-20]\n\n")
    assert "Preceding:" in figure.embedding_text
    assert "Following:" in figure.embedding_text
    assert "Preceding:" not in figure.content


def test_token_ceiling_never_splits_a_figure() -> None:
    from vo_rag.core.chunking.counters import naive_counter
    from vo_rag.core.chunking.token_ceiling import TokenCeilingBackstop

    window = "Preceding: " + "neighbour words here. " * 400
    figure = RetrievalChunk(
        chunk_id="f",
        chunk_type="figure_composite",
        document_id="d",
        source_id="s",
        page_number=3,
        section_path="[Book]",
        section_title="Book",
        content="Caption: Figure 1.\n\nVisual: " + "a long description. " * 300,
        embedding_text=f"[Book]\n\n{window}",
        token_count=naive_counter(f"[Book]\n\n{window}"),
        has_image=True,
        image_id="img",
        image_ref="src/img.png",
    )
    backstop = TokenCeilingBackstop(counter=naive_counter, max_tokens=600, overlap=0)
    assert backstop.enforce([figure]) == [figure]

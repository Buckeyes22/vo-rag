from __future__ import annotations

from vo_rag.core.chunking.composite_emitter import CompositeChunk
from vo_rag.core.chunking.hierarchical import HierarchicalChunker, _chunk_id
from vo_rag.core.config.settings import ChunkingSettings
from vo_rag.core.parsing.contract import ElementType, PageQA, ParsedDocument, ParsedElement


def _word_count(text: str) -> int:
    return len(text.split())


def _doc(elements: list[ParsedElement], *, total_pages: int = 3) -> ParsedDocument:
    return ParsedDocument(
        document_id="d",
        source_id="s",
        title="Book",
        total_pages=total_pages,
        elements=elements,
        images=[],
        page_qa=[
            PageQA(page_number=page, ocr_confidence=0.9, parser_used="p")
            for page in range(1, total_pages + 1)
        ],
        markdown="",
        heading_tree=[],
        parser_used="p",
        quarantined_pages=[],
    )


def test_default_chunking_strategy_keeps_hierarchical_output() -> None:
    assert ChunkingSettings().strategy == "hierarchical"
    elements = [
        ParsedElement(
            element_id="c1",
            element_type=ElementType.TITLE,
            text="Chapter 1",
            page_number=1,
            section_depth=1,
        ),
        ParsedElement(
            element_id="p1",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Brake pressure data introduces the table.",
            page_number=1,
        ),
        ParsedElement(
            element_id="t1",
            element_type=ElementType.TABLE,
            text="First gear ratio: value is 2.92.\nSecond gear ratio: value is 1.88.",
            page_number=1,
        ),
        ParsedElement(
            element_id="p2",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Throttle trace data follows the table.",
            page_number=1,
        ),
    ]

    chunks = HierarchicalChunker(target=1000, max_=1200, min_=1, counter=_word_count).chunk(
        _doc(elements),
        composites=[],
    )

    assert [
        {
            "chunk_id": chunk.chunk_id,
            "chunk_type": chunk.chunk_type,
            "page_number": chunk.page_number,
            "page_range": chunk.page_range,
            "section_path": chunk.section_path,
            "section_title": chunk.section_title,
            "content": chunk.content,
            "embedding_text": chunk.embedding_text,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ] == [
        {
            "chunk_id": _chunk_id(
                "s",
                "Brake pressure data introduces the table.",
                "text",
            ),
            "chunk_type": "text",
            "page_number": 1,
            "page_range": "1",
            "section_path": "[Book > Chapter 1]",
            "section_title": "Chapter 1",
            "content": "Brake pressure data introduces the table.",
            "embedding_text": "[Book > Chapter 1]\n\nBrake pressure data introduces the table.",
            "metadata": {},
        },
        {
            "chunk_id": _chunk_id(
                "s",
                "First gear ratio: value is 2.92.\nSecond gear ratio: value is 1.88.",
                "table",
            ),
            "chunk_type": "table",
            "page_number": 1,
            "page_range": "1",
            "section_path": "[Book > Chapter 1]",
            "section_title": "Chapter 1",
            "content": "First gear ratio: value is 2.92.\nSecond gear ratio: value is 1.88.",
            "embedding_text": (
                "[Book > Chapter 1]\n\n"
                "First gear ratio: value is 2.92.\nSecond gear ratio: value is 1.88."
            ),
            "metadata": {"element_id": "t1"},
        },
        {
            "chunk_id": _chunk_id(
                "s",
                "Throttle trace data follows the table.",
                "text",
            ),
            "chunk_type": "text",
            "page_number": 1,
            "page_range": "1",
            "section_path": "[Book > Chapter 1]",
            "section_title": "Chapter 1",
            "content": "Throttle trace data follows the table.",
            "embedding_text": "[Book > Chapter 1]\n\nThrottle trace data follows the table.",
            "metadata": {},
        },
    ]


class FakeProseSplitter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def split_text(self, content: str, section_path: str) -> list[str]:
        self.calls.append((content, section_path))
        return [f"semantic-first: {content}", f"semantic-second: {section_path}"]


def test_injected_prose_splitter_only_handles_prose() -> None:
    elements = [
        ParsedElement(
            element_id="c1",
            element_type=ElementType.TITLE,
            text="Chapter 1",
            page_number=1,
            section_depth=1,
        ),
        ParsedElement(
            element_id="p1",
            element_type=ElementType.NARRATIVE_TEXT,
            text=(
                "Brake pressure and throttle traces describe the first prose boundary. "
                "Steering and apex data keep this paragraph retrieval-worthy."
            ),
            page_number=1,
        ),
        ParsedElement(
            element_id="t1",
            element_type=ElementType.TABLE,
            text="First gear ratio: value is 2.92.\nSecond gear ratio: value is 1.88.",
            page_number=1,
        ),
        ParsedElement(
            element_id="p2",
            element_type=ElementType.NARRATIVE_TEXT,
            text=(
                "Throttle traces after the table describe a second prose boundary. "
                "Brake release and corner exit speed remain technical enough."
            ),
            page_number=2,
        ),
    ]
    composite = CompositeChunk(
        chunk_id="fig-1",
        section_title="Chapter 1",
        preceding_text="",
        vlm_description="driver trace plot",
        following_text="",
        image_id="img-1",
        image_ref="figures/img-1.png",
        embedding_text="[IMAGE: driver trace plot]",
        token_count=4,
        page_number=1,
        source_id="s",
        element_index=2,
    )
    fake = FakeProseSplitter()

    chunks = HierarchicalChunker(
        target=1000,
        max_=1200,
        min_=1,
        counter=_word_count,
        prose_splitter=fake,
    ).chunk(_doc(elements), composites=[composite])

    assert len(fake.calls) == 2
    assert all(section_path == "[Book > Chapter 1]" for _, section_path in fake.calls)
    assert [chunk.chunk_type for chunk in chunks] == [
        "text",
        "text",
        "figure_composite",
        "table",
        "text",
        "text",
    ]
    assert chunks[0].content.startswith("semantic-first:")
    assert chunks[1].content == "semantic-second: [Book > Chapter 1]"
    # Stored content is the figure's own text (VOR-1452); the builder's
    # window rides in embedding_text only.
    assert chunks[2].content == "Visual: driver trace plot"
    assert chunks[2].embedding_text == "[Book > Chapter 1]\n\n[IMAGE: driver trace plot]"
    assert (
        chunks[3].content == "First gear ratio: value is 2.92.\nSecond gear ratio: value is 1.88."
    )
    assert chunks[4].content.startswith("semantic-first:")
    assert chunks[5].content == "semantic-second: [Book > Chapter 1]"

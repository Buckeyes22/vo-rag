from __future__ import annotations

from vo_rag.core.chunking.models import PendingShortText, RetrievalChunk
from vo_rag.core.chunking.short_text import ShortTextMerger


def _chunk(page: int, content: str) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=f"c-{page}",
        chunk_type="text",
        document_id="doc",
        source_id="source",
        page_number=page,
        page_range=str(page),
        section_path="[Book]",
        section_title="Book",
        chapter_path="[Book]",
        content=content,
        embedding_text=f"[Book]\n\n{content}",
        token_count=len(content.split()),
    )


def test_short_text_merger_merges_useful_short_text_into_adjacent_chunk() -> None:
    drops: list[tuple[int | None, str]] = []
    merger = ShortTextMerger(counter=lambda text: len(text.split()))
    chunks = [_chunk(1, "Brake pressure trace."), _chunk(3, "Throttle trace.")]
    candidates = [
        PendingShortText(
            content="Smooth hands settle chassis balance.",
            pages=[2],
            section_path="[Book]",
            section_title="Book",
        )
    ]

    out = merger.merge_or_record(
        chunks,
        candidates,
        document_id="doc",
        source_id="source",
        doc_title="Book",
        record_drop=lambda page, reason: drops.append((page, reason)),
    )

    assert out is chunks
    assert "Smooth hands settle chassis balance." in chunks[0].content
    assert chunks[0].page_range == "1-2"
    assert drops == []


def test_short_text_merger_never_drops_text_on_an_already_covered_page() -> None:
    # VOR-1452: this used to record a below_min_words drop and lose the text.
    drops: list[tuple[int | None, str]] = []
    merger = ShortTextMerger(counter=lambda text: len(text.split()))
    chunks = [_chunk(2, "Existing technical content.")]
    candidates = [
        PendingShortText(
            content="Tiny note",
            pages=[2],
            section_path="[Book]",
            section_title="Book",
        )
    ]

    out = merger.merge_or_record(
        chunks,
        candidates,
        document_id="doc",
        source_id="source",
        doc_title="Book",
        record_drop=lambda page, reason: drops.append((page, reason)),
    )

    assert drops == []
    assert [chunk.content for chunk in out] == ["Existing technical content.\n\nTiny note"]


def test_short_text_merger_keeps_text_of_another_section_standalone() -> None:
    drops: list[tuple[int | None, str]] = []
    merger = ShortTextMerger(counter=lambda text: len(text.split()))
    chunks = [_chunk(2, "Existing technical content.")]
    candidates = [
        PendingShortText(
            content="Tiny note",
            pages=[2],
            section_path="[Book > Other]",
            section_title="Other",
        )
    ]

    out = merger.merge_or_record(
        chunks,
        candidates,
        document_id="doc",
        source_id="source",
        doc_title="Book",
        record_drop=lambda page, reason: drops.append((page, reason)),
    )

    assert drops == []
    assert [chunk.content for chunk in out] == ["Existing technical content.", "Tiny note"]
    assert out[1].section_path == "[Book > Other]"

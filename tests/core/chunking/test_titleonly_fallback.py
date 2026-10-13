"""Low-heading-confidence page-bucketing fallback."""

from __future__ import annotations

from vo_rag.core.chunking.hierarchical import HierarchicalChunker, RetrievalChunk
from vo_rag.core.parsing.contract import ParsedDocument


def _doc(title: str, total_pages: int) -> ParsedDocument:
    return ParsedDocument(
        document_id="d",
        source_id="s",
        title=title,
        total_pages=total_pages,
        elements=[],
        images=[],
        page_qa=[],
        markdown="",
        heading_tree=[],
        parser_used="x",
    )


def _title_chunk(title: str, i: int) -> RetrievalChunk:
    so = f"[{title}]"
    return RetrievalChunk(
        chunk_id=f"c{i}",
        chunk_type="text",
        document_id="d",
        source_id="s",
        page_number=i,
        page_range=str(i),
        section_path=so,
        section_title=title,
        chapter_path=so,
        content=f"content {i}",
        embedding_text=f"{so}\n\ncontent {i}",
        token_count=5,
    )


def test_oversized_titleonly_section_is_page_bucketed() -> None:
    chunker = HierarchicalChunker()
    title = "Big Structureless Book"
    chunks = [_title_chunk(title, i) for i in range(1, 41)]  # 40 > threshold 10
    out = chunker._page_section_titleonly_fallback(chunks, _doc(title, 120))

    sections = {c.section_path for c in out}
    chapters = {c.chapter_path for c in out}
    assert f"[{title}]" not in sections  # the giant title-only section is gone
    assert any("pp. 1-10" in s for s in sections)
    assert len(sections) > 1
    # chapters (50-page buckets) aggregate sections (10-page buckets)
    assert len(chapters) < len(sections)
    # embedding_text/section_title kept consistent with the new section_path
    for c in out:
        assert c.embedding_text.startswith(c.section_path)
        assert c.section_title.startswith("pp. ")


def test_small_titleonly_section_left_untouched() -> None:
    chunker = HierarchicalChunker()
    title = "Small Book"
    chunks = [_title_chunk(title, i) for i in range(1, 6)]  # 5 <= threshold
    out = chunker._page_section_titleonly_fallback(chunks, _doc(title, 5))
    assert all(c.section_path == f"[{title}]" for c in out)  # unchanged


def test_real_headings_not_resectioned() -> None:
    chunker = HierarchicalChunker()
    title = "Well Headed"
    # 40 chunks but all carry real section paths (not title-only) -> untouched
    chunks = []
    for i in range(1, 41):
        sp = f"[{title} > Chapter {i % 5}]"
        chunks.append(
            RetrievalChunk(
                chunk_id=f"c{i}",
                chunk_type="text",
                document_id="d",
                source_id="s",
                page_number=i,
                page_range=str(i),
                section_path=sp,
                section_title=f"Chapter {i % 5}",
                chapter_path=sp,
                content=f"c{i}",
                embedding_text=f"{sp}\n\nc{i}",
                token_count=5,
            )
        )
    out = chunker._page_section_titleonly_fallback(chunks, _doc(title, 40))
    assert all("pp. " not in c.section_path for c in out)  # no page-bucketing applied

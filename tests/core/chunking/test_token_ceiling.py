from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from vo_rag.core.chunking.counters import naive_counter
from vo_rag.core.chunking.models import RetrievalChunk
from vo_rag.core.chunking.token_ceiling import TokenCeilingBackstop


def _chunk(content: str, *, chunk_type: str = "text") -> RetrievalChunk:
    embedding_text = f"[Book]\n\n{content}"
    return RetrievalChunk(
        chunk_id="chunk",
        chunk_type=chunk_type,
        document_id="doc",
        source_id="source",
        page_number=1,
        page_range="1",
        section_path="[Book]",
        section_title="Book",
        chapter_path="[Book]",
        content=content,
        embedding_text=embedding_text,
        token_count=naive_counter(embedding_text),
    )


def test_token_ceiling_backstop_resplits_oversized_chunks_with_metadata() -> None:
    backstop = TokenCeilingBackstop(counter=lambda text: len(text.split()), max_tokens=10)
    content = " ".join(f"word{i}" for i in range(45))

    out = backstop.enforce([_chunk(content)])

    assert len(out) > 1
    assert all(chunk.token_count <= int(10 * 1.15) for chunk in out)
    assert [chunk.metadata["ceiling_split_part"] for chunk in out] == list(range(1, len(out) + 1))
    assert {chunk.metadata["ceiling_split_total"] for chunk in out} == {len(out)}


def test_oversized_table_row_keeps_all_values_and_bounds_repeated_header() -> None:
    backstop = TokenCeilingBackstop(counter=naive_counter, max_tokens=32, overlap=50)
    values = [f"v{index:03}" for index in range(80)]
    content = "Comparison across columns: Item, Value.\nRow 1: " + " ".join(values)

    pieces = backstop.split_table(content, "[Book]")

    assert backstop.overlap == 8
    assert len(pieces) > 1
    assert all(backstop.embedding_token_count("[Book]", piece) <= backstop.max for piece in pieces)
    assert all("Comparison across columns: Item, Value." in piece for piece in pieces)
    assert all(any(value in piece for piece in pieces) for value in values)


def test_oversized_table_header_is_split_without_aborting() -> None:
    backstop = TokenCeilingBackstop(counter=naive_counter, max_tokens=20, overlap=50)
    values = [f"header{index:03}" for index in range(40)]
    content = "Comparison across columns: " + " ".join(values)

    pieces = backstop.split_table(content, "[Book]")

    assert len(pieces) > 1
    assert all(backstop.embedding_token_count("[Book]", piece) <= backstop.max for piece in pieces)
    assert all(any(value in piece for piece in pieces) for value in values)


@given(
    st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
        min_size=1,
        max_size=2000,
    )
)
@settings(max_examples=40, deadline=None)
def test_token_ceiling_never_exceeded_property(content: str) -> None:
    backstop = TokenCeilingBackstop(counter=naive_counter, max_tokens=60)
    out = backstop.enforce([_chunk(content or "x")])

    assert all(chunk.token_count <= int(60 * 1.15) for chunk in out)

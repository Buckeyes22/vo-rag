from __future__ import annotations

from vo_rag.core.chunking.composite_emitter import CompositeChunk, CompositeEmitter
from vo_rag.core.chunking.table_linearizer import TableLinearizer
from vo_rag.core.parsing.contract import ElementType, ParsedElement


def _word_count(text: str) -> int:
    return len(text.split())


def test_composite_emitter_builds_figure_composite_with_image_metadata() -> None:
    emitter = CompositeEmitter(counter=_word_count)
    composite = CompositeChunk(
        chunk_id="source-composite",
        section_title="Figure Section",
        preceding_text="",
        vlm_description="damper histogram",
        following_text="",
        image_id="img-1",
        image_ref="garage/img-1.png",
        embedding_text="[IMAGE: damper histogram]",
        token_count=3,
        page_number=4,
        source_id="source",
        element_index=8,
        metadata={"caption": "Figure 4"},
    )

    chunk = emitter.emit_composite(
        composite,
        document_id="doc",
        source_id="source",
        section_path="[Book > Chapter]",
        section_title="Chapter",
        doc_title="Book",
    )

    assert chunk.chunk_type == "figure_composite"
    assert chunk.has_image is True
    assert chunk.image_id == "img-1"
    assert chunk.image_ref == "garage/img-1.png"
    assert chunk.metadata == {"element_index": 8, "caption": "Figure 4"}


def test_composite_emitter_emits_tables_with_split_metadata() -> None:
    emitter = CompositeEmitter(counter=_word_count, max_tokens=6, overlap=0)
    table = ParsedElement(
        element_id="t1",
        element_type=ElementType.TABLE,
        text="\n".join(
            [
                "ROW 1 has several values.",
                "ROW 2 has several values.",
                "ROW 3 has several values.",
            ]
        ),
        page_number=2,
    )

    chunks = emitter.emit_table(
        table,
        document_id="doc",
        source_id="source",
        section_path="[Book]",
        section_title="Book",
        doc_title="Book",
    )

    assert [chunk.chunk_type for chunk in chunks] == ["table", "table", "table"]
    assert [chunk.metadata["table_split_part"] for chunk in chunks] == [1, 2, 3]


def test_table_ingestion_keeps_all_rows_when_over_budget() -> None:
    emitter = CompositeEmitter(
        counter=lambda text: max(1, len(text) // 4),
        max_tokens=30,
        overlap=0,
        table_linearizer=TableLinearizer(counter=lambda text: max(1, len(text) // 4)),
    )
    rows = "".join(f"<tr><td>R{i}</td><td>V{i}</td></tr>" for i in range(100))
    table = ParsedElement(
        element_id="t-large",
        element_type=ElementType.TABLE,
        metadata={"html": f"<table><tr><th>Item</th><th>Value</th></tr>{rows}</table>"},
    )

    chunks = emitter.emit_table(
        table,
        document_id="doc",
        source_id="source",
        section_path="[Book]",
        section_title="Book",
        doc_title="Book",
    )

    assert len(chunks) > 1
    assert all(any(f"R{i}: Value=V{i}." in chunk.content for chunk in chunks) for i in range(100))
    assert all(chunk.token_count <= 30 for chunk in chunks)

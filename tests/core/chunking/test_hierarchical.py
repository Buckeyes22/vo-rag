from __future__ import annotations

from collections.abc import Sequence

from vo_rag.core.chunking.composite_emitter import CompositeChunk
from vo_rag.core.chunking.hierarchical import HierarchicalChunker, _chunk_id
from vo_rag.core.parsing.contract import ElementType, PageQA, ParsedDocument, ParsedElement


def _word_count(text: str) -> int:
    return len(text.split())


def test_mathematical_page_pixels_accompany_prose_without_replacing_equation_crop() -> None:
    visual = {
        "status": "source_bound",
        "physical_page": 2,
        "sha256": "ab" * 32,
        "image_ref": "cd" * 32 + "/p002_pg" + "ab" * 32 + ".png",
    }
    equation_visual = {
        "status": "source_bound",
        "sha256": "ef" * 32,
        "image_ref": "cd" * 32 + "/p002_eq" + "ef" * 32 + ".png",
    }
    prose = ParsedElement(
        element_id="prose",
        element_type=ElementType.NARRATIVE_TEXT,
        text="The mathematical relationship occurs within this original printed paragraph "
        "alongside the surrounding explanation.",
        page_number=2,
    )
    equation = ParsedElement(
        element_id="eq",
        element_type=ElementType.EQUATION,
        text="numerator denominator --",
        page_number=2,
        metadata={"source_page_visual": visual, "equation_visual": equation_visual},
    )
    chunks = HierarchicalChunker(counter=_word_count).chunk(
        _doc([prose, equation], total_pages=2), composites=[]
    )
    body = next(c for c in chunks if c.chunk_type == "text")
    assert body.content == prose.text and body.has_image and body.image_ref == visual["image_ref"]
    assert body.vlm_description is None
    crop = next(c for c in chunks if c.chunk_type == "figure_composite")
    assert crop.image_ref == equation_visual["image_ref"]
    assert all(c.metadata["source_page_visuals"] == [visual] for c in chunks)


def test_small_table_merged_into_prose_retains_original_layout() -> None:
    visual = {
        "status": "source_bound",
        "physical_page": 2,
        "sha256": "ab" * 32,
        "image_ref": "cd" * 32 + "/p002_tbl" + "ab" * 32 + ".png",
    }
    prose = ParsedElement(
        element_id="prose",
        element_type=ElementType.NARRATIVE_TEXT,
        text="The measured values in the following table explain the complete relationship.",
        page_number=2,
    )
    table = ParsedElement(
        element_id="small-table",
        element_type=ElementType.TABLE,
        text="| Speed | Force |\n|---|---|\n| 12 | 24 |",
        page_number=2,
        metadata={"table_visual": visual},
    )
    chunks = HierarchicalChunker(counter=_word_count).chunk(
        _doc([prose, table], total_pages=2), composites=[]
    )
    assert len(chunks) == 1
    merged = chunks[0]
    assert merged.chunk_type == "text"
    assert prose.text in merged.content and "12" in merged.content and "24" in merged.content
    assert merged.metadata["merged_table_element_ids"] == [table.element_id]
    assert merged.metadata["source_table_visuals"] == [
        {"source_element_id": table.element_id, **visual}
    ]
    assert merged.has_image and merged.image_ref == visual["image_ref"]
    assert merged.vlm_description is None


def test_original_table_layout_accompanies_every_normal_text_part() -> None:
    visual = {
        "status": "source_bound",
        "sha256": "ab" * 32,
        "image_ref": "cd" * 32 + "/p002_tbl" + "ab" * 32 + ".png",
    }
    table = ParsedElement(
        element_id="table-2",
        element_type=ElementType.TABLE,
        text="| Question | Answer |\n|---|---|\n"
        + "\n".join(
            f"| {i} | The complete original answer has several words. |" for i in range(20)
        ),
        page_number=2,
        metadata={"table_visual": visual},
    )
    chunks = HierarchicalChunker(counter=_word_count, target=40, max_=50, min_=1).chunk(
        _doc([table], total_pages=2), composites=[]
    )
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.chunk_type == "table"
        assert chunk.has_image and chunk.image_ref == visual["image_ref"]
        assert chunk.vlm_description is None
        assert chunk.metadata["table_source_element_id"] == table.element_id
        assert chunk.metadata["table_visual"] == visual
        assert chunk.metadata["symbolic_transcription_verified"] is False
    assert all(
        any(
            f"| {i} | The complete original answer has several words. |" in c.content
            for c in chunks
        )
        for i in range(20)
    )


def test_original_equation_crop_is_retrievable_without_vlm_or_parsed_images() -> None:
    original = "rpm cutting speed, fpm circumference in feet ---- ="
    visual = {
        "status": "source_bound",
        "sha256": "ab" * 32,
        "image_ref": "cd" * 32 + "/p002_eq" + "ab" * 32 + ".png",
    }
    equation = ParsedElement(
        element_id="eq-2",
        element_type=ElementType.EQUATION,
        text=original,
        page_number=2,
        metadata={"docling_orig": original, "equation_visual": visual},
    )
    doc = _doc([equation], total_pages=2)
    assert doc.images == []
    chunks = HierarchicalChunker(counter=_word_count).chunk(doc, composites=[])
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_type == "figure_composite"
    assert chunk.content == original
    assert chunk.has_image and chunk.image_ref == visual["image_ref"]
    assert chunk.page_number == 2 and chunk.page_range == "2"
    assert chunk.vlm_description is None
    assert chunk.metadata["symbolic_transcription_verified"] is False
    assert chunk.metadata["equation_source_element_id"] == "eq-2"


def _doc(
    elements: list[ParsedElement],
    quarantined: list[int] | None = None,
    *,
    total_pages: int = 3,
) -> ParsedDocument:
    return ParsedDocument(
        document_id="d",
        source_id="s",
        title="Book",
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
        quarantined_pages=list(quarantined or []),
    )


def _covered_pages(chunks: Sequence[object]) -> set[int]:
    pages: set[int] = set()
    for chunk in chunks:
        page_range = getattr(chunk, "page_range", None)
        if page_range and "-" in page_range:
            start, end = [int(part) for part in page_range.split("-", maxsplit=1)]
            pages.update(range(start, end + 1))
            continue
        if page_range:
            pages.add(int(page_range))
            continue
        page_number = getattr(chunk, "page_number", None)
        if page_number is not None:
            pages.add(int(page_number))
    return pages


def test_numeric_table_rows_survive_narrative_chunk_boundaries() -> None:
    """Numeric rows stay present when packing flushes before or within the table."""
    rows = [f"{number}/5 = {number * 3}%" for number in range(1, 52)]
    elements = [
        ParsedElement(
            element_id="intro",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Convert these numeric chances into odds using the table below.",
            page_number=1,
        ),
        *[
            ParsedElement(
                element_id=f"row-{index}",
                element_type=ElementType.NARRATIVE_TEXT,
                text=row,
                page_number=1,
            )
            for index, row in enumerate(rows)
        ],
    ]
    chunks = HierarchicalChunker(counter=_word_count, target=20, max_=30, min_=1).chunk(
        _doc(elements, total_pages=1), composites=[]
    )
    assert len(chunks) > 1
    assert all(any(row in chunk.content for chunk in chunks) for row in rows)


def test_hard_break_at_depth_one_title() -> None:
    els = [
        ParsedElement(
            element_id="c1",
            element_type=ElementType.TITLE,
            text="Ch1",
            page_number=1,
            section_depth=1,
        ),
        ParsedElement(
            element_id="p1",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Trail braking adjusts brake pressure, steering, throttle, and apex timing.",
            page_number=1,
        ),
        ParsedElement(
            element_id="c2",
            element_type=ElementType.TITLE,
            text="Ch2",
            page_number=2,
            section_depth=1,
        ),
        ParsedElement(
            element_id="p2",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Corner exit throttle traces help compare speed and steering corrections.",
            page_number=2,
        ),
    ]
    chunks = HierarchicalChunker(target=50, max_=80, min_=5).chunk(_doc(els), composites=[])
    for c in chunks:
        assert not ("text a" in c.content and "text b" in c.content)


def test_quarantined_pages_skipped() -> None:
    els = [
        ParsedElement(
            element_id="p1",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Brake pressure trace remains useful.",
            page_number=1,
        ),
        ParsedElement(
            element_id="p2",
            element_type=ElementType.NARRATIVE_TEXT,
            text="bad-quarantined",
            page_number=2,
        ),
    ]
    chunks = HierarchicalChunker(target=10, max_=20, min_=1).chunk(
        _doc(els, quarantined=[2]), composites=[]
    )
    assert all("bad-quarantined" not in c.content for c in chunks)


def test_short_sole_page_text_is_merged_not_dropped() -> None:
    els = [
        ParsedElement(
            element_id="c1",
            element_type=ElementType.TITLE,
            text="Corner Entry",
            page_number=1,
            section_depth=1,
        ),
        ParsedElement(
            element_id="p1",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Brake pressure traces identify release timing before turn-in.",
            page_number=1,
        ),
        ParsedElement(
            element_id="c2",
            element_type=ElementType.TITLE,
            text="Balance Note",
            page_number=2,
            section_depth=1,
        ),
        ParsedElement(
            element_id="p2",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Smooth hands settle chassis balance.",
            page_number=2,
        ),
        ParsedElement(
            element_id="c3",
            element_type=ElementType.TITLE,
            text="Corner Exit",
            page_number=3,
            section_depth=1,
        ),
        ParsedElement(
            element_id="p3",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Throttle traces show steering correction at apex exit.",
            page_number=3,
        ),
    ]
    chunker = HierarchicalChunker(target=1000, max_=1200, min_=1)

    chunks = chunker.chunk(_doc(els), composites=[])

    assert "Smooth hands settle chassis balance." in "\n\n".join(c.content for c in chunks)
    assert 2 in _covered_pages(chunks)
    ledger = getattr(chunker, "last_page_drop_ledger", [])
    assert all(drop.page_number != 2 or drop.reason != "below_min_words" for drop in ledger)


def test_empty_after_normalize_page_is_recorded_in_drop_ledger() -> None:
    els = [
        ParsedElement(
            element_id="p1",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Brake pressure traces identify release timing before turn-in.",
            page_number=1,
        ),
        ParsedElement(
            element_id="p2",
            element_type=ElementType.NARRATIVE_TEXT,
            text="2",
            page_number=2,
        ),
    ]
    chunker = HierarchicalChunker(target=1000, max_=1200, min_=1)

    chunker.chunk(_doc(els, total_pages=2), composites=[])

    ledger = getattr(chunker, "last_page_drop_ledger", [])
    assert [(drop.page_number, drop.reason) for drop in ledger] == [(2, "empty_after_normalize")]
    assert getattr(chunker, "last_drop_summary", {}) == {"empty_after_normalize": 1}


def test_drop_ledger_summary_counts_reasons() -> None:
    els = [
        ParsedElement(
            element_id="p1",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Brake pressure traces identify release timing before turn-in.",
            page_number=1,
        ),
        ParsedElement(
            element_id="c2",
            element_type=ElementType.TITLE,
            text="Covered Short Note",
            page_number=2,
            section_depth=1,
        ),
        ParsedElement(
            element_id="short-covered",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Smooth hands settle chassis balance.",
            page_number=2,
        ),
        ParsedElement(
            element_id="t2",
            element_type=ElementType.TABLE,
            text="Gear 1 ratio 2.92\nGear 2 ratio 1.88",
            page_number=2,
        ),
        ParsedElement(
            element_id="q3",
            element_type=ElementType.NARRATIVE_TEXT,
            text="This page failed parser QA and must not reach retrieval.",
            page_number=3,
        ),
    ]
    chunker = HierarchicalChunker(target=1000, max_=1200, min_=1)

    chunker.chunk(_doc(els, quarantined=[3], total_pages=4), composites=[])

    ledger = getattr(chunker, "last_page_drop_ledger", [])
    # VOR-1452: short text on a page another chunk covers is kept (merged or
    # standalone), never dropped as below_min_words.
    assert [(drop.page_number, drop.reason) for drop in ledger] == [
        (3, "qa_quarantined"),
        (4, "no_text_extracted"),
    ]
    assert getattr(chunker, "last_drop_summary", {}) == {
        "no_text_extracted": 1,
        "qa_quarantined": 1,
    }


def test_chunk_id_deterministic() -> None:
    # Unversioned content-addressed handle: same (source_id, content, stage)
    # → same id; differing content or stage → different id.
    a = _chunk_id("src1", "hello", "text")
    b = _chunk_id("src1", "hello", "text")
    assert a == b
    assert _chunk_id("src1", "world", "text") != a
    assert _chunk_id("src1", "hello", "table") != a


def test_composite_interleaving_flushes_text() -> None:
    els = [
        ParsedElement(
            element_id="p1",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Brake trace before apex.",
            page_number=1,
        ),
        ParsedElement(
            element_id="p2",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Throttle trace after apex.",
            page_number=1,
        ),
    ]
    comp = CompositeChunk(
        chunk_id="c",
        section_title="Sec",
        preceding_text="",
        vlm_description="fig",
        following_text="",
        image_id="img",
        image_ref="g/key",
        embedding_text="[IMAGE: fig]",
        token_count=3,
        page_number=1,
        source_id="s",
        element_index=1,
    )
    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(_doc(els), composites=[comp])
    assert [c.chunk_type for c in out] == ["text", "figure_composite", "text"]
    assert out[0].content == "Brake trace before apex."
    # content is the figure's own text; the builder's window is embedding-only (VOR-1452).
    assert out[1].content == "Visual: fig"
    assert "IMAGE" in out[1].embedding_text
    assert out[2].content == "Throttle trace after apex."


def test_table_becomes_own_chunk() -> None:
    els = [
        ParsedElement(
            element_id="p1",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Brake pressure data introduces the table.",
            page_number=1,
        ),
        ParsedElement(
            element_id="t1", element_type=ElementType.TABLE, text="table raw", page_number=1
        ),
        ParsedElement(
            element_id="p2",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Throttle trace data follows the table.",
            page_number=1,
        ),
    ]
    out = HierarchicalChunker(
        target=1000,
        max_=1200,
        min_=1,
        table_linearizer=lambda e: f"linearized: {e.text}",
    ).chunk(_doc(els), composites=[])
    assert [c.chunk_type for c in out] == ["text", "table", "text"]
    assert out[1].content == "linearized: table raw"


def test_oversized_single_text_element_is_hard_split_with_overlap() -> None:
    words = [f"word{i:04d}" for i in range(5000)]
    els = [
        ParsedElement(
            element_id="long",
            element_type=ElementType.NARRATIVE_TEXT,
            text=" ".join(words),
            page_number=7,
        )
    ]
    chunker = HierarchicalChunker(
        target=500,
        max_=600,
        min_=1,
        overlap=25,
        counter=_word_count,
    )

    out = chunker.chunk(_doc(els, total_pages=10), composites=[])

    ceiling = int(chunker.max * 1.15)
    assert len(out) > 1
    assert all(chunk.chunk_type == "text" for chunk in out)
    assert all(chunk.token_count <= ceiling for chunk in out)

    covered_words: list[str] = []
    for index, chunk in enumerate(out):
        chunk_words = chunk.content.split()
        if index > 0:
            overlap_words = out[index - 1].content.split()[-chunker.overlap :]
            assert chunk_words[: chunker.overlap] == overlap_words
            chunk_words = chunk_words[chunker.overlap :]
        covered_words.extend(chunk_words)
    assert covered_words == words


def test_oversized_table_is_hard_split_at_row_boundaries() -> None:
    rows = [
        f"ROW {row:03d}: " + " ".join(f"value{row:03d}_{cell:02d}" for cell in range(48)) + "."
        for row in range(100)
    ]
    els = [
        ParsedElement(
            element_id="big-table",
            element_type=ElementType.TABLE,
            text="\n".join(rows),
            page_number=4,
        )
    ]
    chunker = HierarchicalChunker(
        target=500,
        max_=600,
        min_=1,
        overlap=25,
        counter=_word_count,
    )

    out = chunker.chunk(_doc(els, total_pages=10), composites=[])

    ceiling = int(chunker.max * 1.15)
    assert len(out) > 1
    assert all(chunk.chunk_type == "table" for chunk in out)
    assert all(chunk.token_count <= ceiling for chunk in out)
    assert [chunk.metadata["table_split_part"] for chunk in out] == list(range(1, len(out) + 1))

    covered_rows = [
        line for chunk in out for line in chunk.content.splitlines() if line.startswith("ROW ")
    ]
    assert covered_rows == rows


def test_zero_whitespace_run_is_capped_under_naive_counter() -> None:
    # Regression: a malformed wide table whose row is a single whitespace-free
    # run (e.g. a corrupt TOC '|309||309||309|...') cannot be split on any
    # linguistic boundary. Under the production ``_naive_counter`` (chars//4)
    # such a run is thousands of tokens. The old boundary-only splitter leaked
    # it whole via the first-word fallback (re.match(r"\S+") matched the entire
    # run). The chunker must still guarantee every chunk is within the ceiling.
    run = "|309|" * 2000  # 10_000 chars, no internal whitespace -> ~2500 tokens
    els = [
        ParsedElement(
            element_id="corrupt-toc",
            element_type=ElementType.TABLE,
            text=run,
            page_number=4,
        )
    ]
    # Default counter (_naive_counter) -- mirrors production, NOT _word_count.
    chunker = HierarchicalChunker(target=500, max_=600, min_=1, overlap=25)

    out = chunker.chunk(_doc(els, total_pages=10), composites=[])

    ceiling = int(chunker.max * 1.15)
    assert len(out) > 1
    assert all(chunk.token_count <= ceiling for chunk in out)
    # No data loss: the de-whitespaced concatenation reconstructs the run.
    rebuilt = "".join("".join(chunk.content.split()) for chunk in out)
    assert rebuilt == run


def test_long_single_word_text_is_char_chopped() -> None:
    # A single narrative "word" with no whitespace (e.g. a base64 blob or a
    # run-together OCR artifact) longer than the budget must be char-chopped,
    # not emitted whole. Varied chars so it is not filtered as low-signal.
    blob = "abcdefghij0123456789" * 600  # 12_000 chars, ~3000 tokens
    els = [
        ParsedElement(
            element_id="blob",
            element_type=ElementType.NARRATIVE_TEXT,
            text=blob,
            page_number=2,
        )
    ]
    chunker = HierarchicalChunker(target=500, max_=600, min_=1, overlap=25)

    out = chunker.chunk(_doc(els, total_pages=10), composites=[])

    ceiling = int(chunker.max * 1.15)
    assert len(out) > 1
    assert all(chunk.token_count <= ceiling for chunk in out)
    rebuilt = "".join("".join(chunk.content.split()) for chunk in out)
    assert rebuilt == blob


def test_small_document_chunk_output_is_unchanged() -> None:
    els = [
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
    chunker = HierarchicalChunker(target=1000, max_=1200, min_=1, counter=_word_count)

    out = chunker.chunk(_doc(els), composites=[])

    expected = [
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
    actual = [
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
        for chunk in out
    ]
    assert actual == expected


def test_embedding_text_prefixed_with_section_path() -> None:
    els = [
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
            text="Brake pressure trace.",
            page_number=1,
        ),
    ]
    out = HierarchicalChunker(target=10, max_=20, min_=1).chunk(_doc(els), composites=[])
    assert out
    assert out[0].section_path.startswith("[Book")
    assert out[0].embedding_text.startswith(out[0].section_path)


def test_standalone_narrative_headings_start_new_chunks() -> None:
    els = [
        ParsedElement(
            element_id="t1",
            element_type=ElementType.TITLE,
            text="Goals",
            page_number=1,
            section_depth=1,
        ),
        ParsedElement(
            element_id="h1",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Warnings!",
            page_number=1,
        ),
        ParsedElement(
            element_id="p1",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Drivers should understand data tool limitations.",
            page_number=1,
        ),
        ParsedElement(
            element_id="h2",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Definitions",
            page_number=1,
        ),
        ParsedElement(
            element_id="p2",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Channels are measured parameters such as speed and RPM.",
            page_number=1,
        ),
    ]
    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(_doc(els), composites=[])
    assert len(out) == 2
    assert out[0].content.startswith("Warnings!")
    assert "Drivers should understand data tool limitations." in out[0].content
    assert out[1].content.startswith("Definitions")
    assert "Channels are measured parameters such as speed and RPM." in out[1].content


def test_consecutive_standalone_headings_are_carried_into_next_chunk() -> None:
    els = [
        ParsedElement(
            element_id="t1",
            element_type=ElementType.TITLE,
            text="Goals",
            page_number=1,
            section_depth=1,
        ),
        ParsedElement(
            element_id="h1",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Overview",
            page_number=1,
        ),
        ParsedElement(
            element_id="h2",
            element_type=ElementType.NARRATIVE_TEXT,
            text="The Basics",
            page_number=1,
        ),
        ParsedElement(
            element_id="p1",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Drivers should compare lap data before changing technique.",
            page_number=1,
        ),
    ]
    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(_doc(els), composites=[])
    assert len(out) == 1
    assert out[0].content.startswith("Overview\n\nThe Basics")


def test_numeric_only_values_outside_page_number_range_are_kept() -> None:
    els = [
        ParsedElement(
            element_id="p1",
            element_type=ElementType.NARRATIVE_TEXT,
            text="4\n5\n5\n5\n5\n6\n7",
            page_number=1,
        ),
    ]
    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(_doc(els), composites=[])
    assert "4\n5\n5\n5\n5\n6\n7" in "\n".join(chunk.content for chunk in out)


def test_agenda_outline_fragments_are_kept_with_the_definitions() -> None:
    """An outline of short lines is kept -- in these books
    it is a session plan or a list of drills, not an agenda slide."""
    els = [
        ParsedElement(
            element_id="p1",
            element_type=ElementType.NARRATIVE_TEXT,
            text=("Question\n\nWhat We'll Cover\n\nOverview\nExamples\nProcess\nResources\nQ&A"),
            page_number=1,
        ),
        ParsedElement(
            element_id="h2",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Definitions",
            page_number=2,
        ),
        ParsedElement(
            element_id="p2",
            element_type=ElementType.NARRATIVE_TEXT,
            text=(
                "Hardware: Data logger, dash, wiring and sensors\n\nChannels: Speed, throttle, RPM"
            ),
            page_number=2,
        ),
    ]
    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(_doc(els), composites=[])
    text = "\n".join(chunk.content for chunk in out)
    assert "What We'll Cover" in text and "Resources" in text
    assert any("Data logger" in chunk.content for chunk in out)


def test_individual_outline_headings_are_not_carried_into_next_chunk() -> None:
    els = [
        ParsedElement(
            element_id=f"h{idx}",
            element_type=ElementType.NARRATIVE_TEXT,
            text=text,
            page_number=2,
        )
        for idx, text in enumerate(
            ["Question", "What We'll Cover", "Overview", "Examples", "Process", "Resources", "Q&A"]
        )
    ]
    els.extend(
        [
            ParsedElement(
                element_id="def",
                element_type=ElementType.NARRATIVE_TEXT,
                text="Definitions",
                page_number=3,
            ),
            ParsedElement(
                element_id="body",
                element_type=ElementType.NARRATIVE_TEXT,
                text="Channels are logged values such as speed, throttle position, and RPM.",
                page_number=3,
            ),
        ]
    )
    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(_doc(els), composites=[])

    assert len(out) == 1
    assert out[0].page_number == 3
    assert out[0].section_title == "Definitions"
    assert out[0].content.startswith("Definitions")
    assert "What We'll Cover" not in out[0].content
    assert "Q&A" not in out[0].content


def test_resources_and_qa_only_page_is_filtered() -> None:
    els = [
        ParsedElement(
            element_id="p17",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Resources\n\nMore in Data Logger Examples...\n\nQ&A",
            page_number=17,
        ),
    ]
    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(_doc(els), composites=[])
    assert out == []


def test_headers_footers_are_skipped_but_interior_repeated_titles_are_kept() -> None:
    els: list[ParsedElement] = []
    for page in range(1, 4):
        els.extend(
            [
                ParsedElement(
                    element_id=f"h{page}",
                    element_type=ElementType.HEADER,
                    text="Book Running Header",
                    page_number=page,
                ),
                ParsedElement(
                    element_id=f"r{page}",
                    element_type=ElementType.NARRATIVE_TEXT,
                    text="Repeated Running Title",
                    page_number=page,
                ),
                ParsedElement(
                    element_id=f"p{page}",
                    element_type=ElementType.NARRATIVE_TEXT,
                    text=f"Brake pressure trace page {page} keeps retrieval value.",
                    page_number=page,
                ),
                ParsedElement(
                    element_id=f"f{page}",
                    element_type=ElementType.FOOTER,
                    text=str(page),
                    page_number=page,
                ),
            ]
        )

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(_doc(els), composites=[])

    assert out
    joined = "\n".join(chunk.content for chunk in out)
    assert "Book Running Header" not in joined
    assert "Repeated Running Title" in joined
    assert "Brake pressure trace page 3" in joined


def test_repeated_running_lines_inside_body_text_are_stripped() -> None:
    els = [
        ParsedElement(
            element_id=f"p{page}",
            element_type=ElementType.NARRATIVE_TEXT,
            text=(
                f"GCR - {548 + page}\n"
                f"Brake pressure trace page {page} keeps retrieval value "
                "for damper calibration analysis.\n"
                "Figure 4.10 cont'd"
            ),
            page_number=page,
        )
        for page in range(1, 6)
    ]

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(
        _doc(els, total_pages=5), composites=[]
    )

    assert out
    joined = "\n".join(chunk.content for chunk in out)
    assert "GCR -" not in joined
    assert "Figure 4.10" not in joined
    assert "Brake pressure trace page 5" in joined


def test_repeated_distinctive_interior_lines_are_not_stripped() -> None:
    els = [
        ParsedElement(
            element_id=f"p{page}",
            element_type=ElementType.NARRATIVE_TEXT,
            text=(
                f"Brake pressure trace page {page} starts with retrieval value.\n"
                f"GCR - {548 + page}\n"
                "Damper calibration analysis keeps enough body text for a chunk."
            ),
            page_number=page,
        )
        for page in range(1, 6)
    ]

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(
        _doc(els, total_pages=5), composites=[]
    )

    assert out
    joined = "\n".join(chunk.content for chunk in out)
    assert "GCR - 553" in joined
    assert "Damper calibration analysis" in joined


def test_title_page_toc_and_copyright_fragments_are_kept() -> None:
    """Title page, contents and imprint text are kept."""
    els = [
        ParsedElement(
            element_id="cover",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Book\nDemo Writer\nExample Publishing",
            page_number=1,
        ),
        ParsedElement(
            element_id="toc",
            element_type=ElementType.NARRATIVE_TEXT,
            text=(
                "Contents\nChapter 1 Getting Started .... 3\n"
                "Chapter 2 Data .... 11\nPart 3 Setup .... 27"
            ),
            page_number=2,
        ),
        ParsedElement(
            element_id="copy",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Copyright 2026 Example Publishing. ISBN 123-4-5678-9012-3.",
            page_number=2,
        ),
    ]

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(_doc(els), composites=[])

    text = "\n".join(chunk.content for chunk in out)
    for kept in ("Demo Writer", "Chapter 2 Data .... 11", "ISBN 123-4-5678-9012-3"):
        assert kept in text, kept


def test_ocr_split_front_matter_toc_rows_are_kept() -> None:
    """Contents rows are kept as text, never dropped."""
    els = [
        ParsedElement(
            element_id="toc1",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Chapter 6 More channels, more data! 63",
            page_number=1,
        ),
        ParsedElement(
            element_id="toc2",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Chapter 11 Software extras 110 Jurisdictions with emissions laws",
            page_number=1,
        ),
        ParsedElement(
            element_id="real",
            element_type=ElementType.NARRATIVE_TEXT,
            text="GPS speed, brake pressure, RPM, and throttle position channels are logged.",
            page_number=4,
        ),
    ]

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(
        _doc(els, total_pages=10),
        composites=[],
    )

    text = "\n".join(chunk.content for chunk in out)
    assert "Chapter 6 More channels" in text
    assert "Chapter 11 Software extras" in text
    [real] = [chunk for chunk in out if "GPS speed" in chunk.content]
    assert "Chapter 6" not in real.content or real.page_number == 1


def test_back_matter_index_fragments_are_kept() -> None:
    """Index text stays as index body, never dropped."""
    els = [
        ParsedElement(
            element_id="index1",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Throttle position 9,10,21, 50, 63, 76-83, 110, 112-116, 135,138, 140",
            page_number=74,
        ),
        ParsedElement(
            element_id="index2",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Wheelspin 40,48-49, 82, 103, 111 Oversteer 49, 56, 81, 82, 89, 90",
            page_number=74,
        ),
        ParsedElement(
            element_id="index3",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Roll 13, 120, 140 Cornering behaviour 59",
            page_number=74,
        ),
        ParsedElement(
            element_id="index4",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Pi VIDS 128 Position sensors 21 Bar graph 34 Brake sensors 125",
            page_number=74,
        ),
        ParsedElement(
            element_id="index5",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Simulations 144-145 Driver coaching 133-136",
            page_number=74,
        ),
        ParsedElement(
            element_id="real",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Throttle position traces identify where the driver lifts before corner entry.",
            page_number=20,
        ),
    ]

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(
        _doc(els, total_pages=74),
        composites=[],
    )

    text = "\n".join(chunk.content for chunk in out)
    for line in (
        "Throttle position traces",
        "Wheelspin 40",
        "Position sensors 21",
        "Simulations 144-145",
    ):
        assert line in text, line


def test_long_front_matter_rights_block_is_kept() -> None:
    """Nothing the chunker is given is dropped for what it
    says; a rights notice is kept (its first sentence here is the book's)."""
    els = [
        ParsedElement(
            element_id="rights",
            element_type=ElementType.NARRATIVE_TEXT,
            text=(
                "How to measure grip, balance and stability with low-cost techniques. "
                "This work is copyright Jordan Avery 2020. ALL RIGHTS RESERVED. "
                "Any unauthorised reprint or use of this material is prohibited. "
                "No part of this booklet may be reproduced or transmitted in any form "
                "by any means without express written permission from the publisher."
            ),
            page_number=5,
        ),
    ]

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(_doc(els), composites=[])

    assert len(out) == 1
    assert "How to measure grip" in out[0].content
    assert "ALL RIGHTS RESERVED" in out[0].content


def test_front_matter_text_before_a_rights_notice_is_kept() -> None:
    """The book's text is never cut off at a rights clause."""
    els = [
        ParsedElement(
            element_id="front",
            element_type=ElementType.NARRATIVE_TEXT,
            text=(
                "The book discusses driver neck strength, G-force tolerance, and "
                "concussion recovery for race drivers.\n\n"
                "All rights reserved. No part of this book may be reprinted or reproduced."
            ),
            page_number=2,
        ),
    ]

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(_doc(els), composites=[])

    assert len(out) == 1
    assert "driver neck strength" in out[0].content
    assert "All rights reserved" in out[0].content


def test_private_use_pdf_glyphs_are_normalized_before_embedding() -> None:
    els = [
        ParsedElement(
            element_id="glyphs",
            element_type=ElementType.NARRATIVE_TEXT,
            text="\ue049e driver resear\ue03bed a la\ue03a of grip a\ue039er wri\ue03cen notes.",
            page_number=5,
        ),
    ]

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(_doc(els), composites=[])

    assert len(out) == 1
    assert out[0].content == "The driver researched a lack of grip after written notes."


def test_distribution_only_footer_chunk_is_dropped() -> None:
    els = [
        ParsedElement(
            element_id="footer",
            element_type=ElementType.NARRATIVE_TEXT,
            text="This file was downloaded from example-site.org.",
            page_number=3,
        ),
    ]

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(_doc(els), composites=[])

    assert out == []


def test_distribution_tail_is_stripped_from_technical_chunk() -> None:
    els = [
        ParsedElement(
            element_id="mixed",
            element_type=ElementType.NARRATIVE_TEXT,
            text=(
                "Measured downforce and reduced drag came from front and rear undertrays, "
                "a front spoiler, external air curtains, rear fins, and separation edges. "
                "This file was downloaded from example-site.org."
            ),
            page_number=3,
        ),
    ]

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(_doc(els), composites=[])

    assert len(out) == 1
    assert "Measured downforce" in out[0].content
    assert "example-site.org" not in out[0].content
    assert "downloaded from" not in out[0].content


def test_front_matter_author_bio_is_kept() -> None:
    """An author biography is text of the book, kept."""
    els = [
        ParsedElement(
            element_id="bio",
            element_type=ElementType.NARRATIVE_TEXT,
            text=(
                "Jordan Avery, who has written more than 20 books, started their working "
                "life freelancing for hobby magazines. They then worked as a "
                "secondary school teacher before becoming a full-time technical writer."
            ),
            page_number=3,
        ),
    ]

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(_doc(els), composites=[])

    assert len(out) == 1
    assert "who has written more than 20 books" in out[0].content


def test_front_matter_contact_block_is_kept() -> None:
    """Imprint contact details are kept as text."""
    els = [
        ParsedElement(
            element_id="contact",
            element_type=ElementType.NARRATIVE_TEXT,
            text=(
                "By Sam Ellis 100 Example Street Springfield, ST 00001-0001 USA "
                "E-mail: CustomerService@example.org Phone: 555-010-0142 Fax: 555-010-0143"
            ),
            page_number=2,
        ),
    ]

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(
        _doc(els, total_pages=100),
        composites=[],
    )

    assert len(out) == 1
    assert "CustomerService@example.org" in out[0].content


def test_tiny_junk_dropped_but_short_technical_chunks_preserved() -> None:
    els = [
        ParsedElement(
            element_id="junk",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Acknowledgments",
            page_number=2,
        ),
        ParsedElement(
            element_id="tech",
            element_type=ElementType.NARRATIVE_TEXT,
            text="GPS Speed, brake pressure, RPM, throttle position.",
            page_number=3,
        ),
    ]

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(_doc(els), composites=[])

    assert len(out) == 1
    assert out[0].page_number == 3
    assert "GPS Speed" in out[0].content


def test_back_matter_index_fragment_with_technical_terms_is_kept() -> None:
    """Index text stays as index body."""
    els = [
        ParsedElement(
            element_id="idx",
            element_type=ElementType.NARRATIVE_TEXT,
            text=(
                "Trailers\n\n"
                "with brakes (Cont.)\n\n"
                "electric brakes\n277\n\n"
                "surge brakes\n277-278\n\n"
                "without brakes\n275-276"
            ),
            page_number=433,
        ),
    ]

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(
        _doc(els, total_pages=434),
        composites=[],
    )

    assert len(out) == 1
    assert "surge brakes" in out[0].content


def test_short_back_matter_index_line_is_kept() -> None:
    """Index text stays as index body."""
    els = [
        ParsedElement(
            element_id="idx",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Trailers with brakes braking dynamics of 276-278",
            page_number=432,
        ),
    ]

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(
        _doc(els, total_pages=434),
        composites=[],
    )

    assert [chunk.content for chunk in out] == ["Trailers with brakes braking dynamics of 276-278"]


def test_short_back_matter_index_heading_without_page_ref_is_kept() -> None:
    """Index text stays as index body."""
    els = [
        ParsedElement(
            element_id="idx",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Compression Deceleration Driver reaction time four phases",
            page_number=421,
        ),
    ]

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(
        _doc(els, total_pages=434),
        composites=[],
    )

    assert [chunk.content for chunk in out] == [
        "Compression Deceleration Driver reaction time four phases"
    ]


def test_short_back_matter_continued_index_heading_is_kept() -> None:
    """Index text stays as index body."""
    els = [
        ParsedElement(
            element_id="idx",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Calipers Compression Deceleration Driver reaction time (Cont.)",
            page_number=422,
        ),
    ]

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(
        _doc(els, total_pages=434),
        composites=[],
    )

    assert [chunk.content for chunk in out] == [
        "Calipers Compression Deceleration Driver reaction time (Cont.)"
    ]


def test_back_matter_index_before_the_last_three_pages_is_kept() -> None:
    """Index text stays as index body."""
    els = [
        ParsedElement(
            element_id="idx",
            element_type=ElementType.NARRATIVE_TEXT,
            text=(
                "Brake balance 212-214 hydraulic ABS 353-360 wheel-lockup analysis 328-343 "
                "Automatic slack adjuster 185 Average displacement formula for"
            ),
            page_number=410,
        ),
        ParsedElement(
            element_id="real",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Brake pressure traces identify where the driver releases pedal force.",
            page_number=100,
        ),
    ]

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(
        _doc(els, total_pages=434),
        composites=[],
    )

    text = "\n".join(chunk.content for chunk in out)
    assert "Brake pressure traces" in text
    assert "Automatic slack adjuster 185" in text


def test_chunker_waits_for_sentence_boundary_before_target_flush() -> None:
    els = [
        ParsedElement(
            element_id="a",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Brake response begins when the driver applies pedal force and line pressure has",
            page_number=10,
        ),
        ParsedElement(
            element_id="b",
            element_type=ElementType.NARRATIVE_TEXT,
            text="risen enough to move the caliper piston.",
            page_number=10,
        ),
        ParsedElement(
            element_id="c",
            element_type=ElementType.NARRATIVE_TEXT,
            text="The next paragraph can start a separate retrieval unit.",
            page_number=10,
        ),
    ]

    out = HierarchicalChunker(
        target=12,
        max_=28,
        min_=1,
        counter=lambda text: len(text.split()),
    ).chunk(_doc(els, total_pages=20), composites=[])

    assert out
    assert "line pressure has\n\nrisen enough" in out[0].content
    assert out[0].content.endswith("caliper piston.")


def test_page_number_furniture_is_stripped_from_joined_text() -> None:
    els = [
        ParsedElement(
            element_id="a",
            element_type=ElementType.NARRATIVE_TEXT,
            text=(
                "After the reaction time elapsed, the driver applies pressure "
                "and the system has\n\n4"
            ),
            page_number=14,
        ),
        ParsedElement(
            element_id="b",
            element_type=ElementType.NARRATIVE_TEXT,
            text="risen enough to move the caliper piston.",
            page_number=15,
        ),
    ]

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(
        _doc(els, total_pages=20),
        composites=[],
    )

    assert len(out) == 1
    assert "\n4" not in out[0].content
    assert "system has\n\nrisen enough" in out[0].content


def test_incomplete_text_is_not_flushed_before_image_composite() -> None:
    els = [
        ParsedElement(
            element_id="before",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Brake line pressure has",
            page_number=10,
        ),
        ParsedElement(
            element_id="image",
            element_type=ElementType.IMAGE,
            text="",
            page_number=11,
        ),
        ParsedElement(
            element_id="after",
            element_type=ElementType.NARRATIVE_TEXT,
            text="risen enough to move the caliper piston.",
            page_number=11,
        ),
    ]
    comp = CompositeChunk(
        chunk_id="c",
        section_title="Sec",
        preceding_text="",
        vlm_description="fig",
        following_text="",
        image_id="img",
        image_ref="g/key",
        embedding_text="[IMAGE: fig]",
        token_count=3,
        page_number=11,
        source_id="s",
        element_index=1,
    )

    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(
        _doc(els, total_pages=20),
        composites=[comp],
    )

    text_chunks = [chunk for chunk in out if chunk.chunk_type == "text"]
    assert len(text_chunks) == 1
    assert "pressure has\n\nrisen enough" in text_chunks[0].content
    assert any(chunk.chunk_type == "figure_composite" for chunk in out)


def _composite(section_title: str, element_index: int) -> CompositeChunk:
    return CompositeChunk(
        chunk_id=f"c{element_index}",
        section_title=section_title,
        preceding_text="",
        vlm_description="fig",
        following_text="",
        image_id=f"img{element_index}",
        image_ref="g/key",
        embedding_text="[IMAGE: fig]",
        token_count=3,
        page_number=1,
        source_id="s",
        element_index=element_index,
    )


def test_composite_takes_the_section_the_chunker_is_in() -> None:
    """Figure composites carry the same section as the text around them; the
    builder's own title (a book-title fallback) must not override it. They were
    all "Visual", then all the book title, in image-heavy EPUBs (VOR-1427)."""
    els = [
        ParsedElement(
            element_id="h1",
            element_type=ElementType.TITLE,
            text="6 Tensor products",
            page_number=1,
            section_depth=1,
        ),
        ParsedElement(
            element_id="p1",
            element_type=ElementType.NARRATIVE_TEXT,
            text=(
                "The tensor product of two vector spaces is defined as follows. "
                "Given bases for each space, the products of basis vectors form a basis "
                "for the tensor product, and every bilinear map factors through it "
                "uniquely, which is the universal property used throughout this chapter."
            ),
            page_number=1,
        ),
    ]
    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(
        _doc(els), composites=[_composite("Linear Algebra", 1)]
    )
    figure = [c for c in out if c.chunk_type == "figure_composite"]
    text = [c for c in out if c.chunk_type == "text"]

    assert figure[0].section_title == text[0].section_title == "6 Tensor products"


def test_composite_keeps_its_own_title_when_no_section_is_open() -> None:
    els = [
        ParsedElement(
            element_id="p1",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Front matter before any heading.",
            page_number=1,
        ),
    ]
    out = HierarchicalChunker(target=1000, max_=1200, min_=1).chunk(
        _doc(els), composites=[_composite("Linear Algebra", 0)]
    )
    figure = [c for c in out if c.chunk_type == "figure_composite"]

    assert figure[0].section_title
    assert figure[0].section_title != "Visual"


def test_promoted_postal_address_keeps_street_number_and_page() -> None:
    """A promoted address is still source data, with its number and physical page."""
    elements = [
        ParsedElement(
            element_id="heading",
            element_type=ElementType.TITLE,
            text="Publisher contact",
            page_number=3,
            section_depth=1,
        ),
        ParsedElement(
            element_id="address",
            element_type=ElementType.NARRATIVE_TEXT,
            text="20 Example Road",
            page_number=3,
        ),
        ParsedElement(
            element_id="body",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Please send correspondence to the publisher at this address. "
            "Include the complete destination and the return address so that "
            "the publisher can respond to the request without delay.",
            page_number=3,
        ),
    ]
    chunks = HierarchicalChunker(counter=_word_count, min_=1).chunk(_doc(elements), composites=[])
    assert any("20 Example Road" in chunk.content for chunk in chunks)
    assert _covered_pages(chunks) == {3}


def test_short_numeric_code_output_and_wrapped_fraction_reach_chunk_content() -> None:
    """Original code output and fraction components remain at real page boundaries."""
    values = [
        (ElementType.CODE_BLOCK, "42", 22),
        (ElementType.NARRATIVE_TEXT, "The cube of 245\n/\n8", 23),
        (ElementType.NARRATIVE_TEXT, "$10.00", 24),
    ]
    elements = [
        ParsedElement(element_id=f"numeric-{i}", element_type=kind, text=text, page_number=page)
        for i, (kind, text, page) in enumerate(values)
    ]
    chunks = HierarchicalChunker(counter=_word_count, min_=1).chunk(
        _doc(elements, total_pages=24), composites=[]
    )
    for _, value, page in values:
        assert any(value in chunk.content and page in _covered_pages([chunk]) for chunk in chunks)


def test_numbered_answers_and_short_formula_titles_remain_in_chunk_bodies() -> None:
    """An answer classified as TITLE keeps its complete value and physical page."""
    elements = [
        ParsedElement(
            element_id="context",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Use these worked answers to calculate volume and circumference.",
            page_number=1,
        ),
        ParsedElement(
            element_id="answer",
            element_type=ElementType.TITLE,
            text="10 410.5024 cubic inches",
            page_number=1,
        ),
        ParsedElement(
            element_id="reference",
            element_type=ElementType.TITLE,
            text="8 Handbook page 2542",
            page_number=1,
        ),
        ParsedElement(
            element_id="formula", element_type=ElementType.TITLE, text="πD", page_number=2
        ),
        ParsedElement(
            element_id="minus", element_type=ElementType.NARRATIVE_TEXT, text="−", page_number=2
        ),
        ParsedElement(
            element_id="prime", element_type=ElementType.NARRATIVE_TEXT, text="′", page_number=2
        ),
        ParsedElement(
            element_id="tail",
            element_type=ElementType.NARRATIVE_TEXT,
            text="The result gives the circumference for the supplied diameter.",
            page_number=2,
        ),
    ]
    chunks = HierarchicalChunker(target=100, max_=120, overlap=0).chunk(
        _doc(elements), composites=[]
    )
    for text, page in [
        ("10 410.5024 cubic inches", 1),
        ("8 Handbook page 2542", 1),
        ("πD", 2),
        ("−", 2),
        ("′", 2),
    ]:
        assert any(text in chunk.content and page in _covered_pages([chunk]) for chunk in chunks)


def test_isolated_operands_and_code_numbers_survive_the_printed_page_range() -> None:
    """Page-edge folios are removed while separately parsed scalar data stays."""
    values = [
        (ElementType.NARRATIVE_TEXT, "12"),
        (ElementType.NARRATIVE_TEXT, "Calculate the quotient from these separately parsed values."),
        (ElementType.NARRATIVE_TEXT, "22"),
        (ElementType.NARRATIVE_TEXT, "/"),
        (ElementType.NARRATIVE_TEXT, "2"),
        (ElementType.CODE_BLOCK, "12"),
        (
            ElementType.NARRATIVE_TEXT,
            "These well-known operands and this code output are source values.",
        ),
        (ElementType.NARRATIVE_TEXT, "12"),
    ]
    elements = [
        ParsedElement(element_id=f"folio-value-{i}", element_type=kind, text=text, page_number=12)
        for i, (kind, text) in enumerate(values)
    ]
    chunks = HierarchicalChunker(target=100, max_=120, overlap=0).chunk(
        _doc(elements, total_pages=12), composites=[]
    )
    paragraphs = [part.strip() for chunk in chunks for part in chunk.content.split("\n\n")]
    assert paragraphs.count("12") == 1
    assert "22" in paragraphs and "2" in paragraphs and "/" in paragraphs
    assert 12 in _covered_pages(chunks)


def test_rejected_interior_numeric_title_stays_with_its_answer() -> None:
    """A source answer label typed as TITLE is data inside a physical page."""
    elements = [
        ParsedElement(
            element_id="answer-folio",
            element_type=ElementType.NARRATIVE_TEXT,
            text="233",
            page_number=233,
        ),
        ParsedElement(
            element_id="answer-context",
            element_type=ElementType.NARRATIVE_TEXT,
            text="These worked answers give the dimensions calculated for each question.",
            page_number=233,
        ),
        ParsedElement(
            element_id="answer-label", element_type=ElementType.TITLE, text="32", page_number=233
        ),
        ParsedElement(
            element_id="answer-value",
            element_type=ElementType.NARRATIVE_TEXT,
            text="1.5705 inches",
            page_number=233,
        ),
        ParsedElement(
            element_id="answer-tail",
            element_type=ElementType.NARRATIVE_TEXT,
            text="Use these values when checking the dimensions of the finished part.",
            page_number=233,
        ),
    ]
    chunks = HierarchicalChunker(target=100, max_=120, overlap=0).chunk(
        _doc(elements, total_pages=233), composites=[]
    )
    assert any("32\n\n1.5705 inches" in chunk.content for chunk in chunks)
    assert not any(
        part.strip() == "233" for chunk in chunks for part in chunk.content.split("\n\n")
    )
    assert 233 in _covered_pages(chunks)


def test_detached_angular_units_and_reference_delimiter_remain_source_data() -> None:
    """Docling separates degrees and a reference parenthesis from their prose."""
    source = [
        "For angles 45 to 90, read the table from the bottom using the function labels.",
        "° °",
        "Kd = 12,000 ( Table 8, page",
        ")",
        "1090",
        "Apply the coefficient to the cutting force calculation for this material.",
    ]
    elements = [
        ParsedElement(
            element_id=f"angular-{i}",
            element_type=ElementType.NARRATIVE_TEXT,
            text=text,
            page_number=52,
        )
        for i, text in enumerate(source)
    ]
    chunks = HierarchicalChunker(target=100, max_=120, overlap=0).chunk(
        _doc(elements, total_pages=200), composites=[]
    )
    paragraphs = [part.strip() for chunk in chunks for part in chunk.content.split("\n\n")]
    assert "° °" in paragraphs
    assert ")" in paragraphs
    assert "1090" in paragraphs
    assert 52 in _covered_pages(chunks)

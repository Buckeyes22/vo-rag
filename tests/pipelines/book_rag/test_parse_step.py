"""Unit tests for the book_rag parse step.

Verifies do_parse_document and helper functions using fake Garage and
fake ParserRouter (no real PDF parsing).
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock

import pytest

from vo_rag.core.parsing.contract import (
    ElementType,
    PageQA,
    ParsedDocument,
    ParsedElement,
)
from vo_rag.pipelines.book_rag.steps.parse import (
    FORWARD_PATH_DECISION_DOC,
    _apply_parse_quality_gate,
    _corruption_sample_text,
    _enforce_content_corruption,
    _enforce_parse_red_flags,
    _enforce_pdf_forward_path,
    _forward_path_metadata,
    _parsed_text_available,
    do_parse_document,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_garage() -> AsyncMock:
    garage = AsyncMock()
    garage.get_bytes.return_value = b"%PDF-1.4 fake pdf content"
    garage.put_bytes.return_value = None
    return garage


@pytest.fixture
def fake_router() -> AsyncMock:
    router = AsyncMock()
    router.parse.return_value = ParsedDocument(
        document_id="",
        source_id="",
        title="Test Doc",
        total_pages=1,
        elements=[
            ParsedElement(
                element_id="e1",
                element_type=ElementType.NARRATIVE_TEXT,
                text="Hello world",
                page_number=1,
            )
        ],
        images=[],
        page_qa=[
            PageQA(
                page_number=1,
                ocr_confidence=0.95,
                parser_used="docling",
                text_element_count=1,
            )
        ],
        markdown="Hello world",
        heading_tree=[],
        parser_used="docling",
        metadata={"parser_backend": "docling"},
    )
    return router


@pytest.fixture
def sample_doc() -> ParsedDocument:
    return ParsedDocument(
        document_id="doc-id",
        source_id="src-id",
        title="Sample",
        total_pages=1,
        elements=[
            ParsedElement(
                element_id="e1",
                element_type=ElementType.NARRATIVE_TEXT,
                text="Hello world",
                page_number=1,
            )
        ],
        images=[],
        page_qa=[
            PageQA(
                page_number=1,
                ocr_confidence=0.95,
                parser_used="docling",
                text_element_count=1,
            )
        ],
        markdown="Hello world",
        heading_tree=[],
        parser_used="docling",
        metadata={"parser_backend": "docling"},
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class TestParsedTextAvailable:
    def test_markdown_present(self) -> None:
        doc = ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=1,
            elements=[],
            images=[],
            page_qa=[],
            markdown="some text",
            heading_tree=[],
            parser_used="p",
        )
        assert _parsed_text_available(doc) is True

    def test_no_markdown_but_element_text(self) -> None:
        doc = ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=1,
            elements=[
                ParsedElement(
                    element_id="e1",
                    element_type=ElementType.NARRATIVE_TEXT,
                    text="content",
                    page_number=1,
                )
            ],
            images=[],
            page_qa=[],
            markdown="",
            heading_tree=[],
            parser_used="p",
        )
        assert _parsed_text_available(doc) is True

    def test_no_text_at_all(self) -> None:
        doc = ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=1,
            elements=[],
            images=[],
            page_qa=[],
            markdown="",
            heading_tree=[],
            parser_used="p",
        )
        assert _parsed_text_available(doc) is False


class TestCorruptionSampleText:
    def test_prefers_markdown(self) -> None:
        doc = ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=1,
            elements=[],
            images=[],
            page_qa=[],
            markdown="markdown content",
            heading_tree=[],
            parser_used="p",
        )
        assert _corruption_sample_text(doc) == "markdown content"

    def test_falls_back_to_elements(self) -> None:
        doc = ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=1,
            elements=[
                ParsedElement(
                    element_id="e1",
                    element_type=ElementType.NARRATIVE_TEXT,
                    text="element text",
                    page_number=1,
                )
            ],
            images=[],
            page_qa=[],
            markdown="",
            heading_tree=[],
            parser_used="p",
        )
        assert _corruption_sample_text(doc) == "element text"

    def test_respects_max_chars(self) -> None:
        long_text = "x" * 30000
        doc = ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=1,
            elements=[],
            images=[],
            page_qa=[],
            markdown=long_text,
            heading_tree=[],
            parser_used="p",
        )
        sample = _corruption_sample_text(doc)
        assert len(sample) == 20000


class TestForwardPathMetadata:
    def test_docling_forward_path_metadata(self, sample_doc: ParsedDocument) -> None:
        meta = _forward_path_metadata(sample_doc)
        assert meta["decision_doc"] == FORWARD_PATH_DECISION_DOC
        assert meta["ocr_stack"] == "docling-layout-pdf"
        assert meta["parser_used"] == "docling"
        assert meta["parser_backend"] == "docling"
        assert meta["parse_element_count"] == 1
        assert meta["parse_image_count"] == 0
        assert meta["total_pages"] == 1

    def test_docling_stack(self) -> None:
        doc = ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=1,
            elements=[],
            images=[],
            page_qa=[],
            markdown="",
            heading_tree=[],
            parser_used="docling",
        )
        meta = _forward_path_metadata(doc)
        assert meta["ocr_stack"] == "docling-layout-pdf"

    def test_docling_ocr_stack(self) -> None:
        """VOR-1427: scanned PDFs / image-only EPUBs OCR'd on the GPU."""
        doc = ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=1,
            elements=[],
            images=[],
            page_qa=[],
            markdown="",
            heading_tree=[],
            parser_used="docling-ocr",
        )
        meta = _forward_path_metadata(doc)
        assert meta["ocr_stack"] == "docling-rapidocr-torch-cuda"


class TestApplyParseQualityGate:
    def test_quarantines_low_confidence_page(self) -> None:
        doc = ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=1,
            elements=[],
            images=[],
            page_qa=[
                PageQA(
                    page_number=1,
                    ocr_confidence=0.1,
                    parser_used="p",
                    text_element_count=0,
                    table_count=1,
                )
            ],
            markdown="",
            heading_tree=[],
            parser_used="p",
        )
        result = _apply_parse_quality_gate(doc)
        assert 1 in result.quarantined_pages
        assert result.metadata["parse_quality_gate"]["quarantined_page_count"] == 1

    def test_clean_page_passes(self, sample_doc: ParsedDocument) -> None:
        result = _apply_parse_quality_gate(sample_doc)
        assert result.quarantined_pages == []
        assert result.metadata["parse_quality_gate"]["quarantined_page_count"] == 0


class TestEnforceContentCorruption:
    def test_clean_text_passes(self, sample_doc: ParsedDocument) -> None:
        _enforce_content_corruption(sample_doc)  # no raise

    def test_cid_glyph_soup_raises(self) -> None:
        doc = ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=1,
            elements=[],
            images=[],
            page_qa=[],
            markdown="YHUWLFDO\n\nFF]\n\nWDQJHQWLDO\n\nVXEVFULSW\x03V\n\x11\x03)LJXUH",
            heading_tree=[],
            parser_used="p",
        )
        with pytest.raises(RuntimeError, match="content-corruption gate"):
            _enforce_content_corruption(doc)

    def test_control_char_raises_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VORAG_BOOK_RAG_DISABLE_CORRUPTION_GATE", raising=False)
        doc = ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=1,
            elements=[],
            images=[],
            page_qa=[],
            markdown="Hello \x00\x01\x02 world",
            heading_tree=[],
            parser_used="p",
        )
        with pytest.raises(RuntimeError, match="content-corruption gate"):
            _enforce_content_corruption(doc)

    def test_disable_corruption_gate_bypasses_corrupt_input(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VORAG_BOOK_RAG_DISABLE_CORRUPTION_GATE", "1")
        doc = ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=1,
            elements=[],
            images=[],
            page_qa=[],
            markdown="Hello \x00\x01\x02 world",
            heading_tree=[],
            parser_used="p",
        )

        _enforce_content_corruption(doc)


class TestEnforcePdfForwardPath:
    def test_docling_with_text_passes(self) -> None:
        doc = ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=1,
            elements=[
                ParsedElement(
                    element_id="e1",
                    element_type=ElementType.NARRATIVE_TEXT,
                    text="content",
                    page_number=1,
                )
            ],
            images=[],
            page_qa=[
                PageQA(
                    page_number=1,
                    ocr_confidence=0.95,
                    parser_used="docling",
                    text_element_count=1,
                )
            ],
            markdown="",
            heading_tree=[],
            parser_used="docling",
        )
        _enforce_pdf_forward_path(doc)  # no raise

    def test_docling_no_text_raises(self) -> None:
        doc = ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=1,
            elements=[],
            images=[],
            page_qa=[
                PageQA(
                    page_number=1,
                    ocr_confidence=0.95,
                    parser_used="docling",
                    text_element_count=0,
                )
            ],
            markdown="",
            heading_tree=[],
            parser_used="docling",
        )
        with pytest.raises(RuntimeError, match="docling PDF parse produced no text"):
            _enforce_pdf_forward_path(doc)

    def test_docling_ocr_with_text_passes(self) -> None:
        """VOR-1427: docling-ocr (scanned PDF / image-only EPUB GPU OCR) gets
        the same text + page-coverage checks as docling."""
        doc = ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=1,
            elements=[
                ParsedElement(
                    element_id="e1",
                    element_type=ElementType.NARRATIVE_TEXT,
                    text="ocr content",
                    page_number=1,
                )
            ],
            images=[],
            page_qa=[
                PageQA(
                    page_number=1,
                    ocr_confidence=1.0,
                    parser_used="docling-ocr",
                    text_element_count=1,
                )
            ],
            markdown="",
            heading_tree=[],
            parser_used="docling-ocr",
        )
        _enforce_pdf_forward_path(doc)  # no raise

    def test_docling_ocr_no_text_raises(self) -> None:
        doc = ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=1,
            elements=[],
            images=[],
            page_qa=[
                PageQA(
                    page_number=1,
                    ocr_confidence=1.0,
                    parser_used="docling-ocr",
                    text_element_count=0,
                )
            ],
            markdown="",
            heading_tree=[],
            parser_used="docling-ocr",
        )
        with pytest.raises(RuntimeError, match="docling PDF parse produced no text"):
            _enforce_pdf_forward_path(doc)

    def test_docling_ocr_sparse_content_raises(self) -> None:
        doc = ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=20,
            elements=[
                ParsedElement(
                    element_id="e1",
                    element_type=ElementType.NARRATIVE_TEXT,
                    text="only one page has text",
                    page_number=1,
                )
            ],
            images=[],
            page_qa=[
                PageQA(
                    page_number=1,
                    ocr_confidence=1.0,
                    parser_used="docling-ocr",
                    text_element_count=1,
                    table_count=0,
                )
            ]
            + [
                PageQA(
                    page_number=i,
                    ocr_confidence=1.0,
                    parser_used="docling-ocr",
                    text_element_count=0,
                    table_count=0,
                )
                for i in range(2, 21)
            ],
            markdown="only one page has text",
            heading_tree=[],
            parser_used="docling-ocr",
        )
        _enforce_pdf_forward_path(doc)
        assert doc.metadata["quality_warnings"][0]["code"] == "low_page_coverage"

    def test_unapproved_parser_raises(self) -> None:
        doc = ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=1,
            elements=[],
            images=[],
            page_qa=[],
            markdown="content",
            heading_tree=[],
            parser_used="mineru_pipeline",
            metadata={},
        )
        with pytest.raises(RuntimeError, match="did not use an approved forward path"):
            _enforce_pdf_forward_path(doc)

    def test_docling_sparse_content_raises(self) -> None:
        doc = ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=20,
            elements=[
                ParsedElement(
                    element_id="e1",
                    element_type=ElementType.NARRATIVE_TEXT,
                    text="only one page has text",
                    page_number=1,
                )
            ],
            images=[],
            page_qa=[
                PageQA(
                    page_number=1,
                    ocr_confidence=0.95,
                    parser_used="docling",
                    text_element_count=1,
                    table_count=0,
                )
            ]
            + [
                PageQA(
                    page_number=i,
                    ocr_confidence=0.95,
                    parser_used="docling",
                    text_element_count=0,
                    table_count=0,
                )
                for i in range(2, 21)
            ],
            markdown="only one page has text",
            heading_tree=[],
            parser_used="docling",
        )
        _enforce_pdf_forward_path(doc)
        assert doc.metadata["quality_warnings"][0]["code"] == "low_page_coverage"


class TestEnforceParseRedFlags:
    def test_blocks_costly_downstream_work(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VORAG_BOOK_RAG_FAIL_ON_PARSE_RED_FLAGS", "1")
        parsed = ParsedDocument(
            document_id="doc-red-flag",
            source_id="src-red-flag",
            title="Bad Image Ratio",
            total_pages=286,
            elements=[],
            images=[],
            page_qa=[],
            markdown="",
            heading_tree=[],
            parser_used="docling",
            metadata={
                "parser_backend": "docling",
                "native_pdf_image_red_flags": ["high_native_image_count_ratio"],
                "native_pdf_images_available": 4_290,
                "native_pdf_images_available_per_page": 15.0,
                "native_pdf_images_max_images_on_page": 80,
            },
        )

        _enforce_parse_red_flags(parsed)
        assert parsed.metadata["quality_warnings"][0]["code"] == "native_pdf_red_flags"

    def test_can_be_disabled_for_manual_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VORAG_BOOK_RAG_FAIL_ON_PARSE_RED_FLAGS", "0")
        parsed = ParsedDocument(
            document_id="doc-red-flag",
            source_id="src-red-flag",
            title="Manual Run",
            total_pages=10,
            elements=[],
            images=[],
            page_qa=[],
            markdown="",
            heading_tree=[],
            parser_used="docling",
            metadata={
                "parser_backend": "docling",
                "native_pdf_image_count_warning": "high_native_image_count_ratio",
            },
        )

        _enforce_parse_red_flags(parsed)


# ---------------------------------------------------------------------------
# do_parse_document
# ---------------------------------------------------------------------------


class TestDoParseDocument:
    async def test_happy_path(self, fake_garage: AsyncMock, fake_router: AsyncMock) -> None:
        result = await do_parse_document(
            envelope_id="env-123",
            garage_key="uploads/test.pdf",
            source_id="src-123",
            garage=fake_garage,
            bucket="test-bucket",
            parser_router=fake_router,
        )
        fake_garage.get_bytes.assert_awaited_once_with("test-bucket", "uploads/test.pdf")
        fake_router.parse.assert_awaited_once()
        fake_garage.put_bytes.assert_awaited_once()

        expected_hash = hashlib.sha256(b"%PDF-1.4 fake pdf content").hexdigest()
        assert result["parsed_garage_key"] == "parsed/env-123/parsed.json"
        assert result["document_id"] == expected_hash
        assert result["source_id"] == "src-123"
        assert result["title"] == "Test Doc"
        assert result["parser_used"] == "docling"
        assert result["parser_backend"] == "docling"
        assert result["image_count"] == 0
        assert result["element_count"] == 1
        assert "duration_ms" in result
        assert result["quarantined_page_count"] == 0
        assert result["forward_path"]["parser_used"] == "docling"

    async def test_corruption_gate_blocks(self, fake_garage: AsyncMock) -> None:
        router = AsyncMock()
        router.parse.return_value = ParsedDocument(
            document_id="",
            source_id="",
            title="Corrupt",
            total_pages=1,
            elements=[],
            images=[],
            page_qa=[],
            markdown="YHUWLFDO\n\nFF]\n\nWDQJHQWLDO\n\nVXEVFULSW\x03V\n\x11\x03)LJXUH",
            heading_tree=[],
            parser_used="docling",
            metadata={"parser_backend": "docling"},
        )
        with pytest.raises(RuntimeError, match="content-corruption gate"):
            await do_parse_document(
                envelope_id="env-123",
                garage_key="uploads/bad.pdf",
                source_id="src-123",
                garage=fake_garage,
                bucket="test-bucket",
                parser_router=router,
            )

    async def test_unapproved_parser_blocks(self, fake_garage: AsyncMock) -> None:
        router = AsyncMock()
        router.parse.return_value = ParsedDocument(
            document_id="",
            source_id="",
            title="Bad Parser",
            total_pages=1,
            elements=[
                ParsedElement(
                    element_id="e1",
                    element_type=ElementType.NARRATIVE_TEXT,
                    text="content",
                    page_number=1,
                )
            ],
            images=[],
            page_qa=[],
            markdown="content",
            heading_tree=[],
            parser_used="mineru_pipeline",
            metadata={},
        )
        with pytest.raises(RuntimeError, match="did not use an approved forward path"):
            await do_parse_document(
                envelope_id="env-123",
                garage_key="uploads/bad.pdf",
                source_id="src-123",
                garage=fake_garage,
                bucket="test-bucket",
                parser_router=router,
            )

    async def test_non_pdf_does_not_enforce_pdf_path(
        self, fake_garage: AsyncMock, fake_router: AsyncMock
    ) -> None:
        fake_garage.get_bytes.return_value = b"not a pdf"
        result = await do_parse_document(
            envelope_id="env-123",
            garage_key="uploads/test.txt",
            source_id="src-123",
            garage=fake_garage,
            bucket="test-bucket",
            parser_router=fake_router,
        )
        assert result["parsed_garage_key"] == "parsed/env-123/parsed.json"

    async def test_docling_ocr_from_epub_source_still_enforces_pdf_gates(
        self, fake_garage: AsyncMock
    ) -> None:
        """An image-only EPUB's OCR result must not
        skip the no-text gate just because the original upload wasn't a PDF."""
        fake_garage.get_bytes.return_value = b"PK\x03\x04 fake epub bytes, not a pdf"
        router = AsyncMock()
        router.parse.return_value = ParsedDocument(
            document_id="",
            source_id="",
            title="Empty OCR",
            total_pages=1,
            elements=[],
            images=[],
            page_qa=[
                PageQA(
                    page_number=1,
                    ocr_confidence=1.0,
                    parser_used="docling-ocr",
                    text_element_count=0,
                )
            ],
            markdown="",
            heading_tree=[],
            parser_used="docling-ocr",
            metadata={"source_format": "epub_image_only"},
        )
        with pytest.raises(RuntimeError, match="docling PDF parse produced no text"):
            await do_parse_document(
                envelope_id="env-123",
                garage_key="uploads/scan.epub",
                source_id="src-123",
                garage=fake_garage,
                bucket="test-bucket",
                parser_router=router,
            )

    async def test_quality_gate_populates_quarantine(self, fake_garage: AsyncMock) -> None:
        router = AsyncMock()
        router.parse.return_value = ParsedDocument(
            document_id="",
            source_id="",
            title="Low Quality",
            total_pages=1,
            elements=[
                ParsedElement(
                    element_id="e1",
                    element_type=ElementType.NARRATIVE_TEXT,
                    text="some content",
                    page_number=1,
                )
            ],
            images=[],
            page_qa=[
                PageQA(
                    page_number=1,
                    ocr_confidence=0.1,
                    parser_used="docling",
                    text_element_count=0,
                    table_count=1,
                )
            ],
            markdown="some content",
            heading_tree=[],
            parser_used="docling",
            metadata={"parser_backend": "docling"},
        )
        result = await do_parse_document(
            envelope_id="env-123",
            garage_key="uploads/test.pdf",
            source_id="src-123",
            garage=fake_garage,
            bucket="test-bucket",
            parser_router=router,
        )
        assert result["quarantined_page_count"] == 1


# ---------------------------------------------------------------------------
# VOR-1434: the page count reaches the ingest metrics before the parse
# ---------------------------------------------------------------------------


class _StopAfterPageCount(Exception):
    pass


async def test_pdf_page_count_is_recorded_before_the_parser_runs(
    fake_garage: AsyncMock, fake_router: AsyncMock
) -> None:
    fitz = pytest.importorskip("fitz")
    from vo_rag.core.observability.metrics import current_ingest_recorder, ingest_recording

    doc = fitz.open()
    for _ in range(3):
        doc.new_page()
    fake_garage.get_bytes.return_value = doc.tobytes()
    seen: list[int | None] = []

    async def _parse(*_: object) -> ParsedDocument:
        recorder = current_ingest_recorder()
        seen.append(recorder.pages if recorder else None)
        raise _StopAfterPageCount

    fake_router.parse.side_effect = _parse
    with ingest_recording(), pytest.raises(_StopAfterPageCount):
        await do_parse_document(
            envelope_id="env-1",
            garage_key="uploads/book.pdf",
            source_id="src-1",
            garage=fake_garage,
            bucket="b",
            parser_router=fake_router,
        )
    assert seen == [3]


async def test_unreadable_pdf_page_count_never_blocks_the_parse(
    fake_garage: AsyncMock, fake_router: AsyncMock
) -> None:
    from vo_rag.core.observability.metrics import ingest_recording

    with ingest_recording() as recorder:
        result = await do_parse_document(
            envelope_id="env-123",
            garage_key="uploads/test.pdf",
            source_id="src-123",
            garage=fake_garage,  # "%PDF-1.4 fake pdf content": fitz can't open it
            bucket="test-bucket",
            parser_router=fake_router,
        )
    assert result["parsed_garage_key"] == "parsed/env-123/parsed.json"
    # The container count failed, so the page count comes from the parse itself.
    assert recorder.pages == 1


async def test_a_non_pdf_records_the_parsed_page_count(
    fake_garage: AsyncMock, fake_router: AsyncMock
) -> None:
    from vo_rag.core.observability.metrics import ingest_recording

    fake_garage.get_bytes.return_value = b"plain text book"
    fake_router.parse.return_value.total_pages = 240
    with ingest_recording() as recorder:
        await do_parse_document(
            envelope_id="env-1",
            garage_key="uploads/book.txt",
            source_id="src-1",
            garage=fake_garage,
            bucket="b",
            parser_router=fake_router,
        )
    assert recorder.pages == 240


async def test_original_mismatch_fails_before_parser_or_cache_write(
    fake_garage: AsyncMock, fake_router: AsyncMock
) -> None:
    """A stale original cannot be attributed to a different canonical source."""
    with pytest.raises(ValueError, match="registered source identity"):
        await do_parse_document(
            envelope_id="env",
            garage_key="book.pdf",
            source_id="vo-rag-corpus-sha256:" + "0" * 64,
            garage=fake_garage,
            bucket="artifacts",
            parser_router=fake_router,
        )
    fake_router.parse.assert_not_awaited()
    fake_garage.put_bytes.assert_not_awaited()


async def test_parse_binding_hashes_exact_persisted_bytes_and_element_occurrences(
    fake_garage: AsyncMock, fake_router: AsyncMock
) -> None:
    from vo_rag.pipelines.book_rag.parse_provenance import accepted_element_inventory_sha256

    parsed = fake_router.parse.return_value
    original_hash = hashlib.sha256(fake_garage.get_bytes.return_value).hexdigest()
    result = await do_parse_document(
        envelope_id="env",
        garage_key="book.pdf",
        source_id="vo-rag-corpus-sha256:" + original_hash,
        garage=fake_garage,
        bucket="artifacts",
        parser_router=fake_router,
    )
    stored = fake_garage.put_bytes.await_args.args[2]
    binding = result["parse_provenance"]
    assert binding["original_sha256"] == original_hash
    assert binding["accepted_parse_sha256"] == hashlib.sha256(stored).hexdigest()
    assert binding["original_to_parse_fidelity"] == "unverified"
    assert binding["parse_to_chunk_intervals"] == "unverified"
    before = accepted_element_inventory_sha256(parsed)
    parsed.elements.append(parsed.elements[0].model_copy())
    assert accepted_element_inventory_sha256(parsed) != before

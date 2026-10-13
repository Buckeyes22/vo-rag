"""Shared ``ParserRouter`` test doubles (VOR-1427).

``test_router.py`` and ``test_router_scanned_pdf.py`` both dispatch through a
fake ``docling`` / ``epub`` pair; this module is the single definition so the
two files don't drift out of sync. Not a test module itself (no ``test_``
prefix -- pytest never collects it).
"""

from __future__ import annotations

from pathlib import Path

from vo_rag.core.parsing.contract import ElementType, PageQA, ParsedDocument, ParsedElement


class FakeParser:
    def __init__(self, label: str) -> None:
        self.label = label
        self.called = False

    async def parse(self, file_path: str) -> ParsedDocument:
        _ = file_path
        self.called = True
        return ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=1,
            elements=[
                ParsedElement(
                    element_id="e",
                    element_type=ElementType.NARRATIVE_TEXT,
                    text="hello",
                    page_number=1,
                )
            ],
            images=[],
            page_qa=[PageQA(page_number=1, ocr_confidence=0.9, parser_used=self.label)],
            markdown="",
            heading_tree=[],
            parser_used=self.label,
        )


class FakeEbookParser(FakeParser):
    async def parse(self, file_path: str) -> ParsedDocument:
        self.label = "mobi" if Path(file_path).suffix.lower() == ".mobi" else "epub"
        return await super().parse(file_path)


class FakeDocling:
    def __init__(self) -> None:
        self.called = False
        self.called_fmt: str | None = None
        self.called_file_hash: str | None = None
        self.scanned_called = False
        self.scanned_file_path: str | None = None
        self.scanned_file_hash: str | None = None

    async def parse(
        self, file_path: str, fmt: str, *, file_hash: str | None = None
    ) -> ParsedDocument:
        _ = file_path
        self.called = True
        self.called_fmt = fmt
        self.called_file_hash = file_hash
        return ParsedDocument(
            document_id="d",
            source_id="s",
            title="t",
            total_pages=1,
            elements=[],
            images=[],
            page_qa=[PageQA(page_number=1, ocr_confidence=1.0, parser_used="docling")],
            markdown="",
            heading_tree=[],
            parser_used="docling",
        )

    async def parse_scanned(self, file_path: str, *, file_hash: str) -> ParsedDocument:
        self.scanned_called = True
        self.scanned_file_path = file_path
        self.scanned_file_hash = file_hash
        return ParsedDocument(
            document_id=file_hash,
            source_id="s",
            title="t",
            total_pages=1,
            elements=[
                ParsedElement(
                    element_id="e",
                    element_type=ElementType.NARRATIVE_TEXT,
                    text="ocr text",
                    page_number=1,
                )
            ],
            images=[],
            page_qa=[PageQA(page_number=1, ocr_confidence=1.0, parser_used="docling-ocr")],
            markdown="ocr text",
            heading_tree=[],
            parser_used="docling-ocr",
        )

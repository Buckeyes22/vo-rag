"""Page-level and document-level quality gate for vo-rag parsers.

Quarantines pages on:
- low OCR confidence
- zero text on a non-blank page (figures/tables present)
- reading-order inversion (currently a no-op stub)

Warns (non-quarantine) on:
- detected-but-not-extracted tables/figures (re-routing signal)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vo_rag.core.parsing.contract import ElementType, PageQA, ParsedDocument, ParsedElement
from vo_rag.core.parsing.reading_order import reading_order_broken

Severity = Literal["pass", "warn", "quarantine"]

_TEXT_ELEMENT_TYPES = (
    ElementType.NARRATIVE_TEXT,
    ElementType.TITLE,
    ElementType.LIST_ITEM,
    ElementType.TABLE,
)


@dataclass(frozen=True)
class PageVerdict:
    passed: bool
    reasons: list[str]
    severity: Severity


_QUARANTINE_TRIGGERS = frozenset(
    {
        "ocr_confidence",
        "zero_text_on_non_blank_page",
        "reading_order_broken",
    }
)


class QualityGate:
    def __init__(
        self,
        *,
        min_ocr_confidence: float = 0.6,
        min_text_elements: int = 1,
        reroute_on_missing_tables: bool = True,
    ) -> None:
        self.min_ocr_confidence = min_ocr_confidence
        self.min_text_elements = min_text_elements
        self.reroute_on_missing_tables = reroute_on_missing_tables

    def evaluate_page(
        self,
        qa: PageQA,
        elements: list[ParsedElement],
    ) -> PageVerdict:
        reasons: list[str] = []
        if qa.ocr_confidence < self.min_ocr_confidence:
            reasons.append(f"ocr_confidence={qa.ocr_confidence:.2f}")
        # A table with text is extracted text: a page holding only a table
        # (and its running head/footer) is not "zero text". It passed only by
        # accident while running heads mapped to NARRATIVE_TEXT; since they map
        # to HEADER/FOOTER (VOR-1452) such pages were quarantined and the
        # chunker dropped their tables (VOR-1453: 25 of 70 pages of a records
        # thesis).
        page_text_count = sum(
            1
            for element in elements
            if element.page_number == qa.page_number
            and element.element_type in _TEXT_ELEMENT_TYPES
            and element.text.strip()
        )
        if page_text_count == 0 and (qa.table_count > 0 or qa.figure_count > 0):
            reasons.append("zero_text_on_non_blank_page")
        if qa.table_count > 0 and not any(
            element.element_type == ElementType.TABLE
            for element in elements
            if element.page_number == qa.page_number
        ):
            reasons.append("table_detected_but_none_extracted")
        if qa.figure_count > 0 and not any(
            element.element_type == ElementType.IMAGE
            for element in elements
            if element.page_number == qa.page_number
        ):
            reasons.append("figure_detected_but_none_extracted")
        if qa.has_multi_column and reading_order_broken(elements, qa.page_number):
            reasons.append("reading_order_broken")

        if not reasons:
            return PageVerdict(True, [], "pass")

        severe = any(reason.split("=")[0] in _QUARANTINE_TRIGGERS for reason in reasons)
        return PageVerdict(
            passed=not severe,
            reasons=reasons,
            severity="quarantine" if severe else "warn",
        )

    def evaluate_document(self, doc: ParsedDocument) -> ParsedDocument:
        for qa in doc.page_qa:
            elements = [
                element for element in doc.elements if element.page_number == qa.page_number
            ]
            verdict = self.evaluate_page(qa, elements)
            qa.warnings.extend(verdict.reasons)
            if verdict.severity == "quarantine":
                doc.quarantined_pages.append(qa.page_number)
        return doc

"""vo-rag core/parsing: format parsers (pdf/html/code/ebook).

The router has no MinerU/OLMOCR parsers and no vendor-SDK OCR forward path or
accelerated-OCR challenger fallback (VOR-1427) -- scanned PDFs go through the
OCR farm's cache or docling's own GPU OCR instead.
"""

from vo_rag.core.parsing.contract import (
    ElementType,
    ExtractedImage,
    PageQA,
    ParsedDocument,
    ParsedElement,
    element_id,
    image_id,
)
from vo_rag.core.parsing.router import ParserRouter
from vo_rag.core.parsing.text_normalization import normalize_extracted_text

__all__ = [
    # contract
    "ElementType",
    "ExtractedImage",
    "PageQA",
    "ParsedDocument",
    "ParsedElement",
    "element_id",
    "image_id",
    # text normalization
    "normalize_extracted_text",
    # router
    "ParserRouter",
]

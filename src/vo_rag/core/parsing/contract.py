"""Normalized parser output contract for vo-rag.

Every parser in ``vo_rag.core.parsing.parsers/`` (docling, EPUB) produces a
``ParsedDocument``. No parser-specific types leak past this boundary;
downstream chunking / quality-gate / image extraction consumes only what is
declared here.

"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Any

from pydantic import Field

from vo_rag.core.models import VoBaseModel


class ElementType(StrEnum):
    NARRATIVE_TEXT = "narrative_text"
    TITLE = "title"
    LIST_ITEM = "list_item"
    TABLE = "table"
    IMAGE = "image"
    FIGURE_CAPTION = "figure_caption"
    PAGE_BREAK = "page_break"
    EQUATION = "equation"
    CODE_BLOCK = "code_block"
    HEADER = "header"
    FOOTER = "footer"


class ParsedElement(VoBaseModel):
    element_id: str
    element_type: ElementType
    text: str = ""
    page_number: int | None = None
    section_depth: int = 0
    section_title: str = ""
    coordinates: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedImage(VoBaseModel):
    image_id: str
    page_number: int
    image_index: int = 0
    ref: str
    width: int | None = None
    height: int | None = None
    file_size: int | None = None
    mime_type: str = "image/png"
    caption: str | None = None
    element_index: int = 0


class PageQA(VoBaseModel):
    page_number: int
    ocr_confidence: float = Field(ge=0.0, le=1.0)
    table_count: int = 0
    figure_count: int = 0
    equation_count: int = 0
    text_element_count: int = 0
    has_multi_column: bool = False
    parser_used: str
    parse_duration_ms: int = 0
    warnings: list[str] = Field(default_factory=list)


class ParsedDocument(VoBaseModel):
    document_id: str
    source_id: str
    title: str
    author: str = ""
    total_pages: int
    elements: list[ParsedElement]
    images: list[ExtractedImage]
    page_qa: list[PageQA]
    markdown: str
    heading_tree: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)
    parser_used: str
    quarantined_pages: list[int] = Field(default_factory=list)


def element_id(content: str, page_num: int | None, position: int) -> str:
    """Deterministic element ID: SHA-256(content|page|position), 64 hex chars."""
    key = f"{content}|{page_num if page_num is not None else ''}|{position}".encode()
    return hashlib.sha256(key).hexdigest()


def image_id(image_bytes: bytes) -> str:
    """Deterministic image ID: SHA-256(bytes), 64 hex chars."""
    return hashlib.sha256(image_bytes).hexdigest()

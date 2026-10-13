"""Boilerplate-chunk detection (VOR-1427).

A "boilerplate chunk" is a whole chunk whose entire content is a book-
agnostic notice -- blank-page markers, EPUB "About This E-Book" blocks,
copyright pages, publisher marketing blocks -- that carries no book-specific
information and only pollutes retrieval. This module is the single source
of truth for what counts as one: ``normalize_chunk_text`` collapses case,
punctuation and whitespace so trivial OCR/formatting variance (extra spaces,
smart quotes, a trailing period) still matches; ``BOILERPLATE_CHUNKS`` is the
reviewed allow-list of normalized forms; ``is_boilerplate`` is the predicate
every consumer (the ingest-time filter ``chunk_without_boilerplate`` below,
shared by ``pipelines/book_rag/steps/chunk_embed.py`` and the visual
sidecar's ``build_visual_chunks``, the ``corpus health``/``corpus
boilerplate`` matcher hook, ``scripts/scan_boilerplate.py``) calls.

``BOILERPLATE_CHUNKS`` ships here with only the known seed entry. The full
reviewed set is built from a production corpus scan
(``scripts/scan_boilerplate.py``) run while ingestion is frozen.
A newly-reviewed row is added here with a matching parametrized case.

Matching rule (exact match, not prefix match): a chunk that merely *starts*
with boilerplate text but goes on to carry real content -- e.g. "This page
intentionally left blank. Chapter 3 begins with load transfer under
braking." -- must be KEPT. ``is_boilerplate`` checks the chunk's entire
normalized content against the allow-list, so a boilerplate phrase followed
by real prose normalizes to a longer string that is not a member of
``BOILERPLATE_CHUNKS`` and is therefore never dropped.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from vo_rag.core.chunking.section_packer import APPENDED_TEXT_ELEMENT_TYPES, SectionPacker
from vo_rag.core.parsing.contract import ElementType, ParsedDocument, ParsedElement
from vo_rag.core.parsing.text_normalization import normalize_extracted_text

if TYPE_CHECKING:
    from vo_rag.core.chunking.composite_emitter import CompositeChunk
    from vo_rag.core.chunking.hierarchical import HierarchicalChunker
    from vo_rag.core.chunking.models import RetrievalChunk

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalize_chunk_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace.

    Used both to build ``BOILERPLATE_CHUNKS`` (its entries are already in
    this normalized form) and to normalize a candidate chunk's content
    before the membership check in ``is_boilerplate``.
    """
    return _WS.sub(" ", _PUNCT.sub("", text.lower())).strip()


#: Reviewed, normalized boilerplate chunk texts. Only the known seed entry
#: ships by default; entries reviewed from a production scan are appended here
#: with a parametrized case added to
#: `test_known_boilerplate_detected`.
BOILERPLATE_CHUNKS: frozenset[str] = frozenset(
    {
        "this page intentionally left blank",
    }
)


def is_boilerplate(text: str) -> bool:
    """True when *text*, normalized, is an exact match for a reviewed
    boilerplate chunk. A boilerplate PREFIX followed by real content is
    never dropped -- see the module docstring's Matching rule note."""
    return normalize_chunk_text(text) in BOILERPLATE_CHUNKS


#: Element types the element-level filter may drop: exactly the body text
#: the chunker appends (``APPENDED_TEXT_ELEMENT_TYPES``, so the two can't
#: drift) plus list items. Never a TITLE (the reviewed set can grow to include short
#: common texts like "Notes" and "Index"; dropping those headings would bring
#: back stale section titles), a TABLE, or a HEADER/FOOTER
#: (the chunker's own running-line handling owns those).
FILTERABLE_ELEMENT_TYPES: frozenset[ElementType] = frozenset(
    {*APPENDED_TEXT_ELEMENT_TYPES, ElementType.LIST_ITEM}
)


def _is_droppable(element: ParsedElement) -> bool:
    """A body element whose whole text is boilerplate -- unless the chunker
    would take it as a standalone heading (a NARRATIVE_TEXT "Notes" or
    "Index" starts its section, exactly like a TITLE), checked with the
    chunker's own predicate on the text as the chunker normalizes it."""
    if element.element_type not in FILTERABLE_ELEMENT_TYPES:
        return False
    if not is_boilerplate(element.text or ""):
        return False
    return not (
        element.element_type is ElementType.NARRATIVE_TEXT
        and SectionPacker.is_standalone_heading(normalize_extracted_text(element.text or ""))
    )


def filter_boilerplate_elements(
    parsed_doc: ParsedDocument, composites: list[CompositeChunk] | None
) -> tuple[ParsedDocument, list[CompositeChunk]]:
    """Drop boilerplate BODY elements before chunking and remap composites.

    Element level (VOR-1427): the packer merges adjacent
    short elements into one chunk, so a real paragraph followed by a short
    "This page intentionally left blank" on the next page can become ONE
    chunk that no whole-chunk check would match. Only ``elements`` changes;
    every other field passes through.

    Composites (figure/table composites from the visual sidecar) were
    computed against the ORIGINAL element list, and the chunker keys them by
    position in the list it is given, so each ``element_index`` is remapped
    to the count of surviving elements before it. That attaches a composite
    whose own element was dropped to the next surviving element (or the end
    of the document) rather than losing its image/table. The remapped index
    is only the chunker's placement key: the ORIGINAL index -- the one that
    points into ``parsed.json`` -- is kept in the composite's ``metadata``,
    which the emitted chunk (and the stored payload) carries.
    """
    elements = parsed_doc.elements
    dropped = [_is_droppable(element) for element in elements]
    kept_before = [0] * (len(elements) + 1)
    running = 0
    for index, is_dropped in enumerate(dropped):
        kept_before[index] = running
        if not is_dropped:
            running += 1
    kept_before[len(elements)] = running

    remapped = [
        composite.model_copy(
            update={
                "element_index": kept_before[min(max(composite.element_index, 0), len(elements))],
                "metadata": {
                    **dict(composite.metadata or {}),
                    "element_index": composite.element_index,
                },
            }
        )
        for composite in (composites or [])
    ]
    if any(dropped):
        parsed_doc = parsed_doc.model_copy(
            update={
                "elements": [
                    element
                    for element, is_dropped in zip(elements, dropped, strict=True)
                    if not is_dropped
                ]
            }
        )
    return parsed_doc, remapped


def chunk_without_boilerplate(
    chunker: HierarchicalChunker,
    parsed_doc: ParsedDocument,
    composites: list[CompositeChunk] | None = None,
) -> tuple[list[RetrievalChunk], ParsedDocument]:
    """``chunker.chunk`` under the boilerplate policy: body elements filtered
    and composites remapped first (``filter_boilerplate_elements``), then any
    chunk that is still ENTIRELY boilerplate dropped (defense in depth, e.g.
    a composite). Returns the chunks and the filtered document (the caller's
    fallback path re-chunks from it)."""
    filtered_doc, remapped = filter_boilerplate_elements(parsed_doc, composites)
    chunks = chunker.chunk(filtered_doc, composites=remapped)
    return [c for c in chunks if not is_boilerplate(c.content or "")], filtered_doc


__all__ = [
    "BOILERPLATE_CHUNKS",
    "FILTERABLE_ELEMENT_TYPES",
    "chunk_without_boilerplate",
    "filter_boilerplate_elements",
    "is_boilerplate",
    "normalize_chunk_text",
]

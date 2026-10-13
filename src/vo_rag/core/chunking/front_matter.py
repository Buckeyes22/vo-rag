"""Front/back-matter heading classifier for book_rag ingestion.

The chunker tags chunks whose section heading names a front/back-matter division
(List of Symbols, Table of Contents, Index, Bibliography, …) with
``chunk_type="front_matter"`` so the summary builder can exclude them — otherwise
they become low-value "section" nodes (e.g. "Bibliography", "R is the corner
radius") that contaminate the chapter/book rollup.
"""

from __future__ import annotations

import re
from typing import Final, Literal

FrontMatterKind = Literal[
    "list_of_symbols",
    "toc",
    "index",
    "bibliography",
    "glossary",
    "about_author",
    "acknowledgements",
    "copyright",
]

_PHRASE_TO_SLUG: Final[dict[str, FrontMatterKind]] = {
    "list of symbols": "list_of_symbols",
    "nomenclature": "list_of_symbols",
    "notation": "list_of_symbols",
    "symbols": "list_of_symbols",
    "table of contents": "toc",
    "contents": "toc",
    "index": "index",
    "subject index": "index",
    "author index": "index",
    "bibliography": "bibliography",
    "references": "bibliography",
    "works cited": "bibliography",
    "glossary": "glossary",
    "about the author": "about_author",
    "about the authors": "about_author",
    "acknowledgements": "acknowledgements",
    "acknowledgments": "acknowledgements",
    "copyright": "copyright",
    "copyright page": "copyright",
}


def front_matter_kind(heading: str) -> FrontMatterKind | None:
    """Return a kind slug if ``heading`` names a front/back-matter section, else None.

    Normalizes the heading (lowercase, strip leading numbering/bullets, strip a
    trailing colon, collapse whitespace) before an exact phrase lookup.
    """
    if not heading:
        return None
    normalized = re.sub(r"^[\s\d.\-–•()]+", "", heading.lower())
    normalized = normalized.rstrip(":").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return _PHRASE_TO_SLUG.get(normalized)

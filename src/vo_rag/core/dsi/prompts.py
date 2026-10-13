"""DSI extraction prompts — single source of truth for ``DSI_PROMPT_VERSION``.

``schemas.py`` MUST NOT redeclare ``DSI_PROMPT_VERSION``: a second copy can
drift from this one and stamp records with the wrong prompt version.

``DSI_EXTRACTION_PROMPT_HASH`` is the canonical version tag for the extraction
contract: a sha256 over ``system + user`` template. Stored on every
``FacetedBookRecord`` (``prompt_hash``) and folded into the Qdrant
``deterministic_point_id`` so a prompt edit produces a distinct record.
"""

from __future__ import annotations

import hashlib

DSI_PROMPT_VERSION = "v1.3"

DSI_EXTRACTION_SYSTEM_PROMPT = """You are a domain expert in technical book analysis. \
Given a book's metadata and representative chunks, extract a structured faceted routing \
record. Every \
provenance-linked entry must cite the chunk_id of a provided chunk that supports \
it, copied verbatim. Do NOT invent chunk_ids and do NOT write section paths — \
section paths are derived automatically from the chunk you cite. Do not \
hallucinate routing metadata without traceable source backing."""

DSI_EXTRACTION_USER_TEMPLATE = """## Book Metadata
Title: {title}
Author: {author}
Categories: {category_ids}

## Representative Chunks
{chunk_contents}

## Extraction Task
Extract a faceted routing record as a JSON object with these fields:
1. short_abstract (~200 tokens): concise scope/purpose overview.
2. technical_domains (list of strings): fields/disciplines covered.
3. key_concepts (provenance-linked — see format below): important ideas, principles,
   theories, or topics taught — ABSTRACT knowledge (not physical objects).
4. key_components (list of strings): PHYSICAL parts, hardware, devices, assemblies,
   or systems described in the book (e.g. crankshaft, ECU, gearbox, brake caliper,
   oxygen sensor). MUST be tangible hardware — NOT concepts, methods, formulas, or
   processes. If the book is purely theoretical with no physical hardware, return an
   empty list.
5. key_methods (provenance-linked): procedures, techniques, or processes (often
   "-ing" or "how to ...").
6. key_equations_or_models (provenance-linked): formulas, equations, or
   mathematical/physical models.
7. figure_heavy_topics (provenance-linked): topics heavily supported by
   figures/diagrams/charts.
8. synonyms_and_aliases (provenance-linked): alternative names/terms for concepts
   in the book.
9. prerequisite_topics (list of strings): background knowledge a reader needs first.
10. chapter_outline (string): brief narrative of the book's progression.

For each field, extract ALL distinct, supportable items present in the chunks —
aim for completeness, not just a few.
Do not pad with duplicates or vague entries.
For provenance-linked fields only, every item must cite a supporting chunk; omit
provenance-linked items you cannot tie to a chunk.

Plain-string fields: `key_components`, `technical_domains`, and \
`prerequisite_topics` are PLAIN STRINGS
(a JSON array of strings) — NOT objects; do NOT attach chunk_id to them.

Each PROVENANCE-LINKED field is a list of objects, each exactly:
  {{"value": "<short noun phrase naming the concept/method/etc.>",
    "chunk_id": "<the exact chunk_id of a Representative Chunk above that supports it>"}}
Rules for provenance-linked entries:
- "value" must be a non-empty, specific noun phrase (never blank).
- "chunk_id" must be copied verbatim from the [chunk_id=...] of a chunk above.
  Use ONLY chunk_ids that appear above; never invent one.
- Do NOT include a section path — it is derived from the chunk_id you cite.
"""


def _sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


DSI_EXTRACTION_PROMPT_HASH: str = _sha256_of(
    DSI_EXTRACTION_SYSTEM_PROMPT + DSI_EXTRACTION_USER_TEMPLATE
)

__all__ = [
    "DSI_PROMPT_VERSION",
    "DSI_EXTRACTION_SYSTEM_PROMPT",
    "DSI_EXTRACTION_USER_TEMPLATE",
    "DSI_EXTRACTION_PROMPT_HASH",
]

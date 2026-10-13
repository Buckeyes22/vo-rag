"""Text normalization for parser output.

Normalizes text emitted from PDF/OCR parsers before chunking:

- NFKC normalization (standard Unicode ligatures like "ﬁ" -> "fi")
- Private-use glyph mapping (some native PDFs use PUA glyphs instead of text)
- Hard end-of-line hyphen dehyphenation ("co-\\noperate" -> "cooperate")
- F-099 bidi-override codepoint stripping (Trojan-source defense)
"""

from __future__ import annotations

import re
import unicodedata

_PRIVATE_USE_REPLACEMENTS = {
    "\ue039": "ft",
    "\ue03a": "ck",
    "\ue03b": "ch",
    "\ue03c": "tt",
    "\ue048": "Qu",
    "\ue049": "Th",
}

_PRIVATE_USE_RE = re.compile(r"[\ue000-\uf8ff]")
_HARD_LINE_HYPHEN_RE = re.compile(r"(\w)-\n\s*(\w)")

# F-099: explicit bidi-override and isolate codepoints. These survive
# NFKC normalization (different category) and are the standard tool for
# Unicode "bidi" attacks where the visible text on screen reads one way but
# the byte stream another. Strip them out — they have no legitimate use in
# extracted PDF/OCR text.
#  - U+202A LRE  Left-to-Right Embedding
#  - U+202B RLE  Right-to-Left Embedding
#  - U+202C PDF  Pop Directional Formatting
#  - U+202D LRO  Left-to-Right Override
#  - U+202E RLO  Right-to-Left Override (the classic CVE-2021-42574 vector)
#  - U+2066 LRI  Left-to-Right Isolate
#  - U+2067 RLI  Right-to-Left Isolate
#  - U+2068 FSI  First Strong Isolate
#  - U+2069 PDI  Pop Directional Isolate
_BIDI_OVERRIDE_RE = re.compile("[\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069]")


def _dehyphenate_line_wraps(text: str) -> str:
    """Rejoin hard end-of-line OCR/PDF hyphenation artifacts."""

    return _HARD_LINE_HYPHEN_RE.sub(r"\1\2", text)


def normalize_extracted_text(text: str) -> str:
    """Normalize text emitted from PDF/OCR parsers before chunking.

    Some native PDFs use private font encodings that PyMuPDF exposes as
    private-use glyphs instead of normal text. Keep the fix conservative:
    map observed ligature glyphs and let any unknown private-use characters
    disappear rather than pollute retrieval text.

    Also strips Unicode bidi-override codepoints (F-099): these have no
    legitimate use in extracted text and are the standard "homograph" /
    "Trojan source" attack vector when an attacker controls a PDF's content.
    """

    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    for raw, replacement in _PRIVATE_USE_REPLACEMENTS.items():
        normalized = normalized.replace(raw, replacement)
    normalized = _PRIVATE_USE_RE.sub("", normalized)
    normalized = _dehyphenate_line_wraps(normalized)
    return _BIDI_OVERRIDE_RE.sub("", normalized)

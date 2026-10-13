"""Book-title normalization and "already replaced" matching (VOR-1427).

``normalize_title`` folds the many spellings one book arrives under -- catalogue
filenames, truncated uploads, ``.reencoded`` copies -- to one comparable key.
The OCR farm uses it to skip scanned books that already have a completed text
version; the corpus duplicate report reuses it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_BOOK_EXTENSIONS = (
    "pdf|epub|mobi|azw|azw3|djvu|djv|fb2|txt|doc|docx|rtf|lit|chm|cbz|cbr|htm|html|md"
)
_EXTENSION_RE = re.compile(rf"\.(?:{_BOOK_EXTENSIONS})\s*$", re.IGNORECASE)
# Trailing ".reencoded" marker left by re-encoding tools.
_REENCODED_RE = re.compile(r"\.reencoded\s*$", re.IGNORECASE)
_TRAILING_UNDERSCORES_RE = re.compile(r"_+\s*$")
# Trailing catalogue id: Title_12345678 / Title___12345678.
_ID_SUFFIX_RE = re.compile(r"_{1,3}\d{5,}\s*$")
# Bracketed / parenthesized digit runs of 5 or more (catalogue ids).
_BRACKETED_ID_RE = re.compile(r"[(\[]\s*\d{5,}\s*[)\]]")
_YEAR_DIGITS = r"(?:1[4-9]|20)\d{2}"
_YEAR_RE = re.compile(rf"\(\s*{_YEAR_DIGITS}\s*\)")
_YEAR_CAPTURE_RE = re.compile(rf"\(\s*({_YEAR_DIGITS})\s*\)")
_APOSTROPHE_RE = re.compile(r"['’`]")
_PUNCT_RE = re.compile(r"[^\w\s]|_")
_SPACE_RE = re.compile(r"\s+")

#: A truncated failed title must keep at least this many normalized
#: characters before a completed title may extend it.
MIN_PREFIX_CHARS = 12


def _strip_suffixes(title: str) -> str:
    """Peel trailing extension / ``.reencoded`` / ``_`` / catalogue id layers.

    Loops because the layers stack in any order (``Title_.reencoded.pdf``).
    """
    text = title.strip()
    while True:
        before = text
        text = _EXTENSION_RE.sub("", text)
        text = _REENCODED_RE.sub("", text)
        text = _ID_SUFFIX_RE.sub("", text)
        text = _TRAILING_UNDERSCORES_RE.sub("", text)
        text = text.strip()
        if text == before:
            return text


def normalize_title(title: str) -> str:
    """Fold a raw book title / filename to a comparable key.

    Strips a trailing ``.reencoded``, trailing ``_``, catalogue id suffixes
    (``_{1,3}\\d{5,}`` at the end; bracketed or parenthesized digit runs of 5+),
    ``(YYYY)`` years and a file extension; then replaces ``_`` with a space,
    lowercases, strips punctuation and collapses whitespace.
    """
    text = _strip_suffixes(title or "")
    text = _BRACKETED_ID_RE.sub(" ", text)
    text = _YEAR_RE.sub(" ", text)
    text = _strip_suffixes(text)
    text = text.replace("_", " ").lower()
    text = _APOSTROPHE_RE.sub("", text)
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def title_year(title: str) -> str | None:
    """The parenthesized publication year, if any.

    This is the same ``(YYYY)`` ``normalize_title`` strips away -- deliberately
    *not* folded back into ``normalize_title`` itself, since ``plan_backlog``'s
    title matching (and anything else already keying off ``normalize_title``)
    relies on years staying stripped there. Callers that need to tell
    "Applied Strength Methods (2013)" apart from "Applied Strength Methods (2019)" -- the
    corpus duplicate report -- read the year separately with this
    function instead.

    If a title carries two parenthesized years, only the first (leftmost)
    one is returned -- ``re.search`` stops at the first match, and this
    function has never looked past it (a test pins that as
    intentional rather than an oversight).
    """
    match = _YEAR_CAPTURE_RE.search(title or "")
    return match.group(1) if match else None


def is_truncated_title(title: str) -> bool:
    """True when a raw title looks cut off: it ends with ``_`` or carries a
    catalogue id suffix (``_{1,3}\\d{5,}``), ignoring a trailing file
    extension / ``.reencoded``."""
    text = (title or "").strip()
    while True:
        before = text
        text = _EXTENSION_RE.sub("", text)
        text = _REENCODED_RE.sub("", text).strip()
        if text == before:
            break
    return bool(text) and (text.endswith("_") or _ID_SUFFIX_RE.search(text) is not None)


def is_already_replaced(failed_title: str, completed_normalized: Iterable[str]) -> bool:
    """Does a completed book already cover this failed upload?

    Exact normalized equality, OR -- only when the failed raw title is
    truncated -- a completed normalized title that *starts with* the failed
    normalized title of at least ``MIN_PREFIX_CHARS`` characters. The reverse
    direction (the failed title extending a shorter completed one) is never
    used: it matched "practical engines applied dynam" to
    "practical engine".
    """
    failed = normalize_title(failed_title)
    if not failed:
        return False
    completed = set(completed_normalized)
    if failed in completed:
        return True
    if not is_truncated_title(failed_title) or len(failed) < MIN_PREFIX_CHARS:
        return False
    return any(title.startswith(failed) for title in completed if title)

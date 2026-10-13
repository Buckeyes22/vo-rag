"""Content fingerprints: one book under differing file bytes (VOR-1428).

md5/sha256 only catch byte-identical files. The same edition re-saved,
re-stamped or re-uploaded under another catalogue record ("Workshop
Reference 29th Ed" vs "Workshop reference a handbook ...") has new
bytes but the same text. A fingerprint hashes normalized text instead:

- PDF: page count plus the text of pages sampled at fixed fractions, so a
  fingerprint costs a handful of page extractions, not a full parse.
- EPUB: the whole spine's text (EPUBs are small and have no stable pages).

Files whose sampled text is too thin (image-only scans, DRM, broken files)
get no fingerprint rather than a weak one that would collide.
"""

from __future__ import annotations

import hashlib
import posixpath
import re
import warnings
import zipfile
from pathlib import Path
from typing import Any

#: Fractions of the page range sampled for a PDF fingerprint.
PDF_SAMPLE_FRACTIONS = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)
#: Below this many normalized characters the text is too thin to identify a book.
MIN_TEXT_CHARS = 2000

_NON_WORD = re.compile(r"[^0-9a-z]+")


def _normalize(text: str) -> str:
    return _NON_WORD.sub(" ", text.casefold()).strip()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _pdf_fingerprint(path: Path) -> str | None:
    import fitz  # type: ignore[import-untyped]  # pymupdf; core dependency

    with fitz.open(path) as doc:
        pages = doc.page_count
        if pages == 0:
            return None
        indexes = sorted({min(pages - 1, int(pages * f)) for f in PDF_SAMPLE_FRACTIONS})
        text = " ".join(_normalize(doc[i].get_text()) for i in indexes)
    if len(text) < MIN_TEXT_CHARS:
        return None
    return f"pdf:{pages}:{_digest(text)}"


def _xml(data: bytes) -> Any:
    """Parse container/OPF XML with the stdlib-backed parser (lxml is not a dependency)."""
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        return BeautifulSoup(data, "html.parser")


def _epub_spine(archive: zipfile.ZipFile) -> list[str]:
    container = _xml(archive.read("META-INF/container.xml"))
    rootfile = container.find("rootfile")
    opf_path = str(rootfile["full-path"]) if rootfile else ""
    opf = _xml(archive.read(opf_path))
    base = posixpath.dirname(opf_path)
    manifest = {
        str(item["id"]): posixpath.normpath(posixpath.join(base, str(item["href"])))
        for item in opf.find_all("item")
        if item.get("id") and item.get("href")
    }
    return [
        manifest[str(ref["idref"])]
        for ref in opf.find_all("itemref")
        if str(ref.get("idref")) in manifest
    ]


def _epub_fingerprint(path: Path) -> str | None:
    from bs4 import BeautifulSoup

    from vo_rag.core.parsing.parsers.epub import check_zip_bomb

    with zipfile.ZipFile(path) as archive:
        # Before any spine member is decompressed (VOR-1434);
        # a bomb raises ZipBombError, which content_fingerprint turns into
        # "unidentifiable" and preflight's container check then rejects.
        check_zip_bomb(archive, source=str(path))
        names = set(archive.namelist())
        parts: list[str] = []
        for name in _epub_spine(archive):
            if name in names:
                html = BeautifulSoup(archive.read(name), "html.parser")
                parts.append(_normalize(html.get_text(" ")))
    text = " ".join(p for p in parts if p)
    if len(text) < MIN_TEXT_CHARS:
        return None
    return f"epub:{_digest(text)}"


def content_fingerprint(path: Path) -> str | None:
    """Fingerprint of a PDF/EPUB's text, or None when it cannot be identified."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return _pdf_fingerprint(path)
        if suffix == ".epub":
            return _epub_fingerprint(path)
    except Exception:  # a broken file is unidentifiable, never fatal to a run
        return None
    return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def duplicate_groups(fingerprints: dict[str, Any]) -> dict[str, list[str]]:
    """fingerprint -> [sha256, ...] for fingerprints held by more than one file."""
    groups: dict[str, list[str]] = {}
    for sha, fingerprint in sorted(fingerprints.items()):
        if fingerprint:
            groups.setdefault(str(fingerprint), []).append(sha)
    return {fp: shas for fp, shas in sorted(groups.items()) if len(shas) > 1}

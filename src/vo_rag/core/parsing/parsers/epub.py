"""EPUB / MOBI parser (ebooklib + pandoc + Calibre for MOBI).

The heavy ``ebooklib`` / ``pypandoc`` imports live inside the
``_parse_epub_sync`` body so the module imports without those deps
installed (the tests only exercise the pure helpers).
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import io
import mimetypes
import os
import posixpath
import re
import shutil
import subprocess
import tempfile
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import structlog

from vo_rag.core.parsing.contract import (
    ElementType,
    ExtractedImage,
    PageQA,
    ParsedDocument,
    ParsedElement,
)
from vo_rag.core.storage.figures import MIME_BY_EXT, FigureStore, safe_ext

MOBI_CONTENT_TYPE = "application/x-mobipocket-ebook"
MOBI_FORMAT = "mobi"
EPUB_FORMAT = "epub"

mimetypes.add_type(MOBI_CONTENT_TYPE, ".mobi")

log = structlog.get_logger(__name__)

# Some EPUBs (VOR-1427) declare their chapter items `media-type="text/html"` in
# `content.opf` instead of the EPUB-required `application/xhtml+xml`.
# ebooklib doesn't treat those spine items as documents and pandoc 3.1.3
# skips them entirely, so the whole book collapses to just the cover.
#
# The pattern is quote-agnostic (single OR double quote, legal XML
# either way) and case-insensitive (`Text/HTML`, `TEXT/HTML`, ...), and
# tolerant of whitespace inside the quotes -- a naive
# `media-type\s*=\s*"text/html"` silently left single-quoted or mixed-case
# manifests unnormalized, collapsing the book to just the cover with no
# error. Group 2 captures the opening quote character so the replacement
# can echo it back unchanged.
_TEXT_HTML = re.compile(rb'(media-type\s*=\s*)([\'"])\s*text/html\s*\2', re.IGNORECASE)
_TEXT_HTML_REPLACEMENT = rb"\1\2application/xhtml+xml\2"

#: The single source of truth for "does this OPF manifest have a `text/html`
#: chapter item" (``scripts/find_text_html_epubs.py`` imports this function
#: rather than re-deriving it with a narrower double-quote-only regex).


def count_text_html_manifest_items(epub_bytes: bytes) -> int:
    """Count of `text/html`-declared (any quote, any case) chapter items
    across every OPF manifest inside the EPUB ``epub_bytes`` -- ``0`` for a
    normal, unaffected EPUB. Shared by ``_normalize_epub`` (detection) and
    ``scripts/find_text_html_epubs.py`` (the read-only corpus sweep) so the
    two can never diverge on what counts as a match.
    """
    with zipfile.ZipFile(io.BytesIO(epub_bytes)) as zf:
        opf_names = [n for n in zf.namelist() if n.endswith(".opf")]
        return sum(len(_TEXT_HTML.findall(zf.read(name))) for name in opf_names)


# Some real books list a manifest item whose file is not in the archive
# (VOR-1431: a nav document, `OEBPS/toc01.xhtml`, on 01M3DMVV0ZH8QWVAMKW9N54230).
# ebooklib's reader loads every manifest item and raises KeyError on the first
# missing one, failing the whole book; the normalized copy drops such items and
# their spine references so the rest of the book parses.
_MANIFEST_ITEM = re.compile(rb"<item\b[^>]*?/?>", re.IGNORECASE | re.DOTALL)
_ATTR_HREF = re.compile(rb"\bhref\s*=\s*(['\"])(.*?)\1", re.IGNORECASE | re.DOTALL)
_ATTR_ID = re.compile(rb"\bid\s*=\s*(['\"])(.*?)\1", re.IGNORECASE | re.DOTALL)


def _manifest_target(opf_name: str, href: bytes) -> str:
    path = unquote(href.decode("utf-8", "replace")).split("#", 1)[0]
    return posixpath.normpath(posixpath.join(posixpath.dirname(opf_name), path))


def _missing_manifest_ids(opf_name: str, opf_bytes: bytes, members: set[str]) -> list[bytes]:
    """ids of manifest items in ``opf_bytes`` whose href is not a zip member."""
    missing: list[bytes] = []
    for item in _MANIFEST_ITEM.finditer(opf_bytes):
        href = _ATTR_HREF.search(item.group(0))
        if href and _manifest_target(opf_name, href.group(2)) not in members:
            item_id = _ATTR_ID.search(item.group(0))
            missing.append(item_id.group(2) if item_id else b"")
    return missing


def _drop_missing_manifest_items(opf_name: str, opf_bytes: bytes, members: set[str]) -> bytes:
    missing = [i for i in _missing_manifest_ids(opf_name, opf_bytes, members) if i]

    def keep(item: re.Match[bytes]) -> bytes:
        href = _ATTR_HREF.search(item.group(0))
        if href and _manifest_target(opf_name, href.group(2)) not in members:
            return b""
        return item.group(0)

    opf_bytes = _MANIFEST_ITEM.sub(keep, opf_bytes)
    for item_id in missing:
        opf_bytes = re.sub(
            rb"<itemref\b[^>]*\bidref\s*=\s*(['\"])" + re.escape(item_id) + rb"\1[^>]*?/?>",
            b"",
            opf_bytes,
            flags=re.IGNORECASE,
        )
    return opf_bytes


# Some real books list a manifest item whose href resolves outside the OPF's
# own directory (VOR-1434: OPF at
# `OEBPS/content.opf`, `href="../jacket.xhtml"`). pandoc refuses the path
# ("No entry on path: OEBPS/../jacket.xhtml", exit 64). The normalized copy
# places such a member inside the OPF directory under a collision-free name,
# points the manifest href at it, and rewrites the relative links inside the
# moved document against its new location. The original member stays where
# it was, so links from other documents still resolve.
def _escapes_opf_dir(opf_name: str, target: str) -> bool:
    opf_dir = posixpath.dirname(opf_name)
    if not opf_dir:
        return target == ".." or target.startswith("../")
    return not target.startswith(opf_dir + "/")


def _escaping_items(opf_name: str, opf_bytes: bytes) -> list[tuple[bytes, str]]:
    """``(raw href, resolved target)`` of every manifest item in ``opf_bytes``
    whose href resolves outside ``opf_name``'s own directory."""
    escaping: list[tuple[bytes, str]] = []
    for item in _MANIFEST_ITEM.finditer(opf_bytes):
        href = _ATTR_HREF.search(item.group(0))
        if not href:
            continue
        target = _manifest_target(opf_name, href.group(2))
        if _escapes_opf_dir(opf_name, target):
            escaping.append((href.group(2), target))
    return escaping


def _escaping_hrefs(opf_name: str, opf_bytes: bytes) -> list[str]:
    """Manifest item hrefs (decoded) in ``opf_bytes`` that resolve outside
    ``opf_name``'s own directory, e.g. ``../jacket.xhtml`` in
    ``OEBPS/content.opf``. Shared with preflight, which reports them."""
    return [
        unquote(raw.decode("utf-8", "replace")) for raw, _ in _escaping_items(opf_name, opf_bytes)
    ]


def _collision_free_name(directory: str, basename: str, taken: set[str]) -> str:
    stem, ext = posixpath.splitext(basename)
    candidate = posixpath.join(directory, basename)
    n = 0
    while candidate in taken:
        n += 1
        candidate = posixpath.join(directory, f"{stem}-{n}{ext}")
    return candidate


_LINK_ATTR = re.compile(rb"(\b(?:href|src)\s*=\s*)(['\"])(.*?)\2", re.IGNORECASE | re.DOTALL)
_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_MARKUP_EXTENSIONS = (".xhtml", ".html", ".htm", ".xml", ".svg", ".ncx")


_LINK_PATH_END = re.compile(r"[?#]")


def _rebase_links(data: bytes, old_name: str, new_name: str, redirects: dict[str, str]) -> bytes:
    """Rewrite relative ``href``/``src`` values in the markup ``data``, read
    from member ``old_name`` and written as ``new_name``.

    A link whose target is a key of ``redirects`` (original member -> the
    copy moved into the OPF dir) is pointed at the moved copy. Any other
    link in a document that moved (``old_name != new_name``) is rebased so it
    still reaches the same target. Only the path part changes: the original
    percent-encoding of its segments and any ``?query``/``#fragment`` are
    kept verbatim. Fragment-only, absolute, and scheme (``https:``,
    ``data:``, ...) values are left alone.
    """
    old_dir, new_dir = posixpath.dirname(old_name), posixpath.dirname(new_name)

    def rebase(match: re.Match[bytes]) -> bytes:
        value = match.group(3).decode("utf-8", "replace")
        if not value or value.startswith(("#", "/")) or _URL_SCHEME.match(value):
            return match.group(0)
        split = _LINK_PATH_END.search(value)
        raw_path, rest = (value[: split.start()], value[split.start() :]) if split else (value, "")
        if not raw_path:
            return match.group(0)
        target = posixpath.normpath(posixpath.join(old_dir, unquote(raw_path)))
        if target in redirects:
            new_path = quote(posixpath.relpath(redirects[target], new_dir or "."), safe="/")
        elif old_name != new_name:
            # Resolve on the still-encoded path so its segments keep their
            # original encoding; only the zip's own directory names are quoted.
            raw_target = posixpath.normpath(posixpath.join(quote(old_dir, safe="/"), raw_path))
            new_path = posixpath.relpath(raw_target, quote(new_dir, safe="/") or ".")
        else:
            return match.group(0)
        new_value = (new_path + rest).encode()
        return match.group(1) + match.group(2) + new_value + match.group(2)

    return _LINK_ATTR.sub(rebase, data)


def _relocate_escaping_items(
    opf_name: str, opf_bytes: bytes, taken: set[str]
) -> tuple[bytes, dict[str, str]]:
    """Point every escaping manifest href in ``opf_bytes`` at a new member
    inside the OPF directory. Returns the rewritten OPF and a
    ``{new member: original member}`` map for the caller to write; ``taken``
    (every member name already in use) is updated in place."""
    opf_dir = posixpath.dirname(opf_name)
    moves: dict[str, str] = {}
    new_href_for: dict[str, bytes] = {}
    for _raw, target in _escaping_items(opf_name, opf_bytes):
        if target not in taken or target in new_href_for:
            continue  # missing members are dropped elsewhere
        new_name = _collision_free_name(opf_dir, posixpath.basename(target), taken)
        taken.add(new_name)
        moves[new_name] = target
        new_href_for[target] = quote(posixpath.relpath(new_name, opf_dir or "."), safe="/").encode()

    def rewrite(item: re.Match[bytes]) -> bytes:
        text = item.group(0)
        href = _ATTR_HREF.search(text)
        if not href:
            return text
        # Keys are escaping targets only, so a hit is always an escaping href.
        new_href = new_href_for.get(_manifest_target(opf_name, href.group(2)))
        if new_href is None:
            return text
        return text[: href.start(2)] + new_href + text[href.end(2) :]

    return _MANIFEST_ITEM.sub(rewrite, opf_bytes), moves


# Some real books end in a ZIP64 end-of-central-directory record even though
# they are tiny (VOR-1432).
# Python's zipfile reads them, but pandoc's zip reader does not ("Did not find
# end of central directory signature"). The normalized copy is written by
# zipfile, which only emits ZIP64 records when the archive actually needs them.
_EOCD = b"PK\x05\x06"
_ZIP64_EOCD_LOCATOR = b"PK\x06\x07"
_ZIP64_LOCATOR_SIZE = 20
_EOCD_MAX_TAIL = 22 + 0xFFFF + _ZIP64_LOCATOR_SIZE  # EOCD + max comment + locator


def _has_zip64_eocd(path: Path) -> bool:
    """True when the EOCD record in ``path`` is preceded by a ZIP64 locator."""
    with path.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - _EOCD_MAX_TAIL))
        tail = f.read()
    eocd = tail.rfind(_EOCD)
    locator = eocd - _ZIP64_LOCATOR_SIZE
    return locator >= 0 and tail[locator : locator + 4] == _ZIP64_EOCD_LOCATOR


# Zip-bomb guard (VOR-1434). Every reader of an untrusted
# EPUB -- preflight's ``testzip()``, ``_epub_needs_normalization``,
# ``_normalize_epub`` (which holds relocated members in memory), ebooklib and
# pandoc -- decompresses members in full, so the archive's declared sizes are
# checked from the central directory before any member is read. zipfile's
# member reader never returns more than the declared ``file_size``, so the
# declared sizes bound what any later read can inflate to.
ZIP_BOMB_MAX_TOTAL_BYTES = 2 * 1024**3
ZIP_BOMB_RATIO_MIN_MEMBER_BYTES = 100 * 1024**2
ZIP_BOMB_MAX_RATIO = 200


class ZipBombError(ValueError):
    """The EPUB's declared sizes are a decompression bomb; the book fails."""

    reason = "zip_bomb"


def check_zip_bomb(archive: zipfile.ZipFile, *, source: str = "") -> None:
    """Raise ``ZipBombError`` when ``archive`` declares more than
    ``ZIP_BOMB_MAX_TOTAL_BYTES`` in total, or any member over
    ``ZIP_BOMB_RATIO_MIN_MEMBER_BYTES`` compresses at more than
    ``ZIP_BOMB_MAX_RATIO``:1. Reads only the central directory."""
    total = 0
    for info in archive.infolist():
        total += info.file_size
        if (
            info.file_size > ZIP_BOMB_RATIO_MIN_MEMBER_BYTES
            and info.file_size > ZIP_BOMB_MAX_RATIO * max(info.compress_size, 1)
        ):
            raise ZipBombError(
                f"zip_bomb: {source or 'epub'} member {info.filename!r} declares "
                f"{info.file_size} bytes from {info.compress_size} compressed "
                f"(over {ZIP_BOMB_MAX_RATIO}:1)"
            )
    if total > ZIP_BOMB_MAX_TOTAL_BYTES:
        raise ZipBombError(
            f"zip_bomb: {source or 'epub'} declares {total} bytes uncompressed "
            f"(cap {ZIP_BOMB_MAX_TOTAL_BYTES})"
        )


def _epub_needs_normalization(path: Path) -> bool:
    """True when ``path`` ends in a ZIP64 end-of-central-directory record, or
    any OPF manifest in it declares a `text/html` chapter item, an item
    whose file is missing from the archive, or an item whose href resolves
    outside the OPF's own directory. Checked directly against the zip
    on disk (no full in-memory read) so ``_parse_epub_sync`` can decide
    whether to create a ``TemporaryDirectory`` at all.
    """
    with zipfile.ZipFile(path) as zin:
        # Raises ZipBombError before any member is read.
        check_zip_bomb(zin, source=str(path))
        if _has_zip64_eocd(path):
            return True
        members = set(zin.namelist())
        opf_names = [n for n in members if n.endswith(".opf")]
        return any(
            _TEXT_HTML.search(data)
            or _missing_manifest_ids(n, data, members)
            or _escaping_items(n, data)
            for n in opf_names
            for data in (zin.read(n),)
        )


def _normalize_epub(path: Path, workdir: Path) -> Path:
    """Return a path to parse for ``epub.read_epub``/``pypandoc.convert_file``.

    ``path`` unchanged when it has no ZIP64 end record and its OPF manifest(s)
    have no `text/html` chapter items and no items missing from the archive.
    Otherwise writes a plain-zip copy under ``workdir`` (VOR-1432) with every
    `text/html` item's `media-type` rewritten to `application/xhtml+xml`
    (preserving the original quote character) and every missing item dropped
    along with its spine reference (VOR-1431). A member whose manifest href
    resolves outside the OPF directory is copied into that directory under a
    collision-free name, its href rewritten, and the relative links inside it
    rebased on the new location (VOR-1434).
    ``mimetype`` stays the first entry, stored uncompressed, as the EPUB spec
    requires; every other entry is copied byte-for-byte.
    """
    if not _epub_needs_normalization(path):
        return path
    with zipfile.ZipFile(path) as zin:
        check_zip_bomb(zin, source=str(path))
        members = set(zin.namelist())
        opf_names = sorted(n for n in members if n.endswith(".opf"))
        taken = set(members)
        opfs: dict[str, bytes] = {}
        moves: dict[str, str] = {}
        for name in opf_names:
            data = _TEXT_HTML.sub(_TEXT_HTML_REPLACEMENT, zin.read(name))
            data = _drop_missing_manifest_items(name, data, members)
            opfs[name], opf_moves = _relocate_escaping_items(name, data, taken)
            moves.update(opf_moves)
        # Read moved payloads before writing: ``zout.writestr(info, ...)``
        # below reuses (and re-offsets) ``zin``'s own ZipInfo objects.
        moved = {new_name: zin.read(old_name) for new_name, old_name in moves.items()}
        out = workdir / f"{path.stem}.normalized.epub"
        with zipfile.ZipFile(out, "w") as zout:
            zout.writestr(
                zipfile.ZipInfo("mimetype"),
                zin.read("mimetype"),
                compress_type=zipfile.ZIP_STORED,
            )
            # Links anywhere in the book (nav, NCX, other chapters) that
            # reach a moved member's original copy now reach the moved copy.
            redirects = {old_name: new_name for new_name, old_name in moves.items()}
            for info in zin.infolist():
                if info.filename == "mimetype":
                    continue
                data = opfs.get(info.filename) or zin.read(info.filename)
                if redirects and info.filename.lower().endswith(_MARKUP_EXTENSIONS):
                    data = _rebase_links(data, info.filename, info.filename, redirects)
                zout.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
            for new_name, data in moved.items():
                if new_name.lower().endswith(_MARKUP_EXTENSIONS):
                    data = _rebase_links(data, moves[new_name], new_name, redirects)
                zout.writestr(new_name, data, compress_type=zipfile.ZIP_DEFLATED)
    log.info("epub_manifest_normalized", source=str(path))
    return out


# is_image_only() thresholds (VOR-1427): a page-scan EPUB's spine is nearly
# all IMAGE elements with little real text (running headers, faint OCR noise,
# and often a few pages of real front matter or a TOC). Decided on the book as
# a whole, never on a single element: one front-matter paragraph must not
# rescue 400 scanned pages (VOR-1433: a 401-image EPUB carrying only
# about 8,000 characters of text). A real illustrated EPUB carries hundreds of
# characters of prose per image and fails the ratio.
_IMAGE_ONLY_MIN_ELEMENTS = 20
_IMAGE_ONLY_MIN_IMAGE_SHARE = 0.8
_IMAGE_ONLY_MAX_TEXT_CHARS_PER_IMAGE = 50


class _ImageHrefCollector(HTMLParser):
    """Collects <img src> and SVG-wrapped <image href|xlink:href> values in
    document order (VOR-1427). A real HTML/XML tokenizer -- not a regex -- so this is
    robust to attribute order and quoting, never confuses "data-src=" for
    "src=", and never matches "src=" sitting inside an unrelated attribute's
    *value* (e.g. ``alt="x src=fake"``); only an actual ``src``/``href``/
    ``xlink:href`` attribute is ever picked up.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._collect(tag, attrs)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._collect(tag, attrs)

    def _collect(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {name.lower(): value for name, value in attrs}
        href: str | None = None
        if tag.lower() == "img":
            href = attr_map.get("src")
        elif tag.lower() == "image":
            href = attr_map.get("xlink:href") or attr_map.get("href")
        if href:
            self.hrefs.append(href)


def _extract_image_hrefs(html_text: str) -> list[str]:
    parser = _ImageHrefCollector()
    parser.feed(html_text)
    parser.close()
    return parser.hrefs


class ImageOnlyEbook(Exception):
    """Raised by ``_parse_epub_sync`` when the spine is essentially all
    page-scan images (VOR-1427), not real text.

    Carries the Pillow-synthesized PDF path and the ``TemporaryDirectory``
    that owns it (the caller/router is responsible for calling
    ``tmpdir.cleanup()`` once it is done reading ``pdf_path``), plus the
    EPUB's own metadata (document_id/source_id/title/author) so the router
    can restore those onto the OCR'd result -- the synthesized PDF has none
    of that; it is a bare stack of page images.
    """

    def __init__(
        self,
        pdf_path: str,
        tmpdir: tempfile.TemporaryDirectory[str],
        *,
        metadata: dict[str, Any],
    ) -> None:
        super().__init__(f"image-only ebook synthesized to {pdf_path}")
        self.pdf_path = pdf_path
        self.tmpdir = tmpdir
        self.metadata = metadata


def is_image_only(elements: list[ParsedElement]) -> bool:
    """True when ``elements`` looks like a page-scan EPUB, not real text.

    At least 20 elements, at least 80% of them ``ElementType.IMAGE``, and the
    total text across all elements under 50 characters per image. TITLEs that
    a TOC anchor emitted (``metadata["toc_anchor"]``) are left out: a page-scan
    EPUB whose TOC lists every page gets one per page (VOR-1427). Headings the
    book itself carries, such as a caption ``<h2>`` over each plate, still count.
    """
    content = [
        e
        for e in elements
        if not (e.element_type == ElementType.TITLE and (e.metadata or {}).get("toc_anchor"))
    ]
    if len(content) < _IMAGE_ONLY_MIN_ELEMENTS:
        return False

    image_count = 0
    total_text_chars = 0
    for element in content:
        if element.element_type == ElementType.IMAGE:
            image_count += 1
        total_text_chars += len((element.text or "").strip())

    if image_count < _IMAGE_ONLY_MIN_IMAGE_SHARE * len(content):
        return False
    return total_text_chars < _IMAGE_ONLY_MAX_TEXT_CHARS_PER_IMAGE * image_count


_ANCHOR_FILTER = Path(__file__).with_name("epub_anchors.lua")

# pandoc keeps the id of these elements (as a Span, Div or Header identifier)
# and drops it on every other one -- a calibre ``<p id="page_17">`` TOC target
# vanished and its chapter stayed inside the previous one (VOR-1427,
# c8865c4a9058: TOC chapters 1-3 all pointed at ``<p id>`` page markers).
_PANDOC_KEEPS_ID = frozenset({"a", "span", "div", "section", "h1", "h2", "h3", "h4", "h5", "h6"})
_VOID_TAGS = frozenset({"img", "br", "hr", "input", "col", "wbr", "area", "embed", "source"})
# Elements that cannot hold a <span> child (and whose ids pandoc drops): a
# target id on one of them is left alone; the file-start anchor still covers it.
_NO_SPAN_CHILD = frozenset(
    {
        "html",
        "head",
        "body",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "colgroup",
        "ul",
        "ol",
        "dl",
        "select",
        "optgroup",
    }
)
# Linear scans only -- the markup is untrusted. A start tag, then its id
# attribute and name looked up inside that one tag.
_START_TAG_RE = re.compile(rb"<[A-Za-z][^<>]*>")
_TAG_NAME_RE = re.compile(rb"<([A-Za-z][\w:-]*)")
_TAG_ID_RE = re.compile(rb"\sid\s*=\s*([\"'])([^\"'<>]*)\1")
# Members larger than this are passed to pandoc as they are.
_ID_REWRITE_MAX_MEMBER_BYTES = 8 * 1024**2


def _toc_target_id_rewrites(path: Path, toc_tree: list[dict[str, Any]]) -> dict[str, bytes]:
    """Markup members of ``path`` rewritten so every TOC target id on an
    element pandoc drops ids from sits on an empty ``<span id>`` instead
    (inside the element; before it for void elements), where the anchor filter
    sees it. Empty when no member needs it."""
    hrefs = [unquote(str(node.get("href") or "")).partition("#") for node in toc_tree]
    if not any(fragment for _, _, fragment in hrefs):
        return {}

    def expose(wanted: set[bytes], match: re.Match[bytes]) -> bytes:
        tag = match.group(0)
        id_attr = _TAG_ID_RE.search(tag)
        name = _TAG_NAME_RE.match(tag)
        if id_attr is None or name is None or id_attr.group(2) not in wanted:
            return tag
        tag_name = name.group(1).decode("ascii", "replace").lower()
        if tag_name in _PANDOC_KEEPS_ID or tag_name in _NO_SPAN_CHILD:
            return tag
        span = b'<span id="' + id_attr.group(2) + b'"></span>'
        start_tag = tag[: id_attr.start()] + tag[id_attr.end() :]
        if tag.endswith(b"/>") or tag_name in _VOID_TAGS:
            return span + start_tag
        return start_tag + span

    rewritten: dict[str, bytes] = {}
    with zipfile.ZipFile(path) as zin:
        check_zip_bomb(zin, source=str(path))
        # TOC hrefs are relative to the OPF directory; key targets by the
        # member's full archive path so same-named files in two folders differ.
        opf_dirs = {posixpath.dirname(n) for n in zin.namelist() if n.endswith(".opf")} or {""}
        targets: dict[str, set[bytes]] = {}
        for file_part, _, fragment in hrefs:
            if not fragment:
                continue
            for opf_dir in opf_dirs:
                member = posixpath.normpath(posixpath.join(opf_dir, file_part))
                targets.setdefault(member, set()).add(fragment.encode("utf-8"))
        for info in zin.infolist():
            wanted = targets.get(info.filename)
            if (
                not wanted
                or not info.filename.lower().endswith(_MARKUP_EXTENSIONS)
                or info.file_size > _ID_REWRITE_MAX_MEMBER_BYTES
            ):
                continue
            data = zin.read(info.filename)
            new = _START_TAG_RE.sub(functools.partial(expose, wanted), data)
            if new != data:
                rewritten[info.filename] = new
    return rewritten


def _write_with_rewrites(path: Path, rewrites: dict[str, bytes], workdir: Path) -> Path:
    """Copy of ``path`` under ``workdir`` with ``rewrites`` replacing members."""
    out = workdir / f"{path.stem}.anchors.epub"
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(out, "w") as zout:
        for info in zin.infolist():
            data = rewrites.get(info.filename) or zin.read(info.filename)
            compress = zipfile.ZIP_STORED if info.filename == "mimetype" else zipfile.ZIP_DEFLATED
            zout.writestr(info, data, compress_type=compress)
    return out


def _read_epub_and_convert(
    parse_path: str, source_path: str, *, source_format: str
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], str]:
    """``ebooklib.read_epub`` + metadata/TOC + ``pypandoc.convert_file`` on
    ``parse_path`` (either the original file or a normalized copy), with
    identity metadata computed from ``source_path`` (always the original).
    Factored out so ``_parse_epub_sync`` can call it once for the common,
    no-rewrite-needed case and once for the normalized-copy case without
    duplicating the read/convert steps themselves.
    """
    import pypandoc  # type: ignore[import-not-found]
    from ebooklib import epub  # type: ignore[import-not-found]

    book = epub.read_epub(parse_path)
    metadata = _metadata_from_book(book, source_path, source_format=source_format)
    # A one-entry navigation TOC ("Start -> titlepage.xhtml") falls back to the
    # book's own linked contents page (VOR-1427).
    toc_tree = _with_contents_page_fallback(book, _toc_to_heading_tree(book.toc))

    # --wrap=none: one line per paragraph. Pandoc's default hard-wraps at 72
    # columns, and markdown_to_elements makes one element per line, so every
    # paragraph was split into several narrative elements (VOR-1437).
    # epub_anchors.lua keeps the spine-file and source ids GFM would drop, as
    # "⟦vorag-anchor:<id>⟧" lines that place TOC chapters whose titles are
    # styled <p> paragraphs (VOR-1427).
    def convert(pandoc_path: str) -> str:
        return str(
            pypandoc.convert_file(
                pandoc_path,
                to="gfm-raw_html",
                format="epub",
                extra_args=["--wrap=none"],
                filters=[str(_ANCHOR_FILTER)],
            )
        )

    rewrites = _toc_target_id_rewrites(Path(parse_path), toc_tree)
    if not rewrites:
        return book, metadata, toc_tree, convert(parse_path)
    with tempfile.TemporaryDirectory(prefix="vorag_epub_anchors_") as tmp:
        markdown = convert(str(_write_with_rewrites(Path(parse_path), rewrites, Path(tmp))))
    return book, metadata, toc_tree, markdown


class EpubParser:
    def __init__(self, figures: FigureStore) -> None:
        self._figures = figures

    async def parse(self, file_path: str) -> ParsedDocument:
        return await asyncio.to_thread(self._parse_sync, file_path)

    def _parse_sync(self, file_path: str) -> ParsedDocument:
        if _ebook_format(file_path) == MOBI_FORMAT:
            return self._parse_mobi_sync(file_path)
        return self._parse_epub_sync(file_path)

    def _parse_mobi_sync(self, file_path: str) -> ParsedDocument:
        with tempfile.TemporaryDirectory(prefix="vorag_mobi_") as tmp:
            converted = Path(tmp) / f"{Path(file_path).stem}.epub"
            _convert_mobi_to_epub(Path(file_path), converted)
            return self._parse_epub_sync(
                str(converted),
                original_file_path=file_path,
                source_format=MOBI_FORMAT,
            )

    def _parse_epub_sync(
        self,
        file_path: str,
        *,
        original_file_path: str | None = None,
        source_format: str = EPUB_FORMAT,
    ) -> ParsedDocument:
        from vo_rag.core.parsing.parsers.epub_markdown import (
            markdown_to_elements,
            strip_anchor_sentinels,
        )

        source_path = original_file_path or file_path
        # Only pay for a TemporaryDirectory (and the
        # normalized copy inside it) when the OPF actually needs the
        # rewrite -- the common case parses `file_path` directly.
        if _epub_needs_normalization(Path(file_path)):
            with tempfile.TemporaryDirectory(prefix="vorag_epub_normalize_") as tmp:
                parse_path = str(_normalize_epub(Path(file_path), Path(tmp)))
                book, metadata, toc_tree, markdown = _read_epub_and_convert(
                    parse_path, source_path, source_format=source_format
                )
        else:
            book, metadata, toc_tree, markdown = _read_epub_and_convert(
                file_path, source_path, source_format=source_format
            )

        elements = markdown_to_elements(markdown, toc_tree)

        if is_image_only(elements):
            raise _build_image_only_ebook(book, metadata)

        pages = _split_pages(elements)
        page_qa = [
            _synthetic_page_qa(idx + 1, grp, parser_used=source_format)
            for idx, grp in enumerate(pages)
        ]
        images = _extract_embedded_images(
            book, self._figures, metadata["document_id"], elements=elements
        )
        return ParsedDocument(
            document_id=metadata["document_id"],
            source_id=metadata["source_id"],
            title=metadata["title"],
            author=str(metadata.get("author", "")),
            total_pages=len(page_qa),
            elements=elements,
            images=images,
            page_qa=page_qa,
            markdown=strip_anchor_sentinels(markdown),
            heading_tree=toc_tree,
            parser_used=source_format,
            metadata=metadata,
        )


def _ebook_format(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".mobi":
        return MOBI_FORMAT
    return EPUB_FORMAT


def _convert_mobi_to_epub(source: Path, target: Path) -> None:
    converter = shutil.which(os.environ.get("VORAG_MOBI_CONVERTER", "ebook-convert"))
    if converter is None:
        raise RuntimeError(
            "MOBI parsing requires Calibre ebook-convert; install calibre on book_rag "
            "workers or set VORAG_MOBI_CONVERTER to a compatible converter"
        )
    timeout_s = int(os.environ.get("VORAG_MOBI_CONVERT_TIMEOUT_S", "900"))
    try:
        subprocess.run(
            [converter, str(source), str(target)],
            check=True,
            capture_output=True,
            env=_ebook_convert_env(),
            text=True,
            timeout=timeout_s,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"MOBI conversion failed: {detail or exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"MOBI conversion timed out after {timeout_s}s") from exc
    if not target.exists() or target.stat().st_size <= 0:
        raise RuntimeError("MOBI conversion produced no EPUB output")


def _ebook_convert_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


def _metadata_from_book(
    book: Any,
    file_path: str,
    *,
    source_format: str = EPUB_FORMAT,
) -> dict[str, Any]:
    raw_bytes = Path(file_path).read_bytes()
    source_id = hashlib.sha256(raw_bytes).hexdigest()
    title = _first_metadata_value(book, "DC", "title") or Path(file_path).stem
    author = _first_metadata_value(book, "DC", "creator") or ""
    language = _first_metadata_value(book, "DC", "language") or ""
    identifier = _first_metadata_value(book, "DC", "identifier") or source_id

    return {
        "document_id": source_id,
        "source_id": identifier,
        "title": title,
        "author": author,
        "language": language,
        "format": source_format,
        "conversion_source_format": source_format if source_format != EPUB_FORMAT else None,
    }


def _first_metadata_value(book: Any, namespace: str, key: str) -> str:
    values = book.get_metadata(namespace, key)
    if not values:
        return ""
    first = values[0]
    if isinstance(first, tuple):
        return str(first[0])
    return str(first)


def _toc_to_heading_tree(toc: Any) -> list[dict[str, Any]]:
    heading_tree: list[dict[str, Any]] = []

    def _walk(node: Any, depth: int) -> None:
        if isinstance(node, (list, tuple)):
            if (
                len(node) == 2
                and not isinstance(node[0], (str, bytes))
                and isinstance(node[1], (list, tuple))
            ):
                entry = _toc_node(node[0], depth)
                entry["children"] = []
                heading_tree.append(entry)
                for child in node[1]:
                    _walk(child, depth + 1)
                return
            for item in node:
                _walk(item, depth)
            return

        heading_tree.append(_toc_node(node, depth))

    _walk(toc, 1)

    normalized: list[dict[str, Any]] = []
    if not heading_tree:
        return normalized

    min_depth = min(int(item.get("depth", 1)) for item in heading_tree)
    for item in heading_tree:
        normalized_item = dict(item)
        normalized_item["depth"] = int(item.get("depth", 1)) - min_depth + 1
        normalized.append(normalized_item)
    return normalized


class _LinkCollector(HTMLParser):
    """``(href, link text)`` of every ``<a href>`` in document order, plus the
    page's ``<title>``, first ``<h1>``-``<h6>`` and first body text. A real
    tokenizer, not a regex over the document: the markup is untrusted."""

    _HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.title = ""
        self.heading = ""
        self.first_text = ""
        self._href: str | None = None
        self._text: list[str] = []
        self._in_title = False
        self._heading_depth = 0
        self._heading_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = tag.lower()
        if name == "title":
            self._in_title = True
        elif name in self._HEADINGS and not self.heading:
            self._heading_depth += 1
        if name != "a":
            return
        self._close()
        href = {key.lower(): value for key, value in attrs}.get("href")
        if href:
            self._href, self._text = href.strip(), []

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name == "title":
            self._in_title = False
        elif name in self._HEADINGS and self._heading_depth:
            self._heading_depth -= 1
            if not self._heading_depth:
                self.heading = " ".join("".join(self._heading_text).split())[:200]
        elif name == "a":
            self._close()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title = " ".join((self.title + data).split())[:200]
            return
        if self._heading_depth:
            self._heading_text.append(data)
        if not self.first_text and data.strip():
            self.first_text = " ".join(data.split())[:200]
        if self._href is not None:
            self._text.append(data)

    def close(self) -> None:
        super().close()
        self._close()

    def _close(self) -> None:
        if self._href is not None:
            self.links.append((self._href, " ".join("".join(self._text).split())))
        self._href, self._text = None, []


# A spine page with at least this many links (with letters in their text) to
# spine files or fragments is a candidate contents page.
_CONTENTS_PAGE_MIN_LINKS = 5
_CONTENTS_PAGE_MAX_BYTES = 2 * 1024**2
# An unlabelled candidate with more entries than this is an index, not a TOC.
_CONTENTS_PAGE_MAX_LINKS = 150
# Index / notes pattern: over this share of links are fragments, into at most
# max(3, fragment links / _INDEX_LINKS_PER_FILE) distinct files.
_INDEX_FRAGMENT_SHARE = 0.8
_INDEX_LINKS_PER_FILE = 20
_BACK_MATTER_LABEL_RE = re.compile(
    r"(?:index|notes|end\s*notes|foot\s*notes|glossary|bibliography|references|works\s+cited)",
    re.IGNORECASE,
)
# The declared charset of an XHTML page, from its first bytes only.
_CHARSET_SNIFF_BYTES = 1024
_XML_ENCODING_RE = re.compile(rb"<\?xml[^>]{0,200}?encoding\s*=\s*[\"']([A-Za-z0-9._:-]{1,40})")
_META_CHARSET_RE = re.compile(
    rb"<meta[^>]{0,300}?charset\s*=\s*[\"']?([A-Za-z0-9._:-]{1,40})", re.I
)


def _decode_markup(content: bytes | str) -> str:
    """``content`` decoded with its XML-declaration or meta charset, falling
    back to UTF-8 with replacement."""
    if isinstance(content, str):
        return content
    head = content[:_CHARSET_SNIFF_BYTES]
    declared = _XML_ENCODING_RE.search(head) or _META_CHARSET_RE.search(head)
    if declared is not None:
        try:
            return content.decode(declared.group(1).decode("ascii"), errors="replace")
        except LookupError:
            pass
    return content.decode("utf-8", errors="replace")


def _page_label_matches(parser: _LinkCollector, pattern: Any) -> bool:
    return any(
        pattern(label.strip(" :.")) for label in (parser.title, parser.heading, parser.first_text)
    )


def _looks_like_index(entries: list[dict[str, Any]]) -> bool:
    fragment_files = [e["href"].partition("#")[0] for e in entries if "#" in e["href"]]
    if len(fragment_files) <= _INDEX_FRAGMENT_SHARE * len(entries):
        return False
    return len(set(fragment_files)) <= max(3, len(fragment_files) // _INDEX_LINKS_PER_FILE)


def _contents_page_toc(book: Any) -> list[dict[str, Any]]:
    """TOC entries from the book's own linked contents page, for books whose
    navigation TOC has at most one entry (VOR-1427). Candidates are spine
    XHTML pages with at least ``_CONTENTS_PAGE_MIN_LINKS`` lettered links to
    spine files or fragments. The first candidate whose title, first heading
    or first text reads "Contents" (or a translation) wins; otherwise the
    first candidate that is not an index, notes, glossary or bibliography
    page (by label, by size, or by fragment links piled into a few files).
    Entries are ``{depth: 1, title: link text, href}`` in link order,
    deduplicated by href, with hrefs relative to the OPF directory like
    ebooklib's own TOC."""
    from vo_rag.core.parsing.parsers.epub_markdown import is_contents_heading

    documents: list[Any] = []
    for entry in book.spine:
        idref = entry[0] if isinstance(entry, list | tuple) else entry
        item = book.get_item_with_id(idref)
        if item is not None and "html" in str(getattr(item, "media_type", "") or ""):
            documents.append(item)
    spine_files = {str(getattr(item, "file_name", "") or "") for item in documents}
    fallback: list[dict[str, Any]] = []
    for item in documents:
        # The raw member, not get_content(): ebooklib re-renders that as UTF-8
        # (ignoring the declared charset) and drops the <head> and its <title>.
        content = getattr(item, "content", None) or item.get_content()
        if not content or len(content) > _CONTENTS_PAGE_MAX_BYTES:
            continue
        parser = _LinkCollector()
        parser.feed(_decode_markup(content))
        parser.close()
        file_name = str(getattr(item, "file_name", "") or "")
        entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_href, text in parser.links:
            path, _, fragment = raw_href.partition("#")
            target = _resolve_href(file_name, unquote(path)) if path else file_name
            if target not in spine_files or (target == file_name and not fragment):
                continue
            if not any(ch.isalpha() for ch in text):
                continue
            href = f"{target}#{fragment}" if fragment else target
            if href in seen:
                continue
            seen.add(href)
            entries.append({"depth": 1, "title": text, "href": href})
        if len(entries) < _CONTENTS_PAGE_MIN_LINKS:
            continue
        if _page_label_matches(parser, is_contents_heading):
            return entries
        if (
            not fallback
            and len(entries) <= _CONTENTS_PAGE_MAX_LINKS
            and not _looks_like_index(entries)
            and not _page_label_matches(parser, _BACK_MATTER_LABEL_RE.fullmatch)
        ):
            fallback = entries
    return fallback


def _with_contents_page_fallback(book: Any, toc_tree: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``toc_tree``, followed by the in-book contents page's links when the
    navigation TOC has at most one entry with an href (VOR-1427)."""
    if sum(1 for node in toc_tree if node.get("href")) > 1:
        return toc_tree
    known = {str(node.get("href") or "") for node in toc_tree}
    extra = [node for node in _contents_page_toc(book) if node["href"] not in known]
    return [*toc_tree, *extra] if extra else toc_tree


def _toc_node(item: Any, depth: int) -> dict[str, Any]:
    title = str(getattr(item, "title", "") or getattr(item, "label", "") or "").strip()
    href = str(getattr(item, "href", "") or getattr(item, "file_name", "") or "").strip()
    if not title and isinstance(item, tuple) and item:
        title = str(item[0])
    if not title:
        title = "Untitled"
    return {"depth": depth, "title": title, "href": href}


def _build_image_only_ebook(book: Any, metadata: dict[str, Any]) -> ImageOnlyEbook:
    """Synthesize a page-image PDF from ``book``'s spine and wrap it for the
    router to hand to ``DoclingParser.parse_scanned`` (VOR-1427)."""
    images = _spine_ordered_image_bytes(book)
    if not images:
        raise RuntimeError("image-only ebook detected but no spine images were found")

    tmpdir = tempfile.TemporaryDirectory(prefix="vorag_epub_image_only_")
    pdf_path = Path(tmpdir.name) / "synthesized.pdf"
    try:
        _write_pdf_from_images(images, pdf_path)
    except Exception:
        tmpdir.cleanup()
        raise
    return ImageOnlyEbook(str(pdf_path), tmpdir, metadata=metadata)


def _spine_ordered_image_bytes(book: Any) -> list[tuple[bytes, str]]:
    """Walk the EPUB spine in reading order and collect each page's image
    bytes (``<img src>`` and SVG-wrapped ``<image href|xlink:href>``, VOR-1427),
    paired with the image's extension so
    ``_write_pdf_from_images`` can embed it losslessly without re-sniffing
    the format."""
    images: list[tuple[bytes, str]] = []
    for entry in book.spine:
        idref = entry[0] if isinstance(entry, list | tuple) else entry
        item = book.get_item_with_id(idref)
        if item is None:
            continue
        media_type = str(getattr(item, "media_type", "") or "")
        if "html" not in media_type:
            continue
        content = item.get_content()
        if not content:
            continue
        html_text = (
            content.decode("utf-8", errors="replace")
            if isinstance(content, bytes)
            else str(content)
        )
        file_name = str(getattr(item, "file_name", "") or "")
        for raw_href in _extract_image_hrefs(html_text):
            href = raw_href.split("#", 1)[0]
            if not href:
                continue
            image_item = book.get_item_with_href(
                _resolve_href(file_name, href)
            ) or book.get_item_with_href(href)
            if image_item is None:
                continue
            payload = image_item.get_content()
            if not payload:
                continue
            image_media_type = str(getattr(image_item, "media_type", "") or "")
            ext = _image_extension(image_item, image_media_type)
            images.append((payload, ext))
    return images


def _resolve_href(base_file_name: str, href: str) -> str:
    base_dir = posixpath.dirname(base_file_name)
    if not base_dir:
        return href
    return posixpath.normpath(posixpath.join(base_dir, href))


# Pillow format name -> pymupdf's `filetype` string, for formats pymupdf's
# stream API accepts directly. Any Pillow-identified
# format missing from this map (e.g. WEBP: this pymupdf build cannot open
# WEBP streams at all, verified) falls back to _page_pdf_bytes_via_pillow_png
# instead -- still content-sniffed, still lossless from the decoded pixels
# onward, just re-containered into a format pymupdf can ingest.
_PIL_FORMAT_TO_FITZ_FILETYPE = {
    "JPEG": "jpg",
    "PNG": "png",
    "GIF": "gif",
    "BMP": "bmp",
    "TIFF": "tiff",
}


def _sniff_pil_format(payload: bytes) -> str | None:
    """The image's real type per its bytes (e.g. "PNG", "JPEG"), or ``None``
    if Pillow can't identify it at all. Never trusts a manifest extension --
    an EPUB's declared extension can be wrong."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(payload)) as probe:
            return probe.format
    except Exception:  # noqa: BLE001 - any of Pillow's many "not an image" errors
        return None


def _flatten_to_opaque(image: Any) -> Any:
    """Composite any transparency onto an opaque white background instead of
    dropping it: ``Image.convert("RGB")`` on an
    image with an alpha channel discards alpha but keeps whatever RGB values
    sit *under* the transparent pixels -- often ``(0, 0, 0)`` -- turning a
    transparent background black and erasing black text drawn on it.
    Non-alpha images keep a PNG-native mode (``L``/``RGB``) unchanged;
    anything else (CMYK, etc.) converts straight to RGB."""
    from PIL import Image

    has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
    if has_alpha:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    if image.mode in ("L", "RGB"):
        return image
    return image.convert("RGB")


def _page_pdf_bytes_via_pillow_png(payload: bytes, *, page_label: str) -> bytes:
    """Decode via Pillow and re-encode losslessly to PNG, for a
    Pillow-identified format pymupdf's stream API doesn't accept directly
    (e.g. WEBP in this pymupdf build)."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(payload)) as image:
            buf = io.BytesIO()
            _flatten_to_opaque(image).save(buf, format="PNG")
            png_bytes = buf.getvalue()
    except Exception as exc:
        raise RuntimeError(
            f"{page_label}: Pillow identified the image but could not decode/re-encode "
            f"it to PNG: {exc}"
        ) from exc

    try:
        return _convert_stream_to_pdf_bytes(png_bytes, "png")
    except Exception as exc:
        raise RuntimeError(
            f"{page_label}: re-encoded to PNG but pymupdf still could not open it: {exc}"
        ) from exc


def _page_pdf_bytes(payload: bytes, ext: str, *, page_label: str) -> bytes:
    """Single-page PDF bytes for one page image, embedded losslessly.

    Determines the real image type from its bytes via Pillow, not the
    (possibly wrong) manifest extension ``ext``:
    every image ``_write_pdf_from_images`` receives is a page the EPUB's
    spine references, so an undecodable one must raise, never be silently
    dropped.
    """
    pil_format = _sniff_pil_format(payload)
    if pil_format is not None:
        filetype = _PIL_FORMAT_TO_FITZ_FILETYPE.get(pil_format.upper())
        if filetype is not None:
            try:
                return _convert_stream_to_pdf_bytes(payload, filetype)
            except Exception as exc:
                raise RuntimeError(
                    f"{page_label}: Pillow identified {pil_format} but pymupdf could not "
                    f"open it as {filetype!r} (manifest ext={ext!r}): {exc}"
                ) from exc
        return _page_pdf_bytes_via_pillow_png(payload, page_label=page_label)

    # Pillow couldn't identify the format at all -- only now fall back to
    # the (possibly wrong) manifest extension.
    try:
        return _convert_stream_to_pdf_bytes(payload, ext or None)
    except Exception as exc:
        raise RuntimeError(
            f"{page_label}: could not decode image; Pillow did not recognize it and "
            f"pymupdf could not open it via the manifest extension {ext!r}: {exc}"
        ) from exc


def _convert_stream_to_pdf_bytes(payload: bytes, filetype: str | None) -> bytes:
    """Open ``payload`` as ``filetype`` and return single-page PDF bytes.
    Raises on any failure -- both ``fitz.open()`` itself and the later
    ``convert_to_pdf()`` (pymupdf can accept a stream at open() time and only
    fail once a page is actually loaded, e.g. for garbage bytes handed a
    plausible-looking filetype)."""
    import fitz  # type: ignore[import-untyped, unused-ignore]  # PyMuPDF

    img_doc = fitz.open(stream=payload, filetype=filetype)
    try:
        return bytes(img_doc.convert_to_pdf())
    finally:
        img_doc.close()


def _write_pdf_from_images(images: list[tuple[bytes, str]], pdf_path: Path) -> None:
    """Build a page-per-image PDF, one page at a time, without decoding every
    page into memory at once and without lossy re-encoding (VOR-1427): a
    Pillow ``save(..., resolution=200)`` call would force a full in-memory
    RGB decode of every page plus a lossy JPEG re-encode, degrading OCR
    input and risking several GB of RSS on a 500+-page scan. pymupdf embeds
    each image's original bytes losslessly via ``convert_to_pdf()`` +
    ``insert_pdf()``, released immediately after each page.
    """
    import fitz  # type: ignore[import-untyped, unused-ignore]  # PyMuPDF

    doc = fitz.open()
    try:
        total = len(images)
        for index, (payload, ext) in enumerate(images, start=1):
            page_label = f"image-only ebook page {index}/{total}"
            page_pdf_bytes = _page_pdf_bytes(payload, ext, page_label=page_label)
            page_pdf = fitz.open("pdf", page_pdf_bytes)
            try:
                doc.insert_pdf(page_pdf)
            finally:
                page_pdf.close()
        doc.save(str(pdf_path))
    finally:
        doc.close()


# An IMAGE element's text is the markdown pandoc wrote for it, e.g.
# ``![alt](images/fig%200.png "title")``; the target is the archive path of
# the manifest item, URL-encoded (VOR-1435).
_MARKDOWN_IMAGE_TARGET = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)")


def _image_path_key(path: str) -> str:
    path = unquote(path).split("#", 1)[0].split("?", 1)[0]
    return posixpath.normpath(path).lstrip("./").lower()


def _image_element_positions(
    elements: list[ParsedElement],
) -> tuple[dict[str, int], dict[str, int]]:
    """First IMAGE element index per referenced image, keyed by normalized
    archive path and by bare file name (for paths pandoc wrote differently)."""
    by_path: dict[str, int] = {}
    by_name: dict[str, int] = {}
    for index, element in enumerate(elements):
        if element.element_type != ElementType.IMAGE:
            continue
        match = _MARKDOWN_IMAGE_TARGET.search(element.text or "")
        if not match:
            continue
        key = _image_path_key(match.group(1))
        by_path.setdefault(key, index)
        by_name.setdefault(posixpath.basename(key), index)
    return by_path, by_name


def _extract_embedded_images(
    book: Any,
    figures: FigureStore,
    src_hash: str,
    *,
    elements: list[ParsedElement] | None = None,
) -> list[ExtractedImage]:
    """Storable manifest images, each attached to the IMAGE element that
    first shows it (VOR-1435) so its figure composite takes that element's
    section and neighbouring text.

    When ``elements`` is given, manifest images the text never shows are
    dropped (VOR-1436): MathML ``altimg`` fallbacks (the equation is already
    in the text), unused assets and duplicate covers have no position or
    context of their own. De Gruyter's Linear Algebra ships 9,961 of them
    against 22 real figures. Without ``elements`` every image is kept at
    element 0 / page 1."""
    by_path, by_name = _image_element_positions(elements or [])
    images: list[ExtractedImage] = []
    for item in book.get_items():
        media_type = str(getattr(item, "media_type", "") or "")
        if not media_type.startswith("image/"):
            continue
        key = _image_path_key(str(getattr(item, "file_name", "") or ""))
        placed = by_path.get(key, by_name.get(posixpath.basename(key)))
        if placed is None and elements is not None:
            continue
        payload = item.get_content()
        if not payload:
            continue

        ext = safe_ext(_image_extension(item, media_type), media_type)
        if ext is None:
            # svg & friends: not storable/servable — skip, regenerable from source.
            continue
        element_index = placed or 0
        page_number = (
            int(elements[element_index].page_number or 1)
            if elements and element_index < len(elements)
            else 1
        )
        image_index = len(images) + 1
        ref = figures.write(src_hash, page_number, image_index, payload, ext=ext)
        images.append(
            ExtractedImage(
                image_id=hashlib.sha256(payload).hexdigest(),
                page_number=page_number,
                image_index=image_index,
                ref=ref,
                file_size=len(payload),
                mime_type=MIME_BY_EXT[ext],
                element_index=element_index,
            )
        )
    return images


def _image_extension(item: Any, media_type: str) -> str:
    item_name = str(getattr(item, "file_name", "") or "")
    suffix = Path(item_name).suffix.lower().lstrip(".")
    if suffix:
        return suffix
    guessed = mimetypes.guess_extension(media_type) or ".bin"
    return guessed.lstrip(".")


def _split_pages(elements: list[Any]) -> list[list[Any]]:
    if not elements:
        return [[]]

    title_depths = [
        int(element.section_depth or 0)
        for element in elements
        if element.element_type == ElementType.TITLE and int(element.section_depth or 0) > 0
    ]
    chapter_depth = min(title_depths) if title_depths else None

    pages: list[list[Any]] = []
    current: list[Any] = []
    for element in elements:
        is_chapter_break = (
            chapter_depth is not None
            and element.element_type == ElementType.TITLE
            and int(element.section_depth or 0) == chapter_depth
        )
        if is_chapter_break and current:
            pages.append(current)
            current = []
        current.append(element)

    if current:
        pages.append(current)
    if not pages:
        pages = [elements]

    for page_number, group in enumerate(pages, start=1):
        for element in group:
            element.page_number = page_number

    return pages


def _synthetic_page_qa(
    page_number: int,
    elements: list[Any],
    *,
    parser_used: str = EPUB_FORMAT,
) -> PageQA:
    return PageQA(
        page_number=page_number,
        ocr_confidence=1.0,
        table_count=sum(1 for element in elements if element.element_type == ElementType.TABLE),
        figure_count=sum(1 for element in elements if element.element_type == ElementType.IMAGE),
        equation_count=sum(
            1 for element in elements if element.element_type == ElementType.EQUATION
        ),
        text_element_count=sum(
            1
            for element in elements
            if element.element_type
            in {
                ElementType.NARRATIVE_TEXT,
                ElementType.TITLE,
                ElementType.LIST_ITEM,
                ElementType.FIGURE_CAPTION,
            }
        ),
        parser_used=parser_used,
    )

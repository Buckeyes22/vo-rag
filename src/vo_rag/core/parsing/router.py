"""vo-rag parser router.

Detects format (magic bytes → MIME → extension), branches:
- any PDF whose pre-parsed cache (``ocr/<sha256>/docling.json`` in Garage)
  is labelled ``"pipeline": "native"`` in its ``meta.json`` → that cached
  layout-only docling parse, as plain ``docling``
- any PDF whose cache is explicitly labelled ``"pipeline": "ocr"`` → the OCR
  farm's result, as ``docling-ocr``, even when ``_pdf_kind`` calls the PDF
  native: the farm only OCRs books whose live parse failed (e.g. a native
  PDF whose docling parse covered 0/34 pages), so parsing it live again
  would only fail again
- otherwise a native PDF → docling (in-process, do_ocr=False); a cache with
  no ``meta.json`` or no pipeline in it is ignored for it (the pipeline is
  never guessed for native text)
- scanned PDF (any ``_pdf_kind`` other than "native") → the OCR farm's
  cached result (farm output with no pipeline label is treated as OCR) when one
  exists, else docling GPU OCR (``DoclingParser.parse_scanned``, VOR-1427)
- docx / pptx / html / xlsx → docling
- epub / mobi → ebook parser; an image-only EPUB (``ImageOnlyEbook``) is
  synthesized to a PDF and OCR'd the same way as a scanned PDF (farm cache
  first)
- text → plain-text fallback

The vendor-SDK OCR forward path and its accelerated-OCR challenger fallback
(and the page-verify pass that re-OCR'd low-confidence pages for a
disagreement check) are gone (VOR-1427): that stack was never deployed.
Scanned PDFs and image-only EPUBs go through the OCR farm's cache or
docling's own GPU OCR instead.
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

try:
    import magic  # type: ignore[import-not-found]
except ImportError:

    class _NoMagic:
        @staticmethod
        def from_file(path: str, mime: bool = True) -> str:
            _ = (path, mime)
            raise RuntimeError("libmagic unavailable")

    magic = _NoMagic()

from vo_rag.core.parsing.contract import (
    ElementType,
    PageQA,
    ParsedDocument,
    ParsedElement,
    element_id,
)
from vo_rag.core.parsing.corruption_gate import _dict_hit_ratio
from vo_rag.core.parsing.parsers.docling_mapping import docling_to_parsed_document
from vo_rag.core.parsing.parsers.epub import ImageOnlyEbook
from vo_rag.core.storage.figures import FigureStore


class _Parser(Protocol):
    async def parse(self, file_path: str, /) -> ParsedDocument: ...


class _FormatAwareParser(Protocol):
    async def parse(
        self, file_path: str, fmt: str, /, *, file_hash: str | None = None
    ) -> ParsedDocument: ...

    async def parse_scanned(self, file_path: str, /, *, file_hash: str) -> ParsedDocument: ...


class _OcrCacheStore(Protocol):
    """The slice of ``GarageClient`` the farm-OCR cache lookup needs.

    ``object_metadata`` returns ``None`` for a missing object and raises for
    every other failure -- that is the not-found signal ``_cache_entry`` relies on.
    """

    async def object_metadata(self, bucket: str, key: str) -> Any: ...

    async def get_bytes(self, bucket: str, key: str) -> bytes: ...


def farm_ocr_key(file_hash: str) -> str:
    """Garage key of the OCR farm's DoclingDocument JSON for a source file."""
    return f"ocr/{file_hash}/docling.json"


def farm_ocr_meta_key(file_hash: str) -> str:
    """Garage key of the cache sidecar (``{"pipeline": ..., "host": ..., ...}``)."""
    return f"ocr/{file_hash}/meta.json"


PIPELINE_NATIVE = "native"
PIPELINE_OCR = "ocr"

# ``ParserRouter._pdf_kind`` (VOR-1427). A PDF is "native" only when enough of
# a spread sample of its pages carries a real text layer:
#: Pages sampled, evenly spread from the first page to the last.
PDF_KIND_SAMPLE_PAGES = 12
#: Share of sampled pages that must each carry at least
#: ``scanned_char_threshold`` characters. Scans often have text on a cover,
#: an archive stamp or a back page (the 24-page "Fluid mechanics" scan: 1 of
#: 12 sampled pages). Sample behind the value: 207 local PDFs of 8+ pages,
#: only 5 of them scans (2 with text on 1 of 12 sampled pages, 3 with none);
#: the 202 native ones had text on at least 8 of 12 (0.67). 0.4 sits in
#: that gap with room for native books with image covers, blank versos and
#: plate galleries; the scan side rests on few files.
NATIVE_MIN_TEXT_PAGE_FRACTION = 0.4
#: The sampled text must be real words, not a garbled or CID-encoded layer.
NATIVE_MIN_DICT_HIT_RATIO = 0.10


@dataclass(frozen=True)
class _CacheEntry:
    """A pre-parsed ``docling.json`` exists; what its ``meta.json`` says about it.

    ``pipeline`` is ``"native"``, ``"ocr"``, or ``None`` when the sidecar or its
    ``pipeline`` field is missing (older farm output without the label).
    """

    pipeline: str | None
    host: str


def _load_docling_json(data: bytes) -> Any:
    """Load a saved DoclingDocument with docling's own JSON loader."""
    from docling_core.types.doc import (  # type: ignore[import-not-found, unused-ignore]
        DoclingDocument,
    )

    with tempfile.TemporaryDirectory(prefix="vorag_farm_ocr_") as tmp:
        path = Path(tmp) / "docling.json"
        path.write_bytes(data)
        return DoclingDocument.load_from_json(path)


_EXT_MAP = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".html": "html",
    ".htm": "html",
    ".xlsx": "xlsx",
    ".epub": "epub",
    ".mobi": "mobi",
    ".txt": "text",
    ".md": "text",
}

_MIME_MAP = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/html": "html",
    "text/markdown": "text",
    "text/plain": "text",
    "application/epub+zip": "epub",
    "application/x-mobipocket-ebook": "mobi",
    "application/vnd.amazon.ebook": "mobi",
}


class ParserRouter:
    def __init__(
        self,
        docling: _FormatAwareParser,
        epub: _Parser,
        *,
        scanned_char_threshold: int = 50,
        garage: _OcrCacheStore | None = None,
        bucket: str | None = None,
        figures: FigureStore | None = None,
    ) -> None:
        self.docling, self.epub = docling, epub
        # OCR-farm cache (VOR-1427): only consulted when both a
        # Garage client and a bucket are given; mapping the cached JSON writes
        # figures, so a FigureStore is required alongside them.
        if garage is not None and bucket and figures is None:
            raise ValueError("ParserRouter: a Garage OCR cache needs a figures store")
        self.garage = garage
        self.ocr_bucket = bucket if garage is not None else None
        self.figures = figures
        self.scanned_char_threshold = scanned_char_threshold

    async def parse(
        self,
        file_path: str,
        file_hash: str,
        mime_type: str | None = None,
    ) -> ParsedDocument:
        fmt = self._detect_format(file_path, mime_type)
        if fmt == "pdf":
            doc = await self._parse_pdf(file_path, file_hash)
        elif fmt in ("docx", "pptx", "html", "xlsx"):
            doc = await self.docling.parse(file_path, fmt, file_hash=file_hash)
        elif fmt in ("epub", "mobi"):
            doc = await self._parse_ebook(file_path, file_hash)
        elif fmt == "text":
            doc = await self._parse_text(file_path, file_hash)
        else:
            raise ValueError(f"unsupported format: {fmt!r}")
        return doc

    async def _parse_pdf(self, file_path: str, file_hash: str) -> ParsedDocument:
        entry = await self._cache_entry(file_hash)
        if entry is not None and entry.pipeline in (PIPELINE_NATIVE, PIPELINE_OCR):
            # An explicit label is trusted for any PDF. "ocr" also covers a PDF
            # _pdf_kind calls native: the farm only OCRs books whose live parse
            # failed, so a live parse would fail the same way again.
            return await self._from_cache(file_hash, file_path, entry)
        kind = await self._pdf_kind(file_path)
        if kind == "native":
            # An unlabelled cache (no meta.json, or no pipeline in it) never
            # stands in for a native parse: its pipeline isn't guessed, and
            # docling.json is not downloaded.
            return await self.docling.parse(file_path, "pdf", file_hash=file_hash)
        if entry is not None:
            return await self._from_cache(file_hash, file_path, entry)
        return await self.docling.parse_scanned(file_path, file_hash=file_hash)

    async def _parse_ebook(self, file_path: str, file_hash: str) -> ParsedDocument:
        try:
            return await self.epub.parse(file_path)
        except ImageOnlyEbook as image_only:
            try:
                entry = await self._cache_entry(file_hash)
                if entry is not None and entry.pipeline == PIPELINE_NATIVE:
                    # An image-only ebook has no text layer to pre-parse, and
                    # a plain-docling result for a non-PDF upload would skip
                    # the parse step's PDF gates (they key on is_pdf or
                    # docling-ocr). Refuse the contradictory cache.
                    raise ValueError(
                        f"pre-parsed cache for image-only ebook {file_hash} is labelled "
                        f"pipeline={PIPELINE_NATIVE!r}; only OCR output can stand in for it"
                    )
                doc = (
                    await self._from_cache(file_hash, image_only.pdf_path, entry)
                    if entry is not None
                    else await self.docling.parse_scanned(image_only.pdf_path, file_hash=file_hash)
                )
            finally:
                image_only.tmpdir.cleanup()
            doc.document_id = str(image_only.metadata.get("document_id") or file_hash)
            doc.source_id = str(image_only.metadata.get("source_id") or file_hash)
            doc.title = str(image_only.metadata.get("title") or doc.title)
            doc.author = str(image_only.metadata.get("author") or "")
            doc.metadata = dict(doc.metadata or {})
            # "epub" or "mobi": an image-only MOBI is
            # converted to EPUB before parsing, but it's still a MOBI upload,
            # not an EPUB one -- label it accordingly.
            source_format = str(image_only.metadata.get("format") or "epub")
            doc.metadata["source_format"] = f"{source_format}_image_only"
            return doc

    async def _cache_entry(self, file_hash: str) -> _CacheEntry | None:
        """Whether a pre-parsed ``docling.json`` exists for this file, and its label.

        One HEAD for ``ocr/<file_hash>/docling.json``; ``meta.json`` is read
        only when that exists. Returns ``None`` when no cache is configured or
        there is no ``docling.json``. A Garage failure, or a sidecar that
        isn't a JSON object with a known ``pipeline``, raises naming the file
        hash -- it must never silently fall through to a live parse.
        """
        if self.garage is None or not self.ocr_bucket or self.figures is None:
            return None
        garage, bucket = self.garage, self.ocr_bucket
        meta_key = farm_ocr_meta_key(file_hash)
        try:
            if await garage.object_metadata(bucket, farm_ocr_key(file_hash)) is None:
                return None
            raw = (
                await garage.get_bytes(bucket, meta_key)
                if await garage.object_metadata(bucket, meta_key) is not None
                else None
            )
        except Exception as exc:
            raise RuntimeError(
                f"pre-parsed cache check failed for file {file_hash}: {type(exc).__name__}: {exc}"
            ) from exc
        if raw is None:
            return _CacheEntry(pipeline=None, host="unknown")
        try:
            meta = json.loads(raw)
        except ValueError as exc:
            raise ValueError(f"{meta_key} for file {file_hash} is not valid JSON: {exc}") from exc
        if not isinstance(meta, dict):
            raise ValueError(f"{meta_key} for file {file_hash} is not a JSON object")
        pipeline = meta.get("pipeline") or None
        if pipeline not in (None, PIPELINE_NATIVE, PIPELINE_OCR):
            raise ValueError(f"{meta_key} for file {file_hash} has unknown pipeline {pipeline!r}")
        return _CacheEntry(pipeline=pipeline, host=str(meta.get("host") or "unknown"))

    async def _from_cache(
        self, file_hash: str, source_path: str, entry: _CacheEntry
    ) -> ParsedDocument:
        """Map the cached ``ocr/<file_hash>/docling.json`` to a ``ParsedDocument``.

        ``pipeline == "native"`` → plain ``docling`` (an off-box layout-only
        parse); otherwise ``docling-ocr`` (the OCR farm).
        """
        garage, bucket, figures = self.garage, self.ocr_bucket, self.figures
        if garage is None or not bucket or figures is None:  # only reached via _cache_entry
            raise RuntimeError("ParserRouter: no pre-parsed cache is configured")
        native = entry.pipeline == PIPELINE_NATIVE
        try:
            data = await garage.get_bytes(bucket, farm_ocr_key(file_hash))
        except Exception as exc:
            raise RuntimeError(
                f"pre-parsed cache read failed for file {file_hash}: {type(exc).__name__}: {exc}"
            ) from exc

        def _map() -> ParsedDocument:
            doc = _load_docling_json(data)
            return docling_to_parsed_document(
                doc,
                figures,
                source_path,
                "pdf",
                src_hash=file_hash,
                parser_used="docling" if native else "docling-ocr",
            )

        parsed = await asyncio.to_thread(_map)
        # The cached parse ran on a file named after the hash (the farm OCRs
        # ``jobs/<sha256>.pdf``), so docling named the document after the hash;
        # a local parse names it after the source file's stem. Keep the title
        # the local path would produce.
        parsed.title = Path(source_path).stem
        parsed.metadata = dict(parsed.metadata or {})
        if native:
            parsed.metadata["pre_parsed"] = PIPELINE_NATIVE
        else:
            parsed.metadata["ocr_source"] = "farm"
        # For a native pre-parse no OCR ran; the key is still ``ocr_host`` so
        # every cache-served document records where it was parsed under one name.
        parsed.metadata["ocr_host"] = entry.host
        return parsed

    async def _parse_text(self, file_path: str, file_hash: str) -> ParsedDocument:
        def _read() -> str:
            return Path(file_path).read_text(encoding="utf-8", errors="replace")

        text = await asyncio.to_thread(_read)
        title = Path(file_path).stem
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
        elements = [
            ParsedElement(
                element_id=element_id(paragraph, 1, index),
                element_type=ElementType.NARRATIVE_TEXT,
                text=paragraph,
                page_number=1,
            )
            for index, paragraph in enumerate(paragraphs)
        ]
        return ParsedDocument(
            document_id=file_hash,
            source_id=file_hash,
            title=title,
            total_pages=1,
            elements=elements,
            images=[],
            page_qa=[
                PageQA(
                    page_number=1,
                    ocr_confidence=1.0,
                    text_element_count=len(elements),
                    parser_used="text",
                )
            ],
            markdown=text,
            heading_tree=[],
            metadata={"format": "text"},
            parser_used="text",
        )

    def _detect_format(self, path: str, mime: str | None) -> str:
        # Priority 1: magic bytes
        try:
            detected_mime = magic.from_file(path, mime=True)
        except (RuntimeError, OSError):
            detected_mime = None
        if detected_mime and detected_mime in _MIME_MAP:
            return _MIME_MAP[detected_mime]
        # Priority 2: provided MIME
        if mime and mime in _MIME_MAP:
            return _MIME_MAP[mime]
        # Priority 3: extension
        ext = Path(path).suffix.lower()
        if ext in _EXT_MAP:
            return _EXT_MAP[ext]
        raise ValueError(f"cannot detect format for {path!r}")

    async def _pdf_kind(self, path: str) -> str:
        """``"native"`` or ``"scanned"`` from a spread sample of pages.

        Native needs at least ``NATIVE_MIN_TEXT_PAGE_FRACTION`` of the sampled
        pages to carry ``scanned_char_threshold`` characters each, and the
        sampled text to pass the dictionary check. Counting pages, not
        averaging characters, keeps one text-heavy cover or archive stamp
        page from making a scan "native".
        """

        def _probe() -> str:
            import fitz  # type: ignore[import-untyped]

            with fitz.open(path) as pdf:
                page_count = pdf.page_count
                if page_count == 0:
                    return "native"
                # Sample pages spread across the whole document, not just the
                # first few. Books with image-heavy front matter (cover, plates,
                # title/copyright pages) carry no extractable text up front yet
                # are fully text-native in the body; a first-pages-only probe
                # misroutes them to OCR. A true scan reads ~no text on almost
                # every sampled page and stays "scanned".
                sample_size = min(page_count, PDF_KIND_SAMPLE_PAGES)
                if page_count <= sample_size:
                    pages = list(range(page_count))
                else:
                    pages = sorted(
                        {
                            round(i * (page_count - 1) / (sample_size - 1))
                            for i in range(sample_size)
                        }
                    )
                texts = [pdf[i].get_text("text").strip() for i in pages]
                with_text = [t for t in texts if len(t) >= self.scanned_char_threshold]
                if len(with_text) / len(pages) < NATIVE_MIN_TEXT_PAGE_FRACTION:
                    return "scanned"
                if _dict_hit_ratio("\n".join(with_text)) < NATIVE_MIN_DICT_HIT_RATIO:
                    return "scanned"
                return "native"

        return await asyncio.to_thread(_probe)

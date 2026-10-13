"""Docling in-process parser (lazy-imports the heavy ``docling`` package).

The heavy ``DocumentConverter`` import is gated so callers
that only need the contract can import this module without docling
installed.

mypy note: every ``docling.*`` (and ``torch``) import below carries
``# type: ignore[import-not-found, unused-ignore]``. mypy only reports
"import-not-found" the first time it sees a given missing module path in a
file and silently resolves later imports of that same path to ``Any``
without re-reporting it -- which would otherwise flag a plain
``# type: ignore[import-not-found]`` on those later occurrences as "unused"
(and vice versa depending on file order). Listing both codes together makes
each ignore valid regardless of which occurrence -- first or later -- mypy
considers it, so reordering these functions never breaks the build.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import tempfile
import threading
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from vo_rag.core.parsing.contract import ParsedDocument
from vo_rag.core.parsing.parsers.docling_mapping import docling_to_parsed_document
from vo_rag.core.parsing.parsers.pdf_page_rebuild import (
    coverage_shortfall,
    doubled_pages,
    missing_words,
    page_words,
    plain_text_elements,
    rebuild_ranges,
    splice_pages,
    text_layer_pages,
)
from vo_rag.core.storage.figures import FigureStore

if TYPE_CHECKING:
    from docling.datamodel.pipeline_options import (  # type: ignore[import-not-found, unused-ignore]
        PdfPipelineOptions,
    )


log = structlog.get_logger(__name__)

#: A page whose longest side exceeds this (in points) is rescaled before
#: docling opens the PDF (VOR-1434). The limit is 5,000pt: a higher limit
#: (14,400pt) never fired on the real failing page (13,400 x 9,100pt). A0, the
#: largest standard paper size, is 3,370pt on its long
#: side, so no legitimate book, plate or fold-out page is touched. Preflight
#: imports this constant so the two can never disagree.
OVERSIZE_PAGE_PT = 5_000.0

#: Longest side, in points, of a page after rescaling (the hand fix applied on
#: 2026-09-26).
RESCALED_PAGE_MAX_PT = 1_200.0


def _longest_side(page: Any) -> float:
    """Longest side, in points, of a PyMuPDF page as displayed (``page.rect``).
    The single oversize measure; preflight imports it."""
    rect = page.rect
    return max(float(rect.width), float(rect.height))


def rescale_oversize_pages(path: Path, workdir: Path) -> Path | None:
    """Write a copy of the PDF at ``path`` into ``workdir`` (same file name, so
    docling names the document after the original) with every page whose
    longest side exceeds ``OVERSIZE_PAGE_PT`` redrawn at ``RESCALED_PAGE_MAX_PT``
    via PyMuPDF ``show_pdf_page`` (vector content and text layer kept). Page
    count and order are preserved; other pages are copied as-is. ``path``
    itself is never modified.

    ``None`` when no page is oversize, or when PyMuPDF fails anywhere in the
    scan or the rewrite (logged as a warning): this is a repair step that runs
    on every PDF, so a file PyMuPDF can't handle goes to docling unchanged
    and fails, if at all, with docling's own error.
    """
    import fitz  # type: ignore[import-untyped, unused-ignore]  # PyMuPDF

    out = workdir / path.name
    try:
        with fitz.open(path) as src:
            oversize = [page.number for page in src if _longest_side(page) > OVERSIZE_PAGE_PT]
            if not oversize:
                return None
            with fitz.open() as dst:
                dst.insert_pdf(src)
                for index in oversize:
                    source_page = src[index]
                    scale = RESCALED_PAGE_MAX_PT / _longest_side(source_page)
                    # Insert the scaled page in front of the original, draw
                    # the original onto it, then drop the original: order and
                    # count are unchanged.
                    page = dst.new_page(
                        pno=index,
                        width=float(source_page.rect.width) * scale,
                        height=float(source_page.rect.height) * scale,
                    )
                    page.show_pdf_page(page.rect, src, index)
                    dst.delete_page(index + 1)
                # A throwaway temp copy: no garbage collection / recompression.
                dst.save(out)
    except Exception as exc:  # noqa: BLE001 - a repair step must not replace docling's own error
        log.warning(
            "pdf_oversize_rescale_skipped",
            source=str(path),
            error=f"{type(exc).__name__}: {exc}",
        )
        out.unlink(missing_ok=True)
        return None
    log.info("pdf_oversize_pages_rescaled", source=str(path), pages=[i + 1 for i in oversize])
    return out


def pdfium_can_open(path: Path) -> bool:
    """True when pypdfium2 -- docling's PDF backend -- can open ``path``, or
    when pypdfium2 isn't installed (nothing to check against; docling itself
    needs it, so only lean test environments lack it)."""
    try:
        import pypdfium2  # type: ignore[import-not-found, import-untyped, unused-ignore]
    except ImportError:
        return True

    try:
        document = pypdfium2.PdfDocument(str(path))
    except Exception:  # noqa: BLE001 - any load failure means "docling can't read it"
        return False
    document.close()
    return True


def unlock_for_pdfium(path: Path, workdir: Path) -> Path | None:
    """A copy of ``path`` that pypdfium2 can open, when it can't open ``path``
    but PyMuPDF can without a password (VOR-1427).

    Real case: a scanned aerodynamics textbook ships an owner-only
    encryption dictionary whose security scheme PDFium rejects ("Unsupported
    security scheme"), so docling reported the file as "not valid" while
    PyMuPDF and docling-parse read every page. The copy is written without
    encryption; content, text layer and page order are unchanged. ``None`` when
    pypdfium2 already opens the file, when it needs a user password, or when
    the rewritten copy still can't be opened (docling then fails with its own
    error). ``path`` itself is never modified.
    """
    if pdfium_can_open(path):
        return None
    import fitz  # type: ignore[import-untyped, unused-ignore]  # PyMuPDF

    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / path.name
    try:
        with fitz.open(path) as src:
            if src.needs_pass:
                return None
            src.save(out, encryption=fitz.PDF_ENCRYPT_NONE)
    except Exception as exc:  # noqa: BLE001 - a repair step must not replace docling's own error
        log.warning("pdf_unlock_skipped", source=str(path), error=f"{type(exc).__name__}: {exc}")
        out.unlink(missing_ok=True)
        return None
    if not pdfium_can_open(out):
        out.unlink(missing_ok=True)
        return None
    log.info("pdf_unlocked_for_pdfium", source=str(path))
    return out


@contextlib.contextmanager
def _pdf_for_docling(file_path: str) -> Iterator[str]:
    """The path docling should open for the PDF at ``file_path``: the file
    itself, or a temp copy (deleted on exit) when a page is oversize or when
    pypdfium2 can't open the file but PyMuPDF can.
    """
    with tempfile.TemporaryDirectory(prefix="vorag_pdf_rescale_") as tmp:
        rescaled = rescale_oversize_pages(Path(file_path), Path(tmp))
        current = rescaled if rescaled is not None else Path(file_path)
        unlocked = unlock_for_pdfium(current, Path(tmp) / "unlocked")
        yield str(unlocked if unlocked is not None else current)


def _is_pdf(file_path: str, fmt: str | None) -> bool:
    return fmt == "pdf" or (fmt is None and file_path.lower().endswith(".pdf"))


_PYPDFIUM_BACKENDS = ("pypdfium", "pypdfium2")

#: Pages per docling_parse conversion when a whole PDF is redone (VOR-1453).
#: docling_parse's per-glyph preprocessing is what blew lanes up to 20-58 GB
#: on glyph-dense books (VOR-1424), so it never gets a whole book at once.
WHOLE_DOCUMENT_CHUNK_PAGES = 8


def pdfium_page_texts(path: str) -> list[str]:
    """pypdfium2's plain text of every page of the PDF at ``path``."""
    import pypdfium2  # type: ignore[import-not-found, import-untyped, unused-ignore]

    document = pypdfium2.PdfDocument(path)
    try:
        texts: list[str] = []
        for index in range(len(document)):
            page = document[index]
            textpage = page.get_textpage()
            try:
                texts.append(str(textpage.get_text_range()))
            finally:
                textpage.close()
                page.close()
        return texts
    finally:
        document.close()


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore[import-not-found, unused-ignore]
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _item_pages(item: Any) -> set[int]:
    return {
        int(prov.page_no)
        for prov in (getattr(item, "prov", None) or [])
        if isinstance(getattr(prov, "page_no", None), int)
    }


def _docling_items(document: Any) -> list[Any]:
    return [
        *(getattr(document, "texts", None) or []),
        *(getattr(document, "tables", None) or []),
        *(getattr(document, "pictures", None) or []),
    ]


def _docling_content_pages(document: Any) -> set[int]:
    """Pages holding a non-empty docling text item or a table."""
    pages: set[int] = set()
    texts = getattr(document, "texts", None) or []
    tables = getattr(document, "tables", None) or []
    for item in [*texts, *tables]:
        if hasattr(item, "text") and not str(getattr(item, "text", "") or "").strip():
            continue
        pages |= _item_pages(item)
    return pages


def _multi_page_spans(document: Any) -> list[tuple[int, int]]:
    """``(first, last)`` page of every docling item that runs over a page break."""
    spans: list[tuple[int, int]] = []
    for item in _docling_items(document):
        pages = _item_pages(item)
        if len(pages) > 1:
            spans.append((min(pages), max(pages)))
    return spans


def _failed_pages(result: Any, total_pages: int) -> set[int]:
    """Pages docling reported as failed (e.g. CUDA out of memory in the layout
    stage): a partial conversion keeps only the pages that completed."""
    status = getattr(getattr(result, "status", None), "value", getattr(result, "status", None))
    if status in (None, "success"):
        return set()
    completed = {getattr(page, "page_no", None) for page in (getattr(result, "pages", None) or [])}
    return {page for page in range(1, total_pages + 1) if page not in completed}


class OcrRequiresGpu(RuntimeError):
    """Raised when scanned-PDF / image-only-EPUB OCR is attempted without CUDA.

    the operator's rule: OCR never runs on the CPU, on any machine. docling's default
    OCR engine is RapidOCR on its default ONNX-based backend, and only the
    CPU-only build of that runtime is installed here (its GPU-capable
    counterpart is not) -- so that default backend always runs on the CPU,
    silently ignoring docling's own ``use_cuda`` flags. This guard fails
    loudly instead, before any conversion is attempted.
    """


def scanned_pipeline_options() -> PdfPipelineOptions:
    """OCR pipeline options for scanned PDFs / image-only EPUBs (VOR-1427).

    Returns a fresh ``PdfPipelineOptions`` per call -- no module-level
    constant -- so this module's own scanned converter and the OCR
    farm worker each get their own instance while sharing identical settings
    verbatim.

    RapidOCR on the ``torch`` backend is the only OCR backend that actually
    reaches the GPU here: docling's default OCR backend always runs on the
    CPU because only its CPU-only ONNX-based runtime is installed (its
    GPU-capable counterpart is not). The accelerator device is always CUDA
    -- a ``VORAG_DOCLING_DEVICE=cpu`` value is ignored for this pipeline, and
    ``VORAG_DOCLING_DEVICE=cuda:N`` selects a specific GPU -- because OCR
    must never run on the CPU, on any host.
    """
    from docling.datamodel.pipeline_options import (  # type: ignore[import-not-found, unused-ignore]
        PdfPipelineOptions,
        RapidOcrOptions,
    )

    try:  # docling >=2.26 moved accelerator options to their own module
        from docling.datamodel.accelerator_options import (  # type: ignore[import-not-found, unused-ignore]
            AcceleratorDevice,
            AcceleratorOptions,
        )
    except ImportError:  # pragma: no cover - older docling layout
        from docling.datamodel.pipeline_options import (  # type: ignore[import-not-found, unused-ignore]
            AcceleratorDevice,
            AcceleratorOptions,
        )

    raw_device = os.environ.get("VORAG_DOCLING_DEVICE", "").strip().lower()
    device: Any = raw_device if raw_device.startswith("cuda:") else AcceleratorDevice.CUDA

    options = PdfPipelineOptions(do_ocr=True)
    options.ocr_options = RapidOcrOptions(backend="torch", force_full_page_ocr=True)
    options.accelerator_options = AcceleratorOptions(device=device)
    options.generate_picture_images = True
    return options


def _require_cuda() -> None:
    """Fail loudly instead of silently OCR'ing on the CPU."""
    import torch  # type: ignore[import-not-found, unused-ignore]

    if not torch.cuda.is_available():
        raise OcrRequiresGpu(
            "docling scanned-PDF OCR requires a CUDA GPU; torch.cuda.is_available() is False"
        )


class DoclingParser:
    def __init__(
        self,
        figures: FigureStore,
        *,
        do_ocr: bool = True,
        device: str | None = None,
    ) -> None:
        self._do_ocr = do_ocr
        # Accelerator selection. Default (unset / "auto") preserves docling's
        # own device pick. Set device="cuda" or env VORAG_DOCLING_DEVICE=cuda
        # to force the layout/table models onto a GPU.
        dev = (device or os.environ.get("VORAG_DOCLING_DEVICE", "")).strip().lower()
        self._device = dev or "auto"
        self._figures = figures

        # Local docling imports: ruff I001 wants vo_rag.* before external, but
        # keeping the docling import close to the class init avoids a heavy import
        # at module load.
        from docling.datamodel.base_models import (  # type: ignore[import-not-found, unused-ignore]  # noqa: I001
            InputFormat,
        )
        from docling.datamodel.pipeline_options import (  # type: ignore[import-not-found, unused-ignore]
            PdfPipelineOptions,
        )
        from docling.document_converter import (  # type: ignore[import-not-found, unused-ignore]
            DocumentConverter,
            PdfFormatOption,
        )

        try:  # docling >=2.26 moved accelerator options to their own module
            from docling.datamodel.accelerator_options import (  # type: ignore[import-not-found, unused-ignore]
                AcceleratorDevice,
                AcceleratorOptions,
            )
        except ImportError:  # pragma: no cover - older docling layout
            from docling.datamodel.pipeline_options import (  # type: ignore[import-not-found, unused-ignore]
                AcceleratorDevice,
                AcceleratorOptions,
            )

        options = PdfPipelineOptions(do_ocr=do_ocr)
        options.generate_picture_images = True
        if dev in ("cuda", "gpu"):
            options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)
        elif dev == "cpu":
            options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)
        # PDF text backend. Default (unset) keeps docling's own docling_parse.
        # VORAG_DOCLING_PDF_BACKEND=pypdfium swaps in pypdfium2: on glyph-dense
        # PDFs docling_parse's single-threaded preprocess builds per-glyph cells
        # that pushed dispatcher lanes past 20-58 GB RSS and starved the GPU
        # (VOR-1424: a 590-page book took >20 GB and was killed; pypdfium2
        # parsed it in 6 min at 4.4 GB peak).
        backend = os.environ.get("VORAG_DOCLING_PDF_BACKEND", "").strip().lower()
        pdf_option = PdfFormatOption(pipeline_options=options)
        if backend in ("pypdfium", "pypdfium2"):
            from docling.backend.pypdfium2_backend import (  # type: ignore[import-not-found, unused-ignore]
                PyPdfiumDocumentBackend,
            )

            pdf_option = PdfFormatOption(pipeline_options=options, backend=PyPdfiumDocumentBackend)
        self._pdf_backend = backend or "default"
        self._pdf_pipeline_options = options
        self._converter = DocumentConverter(format_options={InputFormat.PDF: pdf_option})
        # docling's default docling_parse backend, same pipeline options, for
        # the pypdfium repair (VOR-1453). Built on first use, one per parser.
        self._docling_parse_converter: Any | None = None
        self._docling_parse_converter_lock = threading.Lock()
        # Built lazily on first parse_scanned() call, not here: __init__ must
        # keep working on hosts without CUDA (only the normal, do_ocr=False
        # converter is used there). _require_cuda() runs once, at that point.
        self._scanned_converter: Any | None = None
        # parse_scanned() runs on a thread pool (asyncio.to_thread); guard
        # against two concurrent first calls each building (and GPU-loading)
        # their own converter.
        self._scanned_converter_lock = threading.Lock()

    async def parse(
        self, file_path: str, fmt: str | None = None, *, file_hash: str | None = None
    ) -> Any:
        return await asyncio.to_thread(self._parse_sync, file_path, fmt, file_hash)

    def _parse_sync(self, file_path: str, fmt: str | None, file_hash: str | None) -> Any:
        try:
            # Canonical source identity: prefer the hash computed by the parse
            # step; fall back to hashing the file so direct callers still get
            # deterministic keys (never a temp-path hash).
            src_hash = file_hash or hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
            if _is_pdf(file_path, fmt):
                with _pdf_for_docling(file_path) as convert_path:
                    result = self._converter.convert(convert_path)
                    if self._pdf_backend in _PYPDFIUM_BACKENDS:
                        return self._repaired_pypdfium_parse(
                            result, convert_path, file_path, fmt, src_hash
                        )
            else:
                result = self._converter.convert(file_path)
            return docling_to_parsed_document(
                result.document, self._figures, file_path, fmt, src_hash=src_hash
            )
        finally:
            if self._device in ("cuda", "gpu"):
                _release_cuda_cache()

    # -- pypdfium repair (VOR-1453) ------------------------------------------

    def _repaired_pypdfium_parse(
        self, result: Any, convert_path: str, file_path: str, fmt: str | None, src_hash: str
    ) -> ParsedDocument:
        """Map a pypdfium-backed conversion, repairing what that backend got
        wrong with docling's default docling_parse backend -- never the whole
        book through docling_parse at once (VOR-1424's memory blow-up):

        - a parse covering too few pages (``coverage_shortfall``) is redone
          whole in ``WHOLE_DOCUMENT_CHUNK_PAGES`` page ranges, and kept when it
          covers more pages;
        - otherwise every page the parse doubled (``doubled_pages``) or docling
          failed to convert is rebuilt in small page ranges and spliced in;
          a page docling_parse fails on, or rebuilds without every word of
          pypdfium2's text (``missing_words``), gets pypdfium2's plain text
          instead.

        docling_parse runs only on CUDA; on a host without it the rebuilt pages
        get pypdfium2's plain text and the whole-document retry is skipped.
        Every repair is recorded in ``metadata``.
        """
        page_texts = pdfium_page_texts(convert_path)
        total_pages = len(page_texts)
        on_gpu = self._device in ("cuda", "gpu") and _cuda_available()

        covered = _docling_content_pages(result.document)
        shortfall = coverage_shortfall(covered, text_layer_pages(page_texts), total_pages)
        fallback_record: dict[str, Any] | None = None
        if shortfall is not None:
            fallback_record = {
                "reason": shortfall,
                "from_backend": "pypdfium2",
                "to_backend": "docling_parse",
                "chunk_pages": WHOLE_DOCUMENT_CHUNK_PAGES,
                "content_pages_before": len(covered),
                "total_pages": total_pages,
            }
            document, error = (
                self._docling_parse_whole_document(convert_path, total_pages)
                if on_gpu
                else (None, "skipped: no CUDA device")
            )
            after = _docling_content_pages(document) if document is not None else set()
            fallback_record["content_pages_after"] = len(after)
            if error:
                fallback_record["error"] = error
            fallback_record["used"] = document is not None and len(after) > len(covered)
            log.warning("pdf_pypdfium_coverage_fallback", source=file_path, **fallback_record)
            if fallback_record["used"]:
                parsed = docling_to_parsed_document(
                    document, self._figures, file_path, fmt, src_hash=src_hash
                )
                parsed.metadata["pdf_backend_fallback"] = fallback_record
                return parsed

        parsed = docling_to_parsed_document(
            result.document, self._figures, file_path, fmt, src_hash=src_hash
        )
        if fallback_record is not None:
            parsed.metadata["pdf_backend_fallback"] = fallback_record

        doubled = doubled_pages(parsed.elements, page_texts)
        reasons = dict.fromkeys(_failed_pages(result, total_pages), "docling_page_failed")
        reasons.update(dict.fromkeys(doubled, "doubled_text"))
        if not reasons:
            return parsed

        replacements: dict[int, list[Any]] = {}
        records: list[dict[str, Any]] = []
        for first, last in rebuild_ranges(reasons, _multi_page_spans(result.document)):
            rebuilt, error = (
                self._docling_parse_range(convert_path, first, last, file_path, fmt, src_hash)
                if on_gpu
                else (None, "skipped: no CUDA device")
            )
            range_texts = [
                element.text
                for page in range(first, last + 1)
                for element in (rebuilt or {}).get(page, [])
            ]
            lost = (
                missing_words(page_texts[first - 1 : last], range_texts)
                if rebuilt is not None
                else Counter[str]()
            )
            for page in range(first, last + 1):
                # The range's missing words, charged to every page that holds
                # one of them (a merged paragraph may file a page's words
                # under the page before it).
                page_lost = sorted(
                    word for word in lost if word in page_words(page_texts[page - 1])
                )
                record: dict[str, Any] = {
                    "page": page,
                    "reason": reasons.get(page, "shares_a_paragraph_with_a_rebuilt_page"),
                }
                if page in doubled:
                    record["doubled_shingles"] = doubled[page].shingles
                    record["extra_words"] = doubled[page].extra_words
                if rebuilt is not None and not page_lost:
                    replacements[page] = rebuilt.get(page, [])
                    record["rebuilt_with"] = "docling_parse"
                else:
                    replacements[page] = plain_text_elements(page, page_texts[page - 1])
                    record["rebuilt_with"] = "pypdfium2_text"
                    if page_lost:
                        record["docling_parse_missing_words"] = page_lost[:20]
                    if error:
                        record["error"] = error
                records.append(record)

        parsed = splice_pages(parsed, replacements)
        parsed.metadata["pdf_page_rebuild"] = {"backend": "docling_parse", "pages": records}
        log.info(
            "pdf_pypdfium_pages_rebuilt",
            source=file_path,
            pages=[record["page"] for record in records],
            plain_text_pages=[
                record["page"] for record in records if record["rebuilt_with"] != "docling_parse"
            ],
        )
        return parsed

    def _get_docling_parse_converter(self) -> Any:
        if self._docling_parse_converter is None:
            with self._docling_parse_converter_lock:
                if self._docling_parse_converter is None:
                    from docling.datamodel.base_models import (  # type: ignore[import-not-found, unused-ignore]  # noqa: I001
                        InputFormat,
                    )
                    from docling.document_converter import (  # type: ignore[import-not-found, unused-ignore]
                        DocumentConverter,
                        PdfFormatOption,
                    )

                    option = PdfFormatOption(pipeline_options=self._pdf_pipeline_options)
                    converter = DocumentConverter(format_options={InputFormat.PDF: option})
                    # docling caches a pipeline (its layout and table models)
                    # per pipeline class and options, not per text backend:
                    # sharing the cache runs docling_parse through the models
                    # the pypdfium converter already put on the GPU instead of
                    # loading a second set next to it (lanes share a GPU).
                    shared = getattr(self._converter, "initialized_pipelines", None)
                    if isinstance(shared, dict):
                        converter.initialized_pipelines = shared
                    self._docling_parse_converter = converter
        return self._docling_parse_converter

    def _docling_parse_range(
        self,
        convert_path: str,
        first: int,
        last: int,
        file_path: str,
        fmt: str | None,
        src_hash: str,
    ) -> tuple[dict[int, list[Any]] | None, str | None]:
        """Elements per page of a docling_parse conversion of pages
        ``first..last``, or ``(None, error)``. The conversion's own figures go
        to a throwaway store: a rebuilt page keeps the original parse's."""
        try:
            result = self._get_docling_parse_converter().convert(
                convert_path, page_range=(first, last)
            )
            with tempfile.TemporaryDirectory(prefix="vorag_page_rebuild_") as tmp:
                rebuilt = docling_to_parsed_document(
                    result.document,
                    FigureStore(tmp),
                    file_path,
                    fmt,
                    src_hash=src_hash,
                    equation_figures=self._figures,
                )
        except Exception as exc:  # noqa: BLE001 - a failed rebuild falls back to plain text
            return None, f"{type(exc).__name__}: {exc}"
        by_page: dict[int, list[Any]] = defaultdict(list)
        for element in rebuilt.elements:
            if element.page_number is not None and first <= element.page_number <= last:
                by_page[element.page_number].append(element)
        failed = sorted(page for page in _failed_pages(result, last) if page >= first)
        return by_page, (f"docling_parse failed on pages {failed}" if failed else None)

    def _docling_parse_whole_document(
        self, convert_path: str, total_pages: int
    ) -> tuple[Any | None, str | None]:
        """The whole PDF through docling_parse, ``WHOLE_DOCUMENT_CHUNK_PAGES``
        pages per conversion, concatenated; or ``(None, error)``."""
        try:
            from docling_core.types.doc import (  # type: ignore[import-not-found, unused-ignore]
                DoclingDocument,
            )

            converter = self._get_docling_parse_converter()
            parts = [
                converter.convert(
                    convert_path,
                    page_range=(first, min(total_pages, first + WHOLE_DOCUMENT_CHUNK_PAGES - 1)),
                ).document
                for first in range(1, total_pages + 1, WHOLE_DOCUMENT_CHUNK_PAGES)
            ]
            return DoclingDocument.concatenate(parts), None
        except Exception as exc:  # noqa: BLE001 - the coverage failure then stands
            return None, f"{type(exc).__name__}: {exc}"

    async def parse_scanned(self, file_path: str, *, file_hash: str) -> ParsedDocument:
        """Full-page OCR on the GPU for scanned PDFs / synthesized EPUB PDFs.

        Uses a second, lazily built converter (``scanned_pipeline_options()``)
        so the normal ``do_ocr=False`` converter above is never touched by
        this path. ``_require_cuda()`` runs once, the first time this
        converter is built.
        """
        return await asyncio.to_thread(self._parse_scanned_sync, file_path, file_hash)

    def _parse_scanned_sync(self, file_path: str, file_hash: str) -> ParsedDocument:
        converter = self._get_scanned_converter()
        try:
            with _pdf_for_docling(file_path) as convert_path:
                result = converter.convert(convert_path)
            doc = docling_to_parsed_document(
                result.document,
                self._figures,
                file_path,
                "pdf",
                src_hash=file_hash,
                parser_used="docling-ocr",
            )
            return doc
        finally:
            _release_cuda_cache()

    def _get_scanned_converter(self) -> Any:
        # Double-checked locking: the fast path (converter already built)
        # never touches the lock; only the first call(s) contend for it, and
        # only one of them actually builds (and GPU-loads) the converter.
        if self._scanned_converter is None:
            with self._scanned_converter_lock:
                if self._scanned_converter is None:
                    _require_cuda()
                    from docling.datamodel.base_models import (  # type: ignore[import-not-found, unused-ignore]  # noqa: I001
                        InputFormat,
                    )
                    from docling.backend.pypdfium2_backend import (  # type: ignore[import-not-found, unused-ignore]
                        PyPdfiumDocumentBackend,
                    )
                    from docling.document_converter import (  # type: ignore[import-not-found, unused-ignore]
                        DocumentConverter,
                        PdfFormatOption,
                    )

                    # pypdfium, never docling_parse: a scanned page has no text
                    # layer to parse (full-page OCR replaces it), and
                    # docling_parse committed ~1.7 GiB per page transiently
                    # (37 GiB peak over 20 pages vs 5.3 GiB; identical output).
                    pdf_option = PdfFormatOption(
                        pipeline_options=scanned_pipeline_options(),
                        backend=PyPdfiumDocumentBackend,
                    )
                    self._scanned_converter = DocumentConverter(
                        format_options={InputFormat.PDF: pdf_option}
                    )
        return self._scanned_converter


def _release_cuda_cache() -> None:
    """Hand cached CUDA blocks back to the driver after a parse (VOR-1425).

    Dispatcher lanes are long-lived processes; PyTorch's caching allocator
    otherwise keeps every block it ever reserved, so after hours of books a
    lane held 4-7 GB of a 12 GB card and a second lane on that GPU hit CUDA
    OOM on every new book. Best-effort: never let cleanup fail a parse.

    No forced ``gc.collect()``: it cost 1-2.5 s per book on the lane's single
    core, and it runs while the parse result is still referenced, so it freed
    nothing ``empty_cache`` doesn't (VOR-1427).
    """
    try:
        import torch  # type: ignore[import-not-found, unused-ignore]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - cleanup must not mask the parse result
        return

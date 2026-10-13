"""Standalone docling GPU-OCR worker for the vo-rag OCR farm (VOR-1427).

Runs on the idle GPU hosts (win, lin), which have no vo-rag install and
no Postgres / Qdrant / Garage credentials. It imports only the stdlib,
docling, docling_core and torch. The coordinator on localhost ships a PDF
here over scp, runs::

    python ocr_worker.py <in.pdf> <out.json>

and copies ``out.json`` (a DoclingDocument with embedded pictures) back.

Exit codes (the coordinator's ``HOST_FAULT_EXITS`` depends on these):

- 0: ok; ``<out.json>`` written.
- 2: usage (wrong arguments).
- 4: GPU guard: the conversion finished but peak allocated GPU memory
  (``torch.cuda.max_memory_allocated(d)`` summed over every visible device)
  is 0, i.e. the OCR did not run on the GPU. No output is written. The
  coordinator treats this as a book failure.
- 5: docling reported the conversion as neither SUCCESS nor PARTIAL_SUCCESS.
  No output is written.
- 69 (``EX_UNAVAILABLE``): CUDA unavailable: ``torch.cuda.is_available()`` is
  False. Checked right after importing torch, before docling is imported or
  any model is loaded. ``--check`` also exits 69 in that case. The host's GPU
  is unusable, so the coordinator treats this as a host fault. Not 3: the
  Windows C runtime's ``abort()`` (e.g. a ``std::bad_alloc`` crash) exits 3,
  which once made an out-of-memory book look like a missing GPU and dropped
  win (2026-09-26).

The worker writes its PID to ``<out stem>.pid`` (``jobs/<sha>.pid``) first
and removes it in a ``finally`` on every normal exit (return or exception).
A hard kill (SIGKILL, an unhandled SIGTERM, a forced Windows process stop)
skips the ``finally``, so the file can outlive the worker and its PID can be
recycled. Two things cover that: the coordinator kills a PID only after
checking its command line is still ``ocr_worker.py`` for that sha, and the
coordinator's per-book cleanup deletes the ``.pid`` file with the others.

Books are converted in ``BATCH_PAGES``-page ranges and the batch documents
concatenated: docling holds every page of a single ``convert()`` in memory,
and on 2026-09-26 four 1,000-4,000-page books at once committed 30-40 GB per
process on win (32 GB) and died with PDFium's out-of-memory error
(``Windows Error 0xe0000008``). Each batch keeps only its document; pages
docling dropped are re-added empty (``add_page``) so every batch is contiguous
and ``DoclingDocument.concatenate`` keeps the original page numbers; the
merged document must have exactly pages 1..N or the book fails (exit 5).

``python ocr_worker.py --check`` prints ``torch <version> cuda <bool> <gpu>``
(used by ``vo-rag corpus ocr-farm bootstrap``; ``python -c`` with newlines
does not survive Windows ssh).

``scanned_pipeline_options()`` below is an inlined copy of
``vo_rag.core.parsing.parsers.docling.scanned_pipeline_options``;
``tests/ocr_farm/test_worker_options_match.py`` keeps the two identical.

Oversize pages (VOR-1434 row 11: a 13,400 x 9,100pt page failed on the farm)
are rescaled before docling opens the PDF, as vo-rag's own docling parser
does. ``_rescale_oversize_pages`` is a pypdfium2 port of
``vo_rag.core.parsing.parsers.docling.rescale_oversize_pages`` (PyMuPDF is not
installed here); ``tests/ocr_farm/test_worker_rescale_match.py`` pins the
thresholds and the resulting page sizes to that implementation.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

EXIT_USAGE = 2  # wrong arguments
GPU_MEM_FRACTION_ENV = "VORAG_OCR_GPU_MEM_FRACTION"  # hosts.GPU_MEM_FRACTION_ENV
EXIT_NO_CUDA = 69  # EX_UNAVAILABLE: torch.cuda.is_available() is False (host fault)
EXIT_GPU_GUARD = 4  # peak GPU memory 0 on every device after conversion (book failure)
EXIT_CONVERSION_FAILED = 5  # docling status neither SUCCESS nor PARTIAL_SUCCESS

#: Pages per docling ``convert()`` call; bounds the worker's memory (see above).
BATCH_PAGES = 100

#: Same values as ``vo_rag.core.parsing.parsers.docling`` (pinned by a test):
#: a page whose longest side exceeds OVERSIZE_PAGE_PT is redrawn with its
#: longest side at RESCALED_PAGE_MAX_PT.
OVERSIZE_PAGE_PT = 5_000.0
RESCALED_PAGE_MAX_PT = 1_200.0


def scanned_pipeline_options() -> Any:
    """OCR pipeline options for scanned PDFs (inlined from vo-rag, keep identical).

    RapidOCR on the ``torch`` backend, full-page OCR, pictures kept, and the
    accelerator always CUDA: ``VORAG_DOCLING_DEVICE=cuda:N`` selects a GPU,
    any other value (including ``cpu``) is ignored.
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


def _check(torch: Any) -> int:
    available = bool(torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if available else ""
    print(f"torch {torch.__version__} cuda {available} {name}".rstrip(), flush=True)
    return 0 if available else EXIT_NO_CUDA


def _page_sizes(pdf: Path) -> list[tuple[float, float]]:
    """(width, height) in points of every page, via pypdfium2 (a docling dependency)."""
    import pypdfium2  # type: ignore[import-not-found, unused-ignore]

    document = pypdfium2.PdfDocument(str(pdf))
    try:
        return [
            (float(width), float(height))
            for width, height in (document[i].get_size() for i in range(len(document)))
        ]
    finally:
        document.close()


def _rescale_oversize_pages(pdf: Path, workdir: Path) -> Path | None:
    """Copy of ``pdf`` in ``workdir`` (same name) with every page whose longest
    side exceeds ``OVERSIZE_PAGE_PT`` redrawn at ``RESCALED_PAGE_MAX_PT``; page
    count and order kept, other pages imported as-is, ``pdf`` never modified.

    ``page_as_xobject`` bakes in the page's rotation and MediaBox origin, so
    the new page takes the displayed size (``get_page_size``) unrotated.
    ``None`` when no page is oversize or pypdfium2 fails anywhere (docling
    then gets the original and reports its own error).
    """
    import pypdfium2  # type: ignore[import-not-found, unused-ignore]

    out = workdir / pdf.name
    src: Any = None
    dst: Any = None
    try:
        src = pypdfium2.PdfDocument(str(pdf))
        sizes = [src.get_page_size(i) for i in range(len(src))]
        oversize = [i for i, (w, h) in enumerate(sizes) if max(w, h) > OVERSIZE_PAGE_PT]
        if not oversize:
            return None
        dst = pypdfium2.PdfDocument.new()
        for index, (width, height) in enumerate(sizes):
            if index not in oversize:
                dst.import_pages(src, [index])
                continue
            scale = RESCALED_PAGE_MAX_PT / max(width, height)
            page = dst.new_page(width * scale, height * scale)
            xobject = src.page_as_xobject(index, dst)
            content = xobject.as_pageobject()
            content.transform(pypdfium2.PdfMatrix().scale(scale, scale))
            page.insert_obj(content)
            page.gen_content()
        dst.save(str(out))
    except Exception as exc:  # noqa: BLE001 - never replace docling's own error
        print(f"ocr_worker: oversize rescale skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
        if dst is not None:
            dst.close()
            dst = None
        out.unlink(missing_ok=True)
        return None
    finally:
        if dst is not None:
            dst.close()
        if src is not None:
            src.close()
    print(f"ocr_worker: rescaled oversize pages {[i + 1 for i in oversize]}", flush=True)
    return out


def _page_batches(page_count: int, size: int = BATCH_PAGES) -> list[tuple[int, int]]:
    """Inclusive 1-based page ranges covering ``1..page_count``."""
    return [(start, min(start + size - 1, page_count)) for start in range(1, page_count + 1, size)]


def _fill_missing_pages(
    document: Any, first: int, last: int, sizes: list[tuple[float, float]], size_cls: Any
) -> None:
    """Re-add pages docling dropped from ``first..last`` (no content, true size)."""
    for page_no in range(first, last + 1):
        if page_no not in document.pages:
            width, height = sizes[page_no - 1]
            document.add_page(page_no=page_no, size=size_cls(width=width, height=height))


def _convert(pdf: Path, out: Path, torch: Any) -> int:
    # Cleanup errors (a Windows file handle still open) must not fail a
    # finished book.
    with tempfile.TemporaryDirectory(
        prefix="vorag_ocr_rescale_", ignore_cleanup_errors=True
    ) as tmp:
        rescaled = _rescale_oversize_pages(pdf, Path(tmp))
        return _convert_pdf(rescaled or pdf, out, torch)


def _convert_pdf(pdf: Path, out: Path, torch: Any) -> int:
    import gc

    from docling.backend.pypdfium2_backend import (  # type: ignore[import-not-found, unused-ignore]
        PyPdfiumDocumentBackend,
    )
    from docling.datamodel.base_models import (  # type: ignore[import-not-found, unused-ignore]
        ConversionStatus,
        InputFormat,
    )
    from docling.document_converter import (  # type: ignore[import-not-found, unused-ignore]
        DocumentConverter,
        PdfFormatOption,
    )
    from docling_core.types.doc import (  # type: ignore[import-not-found, unused-ignore]
        DoclingDocument,
        ImageRefMode,
        Size,
    )

    # pypdfium, as in vo_rag's parse_scanned: docling's default docling_parse
    # backend committed ~1.7 GiB per scanned page transiently, which Windows
    # charges in full (win hit its commit limit on 2026-09-26).
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=scanned_pipeline_options(), backend=PyPdfiumDocumentBackend
            )
        }
    )
    sizes = _page_sizes(pdf)
    batches = _page_batches(len(sizes))
    documents: list[Any] = []
    statuses: list[Any] = []
    for first, last in batches:
        result = converter.convert(str(pdf), page_range=(first, last))
        if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
            print(
                f"ocr_worker: conversion failed: status={result.status} pages={first}-{last}",
                file=sys.stderr,
            )
            return EXIT_CONVERSION_FAILED
        _fill_missing_pages(result.document, first, last, sizes, Size)
        documents.append(result.document)
        statuses.append(result.status)
        # Drop the batch's page buffers before converting the next one.
        del result
        gc.collect()
        torch.cuda.empty_cache()
    document = documents[0] if len(documents) == 1 else DoclingDocument.concatenate(documents)
    document.name = documents[0].name
    if sorted(document.pages) != list(range(1, len(sizes) + 1)):
        print(f"ocr_worker: merged document pages are not 1..{len(sizes)}", file=sys.stderr)
        return EXIT_CONVERSION_FAILED
    # Sum over every visible device: max_memory_allocated() with no argument
    # reads only the current device, and VORAG_DOCLING_DEVICE=cuda:N may put
    # the models on another one.
    peak = sum(
        int(torch.cuda.max_memory_allocated(device))
        for device in range(int(torch.cuda.device_count()))
    )
    if peak == 0:
        print(
            "ocr_worker: GPU guard: max_memory_allocated() == 0 on every device after conversion",
            file=sys.stderr,
        )
        return EXIT_GPU_GUARD
    status = (
        ConversionStatus.SUCCESS
        if all(s == ConversionStatus.SUCCESS for s in statuses)
        else ConversionStatus.PARTIAL_SUCCESS
    )
    partial = out.with_name(out.name + ".part")
    document.save_as_json(partial, image_mode=ImageRefMode.EMBEDDED)
    os.replace(partial, out)
    print(
        f"ocr_worker: ok status={status} pages={len(sizes)} batches={len(batches)} "
        f"gpu_peak_bytes={peak}",
        flush=True,
    )
    return 0


def _cap_gpu_memory(torch: Any) -> bool:
    """Apply ``VORAG_OCR_GPU_MEM_FRACTION`` (0 < f <= 1) to every visible GPU.

    Set by the coordinator for hosts whose GPU has another tenant (lin keeps
    ~3.5 GB free for whisper). The caching allocator then frees cached blocks
    before exceeding the cap instead of growing past it. False on a bad value.
    """
    raw = os.environ.get(GPU_MEM_FRACTION_ENV, "").strip()
    if not raw:
        return True
    try:
        fraction = float(raw)
    except ValueError:
        fraction = -1.0
    if not 0 < fraction <= 1:
        print(f"ocr_worker: {GPU_MEM_FRACTION_ENV} must be in (0, 1], got {raw!r}", file=sys.stderr)
        return False
    for device in range(int(torch.cuda.device_count())):
        torch.cuda.set_per_process_memory_fraction(fraction, device)
    return True


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    check = args == ["--check"]
    if not check and len(args) != 2:
        print("usage: ocr_worker.py <in.pdf> <out.json> | --check", file=sys.stderr)
        return EXIT_USAGE

    if check:
        import torch  # type: ignore[import-not-found, unused-ignore]

        return _check(torch)

    # First thing, before the slow torch import: a timeout during that import
    # must still find the PID.
    out = Path(args[1])
    pid_file = out.with_name(out.stem + ".pid")
    pid_file.write_text(str(os.getpid()), encoding="utf-8")
    try:
        import torch  # type: ignore[import-not-found, unused-ignore]

        if not torch.cuda.is_available():
            print("ocr_worker: CUDA unavailable; OCR never runs on the CPU", file=sys.stderr)
            return EXIT_NO_CUDA
        if not _cap_gpu_memory(torch):
            return EXIT_USAGE
        return _convert(Path(args[0]), out, torch)
    finally:
        # Don't leave a PID behind on a normal exit: it could be recycled. (A
        # hard kill skips this; see the module docstring for what covers it.)
        pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

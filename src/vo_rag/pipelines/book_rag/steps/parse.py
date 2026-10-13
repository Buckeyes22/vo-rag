"""book_rag parse step — download, hash, parse, gate, persist.

Notes:

- docling (native or GPU-OCR) is the only parser path.
- Behavior flags are first-class ``BookRagConfig`` fields, not env flags.
- Visual-sidecar attachment is a separate backfill path.
- Wires ``vo_rag.core.parsing`` corruption / quality / router gates.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from vo_rag.core.observability.metrics import record_ingest_pages
from vo_rag.core.parsing.contract import PageQA, ParsedDocument
from vo_rag.core.parsing.corruption_gate import assess_text_corruption
from vo_rag.core.parsing.quality_gate import QualityGate
from vo_rag.core.parsing.router import ParserRouter
from vo_rag.core.storage.garage import GarageClient
from vo_rag.pipelines.book_rag.parse_provenance import accepted_parse_binding
from vo_rag.pipelines.book_rag.quality_warnings import quality_warning
from vo_rag.pipelines.book_rag.source_identity import canonical_source_id

logger = logging.getLogger(__name__)

# The doc that describes the current PDF forward path (native docling, docling
# GPU OCR, the OCR-farm cache). Stored on every chunk as forward_path_decision_doc.
FORWARD_PATH_DECISION_DOC = "docs/architecture.md"


# ---------------------------------------------------------------------------
# public step
# ---------------------------------------------------------------------------


async def do_parse_document(
    *,
    envelope_id: str,
    garage_key: str,
    source_id: str,
    garage: GarageClient,
    bucket: str,
    parser_router: ParserRouter,
) -> dict[str, Any]:
    """Download file from Garage, run ParserRouter, persist ParsedDocument JSON."""
    t0 = time.perf_counter()
    data = await garage.get_bytes(bucket, garage_key)
    file_hash = hashlib.sha256(data).hexdigest()
    if source_id.startswith("vo-rag-corpus-sha256:") and source_id != canonical_source_id(
        file_hash
    ):
        raise ValueError("Downloaded original bytes do not match the registered source identity")

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / Path(garage_key).name
        local.write_bytes(data)
        is_pdf = local.suffix.lower() == ".pdf" or data.startswith(b"%PDF")
        if is_pdf:
            # VOR-1434: a PDF's page count is known before the (possibly
            # hour-long) parse; give it to the ingest metrics now so the
            # over-budget alert uses the book's real budget during the parse.
            await _record_pdf_page_count(str(local))
        parsed = await parser_router.parse(str(local), file_hash)
        parsed.document_id = file_hash
        parsed.source_id = source_id
        if is_pdf:
            parsed = await _reconcile_pdf_page_count(parsed, str(local))
        # VOR-1434: the parsed page count (any format, both ingest
        # paths) for the over-budget alert and the canary's stage budget.
        record_ingest_pages(parsed.total_pages)
        parsed = _apply_parse_quality_gate(parsed)
        _enforce_content_corruption(parsed)
        # An image-only EPUB/MOBI is OCR'd through the same docling-ocr path
        # as a scanned PDF (VOR-1427), but the *original* upload is not a
        # PDF -- is_pdf stays False for it. The PDF fail-loud
        # gates (no-text / page-coverage / red-flag checks) must
        # still apply whenever the parse result itself is docling-ocr output,
        # whatever the original file type, or an empty OCR would be ingested
        # silently. _reconcile_pdf_page_count stays is_pdf-only: it opens
        # `local` itself with fitz, and for an image-only ebook `local` is
        # the original EPUB/MOBI, not a PDF.
        if is_pdf or parsed.parser_used == "docling-ocr":
            _enforce_pdf_forward_path(parsed)
            parsed.metadata = dict(parsed.metadata or {})
            parsed.metadata["book_rag_forward_path"] = _forward_path_metadata(parsed)
            _enforce_parse_red_flags(parsed)

    parsed_key = f"parsed/{envelope_id}/parsed.json"
    parsed.metadata["original_sha256"] = file_hash
    accepted_bytes = parsed.model_dump_json().encode("utf-8")
    binding = accepted_parse_binding(parsed, file_hash, accepted_bytes)
    await garage.put_bytes(bucket, parsed_key, accepted_bytes, "application/json")
    duration_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        (
            "book_rag.parse_document done envelope_id=%s parser=%s "
            "elements=%d images=%d pages=%d duration_ms=%d"
        ),
        envelope_id,
        parsed.parser_used,
        len(parsed.elements),
        len(parsed.images),
        parsed.total_pages,
        duration_ms,
    )
    return {
        "parsed_garage_key": parsed_key,
        "parse_provenance": binding,
        "document_id": parsed.document_id,
        "source_id": parsed.source_id or source_id,
        "title": parsed.title,
        "total_pages": parsed.total_pages,
        "image_count": len(parsed.images),
        "element_count": len(parsed.elements),
        "parser_used": parsed.parser_used,
        "parser_backend": parsed.metadata.get("parser_backend"),
        "forward_path": parsed.metadata.get("book_rag_forward_path", {}),
        "duration_ms": duration_ms,
        "quality_warnings": parsed.metadata.get("quality_warnings", []),
        "quarantined_page_count": len(parsed.quarantined_pages),
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parsed_text_available(parsed: ParsedDocument) -> bool:
    if (parsed.markdown or "").strip():
        return True
    return any((element.text or "").strip() for element in parsed.elements)


_DOCLING_OCR_STACK = "docling-rapidocr-torch-cuda"


def _ocr_stack(parsed: ParsedDocument, metadata: dict[str, Any]) -> str:
    stack = metadata.get("book_rag_forward_path_stack")
    if stack:
        return str(stack)
    if parsed.parser_used == "docling":
        return "docling-layout-pdf"
    return _DOCLING_OCR_STACK


def _forward_path_metadata(parsed: ParsedDocument) -> dict[str, Any]:
    metadata = dict(parsed.metadata or {})
    return {
        "decision_doc": FORWARD_PATH_DECISION_DOC,
        "ocr_stack": _ocr_stack(parsed, metadata),
        "parser_used": parsed.parser_used,
        "parser_backend": metadata.get("parser_backend"),
        "parse_element_count": len(parsed.elements),
        "parse_image_count": len(parsed.images),
        "total_pages": parsed.total_pages,
        "page_visual_sidecar_enabled": metadata.get("book_rag_page_visual_sidecar_enabled", False),
        "page_visual_count": metadata.get("book_rag_page_visual_count", 0),
        "page_visual_dpi": metadata.get("book_rag_page_visual_dpi"),
    }


def _enforce_pdf_forward_path(parsed: ParsedDocument) -> None:
    """Approved PDF paths: docling (layout-only, text-layer "native" PDFs) and
    docling-ocr (GPU OCR for scanned PDFs / synthesized image-only-EPUB PDFs,
    VOR-1427). Nothing else is accepted.
    """
    if parsed.parser_used not in ("docling", "docling-ocr"):
        raise RuntimeError(
            "book_rag PDF parse did not use an approved forward path "
            f"(parser_used={parsed.parser_used!r}); expected docling (native) or "
            "docling-ocr (scanned)"
        )
    if not _parsed_text_available(parsed):
        raise RuntimeError("book_rag docling PDF parse produced no text")
    total_pages = parsed.total_pages or len(parsed.page_qa) or 1
    pages_with_content = sum(
        1 for qa in parsed.page_qa if (qa.text_element_count or 0) > 0 or (qa.table_count or 0) > 0
    )
    if total_pages >= 10 and pages_with_content / total_pages < 0.10:
        _add_parse_warning(
            parsed,
            "low_page_coverage",
            f"Extracted content on {pages_with_content}/{total_pages} pages; "
            "extraction fidelity unverified",
        )


def _corruption_sample_text(parsed: ParsedDocument, *, max_chars: int = 20000) -> str:
    """Representative text sample for the content-corruption gate."""
    if (parsed.markdown or "").strip():
        return parsed.markdown[:max_chars]
    parts: list[str] = []
    total = 0
    for element in parsed.elements:
        text = (element.text or "").strip()
        if not text:
            continue
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return "\n".join(parts)


def _enforce_content_corruption(parsed: ParsedDocument) -> None:
    """Fail-closed on CID glyph-soup / control-char corruption.

    The unambiguous CID signatures (control-char ratio, caesar-shift recovery)
    can never be legitimate book text and would poison embeddings — hard-fail
    on those before any paid downstream work.

    Operator kill-switch: set
    ``VORAG_BOOK_RAG_DISABLE_CORRUPTION_GATE=1`` to bypass for a known-good doc
    the heuristics misjudge. Default fail-closed (gate ON).
    """
    if _env_enabled("VORAG_BOOK_RAG_DISABLE_CORRUPTION_GATE", default="0"):
        return
    sample = _corruption_sample_text(parsed)
    if not sample.strip():
        return
    verdict = assess_text_corruption(sample)
    if not verdict.is_corrupt:
        return
    if verdict.reason in {"control_char", "cid_caesar_shift"}:
        raise RuntimeError(
            "book_rag content-corruption gate: "
            f"reason={verdict.reason} control_ratio={verdict.control_ratio:.3f} "
            f"dict_hit_ratio={verdict.dict_hit_ratio:.3f} "
            f"(document_id={parsed.document_id}, title={parsed.title!r}); "
            "quarantining before downstream spend"
        )
    logger.warning(
        (
            "book_rag.content_corruption_warning document_id=%s title=%s "
            "reason=%s dict_hit_ratio=%.3f space_ratio=%.3f"
        ),
        parsed.document_id,
        parsed.title,
        verdict.reason,
        verdict.dict_hit_ratio,
        verdict.space_ratio,
    )


def _env_enabled(name: str, *, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _enforce_parse_red_flags(parsed: ParsedDocument) -> None:
    """Record native-PDF quality concerns without blocking usable text.

    The historical flag controls whether this audit runs; enabling it now
    records warnings. Parser identity, no-text, and corruption guards remain
    separate from image-count heuristics.
    """
    if not _env_enabled("VORAG_BOOK_RAG_FAIL_ON_PARSE_RED_FLAGS"):
        return

    metadata = parsed.metadata or {}
    red_flags = list(metadata.get("native_pdf_image_red_flags") or [])
    legacy_warning = metadata.get("native_pdf_image_count_warning")
    if legacy_warning and legacy_warning not in red_flags:
        red_flags.append(str(legacy_warning))

    if not red_flags:
        return

    _add_parse_warning(parsed, "native_pdf_red_flags", ", ".join(sorted(map(str, red_flags))))


def _add_parse_warning(parsed: ParsedDocument, code: str, details: str) -> None:
    """Retain quality evidence while keeping usable accepted passages publishable."""
    parsed.metadata = dict(parsed.metadata or {})
    warnings = list(parsed.metadata.get("quality_warnings", []))
    warning = quality_warning("parse", code, details)
    if warning not in warnings:
        warnings.append(warning)
    parsed.metadata["quality_warnings"] = warnings


def _apply_parse_quality_gate(parsed: ParsedDocument) -> ParsedDocument:
    parsed = QualityGate().evaluate_document(parsed)
    quarantined = sorted(set(parsed.quarantined_pages))
    parsed.quarantined_pages = quarantined
    warning_count = sum(len(qa.warnings) for qa in parsed.page_qa)
    parsed.metadata = dict(parsed.metadata or {})
    parsed.metadata["parse_quality_gate"] = {
        "quarantined_pages": quarantined,
        "quarantined_page_count": len(quarantined),
        "page_warning_count": warning_count,
    }
    if quarantined or warning_count:
        _add_parse_warning(
            parsed,
            "page_quality_findings",
            f"{len(quarantined)} pages isolated; {warning_count} page findings; "
            "usable pages retained",
        )
    return parsed


# ---------------------------------------------------------------------------
# PDF-only helpers
# ---------------------------------------------------------------------------


async def _record_pdf_page_count(pdf_path: str) -> None:
    """Best-effort: an unreadable PDF just leaves the page count unknown."""

    def _count() -> int:
        fitz = importlib.import_module("fitz")
        with fitz.open(pdf_path) as pdf:
            return int(pdf.page_count)

    try:
        record_ingest_pages(await asyncio.to_thread(_count))
    except Exception:  # noqa: BLE001 - metrics must never fail a parse
        logger.debug("book_rag.parse_document page count unavailable", exc_info=True)


async def _reconcile_pdf_page_count(parsed: ParsedDocument, pdf_path: str) -> ParsedDocument:
    """Correct parser-reported page counts against the PDF container."""

    def _count() -> int:
        fitz = importlib.import_module("fitz")
        with fitz.open(pdf_path) as pdf:
            return int(pdf.page_count)

    try:
        page_count = await asyncio.to_thread(_count)
    except Exception as exc:  # noqa: BLE001
        parsed.metadata = dict(parsed.metadata or {})
        parsed.metadata["pdf_page_count_reconcile_error"] = f"{type(exc).__name__}: {exc}"
        return parsed

    if page_count <= 0:
        return parsed

    if parsed.total_pages != page_count:
        parsed.metadata = dict(parsed.metadata or {})
        parsed.metadata["parser_reported_total_pages"] = parsed.total_pages
        parsed.metadata["pdf_total_pages_verified"] = page_count
        parsed.total_pages = page_count

    qa_by_page = {qa.page_number: qa for qa in parsed.page_qa if 1 <= qa.page_number <= page_count}
    parsed.page_qa = [
        qa_by_page.get(page_number)
        or PageQA(
            page_number=page_number,
            ocr_confidence=0.0,
            parser_used=parsed.parser_used,
            warnings=["synthesized_after_page_count_reconciliation"],
        )
        for page_number in range(1, page_count + 1)
    ]
    return parsed

"""Retain original PDF equation and table pixels alongside supplementary parsed characters."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
from typing import Any, Literal, cast

from vo_rag.core.parsing.contract import ElementType, ParsedDocument
from vo_rag.core.storage.figures import SRC_RE, FigureStore

MAX_CROP_BYTES = 16 * 1024 * 1024
MAX_CROP_PIXELS = 4_194_304
CROP_SCALE = 1.67
CROP_EXPANSION = 0.18
MAX_SOURCE_CROP_BYTES = 64 * 1024 * 1024
MAX_SOURCE_CROPS = 4096


class PdfEquationCrops:
    """Render source-bound formula and table crops without performing recognition.

    One verified PDF stays open for the mapping pass. Invalid or unavailable
    geometry has an explicit disposition; source text remains available.
    """

    def __init__(
        self,
        source: Path,
        digest: str,
        figures: FigureStore,
        *,
        max_bytes: int = MAX_SOURCE_CROP_BYTES,
    ) -> None:
        import fitz  # type: ignore[import-untyped, unused-ignore]

        if not 0 <= max_bytes <= MAX_SOURCE_CROP_BYTES:
            raise ValueError("invalid finite equation crop byte budget")
        self._remaining_bytes = max_bytes
        self._occurrences = 0
        actual = hashlib.sha256()
        with source.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                actual.update(block)
        if actual.hexdigest() != digest:
            raise ValueError("equation crop original PDF identity differs")
        self._pdf = fitz.open(source)
        self._digest = digest
        self._figures = figures
        self._mathematical_pages: dict[int, bool] = {}

    def close(self) -> None:
        """Close the verified original PDF."""
        self._pdf.close()

    def has_mathematical_notation(self, page_number: int | None) -> bool:
        """Select original notation even when the native parser missed a formula.

        Read literal mathematical glyphs and PDF superscript font flags. This
        selects supplementary pixels; it does not interpret the relationship
        or manufacture a formula node. Ordinary prose alone selects no image.
        """
        if not isinstance(page_number, int) or not 1 <= page_number <= len(self._pdf):
            return False
        if page_number not in self._mathematical_pages:
            page = self._pdf[page_number - 1]
            # Exclude image bytes from the PDF text dictionary. Stop the
            # Python scan at a finite span count; renders retain their limits.
            data = page.get_text("dict", flags=0)
            selected = False
            inspected = 0
            for block in data.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        inspected += 1
                        if inspected > 100_000:
                            self._mathematical_pages[page_number] = False
                            return False
                        text = str(span.get("text", ""))
                        if any(c in text for c in "=+<>%/⁄π√±−×÷°′″∑∫²³") or (
                            span.get("flags", 0) & 1 and text.strip()
                        ):
                            selected = True
                            break
                    if selected:
                        break
                if selected:
                    break
            self._mathematical_pages[page_number] = selected
        return self._mathematical_pages[page_number]

    def crop(
        self,
        provenance: dict[str, Any],
        native_size: tuple[object, object] | None,
        *,
        kind: Literal["equation", "table", "page"] = "equation",
    ) -> dict[str, object]:
        """Return the exact rendered crop and its finite provenance receipt."""
        import fitz  # type: ignore[import-untyped, unused-ignore]

        unavailable: dict[str, object] = {"status": "unavailable"}
        self._occurrences += 1
        if self._occurrences > MAX_SOURCE_CROPS or self._remaining_bytes <= 0:
            return {**unavailable, "reason": "finite_source_crop_budget_exceeded"}
        occurrences = provenance.get("occurrences")
        if (
            provenance.get("geometry_status") != "available"
            or provenance.get("total_occurrence_count") != 1
            or not isinstance(occurrences, list)
            or len(occurrences) != 1
        ):
            return {**unavailable, "reason": "single_original_region_required"}
        location = occurrences[0]
        page_number = location.get("page_no", location.get("page_number"))
        if "page_idx" in location:
            page_number = location["page_idx"] + 1
        if not isinstance(page_number, int) or not 1 <= page_number <= len(self._pdf):
            return {**unavailable, "reason": "physical_page_unavailable"}
        box = location.get("bbox")
        if not isinstance(box, dict):
            return {**unavailable, "reason": "bbox_unavailable"}
        values = [box.get(axis) for axis in ("l", "t", "r", "b")]
        if any(not isinstance(value, int | float) or not math.isfinite(value) for value in values):
            return {**unavailable, "reason": "bbox_invalid"}
        left, top, right, bottom = map(float, cast(list[int | float], values))
        page = self._pdf[page_number - 1]
        if page.rotation != 0 and kind == "equation":
            return {**unavailable, "reason": "rotated_parser_geometry_unverified"}
        if native_size is None or any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in native_size
        ):
            return {**unavailable, "reason": "native_page_frame_unavailable"}
        if tuple(map(float, cast(tuple[int | float, int | float], native_size))) != (
            float(page.rect.width),
            float(page.rect.height),
        ):
            return {**unavailable, "reason": "native_page_frame_differs"}
        if box.get("coord_origin") == "BOTTOMLEFT":
            top, bottom = float(page.rect.height) - top, float(page.rect.height) - bottom
        elif box.get("coord_origin") != "TOPLEFT":
            return {**unavailable, "reason": "coordinate_origin_unknown"}
        if left >= right or top >= bottom:
            return {**unavailable, "reason": "bbox_empty_or_reversed"}
        region = fitz.Rect(left, top, right, bottom)
        margin_x = max(4.0, region.width * CROP_EXPANSION / 2)
        margin_y = max(4.0, region.height * CROP_EXPANSION / 2)
        expanded = fitz.Rect(left - margin_x, top - margin_y, right + margin_x, bottom + margin_y)
        clipped = expanded & page.rect
        if clipped.is_empty or clipped.width * clipped.height * CROP_SCALE**2 > MAX_CROP_PIXELS:
            return {**unavailable, "reason": "finite_crop_area_exceeded"}
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(CROP_SCALE, CROP_SCALE), clip=clipped, alpha=False
        )
        if pixmap.width * pixmap.height > MAX_CROP_PIXELS:
            return {**unavailable, "reason": "finite_crop_pixels_exceeded"}
        raw = pixmap.tobytes("png")
        if len(raw) > min(MAX_CROP_BYTES, self._remaining_bytes):
            return {**unavailable, "reason": "finite_crop_bytes_exceeded"}
        digest = hashlib.sha256(raw).hexdigest()
        writer = {
            "table": self._figures.write_table,
            "page": self._figures.write_page,
            "equation": self._figures.write_equation,
        }[kind]
        ref = writer(self._digest, page_number, raw)
        self._remaining_bytes -= len(raw)
        return {
            "status": "source_bound",
            "original_sha256": self._digest,
            "physical_page": page_number,
            "original_bbox": box,
            "bbox_space": "docling_parser_page",
            "clip_space": "PyMuPDF_page_rect",
            "native_page_size": list(native_size),
            "page_rect": list(page.rect),
            "media_box": list(page.mediabox),
            "crop_box": list(page.cropbox),
            "rotation": page.rotation,
            "clip_bbox_top_left": list(clipped),
            "image_ref": ref,
            "sha256": digest,
            "size_bytes": len(raw),
            "pixel_width": pixmap.width,
            "pixel_height": pixmap.height,
            "renderer": "PyMuPDF",
            "renderer_version": importlib.metadata.version("PyMuPDF"),
            "scale": CROP_SCALE,
            "expansion": CROP_EXPANSION,
            "minimum_margin_points": 4.0,
            "symbolic_transcription_verified": False,
        }

    def page(
        self, page_number: int | None, native_size: tuple[object, object] | None
    ) -> dict[str, object]:
        """Retain a whole mathematical page for formulas embedded in prose.

        The physical-page frame comes from the genuine converter and must
        equal the original displayed page. No text or formula is inferred.
        """
        if native_size is None or any(
            not isinstance(v, int | float) or isinstance(v, bool) or not math.isfinite(v)
            for v in native_size
        ):
            return {"status": "unavailable", "reason": "native_page_frame_unavailable"}
        width, height = native_size
        value = self.crop(
            {
                "geometry_status": "available",
                "total_occurrence_count": 1,
                "occurrences": [
                    {
                        "page_no": page_number,
                        "bbox": {
                            "l": 0.0,
                            "t": 0.0,
                            "r": width,
                            "b": height,
                            "coord_origin": "TOPLEFT",
                        },
                    }
                ],
            },
            native_size,
            kind="page",
        )
        if value.get("status") == "source_bound":
            value["bbox_space"] = "original_native_display_page"
            value["representation"] = "original_mathematical_page"
        return value


def verify_equation_crops(
    parsed: ParsedDocument, figures: FigureStore, max_total_bytes: int
) -> dict[str, int]:
    """Verify declared mathematical page, equation and table source crops.

    Source text without a declared crop is not upgraded to visual fidelity.
    Limits are checked before reading any image body, including repeated refs.
    """
    declarations: dict[str, tuple[int, str]] = {}
    occurrences = 0
    for element in parsed.elements:
        visuals = [("pg", element.metadata.get("source_page_visual"))]
        if element.element_type is ElementType.EQUATION:
            visuals.append(("eq", element.metadata.get("equation_visual")))
        elif element.element_type is ElementType.TABLE:
            visuals.append(("tbl", element.metadata.get("table_visual")))
        for namespace, visual in visuals:
            if not isinstance(visual, dict) or visual.get("status") != "source_bound":
                continue
            occurrences += 1
            if occurrences > MAX_SOURCE_CROPS:
                raise ValueError("equation_crop_source_count_exceeded")
            digest, size, page = (
                visual.get(key) for key in ("sha256", "size_bytes", "physical_page")
            )
            if (
                not isinstance(digest, str)
                or SRC_RE.fullmatch(digest) is None
                or not isinstance(size, int)
                or isinstance(size, bool)
                or not 0 < size <= MAX_CROP_BYTES
                or page != element.page_number
                or not isinstance(page, int)
                or isinstance(page, bool)
                or visual.get("original_sha256") != parsed.document_id
            ):
                raise ValueError("equation_crop_identity_invalid")
            ref = f"{parsed.document_id}/p{page:03d}_{namespace}{digest}.png"
            if visual.get("image_ref") != ref:
                raise ValueError("equation_crop_reference_invalid")
            if ref in declarations and declarations[ref] != (size, digest):
                raise ValueError("equation_crop_declaration_conflict")
            declarations[ref] = (size, digest)
    total = sum(size for size, _ in declarations.values())
    if total > min(max_total_bytes, MAX_SOURCE_CROP_BYTES):
        raise ValueError("equation_crop_aggregate_budget_exceeded")
    for ref, (size, digest) in declarations.items():
        with figures.path_for(ref).open("rb") as stream:
            if os.fstat(stream.fileno()).st_size != size:
                raise ValueError("equation_crop_size_changed")
            raw = stream.read(size + 1)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("equation_crop_bytes_changed")
    return {"occurrences": occurrences, "unique_images": len(declarations), "bytes": total}


__all__ = [
    "CROP_EXPANSION",
    "CROP_SCALE",
    "MAX_CROP_BYTES",
    "MAX_CROP_PIXELS",
    "MAX_SOURCE_CROP_BYTES",
    "MAX_SOURCE_CROPS",
    "PdfEquationCrops",
    "verify_equation_crops",
]

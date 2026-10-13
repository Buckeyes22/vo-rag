"""Map docling ``Document`` to the vo-rag ``ParsedDocument`` contract.

Docling is an in-process parser (not a worker), so the heavy
``DocumentConverter`` import is kept lazy inside ``parsers/docling.py``.
This module is the pure mapping: walk docling's ``texts`` / ``tables`` /
``pictures`` merged in document reading order, infer heading depth, write
figure images to the FigureStore.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

from vo_rag.core.parsing.contract import (
    ElementType,
    ExtractedImage,
    PageQA,
    ParsedDocument,
    ParsedElement,
    element_id,
    image_id,
)
from vo_rag.core.parsing.image_filters import (
    ImageCandidate,
    average_hash,
    filter_image_candidates,
    image_dimensions,
)
from vo_rag.core.parsing.parsers.pdf_formula import MAX_SOURCE_CROP_BYTES, PdfEquationCrops
from vo_rag.core.storage.figures import MIME_BY_EXT, FigureStore, safe_ext

_MAX_TABLE_PROVENANCE_OCCURRENCES = 16
_MAX_TABLE_PROVENANCE_INSPECTED = 256
_MAX_TABLE_SELF_REF_CHARS = 256
_MAX_TABLE_COORD_ABS = 1_000_000_000
_MAX_TABLE_ORIGIN_CHARS = 64


def docling_to_parsed_document(
    doc: object,
    figures: FigureStore,
    source_path: str,
    fmt: str | None,
    *,
    src_hash: str,
    parser_used: str = "docling",
    equation_figures: FigureStore | None = None,
    equation_crop_max_bytes: int = MAX_SOURCE_CROP_BYTES,
) -> ParsedDocument:
    native_layout = any(True for _ in _iter_values(getattr(doc, "tables", []))) or any(
        _label_to_element_type(_normalize_label(_get_attr(item, "label", "type", "kind", "name")))
        is ElementType.EQUATION
        for item in _iter_values(getattr(doc, "texts", []))
    )
    crops = (
        PdfEquationCrops(
            Path(source_path),
            src_hash,
            equation_figures or figures,
            max_bytes=equation_crop_max_bytes,
        )
        if (fmt or Path(source_path).suffix.lstrip(".")).lower() == "pdf"
        and Path(source_path).is_file()
        and (
            native_layout
            or (
                parser_used == "docling"
                and any(
                    _native_page_size(doc, _extract_page_number(item)) is not None
                    for item in _iter_values(getattr(doc, "texts", []))
                )
            )
        )
        else None
    )
    try:
        return _map_document(
            doc,
            figures,
            source_path,
            fmt,
            src_hash=src_hash,
            parser_used=parser_used,
            equation_crops=crops,
        )
    finally:
        if crops is not None:
            crops.close()


def _map_document(
    doc: object,
    figures: FigureStore,
    source_path: str,
    fmt: str | None,
    *,
    src_hash: str,
    parser_used: str,
    equation_crops: PdfEquationCrops | None,
) -> ParsedDocument:
    elements: list[ParsedElement] = []
    images: list[ExtractedImage] = []

    heading_stack: list[str] = []
    heading_nodes: list[tuple[int, str]] = []
    table_count_by_page: dict[int, int] = {}
    image_count_by_page: dict[int, int] = {}
    text_count_by_page: dict[int, int] = {}
    mathematical_pages: set[int | None] = set()

    texts = _iter_values(getattr(doc, "texts", []))
    tables = _iter_values(getattr(doc, "tables", []))
    pictures = _iter_values(getattr(doc, "pictures", []))
    native_mathematical_pages = {
        _extract_page_number(item)
        for item in texts
        if _label_to_element_type(
            _normalize_label(_get_attr(item, "label", "type", "kind", "name"))
        )
        is ElementType.EQUATION
    }

    def source_page_metadata(page: int | None) -> dict[str, object]:
        if page in mathematical_pages:
            return {}
        if page not in native_mathematical_pages and not (
            equation_crops is not None and equation_crops.has_mathematical_notation(page)
        ):
            return {}
        mathematical_pages.add(page)
        return {
            "source_page_visual": (
                equation_crops.page(page, _native_page_size(doc, page))
                if equation_crops is not None
                else {"status": "unavailable", "reason": "original_pdf_unavailable"}
            )
        }

    # Two-pass image handling: collect candidates, cull the frivolous ones
    # (tiny / duplicate / near-full-page — the image_filters.py rules,
    # VOR-304), then write + emit only the survivors so culled bytes never reach
    # the figure store or the VLM describe pass. ``image_index`` stays docling's
    # picture index so FigureStore keys do not depend on reading order.
    pending: dict[int, tuple[ImageCandidate, bytes, int, str]] = {}
    for image_index, picture_obj in enumerate(pictures):
        img_bytes = _extract_image_bytes(picture_obj)
        if not img_bytes:
            continue
        pic_page = _extract_page_number(picture_obj) or 1
        raw_ext, raw_mime = _image_ext_and_mime(picture_obj)
        ext = safe_ext(raw_ext, raw_mime)
        if ext is None:
            # Unsupported image type (svg/tiff/...) — skip; the bytes stay
            # regenerable from the source document.
            continue
        width, height = image_dimensions(img_bytes) or (0, 0)
        candidate = ImageCandidate(
            image_id=image_id(img_bytes),
            width=width,
            height=height,
            page_number=pic_page,
            average_hash=average_hash(img_bytes),
        )
        pending[image_index] = (candidate, img_bytes, pic_page, ext)

    kept_candidates, image_skip_reasons = filter_image_candidates(
        [item[0] for item in pending.values()]
    )
    kept_ids = {id(candidate) for candidate in kept_candidates}

    # VOR-1417: walk texts, tables and pictures in document reading order so
    # every table / picture sits under the heading it belongs to (previously
    # all tables and pictures were appended after the last text item).
    idx = 0
    for kind, position, item_obj in _reading_order(doc, texts, tables, pictures):
        if kind == _PICTURE:
            entry = pending.get(position)
            if entry is None or id(entry[0]) not in kept_ids:
                continue
            candidate, img_bytes, img_page, ext = entry
            img_id = candidate.image_id
            ref = figures.write(src_hash, img_page, position, img_bytes, ext=ext)
            mime_type = MIME_BY_EXT[ext]

            images.append(
                ExtractedImage(
                    image_id=img_id,
                    page_number=img_page,
                    image_index=position,
                    ref=ref,
                    file_size=len(img_bytes),
                    mime_type=mime_type,
                    element_index=idx,
                )
            )
            elements.append(
                ParsedElement(
                    element_id=element_id(img_id, img_page, idx),
                    element_type=ElementType.IMAGE,
                    text="",
                    page_number=img_page,
                    section_depth=len(heading_stack),
                    section_title=heading_stack[-1] if heading_stack else "",
                    metadata={
                        "image_id": img_id,
                        "image_ref": ref,
                        "mime_type": mime_type,
                    },
                )
            )
            image_count_by_page[img_page] = image_count_by_page.get(img_page, 0) + 1
            idx += 1
            continue

        if kind == _TABLE:
            page_number = _extract_page_number(item_obj)
            table_md = _as_str(
                _call_or_value(
                    item_obj, "export_to_markdown", "to_markdown", "markdown", "md", "text"
                )
            )
            table_html = _as_str(
                _call_or_value(item_obj, "export_to_html", "to_html", "html", "html_text")
            )
            text = table_md or table_html
            if not text:
                continue
            table_label = _normalize_label(_get_attr(item_obj, "label", "type", "kind", "name"))
            table_visual = (
                equation_crops.crop(
                    _table_provenance_metadata(item_obj),
                    _native_page_size(doc, page_number),
                    kind="table",
                )
                if equation_crops is not None
                else {"status": "unavailable", "reason": "original_pdf_unavailable"}
            )
            elements.append(
                ParsedElement(
                    element_id=element_id(text, page_number, idx),
                    element_type=ElementType.TABLE,
                    text=text,
                    page_number=page_number,
                    section_depth=len(heading_stack),
                    section_title=heading_stack[-1] if heading_stack else "",
                    metadata={
                        **source_page_metadata(page_number),
                        "markdown": table_md,
                        "html": table_html,
                        "docling_provenance": _table_provenance_metadata(item_obj),
                        **(
                            {"table_visual": table_visual}
                            if (fmt or Path(source_path).suffix.lstrip(".")).lower() == "pdf"
                            else {}
                        ),
                        # "document_index": a table of contents (VOR-1452); the
                        # chunker files its page as contents (contents_pages).
                        **({"docling_label": table_label} if table_label else {}),
                    },
                )
            )
            if page_number is not None:
                table_count_by_page[page_number] = table_count_by_page.get(page_number, 0) + 1
            idx += 1
            continue

        # Formula enrichment leaves ``text`` empty when disabled, but the
        # original PDF characters remain in ``orig``. Select a nonempty field
        # so detected formulas and other source text do not silently vanish.
        text = next(
            (
                value
                for name in ("text", "orig", "content", "value")
                if (value := _as_str(_get_attr(item_obj, name)))
            ),
            "",
        )
        if not text:
            continue

        page_number = _extract_page_number(item_obj)
        label = _normalize_label(_get_attr(item_obj, "label", "type", "kind", "name"))
        list_metadata: dict[str, object] = {}
        if _label_to_element_type(label) is ElementType.LIST_ITEM:
            # Docling removes genuine list markers from ``text`` and stores
            # them separately. Keep question/step numbers without inferring
            # a sequence or duplicating markers already in the text.
            marker_value = _get_attr(item_obj, "marker")
            marker = marker_value.strip() if isinstance(marker_value, str) else ""
            if marker and len(marker) <= 64 and "\n" not in marker and "\r" not in marker:
                suffix = text[len(marker) :] if text.startswith(marker) else None
                closing_marker = marker.endswith((")", "]", "}")) or marker in (
                    "•",
                    "‣",
                    "◦",
                    "▪",
                    "●",
                    "○",
                )
                original_prefix = bool(
                    suffix is not None and text == _as_str(_get_attr(item_obj, "orig"))
                )
                period_prefix = bool(suffix and marker.endswith(".") and suffix[0].isalpha())
                if suffix is None or (
                    suffix
                    and not suffix[0].isspace()
                    and not closing_marker
                    and not original_prefix
                    and not period_prefix
                ):
                    text = marker + " " + text
                list_metadata["docling_list_marker"] = marker
            enumerated = _get_attr(item_obj, "enumerated")
            if isinstance(enumerated, bool):
                list_metadata["docling_list_enumerated"] = enumerated
        level = _heading_level(item_obj)

        if _is_heading(label) and not _is_caption_text(text):
            level = _infer_heading_level(text, level)
            if level < 1:
                level = 1
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(text)
            heading_nodes.append((level, text))
            element_type = ElementType.TITLE
            section_depth = level
            section_title = text
        else:
            element_type = (
                ElementType.FIGURE_CAPTION
                if (_is_heading(label) and _is_caption_text(text))
                else _label_to_element_type(label)
            )
            section_depth = len(heading_stack)
            section_title = heading_stack[-1] if heading_stack else ""

        formula_metadata: dict[str, object] = {}
        if element_type is ElementType.EQUATION:
            provenance = _table_provenance_metadata(item_obj)
            native_size = _native_page_size(doc, page_number)
            formula_metadata = {
                "docling_orig": _as_str(_get_attr(item_obj, "orig")),
                "docling_provenance": provenance,
                "equation_visual": (
                    equation_crops.crop(provenance, native_size)
                    if equation_crops is not None
                    else {"status": "unavailable", "reason": "original_pdf_unavailable"}
                ),
            }
        elements.append(
            ParsedElement(
                element_id=element_id(text, page_number, idx),
                element_type=element_type,
                text=text,
                page_number=page_number,
                section_depth=section_depth,
                section_title=section_title,
                metadata={
                    **source_page_metadata(page_number),
                    **({"docling_label": label} if label else {}),
                    **list_metadata,
                    **formula_metadata,
                },
            )
        )
        if page_number is not None:
            text_count_by_page[page_number] = text_count_by_page.get(page_number, 0) + 1
        idx += 1

    pages = _sorted_pages(doc, elements, images)
    page_qa = [
        PageQA(
            page_number=page,
            ocr_confidence=1.0,
            table_count=table_count_by_page.get(page, 0),
            figure_count=image_count_by_page.get(page, 0),
            text_element_count=text_count_by_page.get(page, 0),
            parser_used=parser_used,
        )
        for page in pages
    ]

    markdown = _document_markdown(doc, elements)
    title = _as_str(_get_attr(doc, "title", "name")) or Path(source_path).stem

    return ParsedDocument(
        document_id=src_hash,
        source_id=source_path,
        title=title,
        total_pages=max(pages) if pages else 1,
        elements=elements,
        images=images,
        page_qa=page_qa,
        markdown=markdown,
        heading_tree=_build_heading_tree(heading_nodes),
        metadata={
            "format": fmt or Path(source_path).suffix.lstrip("."),
            "image_filter_skipped": dict(image_skip_reasons),
        },
        parser_used=parser_used,
    )


def _native_page_size(doc: object, page_number: int | None) -> tuple[object, object] | None:
    """Return the actual native physical-page frame for crop verification."""
    pages = _get_attr(doc, "pages")
    page = pages.get(page_number, pages.get(str(page_number))) if isinstance(pages, dict) else None
    size = _get_attr(page, "size")
    return (_get_attr(size, "width"), _get_attr(size, "height")) if size else None


_TEXT = 0
_TABLE = 1
_PICTURE = 2


def _reading_order(
    doc: object,
    texts: list[object],
    tables: list[object],
    pictures: list[object],
) -> list[tuple[int, int, object]]:
    """Return ``(kind, position_in_its_list, item)`` in document reading order.

    Ordering source, in preference order:

    1. docling's body tree (``doc.iterate_items()``), the canonical reading
       order of a ``DoclingDocument``. Every content layer and picture children
       are requested so furniture texts and in-figure texts get a rank too.
    2. page + bbox (top-to-bottom) when no tree is available or no item
       carries a ``self_ref``.

    The three lists are *merged*, never re-sorted: each list keeps its own
    order (docling's ``texts`` order is already reading order), and an item
    the ordering source cannot place inherits its predecessor's key so it
    stays right after it.
    """

    sequences = (texts, tables, pictures)
    rank = _tree_rank(doc, sequences)
    keyed: list[list[tuple[tuple[float, ...], object]]] = []
    for items in sequences:
        keys: list[tuple[tuple[float, ...], object]] = []
        carry: tuple[float, ...] = (-1.0, 0.0, 0.0)
        for item in items:
            key: tuple[float, ...] | None
            if rank is not None:
                r = rank.get(_as_str(_get_attr(item, "self_ref")))
                key = None if r is None else (float(r), 0.0, 0.0)
            else:
                key = _page_bbox_key(item)
            if key is None:
                # Unplaceable: sort right after the predecessor in its own list.
                key = (carry[0], carry[1], carry[2] + 1.0)
            keys.append((key, item))
            carry = key
        keyed.append(keys)

    merged: list[tuple[int, int, object]] = []
    heads = [0, 0, 0]
    while True:
        best: int | None = None
        for kind in (_TEXT, _TABLE, _PICTURE):
            if heads[kind] >= len(keyed[kind]):
                continue
            if best is None or keyed[kind][heads[kind]][0] < keyed[best][heads[best]][0]:
                best = kind
        if best is None:
            return merged
        position = heads[best]
        merged.append((best, position, keyed[best][position][1]))
        heads[best] += 1


def _tree_rank(doc: object, sequences: tuple[list[object], ...]) -> dict[str, int] | None:
    iterate = _get_attr(doc, "iterate_items")
    if not callable(iterate):
        return None
    if not any(_as_str(_get_attr(item, "self_ref")) for items in sequences for item in items):
        return None
    layers = {
        layer
        for items in sequences
        for item in items
        if (layer := _get_attr(item, "content_layer")) is not None
    }
    kwargs: dict[str, object] = {"traverse_pictures": True}
    if layers:
        kwargs["included_content_layers"] = layers
    try:
        walked = list(iterate(**kwargs))
    except TypeError:
        try:
            walked = list(iterate())
        except TypeError:
            return None
    rank: dict[str, int] = {}
    for position, entry in enumerate(walked):
        node = entry[0] if isinstance(entry, tuple) else entry
        ref = _as_str(_get_attr(node, "self_ref"))
        if ref and ref not in rank:
            rank[ref] = position
    return rank or None


def _page_bbox_key(obj: object) -> tuple[float, ...] | None:
    page_number = _extract_page_number(obj)
    if page_number is None:
        return None
    top = 0.0
    raw_provenance = _get_attr(obj, "prov", "provenance")
    if not isinstance(raw_provenance, list | tuple | dict):
        return None
    prov = next(_iter_provenance(raw_provenance), None)
    bbox = _get_attr(prov, "bbox") if prov is not None else None
    t = _get_attr(bbox, "t", "top") if bbox is not None else None
    if (
        isinstance(t, int | float)
        and not isinstance(t, bool)
        and abs(t) <= _MAX_TABLE_COORD_ABS
        and (not isinstance(t, float) or math.isfinite(t))
    ):
        origin = _get_attr(bbox, "coord_origin")
        origin = getattr(origin, "value", origin)
        origin_text = (
            origin.upper()
            if isinstance(origin, str) and len(origin) <= _MAX_TABLE_ORIGIN_CHARS
            else ""
        )
        # BOTTOMLEFT (docling's PDF default): larger t is higher on the page.
        top = -float(t) if origin_text == "BOTTOMLEFT" else float(t)
    return (float(page_number), top, 0.0)


def _iter_values(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, list | tuple):
        return list(value)
    try:
        return list(value)  # type: ignore[no-any-return,call-overload]
    except TypeError:
        return [value]


def _as_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _get_attr(obj: object, *names: str) -> object | None:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)  # type: ignore[no-any-return]
        if isinstance(obj, dict) and name in obj:
            return obj[name]  # type: ignore[no-any-return]
    return None


def _call_or_value(obj: object, *names: str) -> object | None:
    for name in names:
        value = _get_attr(obj, name)
        if value is None:
            continue
        if callable(value):
            try:
                return value()  # type: ignore[no-any-return]
            except TypeError:
                continue
        return value
    return None


def _normalize_label(raw: object | None) -> str:
    if raw is None:
        return ""
    value = str(raw).lower().strip()
    if "." in value:
        value = value.split(".")[-1]
    return value


def _is_heading(label: str) -> bool:
    return label in {"title", "heading", "header", "section_header", "section-title"}


_CHAPTER_HEADING_RE = re.compile(
    r"^\s*(chapter|part|section|book|appendix)\s+([0-9]+|[ivxlcdm]+)\b", re.I
)
_CHAPTER_KEYWORD_RE = re.compile(r"^\s*(chapter|part|section|appendix|unit|book)\b", re.I)
_NUM_SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\b")
_APPENDIX_TABLE_RE = re.compile(r"^[A-Z]\d")
_CAPTION_PREFIX_RE = re.compile(r"^(figure|fig\.?|table|photo|image|plate|diagram)\b", re.I)


def _is_caption_text(text: str) -> bool:
    t = (text or "").strip()
    if _CAPTION_PREFIX_RE.match(t):
        return True
    return t.endswith(".") and len(t.split()) > 7


def _infer_heading_level(text: str, docling_level: int) -> int:
    if docling_level >= 2:
        return docling_level
    t = (text or "").strip()
    if not t:
        return 2
    if _CHAPTER_HEADING_RE.match(t) or _CHAPTER_KEYWORD_RE.match(t):
        return 1
    m = _NUM_SECTION_RE.match(t)
    if m:
        return min(m.group(1).count(".") + 1, 4)
    if _APPENDIX_TABLE_RE.match(t):
        return 2
    letters = [c for c in t if c.isalpha()]
    if letters and all(c.isupper() for c in letters) and len(t.split()) <= 9:
        return 1
    return 2


def _heading_level(obj: object) -> int:
    for name in ("level", "heading_level", "section_level"):
        value = _get_attr(obj, name)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    for name in ("label", "type", "kind"):
        raw = _as_str(_get_attr(obj, name))
        digits = "".join(ch for ch in raw if ch.isdigit())
        if digits:
            return int(digits)
    return 0


def _label_to_element_type(label: str) -> ElementType:
    if label in {"table"}:
        return ElementType.TABLE
    if label == "document_index":
        # A table-of-contents line (VOR-1452): body text, never a TITLE; its
        # ``docling_label`` makes the chunker file the page as contents.
        return ElementType.NARRATIVE_TEXT
    if label in {"list_item", "list", "listitem"}:
        return ElementType.LIST_ITEM
    if label in {"caption", "figure_caption"}:
        return ElementType.FIGURE_CAPTION
    if label in {"equation", "formula"}:
        return ElementType.EQUATION
    if label in {"code", "code_block"}:
        return ElementType.CODE_BLOCK
    # docling-core's DocItemLabel values are "page_header"/"page_footer"
    # (VOR-1452): matching only "header" let running heads through as
    # NARRATIVE_TEXT, where the chunker promoted them to section headings.
    if label in {"header", "page_header"}:
        return ElementType.HEADER
    if label in {"footer", "page_footer"}:
        return ElementType.FOOTER
    return ElementType.NARRATIVE_TEXT


def _extract_page_number(obj: object) -> int | None:
    for name in ("page_no", "page_number", "page", "page_idx"):
        value = _get_attr(obj, name)
        page_number = _normalized_page_number(name, value)
        if page_number is not None:
            return page_number
    prov = _get_attr(obj, "prov", "provenance")
    if not isinstance(prov, list | tuple | dict):
        return None
    for p in islice(_iter_provenance(prov), _MAX_TABLE_PROVENANCE_INSPECTED + 1):
        for name in ("page_no", "page_number", "page", "page_idx"):
            page_number = _normalized_page_number(name, _get_attr(p, name))
            if page_number is not None:
                return page_number
    return None


def _normalized_page_number(field: str, value: object | None) -> int | None:
    """Normalize one bounded Docling page field, retaining page_idx semantics."""
    if not isinstance(value, int) or isinstance(value, bool) or value > _MAX_TABLE_COORD_ABS:
        return None
    if field == "page_idx":
        return value + 1 if value >= 0 else None
    return value if value > 0 else None


def _table_provenance_metadata(obj: object) -> dict[str, Any]:
    """Retain bounded Docling table location evidence without asserting fidelity.

    Coordinates are copied as emitted by Docling. This mapper does not convert
    coordinate origins or infer units, and the locator does not prove the
    source region was correctly recognized as a table.
    """
    raw_ref = _get_attr(obj, "self_ref")
    if isinstance(raw_ref, str):
        bounded_ref = raw_ref[:_MAX_TABLE_SELF_REF_CHARS]
        ref_truncated = len(raw_ref) > _MAX_TABLE_SELF_REF_CHARS
        try:
            bounded_ref.encode("utf-8")
        except UnicodeEncodeError:
            source_ref = None
            ref_sha256 = None
            ref_status = "invalid"
        else:
            source_ref = bounded_ref
            # Never hash an unbounded locator. Its omitted suffix has unknown
            # content and is explicitly not claimed to be fingerprinted.
            ref_sha256 = None
            ref_status = "truncated" if ref_truncated else "available"
    else:
        source_ref = None
        ref_truncated = False
        ref_sha256 = None
        ref_status = "unavailable"

    raw_provenance = _get_attr(obj, "prov", "provenance")
    kept: list[dict[str, Any]] = []
    inventory_hash = hashlib.sha256()
    valid_count = 0
    inspected_count = 0
    has_more = False
    for occurrence in _iter_provenance(raw_provenance):
        if inspected_count >= _MAX_TABLE_PROVENANCE_INSPECTED:
            has_more = True
            break
        inspected_count += 1
        record = _table_provenance_occurrence(occurrence)
        if record["status"] == "available":
            valid_count += 1
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        inventory_hash.update(encoded)
        inventory_hash.update(b"\n")
        if len(kept) < _MAX_TABLE_PROVENANCE_OCCURRENCES:
            kept.append(record)

    if has_more:
        geometry_status = "unknown_truncated"
    elif inspected_count == 0:
        geometry_status = "unavailable"
    elif valid_count == inspected_count:
        geometry_status = "available"
    elif valid_count:
        geometry_status = "partial"
    else:
        geometry_status = "invalid"

    return {
        "source_self_ref": source_ref,
        "source_self_ref_status": ref_status,
        "source_self_ref_truncated": ref_truncated,
        "source_self_ref_sha256": ref_sha256,
        "geometry_status": geometry_status,
        "source_alignment": "unverified",
        "occurrences": kept,
        "occurrence_count": inspected_count,
        "occurrence_count_scope": "inspected_prefix",
        "total_occurrence_count": None if has_more else inspected_count,
        "occurrences_truncated": has_more or inspected_count > _MAX_TABLE_PROVENANCE_OCCURRENCES,
        "occurrence_inspection_truncated": has_more,
        "occurrence_inspection_limit": _MAX_TABLE_PROVENANCE_INSPECTED,
        "occurrence_inventory_sha256": inventory_hash.hexdigest(),
        "occurrence_inventory_sha256_scope": "inspected_prefix" if has_more else "complete",
    }


def _table_provenance_occurrence(provenance: object) -> dict[str, Any]:
    """Copy one bounded, JSON-safe page and bbox record from Docling."""
    record: dict[str, Any] = {
        "status": "invalid",
        "page_number_source_field": None,
        "page_number_status": "missing",
    }
    for field in ("page_no", "page_number", "page", "page_idx"):
        value = _get_attr(provenance, field)
        if value is None:
            continue
        record["page_number_source_field"] = field
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and abs(value) <= _MAX_TABLE_COORD_ABS
            and ((field == "page_idx" and value >= 0) or (field != "page_idx" and value > 0))
        ):
            # Keep the original field name and value: page_idx can be zero-based.
            record[field] = value
            record["page_number_status"] = "available"
        else:
            record["page_number_status"] = "invalid"
        break

    bbox = _get_attr(provenance, "bbox")
    if bbox is None:
        record["bbox"] = None
        record["status"] = (
            "unavailable" if record["page_number_status"] == "missing" else "bbox_missing"
        )
        return record

    copied_bbox: dict[str, int | float | str | None] = {}
    bbox_valid = True
    for axis in ("l", "t", "r", "b"):
        value = _get_attr(bbox, axis)
        if isinstance(value, int | float) and not isinstance(value, bool):
            if (isinstance(value, float) and not math.isfinite(value)) or abs(
                value
            ) > _MAX_TABLE_COORD_ABS:
                copied_bbox[axis] = None
                bbox_valid = False
            else:
                copied_bbox[axis] = value
        else:
            copied_bbox[axis] = None
            bbox_valid = False

    origin = _get_attr(bbox, "coord_origin")
    origin = getattr(origin, "value", origin)
    if isinstance(origin, str) and origin and len(origin) <= _MAX_TABLE_ORIGIN_CHARS:
        try:
            origin.encode("utf-8")
        except UnicodeEncodeError:
            copied_bbox["coord_origin"] = None
            bbox_valid = False
        else:
            copied_bbox["coord_origin"] = origin
    else:
        copied_bbox["coord_origin"] = None
        bbox_valid = False
    record["coord_origin_status"] = (
        "known_unverified"
        if copied_bbox["coord_origin"] is not None
        else (
            "invalid_too_long"
            if isinstance(origin, str) and len(origin) > _MAX_TABLE_ORIGIN_CHARS
            else "invalid"
        )
    )
    if copied_bbox["coord_origin"] not in {"TOPLEFT", "BOTTOMLEFT"}:
        record["coord_origin_status"] = (
            "unknown_unverified"
            if copied_bbox["coord_origin"] is not None
            else record["coord_origin_status"]
        )
        bbox_valid = False
    record["bbox"] = copied_bbox
    if record["page_number_status"] != "available":
        record["status"] = "page_number_invalid"
    else:
        record["status"] = "available" if bbox_valid else "bbox_invalid"
    return record


def _iter_provenance(value: object | None) -> Iterator[object]:
    """Iterate provenance without duplicating an arbitrarily long source list."""
    if value is None:
        return iter(())
    if isinstance(value, dict):
        return iter(value.values())
    if isinstance(value, str | bytes | bytearray):
        return iter((value,))
    try:
        return iter(value)  # type: ignore[no-any-return,call-overload]
    except TypeError:
        return iter((value,))


def _pil_to_png_bytes(pil: object) -> bytes:
    """Serialize a PIL.Image-like object (duck-typed) to PNG bytes."""
    if not (hasattr(pil, "save") and hasattr(pil, "size") and hasattr(pil, "mode")):
        return b""
    buf = io.BytesIO()
    try:
        pil.save(buf, format="PNG")
    except (OSError, ValueError):
        return b""
    return buf.getvalue()


def _decode_data_uri(uri: object) -> bytes:
    """Decode a ``data:<mime>;base64,<payload>`` URI to raw bytes."""
    text = str(uri or "")
    marker = ";base64,"
    if text.startswith("data:") and marker in text:
        try:
            return base64.b64decode(text.split(marker, 1)[1])
        except (ValueError, TypeError):
            return b""
    return b""


def _extract_image_bytes(picture_obj: object) -> bytes:
    # docling-core 2.x exposes a PictureItem's image via ``.image`` -> ImageRef,
    # which carries a PIL image (``.pil_image``) and/or a ``data:`` base64 URI.
    # Older docling exposed raw bytes / a ``get_image_bytes`` method. Try the
    # modern PIL + data-URI paths first (they fire on the recursion into the
    # ImageRef), then fall back to the legacy byte-attrs / zero-arg methods.
    try:
        png = _pil_to_png_bytes(_get_attr(picture_obj, "pil_image"))
    except Exception:  # noqa: BLE001 - a flaky image property must not kill the parse
        png = b""
    if png:
        return png
    decoded = _decode_data_uri(_get_attr(picture_obj, "uri"))
    if decoded:
        return decoded
    for name in ("image", "data", "bytes", "blob", "binary"):
        value = _get_attr(picture_obj, name)
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)
        nested = (
            _extract_image_bytes(value) if value is not None and value is not picture_obj else b""
        )
        if nested:
            return nested
    for method_name in ("to_bytes", "export_image", "get_image_bytes"):
        method = _get_attr(picture_obj, method_name)
        if callable(method):
            try:
                result = method()
                if isinstance(result, bytes):
                    return result
            except TypeError:
                continue
    return b""


def _image_ext_and_mime(picture_obj: object) -> tuple[str, str]:
    ext = _as_str(_get_attr(picture_obj, "ext", "extension", "format")).lower()
    if ext.startswith("."):
        ext = ext[1:]
    mime = _as_str(_get_attr(picture_obj, "mime_type", "content_type"))
    if mime:
        guessed = mime.split("/")[-1].lower()
        if not ext and guessed:
            ext = "jpg" if guessed == "jpeg" else guessed
    if not ext:
        ext = "png"
    if not mime:
        mime = f"image/{'jpeg' if ext == 'jpg' else ext}"
    return ext, mime


def _sorted_pages(
    doc: object, elements: list[ParsedElement], images: list[ExtractedImage]
) -> list[int]:
    pages: set[int] = set()
    doc_pages = _get_attr(doc, "pages")
    for p in _iter_values(doc_pages):
        page_number = _extract_page_number(p)
        if page_number is not None:
            pages.add(page_number)
    for element in elements:
        if element.page_number is not None:
            pages.add(element.page_number)
    for image in images:
        pages.add(image.page_number)
    if not pages:
        pages.add(1)
    return sorted(pages)


def _document_markdown(doc: object, elements: list[ParsedElement]) -> str:
    """Markdown built from the mapped elements, in one linear pass.

    Not docling's ``export_to_markdown``: that re-serialises the whole book in
    Python on the lane's single core (18 s for a 3,915-page book, about 40% of
    the mapping; VOR-1427), and the only readers of ``ParsedDocument.markdown``
    (the parse step's has-text check and its 20,000-character corruption
    sample) are served just as well by the elements. ``doc`` is unused and
    kept so callers don't change.
    """
    _ = doc
    lines: list[str] = []
    for element in elements:
        if element.element_type == ElementType.TITLE:
            depth = max(1, element.section_depth)
            lines.append(f"{'#' * depth} {element.text}")
        elif element.element_type == ElementType.TABLE:
            md = _as_str(element.metadata.get("markdown"))
            lines.append(md or element.text)
        elif element.element_type in {ElementType.NARRATIVE_TEXT, ElementType.LIST_ITEM}:
            if element.text:
                lines.append(element.text)
    return "\n\n".join(line for line in lines if line)


def _build_heading_tree(headings: list[tuple[int, str]]) -> list[dict[str, Any]]:
    root: list[dict[str, Any]] = []
    stack: list[tuple[int, dict[str, Any]]] = []
    for level, text in headings:
        node = {"title": text, "level": level, "children": []}
        while stack and stack[-1][0] >= level:
            stack.pop()
        if not stack:
            root.append(node)
        else:
            stack[-1][1]["children"].append(node)
        stack.append((level, node))
    return root

"""Image-eligibility filtering for the docling extraction path (VOR-304).

Frivolous images (tiny icons, exact/near-duplicates, near-full-page renders)
are dropped before the VLM describe pass instead of polluting the figure
store.

The eligibility *decision* (`filter_image_candidates`) is pure and operates on
precomputed inputs — width/height, the byte sha (`image_id`), the average-hash,
and the page bbox — so it is hermetic and importable without Pillow. The pixel
average-hash (`average_hash`) is the only PIL-dependent piece and lazy-imports
Pillow, returning ``None`` when it is unavailable (perceptual dedup then no-ops).
"""

from __future__ import annotations

import io
from collections import Counter
from dataclasses import dataclass

# Thresholds: min 96x96 / 16_384 px², perceptual
# Hamming <= 4, drop renders covering >= 75% of the page, and require at least
# the photo floor of 10% page coverage when a bbox is known.
MIN_WIDTH = 96
MIN_HEIGHT = 96
MIN_AREA = 16_384
PHASH_DISTANCE = 4
MAX_PAGE_FRACTION = 0.75
MIN_BBOX_FRACTION = 0.10


@dataclass
class ImageCandidate:
    """One extracted image plus the geometry/hash needed to judge eligibility."""

    image_id: str  # sha256 of the image bytes (exact-duplicate key)
    width: int
    height: int
    page_number: int | None = None
    average_hash: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    page_width: float = 0.0
    page_height: float = 0.0


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def filter_image_candidates(
    candidates: list[ImageCandidate],
) -> tuple[list[ImageCandidate], Counter[str]]:
    """Return the eligible candidates plus a per-reason cull breakdown.

    Filters applied in order: minimum pixel size; page-coverage bounds (when a
    bbox is known); exact byte dedup; perceptual near-duplicate dedup. The
    perceptual pass also absorbs recurring page-furniture (a logo repeated across
    pages is a near-duplicate of its first occurrence).
    """
    kept: list[ImageCandidate] = []
    reasons: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_hashes: list[int] = []

    for cand in candidates:
        # Only cull as too-small when dimensions are KNOWN (> 0). width/height <= 0
        # means undecodable bytes or Pillow absent — fail open, judge by dedup only.
        if (
            cand.width > 0
            and cand.height > 0
            and (
                cand.width < MIN_WIDTH
                or cand.height < MIN_HEIGHT
                or cand.width * cand.height < MIN_AREA
            )
        ):
            reasons["too_small"] += 1
            continue

        page_area = cand.page_width * cand.page_height
        if cand.bbox is not None and page_area > 0:
            x0, y0, x1, y1 = cand.bbox
            bbox_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            coverage = bbox_area / page_area
            if coverage >= MAX_PAGE_FRACTION:
                reasons["near_full_page"] += 1
                continue
            if coverage < MIN_BBOX_FRACTION:
                reasons["below_min_coverage"] += 1
                continue

        if cand.image_id in seen_ids:
            reasons["exact_duplicate"] += 1
            continue
        if cand.average_hash is not None and any(
            hamming_distance(cand.average_hash, seen) <= PHASH_DISTANCE for seen in seen_hashes
        ):
            reasons["perceptual_duplicate"] += 1
            continue

        seen_ids.add(cand.image_id)
        if cand.average_hash is not None:
            seen_hashes.append(cand.average_hash)
        kept.append(cand)

    return kept, reasons


def average_hash(image_bytes: bytes, *, size: int = 8) -> int | None:
    """8x8 grayscale average-hash (aHash) of the image, or ``None`` on failure.

    Lazy-imports Pillow; returns ``None`` if Pillow is unavailable or the bytes
    cannot be decoded, so the caller treats the image as not-deduplicable rather
    than crashing.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            gray = img.convert("L").resize((size, size))
            values = list(gray.getdata())
    except Exception:  # noqa: BLE001 - a flaky image must not kill the parse
        return None
    if not values:
        return None
    avg = sum(values) / len(values)
    out = 0
    for value in values:
        out = (out << 1) | int(value >= avg)
    return out


def image_dimensions(image_bytes: bytes) -> tuple[int, int] | None:
    """Return ``(width, height)`` in pixels, or ``None`` if undecodable.

    Lazy-imports Pillow. ``None`` (Pillow missing or corrupt bytes) lets the
    caller drop the image as undescribable rather than crash.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return int(img.width), int(img.height)
    except Exception:  # noqa: BLE001 - a flaky image must not kill the parse
        return None


__all__ = [
    "MIN_AREA",
    "MIN_BBOX_FRACTION",
    "MIN_HEIGHT",
    "MIN_WIDTH",
    "MAX_PAGE_FRACTION",
    "PHASH_DISTANCE",
    "ImageCandidate",
    "average_hash",
    "filter_image_candidates",
    "hamming_distance",
    "image_dimensions",
]

"""Tests for image-eligibility filtering (VOR-304).

Frivolous images (tiny icons, exact/near-duplicates, near-full-page renders)
are dropped on the docling path before the VLM describe pass. The logic
operates on precomputed inputs (no PIL), so these tests are hermetic.
"""

from __future__ import annotations

import io

import pytest

from vo_rag.core.parsing.image_filters import (
    MIN_AREA,
    MIN_HEIGHT,
    MIN_WIDTH,
    PHASH_DISTANCE,
    ImageCandidate,
    average_hash,
    filter_image_candidates,
    hamming_distance,
    image_dimensions,
)


def _cand(
    image_id: str,
    *,
    width: int = 200,
    height: int = 200,
    page_number: int | None = 1,
    average_hash: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    page_width: float = 0.0,
    page_height: float = 0.0,
) -> ImageCandidate:
    return ImageCandidate(
        image_id=image_id,
        width=width,
        height=height,
        page_number=page_number,
        average_hash=average_hash,
        bbox=bbox,
        page_width=page_width,
        page_height=page_height,
    )


def test_default_cull_thresholds() -> None:
    assert (MIN_WIDTH, MIN_HEIGHT, MIN_AREA, PHASH_DISTANCE) == (96, 96, 16_384, 4)


def test_hamming_distance() -> None:
    assert hamming_distance(0b1011, 0b1110) == 2
    assert hamming_distance(5, 5) == 0


def test_too_small_images_rejected() -> None:
    cands = [
        _cand("ok", width=200, height=200),
        _cand("narrow", width=50, height=400),  # width < 96
        _cand("short", width=400, height=50),  # height < 96
        _cand("small-area", width=96, height=96),  # 9216 < 16_384
    ]
    kept, reasons = filter_image_candidates(cands)
    assert [c.image_id for c in kept] == ["ok"]
    assert reasons["too_small"] == 3


def test_unknown_dimensions_not_culled_by_size() -> None:
    # width/height <= 0 == dimensions undeterminable (undecodable bytes / no Pillow);
    # fail open rather than silently drop a possibly-real image.
    kept, reasons = filter_image_candidates([_cand("unknown", width=0, height=0)])
    assert [c.image_id for c in kept] == ["unknown"]
    assert reasons["too_small"] == 0


def test_exact_duplicate_dropped_keeping_first() -> None:
    cands = [
        _cand("sha-A"),
        _cand("sha-A"),  # identical bytes -> same image_id
        _cand("sha-B"),
    ]
    kept, reasons = filter_image_candidates(cands)
    assert [c.image_id for c in kept] == ["sha-A", "sha-B"]
    assert reasons["exact_duplicate"] == 1


def test_perceptual_near_duplicate_dropped() -> None:
    # average_hash within Hamming<=4 of an already-kept image -> drop.
    cands = [
        _cand("A", average_hash=0b0),
        _cand("B", average_hash=0b1111),  # hamming 4 -> duplicate
        _cand("C", average_hash=0b11111),  # hamming 5 -> distinct, kept
    ]
    kept, reasons = filter_image_candidates(cands)
    assert [c.image_id for c in kept] == ["A", "C"]
    assert reasons["perceptual_duplicate"] == 1


def test_missing_average_hash_is_not_deduped() -> None:
    cands = [_cand("A", average_hash=None), _cand("B", average_hash=None)]
    kept, reasons = filter_image_candidates(cands)
    assert {c.image_id for c in kept} == {"A", "B"}
    assert reasons["perceptual_duplicate"] == 0


def test_below_min_page_coverage_rejected() -> None:
    # 1000x1000 page; a 200x200-pt placement = 4% coverage (< 10% photo floor).
    cands = [_cand("tiny-on-page", bbox=(0, 0, 200, 200), page_width=1000, page_height=1000)]
    kept, reasons = filter_image_candidates(cands)
    assert kept == []
    assert reasons["below_min_coverage"] == 1


def test_near_full_page_render_rejected() -> None:
    # 90% page coverage -> dropped as a near-full-page raster (>= 0.75).
    cands = [_cand("full-page", bbox=(0, 0, 900, 1000), page_width=1000, page_height=1000)]
    kept, reasons = filter_image_candidates(cands)
    assert kept == []
    assert reasons["near_full_page"] == 1


def test_in_range_coverage_kept() -> None:
    cands = [_cand("good", bbox=(0, 0, 400, 400), page_width=1000, page_height=1000)]  # 16%
    kept, reasons = filter_image_candidates(cands)
    assert [c.image_id for c in kept] == ["good"]
    assert sum(reasons.values()) == 0


def test_no_geometry_skips_coverage_checks() -> None:
    # epub images carry no page bbox; coverage filters must not fire.
    cands = [_cand("epub-img", bbox=None)]
    kept, reasons = filter_image_candidates(cands)
    assert [c.image_id for c in kept] == ["epub-img"]
    assert sum(reasons.values()) == 0


def test_pillow_probes_when_available() -> None:
    pil_image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    pil_image.new("RGB", (120, 96), (128, 128, 128)).save(buf, "PNG")
    png = buf.getvalue()

    assert image_dimensions(png) == (120, 96)
    assert image_dimensions(b"not-an-image") is None

    digest = average_hash(png)
    assert isinstance(digest, int)
    assert average_hash(png) == digest  # deterministic
    assert average_hash(b"not-an-image") is None

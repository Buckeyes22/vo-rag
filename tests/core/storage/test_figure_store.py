"""FigureStore — deterministic-key filesystem store for figure images."""

from __future__ import annotations

import hashlib

import pytest

from vo_rag.core.storage.figures import (
    ALLOWED_EXTS,
    MIME_BY_EXT,
    FigureStore,
    safe_ext,
)

SRC = "ab" * 32  # 64 lowercase hex chars — canonical source identity


def test_equation_crop_publication_is_separate_and_does_not_replace_conflicts(tmp_path) -> None:
    store = FigureStore(tmp_path)
    old = store.write(SRC, 12, 0, b"original-figure", ext="png")
    raw = b"equation-png"
    ref = store.write_equation(SRC, 12, raw)
    assert ref == f"{SRC}/p012_eq{hashlib.sha256(raw).hexdigest()}.png"
    assert store.write_equation(SRC, 12, raw) == ref
    assert store.open_bytes(old) == b"original-figure"
    store.path_for(ref).write_bytes(b"conflicting")
    with pytest.raises(ValueError, match="conflicting bytes"):
        store.write_equation(SRC, 12, raw)
    assert store.open_bytes(ref) == b"conflicting"
    assert not list(store.path_for(ref).parent.glob("*.tmp"))


def test_write_returns_deterministic_ref(tmp_path) -> None:
    store = FigureStore(tmp_path)
    ref1 = store.write(SRC, 7, 2, b"png-bytes", ext="png")
    ref2 = store.write(SRC, 7, 2, b"png-bytes", ext="png")
    assert ref1 == ref2 == f"{SRC}/p007_img002.png"


def test_round_trip_bytes(tmp_path) -> None:
    store = FigureStore(tmp_path)
    ref = store.write(SRC, 1, 0, b"\x89PNG-payload", ext="png")
    assert store.open_bytes(ref) == b"\x89PNG-payload"
    assert store.exists(ref) is True
    assert store.path_for(ref) == tmp_path.resolve() / SRC / "p001_img000.png"


def test_exists_false_for_unwritten_and_malformed(tmp_path) -> None:
    store = FigureStore(tmp_path)
    assert store.exists(f"{SRC}/p001_img000.png") is False
    assert store.exists("../../etc/passwd") is False
    assert store.exists("not-a-ref") is False


def test_wide_page_and_index_numbers(tmp_path) -> None:
    # :03d is minimum width — 4-digit pages (books >999pp) must validate.
    store = FigureStore(tmp_path)
    ref = store.write(SRC, 1500, 1000, b"x", ext="jpg")
    assert ref == f"{SRC}/p1500_img1000.jpg"
    assert store.exists(ref)


@pytest.mark.parametrize(
    "bad_ref",
    [
        "../evil/p001_img001.png",  # traversal in src segment
        f"{SRC}/../p001_img001.png",  # traversal in filename segment
        f"{SRC}/p001_img001.png.tmp",  # in-progress write suffix
        f"{SRC}/p001_img001.svg",  # svg excluded (XSS when served raw)
        "ZZ" * 32 + "/p001_img001.png",  # non-hex src
        f"{SRC}/page001_img001.png",  # old garage-era filename shape
        f"{SRC}",  # no filename
    ],
)
def test_path_for_rejects_malformed(tmp_path, bad_ref: str) -> None:
    store = FigureStore(tmp_path)
    with pytest.raises(ValueError):
        store.path_for(bad_ref)


def test_write_rejects_bad_src_and_ext(tmp_path) -> None:
    store = FigureStore(tmp_path)
    with pytest.raises(ValueError):
        store.write("short-hash", 1, 0, b"x", ext="png")
    with pytest.raises(ValueError):
        store.write(SRC, 1, 0, b"x", ext="svg")


def test_safe_ext_normalizes() -> None:
    assert safe_ext("PNG", "image/png") == "png"
    assert safe_ext(".jpeg", "image/jpeg") == "jpeg"
    assert safe_ext("", "image/jpeg") == "jpg"  # mime fallback
    assert safe_ext("gif", "image/gif") == "gif"  # EPUBs carry gifs
    assert safe_ext("svg", "image/svg+xml") is None  # refused
    assert safe_ext("tiff", "image/tiff") is None
    assert safe_ext("", "") is None
    assert set(MIME_BY_EXT) == set(ALLOWED_EXTS)


def test_safe_ext_refused_mime_beats_allowed_extension() -> None:
    # Mislabeled-SVG defense: a "cover.png" EPUB item whose media_type is
    # image/svg+xml must be refused — the MIME verdict wins over the ext.
    assert safe_ext("png", "image/svg+xml") is None
    assert safe_ext("jpg", "image/svg") is None


def test_concurrent_same_key_writes_do_not_collide(tmp_path) -> None:
    # Two parallel ingests of the same source/page/index must both succeed
    # (unique temp names; the old shared `<final>.tmp` lost one replace()).
    from concurrent.futures import ThreadPoolExecutor

    store = FigureStore(tmp_path)
    payloads = [bytes([n]) * 64 for n in range(8)]

    def _write(payload: bytes) -> str:
        return store.write(SRC, 1, 0, payload, ext="png")

    with ThreadPoolExecutor(max_workers=8) as pool:
        refs = list(pool.map(_write, payloads))

    assert set(refs) == {f"{SRC}/p001_img000.png"}
    assert store.open_bytes(refs[0]) in payloads  # one writer's bytes, intact
    leftovers = list((tmp_path / SRC).glob("*.tmp"))
    assert leftovers == []  # no abandoned temp files


def test_settings_expose_figures_root() -> None:
    from vo_rag.core.config.settings import FiguresSettings

    assert FiguresSettings().root == "/data/vo-rag/figures"

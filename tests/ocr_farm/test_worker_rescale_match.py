"""The OCR farm worker rescales oversize pages exactly like vo-rag (VOR-1434 row 11).

The real 13,400 x 9,100pt failure happened on the farm, which runs docling on
the original ``jobs/<sha>.pdf``. The worker has no vo-rag or PyMuPDF install,
so it carries a pypdfium2 port of
``vo_rag.core.parsing.parsers.docling.rescale_oversize_pages``. These tests
pin the thresholds, check the worker hands docling the rescaled copy, and
(where pypdfium2 is installed, i.e. ``--extra runtime``) check that both
implementations produce the same page sizes and keep the text.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import fitz  # type: ignore[import-untyped, unused-ignore]
import pytest

from tests.ocr_farm._fakes import (
    FakeDoclingDocument,
    FakeSize,
    fake_torch,
    install_fake_docling,
    load_worker,
)
from vo_rag.core.parsing.parsers import docling as docling_module


def _build_pdf(path: Path, sizes: list[tuple[float, float]], rotate: int = 0) -> Path:
    doc = fitz.open()
    for index, (width, height) in enumerate(sizes):
        page = doc.new_page(width=width, height=height)
        page.insert_text((72, 144), f"Page marker {index}", fontsize=24)
        page.set_rotation(rotate)
    doc.save(path)
    doc.close()
    return path


def _sizes(path: str | Path) -> list[tuple[float, float]]:
    with fitz.open(path) as doc:
        return [(round(p.rect.width, 1), round(p.rect.height, 1)) for p in doc]


def _texts(path: str | Path) -> list[str]:
    with fitz.open(path) as doc:
        return [" ".join(p.get_text("text").split()) for p in doc]


class _Result:
    status = "success"

    def __init__(self) -> None:
        self.document = FakeDoclingDocument(pages={1: FakeSize(width=612.0, height=792.0)})


def _recording_converter(seen: list[tuple[str, list[tuple[float, float]]]]) -> Any:
    class _Converter:
        def __init__(self, *, format_options: dict[str, Any]) -> None:
            self.format_options = format_options

        def convert(self, source: str, page_range: tuple[int, int] = (1, 1)) -> _Result:
            _ = page_range
            seen.append((source, _sizes(source) if Path(source).stat().st_size > 64 else []))
            return _Result()

    return _Converter


def _run(monkeypatch: pytest.MonkeyPatch, worker: Any, pdf: Path, out: Path) -> int:
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "torch", fake_torch(available=True, peak_bytes=4096))
        return int(worker.main([str(pdf), str(out)]))


def test_thresholds_match_vo_rag() -> None:
    worker = load_worker()
    assert worker.OVERSIZE_PAGE_PT == docling_module.OVERSIZE_PAGE_PT
    assert worker.RESCALED_PAGE_MAX_PT == docling_module.RESCALED_PAGE_MAX_PT


def test_converter_receives_the_rescaled_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Runs everywhere: the rescale itself is stubbed, the wiring is real."""
    seen: list[tuple[str, list[tuple[float, float]]]] = []
    install_fake_docling(monkeypatch, _recording_converter(seen))
    worker = load_worker()
    pdf = tmp_path / "job.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    calls: list[tuple[Path, Path]] = []

    def _rescale(source: Path, workdir: Path) -> Path:
        calls.append((source, workdir))
        copy = workdir / source.name
        copy.write_bytes(b"%PDF-1.4\n")
        return copy

    monkeypatch.setattr(worker, "_rescale_oversize_pages", _rescale)

    assert _run(monkeypatch, worker, pdf, tmp_path / "job.json") == 0

    [(source, workdir)] = calls
    assert source == pdf
    assert seen[0][0] == str(workdir / "job.pdf") != str(pdf)
    assert not workdir.exists()  # temp copy cleaned up


def test_converter_receives_the_original_when_nothing_is_oversize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[tuple[str, list[tuple[float, float]]]] = []
    install_fake_docling(monkeypatch, _recording_converter(seen))
    worker = load_worker()
    pdf = tmp_path / "job.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    assert _run(monkeypatch, worker, pdf, tmp_path / "job.json") == 0
    assert seen[0][0] == str(pdf)


# ---------------------------------------------------------------------------
# Real pypdfium2 (installed with docling, the ``runtime`` extra).
# ---------------------------------------------------------------------------


def test_worker_rescales_an_oversize_page_before_docling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdfium = pytest.importorskip("pypdfium2")
    seen: list[tuple[str, list[tuple[float, float]]]] = []
    install_fake_docling(monkeypatch, _recording_converter(seen))
    monkeypatch.setitem(sys.modules, "pypdfium2", pdfium)
    worker = load_worker()
    pdf = _build_pdf(tmp_path / "job.pdf", [(400, 500), (13_400, 9_100)])
    before = pdf.read_bytes()

    assert _run(monkeypatch, worker, pdf, tmp_path / "job.json") == 0

    [(source, sizes)] = seen
    assert source != str(pdf) and not Path(source).exists()
    assert len(sizes) == 2
    assert all(max(w, h) <= docling_module.RESCALED_PAGE_MAX_PT for w, h in sizes[1:])
    assert pdf.read_bytes() == before


@pytest.mark.parametrize("rotate", [0, 90, 180, 270])
def test_worker_and_vo_rag_rescale_to_the_same_pages(tmp_path: Path, rotate: int) -> None:
    pdfium = pytest.importorskip("pypdfium2")
    worker = load_worker()
    src = _build_pdf(tmp_path / "plate.pdf", [(400, 500), (13_400, 9_100), (9_100, 13_400)], rotate)
    (tmp_path / "w").mkdir()
    (tmp_path / "v").mkdir()

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "pypdfium2", pdfium)
        ours = worker._rescale_oversize_pages(src, tmp_path / "w")
    theirs = docling_module.rescale_oversize_pages(src, tmp_path / "v")

    assert ours is not None and theirs is not None
    assert _sizes(ours) == _sizes(theirs)
    assert _texts(ours) == _texts(theirs) == _texts(src)


def test_worker_rescale_falls_back_on_unreadable_pdf(tmp_path: Path) -> None:
    pdfium = pytest.importorskip("pypdfium2")
    worker = load_worker()
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "pypdfium2", pdfium)
        assert worker._rescale_oversize_pages(broken, tmp_path) is None

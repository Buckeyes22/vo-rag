"""PDFs pypdfium2 can't open but PyMuPDF can get an unlocked copy (VOR-1427).

Real case: a scanned aerodynamics textbook carries an owner-only encryption
dictionary whose security scheme PDFium rejects, so docling (pypdfium2
backend) called the file "not valid" while PyMuPDF read every page.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # type: ignore[import-untyped, unused-ignore]
import pytest

from vo_rag.core.parsing.parsers import docling as docling_module
from vo_rag.core.parsing.parsers.docling import unlock_for_pdfium

pytest.importorskip("pypdfium2")


def _build_pdf(path: Path, pages: int = 3, **save_kwargs: object) -> Path:
    doc = fitz.open()
    for i in range(pages):
        doc.new_page(width=612, height=792).insert_text((72, 72), f"page {i + 1} text")
    doc.save(path, **save_kwargs)
    doc.close()
    return path


def _texts(path: Path) -> list[str]:
    with fitz.open(path) as doc:
        return [page.get_text().strip() for page in doc]


def _pdfium_rejects(original: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make pypdfium2 fail on ``original`` only, like the unsupported scheme."""
    real = docling_module.pdfium_can_open
    monkeypatch.setattr(
        docling_module,
        "pdfium_can_open",
        lambda path: False if Path(path) == original else real(path),
    )


def test_readable_pdf_is_left_alone(tmp_path: Path) -> None:
    pdf = _build_pdf(tmp_path / "ok.pdf")
    assert unlock_for_pdfium(pdf, tmp_path / "out") is None


def test_pdf_pdfium_rejects_gets_an_unencrypted_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = _build_pdf(
        tmp_path / "restricted.pdf",
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="",
        permissions=fitz.PDF_PERM_ACCESSIBILITY,
    )
    _pdfium_rejects(pdf, monkeypatch)

    out = unlock_for_pdfium(pdf, tmp_path / "out")

    assert out is not None and out != pdf and out.name == pdf.name
    assert _texts(out) == _texts(pdf) == ["page 1 text", "page 2 text", "page 3 text"]
    with fitz.open(out) as doc:
        assert not doc.is_encrypted
    assert docling_module.pdfium_can_open(out)


def test_pdf_needing_a_user_password_is_not_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = _build_pdf(
        tmp_path / "locked.pdf",
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    _pdfium_rejects(pdf, monkeypatch)
    assert unlock_for_pdfium(pdf, tmp_path / "out") is None
    assert not (tmp_path / "out" / pdf.name).exists()


def test_docling_receives_the_unlocked_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = _build_pdf(tmp_path / "restricted.pdf")
    _pdfium_rejects(pdf, monkeypatch)

    with docling_module._pdf_for_docling(str(pdf)) as convert_path:
        assert Path(convert_path) != pdf
        assert _texts(Path(convert_path)) == _texts(pdf)
        kept = Path(convert_path)
    assert not kept.exists()  # temp copy removed on exit


def test_preflight_reports_a_pdf_the_parser_will_unlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vo_rag.core.parsing import preflight

    pdf = _build_pdf(tmp_path / "restricted.pdf", pages=12)
    _pdfium_rejects(pdf, monkeypatch)

    assert preflight._docling_backend_check(pdf) == "unlocked_for_docling"


def test_preflight_rejects_a_pdf_docling_cannot_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vo_rag.core.parsing import preflight

    pdf = _build_pdf(tmp_path / "broken.pdf", pages=12)
    monkeypatch.setattr(docling_module, "pdfium_can_open", lambda path: False)

    assert preflight._docling_backend_check(pdf) == "unreadable_by_docling"
    result = preflight._inspect_pdf(pdf, "0" * 64, None)
    assert result.verdict == "reject" and "unreadable_by_docling" in result.reasons

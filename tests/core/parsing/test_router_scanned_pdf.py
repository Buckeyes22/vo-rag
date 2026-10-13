"""A non-native PDF is routed to docling's GPU OCR (VOR-1427).

``_pdf_kind`` is monkeypatched so no real PDF is sampled; the fake ``docling``
double stands in for a real ``DoclingParser`` (and its GPU OCR converter),
so this file never touches CUDA or docling.

The "scanned" -> docling.parse_scanned case is covered by
``test_router.py::test_pdf_dispatches_to_docling_ocr_forward_path``; this
file covers the generalisation: *any* non-"native" ``_pdf_kind`` result routes
to ``parse_scanned``, not just the literal string "scanned".
"""

from __future__ import annotations

from pathlib import Path

import pytest

import vo_rag.core.parsing.router as router_module
from tests.core.parsing._router_fakes import (
    FakeDocling as _FakeDocling,
)
from tests.core.parsing._router_fakes import (
    FakeParser as _FakeParser,
)
from vo_rag.core.parsing.router import ParserRouter


def _router() -> ParserRouter:
    return ParserRouter(
        docling=_FakeDocling(),
        epub=_FakeParser("epub"),
    )


@pytest.fixture(autouse=True)
def _stub_magic(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_from_file(path: str, mime: bool = True) -> str:
        _ = (path, mime)
        return "application/octet-stream"

    monkeypatch.setattr(router_module.magic, "from_file", fake_from_file)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_pdf_kind_other_than_native_routes_to_docling_parse_scanned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Any _pdf_kind result other than "native" goes to parse_scanned."""
    router = _router()

    async def fake_pdf_kind(self: ParserRouter, path: str) -> str:
        _ = (self, path)
        return "some-other-non-native-kind"

    monkeypatch.setattr(ParserRouter, "_pdf_kind", fake_pdf_kind)

    path = tmp_path / "x.pdf"
    path.write_bytes(b"%PDF-1.4\n")
    doc = await router.parse(str(path), file_hash="h", mime_type="application/pdf")

    assert doc.parser_used == "docling-ocr"
    fake_docling: _FakeDocling = router.docling  # type: ignore[assignment]
    assert fake_docling.scanned_called is True
    assert fake_docling.called is False  # the native docling.parse() path

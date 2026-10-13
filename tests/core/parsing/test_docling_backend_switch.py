"""VORAG_DOCLING_PDF_BACKEND selects docling's PDF text backend (VOR-1424).

docling is not installed in the light CI suite, so a minimal stub package is
injected into sys.modules; the test asserts which backend class the parser
hands to PdfFormatOption.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest


class _PyPdfiumDocumentBackend:
    pass


class _PdfFormatOption:
    def __init__(self, *, pipeline_options: Any, backend: Any = None) -> None:
        self.pipeline_options = pipeline_options
        self.backend = backend


class _Options:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _Converter:
    def __init__(self, *, format_options: dict[str, Any]) -> None:
        self.format_options = format_options


@pytest.fixture
def fake_docling(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = {
        "docling": types.ModuleType("docling"),
        "docling.datamodel": types.ModuleType("docling.datamodel"),
        "docling.datamodel.base_models": types.SimpleNamespace(
            InputFormat=types.SimpleNamespace(PDF="pdf")
        ),
        "docling.datamodel.pipeline_options": types.SimpleNamespace(PdfPipelineOptions=_Options),
        "docling.datamodel.accelerator_options": types.SimpleNamespace(
            AcceleratorDevice=types.SimpleNamespace(CUDA="cuda", CPU="cpu"),
            AcceleratorOptions=_Options,
        ),
        "docling.document_converter": types.SimpleNamespace(
            DocumentConverter=_Converter, PdfFormatOption=_PdfFormatOption
        ),
        "docling.backend": types.ModuleType("docling.backend"),
        "docling.backend.pypdfium2_backend": types.SimpleNamespace(
            PyPdfiumDocumentBackend=_PyPdfiumDocumentBackend
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _pdf_option(monkeypatch: pytest.MonkeyPatch, backend: str | None) -> _PdfFormatOption:
    from vo_rag.core.parsing.parsers.docling import DoclingParser

    if backend is None:
        monkeypatch.delenv("VORAG_DOCLING_PDF_BACKEND", raising=False)
    else:
        monkeypatch.setenv("VORAG_DOCLING_PDF_BACKEND", backend)
    parser = DoclingParser(figures=object(), do_ocr=False)  # type: ignore[arg-type]
    option: _PdfFormatOption = parser._converter.format_options["pdf"]
    return option


@pytest.mark.usefixtures("fake_docling")
def test_default_keeps_doclings_own_pdf_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _pdf_option(monkeypatch, None).backend is None


@pytest.mark.usefixtures("fake_docling")
@pytest.mark.parametrize("value", ["pypdfium", "PyPdfium2", " pypdfium2 "])
def test_pypdfium_switch_selects_pypdfium2_backend(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    assert _pdf_option(monkeypatch, value).backend is _PyPdfiumDocumentBackend


@pytest.mark.usefixtures("fake_docling")
def test_unknown_value_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _pdf_option(monkeypatch, "bogus").backend is None


class _FakeCuda:
    def __init__(self) -> None:
        self.emptied = 0

    def is_available(self) -> bool:
        return True

    def empty_cache(self) -> None:
        self.emptied += 1


@pytest.mark.usefixtures("fake_docling")
@pytest.mark.parametrize(("device", "expected"), [("cuda", 1), ("cpu", 0)])
def test_cuda_cache_released_after_every_parse_even_on_failure(
    monkeypatch: pytest.MonkeyPatch, device: str, expected: int
) -> None:
    """VOR-1425: long-lived lanes must not hoard VRAM between books."""
    from vo_rag.core.parsing.parsers.docling import DoclingParser

    cuda = _FakeCuda()
    parser = DoclingParser(figures=object(), do_ocr=False, device=device)  # type: ignore[arg-type]

    def boom(_path: str) -> None:
        raise RuntimeError("CUDA out of memory")

    parser._converter = types.SimpleNamespace(convert=boom)
    # The fake torch lives only around the parse: where real torch is
    # installed, pytest-randomly reseeds it in the teardown hook, before
    # fixture finalizers, and a fake still in sys.modules there leaks into
    # every later test.
    with monkeypatch.context() as patch, pytest.raises(RuntimeError):
        patch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
        parser._parse_sync("/tmp/book.pdf", "pdf", "abc")
    assert cuda.emptied == expected

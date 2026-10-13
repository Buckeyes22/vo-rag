"""DoclingParser repairs a pypdfium-backed parse with docling_parse (VOR-1453).

docling is not installed in the light CI suite: a stub package stands in for
it, the converters return docling-shaped documents built from SimpleNamespace
items, and pypdfium2's page text is patched in. The stubs record every
conversion, so the tests also check that docling_parse only ever sees small
page ranges.
"""

from __future__ import annotations

import contextlib
import hashlib
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vo_rag.core.parsing.contract import ElementType, ParsedDocument
from vo_rag.core.storage.figures import FigureStore

DOUBLED = (
    "good runs the rotation is between 1.5 to 11.5 revolutions per "
    "second. (b) This is the only mount with recorded positive as well good runs the rotation i "
    "second. (b) This is the only mount with recorded positive as well as negative rotations "
    "about the long axis. For good runs the s per second with the fron as negative "
    "rotations about rotations have ranged from 4.0 degrees per second with the front "
    "er second with the rotations have rang of the rotor turning up to 7.2 degrees per "
    "second with the ng down."
)
CLEAN = (
    "good runs the rotation is between 1.5 to 11.5 revolutions per second. (b) This is "
    "the only mount with recorded positive as well as negative rotations about the long "
    "axis. For good runs the rotations have ranged from 4.0 degrees per second with the "
    "front of the rotor turning up to 7.2 degrees per second with the front of the "
    "rotor turning down."
)
PAGE_ONE = "The mount of the rotor decides how the release velocity is transferred."
PAGE_THREE = "The best mount keeps the rotor on its long axis through the release."


def _text(text: str, page: int, label: str = "text") -> SimpleNamespace:
    return SimpleNamespace(text=text, label=label, prov=[SimpleNamespace(page_no=page)])


def _document(*items: SimpleNamespace, pages: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        texts=list(items),
        tables=[],
        pictures=[],
        pages=[SimpleNamespace(page_no=page) for page in range(1, pages + 1)],
    )


def _result(document: SimpleNamespace, *, failed: set[int] | None = None) -> SimpleNamespace:
    total = len(document.pages)
    completed = [p for p in range(1, total + 1) if p not in (failed or set())]
    return SimpleNamespace(
        document=document,
        status=SimpleNamespace(value="partial_success" if failed else "success"),
        pages=[SimpleNamespace(page_no=p) for p in completed],
    )


class _Converter:
    """Records calls; ``answer(path, page_range)`` produces the result."""

    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.calls: list[tuple[int, int] | None] = []
        self.initialized_pipelines: dict[Any, Any] = {}

    def convert(self, path: str, page_range: tuple[int, int] | None = None) -> Any:
        self.calls.append(page_range)
        return self.answer(path, page_range)


class _DoclingDocument:
    @staticmethod
    def concatenate(docs: list[SimpleNamespace]) -> SimpleNamespace:
        return SimpleNamespace(
            texts=[item for doc in docs for item in doc.texts],
            tables=[],
            pictures=[],
            pages=[page for doc in docs for page in doc.pages],
        )


class _Options:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _PdfFormatOption:
    def __init__(self, *, pipeline_options: Any, backend: Any = None) -> None:
        self.pipeline_options = pipeline_options
        self.backend = backend


@pytest.fixture
def docling_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = {
        "docling": types.ModuleType("docling"),
        "docling.datamodel": types.ModuleType("docling.datamodel"),
        "docling.datamodel.base_models": SimpleNamespace(InputFormat=SimpleNamespace(PDF="pdf")),
        "docling.datamodel.pipeline_options": SimpleNamespace(PdfPipelineOptions=_Options),
        "docling.datamodel.accelerator_options": SimpleNamespace(
            AcceleratorDevice=SimpleNamespace(CUDA="cuda", CPU="cpu"),
            AcceleratorOptions=_Options,
        ),
        "docling.document_converter": SimpleNamespace(
            DocumentConverter=lambda format_options: _Converter(None),
            PdfFormatOption=_PdfFormatOption,
        ),
        "docling.backend": types.ModuleType("docling.backend"),
        "docling.backend.pypdfium2_backend": SimpleNamespace(PyPdfiumDocumentBackend=object),
        "docling_core": types.ModuleType("docling_core"),
        "docling_core.types": types.ModuleType("docling_core.types"),
        "docling_core.types.doc": SimpleNamespace(DoclingDocument=_DoclingDocument),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("VORAG_DOCLING_PDF_BACKEND", "pypdfium")


def _parser(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    pypdfium: Any,
    docling_parse: Any,
    page_texts: list[str],
    cuda: bool = True,
) -> tuple[Any, _Converter, _Converter]:
    from vo_rag.core.parsing.parsers import docling as docling_module

    @contextlib.contextmanager
    def _same_path(path: str) -> Iterator[str]:
        yield path

    monkeypatch.setattr(docling_module, "_pdf_for_docling", _same_path)
    monkeypatch.setattr(docling_module, "pdfium_page_texts", lambda _path: page_texts)
    monkeypatch.setattr(docling_module, "_cuda_available", lambda: cuda)
    monkeypatch.setattr(docling_module, "_release_cuda_cache", lambda: None)
    parser = docling_module.DoclingParser(FigureStore(tmp_path), do_ocr=False, device="cuda")
    first = _Converter(pypdfium)
    second = _Converter(docling_parse)
    parser._converter = first
    parser._docling_parse_converter = second
    return parser, first, second


def _parse(parser: Any) -> ParsedDocument:
    parsed: ParsedDocument = parser._parse_sync("/books/rotor.pdf", "pdf", "ab" * 32)
    return parsed


def _texts(parsed: ParsedDocument, page: int) -> list[str]:
    return [e.text for e in parsed.elements if e.page_number == page]


PAGE_TEXTS = [PAGE_ONE, CLEAN, PAGE_THREE]


@pytest.mark.usefixtures("docling_stub")
def test_rebuilt_equation_crop_survives_temporary_picture_store_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import fitz  # type: ignore[import-untyped, unused-ignore]

    source = tmp_path / "equation.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page(width=200, height=300)
        page.insert_text((50, 65), "A = numerator / denominator")
        pdf.save(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    item = SimpleNamespace(
        text="",
        orig="A numerator denominator = ---",
        label="formula",
        prov=[
            SimpleNamespace(
                page_no=1,
                bbox=SimpleNamespace(l=45.0, t=50.0, r=190.0, b=80.0, coord_origin="TOPLEFT"),
            )
        ],
    )
    document = _document(item, pages=1)
    document.pages = {1: SimpleNamespace(size=SimpleNamespace(width=200.0, height=300.0))}
    parser, _, _ = _parser(
        monkeypatch,
        tmp_path,
        pypdfium=lambda *_: _result(document),
        docling_parse=lambda *_: _result(document),
        page_texts=["A numerator denominator"],
    )
    rows, error = parser._docling_parse_range(str(source), 1, 1, str(source), "pdf", digest)
    assert error is None and rows is not None
    visual = rows[1][0].metadata["equation_visual"]
    assert visual["status"] == "source_bound"
    path = parser._figures.path_for(visual["image_ref"])
    assert path.exists()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == visual["sha256"]


@pytest.mark.usefixtures("docling_stub")
def test_doubled_page_is_rebuilt_with_docling_parse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pypdfium = _document(
        _text("Mounts", 1, "section_header"),
        _text(PAGE_ONE, 1),
        _text(DOUBLED, 2),
        _text("The best mount", 3, "section_header"),
        _text(PAGE_THREE, 3),
    )
    clean_page_two = _document(_text(CLEAN, 2), pages=1)
    parser, _, docling_parse = _parser(
        monkeypatch,
        tmp_path,
        pypdfium=lambda _p, _r: _result(pypdfium),
        docling_parse=lambda _p, _r: _result(clean_page_two),
        page_texts=PAGE_TEXTS,
    )

    parsed = _parse(parser)

    assert docling_parse.calls == [(2, 2)]
    assert _texts(parsed, 2) == [CLEAN]
    assert _texts(parsed, 1) == ["Mounts", PAGE_ONE]
    assert _texts(parsed, 3) == ["The best mount", PAGE_THREE]
    assert parsed.elements[2].section_title == "Mounts"
    [record] = parsed.metadata["pdf_page_rebuild"]["pages"]
    assert record["page"] == 2
    assert record["reason"] == "doubled_text"
    assert record["rebuilt_with"] == "docling_parse"
    assert record["extra_words"] > 0


@pytest.mark.usefixtures("docling_stub")
def test_rebuild_that_drops_a_line_falls_back_to_pypdfium_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pypdfium = _document(_text(PAGE_ONE, 1), _text(DOUBLED, 2), _text(PAGE_THREE, 3))
    lossy = _document(_text(CLEAN.replace("as negative rotations", "negative"), 2), pages=1)
    parser, _, _ = _parser(
        monkeypatch,
        tmp_path,
        pypdfium=lambda _p, _r: _result(pypdfium),
        docling_parse=lambda _p, _r: _result(lossy),
        page_texts=[PAGE_ONE, CLEAN.replace(". ", ".\n"), PAGE_THREE],
    )

    parsed = _parse(parser)

    [element] = [e for e in parsed.elements if e.page_number == 2]
    assert element.element_type == ElementType.NARRATIVE_TEXT
    assert "as negative rotations" in element.text
    [record] = parsed.metadata["pdf_page_rebuild"]["pages"]
    assert record["rebuilt_with"] == "pypdfium2_text"
    assert record["docling_parse_missing_words"] == ["as", "rotations"]


@pytest.mark.usefixtures("docling_stub")
def test_rebuild_covers_a_paragraph_running_over_the_page_break(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    straddling = SimpleNamespace(
        text=PAGE_ONE + " " + "It continues here.",
        label="text",
        prov=[SimpleNamespace(page_no=1), SimpleNamespace(page_no=2)],
    )
    pypdfium = _document(straddling, _text(DOUBLED, 2), _text(PAGE_THREE, 3))
    rebuilt = _document(_text(PAGE_ONE, 1), _text("It continues here. " + CLEAN, 2), pages=2)
    parser, _, docling_parse = _parser(
        monkeypatch,
        tmp_path,
        pypdfium=lambda _p, _r: _result(pypdfium),
        docling_parse=lambda _p, _r: _result(rebuilt),
        page_texts=[PAGE_ONE, "It continues here. " + CLEAN, PAGE_THREE],
    )

    parsed = _parse(parser)

    # Page 1 holds the start of the page-2 paragraph, so both are rebuilt:
    # replacing page 2 alone would drop "It continues here." or repeat it.
    assert docling_parse.calls == [(1, 2)]
    assert [e.text for e in parsed.elements] == [
        PAGE_ONE,
        "It continues here. " + CLEAN,
        PAGE_THREE,
    ]
    reasons = {r["page"]: r["reason"] for r in parsed.metadata["pdf_page_rebuild"]["pages"]}
    assert reasons == {1: "shares_a_paragraph_with_a_rebuilt_page", 2: "doubled_text"}


@pytest.mark.usefixtures("docling_stub")
def test_page_docling_failed_on_is_rebuilt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # e.g. CUDA out of memory in the layout stage: docling keeps the pages
    # that completed and reports a partial success.
    pypdfium = _document(_text(PAGE_ONE, 1), _text(PAGE_THREE, 3))
    parser, _, docling_parse = _parser(
        monkeypatch,
        tmp_path,
        pypdfium=lambda _p, _r: _result(pypdfium, failed={2}),
        docling_parse=lambda _p, _r: _result(_document(_text(CLEAN, 2), pages=1)),
        page_texts=PAGE_TEXTS,
    )

    parsed = _parse(parser)

    assert docling_parse.calls == [(2, 2)]
    assert [e.text for e in parsed.elements] == [PAGE_ONE, CLEAN, PAGE_THREE]
    [record] = parsed.metadata["pdf_page_rebuild"]["pages"]
    assert record["reason"] == "docling_page_failed"


@pytest.mark.usefixtures("docling_stub")
def test_without_cuda_doubled_page_gets_pypdfium_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pypdfium = _document(_text(PAGE_ONE, 1), _text(DOUBLED, 2), _text(PAGE_THREE, 3))
    parser, _, docling_parse = _parser(
        monkeypatch,
        tmp_path,
        pypdfium=lambda _p, _r: _result(pypdfium),
        docling_parse=lambda _p, _r: pytest.fail("docling_parse must not run without CUDA"),
        page_texts=PAGE_TEXTS,
        cuda=False,
    )

    parsed = _parse(parser)

    assert docling_parse.calls == []
    assert _texts(parsed, 2) == [CLEAN]
    [record] = parsed.metadata["pdf_page_rebuild"]["pages"]
    assert record["rebuilt_with"] == "pypdfium2_text"
    assert record["error"] == "skipped: no CUDA device"


def _book(pages: int, *, content_pages: set[int]) -> SimpleNamespace:
    return _document(
        *(
            _text(f"Section {page} text of the sprint training book.", page)
            for page in sorted(content_pages)
        ),
        pages=pages,
    )


def _page_texts(pages: int) -> list[str]:
    return [f"Section {page} text of the sprint training book." for page in range(1, pages + 1)]


@pytest.mark.usefixtures("docling_stub")
def test_thin_parse_is_redone_whole_in_page_ranges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # "covered content on only 1/26 pages": the whole book goes through
    # docling_parse, 8 pages per conversion, never at once.
    def docling_parse(_path: str, page_range: tuple[int, int]) -> SimpleNamespace:
        first, last = page_range
        pages = range(first, last + 1)
        return SimpleNamespace(
            document=SimpleNamespace(
                texts=[
                    _text(f"Section {page} text of the sprint training book.", page)
                    for page in pages
                ],
                tables=[],
                pictures=[],
                pages=[SimpleNamespace(page_no=page) for page in pages],
            ),
            status=SimpleNamespace(value="success"),
            pages=[SimpleNamespace(page_no=page) for page in pages],
        )

    parser, _, converter = _parser(
        monkeypatch,
        tmp_path,
        pypdfium=lambda _p, _r: _result(_book(26, content_pages={1})),
        docling_parse=docling_parse,
        page_texts=_page_texts(26),
    )

    parsed = _parse(parser)

    assert converter.calls == [(1, 8), (9, 16), (17, 24), (25, 26)]
    assert {e.page_number for e in parsed.elements} == set(range(1, 27))
    fallback = parsed.metadata["pdf_backend_fallback"]
    assert fallback["used"] is True
    assert fallback["reason"] == "page_coverage_gate"
    assert (fallback["content_pages_before"], fallback["content_pages_after"]) == (1, 26)
    assert "pdf_page_rebuild" not in parsed.metadata


@pytest.mark.usefixtures("docling_stub")
def test_failed_whole_document_retry_keeps_the_original_parse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(_path: str, _page_range: tuple[int, int]) -> None:
        raise RuntimeError("CUDA out of memory")

    parser, _, _ = _parser(
        monkeypatch,
        tmp_path,
        pypdfium=lambda _p, _r: _result(_book(26, content_pages={1})),
        docling_parse=boom,
        page_texts=_page_texts(26),
    )

    parsed = _parse(parser)

    # The parse step's page-coverage gate then fails the book, as before.
    assert {e.page_number for e in parsed.elements} == {1}
    fallback = parsed.metadata["pdf_backend_fallback"]
    assert fallback["used"] is False
    assert fallback["error"] == "RuntimeError: CUDA out of memory"


@pytest.mark.usefixtures("docling_stub")
def test_default_backend_is_not_repaired(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VORAG_DOCLING_PDF_BACKEND")
    pypdfium = _document(_text(PAGE_ONE, 1), _text(DOUBLED, 2), _text(PAGE_THREE, 3))
    parser, _, docling_parse = _parser(
        monkeypatch,
        tmp_path,
        pypdfium=lambda _p, _r: _result(pypdfium),
        docling_parse=lambda _p, _r: pytest.fail("docling_parse is the default backend here"),
        page_texts=PAGE_TEXTS,
    )

    parsed = _parse(parser)

    assert _texts(parsed, 2) == [DOUBLED]
    assert "pdf_page_rebuild" not in parsed.metadata


@pytest.mark.usefixtures("docling_stub")
def test_docling_parse_converter_shares_the_loaded_models(tmp_path: Path) -> None:
    from vo_rag.core.parsing.parsers.docling import DoclingParser

    parser = DoclingParser(FigureStore(tmp_path), do_ocr=False, device="cuda")
    converter = parser._get_docling_parse_converter()

    assert converter.initialized_pipelines is parser._converter.initialized_pipelines

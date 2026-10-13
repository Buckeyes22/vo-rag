"""OCR must never run on the CPU (VOR-1427).

The scanned-PDF / image-only-EPUB OCR path uses docling's RapidOCR on the
``torch`` backend, forced onto CUDA. docling's default OCR backend
(``onnxruntime``) silently falls back to the CPU because only CPU
``onnxruntime`` is installed here -- so this suite also asserts the string
``onnxruntime`` never appears anywhere under ``src/vo_rag/core/parsing``.

docling and torch are never imported for real: a minimal fake docling
package is injected into ``sys.modules`` (mirrors
``test_docling_backend_switch.py``), and ``torch.cuda.is_available`` is
monkeypatched. No model is ever loaded, on CPU or GPU.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest


class _Options:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _PdfFormatOption:
    def __init__(self, *, pipeline_options: Any, backend: Any = None) -> None:
        self.pipeline_options = pipeline_options
        self.backend = backend


class _Converter:
    def __init__(self, *, format_options: dict[str, Any]) -> None:
        self.format_options = format_options
        self.convert_calls: list[str] = []

    def convert(self, file_path: str) -> Any:
        self.convert_calls.append(file_path)
        raise AssertionError("convert() must never be called in this suite")


@contextlib.contextmanager
def _fake_torch(monkeypatch: pytest.MonkeyPatch, cuda: Any) -> Iterator[None]:
    """Install a fake ``torch`` only for the code that needs it.

    Never for the whole test: where real torch is installed, pytest-randomly
    reseeds it (via thinc -> ``torch.manual_seed``) in the teardown hook,
    *before* fixture finalizers run. With a fake still in ``sys.modules`` that
    reseed raises, the monkeypatch finalizer never runs, and the fake leaks
    into every later test.
    """
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
        yield


@pytest.fixture
def fake_docling(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = {
        "docling": types.ModuleType("docling"),
        "docling.datamodel": types.ModuleType("docling.datamodel"),
        "docling.datamodel.base_models": types.SimpleNamespace(
            InputFormat=types.SimpleNamespace(PDF="pdf")
        ),
        "docling.datamodel.pipeline_options": types.SimpleNamespace(
            PdfPipelineOptions=_Options, RapidOcrOptions=_Options
        ),
        "docling.datamodel.accelerator_options": types.SimpleNamespace(
            AcceleratorDevice=types.SimpleNamespace(CUDA="cuda", CPU="cpu", AUTO="auto"),
            AcceleratorOptions=_Options,
        ),
        "docling.document_converter": types.SimpleNamespace(
            DocumentConverter=_Converter, PdfFormatOption=_PdfFormatOption
        ),
        "docling.backend": types.ModuleType("docling.backend"),
        "docling.backend.pypdfium2_backend": types.SimpleNamespace(
            PyPdfiumDocumentBackend=type("PyPdfiumDocumentBackend", (), {})
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


# ---------------------------------------------------------------------------
# The normal (non-OCR) converter is unchanged.
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("fake_docling")
def test_normal_converter_still_has_ocr_disabled() -> None:
    from vo_rag.core.parsing.parsers.docling import DoclingParser

    parser = DoclingParser(figures=object(), do_ocr=False)  # type: ignore[arg-type]
    option: _PdfFormatOption = parser._converter.format_options["pdf"]
    assert option.pipeline_options.do_ocr is False


# ---------------------------------------------------------------------------
# scanned_pipeline_options()
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("fake_docling")
def test_scanned_pipeline_options_enables_torch_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VORAG_DOCLING_DEVICE", raising=False)
    from vo_rag.core.parsing.parsers.docling import scanned_pipeline_options

    options = scanned_pipeline_options()

    assert options.do_ocr is True
    assert options.ocr_options.backend == "torch"
    assert options.ocr_options.force_full_page_ocr is True
    assert options.accelerator_options.device == "cuda"
    assert options.generate_picture_images is True


@pytest.mark.usefixtures("fake_docling")
def test_scanned_pipeline_options_selects_specific_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VORAG_DOCLING_DEVICE", "cuda:1")
    from vo_rag.core.parsing.parsers.docling import scanned_pipeline_options

    assert scanned_pipeline_options().accelerator_options.device == "cuda:1"


@pytest.mark.usefixtures("fake_docling")
def test_scanned_pipeline_options_cpu_env_value_never_produces_cpu_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VORAG_DOCLING_DEVICE=cpu must not select a CPU device for OCR."""
    monkeypatch.setenv("VORAG_DOCLING_DEVICE", "cpu")
    from vo_rag.core.parsing.parsers.docling import scanned_pipeline_options

    device = scanned_pipeline_options().accelerator_options.device
    assert device == "cuda"
    assert device != "cpu"


@pytest.mark.usefixtures("fake_docling")
def test_scanned_pipeline_options_returns_a_fresh_object_per_call() -> None:
    from vo_rag.core.parsing.parsers.docling import scanned_pipeline_options

    first = scanned_pipeline_options()
    second = scanned_pipeline_options()

    assert first is not second
    assert first.ocr_options is not second.ocr_options
    assert first.accelerator_options is not second.accelerator_options


# ---------------------------------------------------------------------------
# parse_scanned() refuses to run without CUDA
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("fake_docling")
@pytest.mark.asyncio
async def test_parse_scanned_raises_without_cuda(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vo_rag.core.parsing.parsers.docling import DoclingParser, OcrRequiresGpu

    class _NoCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    parser = DoclingParser(figures=object(), do_ocr=False)  # type: ignore[arg-type]
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    with _fake_torch(monkeypatch, _NoCuda()), pytest.raises(OcrRequiresGpu):
        await parser.parse_scanned(str(pdf), file_hash="abc123")

    assert parser._scanned_converter is None


@pytest.mark.usefixtures("fake_docling")
@pytest.mark.asyncio
async def test_parse_scanned_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Exercise parse_scanned's success path."""
    from vo_rag.core.parsing.parsers.docling import DoclingParser

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

    parser = DoclingParser(figures=object(), do_ocr=False)  # type: ignore[arg-type]

    # Swap the scanned-converter's DocumentConverter *after* __init__ already
    # built the normal converter with the fixture's plain _Converter, so this
    # counts only scanned-converter builds.
    build_count = 0
    seen_pipeline_options: list[Any] = []

    class _StubDocument:
        texts: list[Any] = []
        tables: list[Any] = []
        pictures: list[Any] = []

    class _StubResult:
        document = _StubDocument()

    class _CountingConverter:
        def __init__(self, *, format_options: dict[str, Any]) -> None:
            nonlocal build_count
            build_count += 1
            self.format_options = format_options
            seen_pipeline_options.append(format_options["pdf"].pipeline_options)

        def convert(self, file_path: str) -> Any:
            _ = file_path
            return _StubResult()

    monkeypatch.setattr(
        sys.modules["docling.document_converter"], "DocumentConverter", _CountingConverter
    )

    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    with _fake_torch(monkeypatch, _Cuda()):
        doc1 = await parser.parse_scanned(str(pdf), file_hash="hash-1")
        doc2 = await parser.parse_scanned(str(pdf), file_hash="hash-2")

    assert doc1.parser_used == "docling-ocr"
    assert doc1.document_id == "hash-1"  # src_hash=file_hash reaches the mapping
    assert doc2.document_id == "hash-2"
    assert build_count == 1  # converter built once, reused on the second call
    scanned_converter = parser._scanned_converter
    assert scanned_converter is not None
    assert seen_pipeline_options[0] is scanned_converter.format_options["pdf"].pipeline_options
    assert seen_pipeline_options[0].do_ocr is True
    assert seen_pipeline_options[0].ocr_options.backend == "torch"
    # docling_parse committed ~1.7 GiB per page transiently (37 GiB peak over
    # 20 scanned pages on win vs 5.3 GiB with pypdfium, identical output).
    backend = scanned_converter.format_options["pdf"].backend
    assert backend is sys.modules["docling.backend.pypdfium2_backend"].PyPdfiumDocumentBackend


@pytest.mark.usefixtures("fake_docling")
def test_get_scanned_converter_thread_safe_builds_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent first calls must build one converter."""
    from vo_rag.core.parsing.parsers.docling import DoclingParser

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

    build_count = 0
    build_count_lock = threading.Lock()
    thread_count = 8

    class _SlowConverter:
        def __init__(self, *, format_options: dict[str, Any]) -> None:
            nonlocal build_count
            # Widen the race window so concurrent callers really overlap.
            threading.Event().wait(0.05)
            with build_count_lock:
                build_count += 1
            self.format_options = format_options

    # Build the normal (do_ocr=False) converter with the fixture's plain
    # _Converter *before* swapping in _SlowConverter, so build_count only
    # ever counts scanned-converter builds.
    parser = DoclingParser(figures=object(), do_ocr=False)  # type: ignore[arg-type]
    monkeypatch.setattr(
        sys.modules["docling.document_converter"], "DocumentConverter", _SlowConverter
    )
    barrier = threading.Barrier(thread_count)

    def _call() -> None:
        barrier.wait()
        parser._get_scanned_converter()

    threads = [threading.Thread(target=_call) for _ in range(thread_count)]
    with _fake_torch(monkeypatch, _Cuda()):
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert build_count == 1


@pytest.mark.usefixtures("fake_docling")
def test_require_cuda_runs_lazily_not_at_init_or_module_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """__init__ must keep working on hosts without CUDA; _require_cuda() only
    runs the first time the scanned converter is actually built."""

    class _NoCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    from vo_rag.core.parsing.parsers.docling import DoclingParser

    # Construction alone must not raise, even though "torch" reports no CUDA.
    with _fake_torch(monkeypatch, _NoCuda()):
        parser = DoclingParser(figures=object(), do_ocr=False)  # type: ignore[arg-type]
    assert parser._scanned_converter is None


# ---------------------------------------------------------------------------
# never CPU OCR: grep-style assertion
# ---------------------------------------------------------------------------


def test_onnxruntime_backend_never_appears_under_parsing() -> None:
    parsing_dir = Path(__file__).resolve().parents[3] / "src" / "vo_rag" / "core" / "parsing"
    assert parsing_dir.is_dir()
    offenders = [
        path
        for path in parsing_dir.rglob("*.py")
        if "onnxruntime" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []

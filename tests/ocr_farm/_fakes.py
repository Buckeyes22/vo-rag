"""Shared OCR-farm test doubles (not a test module: no ``test_`` prefix).

- ``install_fake_docling`` injects a minimal fake ``docling`` / ``docling_core``
  package into ``sys.modules`` (mirrors ``tests/core/parsing/test_ocr_gpu_only.py``)
  so option-building and worker code run in the locked CI env, which has no
  docling. No model is ever loaded.
- ``load_worker`` imports the standalone worker script by file path, exactly
  as it runs on a GPU host (it is never installed as part of vo-rag there).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

WORKER_DIR = Path(__file__).resolve().parents[2] / "src" / "vo_rag" / "ocr_farm" / "worker"
WORKER_PATH = WORKER_DIR / "ocr_worker.py"
REQUIREMENTS_PATH = WORKER_DIR / "requirements.txt"


class Options:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class PdfFormatOption:
    def __init__(self, *, pipeline_options: Any, backend: Any = None) -> None:
        self.pipeline_options = pipeline_options
        self.backend = backend


class FakeSize:
    def __init__(self, *, width: float, height: float) -> None:
        self.width = width
        self.height = height


class FakeDoclingDocument:
    """``pages`` / ``add_page`` / ``save_as_json`` / ``concatenate`` of a DoclingDocument."""

    def __init__(self, name: str = "book", pages: dict[int, Any] | None = None) -> None:
        self.name = name
        self.pages: dict[int, Any] = dict(pages or {})
        self.saved: list[tuple[Path, Any]] = []

    def add_page(self, *, page_no: int, size: Any) -> None:
        self.pages[page_no] = size

    def save_as_json(self, filename: Path, image_mode: Any = None) -> None:
        self.saved.append((Path(filename), image_mode))
        Path(filename).write_text(
            json.dumps({"schema_name": "DoclingDocument", "pages": sorted(self.pages)}),
            encoding="utf-8",
        )

    @classmethod
    def concatenate(cls, docs: list[Any]) -> FakeDoclingDocument:
        """docling-core 2.78's page rule: each document's pages shift by
        ``max page so far - its first page + 1`` (0 when batches are contiguous)."""
        merged = cls(name=" + ".join(doc.name for doc in docs))
        max_page = 0
        for doc in docs:
            if not doc.pages:
                continue
            delta = max_page - min(doc.pages) + 1
            for page_no, page in doc.pages.items():
                merged.pages[page_no + delta] = page
            max_page = max(page_no + delta for page_no in doc.pages)
        return merged


def fake_pdfium(page_count: int) -> Any:
    class _Page:
        def get_size(self) -> tuple[float, float]:
            return (612.0, 792.0)

    class _PdfDocument:
        def __init__(self, path: str) -> None:
            _ = path

        def __len__(self) -> int:
            return page_count

        def __getitem__(self, index: int) -> _Page:
            return _Page()

        def get_page_size(self, index: int) -> tuple[float, float]:
            return _Page().get_size()

        def close(self) -> None:
            return None

    return types.SimpleNamespace(PdfDocument=_PdfDocument)


def install_fake_docling(
    monkeypatch: pytest.MonkeyPatch, converter_cls: Any = None, *, page_count: int = 1
) -> None:
    class _NeverConverter:
        def __init__(self, *, format_options: dict[str, Any]) -> None:
            raise AssertionError("DocumentConverter must not be built in this test")

    modules: dict[str, Any] = {
        "docling": types.ModuleType("docling"),
        "docling.datamodel": types.ModuleType("docling.datamodel"),
        "docling.datamodel.base_models": types.SimpleNamespace(
            InputFormat=types.SimpleNamespace(PDF="pdf"),
            ConversionStatus=types.SimpleNamespace(
                SUCCESS="success", PARTIAL_SUCCESS="partial_success", FAILURE="failure"
            ),
        ),
        "docling.datamodel.pipeline_options": types.SimpleNamespace(
            PdfPipelineOptions=Options, RapidOcrOptions=Options
        ),
        "docling.datamodel.accelerator_options": types.SimpleNamespace(
            AcceleratorDevice=types.SimpleNamespace(CUDA="cuda", CPU="cpu", AUTO="auto"),
            AcceleratorOptions=Options,
        ),
        "docling.document_converter": types.SimpleNamespace(
            DocumentConverter=converter_cls or _NeverConverter, PdfFormatOption=PdfFormatOption
        ),
        "docling_core": types.ModuleType("docling_core"),
        "docling_core.types": types.ModuleType("docling_core.types"),
        "docling_core.types.doc": types.SimpleNamespace(
            ImageRefMode=types.SimpleNamespace(
                EMBEDDED="embedded", PLACEHOLDER="placeholder", REFERENCED="referenced"
            ),
            DoclingDocument=FakeDoclingDocument,
            Size=FakeSize,
        ),
        "pypdfium2": fake_pdfium(page_count),
        "docling.backend": types.ModuleType("docling.backend"),
        "docling.backend.pypdfium2_backend": types.SimpleNamespace(
            PyPdfiumDocumentBackend=type("PyPdfiumDocumentBackend", (), {})
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def fake_torch(
    *,
    available: bool,
    peak_bytes: int = 1024,
    device_name: str = "NVIDIA GeForce RTX 5070",
    peaks: dict[int, int] | None = None,
) -> Any:
    """``peaks`` maps device index -> peak bytes (default: ``{0: peak_bytes}``);
    ``device_count()`` is one more than the highest index."""
    per_device = dict(peaks) if peaks is not None else {0: peak_bytes}

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return available

        @staticmethod
        def device_count() -> int:
            return max(per_device, default=-1) + 1

        @staticmethod
        def max_memory_allocated(device: Any = None) -> int:
            if device is None:
                raise AssertionError("pass an explicit device index")
            return per_device.get(int(device), 0)

        @staticmethod
        def empty_cache() -> None:
            return None

        memory_fractions: dict[int, float] = {}

        @staticmethod
        def set_per_process_memory_fraction(fraction: float, device: int) -> None:
            _Cuda.memory_fractions[int(device)] = fraction

        @staticmethod
        def get_device_name(device: Any = None) -> str:
            _ = device
            return device_name

    return types.SimpleNamespace(__version__="2.12.0+cu130", cuda=_Cuda())


def load_worker() -> Any:
    spec = importlib.util.spec_from_file_location("vorag_ocr_worker_under_test", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

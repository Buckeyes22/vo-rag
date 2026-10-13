"""The standalone OCR worker's exit contract (VOR-1427).

torch and docling are faked in ``sys.modules``; nothing is loaded on any GPU
or CPU. The worker must:

- exit non-zero when CUDA is unavailable, before building a converter;
- exit non-zero when ``torch.cuda.max_memory_allocated() == 0`` after the
  conversion (the GPU guard) and leave no output file;
- write ``result.document.save_as_json(out, image_mode=ImageRefMode.EMBEDDED)``
  on success;
- print ``torch <ver> cuda True <gpu>`` for ``--check``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.ocr_farm._fakes import (
    WORKER_PATH,
    FakeDoclingDocument,
    FakeSize,
    fake_torch,
    install_fake_docling,
    load_worker,
)


class _Result:
    def __init__(self, status: str = "success") -> None:
        self.document = FakeDoclingDocument(pages={1: FakeSize(width=612.0, height=792.0)})
        self.status = status


def _converter_cls(result: _Result, built: list[Any]) -> Any:
    class _Converter:
        def __init__(self, *, format_options: dict[str, Any]) -> None:
            self.format_options = format_options
            built.append(self)

        def convert(self, source: str, page_range: tuple[int, int] = (1, 1)) -> _Result:
            _ = source, page_range
            return result

    return _Converter


def _main(monkeypatch: pytest.MonkeyPatch, worker: Any, torch: Any, argv: list[str]) -> int:
    """Run ``worker.main`` with a fake ``torch`` visible only during the call.

    Scoped this tightly on purpose: where real torch is installed,
    pytest-randomly reseeds it between test phases, and real torch re-imports
    ``torch.cuda`` from ``sys.modules`` while doing so.
    """
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "torch", torch)
        return int(worker.main(argv))


def _pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    return pdf


def test_exits_nonzero_without_cuda_before_building_converter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built: list[Any] = []
    install_fake_docling(monkeypatch, _converter_cls(_Result(), built))
    torch = fake_torch(available=False)
    worker = load_worker()
    out = tmp_path / "out.json"

    code = _main(monkeypatch, worker, torch, [str(_pdf(tmp_path)), str(out)])

    assert code != 0
    assert built == []
    assert not out.exists()


def test_exits_nonzero_when_gpu_memory_was_never_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built: list[Any] = []
    result = _Result()
    install_fake_docling(monkeypatch, _converter_cls(result, built))
    torch = fake_torch(available=True, peak_bytes=0)
    worker = load_worker()
    out = tmp_path / "out.json"

    code = _main(monkeypatch, worker, torch, [str(_pdf(tmp_path)), str(out)])

    assert code != 0
    assert len(built) == 1
    assert not out.exists()
    assert result.document.saved == []


def test_writes_embedded_json_on_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    built: list[Any] = []
    result = _Result()
    install_fake_docling(monkeypatch, _converter_cls(result, built))
    torch = fake_torch(available=True, peak_bytes=4096)
    worker = load_worker()
    out = tmp_path / "out.json"

    code = _main(monkeypatch, worker, torch, [str(_pdf(tmp_path)), str(out)])

    assert code == 0
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["schema_name"] == "DoclingDocument"
    assert len(result.document.saved) == 1
    _, image_mode = result.document.saved[0]
    assert image_mode == "embedded"
    option = built[0].format_options["pdf"]
    pdfium = sys.modules["docling.backend.pypdfium2_backend"].PyPdfiumDocumentBackend
    assert option.backend is pdfium
    options = option.pipeline_options
    assert options.do_ocr is True
    assert options.ocr_options.backend == "torch"
    assert options.accelerator_options.device == "cuda"
    # No partial file left behind.
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".part")] == []


def test_exits_nonzero_on_failed_conversion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built: list[Any] = []
    install_fake_docling(monkeypatch, _converter_cls(_Result(status="failure"), built))
    torch = fake_torch(available=True)
    worker = load_worker()
    out = tmp_path / "out.json"

    assert _main(monkeypatch, worker, torch, [str(_pdf(tmp_path)), str(out)]) != 0
    assert not out.exists()


def test_check_prints_torch_cuda_and_gpu(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_fake_docling(monkeypatch)
    torch = fake_torch(available=True)
    worker = load_worker()

    assert _main(monkeypatch, worker, torch, ["--check"]) == 0
    assert capsys.readouterr().out.strip() == "torch 2.12.0+cu130 cuda True NVIDIA GeForce RTX 5070"


def test_check_fails_without_cuda(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_fake_docling(monkeypatch)
    torch = fake_torch(available=False)
    worker = load_worker()

    assert _main(monkeypatch, worker, torch, ["--check"]) != 0
    assert "cuda False" in capsys.readouterr().out


def test_gpu_guard_sums_peak_across_all_devices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Memory used only on device 1 (e.g. ``VORAG_DOCLING_DEVICE=cuda:1``) still counts."""
    built: list[Any] = []
    result = _Result()
    install_fake_docling(monkeypatch, _converter_cls(result, built))
    torch = fake_torch(available=True, peaks={0: 0, 1: 8192})
    worker = load_worker()
    out = tmp_path / "out.json"

    code = _main(monkeypatch, worker, torch, [str(_pdf(tmp_path)), str(out)])

    assert code == 0
    assert out.exists()


@pytest.mark.parametrize(
    ("available", "peak", "status", "expected"),
    [
        (True, 4096, "success", 0),
        (True, 0, "success", 4),
        (True, 4096, "failure", 5),
        (False, 4096, "success", 69),
    ],
    ids=["ok", "gpu_guard", "conversion_failed", "no_cuda"],
)
def test_pid_file_exists_while_running_and_is_removed_on_every_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    available: bool,
    peak: int,
    status: str,
    expected: int,
) -> None:
    """The coordinator reads ``jobs/<sha>.pid`` to kill a timed-out worker; a
    stale file after exit could point at a recycled PID, so it must go."""
    sha = "a" * 64
    pid_file = tmp_path / f"{sha}.pid"
    seen: list[str] = []
    result = _Result(status=status)

    class _Converter:
        def __init__(self, *, format_options: dict[str, Any]) -> None:
            self.format_options = format_options

        def convert(self, source: str, page_range: tuple[int, int] = (1, 1)) -> _Result:
            _ = source, page_range
            seen.append(pid_file.read_text(encoding="utf-8"))
            return result

    install_fake_docling(monkeypatch, _Converter)
    torch = fake_torch(available=available, peak_bytes=peak)
    worker = load_worker()

    code = _main(monkeypatch, worker, torch, [str(_pdf(tmp_path)), str(tmp_path / f"{sha}.json")])

    assert code == expected
    if available:
        assert seen == [str(os.getpid())]
    assert not pid_file.exists()


def test_exit_code_docs_match_the_constants() -> None:
    worker_source = WORKER_PATH.read_text(encoding="utf-8")
    worker = load_worker()
    assert (worker.EXIT_USAGE, worker.EXIT_NO_CUDA, worker.EXIT_GPU_GUARD) == (2, 69, 4)
    assert worker.EXIT_CONVERSION_FAILED == 5
    assert "- 69 (``EX_UNAVAILABLE``): CUDA unavailable" in worker_source
    assert "- 4: GPU guard" in worker_source


# ---------------------------------------------------------------------------
# Page batches: memory stays bounded and page numbers survive the merge.
# ---------------------------------------------------------------------------


class _BatchConverter:
    """Returns one document per ``page_range``; ``drop`` pages vanish, as when
    docling's preprocess fails a page; ``fail_at`` gives that batch FAILURE."""

    calls: list[tuple[int, int]] = []

    def __init__(self, *, format_options: dict[str, Any]) -> None:
        self.format_options = format_options

    drop: frozenset[int] = frozenset()
    fail_at: int | None = None

    def convert(self, source: str, page_range: tuple[int, int] = (1, 1)) -> _Result:
        _ = source
        type(self).calls.append(page_range)
        first, last = page_range
        result = _Result(status="failure" if first == type(self).fail_at else "success")
        result.document = FakeDoclingDocument(
            pages={
                n: FakeSize(width=612.0, height=792.0)
                for n in range(first, last + 1)
                if n not in type(self).drop
            }
        )
        return result


def _batch_converter(*, drop: set[int] | None = None, fail_at: int | None = None) -> Any:
    return type(
        "_Converter",
        (_BatchConverter,),
        {"calls": [], "drop": frozenset(drop or ()), "fail_at": fail_at},
    )


def test_page_batches_cover_every_page_once() -> None:
    worker = load_worker()
    assert worker._page_batches(1) == [(1, 1)]
    assert worker._page_batches(100) == [(1, 100)]
    assert worker._page_batches(250) == [(1, 100), (101, 200), (201, 250)]
    assert worker._page_batches(250, size=120) == [(1, 120), (121, 240), (241, 250)]


def test_large_book_is_converted_in_batches_with_original_page_numbers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    converter = _batch_converter()
    install_fake_docling(monkeypatch, converter, page_count=250)
    torch = fake_torch(available=True, peak_bytes=4096)
    worker = load_worker()
    out = tmp_path / "out.json"

    code = _main(monkeypatch, worker, torch, [str(_pdf(tmp_path)), str(out)])

    assert code == 0
    assert converter.calls == [(1, 100), (101, 200), (201, 250)]
    assert json.loads(out.read_text(encoding="utf-8"))["pages"] == list(range(1, 251))


def test_pages_docling_dropped_are_refilled_so_numbers_do_not_shift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without the refill, batch 2 missing page 101 would shift pages 102-200
    down by one in the concatenation (and page 200 missing would shift batch 3)."""
    converter = _batch_converter(drop={101, 150, 200})
    install_fake_docling(monkeypatch, converter, page_count=250)
    torch = fake_torch(available=True, peak_bytes=4096)
    worker = load_worker()
    out = tmp_path / "out.json"

    code = _main(monkeypatch, worker, torch, [str(_pdf(tmp_path)), str(out)])

    assert code == 0
    assert json.loads(out.read_text(encoding="utf-8"))["pages"] == list(range(1, 251))


def test_a_failed_batch_fails_the_book_without_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    converter = _batch_converter(fail_at=101)
    install_fake_docling(monkeypatch, converter, page_count=250)
    torch = fake_torch(available=True, peak_bytes=4096)
    worker = load_worker()
    out = tmp_path / "out.json"

    code = _main(monkeypatch, worker, torch, [str(_pdf(tmp_path)), str(out)])

    assert code == 5
    assert converter.calls == [(1, 100), (101, 200)]
    assert "pages=101-200" in capsys.readouterr().err
    assert not out.exists()


def test_refill_and_concatenate_keep_page_numbers_with_real_docling_core() -> None:
    """The real docling-core concatenate: contiguous (refilled) batches keep numbering."""
    doc_mod = pytest.importorskip("docling_core.types.doc")
    worker = load_worker()
    sizes = [(612.0, 792.0)] * 250
    batches = []
    for first, last in worker._page_batches(250):
        document = doc_mod.DoclingDocument(name="book")
        for page_no in range(first, last + 1):
            if page_no not in {101, 150, 200}:
                document.add_page(page_no=page_no, size=doc_mod.Size(width=612, height=792))
        worker._fill_missing_pages(document, first, last, sizes, doc_mod.Size)
        batches.append(document)

    merged = doc_mod.DoclingDocument.concatenate(batches)

    assert sorted(merged.pages) == list(range(1, 251))


# ---------------------------------------------------------------------------
# GPU memory cap (lin shares its T1000 with whisper).
# ---------------------------------------------------------------------------


def test_gpu_memory_cap_is_applied_to_every_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_fake_docling(monkeypatch, _converter_cls(_Result(), []))
    torch = fake_torch(available=True, peaks={0: 4096, 1: 4096})
    type(torch.cuda).memory_fractions.clear()
    monkeypatch.setenv("VORAG_OCR_GPU_MEM_FRACTION", "0.25")
    worker = load_worker()

    code = _main(monkeypatch, worker, torch, [str(_pdf(tmp_path)), str(tmp_path / "out.json")])

    assert code == 0
    assert type(torch.cuda).memory_fractions == {0: 0.25, 1: 0.25}


def test_no_cap_without_the_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_fake_docling(monkeypatch, _converter_cls(_Result(), []))
    torch = fake_torch(available=True)
    type(torch.cuda).memory_fractions.clear()
    monkeypatch.delenv("VORAG_OCR_GPU_MEM_FRACTION", raising=False)
    worker = load_worker()

    assert _main(monkeypatch, worker, torch, [str(_pdf(tmp_path)), str(tmp_path / "o.json")]) == 0
    assert type(torch.cuda).memory_fractions == {}


@pytest.mark.parametrize("raw", ["0", "1.5", "-0.2", "half"])
def test_bad_gpu_memory_cap_is_a_usage_error_before_converting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: str
) -> None:
    built: list[Any] = []
    install_fake_docling(monkeypatch, _converter_cls(_Result(), built))
    torch = fake_torch(available=True)
    monkeypatch.setenv("VORAG_OCR_GPU_MEM_FRACTION", raw)
    worker = load_worker()

    code = _main(monkeypatch, worker, torch, [str(_pdf(tmp_path)), str(tmp_path / "o.json")])

    assert code == 2
    assert built == []


def test_only_hosts_with_a_cap_pass_it_to_the_worker() -> None:
    from vo_rag.ocr_farm.hosts import DEFAULT_HOSTS

    win, lin = DEFAULT_HOSTS
    assert win.gpu_mem_fraction is None
    assert "VORAG_OCR_GPU_MEM_FRACTION" not in win.worker_command("a.pdf", "a.json")
    assert lin.gpu_mem_fraction == 0.25
    assert "VORAG_OCR_GPU_MEM_FRACTION=0.25 " in lin.worker_command("a.pdf", "a.json")


def test_worker_and_hosts_agree_on_the_env_var_name() -> None:
    from vo_rag.ocr_farm.hosts import GPU_MEM_FRACTION_ENV

    assert load_worker().GPU_MEM_FRACTION_ENV == GPU_MEM_FRACTION_ENV

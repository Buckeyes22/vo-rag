"""scripts/prefetch_ocr_models.py (VOR-1427): fetch RapidOCR's torch models ahead of first use."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "prefetch_ocr_models.py"
TORCH_FILES = (
    "ch_PP-OCRv4_det_mobile.pth",
    "ch_ptocr_mobile_v2.0_cls_mobile.pth",
    "ch_PP-OCRv4_rec_mobile.pth",
)


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("prefetch_ocr_models_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_rapidocr(
    monkeypatch: pytest.MonkeyPatch, writes: tuple[str, ...]
) -> list[dict[str, Any]]:
    """A rapidocr stand-in whose RapidOCR(params) 'downloads' ``writes``."""
    seen: list[dict[str, Any]] = []

    class _Engine:
        TORCH = "torch"

    class _RapidOCR:
        def __init__(self, params: dict[str, Any]) -> None:
            seen.append(params)
            root = Path(params["Global.model_root_dir"])
            for name in writes:
                (root / name).write_bytes(b"weights")

    monkeypatch.setitem(
        sys.modules, "rapidocr", types.SimpleNamespace(RapidOCR=_RapidOCR, __file__="x")
    )
    monkeypatch.setitem(sys.modules, "rapidocr.utils", types.ModuleType("rapidocr.utils"))
    monkeypatch.setitem(
        sys.modules, "rapidocr.utils.typings", types.SimpleNamespace(EngineType=_Engine)
    )
    return seen


def test_builds_torch_engines_on_cpu_into_the_model_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen = _fake_rapidocr(monkeypatch, TORCH_FILES)

    assert _load().main(["--model-root", str(tmp_path)]) == 0

    params = seen[0]
    assert {params[f"{stage}.engine_type"] for stage in ("Det", "Cls", "Rec")} == {"torch"}
    assert params["EngineConfig.torch.use_cuda"] is False
    # rapidocr joins the root with "/", so it must be a Path, not a str.
    assert isinstance(params["Global.model_root_dir"], Path)
    assert "ok" in capsys.readouterr().out


def test_fails_when_a_torch_model_did_not_arrive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_rapidocr(monkeypatch, TORCH_FILES[:2])

    assert _load().main(["--model-root", str(tmp_path)]) == 1
    assert "ch_PP-OCRv4_rec_mobile" in capsys.readouterr().err


def test_bundled_onnx_models_alone_count_as_missing(tmp_path: Path) -> None:
    for name in ("ch_PP-OCRv4_det_infer.onnx", "ch_ppocr_mobile_v2.0_cls_infer.onnx"):
        (tmp_path / name).write_bytes(b"onnx")
    assert len(_load().missing_models(tmp_path)) == 3

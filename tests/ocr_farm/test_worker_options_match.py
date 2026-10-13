"""The standalone OCR worker stays identical to the in-process OCR settings (VOR-1427).

The worker runs on hosts with no vo-rag install, so it inlines
``scanned_pipeline_options()``. These tests keep the two copies in lockstep:

- against a fake docling package (runs everywhere, including the locked CI
  env where docling is absent), field by field, for every device setting;
- against real docling when it is installed (skipped otherwise), by comparing
  the full pydantic dump of both option objects.

It also pins the worker's ``requirements.txt`` to ``uv.lock`` (base
versions, local labels such as ``+cu130`` ignored) and asserts the worker
imports nothing but the stdlib, docling, docling_core and torch.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from tests.ocr_farm._fakes import REQUIREMENTS_PATH, WORKER_PATH, install_fake_docling, load_worker

REPO_ROOT = Path(__file__).resolve().parents[2]


def _fields(options: Any) -> dict[str, Any]:
    return {
        "type": type(options).__name__,
        "do_ocr": options.do_ocr,
        "ocr_options_type": type(options.ocr_options).__name__,
        "ocr_backend": options.ocr_options.backend,
        "force_full_page_ocr": options.ocr_options.force_full_page_ocr,
        "accelerator_options_type": type(options.accelerator_options).__name__,
        "device": options.accelerator_options.device,
        "generate_picture_images": options.generate_picture_images,
    }


@pytest.mark.parametrize("device_env", [None, "", "cuda", "cuda:1", "cpu", "auto"])
def test_worker_options_equal_scanned_pipeline_options_field_by_field(
    monkeypatch: pytest.MonkeyPatch, device_env: str | None
) -> None:
    install_fake_docling(monkeypatch)
    if device_env is None:
        monkeypatch.delenv("VORAG_DOCLING_DEVICE", raising=False)
    else:
        monkeypatch.setenv("VORAG_DOCLING_DEVICE", device_env)
    from vo_rag.core.parsing.parsers.docling import scanned_pipeline_options

    worker = load_worker()
    ours = _fields(scanned_pipeline_options())
    theirs = _fields(worker.scanned_pipeline_options())

    assert theirs == ours
    # CUDA semantics: never a CPU device, whatever the env says.
    assert theirs["device"] != "cpu"
    assert theirs["do_ocr"] is True
    assert theirs["ocr_backend"] == "torch"
    assert theirs["force_full_page_ocr"] is True
    assert theirs["generate_picture_images"] is True


@pytest.mark.parametrize("device_env", [None, "cuda:1", "cpu"])
def test_worker_options_equal_scanned_pipeline_options_with_real_docling(
    monkeypatch: pytest.MonkeyPatch, device_env: str | None
) -> None:
    pytest.importorskip("docling", reason="docling not installed (runtime extra)")
    if device_env is None:
        monkeypatch.delenv("VORAG_DOCLING_DEVICE", raising=False)
    else:
        monkeypatch.setenv("VORAG_DOCLING_DEVICE", device_env)
    from vo_rag.core.parsing.parsers.docling import scanned_pipeline_options

    worker = load_worker()
    ours = scanned_pipeline_options()
    theirs = worker.scanned_pipeline_options()

    assert _fields(theirs) == _fields(ours)
    assert theirs.model_dump() == ours.model_dump()


def _lock_versions() -> dict[str, set[str]]:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    versions: dict[str, set[str]] = {}
    for package in lock.get("package", []):
        versions.setdefault(package["name"].lower(), set()).add(str(package.get("version", "")))
    return versions


def _base(version: str) -> str:
    return version.split("+", 1)[0]


def test_requirements_pin_expected_packages() -> None:
    pins = _requirement_pins()
    assert pins == {
        "docling": "2.97.0",
        "pypdfium2": "5.9.0",
        "rapidocr": "3.8.1",
        "torch": "2.12.0+cu130",
        "torchvision": "0.27.0+cu130",
    }


def test_requirements_versions_match_uv_lock() -> None:
    """Compare base versions; ``+cu130`` is a build variant, not a version."""
    lock = _lock_versions()
    for name, version in _requirement_pins().items():
        assert name in lock, f"{name} missing from uv.lock"
        assert _base(version) in {_base(v) for v in lock[name]}, (
            f"{name}=={version} does not match uv.lock {sorted(lock[name])}"
        )


def _requirement_pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, version = line.partition("==")
        assert sep, f"requirement not pinned with ==: {raw!r}"
        pins[name.strip().lower()] = version.strip()
    return pins


def test_worker_is_standalone() -> None:
    """Stdlib + docling/docling_core/pypdfium2 + torch only: the GPU hosts have no vo-rag."""
    tree = ast.parse(WORKER_PATH.read_text(encoding="utf-8"))
    allowed_third_party = {"docling", "docling_core", "pypdfium2", "torch"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports are not allowed in the worker"
            roots = {(node.module or "").split(".", 1)[0]}
        else:
            continue
        for root in roots:
            assert root in sys.stdlib_module_names or root in allowed_third_party, (
                f"worker imports non-stdlib module {root!r}"
            )

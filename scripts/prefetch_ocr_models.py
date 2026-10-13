"""Fetch RapidOCR's torch models into its model directory ahead of first use.

The scanned-PDF OCR path (``DoclingParser.parse_scanned``) runs RapidOCR on the
torch backend. The rapidocr wheel only bundles its ONNX models, so on first
use it downloads ``ch_PP-OCRv4_det_mobile.pth``, ``ch_ptocr_mobile_v2.0_cls_mobile.pth``
and ``ch_PP-OCRv4_rec_mobile.pth`` into ``site-packages/rapidocr/models``. If
that directory is read-only, every in-lane OCR parse fails with
``[Errno 13] Permission denied``. Run this right after ``uv sync``, while
the environment is still writable.

Building the engines only loads the model weights (on the CPU, which is
enough to trigger the download); no OCR is run. Exits 1 if any torch model
is still missing afterwards.

    .venv/bin/python scripts/prefetch_ocr_models.py [--model-root DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

#: The torch model files docling's RapidOCR (torch backend) loads, by stage.
TORCH_MODEL_PREFIXES = {
    "det": "ch_PP-OCRv4_det_mobile",
    "cls": "ch_ptocr_mobile_v2.0_cls",
    "rec": "ch_PP-OCRv4_rec_mobile",
}


def _params(model_root: Path | None) -> dict[str, Any]:
    from rapidocr.utils.typings import EngineType  # type: ignore[import-not-found, unused-ignore]

    params: dict[str, Any] = {
        "Det.engine_type": EngineType.TORCH,
        "Cls.engine_type": EngineType.TORCH,
        "Rec.engine_type": EngineType.TORCH,
        "EngineConfig.torch.use_cuda": False,
    }
    if model_root is not None:
        params["Global.model_root_dir"] = model_root  # rapidocr joins it with "/"
    return params


def default_model_root() -> Path:
    import rapidocr  # type: ignore[import-not-found, unused-ignore]

    return Path(rapidocr.__file__).resolve().parent / "models"


def missing_models(model_root: Path) -> list[str]:
    """Prefixes from ``TORCH_MODEL_PREFIXES`` with no ``.pth`` file in ``model_root``."""
    present = [p.name for p in model_root.glob("*.pth")]
    return [
        prefix
        for prefix in TORCH_MODEL_PREFIXES.values()
        if not any(name.startswith(prefix) for name in present)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-root", type=Path, default=None)
    args = parser.parse_args(argv)
    model_root = args.model_root or default_model_root()
    model_root.mkdir(parents=True, exist_ok=True)

    from rapidocr import RapidOCR  # type: ignore[import-not-found, unused-ignore]

    RapidOCR(params=_params(args.model_root))
    missing = missing_models(model_root)
    if missing:
        print(
            f"prefetch_ocr_models: missing in {model_root}: {', '.join(missing)}", file=sys.stderr
        )
        return 1
    names = sorted(p.name for p in model_root.glob("*.pth"))
    print(f"prefetch_ocr_models: ok {model_root}: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

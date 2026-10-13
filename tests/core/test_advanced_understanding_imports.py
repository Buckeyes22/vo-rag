from __future__ import annotations

import importlib

ADVANCED_MODULES = [
    "vo_rag.core.eval.judge_calibration",
    "vo_rag.core.eval.non_inferiority",
    "vo_rag.core.summary",
    "vo_rag.core.summary.builder",
    "vo_rag.core.summary.layer1a_retrieval",
    "vo_rag.core.raptor",
    "vo_rag.core.dsi",
    "vo_rag.core.graph",
    "vo_rag.core.contextual",
]


def test_advanced_understanding_modules_import() -> None:
    for module_name in ADVANCED_MODULES:
        assert importlib.import_module(module_name) is not None

"""Resident figure-specialist evaluation contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vo_rag.core.eval.vlm_figures import (
    FigureEvalCase,
    FigureEvalThresholds,
    load_figure_eval_cases,
    run_figure_eval,
    score_figure_result,
    summarize_figure_eval,
)
from vo_rag.pipelines.book_rag.visual_sidecar import VLMDescription


def _description(**overrides: object) -> VLMDescription:
    values: dict[str, object] = {
        "description": "The Intake arrow flows to a Pump labelled P-101.",
        "model_used": "resident",
        "token_count": 30,
        "duration_ms": 500,
        "image_id": "figure-1",
        "page_kind": "diagram",
        "visible_text": ("Intake", "P-101"),
        "confidence": 0.9,
    }
    values.update(overrides)
    return VLMDescription(**values)  # type: ignore[arg-type]


def test_load_cases_resolves_relative_images_and_rejects_duplicates(tmp_path: Path) -> None:
    image = tmp_path / "figure.png"
    image.write_bytes(b"not decoded by the manifest loader")
    payload = {
        "id": "figure-1",
        "image": image.name,
        "expected_page_kinds": ["diagram"],
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    cases = load_figure_eval_cases(manifest)

    assert cases[0].image == str(image.resolve())
    manifest.write_text((json.dumps(payload) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_figure_eval_cases(manifest)


def test_score_result_checks_grounding_terms_and_page_kind() -> None:
    case = FigureEvalCase(
        id="figure-1",
        image="figure.png",
        expected_page_kinds=["diagram", "mixed"],
        required_terms=["Intake", "P-101", "Pump"],
        forbidden_terms=["P-102"],
    )

    score = score_figure_result(case, _description())

    assert score["page_kind_correct"] is True
    assert score["required_term_recall"] == 1.0
    assert score["forbidden_term_hits"] == []


def test_summary_enforces_every_rollout_threshold() -> None:
    results = [
        {
            "id": f"figure-{index}",
            "ok": True,
            "page_kind_correct": True,
            "required_term_recall": 1.0,
            "forbidden_term_hits": [],
            "needs_burst": False,
            "duration_ms": 1_000,
        }
        for index in range(60)
    ]

    passed = summarize_figure_eval(results, FigureEvalThresholds())
    results[0]["forbidden_term_hits"] = ["unsupported"]
    failed = summarize_figure_eval(results, FigureEvalThresholds())

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["checks"]["forbidden_term_violations"] is False


def test_thresholds_reject_a_gate_that_cannot_be_interpreted() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        FigureEvalThresholds(minimum_success_rate=1.1)


@pytest.mark.asyncio
async def test_runner_is_sequential_and_records_case_failures(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    cases = [
        FigureEvalCase(id="first", image=str(first), expected_page_kinds=["diagram"]),
        FigureEvalCase(id="second", image=str(second), expected_page_kinds=["diagram"]),
    ]

    class _Service:
        model = "resident"

        def __init__(self) -> None:
            self.calls: list[str] = []

        async def describe_image(
            self,
            image_bytes: bytes,
            prompt: str,
            image_id: str,
            *,
            max_tokens: int,
        ) -> VLMDescription:
            self.calls.append(image_id)
            if image_id == "second":
                raise RuntimeError("synthetic failure")
            return _description(image_id=image_id)

        async def aclose(self) -> None:
            return None

    service = _Service()
    results = await run_figure_eval(cases, service, max_tokens=1024)

    assert service.calls == ["first", "second"]
    assert results[0]["ok"] is True
    assert results[1] == {"id": "second", "ok": False, "error": "RuntimeError"}

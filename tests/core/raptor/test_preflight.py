from __future__ import annotations

import math

from vo_rag.core.raptor.preflight import (
    RaptorBudgetConfig,
    decide_preflight,
    emit_preflight_report,
    project_budget,
    project_n_clusters,
)


class TestProjectNClusters:
    def test_project_n_clusters_232k_corpus_returns_2906_291_39(self) -> None:
        result = project_n_clusters(n_total_chunks=232_401)
        assert result["level_0"] == 2906
        assert result["level_1"] == 291
        assert result["level_2"] == 39

    def test_project_n_clusters_small_corpus_proportional(self) -> None:
        result = project_n_clusters(n_total_chunks=100)
        assert result["level_0"] == 2
        assert result["level_1"] == 1
        assert result["level_2"] == 1

    def test_project_n_clusters_exact_division(self) -> None:
        result = project_n_clusters(n_total_chunks=800)
        assert result["level_0"] == 10
        assert result["level_1"] == 1
        assert result["level_2"] == 1

    def test_project_n_clusters_zero_chunks(self) -> None:
        result = project_n_clusters(n_total_chunks=0)
        assert result["level_0"] == 0
        assert result["level_1"] == 0
        assert result["level_2"] == 0


class TestProjectBudget:
    def test_project_budget_sums_per_level_breakdown_to_total(self) -> None:
        config = RaptorBudgetConfig()
        projection = project_budget(n_total_chunks=232_000, config=config)

        total_input = sum(v["input_tokens"] for v in projection.breakdown_per_level.values())
        total_output = sum(v["output_tokens"] for v in projection.breakdown_per_level.values())
        total_cost = sum(v["cost_usd"] for v in projection.breakdown_per_level.values())

        assert total_input == projection.total_input_tokens
        assert total_output == projection.total_output_tokens
        assert math.isclose(total_cost, projection.projected_cost_usd, rel_tol=1e-9)

    def test_project_budget_concurrent_calls_reduces_wall_time(self) -> None:
        config_single = RaptorBudgetConfig(concurrent_llm_calls=1)
        config_parallel = RaptorBudgetConfig(concurrent_llm_calls=16)

        projection_single = project_budget(n_total_chunks=10_000, config=config_single)
        projection_parallel = project_budget(n_total_chunks=10_000, config=config_parallel)

        assert (
            projection_single.projected_wall_time_hours
            > projection_parallel.projected_wall_time_hours
        )
        ratio = (
            projection_single.projected_wall_time_hours
            / projection_parallel.projected_wall_time_hours
        )
        assert 15.5 < ratio < 16.5

    def test_project_budget_cost_proportional_to_tokens(self) -> None:
        config = RaptorBudgetConfig()
        projection_4000 = project_budget(n_total_chunks=4000, config=config)
        projection_8000 = project_budget(n_total_chunks=8000, config=config)

        level_0_tokens_4000 = projection_4000.breakdown_per_level["level_0"]["input_tokens"]
        level_0_tokens_8000 = projection_8000.breakdown_per_level["level_0"]["input_tokens"]
        assert level_0_tokens_8000 == 2 * level_0_tokens_4000
        assert projection_8000.projected_cost_usd == 2 * projection_4000.projected_cost_usd

    def test_project_budget_breakdown_has_all_three_levels(self) -> None:
        config = RaptorBudgetConfig()
        projection = project_budget(n_total_chunks=50_000, config=config)
        assert set(projection.breakdown_per_level.keys()) == {"level_0", "level_1", "level_2"}

    def test_project_budget_wall_time_formula_with_concurrency(self) -> None:
        config = RaptorBudgetConfig(concurrent_llm_calls=8)
        projection = project_budget(n_total_chunks=640, config=config)
        level_0_clusters = math.ceil(640 / 80)
        expected_wall_h = (level_0_clusters * config.avg_summary_wall_time_s) / 8 / 3600
        actual_wall_h = projection.breakdown_per_level["level_0"]["wall_time_hours"]
        assert math.isclose(actual_wall_h, expected_wall_h, rel_tol=1e-9)


class TestDecidePreflight:
    def test_decide_preflight_within_budget_proceeds(self) -> None:
        config = RaptorBudgetConfig(
            max_total_input_tokens=100_000_000,
            max_total_output_tokens=10_000_000,
            max_wall_time_hours=48.0,
            max_cost_usd=500.0,
        )
        projection = project_budget(n_total_chunks=50_000, config=config)
        verdict = decide_preflight(projection=projection, budget=config)

        assert verdict.proceed is True
        assert verdict.action == "proceed"
        assert len(verdict.violations) == 0

    def test_decide_preflight_single_axis_breach_yields_reduce_scope(self) -> None:
        config = RaptorBudgetConfig(
            max_total_input_tokens=1,
            max_total_output_tokens=10_000_000,
            max_wall_time_hours=48.0,
            max_cost_usd=500.0,
        )
        projection = project_budget(n_total_chunks=50_000, config=config)
        verdict = decide_preflight(projection=projection, budget=config)

        assert verdict.proceed is False
        assert verdict.action == "reduce_scope"
        assert len(verdict.violations) == 1
        assert "input_tokens" in verdict.violations[0]

    def test_decide_preflight_multi_axis_breach_yields_abort_over_budget(self) -> None:
        config = RaptorBudgetConfig(
            max_total_input_tokens=1,
            max_total_output_tokens=1,
            max_wall_time_hours=0.001,
            max_cost_usd=0.001,
        )
        projection = project_budget(n_total_chunks=50_000, config=config)
        verdict = decide_preflight(projection=projection, budget=config)

        assert verdict.proceed is False
        assert verdict.action == "abort_over_budget"
        assert len(verdict.violations) >= 2

    def test_decide_preflight_records_violation_for_each_breach(self) -> None:
        config = RaptorBudgetConfig(
            max_total_input_tokens=1,
            max_total_output_tokens=1,
            max_wall_time_hours=48.0,
            max_cost_usd=500.0,
        )
        projection = project_budget(n_total_chunks=50_000, config=config)
        verdict = decide_preflight(projection=projection, budget=config)

        assert verdict.proceed is False
        assert len(verdict.violations) == 2
        violation_keys = {v.split()[0] for v in verdict.violations}
        assert "total_input_tokens" in violation_keys
        assert "total_output_tokens" in violation_keys

    def test_decide_preflight_cost_breach_recorded(self) -> None:
        config = RaptorBudgetConfig(
            max_total_input_tokens=100_000_000,
            max_total_output_tokens=10_000_000,
            max_wall_time_hours=48.0,
            max_cost_usd=0.00001,
        )
        projection = project_budget(n_total_chunks=50_000, config=config)
        verdict = decide_preflight(projection=projection, budget=config)

        assert verdict.proceed is False
        assert any("cost_usd" in v for v in verdict.violations)

    def test_decide_preflight_over_budget_returns_abort_over_budget(self) -> None:
        """project_budget/decide_preflight returns abort_over_budget when over budget."""
        config = RaptorBudgetConfig(
            max_total_input_tokens=100_000_000,
            max_total_output_tokens=10_000_000,
            max_wall_time_hours=0.001,
            max_cost_usd=0.00001,
        )
        projection = project_budget(n_total_chunks=50_000, config=config)
        verdict = decide_preflight(projection=projection, budget=config)

        assert verdict.action == "abort_over_budget"
        assert verdict.proceed is False
        assert any("cost_usd" in v for v in verdict.violations)
        assert any("wall_time" in v for v in verdict.violations)


class TestEmitPreflightReport:
    def test_emit_preflight_report_returns_multi_line_with_per_level_breakdown(self) -> None:
        config = RaptorBudgetConfig()
        projection = project_budget(n_total_chunks=50_000, config=config)
        verdict = decide_preflight(projection=projection, budget=config)
        report = emit_preflight_report(verdict)

        lines = report.split("\n")
        assert len(lines) > 10
        assert "level_0" in report
        assert "level_1" in report
        assert "level_2" in report
        assert "Clusters" in report
        assert "Input tok" in report
        assert "Output tok" in report

    def test_emit_preflight_report_contains_budget_caps(self) -> None:
        config = RaptorBudgetConfig(
            max_total_input_tokens=50_000_000,
            max_total_output_tokens=5_000_000,
            max_wall_time_hours=24.0,
            max_cost_usd=200.0,
        )
        projection = project_budget(n_total_chunks=100_000, config=config)
        verdict = decide_preflight(projection=projection, budget=config)
        report = emit_preflight_report(verdict)

        assert "50,000,000" in report
        assert "5,000,000" in report
        assert "24.00" in report
        assert "200.00" in report

    def test_emit_preflight_report_contains_verdict(self) -> None:
        config = RaptorBudgetConfig()
        projection = project_budget(n_total_chunks=50_000, config=config)
        verdict = decide_preflight(projection=projection, budget=config)
        report = emit_preflight_report(verdict)

        assert verdict.action.upper() in report
        assert str(verdict.proceed) in report


class TestBudgetConfigDefaults:
    def test_budget_config_defaults(self) -> None:
        config = RaptorBudgetConfig()
        assert config.max_total_input_tokens == 50_000_000
        assert config.max_total_output_tokens == 5_000_000
        assert config.max_wall_time_hours == 24.0
        assert config.max_cost_usd == 200.0
        assert config.summary_model == "qwen3.6-27b-fp8"
        assert config.input_cost_per_million_tokens_usd == 0.40
        assert config.output_cost_per_million_tokens_usd == 1.20
        assert config.avg_tokens_per_chunk == 384
        assert config.avg_summary_input_tokens == 6000
        assert config.avg_summary_output_tokens == 600
        assert config.avg_summary_wall_time_s == 8.0
        assert config.concurrent_llm_calls == 16

    def test_budget_config_accepts_custom_overrides(self) -> None:
        config = RaptorBudgetConfig(
            max_total_input_tokens=10_000_000,
            max_wall_time_hours=12.0,
            concurrent_llm_calls=32,
        )
        assert config.max_total_input_tokens == 10_000_000
        assert config.max_wall_time_hours == 12.0
        assert config.concurrent_llm_calls == 32
        assert config.max_total_output_tokens == 5_000_000

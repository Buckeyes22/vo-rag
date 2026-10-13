import pytest

from vo_rag.core.config.pipelines import (
    PipelineSpec,
    PipelineStatus,
    load_pipeline_registry,
)


def test_registry_loads_all_ten_pipelines():
    reg = load_pipeline_registry()
    assert len(reg) == 10
    assert "book_rag" in reg
    assert isinstance(reg["book_rag"], PipelineSpec)


def test_book_rag_is_active_others_planned():
    # book_rag is in production (VOR-7); the other pipelines are still planned.
    reg = load_pipeline_registry()
    assert reg["book_rag"].status is PipelineStatus.ACTIVE
    assert reg["code_rag"].status is PipelineStatus.PLANNED


def test_book_rag_is_the_only_active_pipeline():
    reg = load_pipeline_registry()
    active = [s.name for s in reg.values() if s.status is PipelineStatus.ACTIVE]
    assert active == ["book_rag"]


def test_unknown_status_raises(tmp_path):
    bad = tmp_path / "pyproject.toml"
    bad.write_text('[tool.vo_rag]\npipelines = [{ name = "x", status = "bogus", summary = "s" }]\n')
    with pytest.raises(ValueError):
        load_pipeline_registry(bad)

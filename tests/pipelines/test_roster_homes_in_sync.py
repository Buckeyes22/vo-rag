from pathlib import Path

from vo_rag.core.config.pipelines import load_pipeline_registry

PIPELINES_DIR = Path(__file__).resolve().parents[2] / "src" / "vo_rag" / "pipelines"


def _home_dirs() -> set[str]:
    return {p.name for p in PIPELINES_DIR.iterdir() if p.is_dir() and p.name != "__pycache__"}


def test_every_roster_entry_has_a_readme_home():
    reg = load_pipeline_registry()
    for name in reg:
        readme = PIPELINES_DIR / name / "README.md"
        assert readme.is_file(), f"missing README home for roster pipeline {name}"


def test_no_orphan_home_without_roster_entry():
    reg = load_pipeline_registry()
    orphans = _home_dirs() - set(reg)
    assert not orphans, f"pipeline home(s) with no roster entry: {orphans}"

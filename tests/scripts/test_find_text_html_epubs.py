"""Unit contract for the read-only text/html-manifest EPUB sweep
(VOR-1427). Only the pure OPF-detection function is exercised here -- no
DB/Garage needed; the Postgres/Garage side runs against real data separately.
"""

from __future__ import annotations

from pathlib import Path

from tests.fixtures.epub.build_text_html_manifest_epub import _build_epub

from scripts import find_text_html_epubs
from vo_rag.core.parsing.parsers import epub as epub_module

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "epub" / "text_html_manifest.epub"


def test_counts_text_html_manifest_items_in_the_affected_fixture() -> None:
    assert find_text_html_epubs.count_text_html_manifest_items(FIXTURE.read_bytes()) == 3


def test_counts_zero_for_a_normal_epub(tmp_path: Path) -> None:
    valid = _build_epub(tmp_path / "ok.epub", media_type="application/xhtml+xml")
    assert find_text_html_epubs.count_text_html_manifest_items(valid.read_bytes()) == 0


def test_counts_single_quoted_manifest_items(tmp_path: Path) -> None:
    """Single-quoted `media-type='text/html'` (legal
    XML) must be counted, not silently missed by the audit sweep."""
    single_quoted = _build_epub(tmp_path / "single_quoted.epub", media_type="text/html", quote="'")
    assert find_text_html_epubs.count_text_html_manifest_items(single_quoted.read_bytes()) == 3


def test_counts_mixed_case_manifest_items(tmp_path: Path) -> None:
    """Mixed-case `Text/HTML` must be counted too."""
    mixed_case = _build_epub(tmp_path / "mixed_case.epub", media_type="Text/HTML")
    assert find_text_html_epubs.count_text_html_manifest_items(mixed_case.read_bytes()) == 3


def test_script_imports_the_shared_detection_function() -> None:
    """The script must not carry its own copy of the
    regex/counting logic -- it imports the parser module's single source of
    truth, so the two can never diverge on what counts as a match."""
    assert find_text_html_epubs.count_text_html_manifest_items is (
        epub_module.count_text_html_manifest_items
    )


def test_help_documents_readonly_scan(capsys) -> None:
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        find_text_html_epubs.main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "Read-only" in out
    assert "--limit" in out

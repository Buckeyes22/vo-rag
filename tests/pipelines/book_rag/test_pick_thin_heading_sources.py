from __future__ import annotations

import json

import pytest
from scripts.pick_thin_heading_sources import rank_sources


def test_rank_sources_orders_heading_thin_sources_first(tmp_path) -> None:
    export_path = tmp_path / "chunks.jsonl"
    rows = [
        {"source_id": "source-deep", "section_path": "Book > Chapter 1 > Section > Topic"},
        {"source_id": "source-thin", "section_path": "Book"},
        {"source_id": "source-middle", "section_path": "Book > Chapter 1 > Section"},
        {"source_id": "source-thin", "section_path": "Book > Chapter 1"},
        {"source_id": "source-deep", "section_path": "Book > Chapter 2 > Section"},
        {"source_id": "source-middle", "section_path": "Book > Chapter 2"},
    ]
    export_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    ranked = rank_sources(export_path)

    assert [row.source_id for row in ranked] == ["source-thin", "source-middle", "source-deep"]
    assert ranked[0].chunk_count == 2
    assert ranked[0].mean_depth == pytest.approx(0.5)
    assert ranked[0].shallow_pct == pytest.approx(1.0)

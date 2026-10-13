"""Contract tests for the local-only source-bound repair manifest builder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
from pathlib import Path
from typing import Any

import pytest

spec = importlib.util.spec_from_file_location(
    "corpus_repair_manifest", Path(__file__).parents[2] / "scripts" / "corpus_repair_manifest.py"
)
assert spec is not None and spec.loader is not None
manifest_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifest_tool)

ORIGINAL_SHA = "a" * 64
PARSE_SHA = "b" * 64
SOURCE_ID = f"vo-rag-corpus-sha256:{ORIGINAL_SHA}"


def _row(chunk_id: str, text: str, **fields: Any) -> dict[str, Any]:
    return {
        "point_id": f"point-{chunk_id}",
        "source_id": SOURCE_ID,
        "accepted_parse_sha256": PARSE_SHA,
        "chunk_id": chunk_id,
        "chunk_type": "text",
        "section_path": "Chapter 1",
        "token_count": 2,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        **fields,
    }


def _snapshot(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(
            {
                "source_id": SOURCE_ID,
                "original_source_sha256": ORIGINAL_SHA,
                "accepted_parse_sha256": PARSE_SHA,
                "accepted_parse_bytes": 1234,
                "version": "test",
                "chunker_fingerprint": "fingerprint",
                "collection": "vo_rag_books_raw_v1",
                "chunks": rows,
            }
        )
    )


def test_builds_exact_source_bound_replacement_and_preserves_unknown_lineage(
    tmp_path: Path,
) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    _snapshot(before, [_row("old", "before", metadata={"semantic_label": "unknown"})])
    _snapshot(
        after,
        [
            _row(
                "new",
                "after",
                metadata={"semantic_label": "unknown", "citation_ids": ["c1"]},
                table_element_ids=["source-element-hash"],
            )
        ],
    )
    live = tmp_path / "live.json"
    payload = {"source_id": SOURCE_ID, "chunk_id": "old", "content": "before"}
    payload_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    live.write_text(
        json.dumps(
            {
                "collection": "vo_rag_books_raw_v1",
                "source_ids": {"books05": SOURCE_ID},
                "captured_at_utc": "test",
                "capture_host": "test-read-only",
                "truncated_sources": [],
                "source_point_counts_captured": {"books05": 1},
                "points": [
                    {"point_id": "live-old", "payload_sha256": payload_hash, "payload": payload}
                ],
            }
        )
    )

    result = manifest_tool.build_manifest(
        before,
        after,
        source_id=SOURCE_ID,
        original_sha256=ORIGINAL_SHA,
        accepted_parse_sha256=PARSE_SHA,
        live_points_path=live,
    )

    assert result["eligible"] is False
    assert result["mode"] == "dry_run_only"
    assert result["writer"]["production_writer"] is None
    assert result["lineage"]["semantic_labels"] == "unknown_preserved"
    assert result["lineage"]["citation_lineage"] == "unknown_preserved"
    assert "production_writer_contract_unknown" in result["ineligibility_reasons"]
    assert "before_point_id_mismatch:old" in result["ineligibility_reasons"]
    assert result["live_points"]["matched_to_before_chunks"] == 1
    assert result["live_points"]["preserved_unmatched_count"] == 0
    assert [change["operation"] for change in result["changes"]] == ["add", "remove"]
    assert result["changes"][0]["after"]["text"] == "after"
    assert result["changes"][1]["before"]["text"] == "before"


def test_source_or_parse_mismatch_is_explicitly_ineligible(tmp_path: Path) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    _snapshot(before, [_row("c1", "before")])
    _snapshot(after, [_row("c1", "after")])

    result = manifest_tool.build_manifest(
        before,
        after,
        source_id="vo-rag-corpus-sha256:" + "c" * 64,
        original_sha256="c" * 64,
        accepted_parse_sha256=PARSE_SHA,
    )

    assert result["eligible"] is False
    assert result["disposition"] == "ineligible"
    assert "before_source_id_mismatch" in result["ineligibility_reasons"]
    assert "after_source_id_mismatch" in result["ineligibility_reasons"]


def test_snapshot_rejects_tampered_exact_text_digest(tmp_path: Path) -> None:
    snapshot = tmp_path / "tampered.json"
    row = _row("c1", "source text")
    row["text"] = "changed after hash"
    _snapshot(snapshot, [row])

    with pytest.raises(ValueError, match="text_sha256 mismatch"):
        manifest_tool._read_chunks(snapshot)


def test_manifest_rejects_duplicate_point_ids_and_collection_scope_mismatch(
    tmp_path: Path,
) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    _snapshot(
        before,
        [
            _row("old-1", "one", point_id="duplicate"),
            _row("old-2", "two", point_id="duplicate"),
        ],
    )
    _snapshot(after, [_row("new", "replacement")])
    data = json.loads(after.read_text())
    data["collection"] = "other_collection"
    after.write_text(json.dumps(data))

    result = manifest_tool.build_manifest(
        before,
        after,
        source_id=SOURCE_ID,
        original_sha256=ORIGINAL_SHA,
        accepted_parse_sha256=PARSE_SHA,
    )

    assert result["eligible"] is False
    assert "before_snapshot_point_ids_not_unique" in result["ineligibility_reasons"]
    assert "before_after_collection_scope_mismatch" in result["ineligibility_reasons"]


def test_duplicate_chunk_ids_are_retained_as_separate_point_occurrences(tmp_path: Path) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    _snapshot(
        before,
        [_row("same", "copy", point_id="p1"), _row("same", "copy", point_id="p2")],
    )
    _snapshot(after, [_row("same", "copy", point_id="p1")])

    result = manifest_tool.build_manifest(
        before,
        after,
        source_id=SOURCE_ID,
        original_sha256=ORIGINAL_SHA,
        accepted_parse_sha256=PARSE_SHA,
    )

    assert result["eligible"] is False
    assert len(result["changes"]) == 2
    assert {change["operation"] for change in result["changes"]} == {"replace", "remove"}
    assert all(change["chunk_id"] == "same" for change in result["changes"])
    assert any(change["before"]["point_id"] == "p2" for change in result["changes"])
    assert "live_point_snapshot_missing" in result["ineligibility_reasons"]


def test_cli_writes_manifest_without_applying_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    output = tmp_path / "private" / "manifest.json"
    _snapshot(before, [_row("c1", "before")])
    _snapshot(after, [_row("c1", "after")])

    status = manifest_tool.main(
        [
            "--before",
            str(before),
            "--after",
            str(after),
            "--source-id",
            SOURCE_ID,
            "--original-sha256",
            ORIGINAL_SHA,
            "--accepted-parse-sha256",
            PARSE_SHA,
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert json.loads(output.read_text())["mode"] == "dry_run_only"
    assert json.loads(capsys.readouterr().out)["eligible"] is False
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert stat.S_IMODE(output.parent.stat().st_mode) == 0o700

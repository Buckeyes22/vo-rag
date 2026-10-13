#!/usr/bin/env python3
"""Build a source-bound, dry-run replacement manifest from private chunk snapshots.

The tool performs no network or database access. It consumes JSON snapshots made
by an independently reviewed shadow parse/chunk run and writes an explicit
replacement plan; producing a manifest never applies it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "vo-rag.corpus-repair-manifest.v1"
WRITER_IDENTITY = "scripts/corpus_repair_manifest.py:dry-run-only"
MAX_SNAPSHOT_BYTES = 50_000_000
MAX_CHUNKS_PER_SNAPSHOT = 5_000
MAX_TOTAL_TEXT_BYTES = 20_000_000


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_chunks(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read one source-bound snapshot JSON object, optionally after log lines."""
    if path.stat().st_size > MAX_SNAPSHOT_BYTES:
        raise ValueError(f"{path}: snapshot exceeds {MAX_SNAPSHOT_BYTES} byte limit")
    raw = path.read_text(encoding="utf-8")
    rows: Any = None
    source_id: Any = None
    parse_sha: Any = None
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        document = None
        # Shadow-run logs can precede one JSON snapshot on stdout. Do not
        # salvage multiple snapshots or skip malformed JSON-looking records.
        candidates: list[Any] = []
        for line in raw.splitlines():
            if not line.lstrip().startswith("{"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: malformed JSON record in snapshot output") from exc
            if isinstance(value, dict) and isinstance(value.get("chunks"), list):
                candidates.append(value)
        if len(candidates) == 1:
            document = candidates[0]
        else:
            raise ValueError(
                f"{path}: expected exactly one complete bound snapshot object"
            ) from None

    if isinstance(document, dict):
        source_id = document.get("source_id")
        parse_sha = document.get("accepted_parse_sha256")
        rows = document.get("chunks")
        original_sha = document.get("original_source_sha256")
        binding = {
            "source_id": source_id,
            "accepted_parse_sha256": parse_sha,
            "original_source_sha256": original_sha,
            "accepted_parse_bytes": document.get("accepted_parse_bytes"),
            "version": document.get("version"),
            "chunker_fingerprint": document.get("chunker_fingerprint"),
            "collection": document.get("collection"),
        }
    else:
        raise ValueError(f"{path}: expected a complete bound snapshot object")

    if not isinstance(source_id, str) or not source_id:
        raise ValueError(f"{path}: missing source_id")
    if not isinstance(parse_sha, str) or len(parse_sha) != 64:
        raise ValueError(f"{path}: missing accepted_parse_sha256")
    if not isinstance(original_sha, str) or not _is_sha256(original_sha):
        raise ValueError(f"{path}: missing or invalid original_source_sha256 binding")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected chunks array")
    if len(rows) > MAX_CHUNKS_PER_SNAPSHOT:
        raise ValueError(f"{path}: exceeds {MAX_CHUNKS_PER_SNAPSHOT} chunk limit")

    chunks: list[dict[str, Any]] = []
    text_bytes = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: chunk {index} is not an object")
        chunk_id = row.get("chunk_id")
        text = row.get("text")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"{path}: chunk {index} has no chunk_id")
        if not isinstance(text, str):
            raise ValueError(f"{path}: chunk {chunk_id!r} has no exact text")
        text_bytes += len(text.encode("utf-8"))
        if text_bytes > MAX_TOTAL_TEXT_BYTES:
            raise ValueError(f"{path}: exceeds {MAX_TOTAL_TEXT_BYTES} total text byte limit")
        if row.get("text_sha256") != _sha256(text):
            raise ValueError(f"{path}: chunk {chunk_id!r} text_sha256 mismatch")
        if row.get("source_id", source_id) != source_id:
            raise ValueError(f"{path}: chunk {chunk_id!r} source_id mismatch")
        if row.get("accepted_parse_sha256", parse_sha) != parse_sha:
            raise ValueError(f"{path}: chunk {chunk_id!r} accepted parse binding mismatch")
        if "point_id" in row and not isinstance(row["point_id"], (str, int)):
            raise ValueError(f"{path}: chunk {chunk_id!r} has invalid point_id")
        chunks.append(dict(row))
    return binding, chunks


def _index_occurrences(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["chunk_id"], []).append(row)
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for chunk_id, occurrences in grouped.items():
        stable = sorted(occurrences, key=lambda row: json.dumps(row, sort_keys=True))
        indexed.update({(chunk_id, ordinal): row for ordinal, row in enumerate(stable)})
    return indexed


def _read_live_points(path: Path, source_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path.stat().st_size > MAX_SNAPSHOT_BYTES:
        raise ValueError(f"{path}: live-point snapshot exceeds {MAX_SNAPSHOT_BYTES} byte limit")
    raw = path.read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict) or not isinstance(document.get("points"), list):
        raise ValueError(f"{path}: expected live-point snapshot object")
    matched: list[dict[str, Any]] = []
    for item in document["points"]:
        if not isinstance(item, dict) or not isinstance(item.get("payload"), dict):
            raise ValueError(f"{path}: malformed live-point entry")
        payload = item["payload"]
        if payload.get("source_id") != source_id:
            continue
        point_id, payload_sha = item.get("point_id"), item.get("payload_sha256")
        if not isinstance(point_id, (str, int)) or not isinstance(payload_sha, str):
            raise ValueError(f"{path}: live point lacks point id or payload hash")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != payload_sha:
            raise ValueError(f"{path}: live point {point_id} payload hash mismatch")
        if not isinstance(payload.get("content"), str) or not isinstance(
            payload.get("chunk_id"), str
        ):
            raise ValueError(f"{path}: live point {point_id} lacks content/chunk_id")
        matched.append(
            {
                "point_id": point_id,
                "payload_sha256": payload_sha,
                "chunk_id": payload["chunk_id"],
                "content_sha256": _sha256(payload["content"]),
                "content": payload["content"],
            }
        )
    labels = [
        label
        for label, candidate in (document.get("source_ids") or {}).items()
        if candidate == source_id
    ]
    truncated_labels = set(document.get("truncated_sources") or [])
    truncated = not labels or any(label in truncated_labels for label in labels)
    count_receipt = document.get("source_point_counts_captured") or {}
    expected_count = sum(count_receipt.get(label, 0) for label in labels)
    capture = {
        "path": str(path),
        "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "collection": document.get("collection"),
        "capture_host": document.get("capture_host"),
        "captured_at_utc": document.get("captured_at_utc"),
        "source_complete": not truncated,
        "source_point_count_captured": len(matched),
        "source_labels": labels,
        "filtered_source_count": expected_count if labels else None,
    }
    return capture, matched


def build_manifest(
    before_path: Path,
    after_path: Path,
    *,
    source_id: str,
    original_sha256: str,
    accepted_parse_sha256: str,
    live_points_path: Path | None = None,
    original_file: Path | None = None,
    accepted_parse_file: Path | None = None,
    target_collection: str | None = None,
) -> dict[str, Any]:
    """Construct exact per-ID changes and conservative eligibility evidence."""
    before_binding, before = _read_chunks(before_path)
    after_binding, after = _read_chunks(after_path)
    reasons: list[str] = []
    live_capture: dict[str, Any] | None = None
    live_points: list[dict[str, Any]] = []
    if live_points_path is None:
        reasons.append("live_point_snapshot_missing")
    else:
        live_capture, live_points = _read_live_points(live_points_path, source_id)
        if live_capture.get("source_complete") is not True:
            reasons.append("live_point_capture_truncated_for_source")
        if not live_capture.get("collection"):
            reasons.append("live_point_collection_unknown")
        if not isinstance(live_capture.get("filtered_source_count"), int):
            reasons.append("live_point_count_receipt_missing")
        elif live_capture["filtered_source_count"] != len(live_points):
            reasons.append("live_point_count_receipt_does_not_match_capture")
        if target_collection is None:
            reasons.append("target_collection_not_explicit")
        elif target_collection != live_capture.get("collection"):
            reasons.append("target_collection_does_not_match_live_capture")
        reasons.append("production_writer_contract_unknown")
    before_collection = before_binding.get("collection")
    after_collection = after_binding.get("collection")
    if before_collection != after_collection:
        reasons.append("before_after_collection_scope_mismatch")
    if live_capture is not None:
        if before_collection != live_capture.get("collection"):
            reasons.append("before_live_collection_scope_mismatch")
        if after_collection != live_capture.get("collection"):
            reasons.append("after_live_collection_scope_mismatch")
    before_point_ids = [row["point_id"] for row in before if "point_id" in row]
    if len(before_point_ids) != len(set(before_point_ids)):
        reasons.append("before_snapshot_point_ids_not_unique")
    original_actual = _file_sha256(original_file) if original_file else None
    parse_actual = _file_sha256(accepted_parse_file) if accepted_parse_file else None
    if original_actual != original_sha256:
        reasons.append("original_file_not_independently_verified")
    if parse_actual != accepted_parse_sha256:
        reasons.append("accepted_parse_file_not_independently_verified")
    live_by_text: dict[str, list[dict[str, Any]]] = {}
    all_live_ids = [point["point_id"] for point in live_points]
    if len(all_live_ids) != len(set(all_live_ids)):
        reasons.append("live_snapshot_point_ids_not_unique")
    for point in live_points:
        live_by_text.setdefault(point["content_sha256"], []).append(point)
    matched_live_ids: set[str | int] = set()
    for row in before:
        matches = live_by_text.get(row["text_sha256"], [])
        chosen = next((p for p in matches if p["point_id"] not in matched_live_ids), None)
        if chosen is None:
            row["live_match"] = "unmatched"
            reasons.append(f"before_chunk_unmatched_in_live:{row['chunk_id']}")
        else:
            if row.get("point_id") != chosen["point_id"]:
                reasons.append(f"before_point_id_mismatch:{row['chunk_id']}")
            row["live_match"] = {
                "point_id": chosen["point_id"],
                "payload_sha256": chosen["payload_sha256"],
            }
            matched_live_ids.add(chosen["point_id"])
    for row in after:
        row.pop("point_id", None)
        row["writer_point_identity"] = "unknown_not_produced_by_dry_run"
    for label, binding in (("before", before_binding), ("after", after_binding)):
        if binding["source_id"] != source_id:
            reasons.append(f"{label}_source_id_mismatch")
        if binding["accepted_parse_sha256"] != accepted_parse_sha256:
            reasons.append(f"{label}_accepted_parse_sha256_mismatch")
        recorded_original = binding.get("original_source_sha256")
        if recorded_original != original_sha256:
            reasons.append(f"{label}_original_sha256_mismatch")
    if before_binding.get("accepted_parse_bytes") != after_binding.get("accepted_parse_bytes"):
        reasons.append("accepted_parse_byte_length_changed")
    expected_source = f"vo-rag-corpus-sha256:{original_sha256}"
    if source_id != expected_source:
        reasons.append("source_id_does_not_bind_original_sha256")
    if not _is_sha256(original_sha256) or not _is_sha256(accepted_parse_sha256):
        reasons.append("invalid_sha256_length")

    before_by_id = _index_occurrences(before)
    after_by_id = _index_occurrences(after)
    changes: list[dict[str, Any]] = []
    for occurrence_key in sorted(before_by_id.keys() | after_by_id.keys()):
        chunk_id, occurrence = occurrence_key
        old = before_by_id.get(occurrence_key)
        new = after_by_id.get(occurrence_key)
        if old == new:
            continue
        operation = "add" if old is None else "remove" if new is None else "replace"
        changes.append(
            {
                "operation": operation,
                "chunk_id": chunk_id,
                "occurrence": occurrence,
                "before": old,
                "after": new,
                "before_text_sha256": old.get("text_sha256") if old else None,
                "after_text_sha256": new.get("text_sha256") if new else None,
                "metadata_diff": _metadata_diff(old, new),
            }
        )

    return {
        "schema": MANIFEST_SCHEMA,
        "mode": "dry_run_only",
        "eligible": not reasons and bool(changes),
        "disposition": "review_required"
        if not reasons and changes
        else ("no_changes" if not reasons else "ineligible"),
        "ineligibility_reasons": reasons,
        "source": {
            "source_id": source_id,
            "original_sha256": original_sha256,
            "accepted_parse_sha256": accepted_parse_sha256,
            "accepted_parse_bytes": before_binding.get("accepted_parse_bytes"),
            "origin_verification": "verified_file_bytes"
            if original_actual == original_sha256 and parse_actual == accepted_parse_sha256
            else "snapshot_claim_only_unverified",
            "original_file_sha256_verified": original_actual,
            "accepted_parse_file_sha256_verified": parse_actual,
        },
        "writer": {
            "identity": WRITER_IDENTITY,
            "script_sha256": _sha256(Path(__file__).read_text(encoding="utf-8")),
            "production_writer": None,
            "target_collection": target_collection,
        },
        "lineage": {
            "semantic_labels": "unknown_preserved",
            "citation_lineage": "unknown_preserved",
            "vector_identity": "unknown_requires_reembedding_for_changed_text",
            "tag_identity": "unknown_requires_retagging_for_changed_text",
        },
        "live_points": {
            **(live_capture or {"status": "missing"}),
            "matched_to_before_chunks": len(matched_live_ids),
            "preserved_unmatched_count": max(0, len(live_points) - len(matched_live_ids)),
            "unmatched_policy": "preserve; never propose blanket removal",
        },
        "snapshots": {
            "before": {
                "path": str(before_path),
                "snapshot_sha256": hashlib.sha256(before_path.read_bytes()).hexdigest(),
                "chunk_count": len(before),
                "producer": before_binding.get("version"),
                "chunker_fingerprint": before_binding.get("chunker_fingerprint"),
                "collection": before_binding.get("collection"),
            },
            "after": {
                "path": str(after_path),
                "snapshot_sha256": hashlib.sha256(after_path.read_bytes()).hexdigest(),
                "chunk_count": len(after),
                "producer": after_binding.get("version"),
                "chunker_fingerprint": after_binding.get("chunker_fingerprint"),
                "collection": after_binding.get("collection"),
            },
        },
        "changes": changes,
    }


def _metadata_diff(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    """Report every metadata and row-field change without guessing its meaning."""
    if before is None or after is None:
        return {"row": {"before": before, "after": after}}
    keys = sorted(before.keys() | after.keys())
    return {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in keys
        if before.get(key) != after.get(key)
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before", required=True, type=Path, help="private before chunk snapshot JSON"
    )
    parser.add_argument(
        "--after", required=True, type=Path, help="private after chunk snapshot JSON"
    )
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--original-sha256", required=True)
    parser.add_argument("--accepted-parse-sha256", required=True)
    parser.add_argument("--live-points", type=Path, help="private read-only current point snapshot")
    parser.add_argument("--original-file", type=Path, help="private original bytes to hash")
    parser.add_argument(
        "--accepted-parse-file", type=Path, help="private accepted parse bytes to hash"
    )
    parser.add_argument("--target-collection", help="explicit intended write collection")
    parser.add_argument("--output", required=True, type=Path, help="private output manifest JSON")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    input_paths = [args.before, args.after]
    input_paths.extend(
        path
        for path in (args.live_points, args.original_file, args.accepted_parse_file)
        if path is not None
    )
    if any(path.resolve().is_relative_to(repo_root) for path in input_paths):
        print(
            "corpus repair manifest: chunk snapshots must be outside the repository",
            file=sys.stderr,
        )
        return 2
    if args.output.resolve().is_relative_to(repo_root):
        print("corpus repair manifest: output must be outside the repository", file=sys.stderr)
        return 2
    try:
        manifest = build_manifest(
            args.before,
            args.after,
            source_id=args.source_id,
            original_sha256=args.original_sha256,
            accepted_parse_sha256=args.accepted_parse_sha256,
            live_points_path=args.live_points,
            original_file=args.original_file,
            accepted_parse_file=args.accepted_parse_file,
            target_collection=args.target_collection,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"corpus repair manifest: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent_mode = stat.S_IMODE(args.output.parent.stat().st_mode)
    args.output.parent.chmod(parent_mode & 0o700)
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    fd, temp_name = tempfile.mkstemp(prefix=".repair-manifest-", dir=args.output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_name, 0o600)
        os.link(temp_name, args.output)
    except FileExistsError:
        print("corpus repair manifest: output already exists; choose a new path", file=sys.stderr)
        return 2
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    print(
        json.dumps(
            {
                "manifest": str(args.output),
                "eligible": manifest["eligible"],
                "changes": len(manifest["changes"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

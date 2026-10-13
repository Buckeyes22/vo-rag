#!/usr/bin/env python3
"""Dump a deterministic ingest manifest for one source from a Qdrant collection.

Parity check: ingest is deterministic, so re-ingesting the SAME source bytes
through the LlamaIndex workflow path must produce byte-identical point IDs,
chunk IDs, and payload cores. Dump a manifest before and after, then `diff`
the two files — any difference is a wiring bug in the new path, full stop.

The manifest is one JSON line per point, sorted by point id, carrying only
content-deterministic fields (volatile/operational fields are excluded).

Usage:
  uv run python scripts/source_ingest_manifest.py \
      --qdrant-url http://127.0.0.1:6400 \
      --collection vo_rag_books_raw_v1 \
      --source-id "vo-rag-corpus-sha256:<hash>" \
      --output /tmp/manifest_before.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx

#: Payload fields that are pure functions of source content + pipeline code.
#: Excluded on purpose: timestamps, colbert vector counts (fallback-dependent),
#: operational/run metadata.
_DETERMINISTIC_FIELDS = (
    "chunk_id",
    "source_id",
    "canonical_source_id",
    "document_id",
    "chunk_type",
    "chapter_path",
    "title",
    "author",
    "domain_tags",
)


def _payload_core(payload: dict[str, Any]) -> dict[str, Any]:
    core = {key: payload.get(key) for key in _DETERMINISTIC_FIELDS}
    content = str(payload.get("content") or "")
    core["content_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return core


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument(
        "--filter-key",
        default="source_id",
        help="payload key to match --source-id against (default: source_id)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_url = args.qdrant_url.rstrip("/")
    rows: list[dict[str, Any]] = []
    offset: Any = None
    with httpx.Client(timeout=30.0) as client:
        while True:
            body: dict[str, Any] = {
                "limit": 1000,
                "with_payload": True,
                "with_vector": False,
                "filter": {"must": [{"key": args.filter_key, "match": {"value": args.source_id}}]},
            }
            if offset is not None:
                body["offset"] = offset
            resp = client.post(f"{base_url}/collections/{args.collection}/points/scroll", json=body)
            resp.raise_for_status()
            result = resp.json()["result"]
            for point in result["points"]:
                rows.append({"point_id": str(point["id"]), **_payload_core(point["payload"])})
            offset = result.get("next_page_offset")
            if offset is None:
                break

    rows.sort(key=lambda r: r["point_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    manifest_hash = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "collection": args.collection,
                "source_id": args.source_id,
                "n_points": len(rows),
                "manifest_path": str(args.output),
                "manifest_sha256": manifest_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

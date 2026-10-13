"""Prove archive tree lineage without fabricating missing Reddit comment IDs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from vo_rag.core.ingest.reddit_archive import post_to_markdown
from vo_rag.core.ingest.repair_evidence import (
    SourceArtifact,
    SourceArtifactRole,
    load_reddit_lineage,
)
from vo_rag.core.ingest.repair_reddit_lineage import (
    archive_comment_records,
    archive_record_digest,
    rendered_archive_occurrences,
    verify_archive_tree_manifest,
)


def fixture() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use identical sibling bodies and a short parent with a promoted reply."""
    body = "This substantive archived response preserves its real tree location."
    post: dict[str, Any] = {
        "title": "Fixture discussion",
        "permalink": "/r/example/comments/abc123/thread/",
        "body": "The retained post body provides original source context.",
        "comments": [
            {"author": "a", "body": body, "replies": []},
            {
                "author": "b",
                "body": "ok",
                "replies": [{"author": "c", "body": body, "replies": []}],
            },
            {"author": "d", "body": body, "replies": []},
        ],
    }
    source_id = (
        "vo-rag-corpus-sha256:"
        + hashlib.sha256(post_to_markdown(post, subreddit="example").encode()).hexdigest()
    )
    records = [(None, 0, post)] + [
        (path, i, comment)
        for i, (path, comment) in enumerate(archive_comment_records(post["comments"]), start=1)
        if len(comment["body"]) >= 40
    ]
    rows = [
        {
            "occurrence_id": f"occurrence-{ordinal}",
            "source_id": source_id,
            "kind": "reddit_post" if path is None else "reddit_comment",
            "subreddit": "example",
            "thread_id": "abc123",
            "post_id": "abc123",
            "canonical_permalink": post["permalink"],
            "ordinal": ordinal,
            "identity_basis": "archive_tree_path",
            "archive_post_index": 0,
            "archive_comment_path": list(path) if path is not None else None,
            "archive_record_sha256": archive_record_digest(record),
            "comment_id": None,
            "parent_comment_id": None,
        }
        for path, ordinal, record in records
    ]
    locations = rendered_archive_occurrences(post, "example")
    for row in rows:
        path = tuple(row["archive_comment_path"]) if row["archive_comment_path"] else None
        row["rendered_start_byte"], row["rendered_end_byte"] = locations[path][2:]
    return [post], rows


def test_repeated_bodies_and_promoted_reply_keep_distinct_actual_paths() -> None:
    """Body equality cannot collapse sibling occurrences or forge a reply parent."""
    archive, rows = fixture()
    verify_archive_tree_manifest(archive, rows)
    assert [r["archive_comment_path"] for r in rows] == [None, [0], [1, 0], [2]]
    assert [r["ordinal"] for r in rows] == [0, 1, 3, 4]
    assert all(r["comment_id"] is None for r in rows)


def test_distinct_fragments_of_one_real_comment_are_accepted() -> None:
    """A long comment can anchor separate chunks without inventing another comment."""
    archive, rows = fixture()
    row = rows[1]
    split = row["rendered_start_byte"] + 20
    second = {**row, "occurrence_id": "later-fragment", "rendered_start_byte": split}
    row["rendered_end_byte"] = split
    rows.append(second)
    verify_archive_tree_manifest(archive, rows)


@pytest.mark.parametrize("which", ["before_record", "past_record", "sibling"])
def test_comment_fragment_cannot_use_another_record(which: str) -> None:
    """Real sibling bodies do not authorize a range outside the declared path."""
    archive, rows = fixture()
    if which == "before_record":
        rows[1]["rendered_start_byte"] -= 1
    elif which == "past_record":
        rows[1]["rendered_end_byte"] += 1
    else:
        rows[1]["rendered_start_byte"] = rows[3]["rendered_start_byte"]
        rows[1]["rendered_end_byte"] = rows[3]["rendered_end_byte"]
    with pytest.raises(ValueError):
        verify_archive_tree_manifest(archive, rows)


@pytest.mark.parametrize(
    "mutation",
    [
        "body",
        "path",
        "ordinal",
        "partial",
        "duplicate",
        "post",
        "source",
        "comment_id",
        "permalink",
        "negative",
        "bool_path",
    ],
)
def test_modified_or_partial_archive_lineage_is_rejected(mutation: str) -> None:
    """Reject independently plausible records that do not bind this exact tree."""
    archive, rows = fixture()
    if mutation == "body":
        archive[0]["comments"][0]["body"] += " changed"
    elif mutation == "path":
        rows[1]["archive_comment_path"] = [2]
    elif mutation == "ordinal":
        rows[2]["ordinal"] = 2
    elif mutation == "partial":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[1]))
    elif mutation == "post":
        rows[0]["archive_post_index"] = 1
    elif mutation == "source":
        rows[1]["source_id"] = "vo-rag-corpus-sha256:" + "a" * 64
    elif mutation == "comment_id":
        rows[1]["comment_id"] = "invented-id"
    elif mutation == "permalink":
        rows[0]["canonical_permalink"] = "/r/example/comments/other/thread/"
    elif mutation == "negative":
        rows[1]["archive_comment_path"] = [-1]
    else:
        rows[1]["archive_comment_path"] = [True]
    with pytest.raises(ValueError):
        verify_archive_tree_manifest(archive, rows)


def test_v2_manifest_loader_rechecks_the_actual_archive(tmp_path: Path) -> None:
    """Public loader verifies archive bytes and source reproduction before admission."""
    archive, rows = fixture()
    ap = tmp_path / "archive.json"
    ap.write_text(json.dumps(archive))
    raw = ap.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    mp = tmp_path / "manifest.json"
    mp.write_text(
        json.dumps(
            {
                "schema": "vo-rag.reddit-occurrence-manifest.v2",
                "archive_sha256": sha,
                "occurrences": rows,
            }
        )
    )
    ar = SourceArtifact(
        role=SourceArtifactRole.REDDIT_ARCHIVE, path=ap, sha256=sha, size_bytes=len(raw)
    )
    mr = mp.read_bytes()
    manifest = SourceArtifact(
        role=SourceArtifactRole.REDDIT_LINEAGE_MANIFEST,
        path=mp,
        sha256=hashlib.sha256(mr).hexdigest(),
        size_bytes=len(mr),
    )
    evidence = load_reddit_lineage(ar, manifest)
    assert len(evidence.occurrences) == 4
    assert evidence.occurrences[2].archive_comment_path == (1, 0)
    assert evidence.occurrences[2].comment_id is None

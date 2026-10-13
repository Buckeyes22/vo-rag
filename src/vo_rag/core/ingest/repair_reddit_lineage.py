"""Verify historical Reddit identities directly against retained archive trees.

Yars retained author/body/replies without Reddit comment IDs. Tree paths are
explicit archive identities, never synthetic Reddit IDs. The normal producer's
exact Markdown digest binds the complete rendered thread to its ingested source.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from vo_rag.core.ingest.reddit_archive import (
    DEFAULT_MIN_COMMENT_CHARS,
    _comment_lines,
    _iter_comments,
    post_id_from_permalink,
    post_to_markdown,
    subreddit_from_permalink,
)


def archive_record_digest(record: Mapping[str, Any]) -> str:
    """Hash every retained field of an actual archive record deterministically."""
    return hashlib.sha256(
        json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def archive_comment_records(
    comments: object, prefix: tuple[int, ...] = ()
) -> Iterator[tuple[tuple[int, ...], Mapping[str, Any]]]:
    """Yield retained comment paths in depth-first archive order, including noise."""
    if not isinstance(comments, Sequence) or isinstance(comments, str | bytes):
        return
    for index, comment in enumerate(comments):
        if not isinstance(comment, Mapping):
            continue
        path = (*prefix, index)
        yield path, comment
        yield from archive_comment_records(comment.get("replies"), path)


def rendered_archive_occurrences(
    post: Mapping[str, Any], subreddit: str
) -> dict[tuple[int, ...] | None, tuple[int, Mapping[str, Any], int, int]]:
    """Locate each actual record in the normal producer's exact Markdown bytes.

    Multiple fragments may reference a long comment. Each must stay inside its
    own rendered record, so an identical sibling body cannot supply its lineage.
    """
    markdown = post_to_markdown(post, subreddit=subreddit).encode()
    tree = list(archive_comment_records(post.get("comments")))
    rendered = list(_iter_comments(post.get("comments"), 0, DEFAULT_MIN_COMMENT_CHARS))
    if not rendered:
        return {None: (0, post, 0, len(markdown))}
    prefix = post_to_markdown({**post, "comments": []}, subreddit=subreddit).encode()
    prefix += b"\n## Comments\n\n"
    if not markdown.startswith(prefix):
        raise ValueError("normal archive producer prefix differs")
    index = {id(comment): (ordinal, path) for ordinal, (path, comment) in enumerate(tree, 1)}
    cursor = len(prefix)
    output: dict[tuple[int, ...] | None, tuple[int, Mapping[str, Any], int, int]] = {
        None: (0, post, 0, cursor)
    }
    reconstructed = bytearray(prefix)
    for depth, comment in rendered:
        ordinal, path = index[id(comment)]
        block = ("\n".join(_comment_lines(comment, depth)) + "\n\n").encode()
        output[path] = (ordinal, comment, cursor, min(cursor + len(block), len(markdown)))
        cursor += len(block)
        reconstructed.extend(block)
    if bytes(reconstructed).rstrip() + b"\n" != markdown:
        raise ValueError("normal archive producer occurrence reconstruction differs")
    return output


def verify_archive_tree_manifest(archive: object, rows: list[dict[str, Any]]) -> None:
    """Require complete rendered occurrences, real paths, order and source digest.

    A changed post, wrong path/body/parent order, guessed Reddit ID, or partial
    rendered comment inventory fails. Dropped short parents stay in the archived
    tree; promoted replies retain their actual path rather than a forged parent.
    """
    if not isinstance(archive, list) or not rows:
        raise ValueError("v2 lineage requires a nonempty manifest and a yars post array")
    by_post: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        index = row.get("archive_post_index")
        if (
            row.get("identity_basis") != "archive_tree_path"
            or type(index) is not int
            or index < 0
            or index >= len(archive)
            or row.get("comment_id") is not None
            or row.get("parent_comment_id") is not None
        ):
            raise ValueError("v2 lineage requires actual archive paths without invented Reddit IDs")
        by_post.setdefault(index, []).append(row)
    for index, occurrences in by_post.items():
        post = archive[index]
        if not isinstance(post, Mapping):
            raise ValueError("archive post is not a record")
        permalink = str(post.get("permalink") or "").strip()
        post_id = post_id_from_permalink(permalink)
        subreddit = subreddit_from_permalink(permalink)
        if not post_id or not subreddit:
            raise ValueError("actual archive permalink does not identify a thread")
        markdown = post_to_markdown(post, subreddit=subreddit).encode()
        source_id = "vo-rag-corpus-sha256:" + hashlib.sha256(markdown).hexdigest()
        expected = rendered_archive_occurrences(post, subreddit)
        seen: set[tuple[int, ...] | None] = set()
        seen_spans: set[tuple[tuple[int, ...] | None, int, int]] = set()
        for row in occurrences:
            value = row.get("archive_comment_path")
            if value is not None and (
                not isinstance(value, list)
                or not value
                or any(type(i) is not int or i < 0 for i in value)
            ):
                raise ValueError("archive comment path is not a nonnegative index sequence")
            occurrence_path = tuple(value) if value is not None else None
            if occurrence_path not in expected:
                raise ValueError("archive occurrence is absent from rendered source")
            seen.add(occurrence_path)
            ordinal, actual, record_start, record_end = expected[occurrence_path]
            start, end = row.get("rendered_start_byte"), row.get("rendered_end_byte")
            if (
                type(start) is not int
                or type(end) is not int
                or not record_start <= start < end <= record_end
                or (occurrence_path, start, end) in seen_spans
            ):
                raise ValueError(
                    "archive fragment is duplicated or outside its actual rendered record"
                )
            seen_spans.add((occurrence_path, start, end))
            kind = "reddit_post" if occurrence_path is None else "reddit_comment"
            if any(
                row.get(key) != val
                for key, val in {
                    "kind": kind,
                    "post_id": post_id,
                    "thread_id": post_id,
                    "subreddit": subreddit,
                    "canonical_permalink": permalink,
                    "source_id": source_id,
                    "ordinal": ordinal,
                    "archive_record_sha256": archive_record_digest(actual),
                }.items()
            ):
                raise ValueError("archive record, order or exact rendered source binding differs")
        if seen != set(expected):
            raise ValueError("archive manifest does not cover the complete rendered thread")

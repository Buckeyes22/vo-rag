"""vo-rag ``brain_sync``: index the brain vault into ``vo_rag_brain_v1``.

The brain is a Git repository of Markdown pages (design notes, infrastructure
facts, copied documentation under sources/<product>/). This pipeline
reads a Git ref directly (never the working tree, never intake) and keeps one
Qdrant collection in sync with it:

- one chunk per Markdown heading section, with the page title as context;
- embeddings from the shared BGE-M3 service, points via ``VoPoint`` /
  ``QdrantStore``, ids via ``deterministic_point_id(page_id, chunk_index,
  "brain", content_hash)`` so re-runs change nothing;
- ``delete_source`` before every re-upsert and for every deleted page;
- ``log.md`` and anything that isn't ``.md`` is skipped;
- the last synced commit and last error are recorded in a small state file so
  callers can tell when the index lags Git;
- ``--rebuild`` drops the collection and re-syncs everything.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import AsyncQdrantClient

from vo_rag.core.embedding.client import EmbeddingClient
from vo_rag.core.embedding.point_vectors import (
    downsample_colbert_vectors,
    fallback_colbert_vectors,
    include_colbert_at_ingest,
    sparse_payload_to_int_map,
)
from vo_rag.core.retrieval.bm25_tokenizer import tokenize, tokens_to_bm25_sparse
from vo_rag.core.storage.points import VoPoint
from vo_rag.core.storage.qdrant import (
    QdrantStore,
    _ensure_collection,
    _ensure_payload_indexes,
    deterministic_point_id,
)

BRAIN_COLLECTION = "vo_rag_brain_v1"
STAGE = "brain"
WRITER_VERSION = "brain-v4-governing-context"
SKIP_BASENAMES = {"log.md"}
DEFAULT_STATE_PATH = Path("~/.local/state/vo-rag/brain_sync.json").expanduser()
POLL_INTERVAL_S = 300
#: A single section longer than this is further split on paragraph breaks so
#: no chunk exceeds what BGE-M3 can embed (8192 tokens ≈ 8k characters).
MAX_CHUNK_CHARS = 8_000


def _budget_batches(texts: list[str], *, max_texts: int, max_chars: int) -> list[list[str]]:
    """Group texts into embed requests that stay inside the service's limits."""
    batches: list[list[str]] = []
    current: list[str] = []
    chars = 0
    for text in texts:
        if current and (len(current) >= max_texts or chars + len(text) > max_chars):
            batches.append(current)
            current, chars = [], 0
        current.append(text)
        chars += len(text)
    if current:
        batches.append(current)
    return batches


class BrainSyncSettings(BaseSettings):
    """Just the endpoints brain_sync needs (same env vars the service uses)."""

    model_config = SettingsConfigDict(extra="ignore")

    qdrant_url: str = "http://127.0.0.1:6400"
    qdrant_api_key: str = ""
    embed_url: str = "http://127.0.0.1:8501/embed"


# --------------------------------------------------------------------------- #
# Git reading
# --------------------------------------------------------------------------- #


_GIT: str = shutil.which("git") or "/usr/bin/git"


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        [_GIT, "-C", str(repo), *args], capture_output=True, text=True, timeout=60
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])} failed: {proc.stderr.strip()[:200]}")
    return proc.stdout


def head_commit(repo: Path, ref: str) -> str:
    return _git(repo, "rev-parse", ref).strip()


def list_markdown(repo: Path, ref: str) -> list[str]:
    """Every committed ``.md`` path at ``ref`` except the skipped ones."""
    out = _git(repo, "ls-tree", "-r", "--name-only", "-z", ref)
    return sorted(
        path
        for path in out.split("\0")
        if path.endswith(".md") and Path(path).name.lower() not in SKIP_BASENAMES
    )


def read_at(repo: Path, ref: str, path: str) -> str:
    return _git(repo, "show", f"{ref}:{path}")


# --------------------------------------------------------------------------- #
# Frontmatter + heading splitter
# --------------------------------------------------------------------------- #

_FRONTMATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6}) (.+?)\s*#*\s*$")


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return (frontmatter dict, body). Unparsable or missing -> ({}, text)."""
    match = _FRONTMATTER.match(text)
    if match is None:
        return {}, text
    try:
        data = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(data, dict):
        return {}, text
    return data, text[match.end() :].lstrip("\n")


@dataclass(frozen=True)
class MarkdownPassage:
    """Keep one local section and its exact governing source occurrences."""

    heading: str
    local_text: str
    context: tuple[tuple[int, str, str], ...]
    content: str
    embedding_section: str


def markdown_passages(body: str, *, title_context: str = "") -> list[MarkdownPassage]:
    """Split source sections with complete H1–H6 ancestor preambles.

    Context comes from original section occurrences, never generated summaries.
    Full source context remains in delivered evidence. When it exceeds the
    embedding bound, only the complete local passage is embedded; ranking text
    is never substituted for returned source evidence.
    """
    sections: list[tuple[str, list[str], tuple[int, ...]]] = [("", [], ())]
    levels: list[int] = [0]
    titles: list[str] = [""]
    stack: list[int] = []
    fence: tuple[str, int] | None = None
    for line in body.splitlines():
        stripped = line.lstrip()
        fence_match = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = (marker[0], len(marker))
            elif (
                marker[0] == fence[0]
                and len(marker) >= fence[1]
                and not stripped[len(marker) :].strip()
            ):
                fence = None
            sections[-1][1].append(line)
            continue
        match = None if fence else _HEADING.match(line)
        if match is None:
            sections[-1][1].append(line)
            continue
        level, heading = len(match.group(1)), match.group(2).strip()
        while stack and levels[stack[-1]] >= level:
            stack.pop()
        ancestors = tuple(stack)
        if sections[0][1]:
            ancestors = (0, *ancestors)
        path = " / ".join(
            [*(titles[i] for i in stack if levels[i] > 1), *([heading] if level > 1 else [])]
        )
        sections.append((path, [line], ancestors))
        levels.append(level)
        titles.append(heading)
        stack.append(len(sections) - 1)

    passages: list[MarkdownPassage] = []
    for section_occurrence, (heading, lines, ancestors) in enumerate(sections):
        text = "\n".join(lines).strip("\n")
        if not any(line.strip() and not _HEADING.match(line) for line in lines):
            continue
        context = tuple(
            (index, sections[index][0], "\n".join(sections[index][1]).strip("\n"))
            for index in ancestors
        )
        prefix = "\n\n".join(value for _, _, value in context)
        if prefix:
            prefix += "\n\n"
        identity_prefix = title_context + (" / " + heading if heading else "")
        local_limit = MAX_CHUNK_CHARS - len(identity_prefix) - 2
        if local_limit <= 0:
            raise ValueError("Markdown identity exceeds chunk budget")
        inline_context = len(prefix) < local_limit // 2
        limit = local_limit - len(prefix) if inline_context else local_limit
        pieces = _split_long(text, limit)
        if len(pieces) > 1:
            # Earlier prose, table headers or bullet subjects in this same
            # original unit can govern facts in later split pieces. Preserve
            # the whole unit; local ranking bounds cannot retire its scope.
            context = (*context, (section_occurrence, heading, text))
        for piece in pieces:
            if any(line.strip() and not _HEADING.match(line) for line in piece.splitlines()):
                full = prefix + (text if len(pieces) > 1 else piece)
                if len((identity_prefix + "\n\n" + full).encode("utf-8")) > 262_144:
                    raise ValueError("Complete governing evidence exceeds delivery budget")
                passages.append(
                    MarkdownPassage(
                        heading, piece, context, full, prefix + piece if inline_context else piece
                    )
                )
    return passages


def split_markdown(body: str) -> list[tuple[str, str]]:
    """Return bounded displayed passages including exact governing context."""
    return [(part.heading, part.content) for part in markdown_passages(body)]


def _split_long(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split bounded Markdown while keeping fences and table headers coherent."""
    if len(text) <= limit:
        return [text]
    # Markdown blocks are grouped before packing; code blank lines are not
    # paragraph boundaries and table rows retain their column headings.
    blocks: list[str] = []
    lines: list[str] = []
    fence: tuple[str, int] | None = None
    all_lines = text.splitlines()
    for line_index, line in enumerate(all_lines):
        table_start = (
            "|" in line
            and line_index + 1 < len(all_lines)
            and re.fullmatch(r"[\s|:-]+", all_lines[line_index + 1])
        )
        if table_start and fence is None and lines:
            blocks.append("\n".join(lines))
            lines = []
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                if lines:
                    blocks.append("\n".join(lines))
                    lines = []
                fence = (token[0], len(token))
            elif (
                token[0] == fence[0]
                and len(token) >= fence[1]
                and not line.strip()[len(token) :].strip()
            ):
                fence = None
                lines.append(line)
                blocks.append("\n".join(lines))
                lines = []
                continue
        if not line.strip() and fence is None and lines:
            blocks.append("\n".join(lines))
            lines = []
        else:
            lines.append(line)
    if lines:
        blocks.append("\n".join(lines))
    parts: list[str] = []
    current = ""
    for block in blocks:
        pieces = _split_block(block, limit)
        for piece in pieces:
            if current and len(current) + len(piece) + 2 > limit:
                parts.append(current)
                current = ""
            current = current + "\n\n" + piece if current else piece
    if current:
        parts.append(current)
    return parts


def _split_block(block: str, limit: int) -> list[str]:
    if len(block) <= limit:
        return [block]
    lines = block.splitlines()
    first_fence = re.match(r"^\s*(`{3,}|~{3,})", lines[0]) if lines else None
    table = len(lines) > 1 and "|" in lines[0] and re.match(r"^[\s|:-]+$", lines[1])
    prefix = lines[0] + "\n" if first_fence else "\n".join(lines[:2]) + "\n" if table else ""
    suffix = "\n" + first_fence.group(1) if first_fence else ""
    closed = bool(
        first_fence
        and re.fullmatch(
            re.escape(first_fence.group(1)[0]) + rf"{{{len(first_fence.group(1))},}}\s*",
            lines[-1],
        )
    )
    body = "\n".join(
        lines[1:-1] if closed else lines[1:] if first_fence else lines[2:] if table else lines
    )
    budget = limit - len(prefix) - len(suffix)
    if budget < min(1_024, limit // 4):
        # Imported HTML tables can flatten an entire document into a header,
        # and fence declarations can themselves exceed the excerpt budget.
        # Such wrappers cannot be repeated in bounded excerpts. Preserve the
        # complete source once as plain fragments, including its declaration,
        # rather than rejecting the page or truncating knowledge to fit syntax.
        prefix = suffix = ""
        body = block
        budget = limit
    pieces: list[str] = []
    while body:
        cut = min(len(body), budget)
        if cut < len(body):
            boundary = body.rfind("\n", 0, cut)
            if boundary > 0:
                cut = boundary + 1
            else:
                # Keep ordinary words/identifiers intact when a flattened
                # header or prose line has no newline within the budget.
                for separator in re.finditer(r"\s+", body[:cut]):
                    boundary = separator.end()
                if boundary > 0:
                    cut = boundary
        pieces.append(prefix + body[:cut].rstrip("\n") + suffix)
        body = body[cut:]
    return pieces


# --------------------------------------------------------------------------- #
# Sync state
# --------------------------------------------------------------------------- #


def load_state(state_path: Path) -> dict[str, Any]:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return state if isinstance(state, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(state_path)


def plan_sync(
    previous_files: dict[str, Any], current: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Diff last-synced page hashes against the ref: (changed, deleted) paths."""
    changed = sorted(path for path, digest in current.items() if previous_files.get(path) != digest)
    deleted = sorted(set(previous_files) - set(current))
    return changed, deleted


# --------------------------------------------------------------------------- #
# Point building
# --------------------------------------------------------------------------- #


def page_payload(
    *,
    path: str,
    title: str | None,
    heading: str,
    commit: str,
    frontmatter: dict[str, Any],
    content: str,
) -> dict[str, Any]:
    is_copy = path.startswith("sources/")
    return {
        "content": content,
        "stage": STAGE,
        "path": path,
        "page_id": path,
        "source_id": path,
        # The retriever's intent_aware visual mode filters on chunk_type;
        # brain chunks are all text.
        "chunk_type": "text",
        "heading": heading or "(page top)",
        "commit": commit,
        "updated": frontmatter.get("updated"),
        "kind": "source" if is_copy else "page",
        "product": frontmatter.get("product"),
        # Explicit applicability cannot be reconstructed from a source URL or
        # incidental prose; preserve the authoritative declared scope.
        "provider": frontmatter.get("provider"),
        "platform": frontmatter.get("platform"),
        "service": frontmatter.get("service"),
        "offering": frontmatter.get("offering"),
        "edition": frontmatter.get("edition"),
        "region": frontmatter.get("region"),
        "applicability": frontmatter.get("applicability"),
        "version": frontmatter.get("version"),
        "url": frontmatter.get("url") if is_copy else None,
        "fetched_at": frontmatter.get("fetched_at") if is_copy else None,
        "concept_id": frontmatter.get("concept_id"),
        "source_origin": frontmatter.get("source_origin"),
        "source_lineage": frontmatter.get("source_lineage"),
        "status": frontmatter.get("status"),
        "title": title,
        "tags": frontmatter.get("tags") or [],
        "subject_type": frontmatter.get("subject_type"),
        "page_area": path.split("/", 1)[0] if "/" in path else "root",
        "writer_version": WRITER_VERSION,
    }


def embedding_text(path: str, title: str | None, heading: str, section: str) -> str:
    """Render consistent title and heading context for vectors and hits."""
    context = title or Path(path).stem
    if heading:
        context += " / " + heading
    return context + "\n\n" + section


def build_points(
    *,
    path: str,
    title: str | None,
    frontmatter: dict[str, Any],
    commit: str,
    chunks: list[tuple[str, str]],
    embeddings: dict[str, list[Any]],
    content_hash: str,
    passage_details: list[MarkdownPassage] | None = None,
) -> list[VoPoint]:
    """Turn one page's chunks + embed rows into VoPoints (mirrors book_rag)."""
    dense_rows = embeddings.get("dense") or []
    sparse_rows = embeddings.get("sparse") or []
    colbert_rows = embeddings.get("colbert") or []
    want_colbert = include_colbert_at_ingest()

    points: list[VoPoint] = []
    for chunk_index, (heading, section_text) in enumerate(chunks):
        # The page title is the shared context line for every chunk.
        content = embedding_text(path, title, heading, section_text)
        dense = list(dense_rows[chunk_index]) if chunk_index < len(dense_rows) else []
        sparse = (
            sparse_payload_to_int_map(sparse_rows[chunk_index])
            if chunk_index < len(sparse_rows)
            else {}
        )
        colbert_raw = colbert_rows[chunk_index] if chunk_index < len(colbert_rows) else None
        if want_colbert and colbert_raw:
            colbert = downsample_colbert_vectors(colbert_raw)
        else:
            colbert = fallback_colbert_vectors(dense)
        payload = page_payload(
            path=path,
            title=title,
            heading=heading,
            commit=commit,
            frontmatter=frontmatter,
            content=content,
        )
        payload["chunk_id"] = hashlib.sha256(
            f"{path}\0{chunk_index}\0{heading}\0{section_text}".encode()
        ).hexdigest()
        payload["chunk_index"] = chunk_index
        if passage_details is not None:
            detail = passage_details[chunk_index]
            payload["source_section_text"] = detail.local_text
            payload["governing_context_version"] = 1
            payload["governing_context_complete"] = True
            payload["embedding_text"] = embedding_text(
                path, title, heading, detail.embedding_section
            )
            payload["governing_context"] = [
                {
                    "section_occurrence": occurrence,
                    "heading": label,
                    "text": text,
                    "source_id": path,
                    "collection": BRAIN_COLLECTION,
                    "point_id": deterministic_point_id(
                        path,
                        chunk_index,
                        STAGE,
                        hashlib.sha256(f"{WRITER_VERSION}:{content_hash}".encode()).hexdigest(),
                    ),
                    "source_revision": commit,
                    "content_hash": hashlib.sha256(text.encode()).hexdigest(),
                }
                for occurrence, label, text in detail.context
            ]
        points.append(
            VoPoint(
                source_id=path,
                chunk_index=chunk_index,
                stage=STAGE,
                version_hash=hashlib.sha256(
                    f"{WRITER_VERSION}:{content_hash}".encode()
                ).hexdigest(),
                dense=dense,
                sparse=sparse,
                bm25=tokens_to_bm25_sparse(tokenize(content)),
                colbert=colbert,
                payload=payload,
            )
        )
    return points


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #


def resolve_ref(repo: Path, ref: str, *, fetch: bool = True) -> tuple[str, list[str]]:
    """Fetch origin and derive the ref to actually read.

    With fetch on, a local-only ref like `main` becomes `origin/main`, so the
    poller indexes what the fetch brought in instead of a stale local branch.
    A failed fetch degrades to the last-fetched state (it never crashes the
    poller); that is reported as an error, not fatal.
    """
    errors: list[str] = []
    if fetch:
        try:
            _git(repo, "fetch", "origin", ref)
        except RuntimeError as exc:
            errors.append(f"fetch failed (indexing local {ref}): {exc}")
        if "/" not in ref:
            ref = f"origin/{ref}"
    return ref, errors


async def sync_once(
    repo: Path,
    ref: str,
    state_path: Path = DEFAULT_STATE_PATH,
    *,
    settings: BrainSyncSettings | None = None,
    rebuild: bool = False,
    fetch: bool = True,
) -> dict[str, Any]:
    """One sync pass. Returns a summary dict; never raises for a bad page.

    The pass fetches `origin` first by default, so a poller's local clone sees
    the remote's `main` without a separate cron step; `--no-fetch` turns that
    off for reads of a purely local ref.
    """
    settings = settings or BrainSyncSettings()
    started = time.monotonic()
    previous_state = {} if rebuild else load_state(state_path)
    previous_files = previous_state.get("files") or {}
    previous_versions = previous_state.get("file_writer_versions") or {}

    summary: dict[str, Any] = {"ref": ref, "rebuild": rebuild, "errors": []}
    try:
        ref, fetch_errors = resolve_ref(repo, ref, fetch=fetch)
        summary["errors"].extend(fetch_errors)
        summary["ref"] = ref
        commit = head_commit(repo, ref)
        paths = list_markdown(repo, ref)
    except RuntimeError as exc:
        summary["error"] = str(exc)
        state = {**previous_state, "last_error": str(exc), "last_error_at": _now()}
        save_state(state_path, state)
        summary["state"] = state
        return summary

    # Read + hash every page. A file that fails to read is skipped and reported.
    current: dict[str, str] = {}
    contents: dict[str, str] = {}
    for path in paths:
        try:
            text = read_at(repo, ref, path)
        except RuntimeError as exc:
            summary["errors"].append(f"{path}: {exc}")
            continue
        contents[path] = text
        current[path] = hashlib.sha256(text.encode("utf-8")).hexdigest()

    changed, deleted = plan_sync(previous_files, current)
    # A listed page whose read failed still exists. Only a confirmed Git
    # removal may delete its last searchable version.
    deleted = sorted(set(previous_files) - set(paths))
    changed = sorted(
        set(changed) | {path for path in current if previous_versions.get(path) != WRITER_VERSION}
    )
    summary.update(
        commit=commit, pages=len(paths), changed=len(changed), deleted=len(deleted), points=0
    )

    client = AsyncQdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, prefer_grpc=False
    )
    store = QdrantStore(
        settings.qdrant_url, api_key=settings.qdrant_api_key or None, prefer_grpc=False
    )
    try:
        if rebuild:
            await client.delete_collection(BRAIN_COLLECTION)
        await _ensure_collection(client, BRAIN_COLLECTION)
        await _ensure_payload_indexes(client, BRAIN_COLLECTION)

        completed_files = dict(previous_files)
        completed_versions = dict(previous_versions)

        def checkpoint(path: str) -> None:
            completed_files[path] = current[path]
            completed_versions[path] = WRITER_VERSION
            save_state(
                state_path,
                {
                    **previous_state,
                    "files": completed_files,
                    "file_writer_versions": completed_versions,
                    "migration_target_version": WRITER_VERSION,
                    "last_error": "writer migration/sync in progress",
                },
            )

        for path in deleted:
            await store.delete_source(BRAIN_COLLECTION, path)
            completed_files.pop(path, None)
            completed_versions.pop(path, None)

        upserted: list[str] = []
        page_errors: list[str] = []
        point_count = 0
        if changed:
            upserted, point_count, page_errors = await _upsert_pages(
                store, settings, repo, ref, commit, changed, contents, current, checkpoint
            )
        summary["errors"].extend(page_errors)
        summary["points"] = point_count

        # Only pages that actually synced advance the recorded hashes, so a
        # failed page is retried (and its old points stay until it succeeds).
        state = {
            "files": completed_files,
            "file_writer_versions": completed_versions,
            "writer_version": WRITER_VERSION,
            "last_commit": commit if not summary["errors"] else previous_state.get("last_commit"),
            "last_ref": ref,
            "last_sync_at": _now(),
            "last_error": "; ".join(summary["errors"]) if summary["errors"] else None,
        }
        save_state(state_path, state)
        summary["state"] = state
        summary["seconds"] = round(time.monotonic() - started, 1)
        return summary
    finally:
        await client.close()
        # QdrantStore wraps its own client with no close() surface.
        await store._client.close()


async def _upsert_pages(
    store: QdrantStore,
    settings: BrainSyncSettings,
    repo: Path,
    ref: str,
    commit: str,
    changed: list[str],
    contents: dict[str, str],
    hashes: dict[str, str],
    checkpoint: Callable[[str], None] | None = None,
) -> tuple[list[str], int, list[str]]:
    """Embed + upsert every changed page.

    Returns (successfully synced paths, points upserted, "path: error"
    strings). A bad page is skipped and reported; it never blocks the others.
    """
    synced: list[str] = []
    errors: list[str] = []
    point_count = 0
    embedder = EmbeddingClient(settings.embed_url, timeout_s=60.0)
    try:
        for path in changed:
            try:
                front, body = parse_frontmatter(contents[path])
                title = front.get("title") if isinstance(front.get("title"), str) else None
                details = markdown_passages(body, title_context=title or Path(path).stem)
                chunks = [(part.heading, part.content) for part in details]
                if not chunks:
                    await store.delete_source(BRAIN_COLLECTION, path)
                else:
                    texts = [
                        embedding_text(path, title, part.heading, part.embedding_section)
                        for part in details
                    ]
                    flat: dict[str, list[Any]] = {"dense": [], "sparse": []}
                    if include_colbert_at_ingest():
                        flat["colbert"] = []
                    for batch in _budget_batches(texts, max_texts=32, max_chars=12_000):
                        for attempt in range(2):
                            try:
                                rows = await embedder.embed(
                                    batch,
                                    return_dense=True,
                                    return_sparse=True,
                                    return_colbert=include_colbert_at_ingest(),
                                )
                                break
                            except httpx.HTTPStatusError as exc:
                                if exc.response.status_code != 500 or attempt == 1:
                                    raise
                                await asyncio.sleep(2.0)
                        for key in flat:
                            flat[key].extend(rows[key])
                    points = build_points(
                        path=path,
                        title=title,
                        frontmatter=front,
                        commit=commit,
                        chunks=chunks,
                        embeddings=flat,
                        content_hash=hashes[path],
                        passage_details=details,
                    )
                    # Keep the old searchable version until the replacement is
                    # acknowledged. Interrupted cleanup repeats idempotently.
                    await store.upsert_vectors(BRAIN_COLLECTION, points)
                    await store.delete_source_except(BRAIN_COLLECTION, path, points[0].version_hash)
                    point_count += len(points)
                synced.append(path)
                if checkpoint:
                    checkpoint(path)
            except Exception as exc:  # noqa: BLE001 — isolate bad pages
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
    finally:
        await embedder.aclose()
    return synced, point_count, errors


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


async def _run(args: argparse.Namespace) -> int:
    settings = BrainSyncSettings()
    while True:
        summary = await sync_once(
            args.repo,
            args.ref,
            args.state,
            settings=settings,
            rebuild=args.rebuild,
            fetch=not args.no_fetch,
        )
        args.rebuild = False  # rebuild applies to the first pass only
        print(
            f"brain_sync {args.repo}@{summary.get('ref', args.ref)} "
            f"commit={summary.get('commit', '?')} "
            f"pages={summary.get('pages', 0)} changed={summary.get('changed', 0)} "
            f"deleted={summary.get('deleted', 0)} points={summary.get('points', 0)} "
            f"errors={len(summary.get('errors', []))}"
        )
        for error in summary.get("errors", []):
            print(f"  error: {error}")
        if summary.get("error"):
            print(f"  fatal: {summary['error']}")
        if not args.loop:
            return 1 if summary.get("errors") or summary.get("error") else 0
        await asyncio.sleep(args.interval_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vo-rag brain_sync", description="Index the brain vault into vo_rag_brain_v1."
    )
    parser.add_argument("--repo", type=Path, default=Path("./var/brain"))
    parser.add_argument("--ref", default="main", help="brain Git ref to index (default: main)")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--loop", action="store_true", help="poll forever")
    parser.add_argument("--interval-s", type=int, default=POLL_INTERVAL_S)
    parser.add_argument("--rebuild", action="store_true", help="drop the collection first")
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="do not fetch origin first (index the clone exactly as it is)",
    )
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "BrainSyncSettings",
    "build_points",
    "embedding_text",
    "main",
    "plan_sync",
    "split_markdown",
    "sync_once",
]

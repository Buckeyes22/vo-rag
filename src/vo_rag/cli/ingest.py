"""Ingest CLI commands."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import hashlib
import json
import mimetypes
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO, cast

import click
import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from vo_rag.cli.client import VoRagApiClient, _client_from_ctx
from vo_rag.cli.render import emit_json
from vo_rag.web.url_safety import SSRFBlocked, validate_external_url

mimetypes.add_type("application/x-mobipocket-ebook", ".mobi")

#: How many header bytes to sniff for magic bytes (VOR-1434). Covers
#: MOBI's ``BOOKMOBI`` marker at offset 60 and an EPUB zip's ``mimetype``
#: entry, whose name+content sit within the first local-file-header record.
_SNIFF_HEAD_BYTES = 4096
_MOBI_MAGIC_OFFSET = 60
_MOBI_MAGIC = b"BOOKMOBI"


def _sniff_content_type(path: Path) -> str:
    """Content type for an upload's ``Content-Type`` header: magic bytes
    first, then the extension, then ``application/octet-stream`` — never
    empty/``None``.

    An upload sent without a content type is rejected server-side with
    ``content_type_not_allowed: actual content type is missing`` (VOR-1434), and
    an extension-only guess is fragile (renamed files, missing/odd extensions).
    Only the three formats this CLI actually verifies by magic bytes
    elsewhere are sniffed here; everything else
    falls back to ``mimetypes.guess_type``.
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(_SNIFF_HEAD_BYTES)
    except OSError:
        head = b""
    if head.startswith(b"%PDF-"):
        return "application/pdf"
    if head.startswith(b"PK\x03\x04") and b"application/epub+zip" in head:
        return "application/epub+zip"
    if head[_MOBI_MAGIC_OFFSET : _MOBI_MAGIC_OFFSET + len(_MOBI_MAGIC)] == _MOBI_MAGIC:
        return "application/x-mobipocket-ebook"
    guessed = mimetypes.guess_type(path.name)[0]
    return guessed or "application/octet-stream"


def register_ingest_commands(root: click.Group) -> None:
    @root.group("ingest")
    def ingest() -> None:
        """Ingest text, URL, idea, or file inputs."""

    @ingest.command("text")
    @click.argument("text", type=str)
    @click.option("--title", default=None)
    @click.option("--tags", default="")
    @click.option("--sovereignty", default="cloud_ok")
    @click.pass_context
    def ingest_text(
        ctx: click.Context,
        text: str,
        title: str | None,
        tags: str,
        sovereignty: str,
    ) -> None:
        client = _client_from_ctx(ctx)
        payload = {
            "source_type": "text_prompt",
            "raw_content": {"text": text, "title": title},
            "metadata": _metadata(tags, title=title),
            "sovereignty": sovereignty,
        }
        response = asyncio.run(client.post_json("/api/v1/intake", payload))
        _print_envelope(response)

    @ingest.command("url")
    @click.argument("url", type=str)
    @click.option("--title", default=None)
    @click.option("--sovereignty", default="cloud_ok")
    @click.pass_context
    def ingest_url(ctx: click.Context, url: str, title: str | None, sovereignty: str) -> None:
        try:
            asyncio.run(validate_external_url(url))
        except SSRFBlocked as exc:
            raise click.ClickException(f"URL validation failed: {exc}") from exc

        client = _client_from_ctx(ctx)
        payload = {
            "source_type": "url",
            "raw_content": {"url": url, "title": title},
            "metadata": _metadata(None, title=title),
            "sovereignty": sovereignty,
        }
        response = asyncio.run(client.post_json("/api/v1/intake", payload))
        _print_envelope(response)

    @ingest.command("idea")
    @click.argument("idea", type=str)
    @click.option("--tags", default="")
    @click.option("--title", default=None)
    @click.option("--sovereignty", default="cloud_ok")
    @click.pass_context
    def ingest_idea(
        ctx: click.Context,
        idea: str,
        tags: str,
        title: str | None,
        sovereignty: str,
    ) -> None:
        client = _client_from_ctx(ctx)
        tag_list = _split_tags(tags)
        if "idea" not in tag_list:
            tag_list.append("idea")

        payload = {
            "source_type": "text_prompt",
            "raw_content": {"text": idea, "title": title},
            "metadata": {"tags": tag_list, "title": title},
            "sovereignty": sovereignty,
        }
        response = asyncio.run(client.post_json("/api/v1/intake", payload))
        _print_envelope(response)

    @ingest.command("file-drop")
    @click.argument("targets", nargs=-1, required=True)
    @click.option("--title", default=None)
    @click.option("--author", default=None)
    @click.option("--tags", default="")
    @click.option("--sovereignty", default="cloud_ok")
    @click.option(
        "--preflight",
        "preflight_path",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="a `vo-rag preflight --out report.json` report. "
        "TARGETS become local file paths (uploaded like `ingest file`), each matched "
        "against the report by path, else by its computed sha256, and its "
        "metadata.preflight = {route, pages, kind, verdict, sha256} attached. A file "
        "the report marks reject is skipped (non-zero exit) unless --allow-rejects.",
    )
    @click.option(
        "--allow-rejects",
        is_flag=True,
        default=False,
        help="submit files the preflight report marks reject instead of skipping them",
    )
    @click.pass_context
    def ingest_file_drop(
        ctx: click.Context,
        targets: tuple[str, ...],
        title: str | None,
        author: str | None,
        tags: str,
        sovereignty: str,
        preflight_path: Path | None,
        allow_rejects: bool,
    ) -> None:
        client = _client_from_ctx(ctx)
        if preflight_path is None:
            # Legacy behaviour: each target is a garage_key already uploaded
            # to Garage -- no local file access, no upload.
            for garage_key in targets:
                payload = {
                    "source_type": "file_drop",
                    "source_data": {"garage_key": garage_key},
                    "raw_content": {"garage_key": garage_key},
                    "metadata": {
                        "title": title,
                        "author": author,
                        "tags": _split_tags(tags),
                    },
                    "sovereignty": sovereignty,
                }
                response = asyncio.run(client.post_json("/api/v1/intake", payload))
                _print_envelope(response)
            return

        entries = _load_preflight_entries(preflight_path)
        skipped = 0
        for target in targets:
            path = Path(target)
            entry = _match_preflight_by_path(entries, path)
            if entry is not None and entry.get("verdict") == "reject" and not allow_rejects:
                click.echo(f"skipping {path} (preflight verdict: reject)", err=True)
                skipped += 1
                continue

            submitted, skip_verdict = asyncio.run(
                _ingest_file_drop_with_preflight(
                    client=client,
                    path=path,
                    title=title,
                    author=author,
                    tags=_split_tags(tags),
                    sovereignty=sovereignty,
                    entries=entries,
                    matched_by_path=entry,
                    allow_rejects=allow_rejects,
                )
            )
            if submitted is None:
                click.echo(
                    f"skipping {path} (preflight verdict: {skip_verdict}, matched by sha256)",
                    err=True,
                )
                skipped += 1
                continue
            _print_envelope(submitted)

        if skipped and not allow_rejects:
            raise click.ClickException(
                f"{skipped} file(s) skipped: preflight verdict=reject "
                "(pass --allow-rejects to submit them anyway)"
            )

    @ingest.command("file")
    @click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
    @click.option("--title", default=None)
    @click.option("--author", default=None)
    @click.option("--tags", default="")
    @click.option("--sovereignty", default="cloud_ok")
    @click.pass_context
    def ingest_file(
        ctx: click.Context,
        path: Path,
        title: str | None,
        author: str | None,
        tags: str,
        sovereignty: str,
    ) -> None:
        client = _client_from_ctx(ctx)
        response = asyncio.run(
            _ingest_file_impl(
                client=client,
                path=path,
                title=title,
                author=author,
                tags=_split_tags(tags),
                sovereignty=sovereignty,
            )
        )
        _print_envelope(response)

    @ingest.command("run")
    @click.option("--envelope-id", required=True, help="Stored intake envelope id to run.")
    @click.option(
        "--chunking-strategy",
        type=click.Choice(["hierarchical", "semantic"]),
        default=None,
        envvar="VORAG_CHUNKING_STRATEGY",
        help="Override chunking.strategy for this ingest run.",
    )
    def ingest_run_default(envelope_id: str, chunking_strategy: str | None) -> None:
        """Run one stored envelope through the book_rag ingest DAG."""
        output = asyncio.run(
            _run_book_rag_envelope_impl(envelope_id, chunking_strategy=chunking_strategy)
        )
        emit_json(dataclasses.asdict(output))


async def _upload_and_hash(client: VoRagApiClient, path: Path) -> tuple[str, str, str]:
    """Sniff PATH's content type, request an upload URL, and PUT its bytes.

    Returns ``(mime, garage_key, file_hash)``. Shared by ``ingest file`` and
    the bulk ``ingest file-drop --preflight`` path (VOR-1434) so a
    reject decided only after upload (a sha256-only match) never re-reads or
    re-hashes the file.
    """
    mime = _sniff_content_type(path)
    upload_request = {
        "filename": path.name,
        "content_type": mime,
        "size_bytes": path.stat().st_size,
    }
    upload = await client.post_json("/api/v1/intake/upload-url", upload_request)
    upload_url = str(upload["upload_url"])
    garage_key = str(upload["garage_key"])
    file_hash = await asyncio.to_thread(_upload_file_sync, path, upload_url, mime)
    return mime, garage_key, file_hash


def _file_drop_payload(
    *,
    path: Path,
    garage_key: str,
    mime: str,
    file_hash: str,
    title: str | None,
    author: str | None,
    tags: list[str],
    sovereignty: str,
    preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "title": title,
        "author": author,
        "tags": tags,
    }
    if preflight is not None:
        metadata["preflight"] = preflight
    return {
        "source_type": "file_drop",
        "source_data": {"garage_key": garage_key},
        "raw_content": {
            "garage_key": garage_key,
            "filename": path.name,
            "content_type": mime,
            "size_bytes": path.stat().st_size,
            "file_hash": file_hash,
        },
        "metadata": metadata,
        "sovereignty": sovereignty,
    }


async def _ingest_file_impl(
    *,
    client: VoRagApiClient,
    path: Path,
    title: str | None,
    author: str | None,
    tags: list[str],
    sovereignty: str,
    preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mime, garage_key, file_hash = await _upload_and_hash(client, path)
    payload = _file_drop_payload(
        path=path,
        garage_key=garage_key,
        mime=mime,
        file_hash=file_hash,
        title=title,
        author=author,
        tags=tags,
        sovereignty=sovereignty,
        preflight=preflight,
    )
    return await client.post_json("/api/v1/intake", payload)


async def _ingest_file_drop_with_preflight(
    *,
    client: VoRagApiClient,
    path: Path,
    title: str | None,
    author: str | None,
    tags: list[str],
    sovereignty: str,
    entries: list[dict[str, Any]],
    matched_by_path: dict[str, Any] | None,
    allow_rejects: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    """Upload PATH and submit it for ``ingest file-drop --preflight``.

    Returns ``(response, skip_verdict)`` — exactly one is ``None``. When
    ``matched_by_path`` is ``None`` (the report used a different path
    representation, or the caller listed the file by garage key), the report
    entry is looked up again by the sha256 computed during upload before
    deciding whether to submit.
    """
    mime, garage_key, file_hash = await _upload_and_hash(client, path)
    entry = (
        matched_by_path
        if matched_by_path is not None
        else _match_preflight_by_sha256(entries, file_hash)
    )
    if entry is not None and entry.get("verdict") == "reject" and not allow_rejects:
        return None, "reject"

    payload = _file_drop_payload(
        path=path,
        garage_key=garage_key,
        mime=mime,
        file_hash=file_hash,
        title=title,
        author=author,
        tags=tags,
        sovereignty=sovereignty,
        preflight=_preflight_metadata(entry) if entry is not None else None,
    )
    response = await client.post_json("/api/v1/intake", payload)
    return response, None


def _load_preflight_entries(preflight_path: Path) -> list[dict[str, Any]]:
    """``results`` from a ``vo-rag preflight --out report.json`` report."""
    data = json.loads(preflight_path.read_text(encoding="utf-8"))
    results = data.get("results") if isinstance(data, dict) else None
    return [r for r in results if isinstance(r, dict)] if isinstance(results, list) else []


def _match_preflight_by_path(entries: list[dict[str, Any]], path: Path) -> dict[str, Any] | None:
    """The report entry for PATH, matched by its literal or resolved form."""
    candidates = {str(path)}
    with contextlib.suppress(OSError):
        candidates.add(str(path.resolve()))
    for entry in entries:
        entry_path = entry.get("path")
        if not entry_path:
            continue
        if entry_path in candidates:
            return entry
        with contextlib.suppress(OSError):
            if str(Path(entry_path).resolve()) in candidates:
                return entry
    return None


def _match_preflight_by_sha256(entries: list[dict[str, Any]], sha256: str) -> dict[str, Any] | None:
    for entry in entries:
        if entry.get("sha256") == sha256:
            return entry
    return None


def _preflight_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    """``metadata.preflight`` shape written on the submitted envelope."""
    return {
        "route": entry.get("route"),
        "pages": entry.get("pages"),
        "kind": entry.get("kind"),
        "verdict": entry.get("verdict"),
        "sha256": entry.get("sha256"),
    }


def _llamaindex_extra_present() -> bool:
    """True when the optional ``llamaindex`` extra is installed.

    ``vo-rag ingest run`` streams live per-step progress through the LlamaIndex
    ``BookRagWorkflow`` when the extra is present, and falls back to the
    non-streaming core runner otherwise. book_rag ingest is core functionality
    that ships in the base install, so the CLI must never hard-require the
    optional extra (the extra stays optional). Importing
    ``book_rag.events`` (which subclasses ``llama_index.core.workflow.Event``)
    is what would couple this command to the extra — hence the spec probe.
    """
    import importlib.util

    try:
        return importlib.util.find_spec("llama_index.core") is not None
    except ModuleNotFoundError:
        return False


async def _run_book_rag_envelope_impl(
    envelope_id: str, *, chunking_strategy: str | None = None
) -> Any:
    import logging

    from vo_rag.core.config import get_settings
    from vo_rag.core.db.postgres import close_pool, get_pool
    from vo_rag.core.db.queries.envelopes import fail_envelope, get_envelope

    log = logging.getLogger("vo_rag.cli.ingest")
    settings = _settings_with_chunking_strategy(get_settings(), chunking_strategy)
    pool = await get_pool(
        settings.database.url,
        min_size=settings.database.pool_min,
        max_size=settings.database.pool_max,
    )
    try:
        envelope = await get_envelope(pool, envelope_id)
        if envelope is None:
            raise click.ClickException(f"envelope not found: {envelope_id}")
        try:
            if _llamaindex_extra_present():
                return await _run_book_rag_streamed(envelope, settings, pool)
            return await _run_book_rag_unstreamed(envelope, settings, pool)
        except Exception as exc:
            # Persist FAILED so a workflow failure does not leave the envelope
            # stuck at `received`. The SSE route and MCP tool persist terminal
            # status at their own surface; the CLI `ingest run`
            # direct path persists it here (without this, a parse quality-gate
            # failure left status=received).
            try:
                await fail_envelope(pool, str(envelope.id), str(exc) or exc.__class__.__name__)
            except Exception as persist_exc:
                log.error(
                    "book_rag_cli_ingest_fail_persist_failed",
                    extra={"envelope_id": str(envelope.id)},
                    exc_info=persist_exc,
                )
            raise
    finally:
        await close_pool()


def _settings_with_chunking_strategy(settings: Any, strategy: str | None) -> Any:
    if strategy is None:
        return settings

    chunking = getattr(settings, "chunking", None)
    if chunking is None:
        raise click.ClickException("settings.chunking is required for --chunking-strategy")

    model_copy = getattr(chunking, "model_copy", None)
    if callable(model_copy):
        updated_chunking = model_copy(update={"strategy": strategy})
    else:
        updated_chunking = copy.copy(chunking)
        updated_chunking.strategy = strategy

    settings_model_copy = getattr(settings, "model_copy", None)
    if callable(settings_model_copy):
        return settings_model_copy(update={"chunking": updated_chunking})

    updated_settings = copy.copy(settings)
    updated_settings.chunking = updated_chunking
    return updated_settings


async def _run_book_rag_streamed(envelope: Any, settings: Any, pool: Any) -> Any:
    """Stream per-step ingest progress through ``BookRagWorkflow``.

    Requires the optional ``llamaindex`` extra (``events`` subclasses the
    LlamaIndex ``Event``). Selected only when :func:`_llamaindex_extra_present`.
    """
    from vo_rag.pipelines.book_rag.events import INGEST_STEP_ORDER, IngestProgress
    from vo_rag.pipelines.book_rag.runner import stream_book_rag_ingest
    from vo_rag.pipelines.book_rag.workflow import PipelineOutput

    terminal: PipelineOutput | None = None
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("ingest", total=len(INGEST_STEP_ORDER))
        async for ev in stream_book_rag_ingest(envelope, settings=settings, pg_pool=pool):
            if isinstance(ev, IngestProgress):
                progress.update(
                    task,
                    description=f"{ev.step}: {ev.detail}",
                    completed=INGEST_STEP_ORDER.index(ev.step) + 1,
                )
            elif isinstance(ev, PipelineOutput):
                terminal = ev

    if terminal is None:
        raise click.ClickException("ingest stream completed without output")
    for warning in terminal.quality_warnings:
        click.echo(
            f"quality warning: {warning.get('stage', 'quality')}:{warning.get('code', 'unknown')}"
        )
    return terminal


async def _run_book_rag_unstreamed(envelope: Any, settings: Any, pool: Any) -> Any:
    """Non-streaming ingest fallback for base installs (no ``llamaindex`` extra).

    Runs the identical book_rag pipeline via the core
    :func:`~vo_rag.pipelines.book_rag.runner.run_book_rag_ingest` — only the
    live progress stream is unavailable without the extra.
    """
    from vo_rag.pipelines.book_rag.runner import run_book_rag_ingest

    click.echo("ingesting (install the 'llamaindex' extra for live progress)…")
    return await run_book_rag_ingest(envelope, settings=settings, pg_pool=pool)


def _upload_file_sync(path: Path, upload_url: str, content_type: str) -> str:
    size_bytes = path.stat().st_size
    digest = hashlib.sha256()
    with Progress(
        TextColumn("[bold blue]upload"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        # stdout carries the command's single JSON envelope (VOR-1430).
        console=Console(stderr=True),
    ) as progress:
        task_id = progress.add_task(path.name, total=size_bytes)
        with path.open("rb") as fh:
            response = httpx.put(
                upload_url,
                headers={"Content-Type": content_type, "Content-Length": str(size_bytes)},
                content=_iter_file_chunks(cast(BinaryIO, fh), progress, task_id, hasher=digest),
                timeout=120.0,
            )
        response.raise_for_status()
    return digest.hexdigest()


def _iter_file_chunks(
    fh: BinaryIO,
    progress: Progress,
    task_id: TaskID,
    chunk_size: int = 1024 * 1024,
    *,
    hasher: Any | None = None,
) -> Iterator[bytes]:
    while True:
        chunk = fh.read(chunk_size)
        if not chunk:
            break
        if hasher is not None:
            hasher.update(chunk)
        progress.advance(task_id, len(chunk))
        yield chunk


def _metadata(tags: str | None, *, title: str | None = None) -> dict[str, Any]:
    return {
        "title": title,
        "tags": _split_tags(tags),
    }


def _split_tags(tags: str | None) -> list[str]:
    if not tags:
        return []
    return [tag.strip() for tag in tags.split(",") if tag.strip()]


def _print_envelope(response: dict[str, Any]) -> None:
    envelope_id = response.get("envelope_id")
    if envelope_id is None:
        raise click.ClickException("missing envelope_id in API response")
    click.echo(str(envelope_id))

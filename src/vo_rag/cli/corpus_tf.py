"""`vo-rag corpus tf-books` / `tf-chunks` — track-and-field contextual tagging
(VOR-1443). Runs on the API host; classification itself happens in
scripts/tf_classify_lane.py, which consumes the job files written here."""

from __future__ import annotations

import asyncio
import csv
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import click

from vo_rag.cli.corpus_ops import _pool_only
from vo_rag.core.corpus.health import scroll_book_payloads
from vo_rag.core.corpus.source_format import SHA_ENVELOPE_MATCH, source_sha256
from vo_rag.core.corpus.tf_prompts import (
    book_prompt,
    chunk_batch_prompt,
    chunk_label,
    chunk_tag_input_sha256,
    chunk_tag_job_sha256,
    chunk_tag_prompt_sha256,
    chunk_tag_recipe_sha256,
)
from vo_rag.core.corpus.tf_schema import (
    TF_TAG_VERSION,
    TF_TAGGER_MODEL,
    ChunkTags,
    TagValidationError,
    chunk_payload,
    legacy_content_sha256,
    validate_batch,
    validate_book,
)
from vo_rag.core.corpus.tf_store import (
    apply_payloads,
    book_chunks,
    book_records,
    chunk_tags_by_hash,
    document_order_key,
    rekey_chunk_tags,
    restamp_source,
    tf_source_ids,
    upsert_book,
    upsert_chunk_tags,
)

#: One row per draft. `title`/`author` are informational (envelope metadata) and
#: `error` carries a validation failure so the reviewer sees the offending value
#: in place; neither is read back by `load`.
_SHEET_COLUMNS = [
    "source_id",
    "title",
    "author",
    "tf_source_type",
    "tf_evidence_type",
    "tf_pub_year",
    "tf_book_athlete_level",
    "tf_book_age_group",
    "tf_book_sex",
    "tf_limitations",
    "rationale",
    "error",
]
#: The columns `load` turns back into a BookTags record.
_TAG_COLUMNS = [c for c in _SHEET_COLUMNS if c not in {"source_id", "title", "author", "error"}]
_LIST_FIELDS = {"tf_book_athlete_level", "tf_book_age_group"}
_FRONT_MATTER_CHUNKS = 15
#: Chunks per model call. 10 is the configuration the v2 prompt was measured at
#: (VOR-1451: 91% strict accuracy, 82 chunks/min, 0 failures on the local model);
#: its answer stays well inside the API call's max_tokens.
_BATCH_DEFAULT = 10


@asynccontextmanager
async def _resources() -> AsyncIterator[tuple[Any, Any]]:
    from vo_rag.cli.corpus_ops import _resources as ops_resources

    async with ops_resources() as (pool, store):
        yield pool, store


#: Title and author per book from its latest completed intake envelope.
#: ``DISTINCT ON (k.source_id)`` keeps the newest of a book's completed
#: envelopes; the envelope is found the canonical way, through the sha256 in the
#: ``source_id`` (``source_format.SHA_ENVELOPE_MATCH``) -- an acquired book's
#: envelope keys its garage by hash, not by ``raw_content.file_hash``. No
#: publication year comes back: nothing writes one, and the model reads the
#: edition's year off the front matter itself.
_BOOK_IDENTITIES_SQL = f"""SELECT DISTINCT ON (k.source_id) k.source_id,
           e.metadata->>'title' AS title, e.metadata->>'author' AS author
    FROM unnest($1::text[], $2::text[]) AS k(source_id, sha256)
    JOIN intake_envelopes e
      ON e.status = 'completed'
     AND {SHA_ENVELOPE_MATCH}
    ORDER BY k.source_id, e.updated_at DESC"""  # nosec B608 -- source_format's own predicate


async def _book_identities(pool: Any, source_ids: list[str]) -> dict[str, tuple[str, str]]:
    """``source_id -> (title, author)`` for every id, from one batched query.

    An id that carries no sha256 is not queried at all -- the chunk payload
    stores no envelope id, so the hash is the only link, exactly as
    ``source_format``'s docstring says -- and falls back to ``(source_id, "")``,
    as does an id with no completed envelope and an envelope that names no
    title.
    """
    keyed = [(sid, sha) for sid in source_ids if (sha := source_sha256(sid)) is not None]
    found: dict[str, tuple[str, str]] = {}
    if keyed:
        rows = await pool.fetch(
            _BOOK_IDENTITIES_SQL, [sid for sid, _ in keyed], [sha for _, sha in keyed]
        )
        for row in rows:
            sid = str(row["source_id"])
            found[sid] = (str(row["title"] or sid), str(row["author"] or ""))
    return {sid: found.get(sid, (sid, "")) for sid in source_ids}


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _read_chunk_results(path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``kind: "chunks"`` records from one results file, plus the lines that could
    not be read as one.

    Every line is read and parsed on its own, so one truncated or non-JSON line is
    a failed job that names its line number instead of the end of the run: a lane
    writes its results while it is still classifying, and losing the whole file to
    its last bad line would throw away the hours of work around it. The bytes are
    decoded per line rather than with ``errors="replace"`` so a mangled character
    is reported instead of silently becoming a ``U+FFFD`` inside a prompt or a
    chunk id. A record that is not a ``chunks`` result is reported too -- a lane
    that answered the wrong question is only visible if the file says so. A
    leading UTF-8 BOM on the file's first line is stripped before parsing --
    some editors and Windows-side tools write one, and it would otherwise sit in
    front of the JSON and fail the whole line.
    """
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    with open(path, "rb") as fh:
        for number, raw in enumerate(fh, start=1):
            if number == 1:
                raw = raw.removeprefix(b"\xef\xbb\xbf")
            if not raw.strip():
                continue
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                rejected.append(
                    _reject("", f"line {number}: not valid UTF-8: {exc}"[:300], f"line {number}")
                )
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                rejected.append(
                    _reject("", f"line {number}: not valid JSON: {exc}"[:300], f"line {number}")
                )
                continue
            if not isinstance(record, dict):
                rejected.append(_reject("", f"line {number}: not a JSON object", f"line {number}"))
                continue
            if record.get("kind") != "chunks":
                rejected.append(
                    _reject(
                        str(record.get("source_id") or "").strip(),
                        "not a chunks result",
                        record.get("job_id"),
                    )
                )
                continue
            records.append(record)
    return records, rejected


def _cell(value: Any) -> str:
    """One sheet cell: a list becomes `a|b`, everything else a string."""
    if isinstance(value, list):
        return "|".join(str(v) for v in value)
    return "" if value is None else str(value)


def _has_empty_segment(cell: str) -> bool:
    """True for a `youth||masters` cell. `split("|")` would drop the gap and
    store a book as if it had said two values, so it is an error to report, not
    a typo to normalise away. A blank cell is left to the schema, which says
    "at least one value required"."""
    return bool(cell) and any(not part.strip() for part in cell.split("|"))


def _reject(
    source_id: str, error: str, job_id: Any = None, *, chunk_id: str | None = None
) -> dict[str, Any]:
    """A rejection line: `source_id` and `error` always; `job_id` only when the
    record actually carried one (a hand-edited sheet row has none); `chunk_id`
    only when the failure is about one chunk within an otherwise-valid job
    (VOR-1443)."""
    entry: dict[str, Any] = {"source_id": source_id}
    if job_id:
        entry["job_id"] = job_id
    if chunk_id:
        entry["chunk_id"] = chunk_id
    entry["error"] = error
    return entry


def register(group: click.Group) -> None:
    @group.group("tf-books")
    def tf_books() -> None:
        """Book-level track-and-field records (draft, review sheet, load)."""

    @tf_books.command("inputs")
    @click.option(
        "--out", "out_path", required=True, type=click.Path(dir_okay=False, writable=True)
    )
    def inputs(out_path: str) -> None:
        """Write one book-draft job per track-and-field source."""

        async def _run() -> list[dict[str, Any]]:
            jobs = []
            async with _resources() as (pool, store):
                source_ids = await tf_source_ids(store._client)  # noqa: SLF001
                idents = await _book_identities(pool, source_ids)
                for sid in source_ids:
                    title, author = idents[sid]
                    chunks = await book_chunks(store._client, sid)  # noqa: SLF001
                    front = [c["content"] for c in chunks[:_FRONT_MATTER_CHUNKS]]
                    sections = list(dict.fromkeys(c["section_path"] for c in chunks))
                    jobs.append(
                        {
                            "job_id": f"book:{sid}",
                            "kind": "book",
                            "source_id": sid,
                            "title": title,
                            "author": author,
                            "prompt": book_prompt(title, author, None, front, sections),
                        }
                    )
            return jobs

        jobs = asyncio.run(_run())
        with open(out_path, "w", encoding="utf-8") as fh:
            for job in jobs:
                fh.write(json.dumps(job, ensure_ascii=False) + "\n")
        click.echo(json.dumps({"jobs": len(jobs)}))

    @tf_books.command("load-drafts")
    @click.argument("results_path", type=click.Path(exists=True, dir_okay=False))
    def load_drafts(results_path: str) -> None:
        """Store every valid model draft as the book's record (reviewed=false); no human step.

        loaded counts records actually written: a draft that loses to an already
        reviewed record is counted as skipped_reviewed, not loaded."""
        parsed, rejected = [], []
        skipped_other = 0
        for r in _read_jsonl(results_path):
            if r.get("kind") != "book":
                skipped_other += 1
                continue
            source_id = str(r.get("source_id") or "").strip()
            raw = r.get("result")
            job_id = r.get("job_id")
            if not source_id:
                rejected.append(_reject(source_id, "source_id is required", job_id))
                continue
            if not isinstance(raw, dict):
                rejected.append(
                    _reject(
                        source_id,
                        f"result must be a JSON object, got {type(raw).__name__}",
                        job_id,
                    )
                )
                continue
            try:
                parsed.append((source_id, validate_book(raw)))
            except TagValidationError as exc:
                rejected.append(_reject(source_id, str(exc)[:200], job_id))

        async def _run() -> int:
            written = 0
            async with _pool_only() as pool:
                for sid, book in parsed:
                    if await upsert_book(pool, sid, book, reviewed=False):
                        written += 1
            return written

        written = asyncio.run(_run())
        click.echo(
            json.dumps(
                {
                    "loaded": written,
                    "skipped_reviewed": len(parsed) - written,
                    "skipped_other": skipped_other,
                    "rejected": rejected,
                }
            )
        )

    @tf_books.command("sheet")
    @click.argument("results_path", type=click.Path(exists=True, dir_okay=False))
    @click.option(
        "--out", "out_path", required=True, type=click.Path(dir_okay=False, writable=True)
    )
    def sheet(results_path: str, out_path: str) -> None:
        """Write the model's book drafts as a review sheet (TSV), one row per draft.

        The title and author columns are informational: they come from the intake
        envelope so the sheet is readable, and `tf-books load` never stores them.
        A draft the schema rejected keeps its row, with the failure in the error
        column and its values in place, so it can be corrected here."""

        async def _identities(ids: list[str]) -> dict[str, tuple[str, str]]:
            async with _pool_only() as pool:
                return await _book_identities(pool, ids)

        records = _read_jsonl(results_path)
        results = [r for r in records if r.get("kind") == "book"]
        # Stripped, so the lookup key matches the stripped source_id every row
        # (and _book_identities) uses below -- a padded id would otherwise miss
        # the identities dict and fall back to a blank title/author.
        idents = asyncio.run(_identities([str(r.get("source_id") or "").strip() for r in results]))
        rejected = []
        rows = 0
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_SHEET_COLUMNS, delimiter="\t")
            writer.writeheader()
            for r in results:
                source_id = str(r.get("source_id") or "").strip()
                raw = r.get("result")
                draft = raw if isinstance(raw, dict) else {}
                job_id = r.get("job_id")
                try:
                    values = validate_book(raw).model_dump()
                except TagValidationError as exc:
                    rejected.append(_reject(source_id, str(exc)[:200], job_id))
                    values = {**draft, "error": str(exc)[:200]}
                title, author = idents.get(source_id, (source_id, ""))
                writer.writerow(
                    {
                        **{k: _cell(v) for k, v in values.items() if k in _TAG_COLUMNS},
                        "source_id": source_id,
                        "title": title,
                        "author": author,
                        "rationale": _cell(draft.get("rationale")),
                        "error": _cell(values.get("error")),
                    }
                )
                rows += 1
        click.echo(
            json.dumps(
                {
                    "rows": rows,
                    "skipped_other": len(records) - len(results),
                    "rejected": rejected,
                }
            )
        )

    @tf_books.command("load")
    @click.argument("sheet_path", type=click.Path(exists=True, dir_okay=False))
    def load(sheet_path: str) -> None:
        """Load the reviewed sheet: every valid row is stored with reviewed=true.

        loaded counts records actually written: a row that loses to an already
        reviewed record is counted as skipped_reviewed, not loaded."""
        # utf-8-sig: a sheet re-saved by a Windows-side editor may carry a BOM,
        # which would otherwise sit in front of the header's first column name
        # and break every row (VOR-1443).
        with open(sheet_path, encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        parsed, rejected = [], []
        for row in rows:
            source_id = str(row.get("source_id") or "").strip()
            if not source_id:
                rejected.append({"source_id": source_id, "error": "source_id is required"})
                continue
            raw: dict[str, Any] = {k: row.get(k, "") for k in _TAG_COLUMNS}
            gap = next((f for f in sorted(_LIST_FIELDS) if _has_empty_segment(str(raw[f]))), None)
            if gap is not None:
                rejected.append(
                    {"source_id": source_id, "error": f"{gap}: empty value in a |-separated list"}
                )
                continue
            for f in _LIST_FIELDS:
                raw[f] = [v for v in str(raw[f]).split("|") if v]
            # A non-empty cell that is not a plain integer rejects the row rather
            # than silently becoming None (VOR-1443).
            year_cell = str(raw["tf_pub_year"]).strip()
            if year_cell and not year_cell.isdigit():
                rejected.append(
                    {
                        "source_id": source_id,
                        "error": f"tf_pub_year: not a plain integer {year_cell!r}",
                    }
                )
                continue
            raw["tf_pub_year"] = int(year_cell) if year_cell else None
            try:
                parsed.append((source_id, validate_book(raw)))
            except TagValidationError as exc:
                rejected.append({"source_id": source_id, "error": str(exc)[:200]})

        async def _run() -> int:
            written = 0
            async with _pool_only() as pool:
                for sid, book in parsed:
                    if await upsert_book(pool, sid, book, reviewed=True):
                        written += 1
            return written

        written = asyncio.run(_run())
        click.echo(
            json.dumps(
                {"loaded": written, "skipped_reviewed": len(parsed) - written, "rejected": rejected}
            )
        )

    @group.group("tf-chunks")
    def tf_chunks() -> None:
        """Chunk-level track-and-field tags (export jobs, load results, re-apply, status)."""

    @tf_chunks.command("export")
    @click.option("--shards", type=click.IntRange(min=1), default=5, show_default=True)
    @click.option("--out-dir", "out_dir", required=True, type=click.Path(file_okay=False))
    @click.option(
        "--source",
        "sources",
        multiple=True,
        help="Limit to these source ids (re-runs, single books).",
    )
    @click.option("--batch", type=click.IntRange(min=1), default=_BATCH_DEFAULT, show_default=True)
    def export(shards: int, out_dir: str, sources: tuple[str, ...], batch: int) -> None:
        """Write per-lane job files for untagged or stale chunks of every book with a record.

        A chunk is exported only when its stamped `tf_tag_version` is not the
        current one, so a re-run picks up exactly what is missing. A chunk whose
        content hash already has current-version tags cached in Postgres is
        skipped too and counted as `cached_skipped` -- `tf-chunks apply` stamps
        those from the cache without calling the model. A book is never split
        across lanes: all of its jobs land in the same shard, and the books are
        placed largest-first so one big book cannot leave a lane idle. A book
        with a record and nothing left to tag is counted as
        `already_tagged_books`, so a finished run reads differently from a
        mis-pointed one where no book had a record at all. Source ids are
        de-duplicated, so a repeated `--source` is one book and never a second
        batch of jobs colliding with the first.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        async def _run() -> tuple[list[list[dict[str, Any]]], dict[str, int]]:
            async with _resources() as (pool, store):
                books = await book_records(pool)
                ids = list(dict.fromkeys(sources or tuple(await tf_source_ids(store._client))))  # noqa: SLF001
                per_book: list[tuple[str, list[dict[str, Any]]]] = []
                skipped = 0
                current = 0
                cached_skipped = 0
                for sid in ids:
                    if sid not in books:
                        skipped += 1
                        continue
                    chunks = await book_chunks(store._client, sid)  # noqa: SLF001
                    cache = await chunk_tags_by_hash(pool, sid)
                    todo = []
                    for c in chunks:
                        if c["tf_tag_version"] == TF_TAG_VERSION:
                            continue
                        if c["content_sha256"] in cache:
                            cached_skipped += 1
                            continue
                        todo.append(c)
                    # One tag classification covers each chunk identity once; if
                    # multiple stored points share it, load fans the result back
                    # to every point after checking that their text agrees.
                    todo_by_chunk_id = {c["chunk_id"]: c for c in todo}
                    todo = list(todo_by_chunk_id.values())
                    if todo:
                        per_book.append((sid, todo))
                    else:
                        current += 1
                per_book.sort(key=lambda item: -len(item[1]))
                # One batched lookup for every book in the run, not one per book:
                # the prompt needs a title and an author and the intake envelope
                # is the only place either is written.
                idents = await _book_identities(pool, [sid for sid, _ in per_book])
                lanes: list[list[dict[str, Any]]] = [[] for _ in range(shards)]
                loads = [0] * shards
                n_chunks = 0
                for sid, todo in per_book:
                    lane = loads.index(min(loads))
                    title, author = idents[sid]
                    book, _reviewed = books[sid]
                    for n, start in enumerate(range(0, len(todo), batch)):
                        part = todo[start : start + batch]
                        chunk_ids = [c["chunk_id"] for c in part]
                        hashes = {c["chunk_id"]: c["content_sha256"] for c in part}
                        input_sha256 = {
                            c["chunk_id"]: chunk_tag_input_sha256(c["content"]) for c in part
                        }
                        prompt = chunk_batch_prompt(
                            title,
                            author,
                            book,
                            [(c["chunk_id"], c["section_path"], c["content"]) for c in part],
                        )
                        recipe_sha256 = chunk_tag_recipe_sha256()
                        prompt_sha256 = chunk_tag_prompt_sha256(prompt)
                        job_sha256 = chunk_tag_job_sha256(
                            source_id=sid,
                            tag_version=TF_TAG_VERSION,
                            chunk_ids=chunk_ids,
                            hashes=hashes,
                            input_sha256=input_sha256,
                            recipe_sha256=recipe_sha256,
                            prompt_sha256=prompt_sha256,
                        )
                        lanes[lane].append(
                            {
                                "job_id": f"chunks:{sid}:{n}",
                                "kind": "chunks",
                                "source_id": sid,
                                "chunk_ids": chunk_ids,
                                "job_chunk_ids": chunk_ids,
                                "hashes": hashes,
                                "input_sha256": input_sha256,
                                # The model answers under p1..pN, not chunk ids
                                # (VOR-1451); the lane maps them back with this.
                                "keys": {chunk_label(i): c["chunk_id"] for i, c in enumerate(part)},
                                # Load refuses results from any other prompt version.
                                "tag_version": TF_TAG_VERSION,
                                "recipe_sha256": recipe_sha256,
                                "export_prompt_sha256": prompt_sha256,
                                "job_sha256": job_sha256,
                                "prompt": prompt,
                            }
                        )
                    loads[lane] += len(todo)
                    n_chunks += len(todo)
                return (
                    lanes,
                    {
                        "jobs": sum(map(len, lanes)),
                        "chunks": n_chunks,
                        "books": len(per_book),
                        "skipped_no_record": skipped,
                        "already_tagged_books": current,
                        "cached_skipped": cached_skipped,
                    },
                )

        lanes, summary = asyncio.run(_run())
        for i, jobs in enumerate(lanes):
            with open(out / f"shard-{i}.jsonl", "w", encoding="utf-8") as fh:
                for job in jobs:
                    fh.write(json.dumps(job, ensure_ascii=False) + "\n")
        click.echo(json.dumps(summary))

    @tf_chunks.command("load")
    @click.argument(
        "results_paths", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False)
    )
    def load_chunks(results_paths: tuple[str, ...]) -> None:
        """Validate chunk-batch results, store them, and stamp the Qdrant payload.

        Each record is validated against the job's own `chunk_ids`, so a lane that
        drops or invents a chunk fails its job rather than corrupting a book. No
        single record can end the run, and none is dropped in silence: a line that
        is not valid UTF-8 or not valid JSON, a record that is not a `chunks`
        result, a record with no `source_id` or no chunks, a result the schema
        rejects and a store that throws are all failed jobs, appended to
        `failed.jsonl` beside the first results file, and the run carries on with
        the next one. The hash stored with the tags is the one the corpus holds for
        the chunk now, not the one the job file claimed. The exported prompt
        digest is checked against current title/author/book tags/section paths;
        the lane separately records the digest of the exact successful attempt."""
        failed: list[dict[str, Any]] = []
        parsed: list[dict[str, Any]] = []
        for path in results_paths:
            records, unreadable = _read_chunk_results(path)
            failed.extend(unreadable)
            for r in records:
                source_id = str(r.get("source_id") or "").strip()
                job_id = r.get("job_id")
                if not source_id:
                    failed.append(_reject(source_id, "source_id is required", job_id))
                    continue
                chunk_ids = r.get("chunk_ids")
                if isinstance(chunk_ids, list) and not chunk_ids:
                    failed.append(_reject(source_id, "empty chunk_ids", job_id))
                    continue
                if not isinstance(chunk_ids, list) or not all(
                    isinstance(cid, str) and cid for cid in chunk_ids
                ):
                    failed.append(
                        _reject(source_id, "chunk_ids must be a list of chunk ids", job_id)
                    )
                    continue
                parsed.append({**r, "source_id": source_id, "chunk_ids": chunk_ids})
        failed_path = Path(results_paths[0]).parent / "failed.jsonl"

        async def _run() -> int:
            loaded = 0
            async with _resources() as (pool, store):
                books = await book_records(pool)
                source_ids = sorted({r["source_id"] for r in parsed} & set(books))
                idents = await _book_identities(pool, source_ids)
                # Nothing to stamp without a book record, so those books are not
                # scrolled either -- an unknown source_id is a failed job below.
                # chunk_id is content-addressed, so more than one live point can
                # share it; every point for a validated chunk_id must be stamped
                # (VOR-1443), so this groups into a list, not a last-write-wins
                # dict.
                points: dict[str, dict[str, list[dict[str, Any]]]] = {}
                for sid in {r["source_id"] for r in parsed} & set(books):
                    by_chunk: dict[str, list[dict[str, Any]]] = {}
                    for c in await book_chunks(store._client, sid):  # noqa: SLF001
                        by_chunk.setdefault(c["chunk_id"], []).append(c)
                    points[sid] = by_chunk
                for r in parsed:
                    sid = r["source_id"]
                    job_id = r.get("job_id")
                    # The lane records the model that produced the tags, so a
                    # re-run under a different model is visible in provenance.
                    model = str(r.get("model") or "").strip() or TF_TAGGER_MODEL
                    try:
                        if r.get("tag_version") != TF_TAG_VERSION:
                            # An older prompt's answer (or one that names no version)
                            # must never be stamped current: the version-filtered cache
                            # would re-apply it forever (VOR-1451).
                            found = r.get("tag_version")
                            failed.append(
                                _reject(
                                    sid,
                                    f"results from tag version {found}, current is "
                                    f"{TF_TAG_VERSION}: re-export and re-tag",
                                    job_id,
                                )
                            )
                            continue
                        if sid not in points:
                            failed.append(_reject(sid, "no book record to stamp", job_id))
                            continue
                        job_chunk_ids = r.get("job_chunk_ids")
                        hashes = r.get("hashes")
                        input_hashes = r.get("input_sha256")
                        recipe_sha256 = r.get("recipe_sha256")
                        export_prompt_sha256 = r.get("export_prompt_sha256")
                        job_sha256 = r.get("job_sha256")
                        if (
                            not isinstance(job_chunk_ids, list)
                            or not job_chunk_ids
                            or not all(isinstance(cid, str) and cid for cid in job_chunk_ids)
                            or len(set(job_chunk_ids)) != len(job_chunk_ids)
                            or len(set(r["chunk_ids"])) != len(r["chunk_ids"])
                            or not isinstance(hashes, dict)
                            or set(hashes) != set(job_chunk_ids)
                            or not all(isinstance(value, str) for value in hashes.values())
                            or not isinstance(input_hashes, dict)
                            or set(input_hashes) != set(job_chunk_ids)
                            or not all(isinstance(value, str) for value in input_hashes.values())
                            or not set(r["chunk_ids"]).issubset(job_chunk_ids)
                            or recipe_sha256 != chunk_tag_recipe_sha256()
                            or not isinstance(export_prompt_sha256, str)
                            or not isinstance(job_sha256, str)
                            or job_sha256
                            != chunk_tag_job_sha256(
                                source_id=sid,
                                tag_version=TF_TAG_VERSION,
                                chunk_ids=job_chunk_ids,
                                hashes=hashes,
                                input_sha256=input_hashes,
                                recipe_sha256=recipe_sha256,
                                prompt_sha256=export_prompt_sha256,
                            )
                        ):
                            failed.append(
                                _reject(
                                    sid,
                                    "missing or inconsistent exported job provenance; "
                                    "re-export and re-tag",
                                    job_id,
                                )
                            )
                            continue
                        tags = validate_batch(r.get("result"), r["chunk_ids"])
                        live = points[sid]
                        book, reviewed = books[sid]
                        current_job_chunks: list[tuple[str, str, str]] = []
                        current_job_context_valid = True
                        for cid in job_chunk_ids:
                            if cid not in live:
                                current_job_context_valid = False
                                break
                            point = live[cid][0]
                            content = point.get("content")
                            section_path = point.get("section_path")
                            if not isinstance(content, str) or not isinstance(section_path, str):
                                current_job_context_valid = False
                                break
                            current_job_chunks.append((cid, section_path, content))
                        title, author = idents.get(sid, (sid, ""))
                        current_prompt_sha = (
                            chunk_tag_prompt_sha256(
                                chunk_batch_prompt(title, author, book, current_job_chunks)
                            )
                            if current_job_context_valid
                            else None
                        )
                        if current_prompt_sha != export_prompt_sha256:
                            failed.append(
                                _reject(
                                    sid,
                                    "exported prompt context changed; re-export and re-tag",
                                    job_id,
                                )
                            )
                            continue
                        successful_prompt_sha = r.get("successful_prompt_sha256")
                        attempt_count = r.get("attempts")
                        if (
                            not isinstance(successful_prompt_sha, str)
                            or not all(
                                c in "0123456789abcdef" for c in successful_prompt_sha.lower()
                            )
                            or len(successful_prompt_sha) != 64
                            or r.get("successful_prompt_identity_status")
                            != "computed_actual_attempt"
                            or type(attempt_count) is not int
                            or attempt_count < 1
                            or (
                                attempt_count == 1 and successful_prompt_sha != export_prompt_sha256
                            )
                        ):
                            failed.append(
                                _reject(
                                    sid,
                                    "missing actual successful-attempt prompt identity",
                                    job_id,
                                )
                            )
                            continue
                        rows: list[tuple[str, ChunkTags]] = []
                        items: list[tuple[Any, dict[str, Any]]] = []
                        seen_hashes: set[str] = set()
                        provenance_by_hash: dict[str, dict[str, object]] = {}
                        hash_input_identity: dict[str, str] = {}
                        for cid, t in tags.items():
                            if cid not in live:
                                # A classified chunk the live corpus no longer
                                # holds (e.g. a re-ingest shifted the chunk
                                # boundaries): a failed job, not a silent drop
                                # (VOR-1443).
                                failed.append(
                                    _reject(sid, "chunk no longer present", job_id, chunk_id=cid)
                                )
                                continue
                            expected_hash = hashes.get(cid)
                            expected_input = input_hashes.get(cid)
                            current_matches = all(
                                expected_hash == point["content_sha256"]
                                and expected_input == chunk_tag_input_sha256(point["content"])
                                for point in live[cid]
                            )
                            if not current_matches:
                                failed.append(
                                    _reject(
                                        sid,
                                        "chunk content changed since exported tag job; "
                                        "re-export and re-tag",
                                        job_id,
                                        chunk_id=cid,
                                    )
                                )
                                continue
                            live_hash = live[cid][0]["content_sha256"]
                            point_provenance: dict[str, object] = {
                                "tf_tag_recipe_sha256": recipe_sha256,
                                "tf_tag_prompt_sha256": successful_prompt_sha,
                                "tf_tag_export_prompt_sha256": export_prompt_sha256,
                                "tf_tag_job_sha256": job_sha256,
                                "tf_tag_prompt_identity_status": (
                                    "lane_recorded_actual_attempt_export_context_verified"
                                ),
                                "tf_tag_input_sha256": expected_input,
                                "tf_tag_input_identity_status": "verified_export_match",
                                "tf_tag_source_id": sid,
                                "tf_tag_content_sha256": live_hash,
                                # The result file carries the requested alias only;
                                # it does not prove an immutable served runtime.
                                "tf_tag_identity_status": "unknown_runtime_revision",
                            }
                            for pt in live[cid]:
                                if pt["content_sha256"] not in seen_hashes:
                                    seen_hashes.add(pt["content_sha256"])
                                    rows.append((pt["content_sha256"], t))
                                    provenance_by_hash[pt["content_sha256"]] = dict(
                                        point_provenance
                                    )
                                    hash_input_identity[pt["content_sha256"]] = str(expected_input)
                                elif (
                                    hash_input_identity.get(pt["content_sha256"]) != expected_input
                                ):
                                    # The cache key deliberately ignores spacing. If
                                    # equivalent normalized text has two exact model
                                    # inputs, retain the tags but mark shared provenance
                                    # ambiguous instead of selecting one silently.
                                    provenance_by_hash[pt["content_sha256"]].pop(
                                        "tf_tag_input_sha256", None
                                    )
                                    provenance_by_hash[pt["content_sha256"]][
                                        "tf_tag_input_identity_status"
                                    ] = "ambiguous_normalized_content"
                                items.append(
                                    (
                                        pt["point_id"],
                                        chunk_payload(
                                            t,
                                            book,
                                            reviewed=reviewed,
                                            model=model,
                                            provenance=point_provenance,
                                        ),
                                    )
                                )
                        await upsert_chunk_tags(
                            pool,
                            sid,
                            rows,
                            model=model,
                            provenance_by_hash=provenance_by_hash,
                        )
                        loaded += await apply_payloads(store._client, items)  # noqa: SLF001
                    except Exception as exc:  # noqa: BLE001 - a failed job must not end the run
                        failed.append(_reject(sid, str(exc)[:300], job_id))
            return loaded

        loaded = asyncio.run(_run())
        if failed:
            with open(failed_path, "a", encoding="utf-8") as fh:
                for f in failed:
                    fh.write(json.dumps(f) + "\n")
        click.echo(json.dumps({"loaded": loaded, "failed_jobs": len(failed)}))

    @tf_chunks.command("apply")
    @click.option("--source", "sources", multiple=True)
    def apply_cmd(sources: tuple[str, ...]) -> None:
        """Re-stamp Qdrant from stored tags without calling the model (after re-ingest).

        Source ids are de-duplicated, so a repeated `--source` restamps and counts
        its book once."""

        async def _run() -> dict[str, Any]:
            per: dict[str, Any] = {}
            async with _resources() as (pool, store):
                books = await book_records(pool)
                for sid in dict.fromkeys(sources or tuple(await tf_source_ids(store._client))):  # noqa: SLF001
                    if sid in books:
                        book, reviewed = books[sid]
                        per[sid] = await restamp_source(  # noqa: SLF001
                            pool, store._client, sid, book, reviewed=reviewed
                        )
            return per

        per = asyncio.run(_run())
        for sid, (stamped, untagged) in per.items():
            click.echo(json.dumps({"source_id": sid, "stamped": stamped, "untagged": untagged}))
        click.echo(
            json.dumps(
                {
                    "books": len(per),
                    "stamped": sum(s for s, _ in per.values()),
                    "untagged": sum(u for _, u in per.values()),
                }
            )
        )

    @tf_chunks.command("rekey")
    @click.option("--source", "sources", multiple=True)
    def rekey_cmd(sources: tuple[str, ...]) -> None:
        """Move stored chunk tags from their pre-VOR-1450 keys to the current ones.

        Tags used to be keyed by the exact stripped text, which a re-parse that
        reflows whitespace no longer matches; the key now ignores whitespace. For
        every live chunk this pairs the old key with the new one and moves the
        stored row, so `apply` finds it again. Safe to re-run; run it once after
        deploying VOR-1450, before any purge and re-ingest."""

        async def _run() -> dict[str, int]:
            per: dict[str, int] = {}
            async with _resources() as (pool, store):
                known = tuple(await tf_source_ids(store._client))  # noqa: SLF001
                # Like `apply`, an id that is not a T&F book is skipped, so a typo
                # never reads as a processed book.
                ids = [sid for sid in dict.fromkeys(sources or known) if sid in known]
                for sid in ids:
                    chunks = await book_chunks(store._client, sid)  # noqa: SLF001
                    pairs = [
                        (legacy_content_sha256(c["content"]), c["content_sha256"]) for c in chunks
                    ]
                    per[sid] = await rekey_chunk_tags(pool, sid, pairs)
            return per

        per = asyncio.run(_run())
        click.echo(json.dumps({"books": len(per), "rekeyed": sum(per.values())}))

    @tf_chunks.command("status")
    @click.option(
        "--failed",
        "failed_paths",
        multiple=True,
        type=click.Path(exists=True, dir_okay=False),
        help="Count job lines from these failed.jsonl files into the 'failed' total (repeatable).",
    )
    def status(failed_paths: tuple[str, ...]) -> None:
        """Tagged / stale / untagged point counts per T&F book, then totals.

        `stale` is a chunk stamped under an earlier `TF_TAG_VERSION`; `untagged`
        is a chunk with no `tf_tag_version` at all, so a version bump reads
        differently from a first run. `failed` is not corpus state -- it is a
        property of the lane's own `failed.jsonl` files, counted only when
        `--failed` names them.
        """

        async def _run() -> list[dict[str, Any]]:
            rows = []
            async with _resources() as (_pool, store):
                for sid in await tf_source_ids(store._client):  # noqa: SLF001
                    chunks = await book_chunks(store._client, sid)  # noqa: SLF001
                    tagged = sum(1 for c in chunks if c["tf_tag_version"] == TF_TAG_VERSION)
                    untagged = sum(1 for c in chunks if c["tf_tag_version"] is None)
                    stale = len(chunks) - tagged - untagged
                    rows.append(
                        {
                            "source_id": sid,
                            "total": len(chunks),
                            "tagged": tagged,
                            "stale": stale,
                            "untagged": untagged,
                        }
                    )
            return rows

        rows = asyncio.run(_run())
        failed = sum(
            1
            for p in failed_paths
            for line in Path(p).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        for row in rows:
            click.echo(json.dumps(row))
        click.echo(
            json.dumps(
                {
                    "books": len(rows),
                    "total": sum(r["total"] for r in rows),
                    "tagged": sum(r["tagged"] for r in rows),
                    "stale": sum(r["stale"] for r in rows),
                    "untagged": sum(r["untagged"] for r in rows),
                    "failed": failed,
                }
            )
        )

    @tf_chunks.command("sample")
    @click.option("--source", "sources", multiple=True, required=True)
    @click.option("--n", "per_book", type=click.IntRange(min=1), default=20, show_default=True)
    @click.option(
        "--out", "out_path", required=True, type=click.Path(dir_okay=False, writable=True)
    )
    def sample(sources: tuple[str, ...], per_book: int, out_path: str) -> None:
        """Review sheet: `per_book` tagged chunks per source, spread across the book.

        Rows are taken every `len // per_book` tagged chunks rather than the first
        `per_book`, so the reviewer sees the whole book and not only its opening
        chapter. `correct(y/n)` and `note` are left empty for the reviewer. Source
        ids are de-duplicated, so a repeated `--source` is scrolled once and never
        doubles a book's rows."""
        cols = [
            "source_id",
            "chunk_id",
            "section_path",
            "excerpt",
            "tf_event_family",
            "tf_event",
            "tf_quality",
            "tf_discipline",
            "tf_training_phase",
            "tf_athlete_level",
            "tf_age_group",
            "tf_sex",
            # "chunk" = the passage stated the population; "book" = the book's
            # population filled in because the passage did not (VOR-1451).
            "tf_population_source",
            "tf_intervention",
            "tf_claim_basis",
            "correct(y/n)",
            "note",
        ]

        async def _run() -> list[dict[str, Any]]:
            rows = []
            async with _resources() as (_pool, store):
                fields = [
                    "chunk_id",
                    "section_path",
                    "content",
                    "page_number",
                    *[c for c in cols if c.startswith("tf_")],
                    "tf_tag_version",
                ]
                for sid in dict.fromkeys(sources):
                    pts: list[dict[str, Any]] = []
                    async for p in scroll_book_payloads(  # noqa: SLF001
                        store._client, sid, with_payload=fields
                    ):
                        if (p.payload or {}).get("tf_tag_version") == TF_TAG_VERSION:
                            pts.append(p.payload)
                    # The scroll cursor's order is not document order -- restore
                    # it through the same key book_chunks sorts with before
                    # striding, so the reviewer sees the book front to back
                    # (VOR-1443).
                    pts.sort(key=document_order_key)
                    step = max(1, len(pts) // per_book)
                    for p in pts[::step][:per_book]:
                        row = {
                            c: ("|".join(p[c]) if isinstance(p.get(c), list) else p.get(c, ""))
                            for c in cols
                            if c.startswith("tf_")
                        }
                        rows.append(
                            {
                                **row,
                                "source_id": sid,
                                "chunk_id": p.get("chunk_id"),
                                "section_path": p.get("section_path"),
                                "excerpt": str(p.get("content", ""))[:400].replace("\n", " "),
                            }
                        )
            return rows

        rows = asyncio.run(_run())
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        click.echo(json.dumps({"rows": len(rows)}))

"""`vo-rag corpus tf-chunks export|load|apply|status|sample` (VOR-1443).

`_resources`, `tf_source_ids`, `book_records`, `book_chunks`, `_book_identities`,
`upsert_chunk_tags`, `apply_payloads`, `restamp_source` and
`scroll_book_payloads` are monkeypatched, so nothing real (Qdrant/Postgres) is
touched. `chunk_batch_prompt`, `validate_batch` and `chunk_payload` are exercised
for real, so the job files the lane consumes, the results `load` validates and
the payload it stamps are all produced by the actual code.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Collection
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

from vo_rag.cli import corpus_tf
from vo_rag.cli.main import cli
from vo_rag.core.corpus.tf_prompts import (
    chunk_batch_prompt,
    chunk_tag_input_sha256,
    chunk_tag_job_sha256,
    chunk_tag_prompt_sha256,
    chunk_tag_recipe_sha256,
)
from vo_rag.core.corpus.tf_schema import (
    TF_TAG_VERSION,
    TF_TAGGER_MODEL,
    content_sha256,
    legacy_content_sha256,
    validate_book,
)
from vo_rag.core.corpus.tf_store import document_order_key

BOOK = validate_book(
    {
        "tf_source_type": "coaching_manual",
        "tf_evidence_type": "coaching_practice",
        "tf_pub_year": 1987,
        "tf_book_athlete_level": ["general"],
        "tf_book_age_group": ["not_stated"],
        "tf_book_sex": "not_stated",
    }
)
TAGS = {
    "tf_event_family": ["relay"],
    "tf_event": ["4x100"],
    "tf_quality": ["technique"],
    "tf_discipline": ["training_methodology"],
    "tf_training_phase": ["competition"],
    "tf_athlete_level": ["not_stated"],
    "tf_age_group": ["not_stated"],
    "tf_sex": "not_stated",
    "tf_intervention": ["drill"],
    "tf_claim_basis": "coach_experience",
}


def _chunks(sid: str, n: int, tagged: Collection[int] = frozenset()) -> list[dict[str, Any]]:
    return [
        {
            "point_id": i,
            "chunk_id": f"{sid}-c{i}",
            "section_path": "S",
            "content": f"text {sid} {i}",
            "content_sha256": f"h{sid}{i}",
            "tf_tag_version": TF_TAG_VERSION if i in tagged else None,
        }
        for i in range(n)
    ]


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {
        "stored": [],
        "applied": [],
        "restamped": [],
        # Set to a source id to make that book's storing raise, the way a
        # transient DB/Qdrant error would.
        "raise_upsert_for": None,
        "provenance": [],
        "books": {"s1": (BOOK, False), "s2": (BOOK, False)},
        "chunks": {"s1": _chunks("s1", 45, {0}), "s2": _chunks("s2", 5), "s3": _chunks("s3", 7)},
        # source_id -> {content_sha256: (ChunkTags, tagger_model)}, as returned by
        # tf_store.chunk_tags_by_hash (VOR-1443). Empty by default so export
        # behaves exactly as before unless a test opts a source into the cache.
        "cache": {},
    }

    @asynccontextmanager
    async def fake_resources() -> Any:
        yield "pool", SimpleNamespace(_client="client")

    async def fake_ids(client: Any) -> list[str]:
        return ["s1", "s2", "s3"]

    async def fake_records(pool: Any) -> dict[str, Any]:
        return dict(state["books"])

    async def fake_chunks(client: Any, sid: str) -> list[dict[str, Any]]:
        return list(state["chunks"][sid])

    async def fake_identities(pool: Any, source_ids: list[str]) -> dict[str, tuple[str, str]]:
        return dict.fromkeys(source_ids, ("Field Guide to Sprint Mechanics", "Jordan Avery"))

    async def fake_upsert(
        pool: Any,
        sid: str,
        rows: list[Any],
        *,
        model: str,
        provenance_by_hash: Any = None,
    ) -> int:
        if sid == state["raise_upsert_for"]:
            raise RuntimeError("deadlock detected")
        state["stored"].append((sid, rows))
        state["provenance"].append(provenance_by_hash or {})
        return len(rows)

    async def fake_apply(client: Any, items: list[Any]) -> int:
        state["applied"].extend(items)
        return len(items)

    async def fake_restamp(
        pool: Any, client: Any, sid: str, book: Any, *, reviewed: bool
    ) -> tuple[int, int]:
        state["restamped"].append((sid, reviewed))
        return (3, 1)

    async def fake_cache(pool: Any, sid: str) -> dict[str, Any]:
        return dict(state["cache"].get(sid, {}))

    for name, fn in {
        "_resources": fake_resources,
        "tf_source_ids": fake_ids,
        "book_records": fake_records,
        "book_chunks": fake_chunks,
        "_book_identities": fake_identities,
        "upsert_chunk_tags": fake_upsert,
        "apply_payloads": fake_apply,
        "restamp_source": fake_restamp,
        "chunk_tags_by_hash": fake_cache,
    }.items():
        monkeypatch.setattr(corpus_tf, name, fn)
    return state


def _last_json(output: str) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(output.strip().splitlines()[-1])
    return parsed


def _job(
    job_id: str, sid: str, count: int, chunks: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """A well-formed result record for the book's first `count` chunks."""
    chunks = chunks or _chunks(sid, count)
    unique_chunks: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        unique_chunks.setdefault(chunk["chunk_id"], chunk)
    chunks = list(unique_chunks.values())
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    hashes = {chunk["chunk_id"]: chunk["content_sha256"] for chunk in chunks}
    input_hashes = {chunk["chunk_id"]: chunk_tag_input_sha256(chunk["content"]) for chunk in chunks}
    prompt = chunk_batch_prompt(
        "Field Guide to Sprint Mechanics",
        "Jordan Avery",
        BOOK,
        [(chunk["chunk_id"], chunk["section_path"], chunk["content"]) for chunk in chunks],
    )
    recipe_sha256 = chunk_tag_recipe_sha256()
    prompt_sha256 = chunk_tag_prompt_sha256(prompt)
    record = {
        "job_id": job_id,
        "kind": "chunks",
        "source_id": sid,
        "chunk_ids": chunk_ids,
        "job_chunk_ids": chunk_ids,
        "hashes": hashes,
        "input_sha256": input_hashes,
        "recipe_sha256": recipe_sha256,
        "export_prompt_sha256": prompt_sha256,
        "result": dict.fromkeys(chunk_ids, TAGS),
        "tag_version": TF_TAG_VERSION,
        "attempts": 1,
        "successful_prompt_sha256": prompt_sha256,
        "successful_prompt_identity_status": "computed_actual_attempt",
    }
    exported_prompt_sha = record["export_prompt_sha256"]
    assert isinstance(exported_prompt_sha, str)
    record["job_sha256"] = chunk_tag_job_sha256(
        source_id=sid,
        tag_version=TF_TAG_VERSION,
        chunk_ids=chunk_ids,
        hashes=hashes,
        input_sha256=input_hashes,
        recipe_sha256=recipe_sha256,
        prompt_sha256=exported_prompt_sha,
    )
    return record


def _rebind_job(record: dict[str, Any]) -> None:
    record["job_sha256"] = chunk_tag_job_sha256(
        source_id=record["source_id"],
        tag_version=record["tag_version"],
        chunk_ids=record["job_chunk_ids"],
        hashes=record["hashes"],
        input_sha256=record["input_sha256"],
        recipe_sha256=record["recipe_sha256"],
        prompt_sha256=record["export_prompt_sha256"],
    )


def test_export_skips_books_without_a_record_and_already_tagged_chunks_and_keeps_books_whole(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    result = CliRunner().invoke(
        cli, ["corpus", "tf-chunks", "export", "--shards", "2", "--out-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    summary = _last_json(result.output)
    assert summary == {
        "jobs": 6,
        "chunks": 49,
        "books": 2,
        "skipped_no_record": 1,
        "already_tagged_books": 0,
        "cached_skipped": 0,
    }
    shard_sources = [
        {
            json.loads(line)["source_id"]
            for line in (tmp_path / f"shard-{i}.jsonl").read_text().splitlines()
        }
        for i in range(2)
    ]
    assert shard_sources[0].isdisjoint(shard_sources[1])
    first = json.loads((tmp_path / "shard-0.jsonl").read_text().splitlines()[0])
    assert (
        first["kind"] == "chunks"
        and len(first["chunk_ids"]) == 10
        and "s1-c0" not in first["chunk_ids"]
    )


def test_export_names_every_book_identity_in_one_batched_lookup(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    result = CliRunner().invoke(
        cli, ["corpus", "tf-chunks", "export", "--shards", "1", "--out-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    jobs = [json.loads(line) for line in (tmp_path / "shard-0.jsonl").read_text().splitlines()]
    assert _last_json(result.output)["chunks"] == 49
    # Every book prompt carries the title/author the batched lookup returned, and
    # every job hashes the content the corpus actually holds.
    assert all("Field Guide to Sprint Mechanics — Jordan Avery" in j["prompt"] for j in jobs)
    assert [j["job_id"] for j in jobs] == [
        "chunks:s1:0",
        "chunks:s1:1",
        "chunks:s1:2",
        "chunks:s1:3",
        "chunks:s1:4",
        "chunks:s2:0",
    ]
    assert jobs[0]["hashes"]["s1-c1"] == "hs11" and len(jobs[-1]["chunk_ids"]) == 5


def test_export_keeps_a_whole_book_in_one_lane(tmp_path: Path, fakes: dict[str, Any]) -> None:
    result = CliRunner().invoke(
        cli, ["corpus", "tf-chunks", "export", "--shards", "3", "--out-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    lanes = [
        [
            json.loads(line)["source_id"]
            for line in (tmp_path / f"shard-{i}.jsonl").read_text().splitlines()
        ]
        for i in range(3)
    ]
    # A book is never split across lanes: its chunks would lose the batch context
    # and a lane could only stamp part of it.
    assert [set(lane) for lane in lanes] == [{"s1"}, {"s2"}, set()]


def test_export_counts_a_book_whose_chunks_are_all_current(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    fakes["chunks"]["s1"] = _chunks("s1", 3, {0, 1, 2})
    result = CliRunner().invoke(
        cli, ["corpus", "tf-chunks", "export", "--shards", "2", "--out-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    # A finished re-run says so: a book with a record and no work left is counted
    # apart from a book with no record at all.
    assert _last_json(result.output) == {
        "jobs": 1,
        "chunks": 5,
        "books": 1,
        "skipped_no_record": 1,
        "already_tagged_books": 1,
        "cached_skipped": 0,
    }
    assert (tmp_path / "shard-0.jsonl").read_text().count("\n") == 1
    assert (tmp_path / "shard-1.jsonl").read_text() == ""


def test_export_can_be_limited_to_one_source(tmp_path: Path, fakes: dict[str, Any]) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "corpus",
            "tf-chunks",
            "export",
            "--shards",
            "1",
            "--source",
            "s2",
            "--out-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # Only the named source is looked up, so the book without a record is not
    # even counted as skipped.
    assert _last_json(result.output) == {
        "jobs": 1,
        "chunks": 5,
        "books": 1,
        "skipped_no_record": 0,
        "already_tagged_books": 0,
        "cached_skipped": 0,
    }
    jobs = [json.loads(line) for line in (tmp_path / "shard-0.jsonl").read_text().splitlines()]
    assert [j["source_id"] for j in jobs] == ["s2"]


def test_export_deduplicates_repeated_source_ids(tmp_path: Path, fakes: dict[str, Any]) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "corpus",
            "tf-chunks",
            "export",
            "--shards",
            "2",
            "--source",
            "s2",
            "--source",
            "s2",
            "--source",
            "s1",
            "--out-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # A repeated id is one book: counted once, exported once, so the job ids stay
    # unique and the chunks are not counted twice.
    assert _last_json(result.output) == {
        "jobs": 6,
        "chunks": 49,
        "books": 2,
        "skipped_no_record": 0,
        "already_tagged_books": 0,
        "cached_skipped": 0,
    }
    job_ids = [
        json.loads(line)["job_id"]
        for i in range(2)
        for line in (tmp_path / f"shard-{i}.jsonl").read_text().splitlines()
    ]
    assert job_ids == [
        "chunks:s1:0",
        "chunks:s1:1",
        "chunks:s1:2",
        "chunks:s1:3",
        "chunks:s1:4",
        "chunks:s2:0",
    ]


def test_export_skips_chunks_whose_content_hash_has_current_cached_tags(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    """A chunk whose content hash already carries current-version tags in
    Postgres is not re-classified by the model -- `tf-chunks apply` stamps it
    from the cache instead. The rest of the book still exports (VOR-1443)."""
    fakes["cache"]["s2"] = {"hs21": ("cached", "m")}  # s2's chunk index 1
    result = CliRunner().invoke(
        cli, ["corpus", "tf-chunks", "export", "--source", "s2", "--out-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    summary = _last_json(result.output)
    assert summary == {
        "jobs": 1,
        "chunks": 4,
        "books": 1,
        "skipped_no_record": 0,
        "already_tagged_books": 0,
        "cached_skipped": 1,
    }
    jobs = [json.loads(line) for line in (tmp_path / "shard-0.jsonl").read_text().splitlines()]
    assert "s2-c1" not in jobs[0]["chunk_ids"]
    assert set(jobs[0]["chunk_ids"]) == {"s2-c0", "s2-c2", "s2-c3", "s2-c4"}


def test_export_counts_a_book_entirely_covered_by_the_cache_as_already_tagged(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    fakes["cache"]["s2"] = {f"hs2{i}": ("cached", "m") for i in range(5)}
    result = CliRunner().invoke(
        cli, ["corpus", "tf-chunks", "export", "--source", "s2", "--out-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert _last_json(result.output) == {
        "jobs": 0,
        "chunks": 0,
        "books": 0,
        "skipped_no_record": 0,
        "already_tagged_books": 1,
        "cached_skipped": 5,
    }


def test_shards_batch_and_rows_per_book_must_be_at_least_one(tmp_path: Path) -> None:
    for args, option in (
        (["export", "--shards", "0", "--out-dir", str(tmp_path)], "--shards"),
        (["export", "--shards", "2", "--batch", "0", "--out-dir", str(tmp_path)], "--batch"),
        (["sample", "--source", "s1", "--n", "0", "--out", str(tmp_path / "s.tsv")], "--n"),
    ):
        result = CliRunner().invoke(cli, ["corpus", "tf-chunks", *args])
        assert result.exit_code == 2, result.output
        assert f"Invalid value for {option!r}" in result.output


def test_load_stores_valid_batches_and_records_failed_jobs(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    exported = CliRunner().invoke(
        cli, ["corpus", "tf-chunks", "export", "--shards", "1", "--out-dir", str(tmp_path)]
    )
    assert exported.exit_code == 0, exported.output
    jobs = [json.loads(line) for line in (tmp_path / "shard-0.jsonl").read_text().splitlines()]
    s2 = next(j for j in jobs if j["source_id"] == "s2")
    good = {
        **s2,
        "job_id": s2["job_id"],
        "kind": "chunks",
        "tag_version": TF_TAG_VERSION,
        "attempts": 1,
        "successful_prompt_sha256": s2["export_prompt_sha256"],
        "successful_prompt_identity_status": "computed_actual_attempt",
        "source_id": "s2",
        "chunk_ids": s2["chunk_ids"],
        "result": dict.fromkeys(s2["chunk_ids"], TAGS),
    }
    bad = {**good, "job_id": "chunks:s2:bogus", "result": {s2["chunk_ids"][0]: TAGS}}
    results = tmp_path / "results-0.jsonl"
    results.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n")
    out = _last_json(CliRunner().invoke(cli, ["corpus", "tf-chunks", "load", str(results)]).output)
    assert out == {"loaded": 5, "failed_jobs": 1}
    assert len(fakes["applied"]) == 5 and fakes["applied"][0][1]["tf_population_source"] == "book"
    assert fakes["applied"][0][1]["tf_book_reviewed"] is False
    # The stored row is keyed by the hash the corpus holds for that chunk, not by
    # anything the job file claimed.
    sid, rows = fakes["stored"][0]
    assert sid == "s2" and [h for h, _ in rows] == [f"hs2{i}" for i in range(5)]
    assert (
        json.loads((tmp_path / "failed.jsonl").read_text().splitlines()[0])["job_id"]
        == "chunks:s2:bogus"
    )


def test_load_survives_a_malformed_record_and_still_loads_the_valid_ones(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    results = tmp_path / "results-0.jsonl"
    results.write_text(
        "\n".join(
            [
                # A padded source_id is stripped, so it still finds its book.
                json.dumps({**_job("chunks:s2:0", "s2", 5), "source_id": " s2 "}),
                "{ not json at all",
                json.dumps(
                    {
                        "job_id": "chunks:x:0",
                        "kind": "chunks",
                        "tag_version": TF_TAG_VERSION,
                        "chunk_ids": ["x-c0"],
                        "result": {},
                    }
                ),
                json.dumps(
                    {
                        "job_id": "chunks:s2:1",
                        "kind": "chunks",
                        "tag_version": TF_TAG_VERSION,
                        "source_id": "s2",
                        "result": {},
                    }
                ),
                "",
            ]
        )
        + "\n"
    )
    result = CliRunner().invoke(cli, ["corpus", "tf-chunks", "load", str(results)])
    assert result.exit_code == 0, result.output
    assert _last_json(result.output) == {"loaded": 5, "failed_jobs": 3}
    assert len(fakes["applied"]) == 5
    # Every malformed line is a failed job naming where it came from, and the
    # unreadable one is located by its line number.
    failed = [json.loads(line) for line in (tmp_path / "failed.jsonl").read_text().splitlines()]
    assert [(f["source_id"], f["job_id"]) for f in failed] == [
        ("", "line 2"),
        ("", "chunks:x:0"),
        ("s2", "chunks:s2:1"),
    ]
    assert "line 2" in failed[0]["error"] and "source_id" in failed[1]["error"]
    assert "chunk_ids" in failed[2]["error"]


def test_load_keeps_going_when_a_line_is_not_valid_utf8(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    results = tmp_path / "results-0.jsonl"
    # A lane whose file was cut off mid-character: bytes on one line, results
    # either side of it.
    results.write_bytes(
        b"\n".join(
            [
                json.dumps(_job("chunks:s2:0", "s2", 5)).encode(),
                b'{"note": "\xe2\x82',
                json.dumps(_job("chunks:s1:0", "s1", 20)).encode(),
            ]
        )
        + b"\n"
    )
    result = CliRunner().invoke(cli, ["corpus", "tf-chunks", "load", str(results)])
    assert result.exit_code == 0, result.output
    assert _last_json(result.output) == {"loaded": 25, "failed_jobs": 1}
    assert [sid for sid, _ in fakes["stored"]] == ["s2", "s1"]
    failed = json.loads((tmp_path / "failed.jsonl").read_text().splitlines()[0])
    assert failed["job_id"] == "line 2" and "utf-8" in failed["error"].lower()


def test_load_tolerates_a_leading_utf8_bom(tmp_path: Path, fakes: dict[str, Any]) -> None:
    """A results file saved with a UTF-8 BOM (e.g. by a Windows-side lane) still
    parses its first line instead of failing it as bad JSON."""
    results = tmp_path / "results-0.jsonl"
    results.write_bytes(
        b"\xef\xbb\xbf"
        + json.dumps(_job("chunks:s2:0", "s2", 5)).encode()
        + b"\n"
        + json.dumps(_job("chunks:s1:0", "s1", 20)).encode()
        + b"\n"
    )
    result = CliRunner().invoke(cli, ["corpus", "tf-chunks", "load", str(results)])
    assert result.exit_code == 0, result.output
    assert _last_json(result.output) == {"loaded": 25, "failed_jobs": 0}
    assert [sid for sid, _ in fakes["stored"]] == ["s2", "s1"]
    assert not (tmp_path / "failed.jsonl").exists()


def test_load_records_a_result_that_is_not_a_chunks_record(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    results = tmp_path / "results-0.jsonl"
    results.write_text(
        "\n".join(
            [
                json.dumps({**_job("book:s2", "s2", 1), "kind": "book"}),
                json.dumps(
                    {
                        "job_id": "chunks:s2:9",
                        "source_id": "s2",
                        "chunk_ids": ["s2-c0"],
                        "result": {},
                    }
                ),
                json.dumps(_job("chunks:s2:0", "s2", 5)),
            ]
        )
        + "\n"
    )
    result = CliRunner().invoke(cli, ["corpus", "tf-chunks", "load", str(results)])
    assert result.exit_code == 0, result.output
    # A book draft and a record with no `kind` are traced, not dropped in silence:
    # a lane that answered the wrong question looks the same as one that did not
    # answer at all unless the file says so.
    assert _last_json(result.output) == {"loaded": 5, "failed_jobs": 2}
    failed = [json.loads(line) for line in (tmp_path / "failed.jsonl").read_text().splitlines()]
    assert [(f["source_id"], f["job_id"], f["error"]) for f in failed] == [
        ("s2", "book:s2", "not a chunks result"),
        ("s2", "chunks:s2:9", "not a chunks result"),
    ]


def test_load_records_a_job_with_no_chunks(tmp_path: Path, fakes: dict[str, Any]) -> None:
    results = tmp_path / "results-0.jsonl"
    results.write_text(
        json.dumps(
            {
                "job_id": "chunks:s2:0",
                "kind": "chunks",
                "tag_version": TF_TAG_VERSION,
                "source_id": "s2",
                "chunk_ids": [],
                "result": {},
            }
        )
        + "\n"
        + json.dumps(_job("chunks:s2:1", "s2", 5))
        + "\n"
    )
    result = CliRunner().invoke(cli, ["corpus", "tf-chunks", "load", str(results)])
    assert result.exit_code == 0, result.output
    # A batch that covers no chunk is a lane that returned nothing for the job it
    # was given; it is reported instead of passing as a silent success.
    assert _last_json(result.output) == {"loaded": 5, "failed_jobs": 1}
    assert [sid for sid, _ in fakes["stored"]] == ["s2"]
    failed = json.loads((tmp_path / "failed.jsonl").read_text().splitlines()[0])
    assert failed["source_id"] == "s2" and failed["error"] == "empty chunk_ids"


def test_load_continues_after_a_job_fails_to_store(tmp_path: Path, fakes: dict[str, Any]) -> None:
    fakes["raise_upsert_for"] = "s2"
    results = tmp_path / "results-0.jsonl"
    results.write_text(
        json.dumps(_job("chunks:s2:0", "s2", 5))
        + "\n"
        + json.dumps(_job("chunks:s1:0", "s1", 20))
        + "\n"
    )
    result = CliRunner().invoke(cli, ["corpus", "tf-chunks", "load", str(results)])
    assert result.exit_code == 0, result.output
    # The store blew up on s2; its job is a failed job and s1's still loads.
    assert _last_json(result.output) == {"loaded": 20, "failed_jobs": 1}
    assert [sid for sid, _ in fakes["stored"]] == ["s1"]
    assert len(fakes["applied"]) == 20
    failed = json.loads((tmp_path / "failed.jsonl").read_text().splitlines()[0])
    assert failed["source_id"] == "s2" and "deadlock" in failed["error"]


def test_load_writes_failed_jobs_beside_the_first_results_file(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    first, second = tmp_path / "lane-0", tmp_path / "lane-1"
    first.mkdir()
    second.mkdir()
    (first / "results-0.jsonl").write_text(json.dumps(_job("chunks:s2:0", "s2", 5)) + "\n")
    (second / "results-0.jsonl").write_text(
        json.dumps({**_job("chunks:s2:9", "s2", 5), "result": {}}) + "\n"
    )
    out = _last_json(
        CliRunner()
        .invoke(
            cli,
            [
                "corpus",
                "tf-chunks",
                "load",
                str(first / "results-0.jsonl"),
                str(second / "results-0.jsonl"),
            ],
        )
        .output
    )
    assert out == {"loaded": 5, "failed_jobs": 1}
    assert json.loads((first / "failed.jsonl").read_text().splitlines()[0])["source_id"] == "s2"
    assert not (second / "failed.jsonl").exists()


def test_load_reports_a_result_for_a_book_with_no_record_as_a_failed_job(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    results = tmp_path / "results-0.jsonl"
    results.write_text(
        json.dumps(
            {
                "job_id": "chunks:s9:0",
                "kind": "chunks",
                "tag_version": TF_TAG_VERSION,
                "source_id": "s9",
                "chunk_ids": ["s9-c0"],
                "result": {"s9-c0": TAGS},
            }
        )
        + "\n"
    )
    result = CliRunner().invoke(cli, ["corpus", "tf-chunks", "load", str(results)])
    assert result.exit_code == 0, result.output
    assert _last_json(result.output) == {"loaded": 0, "failed_jobs": 1}
    assert fakes["stored"] == [] and fakes["applied"] == []
    failed = json.loads((tmp_path / "failed.jsonl").read_text().splitlines()[0])
    assert failed["source_id"] == "s9" and "no book record" in failed["error"]


def test_load_stamps_the_model_the_result_record_carries(
    tmp_path: Path, fakes: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provenance is the model that produced the tags, not whatever the default
    is now: the record's own `model` reaches both the stored row and the payload,
    and a record without one falls back to the default."""
    models: list[str] = []

    async def capture(
        pool: Any, sid: str, rows: list[Any], *, model: str, provenance_by_hash: Any = None
    ) -> int:
        models.append(model)
        fakes["stored"].append((sid, rows))
        return len(rows)

    monkeypatch.setattr(corpus_tf, "upsert_chunk_tags", capture)
    results = tmp_path / "results-0.jsonl"
    results.write_text(
        json.dumps({**_job("chunks:s2:0", "s2", 5), "model": "opencode/glm-5"})
        + "\n"
        + json.dumps({**_job("chunks:s1:0", "s1", 20), "model": "  "})
        + "\n"
    )
    out = _last_json(CliRunner().invoke(cli, ["corpus", "tf-chunks", "load", str(results)]).output)
    assert out == {"loaded": 25, "failed_jobs": 0}
    assert models == ["opencode/glm-5", TF_TAGGER_MODEL]
    assert fakes["applied"][0][1]["tf_tagger_model"] == "opencode/glm-5"
    assert fakes["applied"][5][1]["tf_tagger_model"] == TF_TAGGER_MODEL


def test_load_stamps_every_point_sharing_a_content_addressed_chunk_id(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    """chunk_id is content-addressed: two live points can share one when their
    content is identical. Every point for a validated chunk_id must be stamped,
    and the stored tags row is unique per content hash (VOR-1443)."""
    fakes["chunks"]["s2"] = [
        {
            "point_id": 100,
            "chunk_id": "shared",
            "section_path": "S",
            "content": "Baton exchange zones.",
            "content_sha256": "hshared",
            "tf_tag_version": None,
        },
        {
            "point_id": 101,
            "chunk_id": "shared",
            "section_path": "S",
            "content": "Baton exchange zones.",
            "content_sha256": "hshared",
            "tf_tag_version": None,
        },
    ]
    duplicate_chunks = list(fakes["chunks"]["s2"])
    results = tmp_path / "results-0.jsonl"
    results.write_text(json.dumps(_job("chunks:s2:0", "s2", 1, duplicate_chunks)) + "\n")
    out = _last_json(CliRunner().invoke(cli, ["corpus", "tf-chunks", "load", str(results)]).output)
    assert out == {"loaded": 2, "failed_jobs": 0}
    assert sorted(pid for pid, _ in fakes["applied"]) == [100, 101]
    sid, rows = fakes["stored"][0]
    assert sid == "s2"
    # Duplicate hash within the call collapses to one row, not one per point.
    assert [h for h, _ in rows] == ["hshared"]


def test_load_records_a_chunk_no_longer_present_as_a_failed_job(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    """A classified chunk_id the live corpus no longer holds (e.g. the book was
    re-ingested and the chunk boundaries shifted) is a failed job, not a silent
    drop -- the other, still-live chunks in the same batch still load
    (VOR-1443)."""
    chunk_ids = [f"s2-c{i}" for i in range(5)] + ["s2-cGONE"]
    result_record = _job("chunks:s2:0", "s2", 5)
    result_record["chunk_ids"] = chunk_ids
    result_record["job_chunk_ids"] = chunk_ids
    result_record["hashes"]["s2-cGONE"] = "missing-hash"
    result_record["input_sha256"]["s2-cGONE"] = chunk_tag_input_sha256("gone text")
    result_record["result"]["s2-cGONE"] = TAGS
    _rebind_job(result_record)
    results = tmp_path / "results-0.jsonl"
    results.write_text(json.dumps(result_record) + "\n")
    out = _last_json(CliRunner().invoke(cli, ["corpus", "tf-chunks", "load", str(results)]).output)
    assert out == {"loaded": 0, "failed_jobs": 1}
    failed = json.loads((tmp_path / "failed.jsonl").read_text().splitlines()[0])
    assert failed["source_id"] == "s2"
    assert "prompt context changed" in failed["error"]


def test_load_rejects_stale_section_context_even_when_chunk_content_is_unchanged(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    record = _job("chunks:s2:section-stale", "s2", 5)
    fakes["chunks"]["s2"][0]["section_path"] = "Changed section after export"
    results = tmp_path / "stale-context.jsonl"
    results.write_text(json.dumps(record) + "\n")

    output = CliRunner().invoke(cli, ["corpus", "tf-chunks", "load", str(results)])

    assert output.exit_code == 0, output.output
    assert _last_json(output.output) == {"loaded": 0, "failed_jobs": 1}
    assert not fakes["stored"]
    failed = json.loads((tmp_path / "failed.jsonl").read_text().splitlines()[0])
    assert failed["error"] == "exported prompt context changed; re-export and re-tag"


def test_apply_restamps_every_book_with_a_record_at_its_reviewed_flag(
    fakes: dict[str, Any],
) -> None:
    result = CliRunner().invoke(cli, ["corpus", "tf-chunks", "apply"])
    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.output.strip().splitlines()]
    assert [line["source_id"] for line in lines[:-1]] == ["s1", "s2"]
    assert all(line["stamped"] == 3 and line["untagged"] == 1 for line in lines[:-1])
    assert lines[-1] == {"books": 2, "stamped": 6, "untagged": 2}
    # s3 is a T&F book with no stored record, so it has nothing to stamp.
    assert fakes["restamped"] == [("s1", False), ("s2", False)]


def test_apply_can_be_limited_to_one_source(fakes: dict[str, Any]) -> None:
    result = CliRunner().invoke(cli, ["corpus", "tf-chunks", "apply", "--source", "s2"])
    assert result.exit_code == 0, result.output
    assert fakes["restamped"] == [("s2", False)]
    assert _last_json(result.output) == {"books": 1, "stamped": 3, "untagged": 1}


def test_apply_restamps_a_repeated_source_once(fakes: dict[str, Any]) -> None:
    result = CliRunner().invoke(
        cli, ["corpus", "tf-chunks", "apply", "--source", "s2", "--source", "s2"]
    )
    assert result.exit_code == 0, result.output
    assert fakes["restamped"] == [("s2", False)]
    assert _last_json(result.output) == {"books": 1, "stamped": 3, "untagged": 1}


def test_status_counts_tagged_points(fakes: dict[str, Any]) -> None:
    out = _last_json(CliRunner().invoke(cli, ["corpus", "tf-chunks", "status"]).output)
    assert out == {"books": 3, "total": 57, "tagged": 1, "stale": 0, "untagged": 56, "failed": 0}


def test_status_reports_each_book_before_the_totals(fakes: dict[str, Any]) -> None:
    result = CliRunner().invoke(cli, ["corpus", "tf-chunks", "status"])
    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.output.strip().splitlines()]
    assert lines == [
        {"source_id": "s1", "total": 45, "tagged": 1, "stale": 0, "untagged": 44},
        {"source_id": "s2", "total": 5, "tagged": 0, "stale": 0, "untagged": 5},
        {"source_id": "s3", "total": 7, "tagged": 0, "stale": 0, "untagged": 7},
        {"books": 3, "total": 57, "tagged": 1, "stale": 0, "untagged": 56, "failed": 0},
    ]


def test_status_counts_stale_chunks_separately_from_untagged(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    """A chunk stamped under an earlier TF_TAG_VERSION is `stale`, distinct from
    one that has never been tagged at all -- a re-run after a version bump reads
    differently from a first run (VOR-1443)."""
    chunks = _chunks("s3", 7)
    chunks[0]["tf_tag_version"] = 0  # an earlier version, not the current one
    fakes["chunks"]["s3"] = chunks
    result = CliRunner().invoke(cli, ["corpus", "tf-chunks", "status"])
    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.output.strip().splitlines()]
    s3_row = next(r for r in lines if r.get("source_id") == "s3")
    assert s3_row == {"source_id": "s3", "total": 7, "tagged": 0, "stale": 1, "untagged": 6}
    assert lines[-1]["stale"] == 1
    assert lines[-1]["untagged"] == 55


def test_status_counts_lines_across_the_optional_failed_files(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    failed_a = tmp_path / "lane-0" / "failed.jsonl"
    failed_a.parent.mkdir()
    failed_a.write_text('{"source_id": "s1"}\n{"source_id": "s2"}\n\n')
    failed_b = tmp_path / "lane-1" / "failed.jsonl"
    failed_b.parent.mkdir()
    failed_b.write_text('{"source_id": "s3"}\n')
    result = CliRunner().invoke(
        cli, ["corpus", "tf-chunks", "status", "--failed", str(failed_a), "--failed", str(failed_b)]
    )
    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.output.strip().splitlines()]
    assert lines[-1]["failed"] == 3


def test_status_failed_defaults_to_zero_without_the_option(fakes: dict[str, Any]) -> None:
    out = _last_json(CliRunner().invoke(cli, ["corpus", "tf-chunks", "status"]).output)
    assert out["failed"] == 0


def test_sample_spreads_the_review_sheet_over_the_tagged_chunks(
    tmp_path: Path, fakes: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _payload(i: int, version: int | None) -> dict[str, Any]:
        return {
            "chunk_id": f"c{i}",
            "section_path": f"Ch{i}",
            "content": f"body {i}\nsecond line",
            **TAGS,
            "tf_tag_version": version,
        }

    points = [SimpleNamespace(payload=_payload(i, TF_TAG_VERSION)) for i in range(4)]
    points.append(SimpleNamespace(payload=_payload(4, None)))

    async def fake_scroll(client: Any, sid: str, *, with_payload: list[str], **_: Any) -> Any:
        for p in points:
            yield p

    monkeypatch.setattr(corpus_tf, "scroll_book_payloads", fake_scroll)
    sheet = tmp_path / "review.tsv"
    result = CliRunner().invoke(
        cli, ["corpus", "tf-chunks", "sample", "--source", "s1", "--n", "2", "--out", str(sheet)]
    )
    assert result.exit_code == 0, result.output
    assert _last_json(result.output) == {"rows": 2}
    rows = list(csv.DictReader(sheet.open(), delimiter="\t"))
    # Every second tagged chunk, never the untagged one, with lists joined by |.
    assert [r["chunk_id"] for r in rows] == ["c0", "c2"]
    assert rows[0]["source_id"] == "s1" and rows[0]["section_path"] == "Ch0"
    assert rows[0]["tf_event_family"] == "relay" and rows[0]["tf_claim_basis"] == "coach_experience"
    assert rows[0]["excerpt"] == "body 0 second line"
    assert rows[0]["correct(y/n)"] == "" and rows[0]["note"] == ""


def test_sample_deduplicates_repeated_source_ids(
    tmp_path: Path, fakes: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repeated `--source` is scrolled once, so the sheet has no duplicate rows."""
    payload = {
        "chunk_id": "s1-c0",
        "section_path": "Ch0",
        "content": "body",
        **TAGS,
        "tf_tag_version": TF_TAG_VERSION,
    }
    calls: list[str] = []

    async def fake_scroll(client: Any, sid: str, *, with_payload: list[str], **_: Any) -> Any:
        calls.append(sid)
        yield SimpleNamespace(payload=payload)

    monkeypatch.setattr(corpus_tf, "scroll_book_payloads", fake_scroll)
    sheet = tmp_path / "review.tsv"
    result = CliRunner().invoke(
        cli,
        [
            "corpus",
            "tf-chunks",
            "sample",
            "--source",
            "s1",
            "--source",
            "s1",
            "--n",
            "2",
            "--out",
            str(sheet),
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == ["s1"]
    assert _last_json(result.output) == {"rows": 1}
    rows = list(csv.DictReader(sheet.open(), delimiter="\t"))
    assert [r["chunk_id"] for r in rows] == ["s1-c0"]


def test_sample_strides_over_document_order_not_scroll_order(
    tmp_path: Path, fakes: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scroll cursor's order is not document order; `sample` must restore it
    (ascending page_number, missing last) before striding, the same as
    `book_chunks` (VOR-1443)."""

    def _payload(chunk_id: str, page_number: int | None) -> dict[str, Any]:
        return {
            "chunk_id": chunk_id,
            "section_path": "S",
            "content": f"body {chunk_id}",
            **TAGS,
            "tf_tag_version": TF_TAG_VERSION,
            "page_number": page_number,
        }

    # Handed back out of document order, as a shuffled scroll cursor would.
    shuffled = [("c3", 4), ("c1", 1), ("c4", None), ("c2", 2)]

    async def fake_scroll(client: Any, sid: str, *, with_payload: list[str], **_: Any) -> Any:
        assert "page_number" in with_payload
        for chunk_id, page_number in shuffled:
            yield SimpleNamespace(payload=_payload(chunk_id, page_number))

    monkeypatch.setattr(corpus_tf, "scroll_book_payloads", fake_scroll)
    sheet = tmp_path / "review.tsv"
    result = CliRunner().invoke(
        cli, ["corpus", "tf-chunks", "sample", "--source", "s1", "--n", "4", "--out", str(sheet)]
    )
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(sheet.open(), delimiter="\t"))
    assert [r["chunk_id"] for r in rows] == ["c1", "c2", "c3", "c4"]


def test_sample_sorts_a_non_integer_page_number_last_like_book_chunks(
    tmp_path: Path, fakes: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `page_number` that is not an int is unnumbered, not page 0 and not a
    number to compare against: `sample` must order it behind every real page
    (tied by chunk_id) exactly as `book_chunks` does, through the one shared
    document-order key -- comparing it as a string against an int would raise
    (VOR-1443)."""

    def _payload(chunk_id: str, page_number: Any) -> dict[str, Any]:
        return {
            "chunk_id": chunk_id,
            "section_path": "S",
            "content": f"body {chunk_id}",
            **TAGS,
            "tf_tag_version": TF_TAG_VERSION,
            "page_number": page_number,
        }

    shuffled = [("c1", 1), ("c2", "12"), ("c3", None), ("c4", 2)]

    async def fake_scroll(client: Any, sid: str, *, with_payload: list[str], **_: Any) -> Any:
        for chunk_id, page_number in shuffled:
            yield SimpleNamespace(payload=_payload(chunk_id, page_number))

    monkeypatch.setattr(corpus_tf, "scroll_book_payloads", fake_scroll)
    sheet = tmp_path / "review.tsv"
    result = CliRunner().invoke(
        cli, ["corpus", "tf-chunks", "sample", "--source", "s1", "--n", "4", "--out", str(sheet)]
    )
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(sheet.open(), delimiter="\t"))
    assert [r["chunk_id"] for r in rows] == ["c1", "c4", "c2", "c3"]
    # The same key, shared with book_chunks, is what produced that order.
    assert document_order_key({"chunk_id": "c2", "page_number": "12"}) == (True, 0, "c2")


def test_rekey_moves_every_live_chunk_from_its_legacy_key(
    fakes: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each live chunk's legacy key (stripped text) is paired with its current key
    (whitespace-folded text) and handed to rekey_chunk_tags, book by book, for every
    T&F book; the total is what rekey_chunk_tags reports moved (VOR-1450)."""
    fakes["chunks"] = {
        "s1": [
            {
                "chunk_id": "a",
                "content": "two  spaces",
                "content_sha256": content_sha256("two  spaces"),
            }
        ],
        "s2": [
            {"chunk_id": "b", "content": "nospace", "content_sha256": content_sha256("nospace")}
        ],
        "s3": [],
    }
    calls: list[tuple[str, list[tuple[str, str]]]] = []

    async def fake_rekey(pool: Any, sid: str, pairs: list[tuple[str, str]]) -> int:
        calls.append((sid, pairs))
        return sum(1 for old, new in pairs if old != new)

    monkeypatch.setattr(corpus_tf, "rekey_chunk_tags", fake_rekey)
    result = CliRunner().invoke(cli, ["corpus", "tf-chunks", "rekey"])
    assert result.exit_code == 0, result.output
    assert [sid for sid, _ in calls] == ["s1", "s2", "s3"]
    assert calls[0][1] == [(legacy_content_sha256("two  spaces"), content_sha256("two spaces"))]
    assert _last_json(result.output) == {"books": 3, "rekeyed": 1}


def test_rekey_can_be_limited_to_one_source(
    fakes: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    async def fake_rekey(pool: Any, sid: str, pairs: list[tuple[str, str]]) -> int:
        seen.append(sid)
        return 0

    monkeypatch.setattr(corpus_tf, "rekey_chunk_tags", fake_rekey)
    result = CliRunner().invoke(
        cli,
        ["corpus", "tf-chunks", "rekey", "--source", "s2", "--source", "s2", "--source", "typo"],
    )
    assert result.exit_code == 0, result.output
    # A repeated id is processed once; an id that is not a T&F book is skipped.
    assert seen == ["s2"]
    assert _last_json(result.output) == {"books": 1, "rekeyed": 0}


def test_load_rejects_results_from_another_tag_version(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    """A results file from an older prompt (or none recorded) must never be stamped as
    the current version: the version-filtered cache would re-apply it forever
    (VOR-1451)."""
    old = {**_job("chunks:s1:0", "s1", 2), "tag_version": TF_TAG_VERSION - 1}
    unversioned = {k: v for k, v in _job("chunks:s2:0", "s2", 2).items() if k != "tag_version"}
    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps(old) + "\n" + json.dumps(unversioned) + "\n")
    result = CliRunner().invoke(cli, ["corpus", "tf-chunks", "load", str(results)])
    assert result.exit_code == 0, result.output
    assert _last_json(result.output) == {"loaded": 0, "failed_jobs": 2}
    assert fakes["stored"] == [] and fakes["applied"] == []
    errors = [
        json.loads(line)["error"] for line in (tmp_path / "failed.jsonl").read_text().splitlines()
    ]
    assert all("re-export" in e for e in errors)


def test_export_jobs_carry_the_label_map_and_tag_version(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    """The prompt keys passages p1..pN in chunk order; the job's `keys` must map the
    same labels to the same chunks, and the job names its tag version (VOR-1451)."""
    result = CliRunner().invoke(
        cli, ["corpus", "tf-chunks", "export", "--source", "s2", "--out-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    job = json.loads((tmp_path / "shard-0.jsonl").read_text().splitlines()[0])
    assert job["tag_version"] == TF_TAG_VERSION
    assert job["keys"] == {f"p{i + 1}": cid for i, cid in enumerate(job["chunk_ids"])}
    first_text = next(
        c["content"] for c in fakes["chunks"]["s2"] if c["chunk_id"] == job["chunk_ids"][0]
    )
    assert f'<chunk id="p1" section="S">\n{first_text}' in job["prompt"]

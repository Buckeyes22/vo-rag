"""Unit contract for the track-and-field tag lane runner (VOR-1443).

The lane is exercised end to end with a fake ``call``: JSON extraction,
validate-then-retry, resume-from-results, and the still-invalid path to
``RESULTS.failed``. No shim and no model are involved: the ``shim_call`` tests
capture the command the lane hands ``_run_shim``, and the ``_run_shim`` tests run
plain ``bash``/``sleep`` processes to prove an overrun is reaped.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vo_rag.core.corpus.tf_schema import TF_TAGGER_MODEL

spec = importlib.util.spec_from_file_location(
    "lane", Path(__file__).parents[2] / "scripts" / "tf_classify_lane.py"
)
assert spec is not None and spec.loader is not None
lane = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lane)

SHIM_PATH = Path("/path/to/agent-shim.sh")

TAGS = {
    "tf_event_family": ["sprint"],
    "tf_event": ["100m"],
    "tf_quality": ["acceleration"],
    "tf_discipline": ["training_methodology"],
    "tf_training_phase": ["not_stated"],
    "tf_athlete_level": ["not_stated"],
    "tf_age_group": ["not_stated"],
    "tf_sex": "not_stated",
    "tf_intervention": ["drill"],
    "tf_claim_basis": "coach_experience",
}
BOOK_TAGS = {
    "tf_source_type": "coaching_manual",
    "tf_evidence_type": "coaching_practice",
    "tf_pub_year": 1987,
    "tf_book_athlete_level": ["general"],
    "tf_book_age_group": ["not_stated"],
    "tf_book_sex": "not_stated",
    "tf_limitations": "Club-level and pre-2000 conventions.",
    "rationale": "A coach's manual for club athletes.",
}


def test_extract_json_takes_the_last_fenced_block_before_the_sentinel() -> None:
    out = (
        'thinking...\n```json\n{"a": 1}\n```\nnote\n'
        '```json\n{"c1": {"x": 2}}\n```\nSHIM-DONE exit=0\n'
    )
    assert lane.extract_json(out) == {"c1": {"x": 2}}


def test_extract_json_handles_bare_json_with_prose() -> None:
    assert lane.extract_json('Here you go: {"c1": {"y": [1]}} done\nSHIM-DONE exit=0') == {
        "c1": {"y": [1]}
    }


def test_extract_json_falls_back_to_the_body_when_the_fence_holds_no_json() -> None:
    """A fence of prose plus a JSON answer outside it: the body is still a
    candidate, and stray braces around the answer do not swallow it."""
    out = (
        "I cannot follow that schema.\n```json\nsee the table below\n```\n"
        'Shape: {any} — use {"c1": {"x": 2}}.\nSHIM-DONE exit=0\n'
    )
    assert lane.extract_json(out) == {"c1": {"x": 2}}


def test_extract_json_takes_a_top_level_array() -> None:
    out = '```json\n[{"chunk_id": "c1", "tf_quality": ["speed"]}]\n```\nSHIM-DONE exit=0\n'
    assert lane.extract_json(out) == [{"chunk_id": "c1", "tf_quality": ["speed"]}]
    # A body that opens with an array prefers it over any brace that follows.
    assert lane.extract_json('Result: [{"c1": 1}] not {"c1": 2}\nSHIM-DONE exit=0') == [{"c1": 1}]


def _job(tmp: Path) -> Path:
    p = tmp / "jobs.jsonl"
    p.write_text(
        json.dumps(
            {
                "job_id": "chunks:s:0",
                "kind": "chunks",
                "source_id": "s",
                "chunk_ids": ["c1"],
                "prompt": "P",
            }
        )
        + "\n"
    )
    return p


def _book_job(tmp: Path) -> Path:
    p = tmp / "book-jobs.jsonl"
    p.write_text(
        json.dumps({"job_id": "book:s", "kind": "book", "source_id": "s", "prompt": "P"}) + "\n"
    )
    return p


def _chunks_job(tmp: Path, chunk_ids: list[str]) -> Path:
    p = tmp / "chunks-jobs.jsonl"
    p.write_text(
        json.dumps(
            {
                "job_id": "chunks:s:0",
                "kind": "chunks",
                "source_id": "s",
                "chunk_ids": chunk_ids,
                "prompt": "P",
            }
        )
        + "\n"
    )
    return p


def test_retries_once_with_the_validation_error_then_succeeds(tmp_path: Path) -> None:
    prompts: list[str] = []
    replies = iter(['{"c1": {"tf_quality": ["speed"]}}', json.dumps({"c1": TAGS})])

    def call(prompt: str) -> str:
        prompts.append(prompt)
        return next(replies) + "\nSHIM-DONE exit=0\n"

    stats = lane.run_lane(_job(tmp_path), tmp_path / "out.jsonl", call=call)
    assert stats == {"done": 1, "partial": 0, "failed": 0, "skipped": 0}
    # the chunks retry path names the ids it is asking about, unlike a book job's
    assert "previous answer was invalid for these chunk ids only" in prompts[1]
    assert json.loads((tmp_path / "out.jsonl").read_text())["attempts"] == 2


def test_lane_result_retains_exported_tag_job_binding(tmp_path: Path) -> None:
    job_path = tmp_path / "bound-jobs.jsonl"
    binding = {
        "job_chunk_ids": ["c1", "c2"],
        "hashes": {"c1": "content-1", "c2": "content-2"},
        "input_sha256": {"c1": "input-1", "c2": "input-2"},
        "recipe_sha256": "recipe",
        "export_prompt_sha256": "prompt",
        "job_sha256": "job",
    }
    job_path.write_text(
        json.dumps(
            {
                "job_id": "chunks:s:0",
                "kind": "chunks",
                "source_id": "s",
                "chunk_ids": ["c1", "c2"],
                "prompt": "P",
                **binding,
            }
        )
        + "\n"
    )
    stats = lane.run_lane(
        job_path,
        tmp_path / "bound-results.jsonl",
        call=lambda p: json.dumps({"c1": TAGS, "c2": TAGS}) + "\nSHIM-DONE exit=0\n",
    )
    assert stats["done"] == 1
    record = json.loads((tmp_path / "bound-results.jsonl").read_text())
    assert {key: record[key] for key in binding} == binding
    assert record["successful_prompt_identity_status"] == "computed_actual_attempt"


def test_resume_skips_jobs_already_in_results(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    out.write_text(
        json.dumps({"job_id": "chunks:s:0", "kind": "chunks", "chunk_ids": ["c1"], "result": {}})
        + "\n"
    )
    stats = lane.run_lane(
        _job(tmp_path), out, call=lambda p: (_ for _ in ()).throw(AssertionError("called"))
    )
    assert stats == {"done": 0, "partial": 0, "failed": 0, "skipped": 1}


def test_resume_skips_and_counts_an_unparseable_results_line(tmp_path: Path) -> None:
    """A lane killed mid-write leaves a partial line: it is counted, not fatal, and
    its job is run again -- on a line of its own, so the torn text is not glued
    to the new record."""
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(
        "".join(
            json.dumps(
                {
                    "job_id": f"chunks:s:{n}",
                    "kind": "chunks",
                    "source_id": "s",
                    "chunk_ids": [f"c{n + 1}"],
                    "prompt": "P",
                }
            )
            + "\n"
            for n in (0, 1)
        )
    )
    out = tmp_path / "out.jsonl"
    out.write_text(
        json.dumps({"job_id": "chunks:s:0", "kind": "chunks", "chunk_ids": ["c1"], "result": {}})
        + "\n"
        + '{"job_id": "chunks:s:1", "resu'
    )
    seen: list[str] = []

    def call(prompt: str) -> str:
        seen.append(prompt)
        return json.dumps({"c2": TAGS}) + "\nSHIM-DONE exit=0\n"

    stats = lane.run_lane(jobs, out, call=call)
    assert stats == {"done": 1, "partial": 0, "failed": 0, "skipped": 2}
    assert len(seen) == 1
    lines = out.read_text().splitlines()
    # the torn line is left as it was, and the re-run record is a whole line
    assert lines[1] == '{"job_id": "chunks:s:1", "resu'
    record = json.loads(lines[-1])
    assert record["job_id"] == "chunks:s:1"
    assert record["result"] == {"c2": TAGS}


def test_resume_skips_and_counts_an_undecodable_results_line(tmp_path: Path) -> None:
    """A results line that is not valid UTF-8 (a lane process killed mid-write to
    a multi-byte character, or a foreign encoding) must not crash resume -- it is
    counted as skipped and its job runs again, on a line of its own, exactly like
    a torn/unparseable JSON line (VOR-1443)."""
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(
        "".join(
            json.dumps(
                {
                    "job_id": f"chunks:s:{n}",
                    "kind": "chunks",
                    "source_id": "s",
                    "chunk_ids": [f"c{n + 1}"],
                    "prompt": "P",
                }
            )
            + "\n"
            for n in (0, 1)
        )
    )
    out = tmp_path / "out.jsonl"
    out.write_bytes(
        json.dumps(
            {"job_id": "chunks:s:0", "kind": "chunks", "chunk_ids": ["c1"], "result": {}}
        ).encode()
        + b"\n"
        # A line that is not valid UTF-8 at all -- not just truncated JSON.
        + b'{"job_id": "chunks:s:1", "note": "\xe2\x82"}\n'
    )
    seen: list[str] = []

    def call(prompt: str) -> str:
        seen.append(prompt)
        return json.dumps({"c2": TAGS}) + "\nSHIM-DONE exit=0\n"

    stats = lane.run_lane(jobs, out, call=call)
    assert stats == {"done": 1, "partial": 0, "failed": 0, "skipped": 2}
    assert len(seen) == 1
    # The file itself still carries the undecodable line, so read it as bytes.
    record = json.loads(out.read_bytes().splitlines()[-1].decode("utf-8"))
    assert record["job_id"] == "chunks:s:1"
    assert record["result"] == {"c2": TAGS}


def test_a_torn_line_in_the_failed_file_is_fenced_before_the_next_record(
    tmp_path: Path,
) -> None:
    """The same torn-file protection ``_append_record`` gives ``RESULTS`` must
    cover ``RESULTS.failed`` too: a lane killed mid-write to the failed file
    must not glue the next failure record onto the fragment."""
    failed = tmp_path / "out.jsonl.failed"
    failed.write_text('{"job_id": "chunks:s:-1", "err')
    stats = lane.run_lane(
        _job(tmp_path), tmp_path / "out.jsonl", call=lambda p: "no json here\nSHIM-DONE exit=0"
    )
    assert stats == {"done": 0, "partial": 0, "failed": 1, "skipped": 0}
    lines = failed.read_text().splitlines()
    assert lines[0] == '{"job_id": "chunks:s:-1", "err'
    record = json.loads(lines[-1])
    assert record["job_id"] == "chunks:s:0"


def test_a_repeated_job_id_within_one_jobs_file_runs_once(tmp_path: Path) -> None:
    """A job_id that appears twice in the jobs file must not be classified
    twice: the repeat is a skip, not a second shim call and a second record."""
    jobs = tmp_path / "jobs.jsonl"
    line = json.dumps(
        {
            "job_id": "chunks:s:0",
            "kind": "chunks",
            "source_id": "s",
            "chunk_ids": ["c1"],
            "prompt": "P",
        }
    )
    jobs.write_text(line + "\n" + line + "\n")
    calls = 0

    def call(prompt: str) -> str:
        nonlocal calls
        calls += 1
        return json.dumps({"c1": TAGS}) + "\nSHIM-DONE exit=0\n"

    stats = lane.run_lane(jobs, tmp_path / "out.jsonl", call=call)
    assert stats == {"done": 1, "partial": 0, "failed": 0, "skipped": 1}
    assert calls == 1
    out_lines = (tmp_path / "out.jsonl").read_text().splitlines()
    assert len(out_lines) == 1


def test_a_repeated_job_id_that_fails_still_runs_once(tmp_path: Path) -> None:
    """The same de-duplication applies when the job ends up in .failed: the
    repeat is skipped, not retried from scratch."""
    jobs = tmp_path / "jobs.jsonl"
    line = json.dumps(
        {
            "job_id": "chunks:s:0",
            "kind": "chunks",
            "source_id": "s",
            "chunk_ids": ["c1"],
            "prompt": "P",
        }
    )
    jobs.write_text(line + "\n" + line + "\n")
    calls = 0

    def call(prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "no json here\nSHIM-DONE exit=0"

    stats = lane.run_lane(jobs, tmp_path / "out.jsonl", call=call, max_attempts=1)
    assert stats == {"done": 0, "partial": 0, "failed": 1, "skipped": 1}
    assert calls == 1
    failed_lines = (tmp_path / "out.jsonl.failed").read_text().splitlines()
    assert len(failed_lines) == 1


def test_a_job_that_stays_invalid_goes_to_failed(tmp_path: Path) -> None:
    stats = lane.run_lane(
        _job(tmp_path), tmp_path / "out.jsonl", call=lambda p: "no json here\nSHIM-DONE exit=0"
    )
    assert stats == {"done": 0, "partial": 0, "failed": 1, "skipped": 0}
    assert json.loads((tmp_path / "out.jsonl.failed").read_text())["job_id"] == "chunks:s:0"


def test_the_failed_line_keeps_every_attempts_error(tmp_path: Path) -> None:
    replies = iter(['{"c1": {"tf_quality": ["speed"]}}', "no json here"])
    stats = lane.run_lane(
        _job(tmp_path),
        tmp_path / "out.jsonl",
        call=lambda p: next(replies) + "\nSHIM-DONE exit=0\n",
    )
    assert stats == {"done": 0, "partial": 0, "failed": 1, "skipped": 0}
    errors = json.loads((tmp_path / "out.jsonl.failed").read_text())["errors"]
    assert len(errors) == 2
    # attempt 1: the real validator rejected the chunk tags; attempt 2: no answer.
    assert errors[0] == "c1: Field required"
    assert errors[1] == "no JSON object or array in model output"


def test_a_batch_keeps_the_chunks_that_validate_and_re_asks_only_the_others(
    tmp_path: Path,
) -> None:
    """A batch is one answer and a few per-chunk errors in it are the common case
    (VOR-1444: a batch of 20 all-or-nothing amplified a few-% failure rate). The
    valid chunks are written straight away under the job's own id with a part
    suffix, and the retry is asked for the one chunk that did not validate."""
    jobs = _chunks_job(tmp_path, ["c1", "c2", "c3"])
    out = tmp_path / "out.jsonl"
    bad = {**TAGS, "tf_quality": ["speed"]}
    replies = iter([json.dumps({"c1": TAGS, "c2": bad, "c3": TAGS}), json.dumps({"c2": TAGS})])
    prompts: list[str] = []

    def call(prompt: str) -> str:
        prompts.append(prompt)
        return next(replies) + "\nSHIM-DONE exit=0\n"

    stats = lane.run_lane(jobs, out, call=call)
    assert stats == {"done": 1, "partial": 0, "failed": 0, "skipped": 0}
    first, second = (json.loads(line) for line in out.read_text().splitlines())
    assert first["job_id"] == "chunks:s:0#p1"
    assert first["kind"] == "chunks"
    assert first["source_id"] == "s"
    assert first["chunk_ids"] == ["c1", "c3"]
    assert set(first["result"]) == {"c1", "c3"}
    assert first["attempts"] == 1
    assert first["model"] == TF_TAGGER_MODEL
    assert second["job_id"] == "chunks:s:0#p2"
    assert second["chunk_ids"] == ["c2"]
    assert second["attempts"] == 2
    # the retry names the one bad chunk id and not the two that already landed
    assert '"c2"' in prompts[1]
    assert "c1" not in prompts[1] and "c3" not in prompts[1]
    assert "previous answer was invalid for these chunk ids only" in prompts[1]
    assert "c2: Value error, tf_quality: not allowed" in prompts[1]
    assert not (tmp_path / "out.jsonl.failed").exists()


def test_a_chunk_id_outside_the_batch_is_never_written(tmp_path: Path) -> None:
    """A batch that carries an id nobody asked for keeps the ids that validate; the
    extra one is ignored, not written and not retried (VOR-1444)."""
    out = tmp_path / "out.jsonl"
    stats = lane.run_lane(
        _chunks_job(tmp_path, ["c1"]),
        out,
        call=lambda p: json.dumps({"c1": TAGS, "c9": TAGS}) + "\nSHIM-DONE exit=0\n",
    )
    assert stats == {"done": 1, "partial": 0, "failed": 0, "skipped": 0}
    record = json.loads(out.read_text())
    assert record["chunk_ids"] == ["c1"]
    assert set(record["result"]) == {"c1"}


def test_a_batch_answered_as_a_list_of_entries_is_accepted_chunk_by_chunk(tmp_path: Path) -> None:
    """A list of entries carrying their own `chunk_id` is the other shape
    `validate_batch` takes, and the whole-batch path accepted it; per-chunk
    acceptance reads it too, and writes the entries as they answered."""
    out = tmp_path / "out.jsonl"
    entries = [{"chunk_id": chunk_id, **TAGS} for chunk_id in ("c1", "c2")]
    stats = lane.run_lane(
        _chunks_job(tmp_path, ["c1", "c2"]),
        out,
        call=lambda p: json.dumps(entries) + "\nSHIM-DONE exit=0\n",
    )
    assert stats == {"done": 1, "partial": 0, "failed": 0, "skipped": 0}
    record = json.loads(out.read_text())
    assert record["chunk_ids"] == ["c1", "c2"]
    assert record["result"] == {"c1": entries[0], "c2": entries[1]}


def test_resume_asks_only_for_the_chunk_ids_no_record_covers(tmp_path: Path) -> None:
    """A job is finished when its chunk ids are all covered, across the job's own
    record and its part records; the ids still missing are asked for on their own,
    and the part numbering continues after the records already written."""
    out = tmp_path / "out.jsonl"
    out.write_text(
        json.dumps(
            {
                "job_id": "chunks:s:0#p1",
                "kind": "chunks",
                "source_id": "s",
                "chunk_ids": ["c1", "c3"],
                "result": {"c1": TAGS, "c3": TAGS},
                "attempts": 1,
                "model": TF_TAGGER_MODEL,
            }
        )
        + "\n"
    )
    prompts: list[str] = []

    def call(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps({"c2": TAGS}) + "\nSHIM-DONE exit=0\n"

    stats = lane.run_lane(_chunks_job(tmp_path, ["c1", "c2", "c3"]), out, call=call)
    assert stats == {"done": 1, "partial": 0, "failed": 0, "skipped": 0}
    assert len(prompts) == 1
    # The prompt limits the ask to the missing id; the job prompt itself (here the
    # stub "P") still carries every passage, as it must (VOR-1448).
    assert prompts[0].endswith('Return JSON ONLY for these chunk ids: ["c2"]')
    record = json.loads(out.read_text().splitlines()[-1])
    assert record["job_id"] == "chunks:s:0#p2"
    assert record["chunk_ids"] == ["c2"]


def test_the_ids_that_never_validate_go_to_failed_with_their_ids(tmp_path: Path) -> None:
    """A job that landed some chunks and lost the rest is partial, and the .failed
    record names the ids still missing so a later pass can re-export just them."""
    out = tmp_path / "out.jsonl"
    replies = iter([json.dumps({"c1": TAGS}), json.dumps({"c2": {"tf_quality": ["speed"]}})])
    stats = lane.run_lane(
        _chunks_job(tmp_path, ["c1", "c2"]),
        out,
        call=lambda p: next(replies) + "\nSHIM-DONE exit=0\n",
    )
    assert stats == {"done": 0, "partial": 1, "failed": 0, "skipped": 0}
    assert json.loads(out.read_text())["chunk_ids"] == ["c1"]
    failed = json.loads((tmp_path / "out.jsonl.failed").read_text())
    assert failed["job_id"] == "chunks:s:0"
    assert failed["source_id"] == "s"
    assert failed["chunk_ids"] == ["c2"]
    # one error per attempt, each naming the id that attempt failed to answer
    assert failed["errors"] == ["c2: no tags in the answer", "c2: Field required"]


def test_a_failed_chunks_record_omits_a_source_id_the_job_never_had(
    tmp_path: Path,
) -> None:
    """A job built without a source_id (the export always sets one, a hand-written
    jobs file need not) must not be given an empty one: the .failed record carries the
    keys the job has, exactly as a results record does, so a later pass re-exports a
    job by the same identity it was asked for."""
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(
        json.dumps({"job_id": "chunks:0", "kind": "chunks", "chunk_ids": ["c1"], "prompt": "P"})
        + "\n"
    )
    stats = lane.run_lane(
        jobs, tmp_path / "out.jsonl", call=lambda p: "no json here\nSHIM-DONE exit=0"
    )
    assert stats == {"done": 0, "partial": 0, "failed": 1, "skipped": 0}
    failed = json.loads((tmp_path / "out.jsonl.failed").read_text())
    assert "source_id" not in failed
    assert failed["job_id"] == "chunks:0"
    assert failed["chunk_ids"] == ["c1"]


def test_a_batch_where_nothing_validates_fails_with_all_of_its_ids(tmp_path: Path) -> None:
    """Zero accepted chunks is a failed job, not a partial one, and every id goes to
    .failed — none is dropped on the way."""
    stats = lane.run_lane(
        _chunks_job(tmp_path, ["c1", "c2"]),
        tmp_path / "out.jsonl",
        call=lambda p: json.dumps({"c1": {"tf_quality": ["speed"]}}) + "\nSHIM-DONE exit=0\n",
    )
    assert stats == {"done": 0, "partial": 0, "failed": 1, "skipped": 0}
    assert not (tmp_path / "out.jsonl").exists()
    failed = json.loads((tmp_path / "out.jsonl.failed").read_text())
    assert failed["chunk_ids"] == ["c1", "c2"]


def test_max_attempts_one_calls_once_and_fails_the_job(tmp_path: Path) -> None:
    prompts: list[str] = []

    def call(prompt: str) -> str:
        prompts.append(prompt)
        return "no json here\nSHIM-DONE exit=0\n"

    stats = lane.run_lane(
        _job(tmp_path),
        tmp_path / "out.jsonl",
        call=call,
        max_attempts=1,
    )
    assert stats == {"done": 0, "partial": 0, "failed": 1, "skipped": 0}
    assert len(prompts) == 1
    assert len(json.loads((tmp_path / "out.jsonl.failed").read_text())["errors"]) == 1


def test_a_non_zero_shim_exit_is_a_shim_failure_not_a_missing_answer(tmp_path: Path) -> None:
    stats = lane.run_lane(
        _job(tmp_path),
        tmp_path / "out.jsonl",
        call=lambda p: "agent-shim: shared runtime is missing\nSHIM-DONE exit=127\n",
    )
    assert stats == {"done": 0, "partial": 0, "failed": 1, "skipped": 0}
    errors = json.loads((tmp_path / "out.jsonl.failed").read_text())["errors"]
    assert len(errors) == 2
    assert all("shim exit 127" in error for error in errors)


def test_a_missing_sentinel_is_a_shim_failure_too(tmp_path: Path) -> None:
    stats = lane.run_lane(
        _job(tmp_path),
        tmp_path / "out.jsonl",
        call=lambda p: json.dumps({"c1": TAGS}) + "\n",
    )
    assert stats == {"done": 0, "partial": 0, "failed": 1, "skipped": 0}
    errors = json.loads((tmp_path / "out.jsonl.failed").read_text())["errors"]
    assert all("sentinel" in error for error in errors)


def test_a_book_job_is_validated_against_the_book_schema(tmp_path: Path) -> None:
    """A book result is one object, not a chunk map: the chunk map is rejected and
    the retry produces the book record."""
    replies = iter([json.dumps({"c1": TAGS}), json.dumps(BOOK_TAGS)])
    stats = lane.run_lane(
        _book_job(tmp_path),
        tmp_path / "out.jsonl",
        call=lambda p: next(replies) + "\nSHIM-DONE exit=0\n",
    )
    assert stats == {"done": 1, "partial": 0, "failed": 0, "skipped": 0}
    record = json.loads((tmp_path / "out.jsonl").read_text())
    assert record["attempts"] == 2
    assert record["result"]["tf_source_type"] == "coaching_manual"
    assert "chunk_ids" not in record
    assert record["successful_prompt_sha256"]
    assert record["successful_prompt_identity_status"] == "computed_actual_attempt"


def test_a_book_job_the_schema_rejects_goes_to_failed(tmp_path: Path) -> None:
    bogus = {**BOOK_TAGS, "tf_source_type": "hobbyist_blog"}
    stats = lane.run_lane(
        _book_job(tmp_path),
        tmp_path / "out.jsonl",
        call=lambda p: json.dumps(bogus) + "\nSHIM-DONE exit=0\n",
    )
    assert stats == {"done": 0, "partial": 0, "failed": 1, "skipped": 0}
    errors = json.loads((tmp_path / "out.jsonl.failed").read_text())["errors"]
    assert all("tf_source_type" in error for error in errors)


def test_every_result_records_the_model_that_produced_it(tmp_path: Path) -> None:
    stats = lane.run_lane(
        _job(tmp_path),
        tmp_path / "out.jsonl",
        call=lambda p: json.dumps({"c1": TAGS}) + "\nSHIM-DONE exit=0\n",
        model="opencode/glm-5",
    )
    assert stats == {"done": 1, "partial": 0, "failed": 0, "skipped": 0}
    assert json.loads((tmp_path / "out.jsonl").read_text())["model"] == "opencode/glm-5"


def test_a_result_falls_back_to_the_configured_default_model(tmp_path: Path) -> None:
    lane.run_lane(
        _job(tmp_path),
        tmp_path / "out.jsonl",
        call=lambda p: json.dumps({"c1": TAGS}) + "\nSHIM-DONE exit=0\n",
    )
    assert json.loads((tmp_path / "out.jsonl").read_text())["model"] == TF_TAGGER_MODEL


def test_shim_call_runs_the_shim_outside_the_repo_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shim needs the system Python 3.14; the repo venv's 3.12 must not win on
    PATH, and nothing may point the shim back at the venv."""
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(stdout="ok\nSHIM-DONE exit=0\n", stderr="")

    monkeypatch.setenv("VIRTUAL_ENV", "/repo/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/repo/.venv")
    monkeypatch.setenv("OPENCODE_CONFIG", "/elsewhere/opencode.json")
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", "/elsewhere")
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", '{"permission": {"*": "allow"}}')
    monkeypatch.setattr(lane.sys, "prefix", "/other/venv")
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join(["/repo/.venv/bin", "/usr/bin", "/other/venv/bin", "/usr/local/bin"]),
    )
    monkeypatch.setattr(lane, "_run_shim", fake_run)
    assert lane.shim_call("provider/model", tmp_path, SHIM_PATH)("P") == "ok\nSHIM-DONE exit=0\n"
    env = captured["env"]
    assert "VIRTUAL_ENV" not in env
    assert not [key for key in env if key.startswith("UV_")]
    assert json.loads(env["OPENCODE_CONFIG_CONTENT"]) == lane.TOOL_POLICY
    assert "OPENCODE_CONFIG" not in env
    assert "OPENCODE_CONFIG_DIR" not in env
    assert env["PATH"].split(os.pathsep) == ["/usr/bin", "/usr/local/bin"]


@pytest.mark.parametrize(
    "exc",
    [
        OSError("[Errno 2] No such file or directory: 'agent-shim.sh'"),
        subprocess.TimeoutExpired(cmd="agent-shim.sh", timeout=900),
    ],
)
def test_a_shim_that_cannot_run_is_a_recorded_attempt_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        raise exc

    monkeypatch.setattr(lane, "_run_shim", fake_run)
    stats = lane.run_lane(
        _job(tmp_path),
        tmp_path / "out.jsonl",
        call=lane.shim_call("m", tmp_path, SHIM_PATH),
        max_attempts=1,
    )
    assert stats == {"done": 0, "partial": 0, "failed": 1, "skipped": 0}
    errors = json.loads((tmp_path / "out.jsonl.failed").read_text())["errors"]
    assert len(errors) == 1
    assert "did not run" in errors[0]
    assert str(exc)[:60] in errors[0]


def test_a_shim_failure_is_retried_like_a_bad_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("shim vanished")
        return SimpleNamespace(stdout=json.dumps({"c1": TAGS}) + "\nSHIM-DONE exit=0\n", stderr="")

    monkeypatch.setattr(lane, "_run_shim", fake_run)
    stats = lane.run_lane(
        _job(tmp_path), tmp_path / "out.jsonl", call=lane.shim_call("m", tmp_path, SHIM_PATH)
    )
    assert stats == {"done": 1, "partial": 0, "failed": 0, "skipped": 0}
    assert calls == 2
    assert json.loads((tmp_path / "out.jsonl").read_text())["attempts"] == 2


def test_a_non_zero_shim_exit_carries_the_stderr_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        return SimpleNamespace(stdout="SHIM-DONE exit=1\n", stderr="boom\ndetail here\n")

    monkeypatch.setattr(lane, "_run_shim", fake_run)
    with pytest.raises(lane.ShimError) as raised:
        lane.shim_call("m", tmp_path, SHIM_PATH)("P")
    assert "shim exit 1" in str(raised.value)
    assert "detail here" in str(raised.value)


def test_the_cli_passes_an_explicit_provider_model_to_the_shim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no API URL the shim path still works for a routable provider/model."""
    seen: dict[str, Any] = {}
    monkeypatch.delenv("VORAG_TF_TAGGER_URL", raising=False)

    def fake_shim_call(model: str, workdir: Path, shim: Path) -> Any:
        seen["shim_model"] = model
        seen["shim_path"] = shim
        return lambda prompt: ""

    def fake_run_lane(jobs: Path, results: Path, *, call: Any, **kwargs: Any) -> dict[str, int]:
        seen.update(kwargs)
        return {"done": 0, "failed": 0, "skipped": 0}

    monkeypatch.setattr(lane, "shim_call", fake_shim_call)
    monkeypatch.setattr(lane, "run_lane", fake_run_lane)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tf_classify_lane.py",
            "jobs.jsonl",
            "results.jsonl",
            "--shim",
            "/path/to/agent-shim.sh",
            "--model",
            "opencode/m",
        ],
    )
    lane.main()
    assert seen["shim_model"] == "opencode/m"
    assert seen["shim_path"] == Path("/path/to/agent-shim.sh")
    assert seen["model"] == "opencode/m"


def test_shim_call_survives_its_workdir_being_removed_mid_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Something outside the lane deleted its temp workdir mid-call in the live run
    (VOR-1445): the prompt file's cleanup raised FileNotFoundError and killed the
    lane. The call still returns its answer, and the next call recreates the dir."""
    workdir = tmp_path / "work"
    workdir.mkdir()

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        assert Path(cmd[2]).parent == workdir
        assert Path(cmd[2]).read_text() == "P"
        shutil.rmtree(workdir)
        return SimpleNamespace(stdout="ok\nSHIM-DONE exit=0\n", stderr="")

    monkeypatch.setattr(lane, "_run_shim", fake_run)
    call = lane.shim_call("m", workdir, SHIM_PATH)
    assert call("P") == "ok\nSHIM-DONE exit=0\n"
    assert call("P") == "ok\nSHIM-DONE exit=0\n"


def test_a_prompt_file_that_cannot_be_written_is_a_shim_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A workdir removed between the mkdir and the write surfaces as a ShimError,
    which run_lane retries, never as an OSError that ends the lane (VOR-1445)."""

    def fake_tempfile(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("workdir gone")

    monkeypatch.setattr(lane.tempfile, "NamedTemporaryFile", fake_tempfile)
    with pytest.raises(lane.ShimError, match="shim workdir not writable"):
        lane.shim_call("m", tmp_path, SHIM_PATH)("P")


def test_the_cli_survives_its_workdir_being_removed_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lane's TemporaryDirectory is removed from outside before main exits:
    its cleanup must not raise (VOR-1445). The stdlib already skips a directory
    that is wholly gone; ``ignore_cleanup_errors`` covers one removed part-way,
    and this pins the contract whichever way the removal lands."""
    seen: dict[str, Path] = {}

    def fake_shim_call(model: str, workdir: Path, shim: Path) -> Any:
        seen["workdir"] = workdir
        return lambda prompt: ""

    def fake_run_lane(jobs: Path, results: Path, *, call: Any, **kwargs: Any) -> dict[str, int]:
        shutil.rmtree(seen["workdir"])
        return {"done": 0, "partial": 0, "failed": 0, "skipped": 0}

    monkeypatch.setattr(lane, "shim_call", fake_shim_call)
    monkeypatch.setattr(lane, "run_lane", fake_run_lane)
    monkeypatch.delenv("VORAG_TF_TAGGER_URL", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tf_classify_lane.py",
            "jobs.jsonl",
            "results.jsonl",
            "--shim",
            "/path/to/agent-shim.sh",
            "--model",
            "opencode/m",
        ],
    )
    lane.main()
    assert not seen["workdir"].exists()


def _alive(pid: int) -> bool:
    """A zombie counts as dead: whether its reaper has collected it yet is not the
    lane's business."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return stat.rpartition(b")")[2].split()[0] != b"Z"


def test_a_shim_that_overruns_is_reaped_with_its_children(tmp_path: Path) -> None:
    """The live run leaked an `opencode run` for 1000s+ after the 900s timeout: the
    timeout killed the shim but not the model process it had spawned (VOR-1446).
    An overrun kills the shim's whole session."""
    pidfile = tmp_path / "child.pid"
    # `set -m` puts the child in its own process group, as the real shim does with
    # `opencode run`: only the session ties it to the shim.
    script = f"set -m; sleep 60 & echo $! > {pidfile}; wait"
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        lane._run_shim(["bash", "-c", script], cwd=tmp_path, env=dict(os.environ), timeout=1)
    # A surviving child holds the output pipes open, so an unreaped one shows up as
    # the call hanging until it exits by itself.
    assert time.monotonic() - started < 10
    child = int(pidfile.read_text())
    for _ in range(50):
        if not _alive(child):
            break
        time.sleep(0.05)
    else:
        os.kill(child, 9)
        pytest.fail("the shim's child outlived the timeout")


def test_run_shim_returns_the_shims_output(tmp_path: Path) -> None:
    out = lane._run_shim(
        ["bash", "-c", "echo hi; echo err >&2"], cwd=tmp_path, env=dict(os.environ), timeout=10
    )
    assert out.stdout == "hi\n"
    assert out.stderr == "err\n"
    assert out.returncode == 0


def test_the_tool_policy_denies_every_tool_that_touches_the_machine() -> None:
    """Pins TOOL_POLICY (see its comment for why bash/read stay offered, VOR-1449)."""
    rules = lane.TOOL_POLICY["permission"]
    assert set(rules) == {
        "*",
        "bash",
        "read",
        "grep",
        "glob",
        "list",
        "edit",
        "external_directory",
        "webfetch",
        "websearch",
        "codesearch",
        "task",
        "skill",
        "lsp",
    }
    assert rules["*"] == "deny"
    assert rules["bash"] == {"*": "deny", "true": "allow"}
    assert rules["read"] == {"*": "deny", "/nonexistent/tf-lane-sentinel": "allow"}
    for tool in set(rules) - {"*", "bash", "read"}:
        assert rules[tool] == "deny", tool


def test_every_shim_call_runs_with_all_model_tools_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The classifier runs under `opencode run --auto`, which approves every tool
    that is not explicitly denied. In the live run it grepped the home directory,
    queried Postgres and Qdrant, and deleted its own prompt file (VOR-1447). The
    workdir carries a deny-all project config on every call, even one the model or
    anything else removed since."""
    seen: list[dict[str, Any]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        seen.append(json.loads((kwargs["cwd"] / "opencode.json").read_text()))
        (kwargs["cwd"] / "opencode.json").unlink()
        return SimpleNamespace(stdout="ok\nSHIM-DONE exit=0\n", stderr="")

    monkeypatch.setattr(lane, "_run_shim", fake_run)
    call = lane.shim_call("m", tmp_path, SHIM_PATH)
    call("P")
    call("P")
    assert seen == [lane.TOOL_POLICY] * 2


def test_every_retry_carries_the_whole_job_prompt(tmp_path: Path) -> None:
    """Each shim call is a fresh model session: a retry that names the ids but not
    the chunks leaves the model nothing to classify. In the live run it went
    looking for the chunks on disk, and some retries validated tags for passages it
    never saw (VOR-1448). The retry is the job prompt plus what was wrong."""
    prompts: list[str] = []
    replies = iter(["no json here", json.dumps({"c1": TAGS, "c2": TAGS})])

    def call(prompt: str) -> str:
        prompts.append(prompt)
        return next(replies) + "\nSHIM-DONE exit=0\n"

    job = _chunks_job(tmp_path, ["c1", "c2"])
    job.write_text(job.read_text().replace('"prompt": "P"', '"prompt": "CHUNK TEXT FOR c1 AND c2"'))
    stats = lane.run_lane(job, tmp_path / "out.jsonl", call=call)
    assert stats == {"done": 1, "partial": 0, "failed": 0, "skipped": 0}
    assert len(prompts) == 2
    assert prompts[1].startswith("CHUNK TEXT FOR c1 AND c2")
    assert "no JSON object or array in model output" in prompts[1]


def test_a_resumed_job_asks_with_the_job_prompt_and_no_invented_error(tmp_path: Path) -> None:
    """A job some record already covers is resumed with the full job prompt, limited
    to the missing ids, and without an error the model never made: the old
    "no tags in any results record" line sent it hunting for a results record
    (VOR-1448)."""
    out = tmp_path / "out.jsonl"
    out.write_text(
        json.dumps(
            {
                "job_id": "chunks:s:0#p1",
                "kind": "chunks",
                "source_id": "s",
                "chunk_ids": ["c1"],
                "result": {"c1": TAGS},
                "attempts": 1,
                "model": TF_TAGGER_MODEL,
            }
        )
        + "\n"
    )
    prompts: list[str] = []

    def call(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps({"c2": TAGS}) + "\nSHIM-DONE exit=0\n"

    job = _chunks_job(tmp_path, ["c1", "c2"])
    job.write_text(job.read_text().replace('"prompt": "P"', '"prompt": "CHUNK TEXT"'))
    lane.run_lane(job, out, call=call)
    assert prompts[0].startswith("CHUNK TEXT")
    assert '["c2"]' in prompts[0]
    assert "invalid" not in prompts[0]
    assert "results record" not in prompts[0]


def test_a_retry_after_a_partial_landing_carries_the_job_prompt_and_the_missing_ids(
    tmp_path: Path,
) -> None:
    """The defect that mis-tagged live chunks: after some ids land, the retry for the
    rest must still carry the passages (VOR-1448)."""
    prompts: list[str] = []
    replies = iter([json.dumps({"c1": TAGS, "c2": {"bad": 1}}), json.dumps({"c2": TAGS})])

    def call(prompt: str) -> str:
        prompts.append(prompt)
        return next(replies) + "\nSHIM-DONE exit=0\n"

    job = _chunks_job(tmp_path, ["c1", "c2"])
    job.write_text(job.read_text().replace('"prompt": "P"', '"prompt": "CHUNK TEXT"'))
    stats = lane.run_lane(job, tmp_path / "out.jsonl", call=call)
    assert stats == {"done": 1, "partial": 0, "failed": 0, "skipped": 0}
    assert prompts[1].startswith("CHUNK TEXT")
    assert 'Return corrected JSON ONLY for these ids: ["c2"]' in prompts[1]


def test_an_answer_keyed_by_short_labels_lands_under_the_real_chunk_ids(tmp_path: Path) -> None:
    """Under batched load the local model garbled long ids it copied back
    ("fd0290a0" → "fd0290290a0"), so the prompt keys passages p1..pN and the job
    carries the label → chunk id map; the runner translates before validating, and
    a retry names labels, never ids (VOR-1451)."""
    job = tmp_path / "jobs.jsonl"
    job.write_text(
        json.dumps(
            {
                "job_id": "chunks:s:0",
                "kind": "chunks",
                "source_id": "s",
                "chunk_ids": ["uuid-a", "uuid-b"],
                "keys": {"p1": "uuid-a", "p2": "uuid-b"},
                "prompt": "CHUNKS",
            }
        )
        + "\n"
    )
    prompts: list[str] = []
    replies = iter([json.dumps({"p1": TAGS, "p2": {"bad": 1}}), json.dumps({"p2": TAGS})])

    def call(prompt: str) -> str:
        prompts.append(prompt)
        return next(replies) + "\nSHIM-DONE exit=0\n"

    out = tmp_path / "out.jsonl"
    stats = lane.run_lane(job, out, call=call)
    assert stats == {"done": 1, "partial": 0, "failed": 0, "skipped": 0}
    covered = {c for line in out.read_text().splitlines() for c in json.loads(line)["chunk_ids"]}
    assert covered == {"uuid-a", "uuid-b"}
    assert '["p2"]' in prompts[1]
    assert "uuid-b" not in prompts[1]


def test_api_call_posts_a_plain_chat_completion_with_no_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The classifier is text in, text out: it calls a model API directly, never an
    agent CLI that could run tools (VOR-1451). Thinking is off and temperature 0."""
    seen: dict[str, Any] = {}

    class Resp:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": '{"p1": {}}'}}]}).encode()

    def fake_urlopen(req: Any, timeout: float) -> Any:
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data)
        seen["auth"] = req.get_header("Authorization")
        return Resp()

    monkeypatch.setenv("TF_KEY", "secret")
    monkeypatch.setattr(lane, "_open", fake_urlopen)
    out = lane.api_call("m1", "http://host/v1/chat/completions", "TF_KEY")("PROMPT")
    assert out.startswith('{"p1": {}}') and "SHIM-DONE exit=0" in out
    body = seen["body"]
    assert body["model"] == "m1" and body["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert body["temperature"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "tools" not in body
    assert seen["auth"] == "Bearer secret"


def test_an_api_failure_is_a_retryable_attempt_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(req: Any, timeout: float) -> Any:
        raise OSError("connection refused")

    monkeypatch.setattr(lane, "_open", boom)
    with pytest.raises(lane.ShimError, match="api call failed"):
        lane.api_call("m", "http://host/x", "NOPE_UNSET")("P")


def test_the_cli_uses_the_api_when_a_url_is_given(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_api_call(model: str, url: str, key_env: str) -> Any:
        seen.update(model=model, url=url, key_env=key_env)
        return lambda prompt: ""

    def fake_shim_call(model: str, workdir: Path, shim: Path) -> Any:
        raise AssertionError("the shim must not be used when an API url is given")

    def fake_run_lane(jobs: Path, results: Path, *, call: Any, **kwargs: Any) -> dict[str, int]:
        seen["model_recorded"] = kwargs["model"]
        return {"done": 0, "partial": 0, "failed": 0, "skipped": 0}

    monkeypatch.setattr(lane, "api_call", fake_api_call)
    monkeypatch.setattr(lane, "shim_call", fake_shim_call)
    monkeypatch.setattr(lane, "run_lane", fake_run_lane)
    monkeypatch.setattr(
        sys,
        "argv",
        ["tf_classify_lane.py", "j.jsonl", "r.jsonl", "--api-url", "http://h/v1/chat/completions"],
    )
    lane.main()
    assert seen["url"] == "http://h/v1/chat/completions"
    assert seen["model"] == TF_TAGGER_MODEL == seen["model_recorded"]
    assert seen["key_env"] == "LLAMA_SWAP_API_KEY"


def test_a_list_shaped_answer_keyed_by_labels_is_translated(tmp_path: Path) -> None:
    job = tmp_path / "jobs.jsonl"
    job.write_text(
        json.dumps(
            {
                "job_id": "chunks:s:0",
                "kind": "chunks",
                "source_id": "s",
                "chunk_ids": ["uuid-a"],
                "keys": {"p1": "uuid-a"},
                "prompt": "P",
            }
        )
        + "\n"
    )
    reply = json.dumps([{"chunk_id": "p1", **TAGS}])
    stats = lane.run_lane(
        job, tmp_path / "out.jsonl", call=lambda p: reply + "\nSHIM-DONE exit=0\n"
    )
    assert stats == {"done": 1, "partial": 0, "failed": 0, "skipped": 0}


def test_records_carry_the_job_tag_version(tmp_path: Path) -> None:
    job = tmp_path / "jobs.jsonl"
    job.write_text(
        json.dumps(
            {
                "job_id": "chunks:s:0",
                "kind": "chunks",
                "source_id": "s",
                "chunk_ids": ["c1"],
                "tag_version": 7,
                "prompt": "P",
            }
        )
        + "\n"
    )
    lane.run_lane(
        job,
        tmp_path / "out.jsonl",
        call=lambda p: json.dumps({"c1": TAGS}) + "\nSHIM-DONE exit=0\n",
    )
    assert json.loads((tmp_path / "out.jsonl").read_text())["tag_version"] == 7


def _fake_response(payload: dict[str, Any]) -> Any:
    class Resp:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(payload).encode()

    return Resp()


def test_a_truncated_completion_is_named_as_such(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = _fake_response(
        {"choices": [{"message": {"content": '{"p1": {'}, "finish_reason": "length"}]}
    )
    monkeypatch.setattr(lane, "_open", lambda req, timeout: resp)
    with pytest.raises(lane.ShimError, match="truncated"):
        lane.api_call("m", "http://h/v1/chat/completions", "NOPE")("P")


def test_api_call_refuses_a_non_http_url_and_ignores_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="http"):
        lane.api_call("m", "file:///etc/passwd", "NOPE")
    # The lane's opener must carry no env-proxy handler, where a default opener
    # built under the same environment does.
    monkeypatch.setenv("http_proxy", "http://evil:3128")
    default = urllib.request.build_opener()

    def proxied(opener: Any) -> bool:
        return any(
            isinstance(h, urllib.request.ProxyHandler) and bool(getattr(h, "proxies", None))
            for h in opener.handlers
        )

    assert proxied(default)
    assert not proxied(lane._OPENER)


def test_the_shim_fallback_needs_a_routable_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an API URL the lane falls back to the shim, which can only route
    provider/model ids (VOR-1451)."""
    monkeypatch.delenv("VORAG_TF_TAGGER_URL", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["tf_classify_lane.py", "j.jsonl", "r.jsonl", "--shim", str(SHIM_PATH)]
    )
    with pytest.raises(SystemExit):
        lane.main()


def test_a_model_without_a_shim_or_api_url_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The shim fallback runs only when --shim is given: --model alone is not enough."""
    monkeypatch.delenv("VORAG_TF_TAGGER_URL", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["tf_classify_lane.py", "j.jsonl", "r.jsonl", "--model", "provider/model"]
    )
    with pytest.raises(SystemExit) as raised:
        lane.main()
    assert raised.value.code == 2
    assert "--shim" in capsys.readouterr().err

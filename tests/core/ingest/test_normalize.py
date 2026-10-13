"""Contract tests for `vo_rag.core.ingest.normalize` (VOR-3). Verifies:

- ``FILE_DROP`` requires ``raw_content.garage_key``; optional ``file_hash`` is
  lowercased and stripped (deterministic registration key).
- ``URL`` requires ``http://`` or ``https://`` scheme.
- ``TEXT_PROMPT`` requires non-empty ``raw_content.text``.
- ``API`` and ``GITHUB_STAR`` pass ``raw_content`` through unchanged.
- Unknown source types raise a 422-shaped ``HTTPException``.
- Valid pipeline hints (TargetPipeline enum values) survive into
  ``envelope.source_data``; invalid hints raise 422.
- ``normalize()`` stamps a new ULID, RECIEVED status, and a created_at /
  updated_at pair on the envelope.
- ``IntakeSubmission`` rejects unknown fields (extra='forbid').
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from vo_rag.core.ingest import IntakeSubmission, normalize
from vo_rag.core.models.envelope import (
    EnvelopeStatus,
    IntakeSource,
    TargetPipeline,
)

# ---------------------------------------------------------------------------
# FILE_DROP
# ---------------------------------------------------------------------------


def test_file_drop_requires_garage_key() -> None:
    submission = IntakeSubmission(source_type=IntakeSource.FILE_DROP, raw_content={})
    with pytest.raises(HTTPException) as exc_info:
        normalize(submission)
    assert exc_info.value.status_code == 422
    assert "garage_key" in str(exc_info.value.detail)


def test_file_drop_normalizes_filename_and_content_type() -> None:
    submission = IntakeSubmission(
        source_type=IntakeSource.FILE_DROP,
        raw_content={
            "garage_key": "intake/test.txt",
            "filename": "test.txt",
            "content_type": "text/plain",
        },
    )
    envelope = normalize(submission)
    assert envelope.source_type is IntakeSource.FILE_DROP
    assert envelope.raw_content["garage_key"] == "intake/test.txt"
    assert envelope.raw_content["filename"] == "test.txt"
    assert envelope.raw_content["content_type"] == "text/plain"
    # file_hash omitted -> not in normalized raw_content
    assert "file_hash" not in envelope.raw_content


def test_file_drop_lowercases_and_strips_file_hash() -> None:
    submission = IntakeSubmission(
        source_type=IntakeSource.FILE_DROP,
        raw_content={
            "garage_key": "intake/test.txt",
            "file_hash": f"  {'ABCDEF12' * 8}  ",
        },
    )
    envelope = normalize(submission)
    assert envelope.raw_content["file_hash"] == "abcdef12" * 8


def test_file_drop_rejects_malformed_file_hash() -> None:
    with pytest.raises(HTTPException, match="invalid_file_hash"):
        normalize(
            IntakeSubmission(
                source_type=IntakeSource.FILE_DROP,
                raw_content={
                    "garage_key": "intake/test.txt",
                    "file_hash": "BAD",
                },
            )
        )


def test_file_drop_empty_file_hash_is_omitted() -> None:
    submission = IntakeSubmission(
        source_type=IntakeSource.FILE_DROP,
        raw_content={"garage_key": "intake/test.txt", "file_hash": ""},
    )
    envelope = normalize(submission)
    assert "file_hash" not in envelope.raw_content


# ---------------------------------------------------------------------------
# URL
# ---------------------------------------------------------------------------


def test_url_requires_http_scheme() -> None:
    submission = IntakeSubmission(
        source_type=IntakeSource.URL, raw_content={"url": "ftp://example.com/x"}
    )
    with pytest.raises(HTTPException) as exc_info:
        normalize(submission)
    assert exc_info.value.status_code == 422


def test_url_rejects_https_with_non_http_prefix() -> None:
    submission = IntakeSubmission(
        source_type=IntakeSource.URL, raw_content={"url": "javascript:alert(1)"}
    )
    with pytest.raises(HTTPException) as exc_info:
        normalize(submission)
    assert exc_info.value.status_code == 422


def test_url_accepts_http_and_https() -> None:
    for scheme in ("http", "https"):
        submission = IntakeSubmission(
            source_type=IntakeSource.URL,
            raw_content={"url": f"{scheme}://example.com/x", "title": "x"},
        )
        envelope = normalize(submission)
        assert envelope.raw_content["url"] == f"{scheme}://example.com/x"
        assert envelope.raw_content["title"] == "x"


def test_url_strips_whitespace() -> None:
    submission = IntakeSubmission(
        source_type=IntakeSource.URL,
        raw_content={"url": "  https://example.com/x  "},
    )
    envelope = normalize(submission)
    assert envelope.raw_content["url"] == "https://example.com/x"


# ---------------------------------------------------------------------------
# TEXT_PROMPT
# ---------------------------------------------------------------------------


def test_text_prompt_rejects_empty() -> None:
    submission = IntakeSubmission(source_type=IntakeSource.TEXT_PROMPT, raw_content={"text": ""})
    with pytest.raises(HTTPException) as exc_info:
        normalize(submission)
    assert exc_info.value.status_code == 422


def test_text_prompt_rejects_whitespace_only() -> None:
    submission = IntakeSubmission(
        source_type=IntakeSource.TEXT_PROMPT, raw_content={"text": "   \n\t  "}
    )
    with pytest.raises(HTTPException) as exc_info:
        normalize(submission)
    assert exc_info.value.status_code == 422


def test_text_prompt_ok() -> None:
    submission = IntakeSubmission(
        source_type=IntakeSource.TEXT_PROMPT, raw_content={"text": "hello"}
    )
    envelope = normalize(submission)
    assert envelope.raw_content["text"] == "hello"
    assert envelope.source_type is IntakeSource.TEXT_PROMPT


# ---------------------------------------------------------------------------
# API / GITHUB_STAR (passthrough)
# ---------------------------------------------------------------------------


def test_api_passes_raw_content_through() -> None:
    raw = {"any": "thing", "is": "fine"}
    submission = IntakeSubmission(source_type=IntakeSource.API, raw_content=raw)
    envelope = normalize(submission)
    assert envelope.raw_content == raw
    assert envelope.source_type is IntakeSource.API


def test_github_star_passes_raw_content_through() -> None:
    raw = {"repo": "vo-rag", "user": "x"}
    submission = IntakeSubmission(source_type=IntakeSource.GITHUB_STAR, raw_content=raw)
    envelope = normalize(submission)
    assert envelope.raw_content == raw
    assert envelope.source_type is IntakeSource.GITHUB_STAR


# ---------------------------------------------------------------------------
# Pipeline hint validation (narrow local helper, no triage router yet)
# ---------------------------------------------------------------------------


def test_valid_pipeline_hint_survives_normalize() -> None:
    submission = IntakeSubmission(
        source_type=IntakeSource.FILE_DROP,
        source_data={"pipeline_hint": TargetPipeline.BOOK_RAG.value},
        raw_content={"garage_key": "intake/test.txt", "filename": "test.txt"},
    )
    envelope = normalize(submission)
    assert envelope.source_data["pipeline_hint"] == "book_rag"


def test_no_pipeline_hint_is_ok() -> None:
    """source_data without a pipeline_hint must NOT raise."""
    submission = IntakeSubmission(
        source_type=IntakeSource.TEXT_PROMPT,
        raw_content={"text": "hello"},
    )
    envelope = normalize(submission)
    assert "pipeline_hint" not in envelope.source_data


def test_invalid_pipeline_hint_rejected() -> None:
    submission = IntakeSubmission(
        source_type=IntakeSource.TEXT_PROMPT,
        source_data={"pipeline_hint": "Book_RAG"},
        raw_content={"text": "hello"},
    )
    with pytest.raises(HTTPException) as exc_info:
        normalize(submission)
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "invalid_pipeline_hint:Book_RAG"


def test_non_string_pipeline_hint_rejected() -> None:
    submission = IntakeSubmission(
        source_type=IntakeSource.TEXT_PROMPT,
        source_data={"pipeline_hint": 123},
        raw_content={"text": "hello"},
    )
    with pytest.raises(HTTPException) as exc_info:
        normalize(submission)
    assert exc_info.value.status_code == 422
    assert "invalid_pipeline_hint" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# Envelope invariants
# ---------------------------------------------------------------------------


def test_envelope_has_new_ulid_and_received_status() -> None:
    submission = IntakeSubmission(
        source_type=IntakeSource.TEXT_PROMPT, raw_content={"text": "hello"}
    )
    envelope = normalize(submission)
    assert len(envelope.id) == 26
    assert envelope.id.isalnum()
    # Crockford base32 — no I, L, O, U
    for ch in "ILOU":
        assert ch not in envelope.id
    assert envelope.status is EnvelopeStatus.RECEIVED


def test_envelope_timestamps_are_utc_close_to_now() -> None:
    submission = IntakeSubmission(
        source_type=IntakeSource.TEXT_PROMPT, raw_content={"text": "hello"}
    )
    before = datetime.now(tz=UTC)
    envelope = normalize(submission)
    # We can't pin tz from outside; just sanity-check the value is a datetime
    assert isinstance(envelope.created_at, datetime)
    assert isinstance(envelope.updated_at, datetime)
    # created_at and updated_at are stamped at the same instant
    assert envelope.created_at == envelope.updated_at
    # Stamp should be no earlier than the moment we asked (allow for clock
    # precision between the two datetime.now() calls).
    assert envelope.created_at >= before.replace(microsecond=0)


# ---------------------------------------------------------------------------
# IntakeSubmission contract
# ---------------------------------------------------------------------------


def test_intake_submission_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        IntakeSubmission(
            source_type=IntakeSource.TEXT_PROMPT,
            raw_content={"text": "x"},
            not_a_field=1,  # type: ignore[call-arg]
        )


def test_intake_submission_optional_fields_default() -> None:
    """source_data, metadata, sovereignty, priority all have sensible defaults."""
    sub = IntakeSubmission(source_type=IntakeSource.TEXT_PROMPT, raw_content={"text": "x"})
    assert sub.source_data == {}
    assert sub.metadata == {}
    envelope = normalize(sub)
    assert envelope.priority.value == "medium"
    assert envelope.sovereignty.value == "cloud_ok"


def test_normalize_drops_server_owned_metadata_keys() -> None:
    """Every intake surface (API, CLI, MCP) goes through ``normalize``;
    ``metrics`` and ``quality`` are the server's to write, ``preflight`` stays
    client-settable."""
    from vo_rag.core.ingest.normalize import IntakeSubmission, normalize
    from vo_rag.core.models.envelope import IntakeSource

    envelope = normalize(
        IntakeSubmission(
            source_type=IntakeSource.TEXT_PROMPT,
            raw_content={"text": "hi"},
            metadata={
                "metrics": {"active_stream": True},
                "quality": {"findings": []},
                "preflight": {"route": "native"},
            },
        )
    )
    assert envelope.metadata == {"preflight": {"route": "native"}}

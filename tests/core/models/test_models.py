"""Contract tests for vo_rag.core.models.

Verifies:
- VoBaseModel config: extra='forbid' rejects unknown fields
- ULID regex: 26 chars, Crockford base32 (excludes I, L, O, U)
- ulid_new() format
- F-031: trust_level defaults to QUARANTINED (fail-closed)
- StrEnum round-trip
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from vo_rag.core.models import (
    ULID,
    ArtifactType,
    ComplianceMode,
    IntakeEnvelope,
    IntakeSource,
    KnowledgeArtifact,
    Priority,
    ProvenanceEntry,
    TargetPipeline,
    TriageResult,
    TrustLevel,
    VoBaseModel,
    ulid_new,
)

# ---------------------------------------------------------------------------
# ulid_new() format
# ---------------------------------------------------------------------------


def test_ulid_new_is_26_char_alnum_string() -> None:
    s = ulid_new()
    assert isinstance(s, str)
    assert len(s) == 26
    assert s.isalnum()
    # Crockford base32: must not contain I, L, O, U (visually ambiguous)
    for ch in "ILOU":
        assert ch not in s, f"ulid contains forbidden char {ch!r}: {s!r}"


def test_ulid_regex_accepts_valid() -> None:
    # A known-valid ULID (Crockford base32, 26 chars, no I/L/O/U)
    valid = "01ARZ3NDEKTSV4RRFFQ69G5FAV"

    class Holder(VoBaseModel):
        ulid_field: ULID

    m = Holder(ulid_field=valid)
    assert m.ulid_field == valid


@pytest.mark.parametrize(
    "bad",
    [
        "short",  # too short
        "x" * 27,  # too long
        "01arz3ndektsv4rrffq69g5fav",  # lowercase
        "01ARZ3NDEKTSV4RRFFQ69G5FAI",  # contains I
        "01ARZ3NDEKTSV4RRFFQ69G5FAL",  # contains L
        "01ARZ3NDEKTSV4RRFFQ69G5FAO",  # contains O
        "01ARZ3NDEKTSV4RRFFQ69G5FAU",  # contains U
        "01ARZ3NDEKTSV4RRFFQ69G5FA!",  # punctuation
    ],
)
def test_ulid_regex_rejects_invalid(bad: str) -> None:
    class Holder(VoBaseModel):
        ulid_field: ULID

    with pytest.raises(ValidationError):
        Holder(ulid_field=bad)


# ---------------------------------------------------------------------------
# VoBaseModel config: extra='forbid', validate_assignment, str_strip_whitespace
# ---------------------------------------------------------------------------


def test_vobasemodel_rejects_unknown_field() -> None:
    class Thing(VoBaseModel):
        name: str

    with pytest.raises(ValidationError):
        Thing(name="ok", not_a_field="boom")  # type: ignore[call-arg]


def test_vobasemodel_strips_string_whitespace() -> None:
    class Thing(VoBaseModel):
        name: str

    t = Thing(name="  padded  ")
    assert t.name == "padded"


def test_vobasemodel_revalidates_on_assignment() -> None:
    class Thing(VoBaseModel):
        n: int

    t = Thing(n=1)
    with pytest.raises(ValidationError):
        t.n = "not an int"  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# F-031: trust_level defaults to QUARANTINED (fail-closed)
# ---------------------------------------------------------------------------


def test_trust_level_defaults_to_quarantined() -> None:
    """F-031: any artifact that doesn't explicitly opt into TRUSTED must be QUARANTINED."""
    art = KnowledgeArtifact(
        id=ulid_new(),
        envelope_id=ulid_new(),
        pipeline=TargetPipeline.BOOK_RAG,
        artifact_type=ArtifactType.SUMMARY,
        title="untitled",
        content={"text": "x"},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    assert art.trust_level is TrustLevel.QUARANTINED


# ---------------------------------------------------------------------------
# StrEnum round-trip
# ---------------------------------------------------------------------------


def test_intake_source_strenum_round_trip() -> None:
    # model_dump(mode='json') round-trips back to the same enum value
    src = IntakeSource.FILE_DROP
    now = datetime.now(UTC)
    restored = IntakeEnvelope.model_validate(
        {
            "id": ulid_new(),
            "source_type": src,
            "raw_content": {},
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }
    )
    assert restored.source_type is IntakeSource.FILE_DROP
    assert restored.source_type.value == "file_drop"
    assert isinstance(restored.source_type, str)


# ---------------------------------------------------------------------------
# ProvenanceEntry.contribution bounds
# ---------------------------------------------------------------------------


def test_provenance_contribution_bounded() -> None:
    with pytest.raises(ValidationError):
        ProvenanceEntry(source_id="s", source_type="book", contribution=1.01)
    with pytest.raises(ValidationError):
        ProvenanceEntry(source_id="s", source_type="book", contribution=-0.01)


# ---------------------------------------------------------------------------
# TriageResult confidence bounds (sanity)
# ---------------------------------------------------------------------------


def test_triage_result_confidence_bounded() -> None:
    from vo_rag.core.models import Disposition, InputType

    TriageResult(
        input_type=InputType.BOOK,
        disposition=Disposition.PROCESS,
        priority=Priority.HIGH,
        target_pipeline=TargetPipeline.BOOK_RAG,
        confidence=0.9,
        reasoning="ok",
    )
    with pytest.raises(ValidationError):
        TriageResult(
            input_type=InputType.BOOK,
            disposition=Disposition.PROCESS,
            priority=Priority.HIGH,
            confidence=1.5,
            reasoning="bad",
        )


# ---------------------------------------------------------------------------
# Make sure ComplianceMode has all 3 expected values (citation tier enums)
# ---------------------------------------------------------------------------


def test_compliance_mode_values_present() -> None:
    assert ComplianceMode.INTERNAL.value == "internal"
    assert ComplianceMode.PUBLICATION.value == "publication"
    assert ComplianceMode.PARTNER.value == "partner"

"""Unit tests for source-publication identity and immutable bindings."""

from __future__ import annotations

import hashlib

import pytest

from vo_rag.core.ingest.representation_receipt import canonical_sha256
from vo_rag.core.ingest.source_publication import (
    PublicationBindingError,
    PublicationState,
    PublicationTransitionError,
    SourceCoordinator,
    SourceFence,
    _advisory_key,
    _require_sha256,
    _validate_no_write_disposition,
    _validate_verification,
)


def test_advisory_key_is_stable_and_namespaced_by_corpus() -> None:
    assert _advisory_key("books", "source-1") == _advisory_key("books", "source-1")
    assert _advisory_key("books", "source-1") != _advisory_key("reddit", "source-1")
    assert all(-(2**31) <= part < 2**31 for part in _advisory_key("books", "source-1"))


@pytest.mark.parametrize("value", ["", "A" * 64, "g" * 64, "0" * 63])
def test_source_sha_must_be_lowercase_sha256(value: str) -> None:
    with pytest.raises(ValueError, match="lowercase 64-character"):
        _require_sha256(value)


def test_source_sha_accepts_lowercase_sha256() -> None:
    _require_sha256("a" * 64)


def test_coordinator_requires_a_corpus_namespace() -> None:
    with pytest.raises(ValueError, match="corpus must be non-empty"):
        SourceCoordinator(object(), corpus=" ")


def test_publication_states_include_retained_terminal_state() -> None:
    assert PublicationState.PUBLISHED.value == "published"


def _verification() -> dict[str, object]:
    point_id = "point-1"

    def digest(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    receipt = {
        "verification": "payload_exact_modalities_complete_numeric_transform_verified",
        "writer": {
            "point_id": point_id,
            "payload_sha256": digest("payload"),
            "content_sha256": digest("content"),
            "embedding_provenance_sha256": digest("embedding"),
            "tag_binding_sha256": digest("tags"),
            "vectors": {"dense": digest("dense")},
        },
        "stored": {
            "point_id": point_id,
            "payload_sha256": digest("payload"),
            "content_sha256": digest("content"),
            "embedding_provenance_sha256": digest("embedding"),
            "tag_binding_sha256": digest("tags"),
            "vectors": {"dense": digest("stored dense")},
        },
    }
    receipts = {point_id: receipt}
    verification: dict[str, object] = {
        "expected_point_ids": [point_id],
        "readback_point_ids": [point_id],
        "expected_receipts": receipts,
        "readback_receipts": receipts,
        "expected_manifest_sha256": canonical_sha256(receipts),
        "readback_manifest_sha256": canonical_sha256(receipts),
    }
    return {"verification": verification}


def test_complete_readback_receipts_allow_verified_disposition() -> None:
    _validate_verification(_verification())


def test_verified_disposition_requires_complete_roster_and_modalities() -> None:
    proof = _verification()
    verification = proof["verification"]
    assert isinstance(verification, dict)
    verification["readback_point_ids"] = []
    with pytest.raises(PublicationTransitionError, match="roster"):
        _validate_verification(proof)


@pytest.mark.parametrize(
    "proof",
    [
        {"no_write": {"qdrant_write_attempts": 1, "acknowledgement_unknown": False}},
        {"no_write": {"qdrant_write_attempts": 0, "acknowledgement_unknown": True}},
        {"no_write": {"qdrant_write_attempts": 0, "acknowledgement_unknown": False}},
    ],
)
def test_preserved_no_write_requires_explicit_no_write_proof(proof: dict[str, object]) -> None:
    with pytest.raises(PublicationTransitionError):
        _validate_no_write_disposition(proof)


@pytest.mark.parametrize("same_job", [True, False])
async def test_supersession_rejects_invalid_identity_or_evidence_before_database(
    same_job: bool,
) -> None:
    class UnusedPool:
        """Weak-referenceable pool that deliberately has no database methods."""

    coordinator = SourceCoordinator(UnusedPool(), corpus="books")
    fence = SourceFence(object(), "books", "source", "failed")
    with pytest.raises(PublicationBindingError):
        await coordinator.supersede_unpublished_and_begin(
            fence,
            failed_job_id="failed",
            successor_job_id="failed" if same_job else "successor",
            collection="books_raw",
            source_id="source",
            source_sha256="a" * 64,
            expected_checkpoint_sha256="b" * 64,
            expected_recipe_sha256="c" * 64,
            retention_evidence={},
            retention_evidence_sha256="d" * 64,
            successor_recipe={},
            expected_successor_revision=2,
        )

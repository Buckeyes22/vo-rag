"""Fail-closed contracts for source repair evidence preflight."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from vo_rag.core.ingest.repair_evidence import (
    CensusPoint,
    LiveCensus,
    NamedVectorFingerprint,
    OccurrenceKind,
    PointDisposition,
    PointEvidence,
    PublicationClass,
    RedditArchiveOccurrence,
    RepairBudget,
    RepairClaimKind,
    RepairCorpus,
    RepairEvidence,
    RepresentationFingerprint,
    SourceArtifact,
    SourceArtifactRole,
    SourceFidelityEvidence,
    SourceOccurrence,
    TerminalDisposition,
    TextRoleDecision,
    TypedRecordRole,
    TypedRecordSchemaPolicy,
    VectorFingerprint,
    WriterRecipeEvidence,
    load_reddit_lineage,
    preflight_repair_evidence,
)

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)
GOOD = "a" * 64
VECTOR_VALUES = (0.125, 0.25, 0.5)
VECTOR_SHA256 = hashlib.sha256(
    json.dumps(list(VECTOR_VALUES), separators=(",", ":")).encode()
).hexdigest()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def artifact(path: Path, role: SourceArtifactRole) -> SourceArtifact:
    raw = path.read_bytes()
    return SourceArtifact(role=role, path=path, sha256=digest(raw), size_bytes=len(raw))


@pytest.mark.parametrize("change", ["offset", "kind", "artifact"])
def test_v2_lineage_must_match_its_original_occurrence(tmp_path: Path, change: str) -> None:
    """Independently valid rendered lineage cannot bind a different source span."""
    from vo_rag.core.ingest.reddit_archive import post_to_markdown
    from vo_rag.core.ingest.repair_reddit_lineage import archive_record_digest

    base = books_evidence(tmp_path)
    post = {
        "title": "Fixture",
        "permalink": "/r/example/comments/abc123/thread/",
        "body": "omitted row; source context",
        "comments": [],
    }
    raw = post_to_markdown(post, subreddit="example").encode()
    original_path = tmp_path / "thread.md"
    original_path.write_bytes(raw)
    original = artifact(original_path, SourceArtifactRole.ORIGINAL)
    source_id = "vo-rag-corpus-sha256:" + original.sha256
    start = raw.index(b"omitted row")
    row = {
        "occurrence_id": "source-occurrence-0",
        "source_id": source_id,
        "kind": "reddit_post",
        "subreddit": "example",
        "post_id": "abc123",
        "thread_id": "abc123",
        "ordinal": 0,
        "canonical_permalink": post["permalink"],
        "identity_basis": "archive_tree_path",
        "archive_post_index": 0,
        "archive_record_sha256": archive_record_digest(post),
        "rendered_start_byte": start,
        "rendered_end_byte": start + 11,
    }
    ap = tmp_path / "archive.json"
    ap.write_text(json.dumps([post]))
    archive = artifact(ap, SourceArtifactRole.REDDIT_ARCHIVE)
    mp = tmp_path / "manifest.json"
    mp.write_text(
        json.dumps(
            {
                "schema": "vo-rag.reddit-occurrence-manifest.v2",
                "archive_sha256": archive.sha256,
                "occurrences": [row],
            }
        )
    )
    lineage = load_reddit_lineage(archive, artifact(mp, SourceArtifactRole.REDDIT_LINEAGE_MANIFEST))
    occurrence = base.occurrences[0].model_copy(
        update={
            "source_id": source_id,
            "kind": OccurrenceKind.REDDIT_POST,
            "artifact_sha256": original.sha256,
            "start_byte": start,
            "end_byte": start + 11,
        }
    )
    evidence = base.model_copy(
        update={
            "corpus": RepairCorpus.REDDIT,
            "publication_class": PublicationClass.REDDIT,
            "original": original,
            "source_id": source_id,
            "reddit_lineage": lineage,
            "occurrences": (occurrence,),
        }
    )
    good = preflight_repair_evidence(evidence, checked_at=NOW)
    assert "reddit_archive_rendered_span_mapping_mismatch" not in good.blockers
    if change == "offset":
        occurrence = occurrence.model_copy(
            update={
                "start_byte": start + 1,
                "end_byte": start + 12,
                "span_sha256": digest(raw[start + 1 : start + 12]),
            }
        )
    elif change == "kind":
        occurrence = occurrence.model_copy(update={"kind": OccurrenceKind.ELEMENT})
    else:
        occurrence = occurrence.model_copy(
            update={"artifact_sha256": evidence.accepted_parse.sha256}
        )
    bad = preflight_repair_evidence(
        evidence.model_copy(update={"occurrences": (occurrence,)}), checked_at=NOW
    )
    assert "reddit_archive_rendered_span_mapping_mismatch" in bad.blockers


def point(
    *,
    point_id: str,
    occurrence_id: str,
    disposition: PointDisposition,
    payload: str,
    content: str,
    replaces: tuple[str, ...] = (),
) -> PointEvidence:
    return PointEvidence(
        point_id=point_id,
        source_occurrence_id=occurrence_id,
        disposition=disposition,
        payload_sha256=digest(payload.encode()),
        content_sha256=digest(content.encode()),
        content_bytes=len(content.encode()),
        content=content,
        vectors=(
            NamedVectorFingerprint(
                name="dense",
                fingerprint=VectorFingerprint(
                    sha256=VECTOR_SHA256, dimensions=3, values=VECTOR_VALUES
                ),
            ),
            NamedVectorFingerprint(
                name="bm25",
                fingerprint=VectorFingerprint(
                    sha256=digest(b'{"indices":[11],"values":[1.0]}'),
                    dimensions=12,
                    kind="sparse",
                    values=(1.0,),
                    indices=(11,),
                ),
            ),
        ),
        representations=(
            RepresentationFingerprint(name="content", sha256=digest(content.encode())),
            RepresentationFingerprint(name="embedding_provenance", sha256=GOOD),
            RepresentationFingerprint(name="tag_binding", sha256=GOOD),
        ),
        replaces_point_ids=replaces,
    )


def books_evidence(tmp_path: Path, **overrides: Any) -> RepairEvidence:
    original_path = tmp_path / "original.bin"
    parse_path = tmp_path / "accepted.json"
    original_path.write_bytes(b"omitted row; source context")
    parse_path.write_bytes(b"accepted parse bytes")
    original = artifact(original_path, SourceArtifactRole.ORIGINAL)
    accepted = artifact(parse_path, SourceArtifactRole.ACCEPTED_PARSE).model_copy(
        update={"parent_sha256": original.sha256}
    )
    source_id = f"vo-rag-corpus-sha256:{original.sha256}"
    occurrence = SourceOccurrence(
        occurrence_id="source-occurrence-0",
        source_id=source_id,
        kind=OccurrenceKind.TABLE,
        artifact_sha256=original.sha256,
        start_byte=0,
        end_byte=11,
        span_sha256=digest(b"omitted row"),
        locator="page:1/table:0",
    )
    live = CensusPoint(
        point_id="old-point",
        source_occurrence_id=occurrence.occurrence_id,
        payload_sha256=digest(GOOD.encode()),
        vectors=(
            NamedVectorFingerprint(
                name="dense",
                fingerprint=VectorFingerprint(
                    sha256=VECTOR_SHA256, dimensions=3, values=VECTOR_VALUES
                ),
            ),
            NamedVectorFingerprint(
                name="bm25",
                fingerprint=VectorFingerprint(
                    sha256=digest(b'{"indices":[11],"values":[1.0]}'),
                    dimensions=12,
                    kind="sparse",
                    values=(1.0,),
                    indices=(11,),
                ),
            ),
        ),
        representations=(
            RepresentationFingerprint(name="content", sha256=digest(b"before")),
            RepresentationFingerprint(name="embedding_provenance", sha256=GOOD),
            RepresentationFingerprint(name="tag_binding", sha256=GOOD),
        ),
    )
    census = LiveCensus(
        collection="vo_rag_books_raw_v1",
        captured_at=NOW - timedelta(minutes=1),
        independent_count_captured_at=NOW - timedelta(minutes=1),
        is_complete=True,
        is_truncated=False,
        atomic_snapshot=False,
        source_point_count=1,
        independent_source_count=1,
        required_vector_names=("dense", "bm25"),
        evidence_bytes=4096,
        points=(live,),
    )
    before = point(
        point_id="old-point",
        occurrence_id=occurrence.occurrence_id,
        disposition=PointDisposition.REPLACE,
        payload=GOOD,
        content="before",
    )
    after = point(
        point_id="new-point",
        occurrence_id=occurrence.occurrence_id,
        disposition=PointDisposition.REPLACE,
        payload="new-payload",
        content="omitted row",
        replaces=("old-point",),
    )
    evidence = RepairEvidence(
        corpus=RepairCorpus.BOOKS,
        source_id=source_id,
        source_revision="catalog-revision-17",
        source_revision_sequence=17,
        replacement_revision_sequence=18,
        target_collection="vo_rag_books_raw_v1",
        publication_class=PublicationClass.BOOKS_RAW,
        original=original,
        accepted_parse=accepted,
        occurrences=(occurrence,),
        occurrence_inventory_complete=True,
        census=census,
        before_points=(before,),
        after_points=(after,),
        after_roster_complete=True,
        claim_kind=RepairClaimKind.DETERMINISTIC_SOURCE_FIDELITY,
        source_fidelity=(
            SourceFidelityEvidence(
                occurrence_id=occurrence.occurrence_id,
                artifact_role=SourceArtifactRole.ORIGINAL,
                artifact_sha256=original.sha256,
                start_byte=0,
                end_byte=11,
                omitted_span_sha256=digest(b"omitted row"),
                retained_text_accounted_for=True,
                restored_in_point_ids=("new-point",),
            ),
        ),
        writer_recipe=WriterRecipeEvidence(
            writer_id="normal-books-writer@54f6905",
            target_collection="vo_rag_books_raw_v1",
            recipe_sha256=GOOD,
            writer_code_sha256=GOOD,
            configuration_sha256=GOOD,
            embedding_request_recipe_sha256=GOOD,
            execution_receipt_sha256=GOOD,
            modality_input_sha256=(
                RepresentationFingerprint(name="dense", sha256=GOOD),
                RepresentationFingerprint(name="bm25", sha256=GOOD),
            ),
            actual_writer_observed=True,
            runtime_identity=None,
            runtime_identity_status="unknown",
            unknown_fields=("historical_model_identity",),
            required_modalities=("dense", "bm25"),
            claims_historical_parity=False,
        ),
        book_metadata={},
        book_metadata_sha256=digest(b"{}"),
        source_envelope_metadata={},
        source_envelope_metadata_sha256=digest(b"{}"),
        parser_metadata={},
        parser_metadata_sha256=digest(b"{}"),
        canonical_source_snapshot={"resolved": False},
        canonical_source_snapshot_sha256=digest(b'{"resolved":false}'),
        quality_warnings=(),
        quality_warnings_sha256=digest(b"[]"),
        trust_level="trusted",
        evidence_snapshot_bytes=8192,
    )
    return evidence.model_copy(update=overrides)


def test_byte_bound_deterministic_omission_can_preflight_without_semantic_gold(
    tmp_path: Path,
) -> None:
    evidence = books_evidence(tmp_path)

    result = preflight_repair_evidence(evidence, checked_at=NOW)

    assert result.eligible is True
    assert result.disposition is TerminalDisposition.ELIGIBLE
    assert result.blockers == ()
    assert evidence.semantic_gold is None


def test_json_artifact_round_trips_through_fail_closed_preflight(tmp_path: Path) -> None:
    evidence = books_evidence(tmp_path)

    result = preflight_repair_evidence(json.loads(evidence.model_dump_json()), checked_at=NOW)

    assert result.eligible is True


def test_omitted_source_span_must_be_present_in_the_exact_replacement(tmp_path: Path) -> None:
    evidence = books_evidence(tmp_path)
    output = evidence.after_points[0].model_copy(
        update={
            "content": "different replacement",
            "content_sha256": digest(b"different replacement"),
            "content_bytes": len(b"different replacement"),
        }
    )
    changed = evidence.model_copy(update={"after_points": (output,)})

    result = preflight_repair_evidence(changed, checked_at=NOW)

    assert result.eligible is False
    assert "omitted_source_span_not_present_in_replacement_content" in result.blockers


def test_semantic_claim_without_human_gold_is_blocked(tmp_path: Path) -> None:
    evidence = books_evidence(tmp_path)
    changed = evidence.model_copy(
        update={"claim_kind": RepairClaimKind.SEMANTIC, "source_fidelity": ()}
    )

    result = preflight_repair_evidence(changed, checked_at=NOW)

    assert result.eligible is False
    assert "human_semantic_gold_missing" in result.blockers


def test_replacement_requires_exact_metadata_warning_and_trust_snapshots(
    tmp_path: Path,
) -> None:
    evidence = books_evidence(tmp_path).model_copy(
        update={
            "book_metadata": None,
            "book_metadata_sha256": None,
            "source_envelope_metadata": None,
            "source_envelope_metadata_sha256": None,
            "parser_metadata": None,
            "parser_metadata_sha256": None,
            "canonical_source_snapshot": None,
            "canonical_source_snapshot_sha256": None,
            "quality_warnings": None,
            "quality_warnings_sha256": None,
            "trust_level": None,
        }
    )

    result = preflight_repair_evidence(evidence, checked_at=NOW)

    assert result.eligible is False
    assert "canonical_book_metadata_snapshot_missing" in result.blockers
    assert "source_envelope_metadata_snapshot_missing" in result.blockers
    assert "canonical_source_record_snapshot_missing" in result.blockers
    assert "document_quality_warning_snapshot_missing" in result.blockers
    assert "normal_writer_trust_decision_missing" in result.blockers


def test_multivector_fingerprint_binds_all_colbert_rows() -> None:
    values = ((0.25, 0.5), (0.75, 1.0))
    fingerprint = digest(json.dumps([list(row) for row in values], separators=(",", ":")).encode())

    captured = VectorFingerprint(
        sha256=fingerprint,
        dimensions=2,
        kind="multivector",
        values=values,
    )

    assert captured.values == values


def test_changed_original_bytes_block_preflight(tmp_path: Path) -> None:
    evidence = books_evidence(tmp_path)
    evidence.original.path.write_bytes(b"changed bytes")

    result = preflight_repair_evidence(evidence, checked_at=NOW)

    assert result.eligible is False
    assert "artifact_hash_or_size_mismatch:original" in result.blockers


def test_incomplete_census_and_unmatched_roster_block(tmp_path: Path) -> None:
    evidence = books_evidence(tmp_path)
    census = evidence.census.model_copy(update={"independent_source_count": 2})
    # Model construction is deliberately strict; malformed receipts never reach eligibility.
    with pytest.raises(ValidationError):
        LiveCensus.model_validate(census.model_dump())
    incomplete = evidence.model_copy(update={"after_roster_complete": False})

    result = preflight_repair_evidence(incomplete, checked_at=NOW)

    assert result.disposition is TerminalDisposition.BLOCKED_MISSING_EVIDENCE
    assert "after_roster_incomplete" in result.blockers


def test_legacy_source_requires_explicit_null_active_revision_token(tmp_path: Path) -> None:
    evidence = books_evidence(tmp_path)
    legacy = evidence.model_copy(update={"source_revision_sequence": None})

    result = preflight_repair_evidence(legacy, checked_at=NOW)

    assert result.eligible is True
    missing_token = evidence.model_dump(mode="json")
    del missing_token["source_revision_sequence"]
    invalid = preflight_repair_evidence(missing_token, checked_at=NOW)
    assert invalid.eligible is False
    assert invalid.blockers == ("evidence_schema_invalid_or_unsupported",)


def test_unresolved_before_or_after_unit_is_preserved(tmp_path: Path) -> None:
    evidence = books_evidence(tmp_path)
    unresolved = evidence.after_points[0].model_copy(
        update={"disposition": PointDisposition.UNRESOLVED}
    )
    changed = evidence.model_copy(update={"after_points": (unresolved,)})

    result = preflight_repair_evidence(changed, checked_at=NOW)

    assert result.eligible is False
    assert "point_disposition_unresolved" in result.blockers


def test_wrong_collection_class_and_brain_are_rejected(tmp_path: Path) -> None:
    values = books_evidence(tmp_path).model_dump(mode="json")
    values["target_collection"] = "vo_rag_brain_v1"
    values["census"]["collection"] = "vo_rag_brain_v1"
    values["writer_recipe"]["target_collection"] = "vo_rag_brain_v1"

    result = preflight_repair_evidence(values, checked_at=NOW)

    assert result.eligible is False
    assert result.disposition is TerminalDisposition.BLOCKED_MISSING_EVIDENCE
    assert result.blockers == ("evidence_schema_invalid_or_unsupported",)


def test_default_budget_is_finite_and_larger_budget_must_be_explicit() -> None:
    assert RepairBudget().max_sources == 1
    assert RepairBudget().max_replacement_chunks == 500
    assert RepairBudget().max_replacement_text_bytes == 20 * 1024 * 1024
    assert RepairBudget().max_evidence_bytes == 50 * 1024 * 1024
    assert RepairBudget().max_wall_seconds == 1800
    assert RepairBudget().concurrency == 1
    assert RepairBudget().inference_requests == 0
    with pytest.raises(ValidationError):
        RepairBudget(max_replacement_chunks=501)
    assert RepairBudget(max_replacement_chunks=501, explicitly_increased=True)


def test_remote_inference_endpoint_and_paid_fallback_are_rejected() -> None:
    with pytest.raises(ValidationError):
        RepairBudget(
            inference_requests=1,
            inference_tokens=50,
            local_endpoint="https://api.example.test/v1",
        )
    with pytest.raises(ValidationError):
        RepairBudget(paid_fallback=True)  # type: ignore[arg-type]
    assert (
        RepairBudget(
            inference_requests=1,
            inference_tokens=50,
            local_endpoint="http://127.0.0.1:8080/v1",
        ).inference_tokens
        == 50
    )


def test_full_vector_values_must_match_the_captured_vector_hash() -> None:
    with pytest.raises(ValidationError, match="vector sha256 does not match"):
        VectorFingerprint(sha256=GOOD, dimensions=3, values=VECTOR_VALUES)


def test_inference_use_must_stay_within_explicit_budget(tmp_path: Path) -> None:
    evidence = books_evidence(tmp_path)
    writer = evidence.writer_recipe.model_copy(update={"inference_requests_used": 1})
    changed = evidence.model_copy(update={"writer_recipe": writer})

    result = preflight_repair_evidence(changed, checked_at=NOW)

    assert "inference_request_budget_exceeded" in result.blockers


def test_unknown_graph_schema_owner_blocks_even_with_collection_scope(tmp_path: Path) -> None:
    evidence = books_evidence(tmp_path)
    graph_evidence = evidence.model_copy(
        update={
            "target_collection": "vo_rag_books_graph_edges_v1",
            "publication_class": PublicationClass.BOOKS_GRAPH_EDGES,
            "record_role": TypedRecordRole.GRAPH_EDGE,
            "schema_policy": TypedRecordSchemaPolicy(
                record_role=TypedRecordRole.GRAPH_EDGE,
                schema_owner=None,
                decision=TextRoleDecision.UNKNOWN,
            ),
            "census": evidence.census.model_copy(
                update={"collection": "vo_rag_books_graph_edges_v1"}
            ),
            "writer_recipe": evidence.writer_recipe.model_copy(
                update={"target_collection": "vo_rag_books_graph_edges_v1"}
            ),
        }
    )

    result = preflight_repair_evidence(graph_evidence, checked_at=NOW)

    assert result.eligible is False
    assert "typed_record_text_policy_unknown_or_owner_missing" in result.blockers


def test_reddit_lineage_adapter_derives_ids_from_manifest(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.json"
    archive_path.write_bytes(b"authoritative archive bytes")
    archive = artifact(archive_path, SourceArtifactRole.REDDIT_ARCHIVE)
    entries = [
        {
            "occurrence_id": "post-occurrence",
            "source_id": "reddit-source-1",
            "kind": "reddit_post",
            "subreddit": "example",
            "thread_id": "abc123",
            "post_id": "abc123",
            "comment_id": None,
            "parent_comment_id": None,
            "ordinal": 0,
            "canonical_permalink": "https://reddit.com/r/example/comments/abc123/thread/",
        },
        {
            "occurrence_id": "comment-occurrence",
            "source_id": "reddit-source-1",
            "kind": "reddit_comment",
            "subreddit": "example",
            "thread_id": "abc123",
            "post_id": "abc123",
            "comment_id": "def456",
            "parent_comment_id": None,
            "ordinal": 1,
            "canonical_permalink": "https://reddit.com/r/example/comments/abc123/thread/",
        },
    ]
    manifest_path = tmp_path / "lineage.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "vo-rag.reddit-occurrence-manifest.v1",
                "archive_sha256": archive.sha256,
                "occurrences": entries,
            }
        )
    )
    manifest = artifact(manifest_path, SourceArtifactRole.REDDIT_LINEAGE_MANIFEST)

    lineage = load_reddit_lineage(archive, manifest)

    assert [row.occurrence_id for row in lineage.occurrences] == [
        "post-occurrence",
        "comment-occurrence",
    ]
    assert all(row.manifest_entry_sha256 for row in lineage.occurrences)


def test_guessed_reddit_mapping_not_in_manifest_blocks(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.json"
    archive_path.write_bytes(b"archive")
    manifest_path = tmp_path / "lineage.json"
    manifest_path.write_text("{}")
    archive = artifact(archive_path, SourceArtifactRole.REDDIT_ARCHIVE)
    manifest = artifact(manifest_path, SourceArtifactRole.REDDIT_LINEAGE_MANIFEST)
    guessed = RedditArchiveOccurrence(
        occurrence_id="guessed",
        source_id="reddit-source-1",
        kind=OccurrenceKind.REDDIT_POST,
        archive_sha256=archive.sha256,
        manifest_sha256=manifest.sha256,
        manifest_entry_sha256=GOOD,
        lineage_source="authoritative_archive_manifest",
        subreddit="example",
        thread_id="abc123",
        post_id="abc123",
        ordinal=0,
        canonical_permalink="https://reddit.com/r/example/comments/abc123/thread/",
    )
    # The adapter refuses an unversioned or caller-invented mapping before it
    # can be used by an evidence envelope.
    with pytest.raises(ValueError, match="invalid authoritative Reddit occurrence manifest"):
        load_reddit_lineage(archive, manifest)
    assert guessed.occurrence_id == "guessed"


def test_modalities_are_bound_by_vectors_separately_from_payload_representations(
    tmp_path: Path,
) -> None:
    evidence = books_evidence(tmp_path)
    assert preflight_repair_evidence(evidence, checked_at=NOW).eligible
    point = evidence.after_points[0].model_copy(
        update={"vectors": (evidence.after_points[0].vectors[-1],)}
    )
    missing = evidence.model_copy(update={"after_points": (point,)})
    assert (
        "replacement_missing_writer_required_modality"
        in preflight_repair_evidence(missing, checked_at=NOW).blockers
    )


def test_empty_sparse_capture_preserves_exact_values_and_digest() -> None:
    """An actual empty lexical vector remains an empty sparse vector in repair evidence."""
    raw = b'{"indices":[],"values":[]}'
    fingerprint = VectorFingerprint(
        kind="sparse", dimensions=1, indices=(), values=(), sha256=digest(raw)
    )
    assert fingerprint.values == ()
    assert fingerprint.indices == ()
    assert VectorFingerprint.model_validate_json(fingerprint.model_dump_json()) == fingerprint


@pytest.mark.parametrize("kind", ["dense", "multivector"])
def test_empty_non_sparse_capture_remains_invalid(kind: str) -> None:
    """A real empty sparse receipt never permits missing dense or ColBERT vectors."""
    with pytest.raises(ValidationError):
        VectorFingerprint.model_validate_json(
            json.dumps({"kind": kind, "dimensions": 1, "values": [], "sha256": digest(b"[]")})
        )


@pytest.mark.parametrize("change", ["missing_indices", "unmatched_index", "digest"])
def test_empty_sparse_capture_still_requires_complete_exact_receipt(change: str) -> None:
    """Empty sparse vectors keep index cardinality and actual canonical digest checks."""
    value: dict[str, Any] = {
        "kind": "sparse",
        "dimensions": 1,
        "indices": [],
        "values": [],
        "sha256": digest(b'{"indices":[],"values":[]}'),
    }
    if change == "missing_indices":
        value.pop("indices")
    elif change == "unmatched_index":
        value["indices"] = [0]
    else:
        value["sha256"] = GOOD
    with pytest.raises(ValidationError):
        VectorFingerprint.model_validate_json(json.dumps(value))

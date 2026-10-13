"""Read-only verification of accepted-parse and source byte bindings.

The verifier binds the exact current Garage original and parse bytes to a fresh
intake-envelope snapshot. It accepts an absent legacy original digest or a
matching explicit digest, and blocks an explicit mismatch. It does not edit
either artifact and does not claim that a historical accepted-parse receipt
existed or that parsing was faithful.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vo_rag.core.ingest.repair_evidence import RepairCorpus, load_reddit_lineage
from vo_rag.core.ingest.representation_receipt import canonical_sha256
from vo_rag.core.parsing.contract import ParsedDocument
from vo_rag.core.storage.garage import GarageClient, GarageObjectTooLarge
from vo_rag.pipelines.book_rag.parse_provenance import accepted_parse_binding

_ENVELOPE_FIELDS = frozenset(
    {
        "id",
        "source_type",
        "source_data",
        "raw_content",
        "metadata",
        "status",
        "target_pipeline",
    }
)


class LegacyParseBlocked(RuntimeError):
    """The supplied legacy parse cannot be bound to its source fail-closed."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise LegacyParseBlocked(code)


async def verify_legacy_parse(
    evidence: Any,
    parsed: ParsedDocument,
    settings: Any,
    pool: Any,
) -> dict[str, object]:
    """Verify current source and parse bytes against a fresh envelope snapshot.

    ``metadata.original_sha256`` may be absent for legacy parse objects; when
    present it must match the exact original source digest.

    All artifact lengths and the aggregate read budget are checked before any
    object body or local artifact bytes are read. Only Garage HEAD/GET, a
    read-only envelope SELECT, and local file reads are performed.
    """
    envelope_id = evidence.source_envelope_id
    snapshot = evidence.source_envelope_snapshot
    snapshot_sha = evidence.source_envelope_snapshot_sha256
    if not isinstance(envelope_id, str) or not envelope_id:
        raise LegacyParseBlocked("legacy_source_envelope_id_missing")
    if not isinstance(snapshot, dict):
        raise LegacyParseBlocked("legacy_source_envelope_snapshot_missing")
    if not isinstance(snapshot_sha, str):
        raise LegacyParseBlocked("legacy_source_envelope_snapshot_digest_mismatch")
    _require(set(snapshot) == _ENVELOPE_FIELDS, "legacy_source_envelope_snapshot_incomplete")
    _require(snapshot.get("id") == envelope_id, "legacy_source_envelope_id_mismatch")
    _require(
        canonical_sha256(snapshot) == snapshot_sha,
        "legacy_source_envelope_snapshot_digest_mismatch",
    )

    original = evidence.original
    accepted = evidence.accepted_parse
    original_sha = original.sha256
    parse_sha = accepted.sha256
    source_id = f"vo-rag-corpus-sha256:{original_sha}"
    _require(parsed.document_id == original_sha, "legacy_parse_document_id_mismatch")
    _require(parsed.source_id == source_id == evidence.source_id, "legacy_parse_source_id_mismatch")
    _require(
        parsed.source_id == f"vo-rag-corpus-sha256:{parsed.document_id}",
        "legacy_parse_source_identity_not_canonical",
    )
    parse_metadata = parsed.metadata if isinstance(parsed.metadata, Mapping) else {}
    metadata_original_sha = parse_metadata.get("original_sha256")
    _require(
        metadata_original_sha is None or metadata_original_sha == original_sha,
        "legacy_parse_original_sha256_mismatch",
    )

    raw_content = _mapping(snapshot.get("raw_content"))
    source_data = _mapping(snapshot.get("source_data"))
    if raw_content is None or source_data is None:
        raise LegacyParseBlocked("legacy_source_envelope_content_missing")
    verification = _mapping(raw_content.get("object_verification"))
    historical_verification_missing = verification is None
    archive_lineage = getattr(evidence, "reddit_lineage", None)
    if verification is None:
        _require(
            "object_verification" not in raw_content
            and getattr(evidence, "corpus", None) is RepairCorpus.REDDIT
            and archive_lineage is not None,
            "legacy_source_object_verification_missing",
        )
        # No historical verification fields are manufactured. The complete
        # actual archive and fresh Garage original/cache reads bind this route.
        verification = {}
    _require(
        raw_content.get("file_hash") == original_sha, "legacy_envelope_original_sha256_mismatch"
    )
    _require(
        raw_content.get("size_bytes") == original.size_bytes
        or (historical_verification_missing and "size_bytes" not in raw_content),
        "legacy_envelope_original_size_mismatch",
    )
    if not historical_verification_missing:
        _require(verification.get("status") == "verified", "legacy_source_object_not_verified")
        _require(
            verification.get("sha256") == original_sha, "legacy_object_verification_sha256_mismatch"
        )
        _require(
            verification.get("size_bytes") == original.size_bytes,
            "legacy_object_verification_size_mismatch",
        )
    original_key = raw_content.get("garage_key")
    if not isinstance(original_key, str) or not original_key:
        raise LegacyParseBlocked("legacy_source_garage_key_missing")
    _require(
        source_data.get("garage_key") == original_key
        and (
            historical_verification_missing or verification.get("authoritative_key") == original_key
        ),
        "legacy_source_garage_key_not_bound",
    )
    if "content_type" in verification:
        _require(
            raw_content.get("content_type") == verification.get("content_type"),
            "legacy_source_content_type_binding_mismatch",
        )

    expected_envelope = {key: snapshot[key] for key in _ENVELOPE_FIELDS}
    try:
        fresh_row = await pool.fetchrow(
            "SELECT id, source_type, source_data, raw_content, metadata, status, target_pipeline "
            "FROM intake_envelopes WHERE id = $1",
            envelope_id,
        )
    except Exception as exc:
        raise LegacyParseBlocked("legacy_source_envelope_read_failed") from exc
    if fresh_row is None:
        raise LegacyParseBlocked("legacy_source_envelope_missing")
    fresh_envelope = {key: fresh_row[key] for key in _ENVELOPE_FIELDS}
    _require(
        canonical_sha256(fresh_envelope) == snapshot_sha and fresh_envelope == expected_envelope,
        "legacy_source_envelope_snapshot_changed",
    )

    try:
        original_path = Path(original.path)
        parse_path = Path(accepted.path)
        local_original_size = original_path.stat().st_size
        local_parse_size = parse_path.stat().st_size
    except OSError as exc:
        raise LegacyParseBlocked("legacy_source_artifact_unavailable") from exc
    _require(local_original_size == original.size_bytes, "legacy_local_original_size_mismatch")
    _require(local_parse_size == accepted.size_bytes, "legacy_local_parse_size_mismatch")

    total_limit = evidence.budget.max_evidence_bytes
    already_counted = int(getattr(getattr(evidence, "census", None), "evidence_bytes", 0))
    already_counted += int(getattr(evidence, "evidence_snapshot_bytes", 0))
    _require(
        local_original_size + local_parse_size + already_counted <= total_limit,
        "legacy_evidence_read_budget_exceeded",
    )

    archive_receipt: dict[str, object] | None = None
    if historical_verification_missing:
        if archive_lineage is None:
            raise LegacyParseBlocked("legacy_source_object_verification_missing")
        try:
            artifacts = (archive_lineage.archive, archive_lineage.manifest)
            lineage_bytes = sum(artifact.path.stat().st_size for artifact in artifacts)
            _require(
                all(artifact.path.stat().st_size == artifact.size_bytes for artifact in artifacts)
                and lineage_bytes + local_original_size + local_parse_size + already_counted
                <= total_limit,
                "legacy_evidence_read_budget_exceeded",
            )
            admitted = load_reddit_lineage(
                archive_lineage.archive,
                archive_lineage.manifest,
                max_bytes=total_limit - already_counted - local_original_size - local_parse_size,
            )
            _require(
                admitted.occurrences == archive_lineage.occurrences
                and all(
                    row.identity_basis == "archive_tree_path" and row.source_id == source_id
                    for row in admitted.occurrences
                ),
                "legacy_reddit_actual_archive_binding_mismatch",
            )
        except LegacyParseBlocked:
            raise
        except (OSError, ValueError) as exc:
            raise LegacyParseBlocked("legacy_reddit_actual_archive_binding_invalid") from exc
        already_counted += lineage_bytes
        archive_receipt = {
            "archive_sha256": archive_lineage.archive.sha256,
            "manifest_sha256": archive_lineage.manifest.sha256,
            "complete_rendered_original_reproduction_verified": True,
            "historical_object_verification_recorded": False,
        }

    parse_key = f"parsed/{envelope_id}/parsed.json"
    bucket = settings.garage.buckets.artifacts
    try:
        garage = GarageClient(
            settings.garage.endpoint,
            settings.garage.access_key,
            settings.garage.secret_key,
            settings.garage.region,
        )
        original_head = await garage.object_metadata(bucket, original_key)
        parse_head = await garage.object_metadata(bucket, parse_key)
    except Exception as exc:
        raise LegacyParseBlocked("legacy_garage_head_failed") from exc
    if original_head is None:
        raise LegacyParseBlocked("legacy_garage_original_missing")
    if parse_head is None:
        raise LegacyParseBlocked("legacy_garage_parse_missing")
    _require(original_head.size == original.size_bytes, "legacy_garage_original_size_mismatch")
    _require(parse_head.size == accepted.size_bytes, "legacy_garage_parse_size_mismatch")
    _require(
        original_head.size + parse_head.size + already_counted <= total_limit,
        "legacy_evidence_read_budget_exceeded",
    )
    expected_etag = verification.get("etag")
    _require(
        expected_etag is None or original_head.etag == expected_etag,
        "legacy_garage_original_etag_mismatch",
    )
    expected_content_type = verification.get("content_type")
    _require(
        expected_content_type is None or original_head.content_type == expected_content_type,
        "legacy_garage_original_content_type_mismatch",
    )

    try:
        # verify_object_sha256 performs a fresh HEAD and a conditional GET when
        # Garage supplies an ETag. Remaining budget is passed to each stream so
        # a size change between our aggregate HEAD check and GET still blocks.
        parse_verified = await garage.verify_object_sha256(
            bucket,
            parse_key,
            max_bytes=total_limit - already_counted,
        )
        if parse_verified is None:
            raise LegacyParseBlocked("legacy_garage_parse_missing")
        parse_verified_size = parse_verified.size
        remaining = total_limit - already_counted - parse_verified_size
        _require(remaining >= 0, "legacy_evidence_read_budget_exceeded")
        original_verified = await garage.verify_object_sha256(
            bucket,
            original_key,
            max_bytes=remaining,
        )
        if original_verified is None:
            raise LegacyParseBlocked("legacy_garage_original_missing")
    except GarageObjectTooLarge as exc:
        raise LegacyParseBlocked("legacy_evidence_read_budget_exceeded") from exc
    except LegacyParseBlocked:
        raise
    except Exception as exc:
        raise LegacyParseBlocked("legacy_garage_body_verification_failed") from exc

    _require(parse_verified.size == accepted.size_bytes, "legacy_garage_parse_size_changed")
    _require(
        hmac.compare_digest(parse_verified.sha256, parse_sha),
        "legacy_garage_parse_digest_mismatch",
    )
    _require(original_verified.size == original.size_bytes, "legacy_garage_original_size_changed")
    _require(
        hmac.compare_digest(original_verified.sha256, original_sha),
        "legacy_garage_original_digest_mismatch",
    )
    _require(
        expected_etag is None or original_verified.etag == expected_etag,
        "legacy_garage_original_etag_mismatch",
    )
    _require(
        expected_content_type is None or original_verified.content_type == expected_content_type,
        "legacy_garage_original_content_type_mismatch",
    )

    try:
        with original_path.open("rb") as stream:
            local_original = stream.read(original.size_bytes + 1)
        with parse_path.open("rb") as stream:
            local_parse = stream.read(accepted.size_bytes + 1)
    except OSError as exc:
        raise LegacyParseBlocked("legacy_local_artifact_read_failed") from exc
    _require(len(local_original) == original.size_bytes, "legacy_local_original_size_changed")
    _require(len(local_parse) == accepted.size_bytes, "legacy_local_parse_size_changed")
    _require(
        hmac.compare_digest(_sha256(local_original), original_sha),
        "legacy_local_original_digest_mismatch",
    )
    _require(
        hmac.compare_digest(_sha256(local_parse), parse_sha), "legacy_local_parse_digest_mismatch"
    )
    try:
        local_parsed = ParsedDocument.model_validate_json(local_parse)
    except Exception as exc:
        raise LegacyParseBlocked("legacy_local_parse_invalid") from exc
    _require(local_parsed == parsed, "legacy_parsed_object_differs_from_local_bytes")

    binding = accepted_parse_binding(parsed, original_sha, local_parse)
    return {
        "schema": "legacy-accepted-parse-current-byte-binding.v1",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "source_envelope_id": envelope_id,
        "source_envelope_snapshot_sha256": snapshot_sha,
        "original_sha256": original_verified.sha256,
        "original_size_bytes": original_verified.size,
        "original_garage_key": original_key,
        "original_garage_etag": original_verified.etag,
        "accepted_parse_sha256": parse_verified.sha256,
        "accepted_parse_size_bytes": parse_verified.size,
        "accepted_parse_garage_key": parse_key,
        "accepted_parse_garage_etag": parse_verified.etag,
        "binding": binding,
        "historical_accepted_parse_binding_claimed": False,
        "original_to_parse_fidelity": "unverified",
        "historical_object_verification_recorded": not historical_verification_missing,
        "current_actual_archive_binding": archive_receipt,
        "parse_to_chunk_intervals": "unverified",
    }


__all__ = ["LegacyParseBlocked", "verify_legacy_parse"]

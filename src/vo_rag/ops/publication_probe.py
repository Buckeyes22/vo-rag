"""Run a typed, zero-inference mechanical publication probe for one reviewed source."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import inspect
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client.models import PointStruct

from vo_rag.core.ingest.representation_receipt import canonical_sha256, representation_receipt
from vo_rag.core.storage.source_revision_writer import SourceRevisionWriter, revision_points

__all__ = [
    "MechanicalProbeCheckpoint",
    "MechanicalProbeResult",
    "execute_mechanical_probe",
    "prepare_mechanical_probe",
    "resume_mechanical_probe",
]

# Content digest of the reviewed authored fixture; no credential.
_SOURCE_SHA256 = (
    "fafdaebe9753c3b68d74252a3accca6fe254777fb284b931b2f83b917732220a"  # pragma: allowlist secret
)
_SOURCE_ID = f"vo-rag-corpus-sha256:{_SOURCE_SHA256}"
_ENVELOPE_ID = "01M40TTQKW097NSJ4YF1E03FT7"
_ARTIFACT_ID = "5FD6126C71862D4AC62036D8D7"
_RAW_COLLECTION = "vo_rag_books_raw_v1"
_RAW_REVISION_ONE_POINT_ID = "c7a64bd1-2a2f-797d-0087-97793f3ff5d7"
_PARSE_KEY = f"parsed/{_ENVELOPE_ID}/parsed.json"
_MAX_POINTS_PER_DERIVED_COLLECTION = 100_000
_MAX_BACKUP_BYTES = 50 * 1024 * 1024
_MAX_WALL_SECONDS = 1800
_MAX_QDRANT_UPSERT_ATTEMPTS = 4
_MECHANICAL_RECIPE = "mechanical-publication-probe-v1"
_DERIVED_COLLECTIONS = (
    "vo_rag_books_sterile_v1",
    "vo_rag_books_summary_tree_v1",
    "vo_rag_books_raptor_v1",
    "vo_rag_books_contextualized_v1",
    "vo_rag_books_graph_nodes_v1",
    "vo_rag_books_graph_edges_v1",
    "vo_rag_books_dsi_v1",
)
_ASSOCIATION_FIELDS = (
    "source_id",
    "canonical_source_id",
    "book_id",
    "source_ids",
    "book_ids",
)


class ProbeBlocked(RuntimeError):
    """A reviewed-fixture precondition or exact writer receipt did not match."""


class MechanicalProbeCheckpoint(BaseModel):
    """Persist a fully bound copy plan and its complete prewrite backup."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_schema: Literal["vo-rag.mechanical-publication-probe.v1"]
    status: Literal[
        "prepared",
        "partial_preserved",
        "published_pending_artifact",
        "superseded_preserved",
        "completed",
    ]
    owner: Literal["vo_rag.ops.publication_probe"]
    recovery: str
    source_id: str
    source_sha256: str
    envelope_id: str
    accepted_parse_sha256: str
    artifact_id: str
    artifact_before: dict[str, Any]
    artifact_before_point_ids: list[str]
    publication_pointer_before: dict[str, Any]
    sequence_before: int
    envelope_evidence: dict[str, Any]
    configuration_sha256: str
    active_revision: int
    allocated_revision: int
    current_point_id: str
    prepared_point: dict[str, Any]
    before_receipts: dict[str, dict[str, Any]]
    before_roster_sha256: str
    derived_census: dict[str, dict[str, Any]]
    census_observations: list[dict[str, dict[str, Any]]]
    evidence_sha256: str
    recipe: dict[str, Any]
    recipe_sha256: str
    job_id: str
    source_path: str
    accepted_parse_path: str
    state_path: str
    backup_path: str
    backup_sha256: str
    backup_bytes: int
    writer_point_sha256: str
    semantic_claim: Literal["none"] = "none"
    producer_refresh: Literal["not_performed"] = "not_performed"
    runtime_identity: Literal["unknown_not_used"] = "unknown_not_used"
    model_budget: Literal[0] = 0
    inference_requests: Literal[0] = 0


class MechanicalProbeResult(BaseModel):
    """Report the publication result without semantic or producer claims."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["prepared", "partial_preserved", "published_pending_artifact", "completed"]
    job_id: str
    source_id: str
    source_revision: int
    active_point_id: str | None = None
    old_revisions_retained: list[int] = Field(default_factory=list)
    inference_requests: Literal[0] = 0
    semantic_claim: Literal["none"] = "none"
    producer_refresh: Literal["not_performed"] = "not_performed"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _private_dir(path: Path) -> Path:
    if path.exists() and path.is_symlink():
        raise ProbeBlocked("checkpoint_directory_is_symlink")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    if path.stat().st_mode & 0o777 != 0o700:
        raise ProbeBlocked("checkpoint_directory_is_not_private")
    return path


def _atomic_bytes(path: Path, value: bytes) -> None:
    parent = _private_dir(path.parent)
    if path.exists() and path.is_symlink():
        raise ProbeBlocked("checkpoint_artifact_is_symlink")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        dir_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json_value(value: Any) -> Any:
    """Convert Qdrant values to finite JSON values without changing their meaning."""
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ProbeBlocked("point_contains_unsupported_payload_or_vector_value")


def _point_data(point: Any) -> dict[str, Any]:
    return {
        "id": str(point.id),
        "payload": _json_value(point.payload or {}),
        "vector": _json_value(point.vector),
    }


def _point_from_data(value: Mapping[str, Any]) -> PointStruct:
    point_id = value.get("id")
    payload = value.get("payload")
    vectors = value.get("vector")
    if (
        not isinstance(point_id, str)
        or not isinstance(payload, dict)
        or not isinstance(vectors, dict)
    ):
        raise ProbeBlocked("prepared_point_shape_invalid")
    return PointStruct(id=point_id, payload=dict(payload), vector=dict(vectors))


def _validate_serving_vectors(point: Any) -> None:
    """Require all four finite serving modalities before accepting a copy input."""
    from qdrant_client.models import SparseVector

    vectors = point.vector
    if not isinstance(vectors, dict) or set(vectors) != {"dense", "colbert", "sparse", "bm25"}:
        raise ProbeBlocked("reviewed_raw_point_does_not_have_all_four_named_modalities")

    def finite_row(value: Any) -> bool:
        return (
            isinstance(value, list)
            and bool(value)
            and all(
                isinstance(number, int | float)
                and not isinstance(number, bool)
                and math.isfinite(float(number))
                for number in value
            )
        )

    if not finite_row(vectors["dense"]):
        raise ProbeBlocked("reviewed_dense_vector_is_not_a_finite_numeric_row")
    colbert = vectors["colbert"]
    if not isinstance(colbert, list) or not colbert or any(not finite_row(row) for row in colbert):
        raise ProbeBlocked("reviewed_colbert_vector_is_not_a_finite_multivector")
    for name in ("sparse", "bm25"):
        vector = vectors[name]
        indices: Any
        values: Any
        if isinstance(vector, SparseVector):
            indices, values = vector.indices, vector.values
        elif isinstance(vector, Mapping):
            raw_indices = vector.get("indices")
            raw_values = vector.get("values")
            indices = raw_indices if isinstance(raw_indices, list) else None
            values = raw_values if isinstance(raw_values, list) else None
        else:
            indices, values = None, None
        if (
            not isinstance(indices, list)
            or not isinstance(values, list)
            or len(indices) != len(values)
            or any(type(index) is not int or index < 0 for index in indices)
            or indices != sorted(set(indices))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                for value in values
            )
        ):
            raise ProbeBlocked(f"reviewed_{name}_vector_is_not_a_finite_sparse_vector")


def _point_equivalent_without_revision(before: Any, after: Any) -> bool:
    before_payload = dict(before.payload or {})
    after_payload = dict(after.payload or {})
    before_payload.pop("source_revision", None)
    after_payload.pop("source_revision", None)
    return before_payload == after_payload


def _is_associated(payload: object, source_id: str) -> bool:
    if not isinstance(payload, Mapping):
        return False
    for field in _ASSOCIATION_FIELDS:
        value = payload.get(field)
        if value == source_id or (isinstance(value, list) and source_id in value):
            return True
    dependencies = payload.get("dependency_revisions")
    if isinstance(dependencies, Mapping) and source_id in dependencies:
        return True
    hashed = payload.get("dependency_revision_keys")
    source_key = hashlib.sha256(source_id.encode()).hexdigest()
    if isinstance(hashed, Mapping) and source_key in hashed:
        return True
    # Conservatively cover schema-adjacent ownership keys without matching prose/content.
    identifiers = {source_id, source_key}
    if source_id.startswith("vo-rag-corpus-sha256:"):
        identifiers.add(source_id.removeprefix("vo-rag-corpus-sha256:"))

    def contains_identifier(value: Any) -> bool:
        if isinstance(value, str):
            return value in identifiers
        if isinstance(value, list):
            return any(contains_identifier(item) for item in value)
        if isinstance(value, Mapping):
            return any(
                str(key) in identifiers or contains_identifier(item) for key, item in value.items()
            )
        return False

    for name, value in payload.items():
        lowered = str(name).lower()
        if not any(token in lowered for token in ("source", "book", "dependenc")):
            continue
        if contains_identifier(value):
            return True
    return False


def _census_is_source_clear(census: Mapping[str, Mapping[str, Any]]) -> bool:
    return set(census) == set(_DERIVED_COLLECTIONS) and all(
        value.get("association_count") == 0 for value in census.values()
    )


def _append_census_observation(
    checkpoint: MechanicalProbeCheckpoint,
    census: dict[str, dict[str, Any]],
) -> MechanicalProbeCheckpoint:
    if not _census_is_source_clear(census):
        raise ProbeBlocked("canary_has_existing_derived_source_associations")
    updated = checkpoint.model_copy(
        update={"census_observations": [*checkpoint.census_observations, census]}
    )
    _atomic_bytes(Path(updated.state_path), updated.model_dump_json(indent=2).encode() + b"\n")
    return updated


async def _derived_census(raw: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for collection in _DERIVED_COLLECTIONS:
        count = await raw.count(collection_name=collection, exact=True, timeout=30)
        if count.count > _MAX_POINTS_PER_DERIVED_COLLECTION:
            raise ProbeBlocked("derived_collection_exceeds_bounded_100000_point_census")
        if count.count == 0:
            result[collection] = {
                "count": 0,
                "census_sha256": canonical_sha256([]),
                "association_count": 0,
            }
            continue
        rows: list[Any] = []
        offset: Any = None
        while True:
            page, offset = await raw.scroll(
                collection_name=collection,
                limit=min(128, _MAX_POINTS_PER_DERIVED_COLLECTION + 1 - len(rows)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
                timeout=30,
            )
            rows.extend(page)
            if len(rows) > _MAX_POINTS_PER_DERIVED_COLLECTION:
                raise ProbeBlocked("derived_collection_scan_exceeded_bounded_ceiling")
            if offset is None:
                break
        if len(rows) != count.count or len({str(row.id) for row in rows}) != count.count:
            raise ProbeBlocked("derived_collection_census_incomplete")
        associated = [str(row.id) for row in rows if _is_associated(row.payload, _SOURCE_ID)]
        if associated:
            raise ProbeBlocked("canary_has_existing_derived_source_associations")
        # The exact scan census is hashed without retaining unrelated point payloads.
        result[collection] = {
            "count": count.count,
            "census_sha256": canonical_sha256(sorted(str(row.id) for row in rows)),
            "association_count": 0,
        }
    return result


@asynccontextmanager
async def _resources() -> Any:
    """Open configured local Qdrant, Postgres, and read-only Garage clients."""
    from vo_rag.core.config import get_settings
    from vo_rag.core.db.postgres import close_pool, get_pool
    from vo_rag.core.storage.garage import GarageClient
    from vo_rag.core.storage.qdrant import QdrantStore

    settings = get_settings()
    store = QdrantStore(
        settings.qdrant.url,
        settings.qdrant.api_key or None,
        prefer_grpc=settings.qdrant.prefer_grpc,
    )
    garage = GarageClient(
        settings.garage.endpoint,
        settings.garage.access_key,
        settings.garage.secret_key,
        settings.garage.region,
    )
    try:
        pool = await get_pool(settings.database.url, min_size=1, max_size=2)
        try:
            yield settings, store, pool, garage
        finally:
            await close_pool()
    finally:
        await store._client.close()  # noqa: SLF001


async def _read_source_rows(
    raw: Any, active_revision: int = 2, *, allow_staged_revision_three: bool = False
) -> list[Any]:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    selected = Filter(must=[FieldCondition(key="source_id", match=MatchValue(value=_SOURCE_ID))])
    count = await raw.count(
        collection_name=_RAW_COLLECTION, count_filter=selected, exact=True, timeout=30
    )
    allowed_counts = {active_revision}
    if active_revision == 2 and allow_staged_revision_three:
        allowed_counts.add(3)
    if active_revision not in {2, 3} or count.count not in allowed_counts:
        raise ProbeBlocked("fixture_must_have_exact_complete_physical_raw_roster")
    rows: list[Any] = []
    offset: Any = None
    while True:
        page, offset = await raw.scroll(
            collection_name=_RAW_COLLECTION,
            scroll_filter=selected,
            limit=min(4, max(3, active_revision + 1) - len(rows)),
            offset=offset,
            with_payload=True,
            with_vectors=True,
            timeout=30,
        )
        rows.extend(page)
        if len(rows) > max(3, active_revision):
            raise ProbeBlocked("fixture_raw_source_roster_exceeds_expected_revisions")
        if offset is None:
            break
    if len(rows) != count.count or len({str(row.id) for row in rows}) != count.count:
        raise ProbeBlocked("fixture_raw_source_census_incomplete")
    by_revision: dict[int, Any] = {}
    for row in rows:
        payload = row.payload or {}
        revision = payload.get("source_revision")
        if (
            payload.get("source_id") != _SOURCE_ID
            or type(revision) is not int
            or revision not in {1, 2, 3}
            or revision in by_revision
            or payload.get("generation_id") is not None
            or payload.get("trust_level") != "trusted"
            or payload.get("ingest_state") != "active"
        ):
            raise ProbeBlocked("fixture_raw_point_ownership_or_revision_invalid")
        _validate_serving_vectors(row)
        by_revision[revision] = row
    expected_revisions = (
        {1, 2, 3}
        if active_revision == 3 or (allow_staged_revision_three and 3 in by_revision)
        else {1, 2}
    )
    if set(by_revision) != expected_revisions:
        raise ProbeBlocked("fixture_raw_source_must_retain_every_prior_revision")
    return [by_revision[revision] for revision in sorted(by_revision)]


async def _load_fixture_evidence(
    source_path: Path,
    accepted_parse_path: Path,
    store: Any,
    pool: Any,
    garage: Any,
    settings: Any,
    *,
    expected_active_revision: int = 2,
    allow_staged_revision_three: bool = False,
) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    from vo_rag.core.parsing.contract import ParsedDocument
    from vo_rag.core.storage.source_revision_writer import SourceRevisionWriter
    from vo_rag.pipelines.book_rag.source_identity import resolve_canonical_source

    if source_path.is_symlink() or not source_path.is_file() or source_path.stat().st_mode & 0o077:
        raise ProbeBlocked("authored_source_must_be_private_regular_file")
    if (
        accepted_parse_path.is_symlink()
        or not accepted_parse_path.is_file()
        or accepted_parse_path.stat().st_mode & 0o077
    ):
        raise ProbeBlocked("accepted_parse_must_be_private_regular_file")
    original = source_path.read_bytes()
    source_sha = _sha(original)
    if source_sha != _SOURCE_SHA256:
        raise ProbeBlocked("authored_source_digest_differs_from_reviewed_fixture")
    parse_bytes = accepted_parse_path.read_bytes()
    parse_sha = _sha(parse_bytes)
    try:
        parsed = ParsedDocument.model_validate_json(parse_bytes)
    except Exception as exc:
        raise ProbeBlocked("accepted_parse_contract_invalid") from exc
    if (
        parsed.source_id != _SOURCE_ID
        or (parsed.metadata or {}).get("original_sha256") != source_sha
    ):
        raise ProbeBlocked("accepted_parse_is_not_bound_to_reviewed_source_bytes")

    envelope = await pool.fetchrow(
        "SELECT id, source_type, source_data, raw_content, metadata, status, target_pipeline "
        "FROM intake_envelopes WHERE id = $1",
        _ENVELOPE_ID,
    )
    if (
        envelope is None
        or str(envelope["id"]) != _ENVELOPE_ID
        or envelope["source_type"] != "file_drop"
        or envelope["status"] != "completed"
        # A NULL target uses the normal router default.  The owned active
        # extraction artifact below remains the required proof that this
        # exact completed intake was published by the Books pipeline.
        or envelope["target_pipeline"] not in (None, "book_rag")
        or not isinstance(envelope["source_data"], dict)
        or not isinstance(envelope["raw_content"], dict)
        or not isinstance(envelope["metadata"], dict)
    ):
        raise ProbeBlocked("reviewed_intake_envelope_is_not_completed_book_file_drop")
    raw_content = dict(envelope["raw_content"])
    source_data = dict(envelope["source_data"])
    garage_key = raw_content.get("garage_key")
    verification = raw_content.get("object_verification")
    if (
        raw_content.get("file_hash") != source_sha
        or raw_content.get("filename") != "technical-canary.md"
        or not isinstance(garage_key, str)
        or not garage_key.startswith("verified/sha256/")
        or not isinstance(verification, dict)
        or verification.get("status") != "verified"
        or verification.get("sha256") != source_sha
        or verification.get("size_bytes") != len(original)
        or verification.get("authoritative_key") != garage_key
        or raw_content.get("size_bytes") != len(original)
        or source_data.get("garage_key") != garage_key
    ):
        raise ProbeBlocked("completed_envelope_does_not_bind_reviewed_Garage_source")
    stored_original = await garage.verify_object_sha256(
        "vorag-artifacts", garage_key, max_bytes=len(original)
    )
    if (
        stored_original is None
        or stored_original.sha256 != source_sha
        or stored_original.size != len(original)
    ):
        raise ProbeBlocked("authoritative_Garage_source_bytes_do_not_match_authored_source")
    stored_parse = await garage.verify_object_sha256(
        "vorag-artifacts", _PARSE_KEY, max_bytes=len(parse_bytes)
    )
    if (
        stored_parse is None
        or stored_parse.sha256 != parse_sha
        or stored_parse.size != len(parse_bytes)
    ):
        raise ProbeBlocked("accepted_parse_bytes_do_not_match_authoritative_Garage_object")

    alias_rows = await pool.fetch(
        "SELECT id, file_hash FROM sources WHERE file_hash = $1 OR id = $2",
        source_sha,
        _SOURCE_ID,
    )
    async with pool.acquire() as connection:
        canonical_source = await resolve_canonical_source(_SOURCE_ID, conn=connection)
    source_snapshot = {
        "resolved": canonical_source is not None,
        "source": canonical_source.model_dump(mode="json") if canonical_source else None,
    }
    if source_snapshot == {"resolved": False, "source": None}:
        if alias_rows:
            raise ProbeBlocked("unresolved_normal_source_has_catalog_alias_rows")
    elif (
        not isinstance(source_snapshot["source"], dict)
        or source_snapshot["source"].get("canonical_id") != _SOURCE_ID
        or (
            source_snapshot["source"].get("file_hash") is not None
            and str(source_snapshot["source"]["file_hash"]).lower() != source_sha
        )
        or any(
            row["file_hash"] is not None and str(row["file_hash"]).lower() != source_sha
            for row in alias_rows
        )
    ):
        raise ProbeBlocked("canonical_source_alias_does_not_match_reviewed_digest")

    artifact_rows = await pool.fetch(
        "SELECT id, envelope_id, pipeline, artifact_type, title, tags, priority, status, "
        "license_class, content, source_provenance, qdrant_collection, qdrant_point_ids "
        "FROM knowledge_artifacts "
        "WHERE envelope_id = $1 ORDER BY id",
        _ENVELOPE_ID,
    )
    coordinator = await store._source_coordinator(_RAW_COLLECTION)
    if coordinator is None or not store.source_publication_enabled:
        raise ProbeBlocked("source_publication_reader_and_writer_flags_must_be_enabled")
    active = await coordinator.read_active(_RAW_COLLECTION, _SOURCE_ID)
    if (
        active is None
        or active.revision != expected_active_revision
        or active.source_sha256 != _SOURCE_SHA256
        or active.source_id != _SOURCE_ID
        or active.collection != _RAW_COLLECTION
        or active.corpus != "books"
    ):
        raise ProbeBlocked("reviewed_raw_source_active_pointer_differs_from_checkpoint")
    sequence = await pool.fetchval(
        "SELECT last_revision FROM source_publication_sequences WHERE corpus = 'books' "
        "AND source_id = $1",
        _SOURCE_ID,
    )
    if sequence not in (
        {expected_active_revision, 3}
        if allow_staged_revision_three and expected_active_revision == 2
        else {expected_active_revision}
    ):
        raise ProbeBlocked("reviewed_raw_source_global_sequence_differs_from_active_pointer")

    rows = await _read_source_rows(
        store._raw_client,
        expected_active_revision,
        allow_staged_revision_three=allow_staged_revision_three,
    )
    rev1 = rows[0]
    by_revision = {int(row.payload["source_revision"]): row for row in rows}
    active_point = by_revision[expected_active_revision]
    if str(rev1.id) != _RAW_REVISION_ONE_POINT_ID:
        raise ProbeBlocked("reviewed_raw_revision_one_physical_id_changed")
    if len(artifact_rows) != 1:
        raise ProbeBlocked("reviewed_envelope_must_own_exactly_one_active_extraction_artifact")
    artifact = dict(artifact_rows[0])
    provenance = artifact.get("source_provenance")
    accepted_artifact_point_rosters = [[str(active_point.id)]]
    if expected_active_revision == 3:
        accepted_artifact_point_rosters.append([str(by_revision[2].id)])
    if (
        artifact.get("id") != _ARTIFACT_ID
        or artifact.get("envelope_id") != _ENVELOPE_ID
        or artifact.get("pipeline") != "book_rag"
        or artifact.get("artifact_type") != "extraction"
        or artifact.get("status") != "active"
        or artifact.get("qdrant_collection") != _RAW_COLLECTION
        or artifact.get("qdrant_point_ids") not in accepted_artifact_point_rosters
        or artifact.get("content") != {"source_id": _SOURCE_ID}
        or not isinstance(provenance, list)
        or len(provenance) != 1
        or not isinstance(provenance[0], dict)
        or provenance[0].get("source_id") != _SOURCE_ID
        or provenance[0].get("source_type") != "file_drop"
        or provenance[0].get("contribution") != 1.0
    ):
        raise ProbeBlocked("reviewed_extraction_artifact_does_not_own_active_raw_point")

    derived_census = await _derived_census(store._raw_client)
    writer = SourceRevisionWriter(store._raw_client, coordinator)
    full_before: dict[str, dict[str, Any]] = {}
    for row in rows:
        full_before[str(row.id)] = representation_receipt(row)
    active_before = await writer.census(_RAW_COLLECTION, _SOURCE_ID, expected_active_revision)
    if len(active_before) != 1 or str(active_before[0].id) != str(active_point.id):
        raise ProbeBlocked("active_revision_census_does_not_match_full_source_roster")
    return (
        {
            "source_sha256": source_sha,
            "accepted_parse_sha256": parse_sha,
            "source_path": str(source_path.resolve(strict=True)),
            "accepted_parse_path": str(accepted_parse_path.resolve(strict=True)),
            "envelope": {
                "id": _ENVELOPE_ID,
                "source_type": "file_drop",
                "filename": raw_content["filename"],
                "garage_key_sha256": _sha(garage_key.encode()),
                "metadata_sha256": canonical_sha256(dict(envelope["metadata"])),
                "source_data_sha256": canonical_sha256(source_data),
                "raw_content_sha256": canonical_sha256(raw_content),
                "authoritative_original_key_sha256": _sha(garage_key.encode()),
                "authoritative_parse_key_sha256": _sha(_PARSE_KEY.encode()),
            },
            "envelope_evidence": {
                "envelope_sha256": canonical_sha256(
                    {
                        "id": _ENVELOPE_ID,
                        "source_type": envelope["source_type"],
                        "source_data": source_data,
                        "raw_content": raw_content,
                        "metadata": dict(envelope["metadata"]),
                        "status": envelope["status"],
                        "target_pipeline": envelope["target_pipeline"],
                    }
                ),
                "original_garage_key_sha256": _sha(garage_key.encode()),
                "accepted_parse_key_sha256": _sha(_PARSE_KEY.encode()),
                "source_object_sha256": stored_original.sha256,
                "accepted_parse_object_sha256": stored_parse.sha256,
            },
            "canonical_source_snapshot": source_snapshot,
            "canonical_source_snapshot_sha256": canonical_sha256(source_snapshot),
            "artifact": artifact,
            "artifact_sha256": canonical_sha256(artifact),
            "publication_pointer": {
                "corpus": active.corpus,
                "collection": active.collection,
                "source_id": active.source_id,
                "revision": active.revision,
                "source_sha256": active.source_sha256,
                "job_id": active.job_id,
            },
            "configuration_sha256": canonical_sha256(settings.model_dump(mode="json")),
            "active_revision": expected_active_revision,
            "allocated_revision": expected_active_revision + 1,
            "current_point_id": str(active_point.id),
            "before_receipts": full_before,
            "before_roster_sha256": canonical_sha256(full_before),
            "derived_census": derived_census,
            "source_sequence": int(sequence),
            "physical_revision_ids": {
                str(revision): str(row.id) for revision, row in by_revision.items()
            },
        },
        rows,
        artifact,
    )


def _generations_enabled() -> bool:
    """Return whether generation-scoped ingest is active in this process."""
    return os.environ.get("VORAG_INGEST_GENERATIONS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _recipe(settings: Any, evidence: dict[str, Any]) -> dict[str, Any]:
    """Bind an exact mechanical copy to code, configuration, and source receipts."""
    code_hashes = _current_code_hashes()
    return {
        "recipe": _MECHANICAL_RECIPE,
        "source_id": _SOURCE_ID,
        "source_sha256": _SOURCE_SHA256,
        "envelope_id": _ENVELOPE_ID,
        "accepted_parse_sha256": evidence["accepted_parse_sha256"],
        "artifact_id": _ARTIFACT_ID,
        "active_revision": 2,
        "replacement_revision": 3,
        "before_roster_sha256": evidence["before_roster_sha256"],
        "artifact_sha256": evidence["artifact_sha256"],
        "envelope_sha256": evidence["envelope_evidence"]["envelope_sha256"],
        "envelope_projection_sha256": canonical_sha256(evidence["envelope"]),
        "canonical_source_snapshot_sha256": evidence["canonical_source_snapshot_sha256"],
        "derived_census_sha256": canonical_sha256(evidence["derived_census"]),
        "code_sha256": code_hashes,
        "configuration_sha256": canonical_sha256(settings.model_dump(mode="json")),
        "inference_requests": 0,
        "model_budget": 0,
        "runtime_identity": "unknown_not_used",
        "semantic_claim": "none",
        "producer_refresh": "not_performed",
    }


def _backup_bytes(evidence: dict[str, Any], rows: list[Any]) -> bytes:
    """Serialize the complete active-revision-two restore point and both old revisions."""
    points = [_point_data(row) for row in rows]
    raw = json.dumps(
        {
            "schema": "vo-rag.mechanical-publication-probe-before.v1",
            "source_id": _SOURCE_ID,
            "collection": _RAW_COLLECTION,
            "active_revision": 2,
            "restore_revision": 2,
            "physical_revisions": [1, 2],
            "before_roster_sha256": evidence["before_roster_sha256"],
            "publication_pointer": evidence["publication_pointer"],
            "source_sequence": evidence["source_sequence"],
            "artifact": evidence["artifact"],
            "envelope_evidence": evidence["envelope_evidence"],
            "configuration_sha256": evidence["configuration_sha256"],
            "derived_census": evidence["derived_census"],
            "points": points,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    compressed = gzip.compress(raw, mtime=0)
    if len(raw) > _MAX_BACKUP_BYTES or len(compressed) > _MAX_BACKUP_BYTES:
        raise ProbeBlocked("full_before_backup_exceeds_50MiB_ceiling")
    return compressed


async def prepare_mechanical_probe(
    source_path: Path, accepted_parse_path: Path, checkpoint_dir: Path
) -> MechanicalProbeCheckpoint:
    """Verify the reviewed fixture and persist a complete no-inference copy plan.

    Args:
        source_path: Private authored bytes whose SHA-256 must match the fixture.
        accepted_parse_path: Private accepted parse bound to the same original and Garage object.
        checkpoint_dir: Private operator directory for the durable checkpoint and backup.

    Returns:
        The validated mechanical probe checkpoint, with its complete before backup.

    Raises:
        ProbeBlocked: If any fixture, runtime gate, source, artifact, or census precondition fails.
        OSError: If the durable backup/checkpoint cannot be written and synced.
    """
    if _generations_enabled():
        raise ProbeBlocked("generation_scoped_ingest_must_be_disabled")
    started = time.monotonic()
    async with _resources() as (settings, store, pool, garage):
        if not store.source_visibility_enabled or not store.source_publication_enabled:
            raise ProbeBlocked("source_publication_readers_and_writes_must_be_enabled")
        async with asyncio.timeout(_MAX_WALL_SECONDS):
            evidence, rows, _artifact = await _load_fixture_evidence(
                source_path, accepted_parse_path, store, pool, garage, settings
            )
        if time.monotonic() - started >= _MAX_WALL_SECONDS:
            raise ProbeBlocked("probe_wall_time_limit_exhausted_before_backup")
        recipe = _recipe(settings, evidence)
        recipe_sha = canonical_sha256(recipe)
        evidence_sha = _probe_evidence_sha(evidence)
        current = next(row for row in rows if int(row.payload["source_revision"]) == 2)
        current_point = _point_data(current)
        intended = revision_points(
            [_point_from_data(current_point)],
            corpus="books",
            collection=_RAW_COLLECTION,
            source_id=_SOURCE_ID,
            revision=3,
        )[0]
        writer_point_sha = canonical_sha256(representation_receipt(intended))
        from vo_rag.core.idempotency import deterministic_uuid

        job_id = deterministic_uuid(
            _MECHANICAL_RECIPE,
            _SOURCE_ID,
            _SOURCE_SHA256,
            evidence["accepted_parse_sha256"],
            evidence["before_roster_sha256"],
            recipe_sha,
        )
        directory = _private_dir(checkpoint_dir)
        checkpoint_path = directory / f"{job_id}.json"
        backup_path = directory / f"{job_id}.before.json.gz"
        if (
            checkpoint_path.exists()
            or checkpoint_path.is_symlink()
            or backup_path.exists()
            or backup_path.is_symlink()
        ):
            raise ProbeBlocked("probe_checkpoint_or_backup_already_exists_use_resume")
        compressed = _backup_bytes(evidence, rows)
        _atomic_bytes(backup_path, compressed)
        backup_sha = _sha(compressed)
        checkpoint = MechanicalProbeCheckpoint(
            checkpoint_schema="vo-rag.mechanical-publication-probe.v1",
            status="prepared",
            owner="vo_rag.ops.publication_probe",
            recovery="Resume this exact checkpoint/job only; preserve rev1 and rev2; no inference.",
            source_id=_SOURCE_ID,
            source_sha256=_SOURCE_SHA256,
            envelope_id=_ENVELOPE_ID,
            accepted_parse_sha256=evidence["accepted_parse_sha256"],
            artifact_id=_ARTIFACT_ID,
            artifact_before=evidence["artifact"],
            artifact_before_point_ids=[evidence["current_point_id"]],
            publication_pointer_before=evidence["publication_pointer"],
            sequence_before=2,
            envelope_evidence=evidence["envelope_evidence"],
            configuration_sha256=evidence["configuration_sha256"],
            active_revision=2,
            allocated_revision=3,
            current_point_id=evidence["current_point_id"],
            prepared_point=current_point,
            before_receipts=evidence["before_receipts"],
            before_roster_sha256=evidence["before_roster_sha256"],
            derived_census=evidence["derived_census"],
            census_observations=[evidence["derived_census"]],
            evidence_sha256=evidence_sha,
            recipe=recipe,
            recipe_sha256=recipe_sha,
            job_id=job_id,
            source_path=evidence["source_path"],
            accepted_parse_path=evidence["accepted_parse_path"],
            state_path=str(checkpoint_path),
            backup_path=str(backup_path),
            backup_sha256=backup_sha,
            backup_bytes=len(compressed),
            writer_point_sha256=writer_point_sha,
        )
        _atomic_bytes(checkpoint_path, checkpoint.model_dump_json(indent=2).encode() + b"\n")
        return checkpoint


def _load_checkpoint(path: Path) -> MechanicalProbeCheckpoint:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o777 != 0o600:
        raise ProbeBlocked("probe_checkpoint_must_be_private_regular_file")
    if path.stat().st_uid != os.geteuid():
        raise ProbeBlocked("probe_checkpoint_owner_mismatch")
    try:
        checkpoint = MechanicalProbeCheckpoint.model_validate_json(path.read_bytes())
    except Exception as exc:
        raise ProbeBlocked("probe_checkpoint_schema_invalid") from exc
    if checkpoint.state_path != str(path.resolve(strict=True)):
        raise ProbeBlocked("probe_checkpoint_path_binding_mismatch")
    backup_path = Path(checkpoint.backup_path)
    if (
        backup_path.is_symlink()
        or not backup_path.is_file()
        or backup_path.stat().st_mode & 0o777 != 0o600
        or backup_path.stat().st_uid != os.geteuid()
        or backup_path.parent != path.parent
        or backup_path.name != f"{checkpoint.job_id}.before.json.gz"
    ):
        raise ProbeBlocked("probe_full_backup_must_be_private_regular_file")
    compressed = backup_path.read_bytes()
    if (
        len(compressed) != checkpoint.backup_bytes
        or _sha(compressed) != checkpoint.backup_sha256
        or len(compressed) > _MAX_BACKUP_BYTES
    ):
        raise ProbeBlocked("probe_full_backup_digest_or_size_mismatch")
    try:
        raw = gzip.decompress(compressed)
        if len(raw) > _MAX_BACKUP_BYTES:
            raise ProbeBlocked("probe_full_backup_exceeds_uncompressed_ceiling")
        backup = json.loads(raw)
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProbeBlocked("probe_full_backup_invalid") from exc
    if (
        not isinstance(backup, dict)
        or backup.get("schema") != "vo-rag.mechanical-publication-probe-before.v1"
        or backup.get("source_id") != _SOURCE_ID
        or backup.get("collection") != _RAW_COLLECTION
        or backup.get("active_revision") != 2
        or backup.get("restore_revision") != 2
        or backup.get("physical_revisions") != [1, 2]
        or backup.get("before_roster_sha256") != checkpoint.before_roster_sha256
        or backup.get("publication_pointer") != checkpoint.publication_pointer_before
        or backup.get("source_sequence") != checkpoint.sequence_before
        or backup.get("artifact") != checkpoint.artifact_before
        or backup.get("envelope_evidence") != checkpoint.envelope_evidence
        or backup.get("configuration_sha256") != checkpoint.configuration_sha256
        or backup.get("derived_census") != checkpoint.derived_census
        or not isinstance(backup.get("points"), list)
        or len(backup["points"]) != 2
    ):
        raise ProbeBlocked("probe_full_backup_scope_or_restore_binding_mismatch")
    observed: dict[str, dict[str, Any]] = {}
    for item in backup["points"]:
        if not isinstance(item, dict):
            raise ProbeBlocked("probe_full_backup_point_invalid")
        point = _point_from_data(item)
        point_id = str(point.id)
        observed[point_id] = representation_receipt(point)
    if (
        len(observed) != 2
        or observed != checkpoint.before_receipts
        or canonical_sha256(observed) != checkpoint.before_roster_sha256
    ):
        raise ProbeBlocked("probe_full_backup_point_receipts_do_not_match_checkpoint")
    current = _point_from_data(checkpoint.prepared_point)
    if (
        str(current.id) != checkpoint.current_point_id
        or (current.payload or {}).get("source_id") != _SOURCE_ID
        or (current.payload or {}).get("source_revision") != 2
    ):
        raise ProbeBlocked("prepared_point_does_not_match_reviewed_revision_two")
    _validate_serving_vectors(current)
    if (
        checkpoint.current_point_id not in checkpoint.before_receipts
        or representation_receipt(current)
        != checkpoint.before_receipts[checkpoint.current_point_id]
    ):
        raise ProbeBlocked("prepared_point_does_not_match_full_revision_two_backup_receipt")
    intended = revision_points(
        [current],
        corpus="books",
        collection=_RAW_COLLECTION,
        source_id=_SOURCE_ID,
        revision=3,
    )[0]
    if canonical_sha256(representation_receipt(intended)) != checkpoint.writer_point_sha256:
        raise ProbeBlocked("prepared_writer_point_receipt_mismatch")
    if (
        checkpoint.source_id != _SOURCE_ID
        or checkpoint.source_sha256 != _SOURCE_SHA256
        or checkpoint.envelope_id != _ENVELOPE_ID
        or checkpoint.artifact_id != _ARTIFACT_ID
        or canonical_sha256(checkpoint.artifact_before) != checkpoint.recipe.get("artifact_sha256")
        or checkpoint.publication_pointer_before.get("revision") != 2
        or checkpoint.publication_pointer_before.get("source_sha256") != _SOURCE_SHA256
        or checkpoint.sequence_before != 2
        or checkpoint.envelope_evidence.get("envelope_sha256")
        != checkpoint.recipe.get("envelope_sha256")
        or checkpoint.configuration_sha256 != checkpoint.recipe.get("configuration_sha256")
        or not checkpoint.census_observations
        or any(not _census_is_source_clear(census) for census in checkpoint.census_observations)
        or checkpoint.active_revision != 2
        or checkpoint.allocated_revision != 3
        or checkpoint.artifact_before_point_ids != [checkpoint.current_point_id]
        or canonical_sha256(checkpoint.recipe) != checkpoint.recipe_sha256
        or checkpoint.recipe.get("recipe") != _MECHANICAL_RECIPE
        or checkpoint.recipe.get("inference_requests") != 0
        or checkpoint.recipe.get("model_budget") != 0
        or checkpoint.recipe.get("semantic_claim") != "none"
        or checkpoint.recipe.get("producer_refresh") != "not_performed"
    ):
        raise ProbeBlocked("probe_checkpoint_reviewed_bindings_invalid")
    from vo_rag.core.idempotency import deterministic_uuid

    expected_job_id = deterministic_uuid(
        _MECHANICAL_RECIPE,
        _SOURCE_ID,
        _SOURCE_SHA256,
        checkpoint.accepted_parse_sha256,
        checkpoint.before_roster_sha256,
        checkpoint.recipe_sha256,
    )
    if checkpoint.job_id != expected_job_id:
        raise ProbeBlocked("probe_checkpoint_deterministic_job_binding_mismatch")
    return checkpoint


def _probe_evidence_sha(evidence: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            key: value
            for key, value in evidence.items()
            if key not in {"before_receipts", "derived_census", "artifact"}
        }
    )


def _current_code_hashes() -> dict[str, str]:
    module_path = Path(__file__).resolve(strict=True)
    writer_path = Path(inspect.getfile(SourceRevisionWriter)).resolve(strict=True)
    return {
        "publication_probe": _sha(module_path.read_bytes()),
        "source_revision_writer": _sha(writer_path.read_bytes()),
    }


class _PrewriteGuard:
    """Check the sequence allocation and dependency census before the first Qdrant write."""

    def __init__(self, raw: Any, pool: Any, checkpoint: MechanicalProbeCheckpoint) -> None:
        self.raw = raw
        self.pool = pool
        self.checkpoint = checkpoint
        self.checked = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw, name)

    async def upsert(self, **kwargs: Any) -> Any:
        if not self.checked:
            row = await self.pool.fetchrow(
                "SELECT j.revision, j.state, j.source_id, j.collection, j.checkpoint, "
                "s.last_revision, "
                "p.active_revision FROM source_publication_jobs AS j "
                "JOIN source_publication_sequences AS s "
                "ON s.corpus = j.corpus AND s.source_id = j.source_id "
                "JOIN source_publications AS p "
                "ON p.corpus = j.corpus AND p.collection = j.collection "
                "AND p.source_id = j.source_id "
                "WHERE j.job_id = $1",
                self.checkpoint.job_id,
            )
            if (
                row is None
                or int(row["revision"]) != 3
                or row["state"] != "preparing"
                or row["source_id"] != _SOURCE_ID
                or row["collection"] != _RAW_COLLECTION
                or int(row["last_revision"]) != 3
                or row["active_revision"] != 2
            ):
                raise ProbeBlocked("global_source_sequence_did_not_allocate_exact_revision_three")
            checkpoint = row["checkpoint"]
            attempts = checkpoint.get("write_attempts") if isinstance(checkpoint, Mapping) else None
            points = kwargs.get("points")
            if (
                type(attempts) is not int
                or attempts < 1
                or attempts > _MAX_QDRANT_UPSERT_ATTEMPTS
                or kwargs.get("collection_name") != _RAW_COLLECTION
                or kwargs.get("wait") is not True
                or not isinstance(points, list)
                or len(points) != 1
            ):
                raise ProbeBlocked("mechanical_singleton_write_or_four_attempt_cap_failed")
            point = points[0]
            payload = point.payload or {}
            if (
                payload.get("source_id") != _SOURCE_ID
                or payload.get("source_revision") != 3
                or canonical_sha256(representation_receipt(point))
                != self.checkpoint.writer_point_sha256
            ):
                raise ProbeBlocked("mechanical_write_point_does_not_match_prepared_revision_three")
            census = await _derived_census(self.raw)
            self.checkpoint = _append_census_observation(self.checkpoint, census)
            self.checked = True
        return await self.raw.upsert(**kwargs)


async def _sync_artifact_point_id(
    _pool: Any,
    coordinator: Any,
    checkpoint: MechanicalProbeCheckpoint,
    new_id: str,
) -> str:
    async with coordinator.fence(_SOURCE_ID, job_id=checkpoint.job_id) as fence:
        connection = fence.connection
        owner = await connection.fetchrow(
            "SELECT p.active_revision, p.active_job_id, j.revision, j.state, j.source_sha256 "
            "FROM source_publications AS p JOIN source_publication_jobs AS j "
            "ON j.job_id = p.active_job_id AND j.corpus = p.corpus "
            "WHERE p.corpus = 'books' AND p.collection = $1 AND p.source_id = $2",
            _RAW_COLLECTION,
            _SOURCE_ID,
        )
        if (
            owner is None
            or int(owner["active_revision"]) != 3
            or owner["active_job_id"] != checkpoint.job_id
            or int(owner["revision"]) != 3
            or owner["state"] != "published"
            or str(owner["source_sha256"]) != _SOURCE_SHA256
        ):
            superseded = checkpoint.model_copy(update={"status": "superseded_preserved"})
            _atomic_bytes(
                Path(superseded.state_path), superseded.model_dump_json(indent=2).encode() + b"\n"
            )
            raise ProbeBlocked("published_revision_three_was_superseded_before_artifact_cas")

        artifact_rows = await connection.fetch(
            "SELECT id, envelope_id, pipeline, artifact_type, title, tags, priority, status, "
            "license_class, content, source_provenance, qdrant_collection, qdrant_point_ids "
            "FROM knowledge_artifacts WHERE envelope_id = $1 ORDER BY id",
            _ENVELOPE_ID,
        )
        if len(artifact_rows) != 1:
            raise ProbeBlocked("artifact_cas_requires_exactly_one_owned_artifact")
        artifact = dict(artifact_rows[0])

        def check_artifact(row: Mapping[str, Any], *, point_ids: list[str]) -> None:
            observed = {key: _json_value(value) for key, value in row.items()}
            expected = dict(checkpoint.artifact_before)
            current_ids = observed.pop("qdrant_point_ids", None)
            expected_ids = expected.pop("qdrant_point_ids", None)
            if observed != expected:
                raise ProbeBlocked("artifact_cas_exact_owner_metadata_changed")
            if current_ids != point_ids or expected_ids != checkpoint.artifact_before_point_ids:
                raise ProbeBlocked("artifact_cas_current_physical_point_id_changed")

        current_ids = list(artifact["qdrant_point_ids"])
        if current_ids == [new_id]:
            check_artifact(artifact, point_ids=[new_id])
            return "already_current"
        check_artifact(artifact, point_ids=checkpoint.artifact_before_point_ids)
        changed = await connection.fetchrow(
            "UPDATE knowledge_artifacts SET qdrant_point_ids = $2::text[] "
            "WHERE id = $1 AND envelope_id = $3 AND pipeline = $4 AND artifact_type = $5 "
            "AND status = $6 AND qdrant_collection = $7 AND qdrant_point_ids = $8::text[] "
            "AND title IS NOT DISTINCT FROM $9 AND tags = $10::text[] "
            "AND priority IS NOT DISTINCT FROM $11 "
            "AND license_class IS NOT DISTINCT FROM $12 AND content = $13::jsonb "
            "AND source_provenance = $14::jsonb RETURNING id",
            _ARTIFACT_ID,
            [new_id],
            _ENVELOPE_ID,
            "book_rag",
            "extraction",
            "active",
            _RAW_COLLECTION,
            checkpoint.artifact_before_point_ids,
            artifact["title"],
            artifact["tags"],
            artifact["priority"],
            artifact["license_class"],
            artifact["content"],
            artifact["source_provenance"],
        )
        if changed is None:
            reread_rows = await connection.fetch(
                "SELECT id, envelope_id, pipeline, artifact_type, title, tags, priority, status, "
                "license_class, content, source_provenance, qdrant_collection, qdrant_point_ids "
                "FROM knowledge_artifacts WHERE envelope_id = $1 ORDER BY id",
                _ENVELOPE_ID,
            )
            if len(reread_rows) == 1:
                reread = dict(reread_rows[0])
                if list(reread.get("qdrant_point_ids") or []) == [new_id]:
                    check_artifact(reread, point_ids=[new_id])
                    return "already_current"
            raise ProbeBlocked("artifact_physical_point_id_compare_and_swap_failed")
        return "updated"


async def _publish_checkpoint_inner(
    checkpoint: MechanicalProbeCheckpoint,
) -> MechanicalProbeResult:
    source_path = Path(checkpoint.source_path)
    parse_path = Path(checkpoint.accepted_parse_path)
    if (
        source_path.is_symlink()
        or not source_path.is_file()
        or _sha(source_path.read_bytes()) != _SOURCE_SHA256
        or parse_path.is_symlink()
        or not parse_path.is_file()
        or _sha(parse_path.read_bytes()) != checkpoint.accepted_parse_sha256
    ):
        raise ProbeBlocked("prepared_source_or_parse_bytes_changed")
    started = time.monotonic()
    async with _resources() as (settings, store, pool, garage):
        coordinator = await store._source_coordinator(_RAW_COLLECTION)
        if (
            coordinator is None
            or coordinator.corpus != "books"
            or not store.source_visibility_enabled
            or not store.source_publication_enabled
            or _generations_enabled()
        ):
            raise ProbeBlocked("mechanical_probe_runtime_gates_changed")
        pointer = await coordinator.read_active(_RAW_COLLECTION, _SOURCE_ID)
        active_revision = pointer.revision if pointer is not None else None
        if active_revision not in {2, 3}:
            raise ProbeBlocked("mechanical_probe_active_pointer_is_not_revision_two_or_three")
        job_row = await pool.fetchrow(
            "SELECT revision, state, source_sha256, expected_active_revision, recipe "
            "FROM source_publication_jobs WHERE job_id = $1 AND corpus = 'books' "
            "AND collection = $2 AND source_id = $3",
            checkpoint.job_id,
            _RAW_COLLECTION,
            _SOURCE_ID,
        )
        allow_staged_revision_three = active_revision == 2 and job_row is not None
        if job_row is not None and (
            int(job_row["revision"]) != 3
            or int(job_row["expected_active_revision"]) != 2
            or str(job_row["source_sha256"]) != _SOURCE_SHA256
            or job_row["recipe"] != checkpoint.recipe
            or str(job_row["state"])
            not in {"preparing", "staged", "verified", "partial_preserved", "published"}
        ):
            raise ProbeBlocked("durable_source_job_does_not_match_exact_mechanical_recipe")
        if active_revision == 3 and (job_row is None or str(job_row["state"]) != "published"):
            raise ProbeBlocked("active_revision_three_lacks_its_exact_published_job")
        evidence, rows, _artifact = await _load_fixture_evidence(
            source_path,
            parse_path,
            store,
            pool,
            garage,
            settings,
            expected_active_revision=active_revision,
            allow_staged_revision_three=allow_staged_revision_three,
        )
        if (
            evidence["source_sha256"] != checkpoint.source_sha256
            or evidence["accepted_parse_sha256"] != checkpoint.accepted_parse_sha256
            or not _census_is_source_clear(evidence["derived_census"])
            or evidence["canonical_source_snapshot_sha256"]
            != checkpoint.recipe["canonical_source_snapshot_sha256"]
            or evidence["envelope_evidence"]["envelope_sha256"]
            != checkpoint.recipe["envelope_sha256"]
            or canonical_sha256(evidence["envelope"])
            != checkpoint.recipe["envelope_projection_sha256"]
            or (
                active_revision == 2
                and evidence["artifact_sha256"] != checkpoint.recipe["artifact_sha256"]
            )
        ):
            raise ProbeBlocked("live_fixture_or_dependency_census_differs_from_checkpoint")
        if (
            canonical_sha256(settings.model_dump(mode="json"))
            != checkpoint.recipe["configuration_sha256"]
            or _current_code_hashes() != checkpoint.recipe["code_sha256"]
        ):
            raise ProbeBlocked("code_or_configuration_differs_from_prepared_probe_recipe")
        prior_receipts = {
            point_id: receipt
            for point_id, receipt in evidence["before_receipts"].items()
            if point_id in checkpoint.before_receipts
        }
        if prior_receipts != checkpoint.before_receipts:
            raise ProbeBlocked("retained_revision_one_or_two_receipt_changed")
        checkpoint = _append_census_observation(checkpoint, evidence["derived_census"])

        current_point = _point_from_data(checkpoint.prepared_point)
        intended = revision_points(
            [current_point],
            corpus="books",
            collection=_RAW_COLLECTION,
            source_id=_SOURCE_ID,
            revision=3,
        )[0]
        intended_id = str(intended.id)
        current_rows = {int((row.payload or {}).get("source_revision", 0)): row for row in rows}
        revision_two = current_rows.get(2)
        if revision_two is None or str(revision_two.id) != checkpoint.current_point_id:
            raise ProbeBlocked("reviewed_revision_two_physical_id_changed")
        if active_revision == 2 and 3 in current_rows:
            if str(current_rows[3].id) != intended_id:
                raise ProbeBlocked("partial_revision_three_id_differs_from_prepared_job")
            from vo_rag.core.ingest.representation_receipt import verify_representation

            verify_representation(intended, current_rows[3])
            if not _point_equivalent_without_revision(current_point, current_rows[3]):
                raise ProbeBlocked("partial_revision_three_is_not_the_exact_mechanical_copy")
        if active_revision == 3:
            if 3 not in current_rows:
                raise ProbeBlocked("published_revision_three_is_not_the_exact_mechanical_copy")
            from vo_rag.core.ingest.representation_receipt import verify_representation

            verify_representation(intended, current_rows[3])
            if not _point_equivalent_without_revision(current_point, current_rows[3]):
                raise ProbeBlocked("published_revision_three_payload_changed_from_mechanical_copy")
            if str(current_rows[3].id) != intended_id:
                raise ProbeBlocked("published_revision_three_physical_id_differs_from_recipe")
        recipe = checkpoint.recipe
        guard = _PrewriteGuard(store._raw_client, pool, checkpoint)
        writer = SourceRevisionWriter(guard, coordinator)
        try:
            async with asyncio.timeout(max(1, _MAX_WALL_SECONDS - int(time.monotonic() - started))):
                ids = await writer.replace(
                    _RAW_COLLECTION,
                    _SOURCE_ID,
                    [current_point],
                    job_id=checkpoint.job_id,
                    source_sha256=_SOURCE_SHA256,
                    expected_active_revision=2,
                    recipe=recipe,
                    expected_before={
                        checkpoint.current_point_id: checkpoint.before_receipts[
                            checkpoint.current_point_id
                        ]["payload_sha256"]
                    },
                    expected_before_receipts={
                        checkpoint.current_point_id: checkpoint.before_receipts[
                            checkpoint.current_point_id
                        ]
                    },
                )
        except BaseException:
            durable_checkpoint = _load_checkpoint(Path(checkpoint.state_path))
            row = await pool.fetchrow(
                "SELECT state FROM source_publication_jobs WHERE job_id = $1",
                durable_checkpoint.job_id,
            )
            if row is not None and str(row["state"]) == "partial_preserved":
                updated = durable_checkpoint.model_copy(update={"status": "partial_preserved"})
                _atomic_bytes(
                    Path(checkpoint.state_path),
                    updated.model_dump_json(indent=2).encode() + b"\n",
                )
            raise
        if ids != [intended_id]:
            raise ProbeBlocked("source_writer_returned_unexpected_mechanical_point_ids")
        final_pointer = await coordinator.read_active(_RAW_COLLECTION, _SOURCE_ID)
        if final_pointer is None or final_pointer.revision != 3:
            raise ProbeBlocked("source_writer_did_not_publish_exact_revision_three")
        rev3 = await writer.census(_RAW_COLLECTION, _SOURCE_ID, 3)
        if len(rev3) != 1 or str(rev3[0].id) != intended_id:
            raise ProbeBlocked("published_revision_three_full_census_mismatch")
        from vo_rag.core.ingest.representation_receipt import verify_representation

        verify_representation(intended, rev3[0])
        if not _point_equivalent_without_revision(current_point, rev3[0]):
            raise ProbeBlocked("published_point_is_not_a_full_payload_and_vector_copy")
        durable_checkpoint = _load_checkpoint(Path(checkpoint.state_path))
        updated = durable_checkpoint.model_copy(update={"status": "published_pending_artifact"})
        _atomic_bytes(
            Path(checkpoint.state_path), updated.model_dump_json(indent=2).encode() + b"\n"
        )
        await _sync_artifact_point_id(pool, coordinator, updated, intended_id)
        completed = updated.model_copy(update={"status": "completed"})
        _atomic_bytes(
            Path(checkpoint.state_path), completed.model_dump_json(indent=2).encode() + b"\n"
        )
        return MechanicalProbeResult(
            status="completed",
            job_id=checkpoint.job_id,
            source_id=_SOURCE_ID,
            source_revision=3,
            active_point_id=intended_id,
            old_revisions_retained=[1, 2],
        )


async def _publish_checkpoint(checkpoint: MechanicalProbeCheckpoint) -> MechanicalProbeResult:
    try:
        async with asyncio.timeout(_MAX_WALL_SECONDS):
            return await _publish_checkpoint_inner(checkpoint)
    except TimeoutError as exc:
        raise ProbeBlocked("mechanical_probe_wall_time_limit_exhausted") from exc


async def execute_mechanical_probe(checkpoint_path: Path) -> MechanicalProbeResult:
    """Publish the prepared unchanged copy through SourceRevisionWriter.

    Args:
        checkpoint_path: Exact private checkpoint returned by preparation.

    Returns:
        A typed, zero-inference publication result.

    Raises:
        ProbeBlocked: If the checkpoint or live source changed since preparation.
        Exception: If publication is partial; the durable job and checkpoint retain recovery.
    """
    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint.status != "prepared":
        raise ProbeBlocked("execute_requires_a_prepared_checkpoint_use_resume")
    return await _publish_checkpoint(checkpoint)


async def resume_mechanical_probe(checkpoint_path: Path) -> MechanicalProbeResult:
    """Resume the same prepared job or finish its pending artifact-ID compare-and-swap.

    Args:
        checkpoint_path: Exact private checkpoint whose publication job is durable.

    Returns:
        The same typed result as the original job after full read-back and artifact CAS.

    Raises:
        ProbeBlocked: If checkpoint, job, source, or publication proof no longer matches.
        Exception: If the existing writer job remains partial; its checkpoint stays recoverable.
    """
    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint.status == "completed":
        raise ProbeBlocked("mechanical_probe_checkpoint_is_already_completed")
    if checkpoint.status not in {"prepared", "partial_preserved", "published_pending_artifact"}:
        raise ProbeBlocked("mechanical_probe_checkpoint_status_is_not_resumable")
    return await _publish_checkpoint(checkpoint)

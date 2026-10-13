"""Bounded, restartable source repair operations for an explicit evidence file.

The command is intentionally source-scoped. ``plan`` is local and read-only;
``preflight`` adds a fresh Qdrant/PostgreSQL read; writes use the ordinary raw
chunk/embed/point producer and the source publication writer. This module does
not create vectors itself or make semantic-parity claims.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import json
import os
import tempfile
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import click
import httpx
from qdrant_client.models import SparseVector

from vo_rag.core.ingest.repair_evidence import (
    PointDisposition,
    PublicationClass,
    RepairCorpus,
    RepairEvidence,
    preflight_repair_evidence,
)
from vo_rag.core.ingest.representation_receipt import canonical_sha256, representation_receipt
from vo_rag.core.storage.points import VoPoint
from vo_rag.core.storage.qdrant import build_point_structs
from vo_rag.core.storage.source_revision_writer import SourceRevisionWriter, revision_points
from vo_rag.ops.repair_artifact_references import (
    ArtifactReferenceBlocked,
    check_artifact_references,
    reconcile_artifact_references,
)


async def _check_artifacts(evidence: RepairEvidence, pool: Any) -> None:
    try:
        await check_artifact_references(evidence, pool)
    except ArtifactReferenceBlocked as exc:
        raise RepairBlocked(str(exc)) from exc


async def _reconcile_artifacts(
    evidence: RepairEvidence, store: Any, job_id: str, *, write: bool = True
) -> str:
    after_ids = [
        point.point_id
        for point in evidence.after_points
        if point.disposition is PointDisposition.REPLACE
    ]
    try:
        return await reconcile_artifact_references(
            evidence, store, job_id=job_id, after_ids=after_ids, write=write
        )
    except ArtifactReferenceBlocked as exc:
        raise RepairBlocked(str(exc)) from exc


_STATE_SCHEMA = "vo-rag.corpus-repair-state.v1"
_REPAIRABLE = {
    (RepairCorpus.BOOKS, PublicationClass.BOOKS_RAW, "vo_rag_books_raw_v1"),
    (RepairCorpus.REDDIT, PublicationClass.REDDIT, "vo_rag_reddit_v1"),
}
_UNSUPPORTED_PUBLICATION_CLASSES = {
    PublicationClass.BOOKS_STERILE,
    PublicationClass.BOOKS_CONTEXTUALIZED,
    PublicationClass.BOOKS_GRAPH_NODES,
    PublicationClass.BOOKS_GRAPH_EDGES,
    PublicationClass.BOOKS_DSI,
}


class RepairBlocked(RuntimeError):
    """A safe, source-local reason that the requested operation cannot proceed."""


def _file_digest(path: Path) -> tuple[str, int, bytes]:
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest(), len(raw), raw


def _load_evidence(path: Path) -> tuple[RepairEvidence, str, bytes]:
    digest, size, raw = _file_digest(path)
    if size <= 0:
        raise RepairBlocked("manifest_empty")
    try:
        evidence = RepairEvidence.model_validate_json(raw)
    except Exception as exc:  # pydantic error details can contain manifest values
        raise RepairBlocked("manifest_schema_invalid_or_unsupported") from exc
    result = preflight_repair_evidence(evidence)
    if not result.eligible:
        raise RepairBlocked("evidence_preflight_blocked:" + ",".join(result.blockers))
    if size + evidence.census.evidence_bytes > evidence.budget.max_evidence_bytes:
        raise RepairBlocked("manifest_and_census_exceed_evidence_budget")
    return evidence, digest, raw


def _job_id(evidence: RepairEvidence, manifest_sha256: str) -> str:
    body = {
        "schema": _STATE_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "source_id": evidence.source_id,
        "collection": evidence.target_collection,
        "expected_active_revision": evidence.source_revision_sequence,
        "replacement_revision_sequence": evidence.replacement_revision_sequence,
        "writer_code_sha256": evidence.writer_recipe.writer_code_sha256,
    }
    return (
        "repair-"
        + hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:40]
    )


def _private_state_dir(path: Path) -> Path:
    if path.exists() and path.is_symlink():
        raise RepairBlocked("state_directory_must_not_be_symlink")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    mode = path.stat().st_mode & 0o777
    if mode != 0o700:
        raise RepairBlocked("state_directory_not_private")
    return path


def _atomic_state(path: Path, value: dict[str, Any]) -> None:
    """Replace one private checkpoint and durably sync file and directory."""
    parent = _private_state_dir(path.parent)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
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


def _atomic_private_bytes(path: Path, raw: bytes) -> None:
    """Persist a private immutable artifact and fsync its containing directory."""
    parent = _private_state_dir(path.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
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


def _private_json_value(value: Any) -> Any:
    """Convert Qdrant's point payload/vector values to finite JSON data."""
    if hasattr(value, "model_dump"):
        return _private_json_value(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _private_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_private_json_value(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise RepairBlocked("before_backup_contains_unsupported_value")


def _before_backup_path(state_path: Path) -> Path:
    return state_path.with_suffix(".before.json.gz")


def _verify_before_backup(
    path: Path,
    *,
    sha256: str,
    source_id: str,
    collection: str,
    max_bytes: int,
) -> dict[str, Any]:
    if path.is_symlink() or path.stat().st_mode & 0o777 != 0o600:
        raise RepairBlocked("before_backup_artifact_not_private")
    compressed = path.read_bytes()
    if len(compressed) > max_bytes or hashlib.sha256(compressed).hexdigest() != sha256:
        raise RepairBlocked("before_backup_artifact_digest_or_budget_mismatch")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as stream:
            raw = stream.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise RepairBlocked("before_backup_uncompressed_budget_exceeded")
        value = json.loads(raw)
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as exc:
        raise RepairBlocked("before_backup_artifact_invalid") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema") != "vo-rag.source-repair-before-backup.v1"
        or value.get("source_id") != source_id
        or value.get("collection") != collection
        or not isinstance(value.get("points"), list)
    ):
        raise RepairBlocked("before_backup_artifact_scope_mismatch")
    return value


def _before_backup_matches_evidence(evidence: RepairEvidence, value: dict[str, Any]) -> bool:
    """Check that the private payload/vector backup is the approved full roster."""
    from qdrant_client.models import PointStruct

    expected = {item.point_id: item for item in evidence.census.points}
    rows = value.get("points")
    if not isinstance(rows, list) or len(rows) != len(expected):
        return False
    observed: set[str] = set()
    for item in rows:
        if not isinstance(item, dict):
            return False
        point_id = item.get("point_id")
        payload = item.get("payload")
        vectors = item.get("vectors")
        if (
            not isinstance(point_id, str)
            or point_id not in expected
            or point_id in observed
            or not isinstance(payload, dict)
            or not isinstance(vectors, dict)
        ):
            return False
        observed.add(point_id)
        try:
            receipt = representation_receipt(
                PointStruct(id=point_id, vector=vectors, payload=payload)
            )
        except Exception:
            return False
        captured = expected[point_id]
        if receipt["payload_sha256"] != captured.payload_sha256:
            return False
        if receipt["vectors"] != {
            vector.name: vector.fingerprint.sha256 for vector in captured.vectors
        }:
            return False
        representation_hashes = {
            "content": receipt["content_sha256"],
            "embedding_provenance": receipt["embedding_provenance_sha256"],
            "tag_binding": receipt["tag_binding_sha256"],
        }
        if representation_hashes != {
            representation.name: representation.sha256
            for representation in captured.representations
        }:
            return False
    return observed == set(expected)


async def _capture_before_backup(
    evidence: RepairEvidence,
    store: Any,
    path: Path,
    *,
    max_bytes: int,
    reserved_bytes: int,
) -> dict[str, Any]:
    """Save exact prior payloads and vectors before any source publication work."""
    if not hasattr(store, "_source_coordinator"):
        raise RepairBlocked("before_backup_source_reader_unavailable")
    from vo_rag.core.storage.source_revision_writer import SourceRevisionWriter

    coordinator = await store._source_coordinator(evidence.target_collection)  # noqa: SLF001
    if coordinator is None:
        raise RepairBlocked("before_backup_source_reader_unavailable")
    writer = SourceRevisionWriter(store._raw_client, coordinator)  # noqa: SLF001
    rows = await writer.census(
        evidence.target_collection,
        evidence.source_id,
        evidence.source_revision_sequence,
    )
    if not _fresh_receipts_match(evidence, rows):
        raise RepairBlocked("before_backup_census_differs_from_approved_hashes")
    points: list[dict[str, Any]] = []
    raw_size_upper_bound = reserved_bytes
    for row in sorted(rows, key=lambda item: str(item.id)):
        item = {
            "point_id": str(row.id),
            "payload": _private_json_value(row.payload or {}),
            "vectors": _private_json_value(row.vector),
        }
        try:
            item_bytes = len(
                json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            )
        except (TypeError, ValueError) as exc:
            raise RepairBlocked("before_backup_contains_non_json_value") from exc
        raw_size_upper_bound += item_bytes + 1
        if raw_size_upper_bound > max_bytes:
            raise RepairBlocked("before_backup_exceeds_evidence_budget")
        points.append(item)
    value = {
        "schema": "vo-rag.source-repair-before-backup.v1",
        "source_id": evidence.source_id,
        "collection": evidence.target_collection,
        "active_revision": evidence.source_revision_sequence,
        "point_count": len(points),
        "points": points,
    }
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    compressed = gzip.compress(raw, mtime=0)
    if reserved_bytes + len(raw) > max_bytes or reserved_bytes + len(compressed) > max_bytes:
        raise RepairBlocked("before_backup_exceeds_evidence_budget")
    _atomic_private_bytes(path, compressed)
    digest = hashlib.sha256(compressed).hexdigest()
    return {
        "path": path.name,
        "sha256": digest,
        "compressed_bytes": len(compressed),
        "uncompressed_bytes": len(raw),
        "point_count": len(points),
        "roster_sha256": canonical_sha256([item["point_id"] for item in points]),
    }


def _state_path(state_dir: Path, manifest_sha256: str) -> Path:
    return _private_state_dir(state_dir) / f"{manifest_sha256}.json"


def _state_load(path: Path, *, manifest_sha256: str, job_id: str) -> dict[str, Any] | None:
    if path.is_symlink():
        raise RepairBlocked("state_checkpoint_must_not_be_symlink")
    try:
        if path.stat().st_mode & 0o777 != 0o600:
            raise RepairBlocked("state_checkpoint_not_private")
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RepairBlocked("state_checkpoint_unreadable") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema") != _STATE_SCHEMA
        or value.get("manifest_sha256") != manifest_sha256
        or value.get("job_id") != job_id
    ):
        raise RepairBlocked("state_checkpoint_binding_mismatch")
    return value


def _evidence_summary(evidence: RepairEvidence, digest: str) -> dict[str, Any]:
    replacements = [p for p in evidence.after_points if p.disposition is PointDisposition.REPLACE]
    dispositions = Counter(
        point.disposition.value for point in (*evidence.before_points, *evidence.after_points)
    )
    unsupported: list[str] = []
    if (evidence.corpus, evidence.publication_class, evidence.target_collection) not in _REPAIRABLE:
        unsupported.append("no normal raw source producer registered for explicit target")
    if evidence.publication_class in _UNSUPPORTED_PUBLICATION_CLASSES:
        unsupported.append("typed or derived producer is owner-controlled and not wired here")
    if any(point.disposition is not PointDisposition.REPLACE for point in evidence.after_points):
        unsupported.append("complete active roster requires explicit retained representations")
    if replacements and evidence.budget.inference_requests == 0:
        unsupported.append("replacement requires embeddings; inference budget is zero")
    return {
        "manifest_sha256": digest,
        "corpus": evidence.corpus.value,
        "source_id": evidence.source_id,
        "target_collection": evidence.target_collection,
        "publication_class": evidence.publication_class.value,
        "expected_active_revision": evidence.source_revision_sequence,
        "before_point_count": len(evidence.before_points),
        "after_point_count": len(evidence.after_points),
        "replacement_point_count": len(replacements),
        "disposition_counts": dict(sorted(dispositions.items())),
        "semantic_gold": "provided_human_labels"
        if evidence.semantic_gold
        else "unknown_not_claimed",
        "writer_runtime": evidence.writer_recipe.runtime_identity_status,
        "budget": evidence.budget.model_dump(mode="json"),
        "unsupported_reasons": unsupported,
    }


def plan_manifest(path: Path) -> dict[str, Any]:
    """Validate local immutable evidence without opening any service client."""
    evidence, digest, _ = _load_evidence(path)
    return _evidence_summary(evidence, digest)


def _vector_payload(value: Any) -> Any:
    return value.model_dump(mode="json") if isinstance(value, SparseVector) else value


def _fresh_receipts_match(evidence: RepairEvidence, rows: list[Any]) -> bool:
    expected = {point.point_id: point for point in evidence.census.points}
    observed = {str(row.id): row for row in rows}
    if set(expected) != set(observed):
        return False
    for point_id, row in observed.items():
        receipt = representation_receipt(row)
        captured = expected[point_id]
        if receipt["payload_sha256"] != captured.payload_sha256:
            return False
        actual_vectors = row.vector
        if not isinstance(actual_vectors, dict):
            return False
        vector_names = {item.name for item in captured.vectors}
        if set(actual_vectors) != vector_names:
            return False
        for item in captured.vectors:
            vector = _vector_payload(actual_vectors[item.name])
            if item.fingerprint.kind == "dense":
                values = vector
                if (
                    not isinstance(values, list)
                    or canonical_sha256(values) != item.fingerprint.sha256
                ):
                    return False
            elif item.fingerprint.kind == "sparse":
                if not isinstance(vector, dict):
                    return False
                if (
                    vector.get("indices") != list(item.fingerprint.indices or ())
                    or vector.get("values") != list(item.fingerprint.values)
                    or canonical_sha256(
                        {
                            "indices": list(item.fingerprint.indices or ()),
                            "values": list(cast(tuple[float, ...], item.fingerprint.values)),
                        }
                    )
                    != item.fingerprint.sha256
                ):
                    return False
            elif vector != [
                list(row) for row in cast(tuple[tuple[float, ...], ...], item.fingerprint.values)
            ]:
                return False
        for representation in captured.representations:
            receipt_key = {
                "content": "content_sha256",
                "embedding_provenance": "embedding_provenance_sha256",
                "tag_binding": "tag_binding_sha256",
            }.get(representation.name)
            if receipt_key is None or receipt[receipt_key] != representation.sha256:
                return False
    return True


@asynccontextmanager
async def _resources() -> Any:
    """Build the configured local Qdrant client and DB pool lazily."""
    from vo_rag.core.config import get_settings
    from vo_rag.core.db.postgres import close_pool, get_pool
    from vo_rag.core.storage.qdrant import QdrantStore

    settings = get_settings()
    store = QdrantStore(
        settings.qdrant.url,
        settings.qdrant.api_key or None,
        prefer_grpc=settings.qdrant.prefer_grpc,
    )
    try:
        pool = await get_pool(settings.database.url, min_size=1, max_size=2)
        try:
            yield settings, store, pool
        finally:
            await close_pool()
    finally:
        await store._client.close()  # noqa: SLF001


async def _fresh_preflight(evidence: RepairEvidence, store: Any) -> dict[str, Any]:
    if not hasattr(store, "_source_coordinator"):
        raise RepairBlocked("source_publication_reader_unavailable")
    coordinator = await store._source_coordinator(evidence.target_collection)  # noqa: SLF001
    if coordinator is None or not getattr(store, "source_publication_enabled", False):
        raise RepairBlocked("source_publication_not_enabled_for_target")
    active = await coordinator.read_active(evidence.target_collection, evidence.source_id)
    actual_revision = active.revision if active is not None else None
    if actual_revision != evidence.source_revision_sequence:
        raise RepairBlocked("active_revision_changed_since_evidence_capture")
    expected_revision = evidence.source_revision_sequence
    writer = SourceRevisionWriter(store._raw_client, coordinator)  # noqa: SLF001
    rows = await writer.census(evidence.target_collection, evidence.source_id, expected_revision)
    if not _fresh_receipts_match(evidence, rows):
        raise RepairBlocked("fresh_complete_census_differs_from_approved_hashes")
    return {
        "active_revision": actual_revision,
        "fresh_point_count": len(rows),
        "fresh_census_sha256": canonical_sha256(
            {
                str(row.id): representation_receipt(row)
                for row in sorted(rows, key=lambda r: str(r.id))
            }
        ),
        "atomic_snapshot": False,
    }


async def preflight_manifest(path: Path, state_dir: Path) -> dict[str, Any]:
    evidence, digest, _ = _load_evidence(path)
    summary = _evidence_summary(evidence, digest)
    if summary["unsupported_reasons"]:
        return {**summary, "eligible": False, "disposition": "preserved_unsupported_producer"}
    async with _resources() as (_, store, pool):
        fresh = await _fresh_preflight(evidence, store)
        if any(p.disposition is PointDisposition.REPLACE for p in evidence.after_points):
            await _check_artifacts(evidence, pool)
    state_path = _state_path(state_dir, digest)
    checkpoint = {
        "schema": _STATE_SCHEMA,
        "manifest_sha256": digest,
        "job_id": _job_id(evidence, digest),
        "source_id": evidence.source_id,
        "target_collection": evidence.target_collection,
        "expected_active_revision": evidence.source_revision_sequence,
        "terminal": "eligible_preflight_read_only",
        "owner": "vo_rag.ops.corpus_repair",
        "recovery": (
            "apply repeats active-token and complete-census checks under source writer admission"
        ),
        "fresh_census": fresh,
    }
    _atomic_state(state_path, checkpoint)
    return {**summary, "eligible": True, "disposition": "eligible_read_only", "fresh_census": fresh}


class _BudgetedEmbeddingClient:
    """Bound each real local embed request by finite request and byte/token caps."""

    def __init__(
        self,
        client: Any,
        *,
        max_requests: int,
        max_tokens: int,
        max_wall_seconds: int,
        started_at: float,
        on_usage: Any,
        initial_usage: dict[str, int] | None = None,
        elapsed_before: float = 0.0,
    ) -> None:
        self._client = client
        self._max_requests = max_requests
        self._max_tokens = max_tokens
        self._max_wall_seconds = max_wall_seconds
        self._started_at = started_at
        self._on_usage = on_usage
        initial = initial_usage or {}
        self._requests = initial.get("requests", 0)
        self._input_bytes = initial.get("input_bytes_upper_bound", 0)
        self._input_tokens = initial.get("reported_input_tokens", 0)
        self._elapsed_before = elapsed_before
        self._max_length = getattr(client, "_max_length", 8192)
        transport = getattr(client, "_client", None)
        self._transport_budgeted = isinstance(transport, httpx.AsyncClient)
        if isinstance(transport, httpx.AsyncClient):
            transport.event_hooks["request"].append(self._before_http_request)

    def _check_wall_time(self) -> None:
        if self._elapsed_before + time.monotonic() - self._started_at >= self._max_wall_seconds:
            raise RepairBlocked("repair_wall_time_budget_exhausted")

    def _charge_request(self, texts: list[str]) -> None:
        self._check_wall_time()
        byte_count = sum(len(text.encode("utf-8")) for text in texts)
        if (
            self._requests + 1 > self._max_requests
            or self._input_bytes + byte_count > self._max_tokens
        ):
            raise RepairBlocked("local_inference_budget_exhausted_before_request")
        self._requests += 1
        self._input_bytes += byte_count
        # Persist consumption before every HTTP attempt, including retries.
        self._on_usage(self.usage)

    async def _before_http_request(self, request: httpx.Request) -> None:
        try:
            body = json.loads(request.content)
        except (ValueError, UnicodeDecodeError) as exc:
            raise RepairBlocked("local_embedding_request_malformed") from exc
        texts = body.get("texts") if isinstance(body, dict) else None
        if (
            request.method != "POST"
            or not isinstance(texts, list)
            or not all(isinstance(text, str) for text in texts)
        ):
            raise RepairBlocked("local_embedding_request_malformed")
        self._charge_request(texts)

    async def embed(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        self._check_wall_time()
        if not self._transport_budgeted:
            self._charge_request(texts)
        remaining = (
            self._max_wall_seconds - self._elapsed_before - (time.monotonic() - self._started_at)
        )
        async with asyncio.timeout(remaining):
            response = await self._client.embed(texts, **kwargs)
        self._check_wall_time()
        if not isinstance(response, dict):
            raise RepairBlocked("local_embedding_response_malformed")
        token_counts = response.get("input_token_counts_per_text")
        if isinstance(token_counts, list):
            valid = [v for v in token_counts if type(v) is int and v >= 0]
            self._input_tokens += sum(valid)
        if self._input_tokens > self._max_tokens:
            raise RepairBlocked("local_inference_reported_tokens_over_budget")
        self._on_usage(self.usage)
        return response

    @property
    def usage(self) -> dict[str, int]:
        return {
            "requests": self._requests,
            "input_bytes_upper_bound": self._input_bytes,
            "reported_input_tokens": self._input_tokens,
        }


def _check_producer_scope(evidence: RepairEvidence) -> None:
    if (evidence.corpus, evidence.publication_class, evidence.target_collection) not in _REPAIRABLE:
        raise RepairBlocked("preserved_unsupported_publication_producer")
    if any(point.disposition is not PointDisposition.REPLACE for point in evidence.after_points):
        raise RepairBlocked("preserved_active_roster_requires_unwired_copy_forward")
    if evidence.publication_class is PublicationClass.REDDIT:
        if evidence.reddit_lineage is None:
            raise RepairBlocked("preserved_reddit_lineage_missing")
        # Reddit's archived thread/comment mapping is evidence only; the accepted
        # ParsedDocument remains the production parser cache used by book_rag.
        if not evidence.occurrence_inventory_complete:
            raise RepairBlocked("preserved_reddit_occurrence_inventory_incomplete")


_WRITER_CODE_PATHS = (
    "src/vo_rag/core/ingest/source_publication.py",
    "src/vo_rag/core/chunking/hierarchical.py",
    "src/vo_rag/core/chunking/artifact_filters.py",
    "src/vo_rag/core/chunking/table_linearizer.py",
    "src/vo_rag/core/chunking/section_packer.py",
    "src/vo_rag/pipelines/book_rag/steps/chunk_embed.py",
    "src/vo_rag/pipelines/book_rag/visual_sidecar.py",
    "src/vo_rag/pipelines/book_rag/workflow.py",
    "src/vo_rag/core/storage/qdrant.py",
    "src/vo_rag/core/storage/passage_contract.py",
    "src/vo_rag/core/storage/source_revision_writer.py",
    "src/vo_rag/ops/corpus_repair.py",
    "src/vo_rag/core/ingest/repair_evidence.py",
    "src/vo_rag/core/ingest/repair_reddit_lineage.py",
    "src/vo_rag/ops/repair_artifact_references.py",
    "src/vo_rag/ops/repair_legacy_parse.py",
    "src/vo_rag/ops/repair_page_parse.py",
    "src/vo_rag/core/parsing/parsers/docling.py",
    "src/vo_rag/core/parsing/parsers/docling_mapping.py",
    "src/vo_rag/core/parsing/parsers/pdf_page_rebuild.py",
    "src/vo_rag/core/parsing/parsers/pdf_formula.py",
    "src/vo_rag/core/storage/figures.py",
)


def _writer_code_sha256() -> str:
    root = Path(__file__).resolve().parents[3]
    return canonical_sha256(
        {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in _WRITER_CODE_PATHS
        }
    )


def _configuration_sha256(
    evidence: RepairEvidence, settings: Any, request_recipe_sha256: str
) -> str:
    return canonical_sha256(
        {
            "chunking": settings.chunking.model_dump(mode="json"),
            "embedding_request_recipe_sha256": request_recipe_sha256,
            "source_envelope_id": evidence.source_envelope_id,
            "source_envelope_snapshot_sha256": evidence.source_envelope_snapshot_sha256,
            "owned_artifact_snapshots_sha256": evidence.owned_artifact_snapshots_sha256,
            "book_metadata_sha256": evidence.book_metadata_sha256,
            "source_envelope_metadata_sha256": evidence.source_envelope_metadata_sha256,
            "canonical_source_snapshot_sha256": evidence.canonical_source_snapshot_sha256,
            "quality_warnings_sha256": evidence.quality_warnings_sha256,
            "trust_level": evidence.trust_level,
            "collection": evidence.target_collection,
            "generation_id": None,
            "ingest_state": "active",
            "embed_batch": 1,
        }
    )


def _recipe_sha256(evidence: RepairEvidence, code_sha256: str, config_sha256: str) -> str:
    return canonical_sha256(
        {
            "writer_id": evidence.writer_recipe.writer_id,
            "writer_code_sha256": code_sha256,
            "configuration_sha256": config_sha256,
            "target_collection": evidence.target_collection,
            "required_modalities": sorted(evidence.writer_recipe.required_modalities),
        }
    )


async def _canonical_source_snapshot(pool: Any, source_id: str) -> dict[str, object]:
    from vo_rag.pipelines.book_rag.source_identity import resolve_canonical_source

    async with pool.acquire() as connection:
        source = await resolve_canonical_source(source_id, conn=connection)
    return {
        "resolved": source is not None,
        "source": source.model_dump(mode="json") if source is not None else None,
    }


def _book_metadata_from_snapshot(
    parsed_title: str,
    envelope_metadata: dict[str, object],
    source_snapshot: dict[str, object],
    source_id: str,
) -> dict[str, object]:
    title = str(envelope_metadata.get("title") or parsed_title or "").strip()
    author = str(envelope_metadata.get("author") or "").strip()
    raw_tags = envelope_metadata.get("tags") or envelope_metadata.get("domain_tags") or []
    tags = [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else []
    source_book = f"{title} ({author})" if title and author else (title or source_id)
    source = source_snapshot.get("source")
    if isinstance(source, dict):
        title = str(source.get("title") or "").strip()
        author = str(source.get("author") or "").strip()
        canonical_id = str(source.get("canonical_id") or "")
        source_book = f"{title} ({author})" if title and author else (title or canonical_id)
        return {
            "source_book": source_book,
            "title": title,
            "author": author or None,
            "domain_tags": tags,
            "license_class": source.get("license_class"),
            "canonical_source_id": canonical_id,
            "project_id": source.get("project_id"),
        }
    return {
        "source_book": source_book,
        "title": title,
        "author": author or None,
        "domain_tags": tags,
        "license_class": str(envelope_metadata.get("license_class") or "unlicensed"),
        "canonical_source_id": source_id,
    }


def _vo_points_path(state_path: Path) -> Path:
    return state_path.with_suffix(".points.json")


def _save_vo_points(path: Path, points: list[VoPoint]) -> str:
    raw = json.dumps(
        [point.model_dump(mode="json") for point in points],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    parent = _private_state_dir(path.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
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
    return hashlib.sha256(raw).hexdigest()


def _load_vo_points(path: Path, expected_sha256: str, expected_count: int) -> list[VoPoint]:
    if path.is_symlink() or path.stat().st_mode & 0o777 != 0o600:
        raise RepairBlocked("durable_writer_points_artifact_not_private")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RepairBlocked("durable_writer_points_artifact_digest_mismatch")
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("not a list")
        points = [VoPoint.model_validate(item) for item in parsed]
    except Exception as exc:
        raise RepairBlocked("durable_writer_points_artifact_invalid") from exc
    if len(points) != expected_count:
        raise RepairBlocked("durable_writer_points_artifact_cardinality_mismatch")
    return points


def _normal_point_structs(points: list[VoPoint], evidence: RepairEvidence) -> list[Any]:
    if evidence.replacement_revision_sequence is None:
        raise RepairBlocked("approved_replacement_revision_sequence_missing")
    return revision_points(
        build_point_structs(points),
        corpus=evidence.corpus.value,
        collection=evidence.target_collection,
        source_id=evidence.source_id,
        revision=evidence.replacement_revision_sequence,
    )


def _bind_occurrences_to_chunks(evidence: RepairEvidence, points: list[VoPoint]) -> None:
    """Bind a replacement point only when its bytes contain one approved occurrence."""
    artifacts = {
        evidence.original.sha256: evidence.original.path.read_bytes(),
        evidence.accepted_parse.sha256: evidence.accepted_parse.path.read_bytes(),
    }
    occurrence_bytes: dict[str, bytes] = {}
    for occurrence in evidence.occurrences:
        artifact = artifacts.get(occurrence.artifact_sha256)
        if artifact is None or occurrence.end_byte > len(artifact):
            raise RepairBlocked("approved_source_occurrence_artifact_unavailable")
        span = artifact[occurrence.start_byte : occurrence.end_byte]
        if hashlib.sha256(span).hexdigest() != occurrence.span_sha256:
            raise RepairBlocked("approved_source_occurrence_span_hash_mismatch")
        occurrence_bytes[occurrence.occurrence_id] = span

    for index, (point, expected) in enumerate(
        zip(
            points,
            (p for p in evidence.after_points if p.disposition is PointDisposition.REPLACE),
            strict=True,
        )
    ):
        chunk_span = occurrence_bytes.get(expected.source_occurrence_id)
        content = str(point.payload.get("content") or "").encode("utf-8")
        if not chunk_span or content.count(chunk_span) != 1:
            raise RepairBlocked(f"source_occurrence_bytes_not_unique_in_chunk_at_{index}")
        payload = dict(point.payload)
        payload["source_occurrence_id"] = expected.source_occurrence_id
        point.payload = payload


def _approved_before_receipts(evidence: RepairEvidence) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for point in evidence.census.points:
        representations = {item.name: item.sha256 for item in point.representations}
        required = {"content", "embedding_provenance", "tag_binding"}
        if not required <= set(representations):
            raise RepairBlocked("approved_before_representation_receipts_incomplete")
        receipts[point.point_id] = {
            "point_id": point.point_id,
            "payload_sha256": point.payload_sha256,
            "vectors": {item.name: item.fingerprint.sha256 for item in point.vectors},
            "content_sha256": representations["content"],
            "embedding_provenance_sha256": representations["embedding_provenance"],
            "tag_binding_sha256": representations["tag_binding"],
        }
    return receipts


def _validate_output_roster(evidence: RepairEvidence, points: list[VoPoint]) -> None:
    after = [
        point for point in evidence.after_points if point.disposition is PointDisposition.REPLACE
    ]
    if len(points) != len(after):
        raise RepairBlocked("ordered_after_roster_cardinality_mismatch")
    staged = _normal_point_structs(points, evidence)
    for index, (point, record, expected) in enumerate(zip(points, staged, after, strict=True)):
        content = str(point.payload.get("content") or "").encode("utf-8")
        if (
            hashlib.sha256(content).hexdigest() != expected.content_sha256
            or len(content) != expected.content_bytes
        ):
            raise RepairBlocked(f"ordered_after_content_mismatch_at_{index}")
        if str(record.id) != expected.point_id:
            raise RepairBlocked(f"stable_after_point_id_mismatch_at_{index}")
        if (record.payload or {}).get("source_occurrence_id") != expected.source_occurrence_id:
            raise RepairBlocked(f"ordered_source_occurrence_mismatch_at_{index}")
        receipt = representation_receipt(record)
        if receipt["payload_sha256"] != expected.payload_sha256:
            raise RepairBlocked(f"ordered_after_payload_receipt_mismatch_at_{index}")
        actual_vectors = point.model_dump(mode="python")
        vector_values: dict[str, Any] = {
            "dense": actual_vectors["dense"],
            "sparse": actual_vectors["sparse"],
            "bm25": actual_vectors["bm25"],
            "colbert": actual_vectors["colbert"],
        }
        expected_vectors = {item.name: item.fingerprint for item in expected.vectors}
        actual_names = set(record.vector) if isinstance(record.vector, dict) else set()
        if actual_names != set(expected_vectors):
            raise RepairBlocked(f"ordered_after_modality_set_mismatch_at_{index}")
        for name, fingerprint in expected_vectors.items():
            value = vector_values[name]
            if fingerprint.kind == "multivector":
                rows = cast(list[list[float]], value)
                actual: Any = tuple(tuple(float(v) for v in row) for row in rows)
            elif fingerprint.kind == "sparse":
                sparse_values = cast(dict[int, float], value)
                items = sorted(sparse_values.items())
                actual_indices = tuple(int(key) for key, _ in items)
                actual_values = tuple(float(item) for _, item in items)
                if actual_indices != fingerprint.indices:
                    raise RepairBlocked(f"ordered_after_sparse_indices_mismatch_at_{index}")
                actual = actual_values
            else:
                dense_values = cast(list[float], value)
                actual = tuple(float(item) for item in dense_values)
            if actual != fingerprint.values:
                raise RepairBlocked(f"ordered_after_vector_values_mismatch_at_{index}")
        for item in expected.representations:
            key = {
                "content": "content_sha256",
                "embedding_provenance": "embedding_provenance_sha256",
                "tag_binding": "tag_binding_sha256",
            }.get(item.name)
            if key is None or receipt[key] != item.sha256:
                raise RepairBlocked(f"ordered_after_representation_mismatch_at_{index}")


async def apply_manifest(path: Path, state_dir: Path, *, resume: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    evidence, manifest_sha256, manifest_raw = _load_evidence(path)
    state_path = _state_path(state_dir, manifest_sha256)
    job_id = _job_id(evidence, manifest_sha256)
    state = _state_load(state_path, manifest_sha256=manifest_sha256, job_id=job_id)
    if resume and state is None:
        raise RepairBlocked("resume_checkpoint_missing")
    if (
        resume
        and state is not None
        and state.get("terminal")
        not in {
            "applying",
            "failed_retryable_preserved",
            "writer_published_needs_verify",
            "published_verified",
            "preserved_no_write",
            "blocked_budget",
        }
    ):
        raise RepairBlocked("resume_checkpoint_has_no_apply_intent")
    if state and (
        state.get("terminal") in {"published_verified", "writer_published_needs_verify"}
        or state.get("writer_upsert_intent") is True
    ):
        async with _resources() as (_, store, _pool):
            already_published = state.get("terminal") in {
                "published_verified",
                "writer_published_needs_verify",
            }
            if not already_published:
                coordinator = await store._source_coordinator(evidence.target_collection)
                pointer = (
                    await coordinator.read_active(evidence.target_collection, evidence.source_id)
                    if coordinator is not None
                    else None
                )
                already_published = (
                    pointer is not None
                    and pointer.job_id == job_id
                    and pointer.revision == evidence.replacement_revision_sequence
                )
            if already_published:
                saved_sha = state.get("writer_points_sha256")
                saved_count = state.get("writer_points_count")
                if not isinstance(saved_sha, str) or type(saved_count) is not int:
                    raise RepairBlocked("published_recovery_prepared_points_checkpoint_missing")
                prepared = _load_vo_points(_vo_points_path(state_path), saved_sha, saved_count)
                _validate_output_roster(evidence, prepared)
                expected_ids = {
                    point.point_id
                    for point in evidence.after_points
                    if point.disposition is PointDisposition.REPLACE
                }
                roster_sha = canonical_sha256(sorted(expected_ids))
                if state.get("writer_roster_sha256") != roster_sha:
                    raise RepairBlocked("published_recovery_prepared_roster_checkpoint_mismatch")
                verified = await _verify_live(
                    evidence,
                    store,
                    expected_job_id=job_id,
                    expected_ids=expected_ids,
                    expected_roster_sha256=roster_sha,
                )
                state.update(
                    {"terminal": "writer_published_needs_verify", "verification": verified}
                )
                _atomic_state(state_path, state)
                verified["artifact_references"] = await _reconcile_artifacts(
                    evidence, store, job_id
                )
                state.update({"terminal": "published_verified", "verification": verified})
                _atomic_state(state_path, state)
                return {
                    "job_id": job_id,
                    "state": "published_verified",
                    "idempotent": True,
                    "inference": state.get("inference", {}),
                    "verification": verified,
                }
    if state and state.get("terminal") in {"preserved_no_write", "blocked_budget"}:
        return {"job_id": job_id, "state": state["terminal"], "idempotent": True}

    replacements = [p for p in evidence.after_points if p.disposition is PointDisposition.REPLACE]
    if not replacements:
        no_write_checkpoint = {
            "schema": _STATE_SCHEMA,
            "manifest_sha256": manifest_sha256,
            "job_id": job_id,
            "source_id": evidence.source_id,
            "target_collection": evidence.target_collection,
            "terminal": "preserved_no_write",
            "owner": "vo_rag.ops.corpus_repair",
            "recovery": "no source write required; retain current active revision",
            "metrics": {"replacement_count": 0},
        }
        _atomic_state(state_path, no_write_checkpoint)
        return {"job_id": job_id, "state": "preserved_no_write", "idempotent": False}

    _check_producer_scope(evidence)

    budget = evidence.budget
    points_path = _vo_points_path(state_path)
    saved_points_sha = state.get("writer_points_sha256") if state else None
    points_count = state.get("writer_points_count") if state else None
    if not isinstance(saved_points_sha, str) and budget.inference_requests == 0:
        budget_checkpoint = {
            "schema": _STATE_SCHEMA,
            "manifest_sha256": manifest_sha256,
            "job_id": job_id,
            "source_id": evidence.source_id,
            "target_collection": evidence.target_collection,
            "terminal": "blocked_budget",
            "owner": "vo_rag.ops.corpus_repair",
            "recovery": "supply an immutable manifest with finite local inference allowance",
            "metrics": {"replacement_count": len(replacements), "inference_requests_used": 0},
        }
        _atomic_state(state_path, budget_checkpoint)
        return {"job_id": job_id, "state": "blocked_budget", "idempotent": False}
    book_metadata_snapshot = getattr(evidence, "book_metadata", None)
    book_metadata_sha256 = getattr(evidence, "book_metadata_sha256", None)
    envelope_metadata_snapshot = getattr(evidence, "source_envelope_metadata", None)
    parser_metadata_snapshot = getattr(evidence, "parser_metadata", None)
    canonical_source_snapshot = getattr(evidence, "canonical_source_snapshot", None)
    canonical_source_snapshot_sha256 = getattr(evidence, "canonical_source_snapshot_sha256", None)
    quality_warnings_snapshot = getattr(evidence, "quality_warnings", None)
    quality_warnings_sha256 = getattr(evidence, "quality_warnings_sha256", None)
    trust_level = getattr(evidence, "trust_level", None)
    request_recipe_sha256_evidence = getattr(
        evidence.writer_recipe, "embedding_request_recipe_sha256", None
    )
    if book_metadata_snapshot is None or book_metadata_sha256 is None:
        raise RepairBlocked("canonical_book_metadata_snapshot_missing_before_admission")
    if (
        envelope_metadata_snapshot is None
        or getattr(evidence, "source_envelope_metadata_sha256", None) is None
        or parser_metadata_snapshot is None
        or getattr(evidence, "parser_metadata_sha256", None) is None
        or canonical_source_snapshot is None
        or canonical_source_snapshot_sha256 is None
    ):
        raise RepairBlocked("canonical_source_or_envelope_snapshot_missing_before_admission")
    if quality_warnings_snapshot is None or quality_warnings_sha256 is None:
        raise RepairBlocked("document_quality_warning_snapshot_missing_before_admission")
    if trust_level is None:
        raise RepairBlocked("normal_writer_trust_decision_missing_before_admission")
    if evidence.writer_recipe.claims_historical_parity and (
        evidence.writer_recipe.runtime_identity_status != "known"
    ):
        raise RepairBlocked("historical_parity_claim_requires_known_runtime_identity")
    if request_recipe_sha256_evidence is None:
        raise RepairBlocked("embedding_request_recipe_fingerprint_missing_before_admission")
    required_modalities = set(evidence.writer_recipe.required_modalities)
    if not required_modalities <= {"dense", "sparse", "bm25", "colbert"}:
        raise RepairBlocked("writer_recipe_requires_unsupported_modality_before_admission")

    # The accepted parse is immutable input and is validated before creating a
    # publication job, so parse errors cannot strand a PREPARING job.
    from vo_rag.core.parsing.contract import ParsedDocument

    try:
        parsed = ParsedDocument.model_validate_json(evidence.accepted_parse.path.read_bytes())
    except Exception as exc:
        raise RepairBlocked("accepted_parse_invalid_before_admission") from exc
    if parsed.source_id != evidence.source_id:
        raise RepairBlocked("accepted_parse_source_id_mismatch")
    if parser_metadata_snapshot != dict(parsed.metadata or {}):
        raise RepairBlocked("parser_metadata_differs_from_accepted_parse_before_admission")
    parse_original_sha = (parsed.metadata or {}).get("original_sha256")
    if parse_original_sha is not None and parse_original_sha != evidence.original.sha256:
        raise RepairBlocked("accepted_parse_original_binding_missing_or_mismatched")

    prior_elapsed_value = (state or {}).get("elapsed_seconds", 0.0)
    prior_elapsed = (
        float(prior_elapsed_value) if isinstance(prior_elapsed_value, int | float) else 0.0
    )
    prior_inference_value = (state or {}).get("inference")
    prior_inference: dict[str, int] = (
        {
            key: value
            for key, value in prior_inference_value.items()
            if isinstance(prior_inference_value, dict)
            and isinstance(key, str)
            and type(value) is int
        }
        if isinstance(prior_inference_value, dict)
        else {}
    )
    checkpoint: dict[str, Any] = {
        "schema": _STATE_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "job_id": job_id,
        "source_id": evidence.source_id,
        "target_collection": evidence.target_collection,
        "expected_active_revision": evidence.source_revision_sequence,
        "terminal": "applying",
        "owner": "vo_rag.ops.corpus_repair",
        "recovery": "resume this manifest to reuse the private prepared VoPoint roster",
        "metrics": {"replacement_count": len(replacements)},
        "inference": prior_inference,
        "elapsed_seconds": prior_elapsed,
    }
    job = None
    local = None
    async with _resources() as (settings, store, pool):
        try:
            if prior_elapsed + time.monotonic() - started >= budget.max_wall_seconds:
                raise RepairBlocked("repair_wall_time_budget_exhausted")
            fresh = await _fresh_preflight(evidence, store)
            await _check_artifacts(evidence, pool)
            from vo_rag.ops.repair_legacy_parse import LegacyParseBlocked
            from vo_rag.ops.repair_page_parse import verify_repair_parse

            try:
                checkpoint["accepted_parse_current_byte_binding"] = await verify_repair_parse(
                    evidence,
                    parsed,
                    settings,
                    pool,
                    deadline=started + budget.max_wall_seconds - prior_elapsed,
                )
            except LegacyParseBlocked as exc:
                raise RepairBlocked(str(exc)) from exc
            if fresh["active_revision"] != evidence.source_revision_sequence:
                raise RepairBlocked("active_revision_changed_before_admission")

            from vo_rag.core.embedding.client import DEFAULT_EMBED_MAX_LENGTH, EmbeddingClient
            from vo_rag.pipelines.book_rag.quality_warnings import collect_quality_warnings
            from vo_rag.pipelines.book_rag.steps import chunk_embed
            from vo_rag.pipelines.book_rag.visual_sidecar import build_extraction_composites

            include_colbert = cast(Any, chunk_embed).include_colbert_at_ingest

            default_request_recipe = chunk_embed._embedding_recipe(
                DEFAULT_EMBED_MAX_LENGTH,
                include_colbert(),
            )
            request_recipe_sha256 = canonical_sha256(default_request_recipe)
            if request_recipe_sha256 != request_recipe_sha256_evidence:
                raise RepairBlocked("active_embedding_request_recipe_differs_from_approval")
            code_sha256 = _writer_code_sha256()
            config_sha256 = _configuration_sha256(evidence, settings, request_recipe_sha256)
            recipe_sha256 = _recipe_sha256(evidence, code_sha256, config_sha256)
            if code_sha256 != evidence.writer_recipe.writer_code_sha256:
                raise RepairBlocked("active_writer_code_differs_from_approval")
            if config_sha256 != evidence.writer_recipe.configuration_sha256:
                raise RepairBlocked("active_writer_configuration_differs_from_approval")
            if recipe_sha256 != evidence.writer_recipe.recipe_sha256:
                raise RepairBlocked("active_writer_recipe_differs_from_approval")

            canonical_snapshot = await _canonical_source_snapshot(pool, evidence.source_id)
            if canonical_sha256(canonical_snapshot) != canonical_source_snapshot_sha256:
                raise RepairBlocked("canonical_source_record_changed_since_approval")
            if canonical_snapshot != canonical_source_snapshot:
                raise RepairBlocked("canonical_source_record_snapshot_mismatch")
            envelope_metadata = cast(dict[str, object], envelope_metadata_snapshot)
            parser_metadata = cast(dict[str, object], parser_metadata_snapshot)
            book_metadata = _book_metadata_from_snapshot(
                parsed.title or "", envelope_metadata, canonical_snapshot, evidence.source_id
            )
            if (
                canonical_sha256(book_metadata) != book_metadata_sha256
                or book_metadata != book_metadata_snapshot
            ):
                raise RepairBlocked("book_metadata_output_differs_from_approved_snapshot")

            backup_path = _before_backup_path(state_path)
            prior_backup = (state or {}).get("before_backup")
            if isinstance(prior_backup, dict):
                if not isinstance(prior_backup.get("sha256"), str):
                    raise RepairBlocked("before_backup_checkpoint_invalid")
                backup_value = _verify_before_backup(
                    backup_path,
                    sha256=prior_backup["sha256"],
                    source_id=evidence.source_id,
                    collection=evidence.target_collection,
                    max_bytes=budget.max_evidence_bytes,
                )
                if len(backup_value["points"]) != prior_backup.get(
                    "point_count"
                ) or not _before_backup_matches_evidence(evidence, backup_value):
                    raise RepairBlocked("before_backup_checkpoint_roster_mismatch")
                backup_info = prior_backup
            else:
                backup_info = await _capture_before_backup(
                    evidence,
                    store,
                    backup_path,
                    max_bytes=budget.max_evidence_bytes,
                    reserved_bytes=len(manifest_raw) + evidence.census.evidence_bytes,
                )
            checkpoint["before_backup"] = backup_info
            _atomic_state(state_path, checkpoint)

            if prior_elapsed + time.monotonic() - started >= budget.max_wall_seconds:
                raise RepairBlocked("repair_wall_time_budget_exhausted_before_admission")

            # Admission is the serialization point and allocates the exact
            # global source sequence. Bind only immutable manifest declarations
            # here; observed output receipts are checkpointed privately later.
            if evidence.replacement_revision_sequence is None:
                raise RepairBlocked("approved_replacement_revision_sequence_missing")
            producer_recipe = {
                "manifest_sha256": manifest_sha256,
                "recipe_sha256": evidence.writer_recipe.recipe_sha256,
                "writer_code_sha256": evidence.writer_recipe.writer_code_sha256,
                "configuration_sha256": evidence.writer_recipe.configuration_sha256,
                "execution_receipt_sha256": evidence.writer_recipe.execution_receipt_sha256,
            }
            job = await store.admit_source_revision(
                evidence.target_collection,
                evidence.source_id,
                job_id,
                evidence.original.sha256,
                expected_active_revision=evidence.source_revision_sequence,
                producer_recipe=producer_recipe,
            )
            if job is None:
                raise RepairBlocked("source_publication_admission_unavailable")
            if job.revision != evidence.replacement_revision_sequence:
                coordinator = await store._source_coordinator(evidence.target_collection)
                if coordinator is not None:
                    async with coordinator.fence(evidence.source_id, job_id=job_id) as fence:
                        await coordinator.checkpoint(
                            fence,
                            job_id,
                            "preserved_no_write",
                            {
                                "write_attempts": 0,
                                "no_write": {
                                    "qdrant_write_attempts": 0,
                                    "acknowledgement_unknown": False,
                                    "reason": (
                                        "approved replacement sequence differs from allocated "
                                        "sequence"
                                    ),
                                },
                            },
                        )
                raise RepairBlocked("allocated_revision_sequence_differs_from_approval_no_write")

            if isinstance(saved_points_sha, str):
                if type(points_count) is not int or points_count <= 0:
                    raise RepairBlocked("durable_writer_points_checkpoint_invalid")
                points = _load_vo_points(points_path, saved_points_sha, points_count)
                inference_usage = dict(prior_inference)
            else:
                local = EmbeddingClient(
                    budget.local_endpoint or "", timeout_s=settings.embedding.timeout_s
                )
                request_recipe = chunk_embed._embedding_recipe(
                    int(getattr(local, "_max_length", DEFAULT_EMBED_MAX_LENGTH)),
                    include_colbert(),
                )
                active_request_recipe_sha256 = canonical_sha256(request_recipe)
                if active_request_recipe_sha256 != request_recipe_sha256:
                    raise RepairBlocked("active_embedding_request_recipe_differs_from_approval")

                def checkpoint_usage(usage: dict[str, int]) -> None:
                    checkpoint["inference"] = usage
                    checkpoint["elapsed_seconds"] = round(
                        prior_elapsed + time.monotonic() - started, 3
                    )
                    _atomic_state(state_path, checkpoint)

                embedder = _BudgetedEmbeddingClient(
                    local,
                    max_requests=budget.inference_requests,
                    max_tokens=budget.inference_tokens,
                    max_wall_seconds=budget.max_wall_seconds,
                    started_at=started,
                    on_usage=checkpoint_usage,
                    initial_usage=prior_inference,
                    elapsed_before=prior_elapsed,
                )
                try:
                    composites = await build_extraction_composites(
                        parsed, document_metadata=book_metadata
                    )
                    chunks = chunk_embed.do_build_chunks(
                        parsed,
                        composites=composites,
                        book_metadata=book_metadata,
                        settings=settings,
                        embedding_client=embedder,
                    )
                    text_bytes = sum(len((chunk.content or "").encode("utf-8")) for chunk in chunks)
                    if len(chunks) > budget.max_replacement_chunks:
                        raise RepairBlocked("replacement_chunk_budget_exceeded")
                    if text_bytes > budget.max_replacement_text_bytes:
                        raise RepairBlocked("replacement_text_budget_exceeded")
                    embeddings = await chunk_embed.do_embed_raw(chunks, embedder, embed_batch=1)
                    provenances = embeddings.get("embedding_provenance")
                    if not isinstance(provenances, list) or len(provenances) != len(chunks):
                        raise RepairBlocked("embedding_provenance_cardinality_mismatch")
                    if any(
                        not isinstance(item, dict)
                        or item.get("request_recipe_sha256") != request_recipe_sha256
                        for item in provenances
                    ):
                        raise RepairBlocked("embedding_request_recipe_receipt_mismatch")
                    if evidence.writer_recipe.runtime_identity_status == "known":
                        identities = {
                            json.dumps(
                                item.get("runtime_identity"),
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            for item in provenances
                            if isinstance(item, dict)
                        }
                        if (
                            len(identities) != 1
                            or next(iter(identities)) != evidence.writer_recipe.runtime_identity
                        ):
                            raise RepairBlocked("observed_embedding_runtime_differs_from_approval")
                    warning_context = SimpleNamespace(
                        envelope=SimpleNamespace(metadata=envelope_metadata),
                        parse_meta=parser_metadata,
                        parsed=parsed,
                        chunks=chunks,
                        quality_warnings=[],
                    )
                    warnings = collect_quality_warnings(warning_context)
                    if canonical_sha256(warnings) != quality_warnings_sha256:
                        raise RepairBlocked("document_quality_warnings_differs_from_approval")
                    points = chunk_embed.build_raw_points(
                        chunks,
                        embeddings,
                        evidence.source_id,
                        collection=evidence.target_collection,
                        trust_level=cast(str, trust_level),
                        generation_id=None,
                        ingest_state="active",
                    )
                    if len(points) != len(replacements):
                        raise RepairBlocked(
                            "normal_producer_output_count_differs_from_approved_roster"
                        )
                    _bind_occurrences_to_chunks(evidence, points)
                    for point in points:
                        payload = dict(point.payload)
                        payload["quality_warnings"] = warnings
                        if warnings:
                            payload["quality_disposition"] = "warning"
                        point.payload = payload
                    _validate_output_roster(evidence, points)
                    staged = _normal_point_structs(points, evidence)
                    expected_receipt_sha256 = canonical_sha256(
                        [representation_receipt(point) for point in staged]
                    )
                    if expected_receipt_sha256 != evidence.writer_recipe.execution_receipt_sha256:
                        raise RepairBlocked("normal_writer_execution_receipt_differs_from_approval")
                    points_sha256 = _save_vo_points(points_path, points)
                    checkpoint.update(
                        {
                            "writer_points_sha256": points_sha256,
                            "writer_points_count": len(points),
                            "writer_points_artifact": points_path.name,
                            "writer_execution_receipt_sha256": expected_receipt_sha256,
                            "inference": embedder.usage,
                            "elapsed_seconds": round(prior_elapsed + time.monotonic() - started, 3),
                        }
                    )
                    _atomic_state(state_path, checkpoint)
                    inference_usage = embedder.usage
                finally:
                    if local is not None:
                        await local._client.aclose()
                        local = None

            _validate_output_roster(evidence, points)
            staged = _normal_point_structs(points, evidence)
            expected_receipt_sha256 = canonical_sha256(
                [representation_receipt(point) for point in staged]
            )
            if expected_receipt_sha256 != evidence.writer_recipe.execution_receipt_sha256:
                raise RepairBlocked("durable_writer_points_receipt_differs_from_approval")
            points_sha256 = saved_points_sha or _save_vo_points(points_path, points)
            checkpoint.update(
                {
                    "writer_points_sha256": points_sha256,
                    "writer_points_count": len(points),
                    "writer_points_artifact": points_path.name,
                    "writer_execution_receipt_sha256": expected_receipt_sha256,
                    "fresh_census_sha256": fresh["fresh_census_sha256"],
                    "terminal": "applying",
                    "writer_upsert_intent": True,
                    "writer_roster_sha256": canonical_sha256(
                        sorted(str(point.id) for point in staged)
                    ),
                    "writer_point_count": len(staged),
                    "inference": inference_usage,
                }
            )
            _atomic_state(state_path, checkpoint)

            if job is None or job.revision != evidence.replacement_revision_sequence:
                raise RepairBlocked("source_publication_job_missing_or_revision_mismatched")
            remaining = budget.max_wall_seconds - prior_elapsed - (time.monotonic() - started)
            if remaining <= 0:
                raise RepairBlocked("repair_wall_time_budget_exhausted_before_write")
            async with asyncio.timeout(remaining):
                result_ids = await store.upsert_vectors(
                    evidence.target_collection,
                    points,
                    publication_job=job,
                    expected_before={
                        point.point_id: point.payload_sha256 for point in evidence.before_points
                    },
                    expected_before_receipts=_approved_before_receipts(evidence),
                )
            expected_ids = {str(point.id) for point in staged}
            if set(map(str, result_ids)) != expected_ids:
                raise RepairBlocked("writer_result_ids_differ_from_approved_after_roster")
            roster_sha256 = canonical_sha256(sorted(expected_ids))
            checkpoint.update(
                {
                    "terminal": "writer_published_needs_verify",
                    "writer_point_count": len(result_ids),
                    "writer_roster_sha256": roster_sha256,
                    "inference": inference_usage,
                    "elapsed_seconds": round(prior_elapsed + time.monotonic() - started, 3),
                }
            )
            _atomic_state(state_path, checkpoint)
            verified = await _verify_live(
                evidence,
                store,
                expected_job_id=job_id,
                expected_ids=expected_ids,
                expected_roster_sha256=roster_sha256,
            )
            verified["artifact_references"] = await _reconcile_artifacts(evidence, store, job_id)
            checkpoint.update({"terminal": "published_verified", "verification": verified})
            _atomic_state(state_path, checkpoint)
            return {
                "job_id": job_id,
                "state": "published_verified",
                "source_id": evidence.source_id,
                "target_collection": evidence.target_collection,
                "writer_point_count": len(result_ids),
                "inference": inference_usage,
                "verification": verified,
            }
        except BaseException as exc:
            no_write_sequence_mismatch = (
                isinstance(exc, RepairBlocked)
                and str(exc) == "allocated_revision_sequence_differs_from_approval_no_write"
            )
            checkpoint.update(
                {
                    "terminal": (
                        "preserved_no_write"
                        if no_write_sequence_mismatch
                        else (
                            "writer_published_needs_verify"
                            if checkpoint.get("terminal") == "writer_published_needs_verify"
                            else "failed_retryable_preserved"
                        )
                    ),
                    "failure_class": type(exc).__name__,
                    "inference": checkpoint.get("inference", prior_inference),
                    "elapsed_seconds": round(prior_elapsed + time.monotonic() - started, 3),
                    "recovery": (
                        "the allocated sequence did not match approval; no model or Qdrant "
                        "work was performed"
                        if no_write_sequence_mismatch
                        else "resume the identical manifest and reuse private prepared points; "
                        "the active pointer remains authoritative"
                    ),
                }
            )
            _atomic_state(state_path, checkpoint)
            raise
        finally:
            if local is not None:
                await local._client.aclose()


async def _verify_live(
    evidence: RepairEvidence,
    store: Any,
    *,
    expected_job_id: str,
    expected_ids: set[str] | None = None,
    expected_roster_sha256: Any = None,
) -> dict[str, Any]:
    coordinator = await store._source_coordinator(evidence.target_collection)  # noqa: SLF001
    active = await coordinator.read_active(evidence.target_collection, evidence.source_id)
    if active is None:
        raise RepairBlocked("publication_pointer_missing_after_apply")
    if active.job_id != expected_job_id:
        raise RepairBlocked("active_publication_job_superseded_or_mismatched")
    expected_revision = evidence.replacement_revision_sequence
    if expected_revision is None:
        raise RepairBlocked("approved_replacement_revision_sequence_missing")
    if active.revision != expected_revision:
        raise RepairBlocked("active_publication_revision_mismatched")
    job = await coordinator.read_job(expected_job_id)
    if job is None or str(job.state) not in {"published", "PublicationState.PUBLISHED"}:
        raise RepairBlocked("durable_published_job_proof_missing")
    if (
        job.source_sha256 != evidence.original.sha256
        or job.source_id != evidence.source_id
        or job.collection != evidence.target_collection
        or job.revision != active.revision
    ):
        raise RepairBlocked("durable_published_job_binding_mismatch")
    durable_verification = job.checkpoint.get("verification")
    durable_receipts = (
        durable_verification.get("point_receipts")
        if isinstance(durable_verification, dict)
        else None
    )
    if not isinstance(durable_receipts, dict) or not durable_receipts:
        raise RepairBlocked("durable_published_representation_receipts_missing")
    writer = SourceRevisionWriter(store._raw_client, coordinator)  # noqa: SLF001
    rows = await writer.census(evidence.target_collection, evidence.source_id, active.revision)
    ids = {str(row.id) for row in rows}
    if expected_ids is not None and ids != expected_ids:
        raise RepairBlocked("published_source_roster_receipt_mismatch")
    if not rows:
        raise RepairBlocked("published_source_roster_empty")
    actual_receipts = {str(row.id): representation_receipt(row) for row in rows}
    approved_ids = [
        point.point_id
        for point in evidence.after_points
        if point.disposition is PointDisposition.REPLACE
    ]
    if ids != set(approved_ids) or set(durable_receipts) != ids:
        raise RepairBlocked("published_source_roster_receipt_mismatch")
    writer_receipts = {}
    stored_receipts = {}
    for point_id, receipt in durable_receipts.items():
        if (
            not isinstance(receipt, dict)
            or not isinstance(receipt.get("writer"), dict)
            or not isinstance(receipt.get("stored"), dict)
            or receipt.get("storage_transform") != "cosine normalization and float32 rounding"
            or receipt.get("verification")
            != "payload_exact_modalities_complete_numeric_transform_verified"
        ):
            raise RepairBlocked("durable_published_representation_receipt_invalid")
        writer_receipts[point_id] = receipt["writer"]
        stored_receipts[point_id] = receipt["stored"]
    if (
        canonical_sha256([writer_receipts[point_id] for point_id in approved_ids])
        != evidence.writer_recipe.execution_receipt_sha256
    ):
        raise RepairBlocked("published_full_representations_differ_from_approved_manifest")
    # The publication writer already validated the only allowed storage
    # transform. Bind the approved pre-write output to its writer receipts,
    # and fresh exact read-back to its separately retained stored receipts.
    if actual_receipts != stored_receipts:
        raise RepairBlocked("published_full_representation_receipts_mismatch")
    if (
        durable_verification.get("manifest_sha256") != canonical_sha256(durable_receipts)
        or durable_verification.get("expected_point_ids") != approved_ids
        or durable_verification.get("readback_point_ids") != sorted(ids)
    ):
        raise RepairBlocked("durable_published_verification_manifest_mismatch")
    roster_sha256 = canonical_sha256(sorted(ids))
    if expected_roster_sha256 is not None and roster_sha256 != expected_roster_sha256:
        raise RepairBlocked("published_source_roster_checkpoint_mismatch")
    first = rows[0]
    vectors = first.vector
    if not isinstance(vectors, dict) or not isinstance(vectors.get("dense"), list):
        raise RepairBlocked("query_smoke_requires_dense_named_vector")
    # This is a technical source/publication-boundary smoke test using a stored
    # vector. It makes no semantic retrieval claim and performs no model call.
    hits = await store.search_single_collection(
        evidence.target_collection,
        vectors["dense"],
        top_k=1,
        filters={"source_id": evidence.source_id},
    )
    if not hits:
        raise RepairBlocked("source_filtered_query_smoke_returned_no_hits")
    payload = hits[0].payload or {}
    if payload.get("source_id") != evidence.source_id:
        raise RepairBlocked("source_filtered_query_smoke_scope_mismatch")
    if payload.get("source_revision") != active.revision or str(hits[0].id) not in ids:
        raise RepairBlocked("source_filtered_query_smoke_publication_boundary_mismatch")
    return {
        "active_revision": active.revision,
        "source_point_count": len(rows),
        "roster_sha256": roster_sha256,
        "query_smoke": "source_filter_and_citation_scope_verified_no_semantic_claim",
        "query_hit_id_sha256": hashlib.sha256(str(hits[0].id).encode()).hexdigest(),
    }


async def verify_manifest(path: Path, state_dir: Path) -> dict[str, Any]:
    evidence, digest, _ = _load_evidence(path)
    job_id = _job_id(evidence, digest)
    state_path = _state_path(state_dir, digest)
    state = _state_load(state_path, manifest_sha256=digest, job_id=job_id)
    if state is None or state.get("terminal") not in {
        "writer_published_needs_verify",
        "published_verified",
    }:
        raise RepairBlocked("verification_requires_published_writer_checkpoint")
    async with _resources() as (_, store, _pool):
        result = await _verify_live(
            evidence,
            store,
            expected_job_id=job_id,
            expected_roster_sha256=state.get("writer_roster_sha256"),
        )
        result["artifact_references"] = await _reconcile_artifacts(
            evidence, store, job_id, write=False
        )
    state.update({"terminal": "published_verified", "verification": result})
    _atomic_state(state_path, state)
    return {"job_id": job_id, "state": "published_verified", "verification": result}


def _run(awaitable: Any) -> Any:
    return asyncio.run(awaitable)


def register(group: click.Group) -> None:
    """Register ``corpus-repair plan|preflight|apply|resume|verify`` commands."""

    @group.group("corpus-repair")
    def corpus_repair() -> None:
        """Plan and run one explicitly manifested, bounded source repair."""

    def common_manifest(fn: Any) -> Any:
        fn = click.option(
            "--manifest",
            type=click.Path(path_type=Path, exists=True, dir_okay=False),
            required=True,
        )(fn)
        return fn

    @corpus_repair.command("plan")
    @common_manifest
    def plan_command(manifest: Path) -> None:
        """Validate immutable evidence locally; performs no service calls."""
        try:
            click.echo(json.dumps(plan_manifest(manifest), sort_keys=True))
        except RepairBlocked as exc:
            click.echo(
                json.dumps({"eligible": False, "disposition": "blocked", "reason": str(exc)})
            )
            raise click.ClickException("repair plan is blocked") from None
        except Exception as exc:
            click.echo(
                json.dumps(
                    {
                        "eligible": False,
                        "disposition": "failed",
                        "failure_class": type(exc).__name__,
                    }
                )
            )
            raise click.ClickException("repair plan failed") from None

    @corpus_repair.command("preflight")
    @common_manifest
    @click.option("--state-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
    def preflight_command(manifest: Path, state_dir: Path) -> None:
        """Read a fresh complete source census and active revision token."""
        try:
            click.echo(json.dumps(_run(preflight_manifest(manifest, state_dir)), sort_keys=True))
        except RepairBlocked as exc:
            click.echo(
                json.dumps({"eligible": False, "disposition": "blocked", "reason": str(exc)})
            )
            raise click.ClickException("repair preflight is blocked") from None
        except Exception as exc:
            click.echo(
                json.dumps(
                    {
                        "eligible": False,
                        "disposition": "failed",
                        "failure_class": type(exc).__name__,
                    }
                )
            )
            raise click.ClickException("repair preflight failed") from None

    @corpus_repair.command("apply")
    @click.option(
        "--manifest", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True
    )
    @click.option("--state-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
    def apply_command(manifest: Path, state_dir: Path) -> None:
        """Apply one eligible source repair with its declared finite budget."""
        try:
            click.echo(json.dumps(_run(apply_manifest(manifest, state_dir)), sort_keys=True))
        except RepairBlocked as exc:
            click.echo(json.dumps({"disposition": "blocked", "reason": str(exc)}))
            raise click.ClickException("repair apply is blocked") from None
        except Exception as exc:
            click.echo(
                json.dumps(
                    {
                        "disposition": "failed_retryable_preserved",
                        "failure_class": type(exc).__name__,
                    }
                )
            )
            raise click.ClickException("repair apply failed") from None

    @corpus_repair.command("resume")
    @click.option(
        "--manifest", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True
    )
    @click.option("--state-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
    def resume_command(manifest: Path, state_dir: Path) -> None:
        """Resume the same immutable source job from private checkpoints."""
        try:
            click.echo(
                json.dumps(_run(apply_manifest(manifest, state_dir, resume=True)), sort_keys=True)
            )
        except RepairBlocked as exc:
            click.echo(json.dumps({"disposition": "blocked", "reason": str(exc)}))
            raise click.ClickException("repair resume is blocked") from None
        except Exception as exc:
            click.echo(
                json.dumps(
                    {
                        "disposition": "failed_retryable_preserved",
                        "failure_class": type(exc).__name__,
                    }
                )
            )
            raise click.ClickException("repair resume failed") from None

    @corpus_repair.command("verify")
    @click.option(
        "--manifest", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True
    )
    @click.option("--state-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
    def verify_command(manifest: Path, state_dir: Path) -> None:
        """Verify publication receipts and source-filtered query boundary."""
        try:
            click.echo(json.dumps(_run(verify_manifest(manifest, state_dir)), sort_keys=True))
        except RepairBlocked as exc:
            click.echo(json.dumps({"disposition": "blocked", "reason": str(exc)}))
            raise click.ClickException("repair verification is blocked") from None
        except Exception as exc:
            click.echo(
                json.dumps(
                    {
                        "disposition": "failed_retryable_preserved",
                        "failure_class": type(exc).__name__,
                    }
                )
            )
            raise click.ClickException("repair verification failed") from None

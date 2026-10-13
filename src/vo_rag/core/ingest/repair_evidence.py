"""Immutable, fail-closed evidence for bounded source repair preflight.

This module validates evidence only. It does not read or write a corpus, invoke
models, or perform a repair. A successful preflight is a necessary evidence
gate, not an authorization or an atomic-publication guarantee.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EvidenceModel(BaseModel):
    """Strict immutable base for serialized evidence records."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RepairCorpus(StrEnum):
    """Corpora supported by source-repair evidence."""

    BOOKS = "books"
    REDDIT = "reddit"


class PublicationClass(StrEnum):
    """Explicit publication boundary for a repair target."""

    BOOKS_RAW = "books_raw"
    BOOKS_STERILE = "books_sterile"
    BOOKS_CONTEXTUALIZED = "books_contextualized"
    BOOKS_GRAPH_NODES = "books_graph_nodes"
    BOOKS_GRAPH_EDGES = "books_graph_edges"
    BOOKS_DSI = "books_dsi"
    REDDIT = "reddit"


class RepairClaimKind(StrEnum):
    """Evidence class for a proposed source change."""

    DETERMINISTIC_SOURCE_FIDELITY = "deterministic_source_fidelity"
    SEMANTIC = "semantic"
    DERIVED_REPRESENTATION = "derived_representation"


class PointDisposition(StrEnum):
    """Required explicit treatment for every before/after point."""

    PRESERVE = "preserve"
    REPLACE = "replace"
    UNRESOLVED = "unresolved"
    EXCLUDED = "excluded"


class TerminalDisposition(StrEnum):
    """Conservative preflight result."""

    ELIGIBLE = "eligible"
    PRESERVED_NO_WRITE = "preserved_no_write"
    BLOCKED_MISSING_EVIDENCE = "blocked_missing_evidence"
    EXCLUDED_NOT_APPLICABLE = "excluded_not_applicable"


class SourceArtifactRole(StrEnum):
    ORIGINAL = "original"
    ACCEPTED_PARSE = "accepted_parse"
    BASELINE_PARSE = "baseline_parse"
    RECOVERED_PAGES_PARSE = "recovered_pages_parse"
    REDDIT_ARCHIVE = "reddit_archive"
    REDDIT_LINEAGE_MANIFEST = "reddit_lineage_manifest"


class OccurrenceKind(StrEnum):
    DOCUMENT = "document"
    PAGE = "page"
    ELEMENT = "element"
    TABLE = "table"
    REDDIT_POST = "reddit_post"
    REDDIT_COMMENT = "reddit_comment"


class TypedRecordRole(StrEnum):
    GRAPH_NODE = "graph_node"
    GRAPH_EDGE = "graph_edge"
    DSI_RECORD = "dsi_record"


class TextRoleDecision(StrEnum):
    EXPECTS_TEXT = "expects_text"
    DOES_NOT_EXPECT_TEXT = "does_not_expect_text"
    UNKNOWN = "unknown"


class HumanSemanticGold(EvidenceModel):
    """Independent human labels required for semantic claims."""

    reviewer_id: str = Field(min_length=1)
    reviewer_is_human: Literal[True]
    sampling_method: str = Field(min_length=1)
    labeled_units: int = Field(gt=0)
    positive_labels: int = Field(ge=0)
    negative_labels: int = Field(ge=0)
    control_units: int = Field(ge=0)
    label_manifest_sha256: str

    @field_validator("label_manifest_sha256")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("label_manifest_sha256 must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _labels_fit_denominator(self) -> HumanSemanticGold:
        if self.positive_labels + self.negative_labels != self.labeled_units:
            raise ValueError("positive and negative labels must equal labeled_units")
        return self


class RepairBudget(EvidenceModel):
    """Finite per-job caps; increasing defaults must be explicit."""

    max_sources: int = Field(default=1, gt=0)
    max_replacement_chunks: int = Field(default=500, gt=0)
    max_replacement_text_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_evidence_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    max_wall_seconds: int = Field(default=1800, gt=0)
    concurrency: int = Field(default=1, gt=0)
    inference_requests: int = Field(default=0, ge=0)
    inference_tokens: int = Field(default=0, ge=0)
    local_endpoint: str | None = None
    paid_fallback: Literal[False] = False
    explicitly_increased: bool = False

    @model_validator(mode="after")
    def _finite_explicit_budget(self) -> RepairBudget:
        defaults = {
            "max_sources": 1,
            "max_replacement_chunks": 500,
            "max_replacement_text_bytes": 20 * 1024 * 1024,
            "max_evidence_bytes": 50 * 1024 * 1024,
            "max_wall_seconds": 1800,
            "concurrency": 1,
        }
        increased = any(getattr(self, key) > value for key, value in defaults.items())
        if increased and not self.explicitly_increased:
            raise ValueError("a budget above defaults requires explicitly_increased=true")
        if not increased and self.explicitly_increased:
            raise ValueError("explicitly_increased is only for a larger finite budget")
        if self.inference_requests == 0:
            if self.inference_tokens != 0 or self.local_endpoint is not None:
                raise ValueError("zero-inference jobs cannot carry endpoint or token allowances")
        else:
            if self.inference_tokens <= 0 or self.local_endpoint is None:
                raise ValueError(
                    "inference requires explicit request, token, and local endpoint caps"
                )
            if not is_local_endpoint(self.local_endpoint):
                raise ValueError("inference endpoint must be loopback or a private local IP")
        return self


DEFAULT_REPAIR_BUDGET = RepairBudget()


def is_local_endpoint(value: str) -> bool:
    """Return whether an endpoint URL names loopback or a private IP literal."""
    try:
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return False
        if parts.username is not None or parts.password is not None:
            return False
        hostname = parts.hostname.lower().rstrip(".")
        if hostname == "localhost" or hostname.endswith(".localhost"):
            return True
        address = ipaddress.ip_address(hostname)
        return address.is_loopback or address.is_private
    except (ValueError, TypeError):
        return False


class SourceArtifact(EvidenceModel):
    """A local source or lineage artifact bound to exact bytes."""

    role: SourceArtifactRole
    path: Path
    sha256: str
    size_bytes: int = Field(ge=0)
    parent_sha256: str | None = None

    @field_validator("sha256")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("sha256 must be lowercase SHA-256")
        return value

    @field_validator("parent_sha256")
    @classmethod
    def _valid_parent_hash(cls, value: str | None) -> str | None:
        if value is not None and SHA256_RE.fullmatch(value) is None:
            raise ValueError("parent_sha256 must be lowercase SHA-256")
        return value


class RecoveredPageParseEvidence(EvidenceModel):
    """Immutable baseline and genuine physical-page output, replayed at admission."""

    baseline_parse: SourceArtifact
    recovered_pages_parse: SourceArtifact
    physical_pages: tuple[int, ...] = Field(min_length=1, max_length=512)
    parser_id: Literal["docling.physical_pages.no_ocr.v1"]
    parser_recipe_sha256: str

    @field_validator("parser_recipe_sha256")
    @classmethod
    def _valid_recipe_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("parser recipe must use lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _physical_page_scope(self) -> RecoveredPageParseEvidence:
        if self.physical_pages != tuple(sorted(set(self.physical_pages))):
            raise ValueError("physical recovery pages must be unique and ordered")
        if any(page <= 0 for page in self.physical_pages):
            raise ValueError("physical recovery pages must be positive")
        if self.baseline_parse.role is not SourceArtifactRole.BASELINE_PARSE:
            raise ValueError("recovery baseline artifact must use baseline_parse role")
        if self.recovered_pages_parse.role is not SourceArtifactRole.RECOVERED_PAGES_PARSE:
            raise ValueError("page output artifact must use recovered_pages_parse role")
        return self


class SourceOccurrence(EvidenceModel):
    """A unique source occurrence with a byte-bound locator."""

    occurrence_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    kind: OccurrenceKind
    artifact_sha256: str
    start_byte: int = Field(ge=0)
    end_byte: int = Field(gt=0)
    span_sha256: str
    locator: str = Field(min_length=1)

    @field_validator("artifact_sha256", "span_sha256")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("artifact_sha256 must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _positive_span(self) -> SourceOccurrence:
        if self.end_byte <= self.start_byte:
            raise ValueError("source occurrence must have a non-empty byte span")
        return self


class RedditArchiveOccurrence(EvidenceModel):
    """Occurrence imported from an authoritative archive manifest only.

    The manifest entry digest and archive digest bind the mapping. A permalink
    or a guessed thread/comment ID by itself is insufficient lineage evidence.
    """

    occurrence_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    kind: Literal[OccurrenceKind.REDDIT_POST, OccurrenceKind.REDDIT_COMMENT]
    archive_sha256: str
    manifest_sha256: str
    manifest_entry_sha256: str
    lineage_source: Literal["authoritative_archive_manifest"]
    subreddit: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    post_id: str = Field(min_length=1)
    comment_id: str | None = None
    parent_comment_id: str | None = None
    ordinal: int = Field(ge=0)
    canonical_permalink: str = Field(min_length=1)
    identity_basis: Literal["reddit_id", "archive_tree_path"] = "reddit_id"
    archive_post_index: int | None = Field(default=None, ge=0)
    archive_comment_path: tuple[int, ...] | None = None
    archive_record_sha256: str | None = None
    rendered_start_byte: int | None = Field(default=None, ge=0)
    rendered_end_byte: int | None = Field(default=None, gt=0)

    @field_validator("archive_sha256", "manifest_sha256", "manifest_entry_sha256")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("lineage digests must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _consistent_archive_ids(self) -> RedditArchiveOccurrence:
        if self.kind is OccurrenceKind.REDDIT_POST and self.comment_id is not None:
            raise ValueError("post occurrence cannot carry comment_id")
        if self.identity_basis == "archive_tree_path":
            if self.archive_post_index is None or self.archive_record_sha256 is None:
                raise ValueError(
                    "archive path identity requires the actual post index and record digest"
                )
            if SHA256_RE.fullmatch(self.archive_record_sha256) is None:
                raise ValueError("archive record digest must be lowercase SHA-256")
            if self.comment_id is not None or self.parent_comment_id is not None:
                raise ValueError("archive path identity cannot invent Reddit comment IDs")
            if self.kind is OccurrenceKind.REDDIT_COMMENT and (
                not self.archive_comment_path or any(i < 0 for i in self.archive_comment_path)
            ):
                raise ValueError("archive comment identity requires a nonnegative tree path")
            if self.kind is OccurrenceKind.REDDIT_POST and self.archive_comment_path is not None:
                raise ValueError("post occurrence cannot carry a comment tree path")
            if (
                self.rendered_start_byte is None
                or self.rendered_end_byte is None
                or self.rendered_end_byte <= self.rendered_start_byte
            ):
                raise ValueError(
                    "archive occurrence requires an exact positive rendered source span"
                )
        elif any(
            value is not None
            for value in (
                self.archive_post_index,
                self.archive_comment_path,
                self.archive_record_sha256,
                self.rendered_start_byte,
                self.rendered_end_byte,
            )
        ):
            raise ValueError("Reddit ID identity cannot carry unchecked archive path fields")
        if (
            self.identity_basis == "reddit_id"
            and self.kind is OccurrenceKind.REDDIT_COMMENT
            and self.comment_id is None
        ):
            raise ValueError("comment occurrence requires an archive comment_id")
        parts = urlsplit(self.canonical_permalink)
        path_parts = parts.path.strip("/").split("/")
        if (
            len(path_parts) < 4
            or path_parts[0] != "r"
            or path_parts[1] != self.subreddit
            or path_parts[2] != "comments"
            or path_parts[3] != self.post_id
        ):
            raise ValueError("permalink does not bind the declared subreddit and post_id")
        return self


class RedditLineageEvidence(EvidenceModel):
    """Verified archive plus authoritative occurrence manifest mapping."""

    archive: SourceArtifact
    manifest: SourceArtifact
    occurrences: tuple[RedditArchiveOccurrence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _roles_and_unique_occurrences(self) -> RedditLineageEvidence:
        if self.archive.role is not SourceArtifactRole.REDDIT_ARCHIVE:
            raise ValueError("archive artifact must use reddit_archive role")
        if self.manifest.role is not SourceArtifactRole.REDDIT_LINEAGE_MANIFEST:
            raise ValueError("manifest artifact must use reddit_lineage_manifest role")
        ids = [o.occurrence_id for o in self.occurrences]
        if len(ids) != len(set(ids)):
            raise ValueError("archive occurrence IDs must be unique")
        if any(o.archive_sha256 != self.archive.sha256 for o in self.occurrences):
            raise ValueError("occurrence archive digest does not match archive artifact")
        if any(o.manifest_sha256 != self.manifest.sha256 for o in self.occurrences):
            raise ValueError("occurrence manifest digest does not match manifest artifact")
        logical_ids = [
            (
                o.post_id,
                (o.archive_comment_path, o.rendered_start_byte, o.rendered_end_byte)
                if o.identity_basis == "archive_tree_path"
                else "post"
                if o.kind is OccurrenceKind.REDDIT_POST
                else o.comment_id,
            )
            for o in self.occurrences
        ]
        if len(logical_ids) != len(set(logical_ids)):
            raise ValueError("archive thread/comment occurrence identities must be unique")
        by_comment = {
            o.comment_id: o
            for o in self.occurrences
            if o.kind is OccurrenceKind.REDDIT_COMMENT and o.comment_id is not None
        }
        ordinals = [
            (o.post_id, o.ordinal, o.rendered_start_byte, o.rendered_end_byte)
            for o in self.occurrences
        ]
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("archive occurrence order must be unique within each thread")
        for occurrence in self.occurrences:
            if occurrence.thread_id != occurrence.post_id:
                raise ValueError("archive thread_id must equal the canonical post_id")
            if occurrence.kind is OccurrenceKind.REDDIT_COMMENT and occurrence.parent_comment_id:
                parent = by_comment.get(occurrence.parent_comment_id)
                if parent is None or parent.post_id != occurrence.post_id:
                    raise ValueError("comment parent must be present in the same archive thread")
                if parent.ordinal >= occurrence.ordinal:
                    raise ValueError("archive parent occurrence must precede its child")
        return self


class _RedditManifestOccurrence(EvidenceModel):
    """One strict archive manifest row; no inferred IDs are accepted."""

    occurrence_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    kind: Literal[OccurrenceKind.REDDIT_POST, OccurrenceKind.REDDIT_COMMENT]
    subreddit: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    post_id: str = Field(min_length=1)
    comment_id: str | None = None
    parent_comment_id: str | None = None
    ordinal: int = Field(ge=0)
    canonical_permalink: str = Field(min_length=1)
    identity_basis: Literal["reddit_id", "archive_tree_path"] = "reddit_id"
    archive_post_index: int | None = Field(default=None, ge=0)
    archive_comment_path: tuple[int, ...] | None = None
    archive_record_sha256: str | None = None
    rendered_start_byte: int | None = Field(default=None, ge=0)
    rendered_end_byte: int | None = Field(default=None, gt=0)


class _RedditManifestDocument(EvidenceModel):
    """Versioned source of truth consumed by the Reddit lineage adapter."""

    schema_id: Literal[
        "vo-rag.reddit-occurrence-manifest.v1", "vo-rag.reddit-occurrence-manifest.v2"
    ] = Field(alias="schema")
    archive_sha256: str
    occurrences: tuple[_RedditManifestOccurrence, ...] = Field(min_length=1)

    @field_validator("archive_sha256")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("archive_sha256 must be lowercase SHA-256")
        return value


class VectorFingerprint(EvidenceModel):
    """Full captured vector values and their canonical fingerprint.

    Empty sparse index/value lists are genuine zero lexical vectors. Dense and
    multivector captures still require finite, nonempty rows and exact dimensions.
    """

    sha256: str
    dimensions: int = Field(gt=0)
    kind: Literal["dense", "sparse", "multivector"] = "dense"
    values: tuple[float, ...] | tuple[tuple[float, ...], ...]
    indices: tuple[int, ...] | None = None

    @field_validator("sha256")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("vector sha256 must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _validate_values(self) -> VectorFingerprint:
        nested = bool(self.values and isinstance(self.values[0], tuple))
        if self.kind == "multivector":
            if not nested or any(
                not row or len(row) != self.dimensions or any(not math.isfinite(v) for v in row)
                for row in self.values
                if isinstance(row, tuple)
            ):
                raise ValueError("multivector values must be finite rows matching dimensions")
            if self.indices is not None:
                raise ValueError("multivectors cannot carry sparse indices")
            canonical: object = [
                list(row) for row in cast(tuple[tuple[float, ...], ...], self.values)
            ]
        elif nested:
            raise ValueError("dense and sparse vectors require flat values")
        elif any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            for value in self.values
        ):
            raise ValueError("vector values must be finite")
        elif self.kind == "dense":
            if self.indices is not None or len(self.values) != self.dimensions:
                raise ValueError("dense vector values must match dimensions and have no indices")
            canonical = list(cast(tuple[float, ...], self.values))
        else:
            if self.indices is None or len(self.indices) != len(self.values):
                raise ValueError("sparse vectors require one index per value")
            if any(index < 0 or index >= self.dimensions for index in self.indices):
                raise ValueError("sparse vector index falls outside dimensions")
            if len(self.indices) != len(set(self.indices)):
                raise ValueError("sparse vector indices must be unique")
            canonical = {
                "indices": list(self.indices),
                "values": list(cast(tuple[float, ...], self.values)),
            }
        actual = hashlib.sha256(
            json.dumps(canonical, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        if actual != self.sha256:
            raise ValueError("vector sha256 does not match captured vector values")
        return self


class NamedVectorFingerprint(EvidenceModel):
    """Immutable vector name and captured fingerprint pair."""

    name: str = Field(min_length=1)
    fingerprint: VectorFingerprint


class RepresentationFingerprint(EvidenceModel):
    """Immutable modality name and output fingerprint pair."""

    name: str = Field(min_length=1)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("representation sha256 must be lowercase SHA-256")
        return value


class CensusPoint(EvidenceModel):
    """One complete source-filtered live point from the captured census."""

    point_id: str = Field(min_length=1)
    source_occurrence_id: str = Field(min_length=1)
    payload_sha256: str
    vectors: tuple[NamedVectorFingerprint, ...] = Field(min_length=1)
    representations: tuple[RepresentationFingerprint, ...] = Field(min_length=1)

    @field_validator("payload_sha256")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("payload_sha256 must be lowercase SHA-256")
        return value

    @field_validator("vectors")
    @classmethod
    def _unique_vectors(
        cls, value: tuple[NamedVectorFingerprint, ...]
    ) -> tuple[NamedVectorFingerprint, ...]:
        names = [item.name for item in value]
        if len(names) != len(set(names)):
            raise ValueError("vector names must be unique")
        return tuple(sorted(value, key=lambda item: item.name))

    @field_validator("representations")
    @classmethod
    def _unique_representations(
        cls, value: tuple[RepresentationFingerprint, ...]
    ) -> tuple[RepresentationFingerprint, ...]:
        names = [item.name for item in value]
        if len(names) != len(set(names)):
            raise ValueError("representation names must be unique")
        return tuple(sorted(value, key=lambda item: item.name))


class LiveCensus(EvidenceModel):
    """Complete bounded live census plus an independent count receipt."""

    collection: str = Field(min_length=1)
    captured_at: datetime
    independent_count_captured_at: datetime
    is_complete: Literal[True]
    is_truncated: Literal[False]
    atomic_snapshot: Literal[False]
    source_point_count: int = Field(ge=0)
    independent_source_count: int = Field(ge=0)
    required_vector_names: tuple[str, ...] = Field(min_length=1)
    evidence_bytes: int = Field(gt=0)
    points: tuple[CensusPoint, ...]

    @field_validator("captured_at", "independent_count_captured_at")
    @classmethod
    def _aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("census timestamps must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _complete_counts_and_vectors(self) -> LiveCensus:
        ids = [p.point_id for p in self.points]
        if len(ids) != len(set(ids)):
            raise ValueError("live census point IDs must be unique")
        if self.source_point_count != len(self.points):
            raise ValueError("source_point_count must equal captured point roster length")
        if self.independent_source_count != self.source_point_count:
            raise ValueError("independent source count does not equal census point count")
        if len(set(self.required_vector_names)) != len(self.required_vector_names):
            raise ValueError("required vector names must be unique")
        required = set(self.required_vector_names)
        for point in self.points:
            if {item.name for item in point.vectors} != required:
                raise ValueError("every census point must carry every required vector fingerprint")
        return self


class PointEvidence(EvidenceModel):
    """Before or after point mapping with an explicit disposition."""

    point_id: str = Field(min_length=1)
    source_occurrence_id: str = Field(min_length=1)
    disposition: PointDisposition
    payload_sha256: str
    content_sha256: str
    content_bytes: int = Field(ge=0)
    content: str | None = None
    vectors: tuple[NamedVectorFingerprint, ...] = Field(min_length=1)
    representations: tuple[RepresentationFingerprint, ...] = Field(min_length=1)
    replaces_point_ids: tuple[str, ...] = ()

    @field_validator("payload_sha256", "content_sha256")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("point digests must be lowercase SHA-256")
        return value

    @field_validator("vectors")
    @classmethod
    def _unique_vectors(
        cls, value: tuple[NamedVectorFingerprint, ...]
    ) -> tuple[NamedVectorFingerprint, ...]:
        names = [item.name for item in value]
        if len(names) != len(set(names)):
            raise ValueError("vector names must be unique")
        return tuple(sorted(value, key=lambda item: item.name))

    @field_validator("representations")
    @classmethod
    def _unique_representations(
        cls, value: tuple[RepresentationFingerprint, ...]
    ) -> tuple[RepresentationFingerprint, ...]:
        names = [item.name for item in value]
        if len(names) != len(set(names)):
            raise ValueError("representation names must be unique")
        return tuple(sorted(value, key=lambda item: item.name))

    @model_validator(mode="after")
    def _content_fingerprint(self) -> PointEvidence:
        if len(set(self.replaces_point_ids)) != len(self.replaces_point_ids):
            raise ValueError("replaces_point_ids must be unique")
        if self.content is not None:
            encoded = self.content.encode("utf-8")
            if len(encoded) != self.content_bytes:
                raise ValueError("content_bytes does not match UTF-8 content")
            if hashlib.sha256(encoded).hexdigest() != self.content_sha256:
                raise ValueError("content_sha256 does not match exact UTF-8 content")
        return self


class SourceFidelityEvidence(EvidenceModel):
    """Digest-checked byte span establishing an exact source omission."""

    occurrence_id: str = Field(min_length=1)
    artifact_role: Literal[SourceArtifactRole.ORIGINAL, SourceArtifactRole.ACCEPTED_PARSE]
    artifact_sha256: str
    start_byte: int = Field(ge=0)
    end_byte: int = Field(gt=0)
    omitted_span_sha256: str
    retained_text_accounted_for: Literal[True]
    restored_in_point_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("artifact_sha256", "omitted_span_sha256")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("source fidelity digests must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _positive_span(self) -> SourceFidelityEvidence:
        if self.end_byte <= self.start_byte:
            raise ValueError("omitted source span must be non-empty")
        if len(set(self.restored_in_point_ids)) != len(self.restored_in_point_ids):
            raise ValueError("restored replacement point IDs must be unique")
        return self


class TypedRecordSchemaPolicy(EvidenceModel):
    """Schema-owner determination for graph/DSI text-bearing record roles."""

    record_role: TypedRecordRole
    schema_owner: str | None = None
    decision: TextRoleDecision
    evidence_sha256: str | None = None

    @field_validator("evidence_sha256")
    @classmethod
    def _valid_optional_hash(cls, value: str | None) -> str | None:
        if value is not None and SHA256_RE.fullmatch(value) is None:
            raise ValueError("evidence_sha256 must be lowercase SHA-256")
        return value


class WriterRecipeEvidence(EvidenceModel):
    """Declared writer contract and observed runtime identity or unknowns."""

    writer_id: str = Field(min_length=1)
    target_collection: str = Field(min_length=1)
    recipe_sha256: str
    writer_code_sha256: str
    configuration_sha256: str
    embedding_request_recipe_sha256: str | None = None
    execution_receipt_sha256: str
    modality_input_sha256: tuple[RepresentationFingerprint, ...] = Field(min_length=1)
    actual_writer_observed: Literal[True]
    inference_requests_used: int = Field(default=0, ge=0)
    inference_tokens_used: int = Field(default=0, ge=0)
    runtime_identity: str | None = None
    runtime_identity_status: Literal["known", "unknown"]
    unknown_fields: tuple[str, ...] = ()
    required_modalities: tuple[str, ...] = Field(min_length=1)
    claims_historical_parity: bool = False

    @field_validator(
        "recipe_sha256", "writer_code_sha256", "configuration_sha256", "execution_receipt_sha256"
    )
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("recipe_sha256 must be lowercase SHA-256")
        return value

    @field_validator("embedding_request_recipe_sha256")
    @classmethod
    def _valid_request_recipe_hash(cls, value: str | None) -> str | None:
        if value is not None and SHA256_RE.fullmatch(value) is None:
            raise ValueError("embedding_request_recipe_sha256 must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _runtime_consistency(self) -> WriterRecipeEvidence:
        if self.runtime_identity_status == "known" and not self.runtime_identity:
            raise ValueError("known runtime identity requires a value")
        if self.runtime_identity_status == "unknown" and self.runtime_identity is not None:
            raise ValueError("unknown runtime identity cannot carry an asserted value")
        if self.runtime_identity_status == "unknown" and not self.unknown_fields:
            raise ValueError("unknown runtime identity must be named in unknown_fields")
        if len(set(self.required_modalities)) != len(self.required_modalities):
            raise ValueError("required modalities must be unique")
        input_names = [item.name for item in self.modality_input_sha256]
        if len(input_names) != len(set(input_names)):
            raise ValueError("writer modality inputs must be unique")
        if not set(self.required_modalities) <= set(input_names):
            raise ValueError("writer recipe must bind the inputs for every required modality")
        return self


class RepairEvidence(EvidenceModel):
    """Complete immutable evidence envelope for one source repair unit."""

    schema_version: Literal[1] = 1
    corpus: RepairCorpus
    source_id: str = Field(min_length=1)
    source_revision: str | None = None
    # Required even when null: null explicitly binds a legacy source with no
    # active publication pointer; omission cannot silently mean "current".
    source_revision_sequence: int | None = Field(ge=0)
    # The exact next publication sequence observed during read-only preflight.
    # Admission must return this value before any inference is permitted.
    replacement_revision_sequence: int | None = Field(default=None, gt=0)
    target_collection: str = Field(min_length=1)
    publication_class: PublicationClass
    record_role: TypedRecordRole | None = None
    schema_policy: TypedRecordSchemaPolicy | None = None
    original: SourceArtifact
    accepted_parse: SourceArtifact
    recovered_page_parse: RecoveredPageParseEvidence | None = None
    occurrences: tuple[SourceOccurrence, ...] = Field(min_length=1)
    occurrence_inventory_complete: bool
    reddit_lineage: RedditLineageEvidence | None = None
    census: LiveCensus
    before_points: tuple[PointEvidence, ...]
    after_points: tuple[PointEvidence, ...]
    after_roster_complete: bool
    claim_kind: RepairClaimKind
    source_fidelity: tuple[SourceFidelityEvidence, ...] = ()
    semantic_gold: HumanSemanticGold | None = None
    writer_recipe: WriterRecipeEvidence
    # A repaired raw point must reproduce the same envelope/catalog metadata
    # inputs and trust decision as ordinary intake. These snapshots are
    # optional for read-only historical evidence, but required for apply.
    source_envelope_id: str | None = Field(default=None, min_length=1)
    source_envelope_snapshot: dict[str, object] | None = None
    source_envelope_snapshot_sha256: str | None = None
    owned_artifact_snapshots: tuple[dict[str, object], ...] | None = None
    owned_artifact_snapshots_sha256: str | None = None
    book_metadata: dict[str, object] | None = None
    book_metadata_sha256: str | None = None
    source_envelope_metadata: dict[str, object] | None = None
    source_envelope_metadata_sha256: str | None = None
    parser_metadata: dict[str, object] | None = None
    parser_metadata_sha256: str | None = None
    canonical_source_snapshot: dict[str, object] | None = None
    canonical_source_snapshot_sha256: str | None = None
    quality_warnings: tuple[dict[str, object], ...] | None = None
    quality_warnings_sha256: str | None = None
    trust_level: Literal["trusted", "quarantined"] | None = None
    budget: RepairBudget = DEFAULT_REPAIR_BUDGET
    evidence_snapshot_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def _static_scope_and_bindings(self) -> RepairEvidence:
        if self.original.role is not SourceArtifactRole.ORIGINAL:
            raise ValueError("original artifact must use original role")
        if self.accepted_parse.role is not SourceArtifactRole.ACCEPTED_PARSE:
            raise ValueError("accepted parse must use accepted_parse role")
        if self.original.parent_sha256 is not None:
            raise ValueError("original source artifact cannot declare a parent digest")
        if self.accepted_parse.parent_sha256 != self.original.sha256:
            raise ValueError("accepted parse must bind to the exact original SHA-256")
        if self.recovered_page_parse is not None:
            if self.corpus is not RepairCorpus.BOOKS:
                raise ValueError("physical PDF recovery is limited to the Books producer")
            for artifact in (
                self.recovered_page_parse.baseline_parse,
                self.recovered_page_parse.recovered_pages_parse,
            ):
                if artifact.parent_sha256 != self.original.sha256:
                    raise ValueError("all recovery artifacts must bind the exact original SHA-256")
        if self.source_id != f"vo-rag-corpus-sha256:{self.original.sha256}":
            raise ValueError("source_id must be content-bound to the original SHA-256")
        if self.census.collection != self.target_collection:
            raise ValueError("census collection must equal explicit target collection")
        if self.writer_recipe.target_collection != self.target_collection:
            raise ValueError("writer recipe target must equal explicit target collection")
        for field_name, snapshot, digest in (
            (
                "source_envelope_snapshot",
                self.source_envelope_snapshot,
                self.source_envelope_snapshot_sha256,
            ),
            (
                "owned_artifact_snapshots",
                self.owned_artifact_snapshots,
                self.owned_artifact_snapshots_sha256,
            ),
            ("book_metadata", self.book_metadata, self.book_metadata_sha256),
            (
                "source_envelope_metadata",
                self.source_envelope_metadata,
                self.source_envelope_metadata_sha256,
            ),
            ("parser_metadata", self.parser_metadata, self.parser_metadata_sha256),
            (
                "canonical_source_snapshot",
                self.canonical_source_snapshot,
                self.canonical_source_snapshot_sha256,
            ),
            ("quality_warnings", self.quality_warnings, self.quality_warnings_sha256),
        ):
            if (snapshot is None) != (digest is None):
                raise ValueError(f"{field_name} snapshot and digest must be supplied together")
            if snapshot is not None:
                actual = hashlib.sha256(
                    json.dumps(
                        snapshot,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
                if actual != digest:
                    raise ValueError(f"{field_name} snapshot digest does not match")
        if self.source_envelope_snapshot is not None and (
            self.source_envelope_id is None
            or self.source_envelope_snapshot.get("id") != self.source_envelope_id
        ):
            raise ValueError("source envelope snapshot must bind its explicit ID")
        if self.source_envelope_snapshot is not None and (
            self.source_envelope_metadata != self.source_envelope_snapshot.get("metadata")
        ):
            raise ValueError("source envelope metadata must match its exact source snapshot")
        expected = {
            (RepairCorpus.BOOKS, PublicationClass.BOOKS_RAW, "vo_rag_books_raw_v1"),
            (RepairCorpus.BOOKS, PublicationClass.BOOKS_STERILE, "vo_rag_books_sterile_v1"),
            (
                RepairCorpus.BOOKS,
                PublicationClass.BOOKS_CONTEXTUALIZED,
                "vo_rag_books_contextualized_v1",
            ),
            (
                RepairCorpus.BOOKS,
                PublicationClass.BOOKS_GRAPH_NODES,
                "vo_rag_books_graph_nodes_v1",
            ),
            (
                RepairCorpus.BOOKS,
                PublicationClass.BOOKS_GRAPH_EDGES,
                "vo_rag_books_graph_edges_v1",
            ),
            (RepairCorpus.BOOKS, PublicationClass.BOOKS_DSI, "vo_rag_books_dsi_v1"),
            (RepairCorpus.REDDIT, PublicationClass.REDDIT, "vo_rag_reddit_v1"),
        }
        if (self.corpus, self.publication_class, self.target_collection) not in expected:
            raise ValueError(
                "corpus, publication class, and exact target collection are not allowed"
            )
        if self.corpus is RepairCorpus.REDDIT and self.reddit_lineage is None:
            raise ValueError("Reddit repair requires authoritative archive/thread/comment lineage")
        if self.corpus is RepairCorpus.BOOKS and self.reddit_lineage is not None:
            raise ValueError("Books repair cannot carry Reddit archive lineage")
        if self.record_role is not None:
            if self.schema_policy is None:
                raise ValueError("typed graph/DSI role requires explicit schema-owner policy")
            if self.schema_policy.record_role is not self.record_role:
                raise ValueError("schema policy role must equal evidence record_role")
        elif self.schema_policy is not None:
            raise ValueError("schema policy requires an explicit typed record_role")
        required_role = {
            PublicationClass.BOOKS_GRAPH_NODES: TypedRecordRole.GRAPH_NODE,
            PublicationClass.BOOKS_GRAPH_EDGES: TypedRecordRole.GRAPH_EDGE,
            PublicationClass.BOOKS_DSI: TypedRecordRole.DSI_RECORD,
        }.get(self.publication_class)
        if required_role is not None and self.record_role is not required_role:
            raise ValueError("graph/DSI collection requires its exact typed record role")
        if required_role is None and self.record_role is not None:
            raise ValueError("typed graph/DSI role cannot target a non-graph/DSI collection")
        if self.claim_kind is RepairClaimKind.SEMANTIC and self.semantic_gold is None:
            raise ValueError("semantic claims require independent human labels")
        if self.claim_kind is not RepairClaimKind.SEMANTIC and self.semantic_gold is not None:
            raise ValueError("human semantic labels must only accompany semantic claims")
        if (
            self.claim_kind is RepairClaimKind.DETERMINISTIC_SOURCE_FIDELITY
            and not self.source_fidelity
        ):
            raise ValueError(
                "deterministic source-fidelity claim requires byte-bound omission spans"
            )
        return self


class PreflightResult(EvidenceModel):
    """Machine-readable eligible or preserved/blocked evidence disposition."""

    eligible: bool
    disposition: TerminalDisposition
    blockers: tuple[str, ...] = ()
    checked_at: datetime

    @field_validator("checked_at")
    @classmethod
    def _aware_checked_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("checked_at must include a timezone")
        return value.astimezone(UTC)


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
    return digest.hexdigest(), byte_count


def load_reddit_lineage(
    archive: SourceArtifact,
    manifest: SourceArtifact,
    *,
    max_bytes: int = DEFAULT_REPAIR_BUDGET.max_evidence_bytes,
) -> RedditLineageEvidence:
    """Build Reddit occurrence evidence solely from a digest-bound archive manifest.

    The manifest schema is intentionally explicit. A permalink list or
    caller-guessed thread/comment identifiers do not establish lineage.
    """
    if archive.role is not SourceArtifactRole.REDDIT_ARCHIVE:
        raise ValueError("archive artifact must use reddit_archive role")
    if manifest.role is not SourceArtifactRole.REDDIT_LINEAGE_MANIFEST:
        raise ValueError("manifest artifact must use reddit_lineage_manifest role")
    if archive.size_bytes + manifest.size_bytes > max_bytes:
        raise ValueError("Reddit lineage artifacts exceed the finite read budget")
    verified_bytes: dict[SourceArtifactRole, bytes] = {}
    for artifact in (archive, manifest):
        with artifact.path.open("rb") as stream:
            if os.fstat(stream.fileno()).st_size != artifact.size_bytes:
                raise ValueError(f"{artifact.role.value} artifact hash or size mismatch")
            raw = stream.read(artifact.size_bytes + 1)
        if hashlib.sha256(raw).hexdigest() != artifact.sha256 or len(raw) != artifact.size_bytes:
            raise ValueError(f"{artifact.role.value} artifact hash or size mismatch")
        verified_bytes[artifact.role] = raw
    try:
        document = _RedditManifestDocument.model_validate_json(
            verified_bytes[SourceArtifactRole.REDDIT_LINEAGE_MANIFEST]
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError("invalid authoritative Reddit occurrence manifest") from exc
    if document.archive_sha256 != archive.sha256:
        raise ValueError("Reddit occurrence manifest does not bind the archive digest")
    path_fields = {
        "identity_basis",
        "archive_post_index",
        "archive_comment_path",
        "archive_record_sha256",
        "rendered_start_byte",
        "rendered_end_byte",
    }
    if document.schema_id == "vo-rag.reddit-occurrence-manifest.v2":
        from vo_rag.core.ingest.repair_reddit_lineage import verify_archive_tree_manifest

        verify_archive_tree_manifest(
            json.loads(verified_bytes[SourceArtifactRole.REDDIT_ARCHIVE]),
            [row.model_dump(mode="json") for row in document.occurrences],
        )
    elif any(row.identity_basis != "reddit_id" for row in document.occurrences):
        raise ValueError("archive tree identity requires the verified v2 manifest")
    occurrences: list[RedditArchiveOccurrence] = []
    for row in document.occurrences:
        values = row.model_dump(mode="json")
        if document.schema_id == "vo-rag.reddit-occurrence-manifest.v1":
            values = {key: value for key, value in values.items() if key not in path_fields}
        entry_bytes = json.dumps(
            values, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        occurrences.append(
            RedditArchiveOccurrence(
                **row.model_dump(),
                archive_sha256=archive.sha256,
                manifest_sha256=manifest.sha256,
                manifest_entry_sha256=hashlib.sha256(entry_bytes).hexdigest(),
                lineage_source="authoritative_archive_manifest",
            )
        )
    return RedditLineageEvidence(archive=archive, manifest=manifest, occurrences=tuple(occurrences))


def _blocked(*reasons: str, checked_at: datetime) -> PreflightResult:
    return PreflightResult(
        eligible=False,
        disposition=TerminalDisposition.BLOCKED_MISSING_EVIDENCE,
        blockers=tuple(dict.fromkeys(reasons)),
        checked_at=checked_at,
    )


def _json_fallback(value: object) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    raise TypeError(f"unsupported evidence JSON value: {type(value).__name__}")


def preflight_repair_evidence(
    evidence: RepairEvidence | dict[str, object] | str | bytes,
    *,
    checked_at: datetime | None = None,
) -> PreflightResult:
    """Rehash and reconcile one bounded repair unit; never mutate a corpus.

    Any missing, stale, ambiguous, mismatched, unsupported, or over-budget
    condition returns ``blocked_missing_evidence``. A zero-change unit returns
    ``preserved_no_write``. The live census is explicitly non-atomic, so this
    result must be followed by the caller's shared fence and fresh preflight.
    """
    now = checked_at or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("checked_at must include a timezone")
    now = now.astimezone(UTC)
    if not isinstance(evidence, RepairEvidence):
        try:
            if isinstance(evidence, str | bytes):
                evidence = RepairEvidence.model_validate_json(evidence)
            else:
                encoded = json.dumps(
                    evidence,
                    default=_json_fallback,
                )
                evidence = RepairEvidence.model_validate_json(encoded)
        except (TypeError, ValueError, ValidationError):
            return _blocked("evidence_schema_invalid_or_unsupported", checked_at=now)
    blockers: list[str] = []

    if not evidence.source_revision:
        blockers.append("authoritative_monotonic_source_revision_missing")
    if evidence.replacement_revision_sequence is None:
        blockers.append("approved_replacement_revision_sequence_missing")

    artifacts = [evidence.original, evidence.accepted_parse]
    if evidence.recovered_page_parse is not None:
        artifacts.extend(
            [
                evidence.recovered_page_parse.baseline_parse,
                evidence.recovered_page_parse.recovered_pages_parse,
            ]
        )
    if evidence.reddit_lineage is not None:
        artifacts.extend([evidence.reddit_lineage.archive, evidence.reddit_lineage.manifest])
    for artifact in artifacts:
        try:
            actual_hash, actual_size = _sha256_file(artifact.path)
        except OSError:
            blockers.append(f"artifact_unavailable:{artifact.role.value}")
            continue
        if actual_hash != artifact.sha256 or actual_size != artifact.size_bytes:
            blockers.append(f"artifact_hash_or_size_mismatch:{artifact.role.value}")

    age = (now - evidence.census.captured_at).total_seconds()
    if age < 0 or age > evidence.budget.max_wall_seconds:
        blockers.append("census_stale_or_from_future")
    skew = abs(
        (
            evidence.census.independent_count_captured_at - evidence.census.captured_at
        ).total_seconds()
    )
    if skew > evidence.budget.max_wall_seconds:
        blockers.append("independent_count_not_captured_with_census")
    if (
        evidence.census.evidence_bytes + evidence.evidence_snapshot_bytes
        > evidence.budget.max_evidence_bytes
    ):
        blockers.append("evidence_snapshot_over_budget")
    if evidence.writer_recipe.inference_requests_used > evidence.budget.inference_requests:
        blockers.append("inference_request_budget_exceeded")
    if evidence.writer_recipe.inference_tokens_used > evidence.budget.inference_tokens:
        blockers.append("inference_token_budget_exceeded")
    if not evidence.occurrence_inventory_complete:
        blockers.append("source_occurrence_inventory_incomplete")
    if not evidence.after_roster_complete:
        blockers.append("after_roster_incomplete")
    if evidence.census.atomic_snapshot is not False:
        blockers.append("snapshot_atomicity_not_explicitly_non_atomic")

    occurrence_ids = [item.occurrence_id for item in evidence.occurrences]
    if len(occurrence_ids) != len(set(occurrence_ids)):
        blockers.append("source_occurrence_ids_not_unique")
    known_occurrences = set(occurrence_ids)
    for occurrence in evidence.occurrences:
        if occurrence.source_id != evidence.source_id:
            blockers.append("source_occurrence_source_id_mismatch")
        if occurrence.artifact_sha256 not in {
            evidence.original.sha256,
            evidence.accepted_parse.sha256,
        }:
            blockers.append("source_occurrence_not_bound_to_original_or_parse")
        bound_artifact = next(
            (a for a in artifacts if a.sha256 == occurrence.artifact_sha256), None
        )
        if bound_artifact is None:
            continue
        if occurrence.end_byte > bound_artifact.size_bytes:
            blockers.append("source_occurrence_span_out_of_bounds")
            continue
        try:
            with bound_artifact.path.open("rb") as stream:
                stream.seek(occurrence.start_byte)
                occurrence_bytes = stream.read(occurrence.end_byte - occurrence.start_byte)
        except OSError:
            blockers.append("source_occurrence_span_unavailable")
            continue
        if len(occurrence_bytes) != occurrence.end_byte - occurrence.start_byte:
            blockers.append("source_occurrence_span_out_of_bounds")
        elif hashlib.sha256(occurrence_bytes).hexdigest() != occurrence.span_sha256:
            blockers.append("source_occurrence_span_hash_mismatch")

    if evidence.reddit_lineage is not None:
        reddit_ids = {row.occurrence_id for row in evidence.reddit_lineage.occurrences}
        if reddit_ids != known_occurrences:
            blockers.append("reddit_occurrences_not_exactly_bound_to_archive_manifest")
        if any(row.source_id != evidence.source_id for row in evidence.reddit_lineage.occurrences):
            blockers.append("reddit_archive_source_id_mismatch")
        source_occurrences = {row.occurrence_id: row for row in evidence.occurrences}
        for row in evidence.reddit_lineage.occurrences:
            bound_occurrence = source_occurrences.get(row.occurrence_id)
            if row.identity_basis == "archive_tree_path" and (
                bound_occurrence is None
                or bound_occurrence.kind != row.kind
                or bound_occurrence.artifact_sha256 != evidence.original.sha256
                or bound_occurrence.start_byte != row.rendered_start_byte
                or bound_occurrence.end_byte != row.rendered_end_byte
            ):
                blockers.append("reddit_archive_rendered_span_mapping_mismatch")
        try:
            reconstructed = load_reddit_lineage(
                evidence.reddit_lineage.archive,
                evidence.reddit_lineage.manifest,
                max_bytes=evidence.budget.max_evidence_bytes,
            )
            if reconstructed.occurrences != evidence.reddit_lineage.occurrences:
                blockers.append("reddit_occurrence_mapping_not_present_in_manifest")
        except (OSError, ValueError):
            blockers.append("reddit_archive_manifest_binding_invalid")

    census_by_id = {point.point_id: point for point in evidence.census.points}
    before_by_id = {point.point_id: point for point in evidence.before_points}
    after_ids = [point.point_id for point in evidence.after_points]
    if len(before_by_id) != len(evidence.before_points):
        blockers.append("before_point_ids_not_unique")
    if len(after_ids) != len(set(after_ids)):
        blockers.append("after_point_ids_not_unique")
    if set(census_by_id) != set(before_by_id):
        blockers.append("before_roster_does_not_cover_full_live_census")
    for point_id, live in census_by_id.items():
        mapped = before_by_id.get(point_id)
        if mapped is None:
            continue
        if mapped.source_occurrence_id != live.source_occurrence_id:
            blockers.append("before_source_occurrence_mapping_mismatch")
        if (
            mapped.payload_sha256 != live.payload_sha256
            or mapped.vectors != live.vectors
            or mapped.representations != live.representations
        ):
            blockers.append("before_payload_or_vector_fingerprint_mismatch")
    for point in (*evidence.before_points, *evidence.after_points):
        if point.source_occurrence_id not in known_occurrences:
            blockers.append("point_source_occurrence_missing_or_ambiguous")
        if point.disposition is PointDisposition.UNRESOLVED:
            blockers.append("point_disposition_unresolved")
    for point in evidence.after_points:
        if point.disposition is PointDisposition.REPLACE and point.content is None:
            blockers.append("replacement_content_unavailable_for_exact_hash_check")
        if point.disposition is PointDisposition.REPLACE and not {
            item.name for item in point.vectors
        } >= set(evidence.writer_recipe.required_modalities):
            blockers.append("replacement_missing_writer_required_modality")
        if point.disposition is PointDisposition.REPLACE and point.point_id in census_by_id:
            blockers.append("replacement_id_reuses_visible_live_point")
    if any(point.disposition is PointDisposition.REPLACE for point in evidence.after_points):
        if evidence.writer_recipe.embedding_request_recipe_sha256 is None:
            blockers.append("embedding_request_recipe_fingerprint_missing")
        if evidence.book_metadata is None or evidence.book_metadata_sha256 is None:
            blockers.append("canonical_book_metadata_snapshot_missing")
        if (
            evidence.source_envelope_metadata is None
            or evidence.source_envelope_metadata_sha256 is None
        ):
            blockers.append("source_envelope_metadata_snapshot_missing")
        if evidence.parser_metadata is None or evidence.parser_metadata_sha256 is None:
            blockers.append("parser_metadata_snapshot_missing")
        if (
            evidence.canonical_source_snapshot is None
            or evidence.canonical_source_snapshot_sha256 is None
        ):
            blockers.append("canonical_source_record_snapshot_missing")
        if evidence.quality_warnings is None or evidence.quality_warnings_sha256 is None:
            blockers.append("document_quality_warning_snapshot_missing")
        if evidence.trust_level is None:
            blockers.append("normal_writer_trust_decision_missing")
    if any(point.replaces_point_ids for point in evidence.before_points):
        blockers.append("before_points_cannot_declare_after_replacement_links")
    for point in evidence.after_points:
        if any(old_id not in census_by_id for old_id in point.replaces_point_ids):
            blockers.append("replacement_targets_point_outside_census")
    for old in evidence.before_points:
        if old.disposition is PointDisposition.REPLACE and not any(
            old.point_id in new.replaces_point_ids for new in evidence.after_points
        ):
            blockers.append("before_replacement_has_no_after_counterpart")
    for new in evidence.after_points:
        for old_id in new.replaces_point_ids:
            replaced = before_by_id.get(old_id)
            if replaced is None or replaced.disposition is not PointDisposition.REPLACE:
                blockers.append("after_replacement_link_has_no_replaced_before_point")

    changed_points = [p for p in evidence.after_points if p.disposition is PointDisposition.REPLACE]
    if len(changed_points) > evidence.budget.max_replacement_chunks:
        blockers.append("replacement_chunk_budget_exceeded")
    declared_replacement_bytes = sum(point.content_bytes for point in changed_points)
    if declared_replacement_bytes > evidence.budget.max_replacement_text_bytes:
        blockers.append("replacement_text_budget_exceeded")
    if evidence.writer_recipe.runtime_identity_status == "unknown":
        if evidence.writer_recipe.claims_historical_parity:
            blockers.append("unknown_runtime_cannot_claim_historical_parity")
        if evidence.claim_kind is not RepairClaimKind.DETERMINISTIC_SOURCE_FIDELITY:
            blockers.append("unknown_runtime_blocks_non_source_fidelity_repair")

    fidelity_artifacts = {a.role: a for a in artifacts}
    for span in evidence.source_fidelity:
        if span.occurrence_id not in known_occurrences:
            blockers.append("source_fidelity_occurrence_missing")
            continue
        fidelity_artifact = fidelity_artifacts.get(span.artifact_role)
        if fidelity_artifact is None or fidelity_artifact.sha256 != span.artifact_sha256:
            blockers.append("source_fidelity_artifact_binding_mismatch")
            continue
        try:
            with fidelity_artifact.path.open("rb") as stream:
                stream.seek(span.start_byte)
                actual_span = stream.read(span.end_byte - span.start_byte)
        except OSError:
            blockers.append("source_fidelity_span_unavailable")
            continue
        if len(actual_span) != span.end_byte - span.start_byte:
            blockers.append("source_fidelity_span_out_of_bounds")
        elif hashlib.sha256(actual_span).hexdigest() != span.omitted_span_sha256:
            blockers.append("source_fidelity_span_hash_mismatch")
        else:
            targets = [
                point
                for point in evidence.after_points
                if point.point_id in span.restored_in_point_ids
            ]
            if len(targets) != len(span.restored_in_point_ids):
                blockers.append("source_fidelity_replacement_point_missing")
            if any(
                point.disposition is not PointDisposition.REPLACE
                or point.source_occurrence_id != span.occurrence_id
                or point.content is None
                or actual_span not in point.content.encode("utf-8")
                for point in targets
            ):
                blockers.append("omitted_source_span_not_present_in_replacement_content")
    if (
        evidence.claim_kind is RepairClaimKind.DETERMINISTIC_SOURCE_FIDELITY
        and not evidence.source_fidelity
    ):
        blockers.append("deterministic_source_fidelity_evidence_missing")
    if evidence.claim_kind is RepairClaimKind.SEMANTIC and evidence.semantic_gold is None:
        blockers.append("human_semantic_gold_missing")
    if evidence.record_role is not None:
        policy = evidence.schema_policy
        if policy is None or policy.decision is TextRoleDecision.UNKNOWN or not policy.schema_owner:
            blockers.append("typed_record_text_policy_unknown_or_owner_missing")
        elif policy.decision is TextRoleDecision.EXPECTS_TEXT and not policy.evidence_sha256:
            blockers.append("typed_record_text_policy_evidence_missing")
        elif policy.decision is TextRoleDecision.DOES_NOT_EXPECT_TEXT:
            if changed_points:
                blockers.append("typed_record_role_is_not_text_bearing")
            else:
                return PreflightResult(
                    eligible=False,
                    disposition=TerminalDisposition.EXCLUDED_NOT_APPLICABLE,
                    blockers=("typed_record_role_is_not_text_bearing",),
                    checked_at=now,
                )

    # Preflight checks exact bytes and span hashes after the top-level file pass.
    # Occurrence byte-span hashes are not re-read here because they describe
    # lineage locators; deterministic omission evidence above is rehashed.
    if blockers:
        return _blocked(*blockers, checked_at=now)
    if not changed_points:
        return PreflightResult(
            eligible=False,
            disposition=TerminalDisposition.PRESERVED_NO_WRITE,
            blockers=("no_replacement_points_declared",),
            checked_at=now,
        )
    return PreflightResult(
        eligible=True,
        disposition=TerminalDisposition.ELIGIBLE,
        blockers=(),
        checked_at=now,
    )


__all__ = [
    "DEFAULT_REPAIR_BUDGET",
    "CensusPoint",
    "HumanSemanticGold",
    "LiveCensus",
    "NamedVectorFingerprint",
    "OccurrenceKind",
    "PointDisposition",
    "PointEvidence",
    "PreflightResult",
    "PublicationClass",
    "RecoveredPageParseEvidence",
    "RedditArchiveOccurrence",
    "RedditLineageEvidence",
    "RepairBudget",
    "RepairClaimKind",
    "RepairCorpus",
    "RepresentationFingerprint",
    "RepairEvidence",
    "SourceArtifact",
    "SourceArtifactRole",
    "SourceFidelityEvidence",
    "SourceOccurrence",
    "TerminalDisposition",
    "TextRoleDecision",
    "TypedRecordRole",
    "TypedRecordSchemaPolicy",
    "VectorFingerprint",
    "WriterRecipeEvidence",
    "is_local_endpoint",
    "load_reddit_lineage",
    "preflight_repair_evidence",
]

"""Track-and-field contextual tag schema (VOR-1443).

Allowed values, validation of model output, population inheritance and the
Qdrant payload a tagged chunk carries. Model output is untrusted: every value
must be in ``ALLOWED``; nothing is coerced.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Mapping
from types import MappingProxyType

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

TF_TAG_VERSION = 2
#: Served-model alias for the tagger; map it to a model on your OpenAI-compatible endpoint.
TF_TAGGER_MODEL = "tf-tagger"

_POP_LEVEL = frozenset(
    {"beginner", "youth", "high_school", "collegiate", "elite", "masters", "general", "not_stated"}
)
_POP_AGE = frozenset({"under_12", "12_14", "15_18", "adult", "masters", "not_stated"})
_POP_SEX = frozenset({"male", "female", "mixed", "not_stated"})

# "not_stated" and "none" mean "the text says nothing"; they are only meaningful
# alone, so a list that pairs one with a real value is a model error, not a
# permissive union that a query filter would read as a stated fact.
_SENTINELS = frozenset({"not_stated", "none"})

# Read-only event → family map: a MappingProxyType, not a dict, so the schema
# cannot be edited from outside this module. Annotated Mapping[str, str] because
# that is what a proxy is; read it, do not mutate it.
EVENT_FAMILY: Mapping[str, str] = MappingProxyType(
    {
        "60m": "sprint",
        "100m": "sprint",
        "200m": "sprint",
        "400m": "sprint",
        "100mh": "hurdles",
        "110mh": "hurdles",
        "400mh": "hurdles",
        "4x100": "relay",
        "4x400": "relay",
        "800m": "distance",
        "1500m": "distance",
        "steeple": "distance",
        "5000m_plus": "distance",
        "long_jump": "jumps",
        "triple_jump": "jumps",
        "high_jump": "jumps",
        "pole_vault": "jumps",
        "shot_put": "throws",
        "discus": "throws",
        "hammer": "throws",
        "javelin": "throws",
        "decathlon": "combined",
        "heptathlon": "combined",
    }
)

ALLOWED: dict[str, frozenset[str]] = {
    "tf_source_type": frozenset(
        {
            "textbook",
            "coaching_manual",
            "federation_manual",
            "training_methodology",
            "program_document",
            "popular_narrative",
        }
    ),
    "tf_evidence_type": frozenset({"empirical_research", "review", "coaching_practice", "opinion"}),
    "tf_book_athlete_level": _POP_LEVEL,
    "tf_book_age_group": _POP_AGE,
    "tf_book_sex": _POP_SEX,
    "tf_event_family": frozenset(
        {"sprint", "hurdles", "jumps", "throws", "distance", "relay", "combined", "general"}
    ),
    "tf_event": frozenset(EVENT_FAMILY) | {"none"},
    "tf_quality": frozenset(
        {
            "acceleration",
            "max_velocity",
            "speed_endurance",
            "special_endurance",
            "aerobic_endurance",
            "strength",
            "power",
            "reactive_strength",
            "technique",
            "mobility",
            "general_fitness",
            "none",
        }
    ),
    "tf_discipline": frozenset(
        {
            "biomechanics",
            "physiology",
            "psychology",
            "training_methodology",
            "injury_rehab",
            "nutrition",
            "testing",
            "history_narrative",
            "none",
        }
    ),
    "tf_training_phase": frozenset(
        {
            "general_prep",
            "specific_prep",
            "pre_competition",
            "competition",
            "transition",
            "multi_year",
            "not_stated",
        }
    ),
    "tf_athlete_level": _POP_LEVEL,
    "tf_age_group": _POP_AGE,
    "tf_sex": _POP_SEX,
    "tf_intervention": frozenset(
        {
            "drill",
            "session",
            "program",
            "technique_cue",
            "strength_exercise",
            "plyometric",
            "testing_protocol",
            "recovery",
            "periodization_model",
            "none",
        }
    ),
    "tf_claim_basis": frozenset(
        {"cites_research", "coach_experience", "textbook_principle", "anecdote", "mixed", "none"}
    ),
}

# Payload fields that get a Qdrant index (qdrant.BOOK_EXTRA_INDEXES reads this).
# The tf_book_* fields are book-level in ALLOWED but are copied onto every
# chunk's payload by chunk_payload (VOR-1443), so they are indexed too.
TF_INDEXED_FIELDS: dict[str, str] = {
    **dict.fromkeys(ALLOWED, "keyword"),
    "tf_pub_year": "integer",
    "tf_pub_decade": "keyword",
    "tf_population_source": "keyword",
    "tf_tag_version": "integer",
    "tf_tagger_model": "keyword",
    "tf_book_reviewed": "bool",
}


class TagValidationError(ValueError):
    """Model output that does not fit the schema."""


def _field_name(info: ValidationInfo) -> str:
    """``info.field_name``, which pydantic always sets for a field validator."""
    if info.field_name is None:
        raise ValueError("validator ran without a field name")
    return info.field_name


def _check(field: str, values: list[str]) -> list[str]:
    allowed = ALLOWED.get(field)
    if allowed is None:
        raise ValueError(f"{field}: no allowed values declared")
    if not values:
        raise ValueError(f"{field}: at least one value required")
    bad = [v for v in values if v not in allowed]
    if bad:
        raise ValueError(f"{field}: not allowed {bad}")
    unique = sorted(set(values))
    if len(unique) > 1 and not _SENTINELS.isdisjoint(unique):
        raise ValueError(f"{field}: {unique} mixes a not_stated/none value with real values")
    return unique


class BookTags(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tf_source_type: str
    tf_evidence_type: str
    tf_pub_year: int | None = Field(default=None, strict=True)
    tf_book_athlete_level: list[str]
    tf_book_age_group: list[str]
    tf_book_sex: str
    tf_limitations: str = Field(default="", max_length=500)

    @field_validator("tf_source_type", "tf_evidence_type", "tf_book_sex")
    @classmethod
    def _one(cls, v: str, info: ValidationInfo) -> str:
        return _check(_field_name(info), [v])[0]

    @field_validator("tf_book_athlete_level", "tf_book_age_group")
    @classmethod
    def _many(cls, v: list[str], info: ValidationInfo) -> list[str]:
        return _check(_field_name(info), v)

    @field_validator("tf_pub_year")
    @classmethod
    def _year(cls, v: int | None) -> int | None:
        if v is not None and not 1850 <= v <= 2030:
            raise ValueError(f"tf_pub_year: implausible {v}")
        return v

    @property
    def tf_pub_decade(self) -> str | None:
        return None if self.tf_pub_year is None else f"{self.tf_pub_year // 10 * 10}s"


class ChunkTags(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tf_event_family: list[str]
    tf_event: list[str]
    tf_quality: list[str]
    tf_discipline: list[str]
    tf_training_phase: list[str]
    tf_athlete_level: list[str]
    tf_age_group: list[str]
    tf_sex: str
    tf_intervention: list[str]
    tf_claim_basis: str

    @field_validator(
        "tf_event_family",
        "tf_event",
        "tf_quality",
        "tf_discipline",
        "tf_training_phase",
        "tf_athlete_level",
        "tf_age_group",
        "tf_intervention",
    )
    @classmethod
    def _many(cls, v: list[str], info: ValidationInfo) -> list[str]:
        return _check(_field_name(info), v)

    @field_validator("tf_sex", "tf_claim_basis")
    @classmethod
    def _one(cls, v: str, info: ValidationInfo) -> str:
        return _check(_field_name(info), [v])[0]

    @model_validator(mode="after")
    def _events_match_families(self) -> ChunkTags:
        if "general" in self.tf_event_family and self.tf_event != ["none"]:
            raise ValueError("tf_event must be ['none'] when tf_event_family is general")
        for event in self.tf_event:
            if event != "none" and EVENT_FAMILY[event] not in self.tf_event_family:
                raise ValueError(f"tf_event {event} needs tf_event_family {EVENT_FAMILY[event]}")
        return self


def validate_book(raw: object) -> BookTags:
    try:
        return BookTags.model_validate(raw)
    except ValidationError as exc:
        raise TagValidationError(str(exc)) from exc


def validate_batch(raw: object, expected_ids: list[str]) -> dict[str, ChunkTags]:
    """``raw`` is ``{chunk_id: tags}`` or ``[{"chunk_id": ..., **tags}]``; it must
    cover exactly ``expected_ids``."""
    if isinstance(raw, list):
        items: dict[str, object] = {}
        for index, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise TagValidationError(
                    f"batch entry {index} must be an object, got {type(entry).__name__}"
                )
            chunk_id = entry.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise TagValidationError(f"batch entry {index} has no chunk_id")
            if chunk_id in items:
                raise TagValidationError(f"duplicate chunk_id {chunk_id} in batch list")
            items[chunk_id] = entry
    elif isinstance(raw, dict):
        items = {str(k): v for k, v in raw.items()}
    else:
        raise TagValidationError("batch result must be an object or a list of objects")
    missing = sorted(set(expected_ids) - set(items))
    unexpected = sorted(set(items) - set(expected_ids))
    if missing or unexpected:
        raise TagValidationError(f"missing={missing} unexpected={unexpected}")
    out: dict[str, ChunkTags] = {}
    errors: list[str] = []
    for chunk_id, tags in items.items():
        try:
            out[chunk_id] = ChunkTags.model_validate(tags)
        except ValidationError as exc:
            errors.append(f"{chunk_id}: {exc.errors()[0]['msg']}")
    if errors:
        raise TagValidationError("; ".join(errors))
    return out


def chunk_payload(
    chunk: ChunkTags,
    book: BookTags,
    *,
    reviewed: bool,
    model: str = TF_TAGGER_MODEL,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The ``tf_*`` payload for one chunk: chunk tags, the book's record, and the
    book's population wherever the chunk states none.

    ``tf_population_source`` is ``chunk`` when the chunk states at least one of
    ``tf_athlete_level`` / ``tf_age_group`` / ``tf_sex`` (the unstated ones
    inherit the book's), and ``book`` when it states none. ``tf_pub_year`` and
    ``tf_pub_decade`` are absent when the book has no publication year.
    """
    stated = (
        chunk.tf_athlete_level != ["not_stated"]
        or chunk.tf_age_group != ["not_stated"]
        or chunk.tf_sex != "not_stated"
    )
    payload: dict[str, object] = chunk.model_dump()
    if chunk.tf_athlete_level == ["not_stated"]:
        payload["tf_athlete_level"] = list(book.tf_book_athlete_level)
    if chunk.tf_age_group == ["not_stated"]:
        payload["tf_age_group"] = list(book.tf_book_age_group)
    if chunk.tf_sex == "not_stated":
        payload["tf_sex"] = book.tf_book_sex
    payload.update(
        tf_population_source="chunk" if stated else "book",
        tf_source_type=book.tf_source_type,
        tf_evidence_type=book.tf_evidence_type,
        tf_limitations=book.tf_limitations,
        tf_book_reviewed=reviewed,
        tf_tag_version=TF_TAG_VERSION,
        tf_tagger_model=model,
        # The book's own population, copied onto every chunk so a chunk-level
        # filter on it needs no join back to Postgres.
        tf_book_athlete_level=list(book.tf_book_athlete_level),
        tf_book_age_group=list(book.tf_book_age_group),
        tf_book_sex=book.tf_book_sex,
    )
    if provenance:
        # Optional identities are stamped only when the classifier result carried
        # them. In particular, a historical row is never made to look like it was
        # produced by today's prompt or runtime during a re-stamp.
        for key in (
            "tf_tag_recipe_sha256",
            "tf_tag_prompt_sha256",
            "tf_tag_export_prompt_sha256",
            "tf_tag_job_sha256",
            "tf_tag_prompt_identity_status",
            "tf_tag_input_sha256",
            "tf_tag_input_identity_status",
            "tf_tag_source_id",
            "tf_tag_content_sha256",
            "tf_tag_source_id",
            "tf_tag_content_sha256",
            "tf_tag_source_id",
            "tf_tag_content_sha256",
            "tf_tagger_runtime_identity",
            "tf_tag_identity_status",
        ):
            if key in provenance:
                payload[key] = provenance[key]
    if book.tf_pub_year is not None:
        payload["tf_pub_year"] = book.tf_pub_year
        payload["tf_pub_decade"] = book.tf_pub_decade
    return payload


#: Invisible characters a parser may add or drop with no visible change: zero-width
#: space/joiners, word joiner, byte-order mark, soft hyphen, left/right-to-left
#: marks and the Mongolian vowel separator (VOR-1450).
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad\u200e\u200f\u180e"))


def content_sha256(text: str) -> str:
    """The key stored tags are found by: the chunk's text with Unicode compatibility
    forms folded (NFKC) and every whitespace and invisible character removed.

    A re-ingest re-parses the book, and the parser is free to reflow whitespace:
    one book came back with "decision-  making" as "decision-making" and every
    doubled space collapsed, so an exact-text hash matched none of its stored tags
    (VOR-1450). Ignoring whitespace keeps the words, and their order, as the key.
    """
    folded = "".join(unicodedata.normalize("NFKC", text).translate(_INVISIBLE).split())
    return hashlib.sha256(folded.encode("utf-8")).hexdigest()


def legacy_content_sha256(text: str) -> str:
    """The key tags were stored under before VOR-1450: the stripped text as-is.
    Only ``tf-chunks rekey`` uses it, to move those rows to ``content_sha256``."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

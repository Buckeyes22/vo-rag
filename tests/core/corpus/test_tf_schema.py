from __future__ import annotations

import hashlib

import pytest

from vo_rag.core.corpus.tf_schema import (
    ALLOWED,
    TF_INDEXED_FIELDS,
    TF_TAG_VERSION,
    TF_TAGGER_MODEL,
    BookTags,
    ChunkTags,
    TagValidationError,
    _check,
    chunk_payload,
    content_sha256,
    legacy_content_sha256,
    validate_batch,
    validate_book,
)

BOOK = {
    "tf_source_type": "coaching_manual",
    "tf_evidence_type": "coaching_practice",
    "tf_pub_year": 1995,
    "tf_book_athlete_level": ["high_school"],
    "tf_book_age_group": ["15_18"],
    "tf_book_sex": "mixed",
    "tf_limitations": "1990s methods.",
}

CHUNK = {
    "tf_event_family": ["sprint"],
    "tf_event": ["100m"],
    "tf_quality": ["acceleration"],
    "tf_discipline": ["training_methodology"],
    "tf_training_phase": ["specific_prep"],
    "tf_athlete_level": ["not_stated"],
    "tf_age_group": ["not_stated"],
    "tf_sex": "not_stated",
    "tf_intervention": ["drill"],
    "tf_claim_basis": "coach_experience",
}


def test_book_record_validates_and_derives_decade() -> None:
    book: BookTags = validate_book(BOOK)
    assert book.tf_pub_decade == "1990s"


def test_unknown_value_is_rejected_not_coerced() -> None:
    with pytest.raises(TagValidationError, match="tf_quality"):
        validate_batch({"c1": {**CHUNK, "tf_quality": ["speed"]}}, ["c1"])


def test_event_must_match_family() -> None:
    with pytest.raises(TagValidationError, match="110mh"):
        validate_batch({"c1": {**CHUNK, "tf_event": ["110mh"]}}, ["c1"])


def test_general_family_requires_event_none() -> None:
    ok = {**CHUNK, "tf_event_family": ["general"], "tf_event": ["none"]}
    assert validate_batch({"c1": ok}, ["c1"])["c1"].tf_event == ["none"]
    with pytest.raises(TagValidationError):
        validate_batch({"c1": {**ok, "tf_event": ["100m"]}}, ["c1"])


def test_batch_must_cover_exactly_the_expected_ids() -> None:
    with pytest.raises(TagValidationError, match="missing"):
        validate_batch({"c1": CHUNK}, ["c1", "c2"])
    with pytest.raises(TagValidationError, match="unexpected"):
        validate_batch({"c1": CHUNK, "cX": CHUNK}, ["c1"])


def test_batch_accepts_a_list_of_objects_with_chunk_id() -> None:
    out = validate_batch([{"chunk_id": "c1", **CHUNK}], ["c1"])
    assert isinstance(out["c1"], ChunkTags)


def test_payload_inherits_book_population_when_not_stated() -> None:
    payload = chunk_payload(
        validate_batch({"c1": CHUNK}, ["c1"])["c1"], validate_book(BOOK), reviewed=False
    )
    assert payload["tf_book_reviewed"] is False
    assert payload["tf_athlete_level"] == ["high_school"]
    assert payload["tf_age_group"] == ["15_18"]
    assert payload["tf_sex"] == "mixed"
    assert payload["tf_population_source"] == "book"
    assert payload["tf_tag_version"] == TF_TAG_VERSION
    assert payload["tf_source_type"] == "coaching_manual"
    assert payload["tf_pub_year"] == 1995


def test_payload_keeps_a_chunk_stated_population() -> None:
    youth = {**CHUNK, "tf_athlete_level": ["youth"], "tf_age_group": ["12_14"]}
    payload = chunk_payload(
        validate_batch({"c1": youth}, ["c1"])["c1"], validate_book(BOOK), reviewed=True
    )
    assert payload["tf_athlete_level"] == ["youth"]
    assert payload["tf_population_source"] == "chunk"


def test_content_hash_ignores_surrounding_whitespace() -> None:
    assert content_sha256("  abc\n") == content_sha256("abc")


def test_every_payload_field_that_is_indexed_is_declared() -> None:
    payload = chunk_payload(
        validate_batch({"c1": CHUNK}, ["c1"])["c1"], validate_book(BOOK), reviewed=False
    )
    assert set(TF_INDEXED_FIELDS) <= set(payload)
    assert "tf_limitations" not in TF_INDEXED_FIELDS
    undated = chunk_payload(
        validate_batch({"c1": CHUNK}, ["c1"])["c1"],
        validate_book({**BOOK, "tf_pub_year": None}),
        reviewed=False,
    )
    assert set(TF_INDEXED_FIELDS) - set(undated) == {"tf_pub_year", "tf_pub_decade"}


def _payload(raw: dict[str, object] | None = None) -> dict[str, object]:
    return chunk_payload(
        validate_batch({"c1": CHUNK if raw is None else raw}, ["c1"])["c1"],
        validate_book(BOOK),
        reviewed=False,
    )


def test_pub_year_rejects_coerced_numbers() -> None:
    for value in ("1995", 1995.0):
        with pytest.raises(TagValidationError):
            validate_book({**BOOK, "tf_pub_year": value})


def test_implausible_pub_year_is_rejected() -> None:
    with pytest.raises(TagValidationError, match="implausible"):
        validate_book({**BOOK, "tf_pub_year": 1700})


def test_empty_value_list_is_rejected() -> None:
    with pytest.raises(TagValidationError, match="at least one value"):
        validate_batch({"c1": {**CHUNK, "tf_quality": []}}, ["c1"])


def test_batch_rejects_a_non_dict_entry() -> None:
    with pytest.raises(TagValidationError, match="must be an object"):
        validate_batch([{"chunk_id": "c1", **CHUNK}, "c2"], ["c1"])


def test_batch_rejects_a_duplicate_chunk_id() -> None:
    with pytest.raises(TagValidationError, match="duplicate chunk_id c1"):
        validate_batch([{"chunk_id": "c1", **CHUNK}, {"chunk_id": "c1", **CHUNK}], ["c1"])


def test_batch_rejects_raw_that_is_neither_object_nor_list() -> None:
    with pytest.raises(TagValidationError, match="object or a list"):
        validate_batch("c1", ["c1"])


def test_population_source_is_chunk_when_any_one_field_is_stated() -> None:
    payload = _payload({**CHUNK, "tf_athlete_level": ["youth"]})
    assert payload["tf_athlete_level"] == ["youth"]
    assert payload["tf_age_group"] == ["15_18"]
    assert payload["tf_sex"] == "mixed"
    assert payload["tf_population_source"] == "chunk"


def test_payload_omits_pub_year_when_the_book_has_none() -> None:
    payload = chunk_payload(
        validate_batch({"c1": CHUNK}, ["c1"])["c1"],
        validate_book({**BOOK, "tf_pub_year": None}),
        reviewed=False,
    )
    assert "tf_pub_year" not in payload
    assert "tf_pub_decade" not in payload


def test_inherited_population_lists_are_copies() -> None:
    book = validate_book(BOOK)
    payload = chunk_payload(validate_batch({"c1": CHUNK}, ["c1"])["c1"], book, reviewed=False)
    inherited = payload["tf_athlete_level"]
    assert isinstance(inherited, list)
    assert inherited is not book.tf_book_athlete_level
    inherited.append("elite")
    assert book.tf_book_athlete_level == ["high_school"]


def test_book_limitations_and_tagger_model_reach_the_payload() -> None:
    payload = _payload()
    assert payload["tf_limitations"] == "1990s methods."
    assert payload["tf_tagger_model"] == TF_TAGGER_MODEL
    overridden = chunk_payload(
        validate_batch({"c1": CHUNK}, ["c1"])["c1"],
        validate_book(BOOK),
        reviewed=False,
        model="opencode/other",
    )
    assert overridden["tf_tagger_model"] == "opencode/other"


def test_every_schema_field_has_an_allowed_entry() -> None:
    for model in (BookTags, ChunkTags):
        for name in model.model_fields:
            if name in {"tf_pub_year", "tf_limitations"}:
                continue
            assert name in ALLOWED, name


def test_check_rejects_an_unknown_field_name() -> None:
    with pytest.raises(ValueError, match="no allowed values"):
        _check("tf_not_a_field", ["x"])


def test_not_stated_must_be_the_only_value_in_a_list() -> None:
    with pytest.raises(TagValidationError, match="tf_athlete_level"):
        validate_batch({"c1": {**CHUNK, "tf_athlete_level": ["not_stated", "youth"]}}, ["c1"])
    with pytest.raises(TagValidationError, match="tf_quality"):
        validate_batch({"c1": {**CHUNK, "tf_quality": ["none", "power"]}}, ["c1"])
    with pytest.raises(TagValidationError, match="tf_book_athlete_level"):
        validate_book({**BOOK, "tf_book_athlete_level": ["not_stated", "elite"]})


def test_a_lone_not_stated_or_none_still_validates() -> None:
    assert validate_batch({"c1": CHUNK}, ["c1"])["c1"].tf_athlete_level == ["not_stated"]
    general = {**CHUNK, "tf_event_family": ["general"], "tf_event": ["none"]}
    assert validate_batch({"c1": general}, ["c1"])["c1"].tf_event == ["none"]


def test_batch_rejects_a_list_entry_without_a_usable_chunk_id() -> None:
    for bad in ({**CHUNK}, {**CHUNK, "chunk_id": ""}, {**CHUNK, "chunk_id": 7}):
        with pytest.raises(TagValidationError, match="batch entry 0 has no chunk_id"):
            validate_batch([bad], ["c1"])


def test_limitations_over_500_characters_is_rejected() -> None:
    with pytest.raises(TagValidationError, match="tf_limitations"):
        validate_book({**BOOK, "tf_limitations": "x" * 501})
    # Exactly 500 is still fine.
    assert validate_book({**BOOK, "tf_limitations": "x" * 500}).tf_limitations == "x" * 500


def test_payload_carries_book_level_population_fields_for_indexing() -> None:
    """The book's population is copied onto every chunk (not just the chunk's own
    inherited/stated tf_athlete_level etc.), so a chunk-level filter on the book's
    population is possible without a join back to Postgres."""
    payload = chunk_payload(
        validate_batch({"c1": CHUNK}, ["c1"])["c1"], validate_book(BOOK), reviewed=False
    )
    assert payload["tf_book_athlete_level"] == ["high_school"]
    assert payload["tf_book_age_group"] == ["15_18"]
    assert payload["tf_book_sex"] == "mixed"


def test_book_level_population_fields_are_indexed() -> None:
    assert TF_INDEXED_FIELDS["tf_book_athlete_level"] == "keyword"
    assert TF_INDEXED_FIELDS["tf_book_age_group"] == "keyword"
    assert TF_INDEXED_FIELDS["tf_book_sex"] == "keyword"
    assert len(TF_INDEXED_FIELDS) == 21


def test_content_hash_ignores_whitespace_and_invisibles_a_parser_may_reflow() -> None:
    """A re-ingest parsed "decision-  making" as "decision-making" and collapsed
    doubled spaces, so no chunk's hash matched its stored tags (VOR-1450). The
    hash ignores whitespace, invisible characters and Unicode compatibility forms;
    the words and their order still count."""
    old = "British  Athletics  apply  a  shared  decision-  making\nprocess"
    new = "British Athletics apply a shared decision-making process"
    assert content_sha256(old) == content_sha256(new)
    assert content_sha256("ﬁnal  sprint") == content_sha256("final sprint")
    assert content_sha256("final sprint") != content_sha256("final sprints")
    assert content_sha256("a b") != content_sha256("b a")
    assert content_sha256("de\u00adcision\u200b making") == content_sha256("decision making")
    assert content_sha256("\u200erun\u200f\u180e") == content_sha256("run")


def test_the_legacy_hash_is_the_one_stored_before_vor_1450() -> None:
    assert legacy_content_sha256("  abc\n") == hashlib.sha256(b"abc").hexdigest()

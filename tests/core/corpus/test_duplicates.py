"""`vo_rag.core.corpus.duplicates` (VOR-1427).

Pure unit tests over `plan_duplicates` -- no Qdrant/Postgres. `BookStats`
values are built directly; `authors`/`ingested_at` are the plain dicts the
CLI's `duplicates --report` command would otherwise fetch from Postgres.
"""

from __future__ import annotations

from vo_rag.core.corpus.duplicates import DupRow, plan_duplicates
from vo_rag.core.corpus.health import BookStats
from vo_rag.core.corpus.titles import normalize_title


def _stats(
    source_id: str,
    source_book: str,
    chunks: int = 0,
    tiny: int = 0,
    boilerplate: int = 0,
) -> BookStats:
    return BookStats(
        source_id=source_id,
        source_book=source_book,
        chunks=chunks,
        tiny=tiny,
        boilerplate=boilerplate,
    )


def test_normalize_title_variants() -> None:
    assert normalize_title("101 garage performance projec_.reencoded") == normalize_title(
        "101 garage performance projec_"
    )
    assert normalize_title("Strengthcraft [12345678]") == normalize_title("Strengthcraft")


def test_normalize_title_apostrophe_punctuation() -> None:
    """Punctuation stripping (incl. apostrophes) folds these to the same key."""
    assert normalize_title("Workshop's Handbook") == normalize_title("Workshops Handbook")


def test_keep_rule_order() -> None:
    a = _stats("A", "Book X", chunks=900, tiny=500)  # degenerate copy
    b = _stats("B", "Book X.reencoded", chunks=800, tiny=0)
    c = _stats("C", "Book X", chunks=700, tiny=0)
    rows = {
        r.source_id: r
        for r in plan_duplicates([a, b, c], authors={}, ingested_at={"A": "1", "B": "2", "C": "3"})
    }
    assert rows["C"].action == "keep"  # clean and not .reencoded beats B's higher chunk count
    assert rows["A"].action == rows["B"].action == "delete"
    assert rows["A"].reason == "more degenerate/boilerplate chunks than keeper"
    assert rows["B"].reason == "reencoded twin"


def test_different_authors_not_grouped() -> None:
    rows = plan_duplicates(
        [_stats("A", "Handbook", 10, 0), _stats("B", "Handbook", 10, 0)],
        authors={"A": "Smith", "B": "Jones"},
        ingested_at={},
    )
    assert rows == []


def test_group_of_one_produces_no_rows() -> None:
    rows = plan_duplicates([_stats("A", "Lonely Book", 10, 0)], authors={}, ingested_at={})
    assert rows == []


def test_untitled_books_are_never_grouped_together() -> None:
    rows = plan_duplicates(
        [_stats("A", "", 10, 0), _stats("B", "", 10, 0), _stats("C", None, 10, 0)],  # type: ignore[arg-type]
        authors={},
        ingested_at={},
    )
    assert rows == []


def test_bad_chunks_is_tiny_plus_boilerplate() -> None:
    a = _stats("A", "Same Title", chunks=100, tiny=5, boilerplate=3)
    b = _stats("B", "Same Title", chunks=100, tiny=0, boilerplate=0)
    rows = {r.source_id: r for r in plan_duplicates([a, b], authors={}, ingested_at={})}
    assert rows["A"].bad_chunks == 8
    assert rows["B"].bad_chunks == 0
    assert rows["B"].action == "keep"
    assert rows["A"].action == "delete"
    assert rows["A"].reason == "more degenerate/boilerplate chunks than keeper"


def test_tie_on_bad_chunks_and_reencoded_breaks_on_chunk_count() -> None:
    a = _stats("A", "Same Title", chunks=100, tiny=0)
    b = _stats("B", "Same Title", chunks=200, tiny=0)
    rows = {r.source_id: r for r in plan_duplicates([a, b], authors={}, ingested_at={})}
    assert rows["B"].action == "keep"
    assert rows["A"].action == "delete"
    assert rows["A"].reason == "fewer chunks than keeper"


def test_final_tiebreak_is_earliest_ingested() -> None:
    a = _stats("A", "Same Title", chunks=100, tiny=0)
    b = _stats("B", "Same Title", chunks=100, tiny=0)
    rows = {
        r.source_id: r
        for r in plan_duplicates(
            [a, b], authors={}, ingested_at={"A": "2026-02-01", "B": "2026-01-01"}
        )
    }
    assert rows["B"].action == "keep"
    assert rows["A"].action == "delete"
    assert rows["A"].reason == "ingested later than keeper"


def test_missing_ingested_at_sorts_last() -> None:
    a = _stats("A", "Same Title", chunks=100, tiny=0)
    b = _stats("B", "Same Title", chunks=100, tiny=0)
    rows = {
        r.source_id: r for r in plan_duplicates([a, b], authors={}, ingested_at={"B": "2026-01-01"})
    }
    assert rows["B"].action == "keep"
    assert rows["A"].action == "delete"
    assert rows["A"].ingested_at is None


def test_author_matching_is_case_and_whitespace_insensitive() -> None:
    rows = plan_duplicates(
        [_stats("A", "Handbook", 10, 0), _stats("B", "Handbook", 10, 0)],
        authors={"A": "  Smith  ", "B": "smith"},
        ingested_at={},
    )
    assert len(rows) == 2


def test_different_years_are_not_grouped() -> None:
    """`normalize_title` folds "(YYYY)" away, so two
    different editions must not be silently merged into one duplicate group."""
    rows = plan_duplicates(
        [
            _stats("A", "Applied Strength Methods (2013)", 100, 0),
            _stats("B", "Applied Strength Methods (2019)", 100, 0),
        ],
        authors={},
        ingested_at={},
    )
    assert rows == []


def test_same_year_is_grouped() -> None:
    rows = plan_duplicates(
        [
            _stats("A", "Applied Strength Methods (2013)", 100, 0),
            _stats("B", "Applied Strength Methods (2013)", 90, 0),
        ],
        authors={},
        ingested_at={},
    )
    assert {r.source_id for r in rows} == {"A", "B"}


def test_no_year_on_either_side_is_grouped() -> None:
    rows = plan_duplicates(
        [
            _stats("A", "Applied Strength Methods", 100, 0),
            _stats("B", "Applied Strength Methods", 90, 0),
        ],
        authors={},
        ingested_at={},
    )
    assert {r.source_id for r in rows} == {"A", "B"}


def test_year_vs_no_year_is_grouped() -> None:
    """A missing year is not evidence of a different edition -- group them
    and let the operator's review of the CSV decide, rather than silently keeping
    a possible duplicate."""
    rows = plan_duplicates(
        [
            _stats("A", "Applied Strength Methods", 100, 0),
            _stats("B", "Applied Strength Methods (2019)", 90, 0),
        ],
        authors={},
        ingested_at={},
    )
    assert {r.source_id for r in rows} == {"A", "B"}


def test_bridge_year_ambiguity_never_deletes_across_two_editions() -> None:
    """A year-less book must not bridge two explicit, differing years into one
    group -- bridging would merge "Applied Strength Methods (2013)" and "... (2019)"
    via the year-less "Applied Strength Methods" and delete one of the two editions.
    Instead, the 2013 and 2019 singletons produce no rows at all (each is alone
    within its own exact-year subgroup), and the year-less book surfaces as
    its own `keep`/`ambiguous_year` row -- never `delete`."""
    rows = plan_duplicates(
        [
            _stats("A", "Applied Strength Methods", 100, 0),  # no year -- would-be bridge
            _stats("B", "Applied Strength Methods (2013)", 90, 0),
            _stats("C", "Applied Strength Methods (2019)", 80, 0),
        ],
        authors={},
        ingested_at={},
    )
    assert not any(row.action == "delete" for row in rows)
    by_id = {row.source_id: row for row in rows}
    assert set(by_id) == {"A"}
    assert by_id["A"].action == "keep"
    assert by_id["A"].reason == "ambiguous_year"


def test_one_explicit_year_plus_yearless_still_group_normally() -> None:
    """Exactly one distinct explicit year in the bucket is unambiguous --
    the year-less book still joins that single year's group and the normal
    keep rule (not `ambiguous_year`) decides keep/delete."""
    rows = plan_duplicates(
        [
            _stats("A", "Applied Strength Methods", chunks=100, tiny=50),  # degenerate, no year
            _stats("B", "Applied Strength Methods (2019)", chunks=90, tiny=0),
        ],
        authors={},
        ingested_at={},
    )
    by_id = {row.source_id: row for row in rows}
    assert set(by_id) == {"A", "B"}
    assert by_id["B"].action == "keep"
    assert by_id["A"].action == "delete"
    assert by_id["A"].reason == "more degenerate/boilerplate chunks than keeper"


def test_two_yearless_plus_two_different_years_are_ambiguous_and_kept() -> None:
    """Two distinct explicit years (2013, 2019), each a singleton, plus two
    year-less books: the singletons produce no rows, and both year-less
    books surface as their own `keep`/`ambiguous_year` rows -- never merged
    with each other under the normal keep rule, and never deleted."""
    rows = plan_duplicates(
        [
            _stats("A", "Applied Strength Methods", 100, 0),
            _stats("D", "Applied Strength Methods", 90, 0),
            _stats("B", "Applied Strength Methods (2013)", 80, 0),
            _stats("C", "Applied Strength Methods (2019)", 70, 0),
        ],
        authors={},
        ingested_at={},
    )
    assert not any(row.action == "delete" for row in rows)
    by_id = {row.source_id: row for row in rows}
    assert set(by_id) == {"A", "D"}
    assert by_id["A"].action == by_id["D"].action == "keep"
    assert by_id["A"].reason == by_id["D"].reason == "ambiguous_year"


def test_fully_tied_rows_break_on_source_id_regardless_of_input_order() -> None:
    """When every other sort key ties, `source_id`
    breaks the tie so the keeper doesn't depend on `collect_stats` order."""
    a = _stats("book-a", "Same Title", chunks=100, tiny=0)
    b = _stats("book-b", "Same Title", chunks=100, tiny=0)

    forward = {r.source_id: r for r in plan_duplicates([a, b], authors={}, ingested_at={})}
    backward = {r.source_id: r for r in plan_duplicates([b, a], authors={}, ingested_at={})}

    assert forward["book-a"].action == backward["book-a"].action == "keep"
    assert forward["book-b"].action == backward["book-b"].action == "delete"


def test_dup_row_is_a_plain_dataclass_with_expected_fields() -> None:
    row = DupRow(
        group="handbook",
        source_id="A",
        source_book="Handbook",
        chunks=1,
        bad_chunks=0,
        reencoded=False,
        ingested_at=None,
        action="keep",
        reason="x",
    )
    assert row.source_id == "A"


# --- ingest dates from intake_envelopes ---


class _EnvelopePool:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        self.calls.append((query, args))
        return self.rows


async def test_envelope_ingested_at_maps_sha_source_ids_to_the_earliest_completed_envelope() -> (
    None
):
    from datetime import UTC, datetime

    from vo_rag.core.corpus.duplicates import envelope_ingested_at

    new_sha, legacy_sha = "b" * 64, "a" * 64
    pool = _EnvelopePool([{"sha256": new_sha, "ingested_at": datetime(2026, 3, 1, tzinfo=UTC)}])
    result = await envelope_ingested_at(
        pool,
        [f"vo-rag-corpus-sha256:{new_sha}", legacy_sha.upper(), "not-a-hash-id"],
    )

    assert result == {
        f"vo-rag-corpus-sha256:{new_sha}": "2026-03-01 00:00:00+00:00",
        legacy_sha.upper(): None,
        "not-a-hash-id": None,
    }
    [(query, args)] = pool.calls
    assert args == (sorted([new_sha, legacy_sha]),)
    assert "JOIN intake_envelopes" in query
    assert "status = 'completed'" in query
    assert "MIN(" in query
    assert "raw_content->>'garage_key'" in query
    assert "source_data->>'garage_key'" in query


async def test_envelope_ingested_at_skips_the_query_without_hash_ids() -> None:
    from vo_rag.core.corpus.duplicates import envelope_ingested_at

    pool = _EnvelopePool([])
    assert await envelope_ingested_at(pool, ["x"]) == {"x": None}
    assert pool.calls == []


def test_book_authors_come_from_the_payload_author() -> None:
    from vo_rag.core.corpus.duplicates import book_authors

    stats = [
        BookStats(source_id="A", source_book="Handbook", author="Smith"),
        BookStats(source_id="B", source_book="Handbook"),
    ]
    assert book_authors(stats) == {"A": "Smith", "B": None}


def test_envelope_queries_share_one_sha_match() -> None:
    from vo_rag.core.corpus.duplicates import ENVELOPE_INGESTED_AT_SQL
    from vo_rag.core.corpus.source_format import SHA_ENVELOPE_MATCH, SOURCE_GARAGE_KEYS_SQL

    assert SHA_ENVELOPE_MATCH in ENVELOPE_INGESTED_AT_SQL
    assert SHA_ENVELOPE_MATCH in SOURCE_GARAGE_KEYS_SQL

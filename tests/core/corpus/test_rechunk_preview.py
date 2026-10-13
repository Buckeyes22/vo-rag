"""Pure metric logic for `vo-rag corpus rechunk-preview` (VOR-1452).

Every case here is a hand-built chunk list -- no Garage/Qdrant/Postgres.
"""

from __future__ import annotations

from types import SimpleNamespace

from vo_rag.core.corpus.rechunk_preview import (
    ChunkView,
    RawCounts,
    add_counts,
    carry_over,
    chunk_view_from_payload,
    chunk_view_from_retrieval,
    coverage_baseline,
    input_coverage,
    is_contents_text,
    junk_path,
    leaf,
    metrics,
    rates,
    raw_counts,
    sentence_coverage,
    word_coverage,
)


def _view(content: str, section_path: str = "", figure: bool = False) -> ChunkView:
    return ChunkView(content=content, section_path=section_path, figure=figure)


# ---------------------------------------------------------------------------
# leaf()
# ---------------------------------------------------------------------------


def test_leaf_takes_last_segment_and_strips_brackets() -> None:
    assert leaf("[Book > Chapter 1 > Index]") == "Index"
    assert leaf("Index") == "Index"
    assert leaf("") == ""


# ---------------------------------------------------------------------------
# frag: <300 chars, any chunk
# ---------------------------------------------------------------------------


def test_frag_boundary_299_counts_300_does_not() -> None:
    counts = raw_counts([_view("x" * 299), _view("x" * 300)])
    assert counts.frag == 1
    assert rates(counts)["frag"] == 0.5


# ---------------------------------------------------------------------------
# frag_text / tiny_text: body (non-figure) chunks only
# ---------------------------------------------------------------------------


def test_frag_text_ignores_figure_chunks() -> None:
    chunks = [_view("x" * 100, figure=False), _view("x" * 100, figure=True)]
    counts = raw_counts(chunks)
    assert counts.body == 1
    assert counts.frag_text == 1
    assert rates(counts)["frag_text"] == 1.0


def test_tiny_text_boundary_149_counts_150_does_not() -> None:
    chunks = [_view("x" * 149), _view("x" * 150)]
    counts = raw_counts(chunks)
    assert counts.tiny_text == 1
    assert rates(counts)["tiny_text"] == 0.5


def test_frag_text_and_tiny_text_denominator_is_body_not_all_chunks() -> None:
    # 1 tiny body chunk, 3 figure chunks: rate must be 1/1, not 1/4.
    chunks = [_view("x" * 10, figure=False)] + [_view("y" * 500, figure=True) for _ in range(3)]
    result = metrics(chunks)
    assert result["tiny_text"] == 1.0
    assert result["frag_text"] == 1.0
    assert result["figures"] == 3


# ---------------------------------------------------------------------------
# junk: back-matter label AND >300 chars
# ---------------------------------------------------------------------------


def test_junk_requires_both_label_match_and_over_300_chars() -> None:
    long_index = _view("x" * 301, section_path="Book > Index")
    short_index = _view("x" * 50, section_path="Book > Index")
    long_normal = _view("x" * 301, section_path="Book > Chapter 3")
    counts = raw_counts([long_index, short_index, long_normal])
    assert counts.junk == 1


def test_junk_matches_known_labels_case_insensitively() -> None:
    for label in (
        "Bibliography",
        "REFERENCES",
        "About the Author",
        "Table of Contents",
        "Acknowledgments",
        "Dedication",
    ):
        chunk = _view("x" * 400, section_path=f"Book > {label}")
        assert raw_counts([chunk]).junk == 1, label


def test_junk_does_not_match_an_ordinary_chapter_title() -> None:
    chunk = _view("x" * 400, section_path="Book > Chapter 3: Training")
    assert raw_counts([chunk]).junk == 0


# ---------------------------------------------------------------------------
# garb: letter ratio < 0.5
# ---------------------------------------------------------------------------


def test_garb_boundary_049_counts_050_does_not() -> None:
    # 49 letters, 51 digits -> ratio 0.49 (garbled)
    below = "a" * 49 + "1" * 51
    # 50 letters, 50 digits -> ratio 0.50 (not garbled: strictly less-than)
    at = "a" * 50 + "1" * 50
    counts = raw_counts([_view(below), _view(at)])
    assert counts.garb == 1


def test_garb_empty_content_is_not_garbled() -> None:
    assert raw_counts([_view("")]).garb == 0


# ---------------------------------------------------------------------------
# fig_bleed: figure content leaking Preceding:/Following:
# ---------------------------------------------------------------------------


def test_fig_bleed_only_counts_figure_chunks_with_neighbour_text() -> None:
    bled_figure = _view("Preceding: some text. Caption: a plot.", figure=True)
    clean_figure = _view("Caption: a plot.", figure=True)
    # A body chunk containing "Following:" is not a figure -> not fig_bleed.
    body_with_word = _view("Following: is a word used in prose.", figure=False)
    counts = raw_counts([bled_figure, clean_figure, body_with_word])
    assert counts.fig_bleed == 1


def test_fig_bleed_on_and_off() -> None:
    on = raw_counts([_view("Following: neighbour prose", figure=True)])
    off = raw_counts([_view("Caption only", figure=True)])
    assert on.fig_bleed == 1
    assert off.fig_bleed == 0


def test_fig_bleed_is_a_window_label_opening_a_line() -> None:
    """VOR-1454: the book's own "Pay attention to the following:" inside a
    figure's page context is no leaked neighbour window."""
    book_words = _view(
        "Page context: Speed Exercises. Pay Attention to the Following: begin with runs.",
        figure=True,
    )
    leaked = _view("Caption: a plot.\n\nFollowing: the next paragraph.", figure=True)
    assert raw_counts([book_words]).fig_bleed == 0
    assert raw_counts([leaked]).fig_bleed == 1


# ---------------------------------------------------------------------------
# path_quality: chunks whose section path has a junk component (VOR-1454)
# ---------------------------------------------------------------------------


def test_junk_path_looks_below_the_document_root() -> None:
    assert junk_path("[Book > WEEK LOAN > Errors during thrust]")
    assert junk_path("[Book > RW ALVG > The Discus Routine]")
    assert junk_path("[Book > DR SAM ELLIS > Learning the Turn]")
    assert not junk_path("[Book > Chapter 2 > Errors during thrust]")
    assert not junk_path("[WEEK LOAN]")  # the root is the document's own title
    assert not junk_path("[Book > pp. 221-230]")
    assert not junk_path("")
    # A heading's own citation bracket is not cut by the path's brackets.
    assert not junk_path("[Book > Chapter 3 > Bronze Medal Winner in Helsinki G. Velmorin [74]]")


def test_path_quality_is_the_share_of_chunks_with_a_junk_path() -> None:
    chunks = [
        _view("a" * 400, "[Book > PROPERTY OF > The Distance Races]"),
        _view("b" * 400, "[Book > The Distance Races]"),
        _view("c" * 400, "[Book > The Sprints]"),
        _view("d" * 400, "[Book > MASS IN POUNDS > Turns From The Ground]"),
    ]
    counts = raw_counts(chunks)
    assert counts.junk_path == 2
    assert rates(counts)["path_quality"] == 0.5


# ---------------------------------------------------------------------------
# junk: a real table of contents under "Contents" is back matter (VOR-1454)
# ---------------------------------------------------------------------------


def test_contents_under_contents_is_not_junk() -> None:
    toc = (
        "1.1. PERFORMANCE RESULT AND FACTORS DETERMINING ITS GROWTH..........28 "
        "1.2. DRILL ROLE IN THE TRAINING PROCESS..........34 1.3. SPECIFIC REGIMES........41"
    )
    foreword = (
        "I am delighted that Pat Marlow, former international athlete and now chief coach "
        "for the hurdles, has written this book for coaches and athletes. " * 3
    )
    counts = raw_counts([_view(toc * 2, "[Book > Contents]"), _view(foreword, "[Book > Contents]")])
    assert counts.back_matter == 1
    assert counts.junk == 1
    assert is_contents_text(toc)
    assert not is_contents_text(foreword)


# ---------------------------------------------------------------------------
# figures / chunks
# ---------------------------------------------------------------------------


def test_figures_and_chunks_counts() -> None:
    chunks = [_view("a", figure=True), _view("b", figure=True), _view("c", figure=False)]
    counts = raw_counts(chunks)
    assert counts.chunks == 3
    assert counts.figures == 2
    assert counts.body == 1


def test_metrics_on_empty_list_does_not_divide_by_zero() -> None:
    result = metrics([])
    assert result["chunks"] == 0
    assert result["frag"] == 0.0
    assert result["frag_text"] == 0.0


# ---------------------------------------------------------------------------
# total weighting: add_counts + rates is an exact micro-average, not an
# average of per-book rates (the prototype's bug for frag_text/tiny_text).
# ---------------------------------------------------------------------------


def test_total_weighting_is_exact_not_average_of_rounded_rates() -> None:
    # Book A: 1 tiny body chunk out of 1 body chunk (rate 1.0).
    book_a = raw_counts([_view("x" * 10, figure=False)])
    # Book B: 0 tiny body chunks out of 99 body chunks (rate 0.0).
    book_b = raw_counts([_view("x" * 500, figure=False) for _ in range(99)])
    total = add_counts(book_a, book_b)
    # Naive average of the two rates would give 0.5; the true weighted rate
    # (1 tiny out of 100 body chunks) is 0.01.
    assert rates(total)["tiny_text"] == 0.01


def test_add_counts_is_commutative_and_fieldwise() -> None:
    a = raw_counts([_view("x" * 10)])
    b = raw_counts([_view("y" * 400, section_path="Book > Index")])
    assert add_counts(a, b) == add_counts(b, a)
    assert add_counts(a, RawCounts()) == a


# ---------------------------------------------------------------------------
# sentence coverage: robust to de-duplication and whitespace reflow
# ---------------------------------------------------------------------------


def test_sentence_coverage_full_when_every_live_sentence_survives() -> None:
    live = [_view("This sentence has exactly six words.")]
    # Reflowed whitespace only (no punctuation change) so folding still
    # yields an exact substring match.
    new = [_view("This   sentence has exactly    six  words. Extra sentence follows here now.")]
    assert sentence_coverage(live, new) == 1.0


def test_sentence_coverage_ignores_short_fragments_under_six_words() -> None:
    # Under 6 words -> not counted at all, so there is nothing to cover
    # (matches the prototype's `sum(...) / (len(ls) or 1)`: 0/1, not a
    # vacuous "full coverage").
    live = [_view("See Fig. 3 above.")]
    new = [_view("Completely unrelated text with no overlap whatsoever here.")]
    assert sentence_coverage(live, new) == 0.0


def test_sentence_coverage_partial_when_a_live_sentence_is_dropped() -> None:
    live = [_view("This sentence has exactly six words. This other sentence also has six words.")]
    new = [_view("This sentence has exactly six words.")]
    assert sentence_coverage(live, new) == 0.5


def test_sentence_coverage_is_robust_to_whitespace_reflow() -> None:
    live = [_view("Doubled  spaces   and\ttabs should still match this sentence right.")]
    new = [_view("Doubled spaces and tabs should still match this sentence right.")]
    assert sentence_coverage(live, new) == 1.0


def test_sentence_coverage_robust_to_deduplication() -> None:
    # Live has the same long sentence twice (across two chunks); it should
    # count once, not twice, and full coverage when it survives once in new.
    sentence = "This particular sentence appears twice in the live payload data."
    live = [_view(sentence), _view(sentence)]
    new = [_view(sentence)]
    assert sentence_coverage(live, new) == 1.0


def test_sentence_coverage_empty_live_does_not_divide_by_zero() -> None:
    assert sentence_coverage([], [_view("anything")]) == 0.0


# ---------------------------------------------------------------------------
# word coverage
# ---------------------------------------------------------------------------


def test_word_coverage_counts_shared_word_occurrences() -> None:
    live = [_view("cat cat dog")]
    new = [_view("cat dog dog")]
    # live words: cat=2, dog=1 (total 3); covered = min(2,1)+min(1,2) = 1+1 = 2
    assert word_coverage(live, new) == round(2 / 3, 4)


def test_word_coverage_ignores_figure_chunks() -> None:
    live = [_view("alpha beta", figure=True)]
    new = [_view("nothing shared", figure=False)]
    # No live words counted at all (both live chunks are figures) -> 0/1, not
    # a vacuous full coverage.
    assert word_coverage(live, new) == 0.0


# ---------------------------------------------------------------------------
# carry_over: content_sha256 key reuse
# ---------------------------------------------------------------------------


def test_carry_over_counts_exact_content_reuse() -> None:
    live = [_view("Exact same text.")]
    new = [_view("Exact same text."), _view("Brand new text.")]
    assert carry_over(live, new) == 0.5


def test_carry_over_is_robust_to_whitespace_reflow() -> None:
    live = [_view("decision-  making   process")]
    new = [_view("decision-making process")]
    assert carry_over(live, new) == 1.0


def test_carry_over_empty_new_does_not_divide_by_zero() -> None:
    assert carry_over([_view("x")], []) == 0.0


# ---------------------------------------------------------------------------
# adapters: RetrievalChunk-like objects and Qdrant payload dicts
# ---------------------------------------------------------------------------


def test_chunk_view_from_retrieval_detects_figure_via_has_image() -> None:
    chunk = SimpleNamespace(content="c", section_path="s", has_image=True, chunk_type="text")
    view = chunk_view_from_retrieval(chunk)
    assert view == {"content": "c", "section_path": "s", "figure": True}


def test_chunk_view_from_retrieval_detects_figure_via_chunk_type() -> None:
    chunk = SimpleNamespace(
        content="c", section_path="s", has_image=False, chunk_type="figure_composite"
    )
    assert chunk_view_from_retrieval(chunk)["figure"] is True


def test_chunk_view_from_retrieval_text_chunk_is_not_a_figure() -> None:
    chunk = SimpleNamespace(content="c", section_path="s", has_image=False, chunk_type="text")
    assert chunk_view_from_retrieval(chunk)["figure"] is False


def test_chunk_view_from_payload_reads_qdrant_fields() -> None:
    payload = {"content": "c", "section_path": "s", "has_image": True, "chunk_type": "figure"}
    assert chunk_view_from_payload(payload) == {"content": "c", "section_path": "s", "figure": True}


def test_chunk_view_from_payload_defaults_missing_fields() -> None:
    assert chunk_view_from_payload({}) == {"content": "", "section_path": "", "figure": False}


def test_coverage_baseline_leaves_out_noise_body_chunks_only() -> None:
    noise = _view("zq xv qq zz vv xx kk jj ww")
    body = _view("The athlete sprints the full sixty metres at race pace today.")
    figure = _view("zq xv qq zz vv xx kk jj ww", figure=True)
    baseline = coverage_baseline([noise, body, figure], lambda text: text.startswith("zq"))
    assert baseline == [body, figure]


def test_sentence_coverage_ignores_gate_dropped_live_noise() -> None:
    noise = _view("zq xv qq zz vv xx kk jj ww zq xv qq zz vv.")
    body = _view("The athlete sprints the full sixty metres at race pace today.")
    new = [body]
    assert sentence_coverage([noise, body], new) == 0.5
    baseline = coverage_baseline([noise, body], lambda text: text.startswith("zq"))
    assert sentence_coverage(baseline, new) == 1.0


# ---------------------------------------------------------------------------
# Junk vs back_matter, input_cov
# ---------------------------------------------------------------------------

_INDEX_TEXT = (
    "abrasion 45\n\naxial bearing wear 54\n\nBrinell hardness test 87\n\n"
    "Brinell scale 61\n\nactive damping 22\n\nactive vertical force peak 121\n\n"
    "acute failure 34, 50\n\nalloy 57\n\nangle-angle diagrams 233-7; hub-rim 234\n\n"
    "axle 59-62; fracture of the shaft 60\n\nanterior mounting bracket 55, 88-90\n\n"
    "articulated joint 21, 44-5\n\nbushing failure 48\n\nbrace 68-71\n\n"
    "mechanics of load failure 1-31\n\ncast iron 18-19"
)
_REFERENCES_TEXT = (
    "Northrop, G.R., Quill, R.T. & Hartwell, G. (1993) Mapping of electrical tissue "
    "stimulation using imaging. Journal of Applied Mechanics 74, 532-537.\n\n"
    "Calder, G.M., Whitfield, S.C. & Marlow, D.K. (1995) Reliability of measurements "
    "of material strength. Materials and Methods 18, 593-600.\n\n"
    "Avery, R.M. (1988) Strength and its development. Sports Engineering 6, 146-168.\n\n"
    "Ellis, E. (1957) Relation between size of cells and their susceptibility to "
    "discharge. Laboratory Notes 126, 1345-1347."
)
_BODY_TEXT = (
    "The hurdler attacks the barrier with the lead leg while the trail leg folds under "
    "the hip. The athlete runs three strides between the barriers and keeps the hips "
    "tall through every clearance, landing on the ball of the foot and running away. "
    "Coaches should film the clearance from the side to see the lead-leg action."
)


def test_a_real_index_under_index_is_back_matter_not_junk() -> None:
    counts = raw_counts([_view(_INDEX_TEXT, section_path="Book > Index")])
    assert (counts.junk, counts.back_matter) == (0, 1)


def test_a_real_reference_list_under_references_is_back_matter_not_junk() -> None:
    counts = raw_counts([_view(_REFERENCES_TEXT, section_path="Book > References")])
    assert (counts.junk, counts.back_matter) == (0, 1)


def test_body_text_under_a_back_matter_label_is_still_junk() -> None:
    for label in ("Index", "References", "Bibliography"):
        counts = raw_counts([_view(_BODY_TEXT, section_path=f"Book > {label}")])
        assert (counts.junk, counts.back_matter) == (1, 0), label


def test_an_index_under_references_is_still_junk() -> None:
    """The content must match its own label: an index is not a reference list."""
    counts = raw_counts([_view(_INDEX_TEXT, section_path="Book > References")])
    assert (counts.junk, counts.back_matter) == (1, 0)


def test_input_coverage_finds_every_input_sentence_or_counts_it_lost() -> None:
    inputs = [
        "The hurdler attacks the barrier with the lead leg. The trail leg folds under the hip.",
        "Coaches should film the clearance from the side every week.",
    ]
    whole = [_view(" ".join(inputs))]
    assert input_coverage(inputs, whole) == 1.0
    lost = [_view(inputs[0])]
    assert input_coverage(inputs, lost) == round(2 / 3, 4)


def test_input_coverage_ignores_the_ocr_gate_drops_and_figure_chunks() -> None:
    inputs = ["Coaches should film the clearance from the side every week."]
    figure_only = [_view(inputs[0], figure=True)]
    assert input_coverage(inputs, figure_only) == 0.0  # a figure never stands in
    assert input_coverage(inputs, figure_only, ocr_dropped=[inputs[0]]) == 1.0


def test_input_coverage_joins_split_chunks_without_their_overlap_prefix() -> None:
    sentence = (
        "hand body-powered prosthesis 647-8 torque efficiency 624-5 velocity in swimming "
        "209, 210, 211 handball 367 heart rate 71 high jump 284-5 approach angle 286."
    )
    first = sentence[:90]
    second = sentence[60:]  # starts with a copy of the first chunk's tail
    assert input_coverage([sentence], [_view(first), _view(second)]) == 1.0


def test_input_coverage_leaves_letterless_runs_out() -> None:
    inputs = ["9 10 11 12 13 14 15 16 17 18", "Coaches should film the clearance every week."]
    assert input_coverage(inputs, [_view(inputs[1])]) == 1.0

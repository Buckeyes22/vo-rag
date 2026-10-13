from __future__ import annotations

from vo_rag.core.chunking.artifact_filters import strip_page_furniture_text
from vo_rag.core.chunking.counters import naive_counter
from vo_rag.core.chunking.section_packer import SectionPacker


def test_numeric_equations_survive_without_alphabetic_labels() -> None:
    """Odds tables and arithmetic remain source text at packing boundaries."""
    packer = SectionPacker(counter=naive_counter, target=50, max_tokens=60, overlap=0)
    for text in ("1/5 = 84%", "100/1 = 1%", "3 * 4 = 12", "0.030 = 0.150 / 5"):
        assert (
            packer.low_signal_element_text_reason(
                text, pages=[24], doc_title="Arithmetic guide", doc_total_pages=24
            )
            is None
        )
    for text in ("...", " "):
        assert packer.low_signal_text_reason(text) == "empty_after_normalize"
    assert packer.clean_text_for_append("24", 24, set()) == ""


def test_scalar_numeric_source_values_are_small_text_instead_of_noise() -> None:
    """Currency, decimal results and code output survive the signal decision."""
    packer = SectionPacker(counter=naive_counter, target=50, max_tokens=60, overlap=0)
    for text in ("42", "$10.00", "2.47", "= $1,331.93."):
        assert packer.low_signal_text_reason(text) in {None, "below_min_words"}
    assert packer.rendered_pending_heading("10 410.5024 cubic inches") == (
        "10 410.5024 cubic inches"
    )


def test_page_furniture_keeps_fraction_denominators_and_math_numeric_runs() -> None:
    """A plausible page number inside a wrapped expression is still math data."""
    for text in ("The cube of 245\n/\n8", "3\n2\n= – 14.8324 feet", "a, tan, ∠, log, θ\n4"):
        assert strip_page_furniture_text(text, [18]) == text
    assert strip_page_furniture_text("42", [22]) == "42"
    assert strip_page_furniture_text("Useful source paragraph.\n24", [24]) == (
        "Useful source paragraph."
    )


def _word_count(text: str) -> int:
    return len(text.split())


def test_section_packer_flushes_on_depth_one_title() -> None:
    packer = SectionPacker(counter=naive_counter, target=50, max_tokens=60, overlap=0)

    assert packer.should_flush_for_title(depth=1) is True
    assert packer.should_flush_for_title(depth=2) is True
    assert packer.should_flush_for_title(depth=3) is False


def test_section_packer_prefers_paragraph_boundaries_when_splitting() -> None:
    packer = SectionPacker(counter=_word_count, target=8, max_tokens=9, overlap=0)
    text = (
        "First paragraph has enough words to stand alone.\n\n"
        "Second paragraph should remain together when split."
    )

    parts = packer.split_text(text, "[Book]")

    assert parts == [
        "First paragraph has enough words to stand alone.",
        "Second paragraph should remain together when split.",
    ]


def test_section_packer_falls_back_to_sentence_boundaries() -> None:
    packer = SectionPacker(counter=_word_count, target=8, max_tokens=7, overlap=0)
    text = "First sentence has a budget. Second sentence also fits."

    parts = packer.split_text(text, "[Book]")

    assert parts == ["First sentence has a budget.", "Second sentence also fits."]


def test_math_operators_and_prime_units_are_mergeable_source_values() -> None:
    """Operators and angular prime units survive between formula fragments."""
    packer = SectionPacker(counter=naive_counter, target=50, max_tokens=60, overlap=0)
    for text in ("−", "–", "-", "/", "*", "′", "πD", "±----\n==", "---\n⎠", "() LX–"):
        assert packer.low_signal_text_reason(text) in {None, "below_min_words"}
        assert packer.rendered_pending_heading(text) == " ".join(text.split())
    assert packer.low_signal_text_reason("...") == "empty_after_normalize"


def test_assembled_body_does_not_reclassify_operands_as_page_furniture() -> None:
    """Body cleaning retains scalar values after element folios were removed."""
    packer = SectionPacker(counter=naive_counter, target=50, max_tokens=60, overlap=0)
    assert packer.clean_text_for_append("2", 12, set(), preserve_numeric_data=True) == "2"
    assert packer.clean_chunk_body("2\n\nA source answer.", [12], set()) == "2\n\nA source answer."

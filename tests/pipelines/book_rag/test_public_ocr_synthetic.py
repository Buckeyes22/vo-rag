"""Synthetic OCR cases; no private corpus or measured quality claims."""

from vo_rag.pipelines.book_rag.ocr_quality import assess, english_dictionary


def test_synthetic_prose_is_kept():
    text = (
        "This synthetic document explains how a service stores passages and ranks search results. "
    ) * 8
    assert assess(text) == "keep"


def test_bundled_dictionary_loads():
    assert "acceleration" in english_dictionary()

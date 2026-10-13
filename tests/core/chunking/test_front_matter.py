import pytest

from vo_rag.core.chunking.front_matter import front_matter_kind


@pytest.mark.parametrize(
    "heading,kind",
    [
        ("List of Symbols", "list_of_symbols"),
        ("Nomenclature", "list_of_symbols"),
        ("Table of Contents", "toc"),
        ("Contents", "toc"),
        ("Index", "index"),
        ("Bibliography", "bibliography"),
        ("References", "bibliography"),
        ("Glossary", "glossary"),
        ("About the Author", "about_author"),
        ("Acknowledgements", "acknowledgements"),
        ("4. Bibliography", "bibliography"),
        ("INDEX", "index"),
    ],
)
def test_front_matter_headings_classified(heading, kind):
    assert front_matter_kind(heading) == kind


@pytest.mark.parametrize(
    "heading",
    [
        "Aerodynamic Downforce",
        "The Understeer Angle",
        "Brake System Design",
        "15.3.4.2 The Marlowe Lateral Tire Force Model",
    ],
)
def test_body_headings_not_front_matter(heading):
    assert front_matter_kind(heading) is None

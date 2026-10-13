from vo_rag.core.chunking.chapter_path import derive_chapter_path


def test_numbered_prefix_aggregates():
    assert (
        derive_chapter_path(
            "[Analysis Techniques > 15.3.4.2 The Marlowe Model]",
            doc_title="Analysis Techniques",
        )
        == "[Analysis Techniques > Ch 15]"
    )
    assert (
        derive_chapter_path(
            "[Analysis Techniques > 15.1 Intro]",
            doc_title="Analysis Techniques",
        )
        == "[Analysis Techniques > Ch 15]"
    )


def test_top_segment_fallback():
    assert (
        derive_chapter_path(
            "[Practical Engine Tuning > SUSPENSION > ROD END BEARINGS]",
            doc_title="Practical Engine Tuning",
        )
        == "[Practical Engine Tuning > SUSPENSION]"
    )


def test_title_only():
    assert (
        derive_chapter_path("[Practical Engine Tuning]", doc_title="Practical Engine Tuning")
        == "[Practical Engine Tuning]"
    )


def test_no_brackets_input():
    assert (
        derive_chapter_path(
            "Analysis Techniques > 7.5 The Understeer Angle",
            doc_title="Analysis Techniques",
        )
        == "[Analysis Techniques > Ch 7]"
    )

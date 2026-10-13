from __future__ import annotations

from vo_rag.core.chunking.heading_tracker import HeadingTracker


def test_heading_tracker_skips_degenerate_headings_at_storage_and_lookup() -> None:
    tracker = HeadingTracker(doc_title="Book")

    assert tracker.update_title("1", depth=1) is False
    assert tracker.section_path() == "[Book]"
    assert tracker.section_title() == "Book"

    assert tracker.update_title("Chapter 1: Setup", depth=1) is True
    assert tracker.update_title("Downloaded from SAE International", depth=2) is False

    assert tracker.section_path() == "[Book > Chapter 1: Setup]"
    assert tracker.section_title() == "Chapter 1: Setup"


def test_heading_tracker_preserves_depth_and_returns_chapter_path() -> None:
    tracker = HeadingTracker(doc_title="Practical Engine Tuning")

    tracker.update_title("SUSPENSION", depth=1)
    tracker.update_title("ROD END BEARINGS", depth=2)

    assert tracker.section_path() == "[Practical Engine Tuning > SUSPENSION > ROD END BEARINGS]"
    assert tracker.section_title() == "ROD END BEARINGS"
    assert tracker.chapter_path() == "[Practical Engine Tuning > SUSPENSION]"


def test_heading_tracker_filters_pending_heading_fragments() -> None:
    tracker = HeadingTracker(doc_title="Book")

    assert tracker.section_path(["Overview", "Q&A", "Definitions"]) == (
        "[Book > Overview > Definitions]"
    )
    assert tracker.section_title(["Overview", "Q&A", "Definitions"]) == "Definitions"

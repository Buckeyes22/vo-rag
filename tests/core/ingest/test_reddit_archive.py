"""Reddit archive -> Markdown conversion (VOR-1404).

The yars scrapes are lists of posts shaped
``{title, author, score, num_comments, created, permalink, url, body, comments[]}``
with ``comments[].replies[]`` nested arbitrarily deep. JSON is not an ingestible
format for vo-rag, so each post becomes one Markdown document that rides the
existing file_drop -> book_rag path.

The load-bearing behaviours tested here:
- header metadata survives (permalink especially -- it is how a retrieved
  passage gets traced back to the thread)
- the comment tree flattens with depth preserved
- short comments are dropped WITHOUT orphaning substantive replies underneath
- real-world grubbiness (empty scores, empty bodies, zero-width unicode)
  does not crash the converter
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from vo_rag.core.ingest.reddit_archive import (
    is_recurring_megathread,
    post_to_markdown,
    reddit_post_filename,
    subreddit_from_permalink,
)

_PERMALINK = "/r/algobetting/comments/g5hi6j/creating_a_collection_of_resources/"


def _post(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "title": "Creating a collection of resources",
        "author": "example_user",
        "score": "187",
        "num_comments": "24",
        "created": "2020-04-21T15:39:36+00:00",
        "permalink": _PERMALINK,
        "url": f"https://www.reddit.com{_PERMALINK}",
        "body": "Please post any resources that have helped you.",
        "comments": [],
    }
    base.update(overrides)
    return base


def _comment(
    body: str, *, author: str = "someone", score: str = "5", replies: Any = None
) -> dict[str, Any]:
    return {"author": author, "score": score, "body": body, "replies": replies or []}


def test_header_carries_the_metadata_needed_to_trace_a_passage() -> None:
    md = post_to_markdown(_post(), subreddit="algobetting")

    assert "Creating a collection of resources" in md
    assert "algobetting" in md
    assert "example_user" in md
    assert "187" in md
    assert "2020-04-21T15:39:36+00:00" in md
    assert _PERMALINK in md


def test_post_body_is_included() -> None:
    md = post_to_markdown(_post(), subreddit="algobetting")

    assert "Please post any resources that have helped you." in md


def test_comments_render_with_author_and_score() -> None:
    post = _post(
        comments=[
            _comment(
                "A genuinely substantive comment about Poisson models.",
                author="stats_guy",
                score="12",
            )
        ]
    )

    md = post_to_markdown(post, subreddit="algobetting")

    assert "stats_guy" in md
    assert "12" in md
    assert "A genuinely substantive comment about Poisson models." in md


def test_nested_replies_are_indented_by_depth() -> None:
    post = _post(
        comments=[
            _comment(
                "Top level comment that is definitely long enough to survive.",
                replies=[
                    _comment(
                        "A nested reply that is also comfortably long enough to survive.",
                        replies=[
                            _comment("A third-level reply, also long enough to be kept here.")
                        ],
                    )
                ],
            )
        ]
    )

    md = post_to_markdown(post, subreddit="algobetting")
    lines = {
        line.lstrip(): len(line) - len(line.lstrip()) for line in md.splitlines() if line.strip()
    }
    depths = [
        indent
        for text, indent in lines.items()
        if "Top level comment" in text or "A nested reply" in text or "A third-level reply" in text
    ]

    assert len(depths) == 3
    assert depths[0] < depths[1] < depths[2]


def test_short_comments_are_dropped_but_their_replies_survive() -> None:
    """A one-word comment is noise; a substantive reply under it is not.

    36% of r/sportsbook comments are under 40 chars. Dropping the parent must
    not take a real answer down with it.
    """
    post = _post(
        comments=[
            _comment(
                "tailing",
                replies=[
                    _comment("Here is the actual reasoning, which is long enough to be kept.")
                ],
            )
        ]
    )

    md = post_to_markdown(post, subreddit="algobetting", min_comment_chars=40)

    assert "tailing" not in md
    assert "Here is the actual reasoning, which is long enough to be kept." in md


def test_min_comment_chars_zero_keeps_everything() -> None:
    post = _post(comments=[_comment("lol")])

    md = post_to_markdown(post, subreddit="algobetting", min_comment_chars=0)

    assert "lol" in md


def test_empty_body_link_post_still_renders_its_comments() -> None:
    """632 posts have no body -- the comments are the entire content."""
    post = _post(
        body="", comments=[_comment("The comments carry all of the value on this link post.")]
    )

    md = post_to_markdown(post, subreddit="sportsbook")

    assert "The comments carry all of the value on this link post." in md
    assert "Creating a collection of resources" in md


def test_blank_scores_and_missing_keys_do_not_crash() -> None:
    """Real records carry score='' and sometimes omit keys entirely."""
    post = {
        "title": "Sparse record",
        "permalink": _PERMALINK,
        "comments": [
            {"body": "A comment with no author and no score at all, long enough to keep."}
        ],
    }

    md = post_to_markdown(post, subreddit="algobetting")

    assert "Sparse record" in md
    assert "A comment with no author and no score at all, long enough to keep." in md


def test_zero_width_unicode_is_preserved_not_mangled() -> None:
    post = _post(body="Poisson​ distribution")

    md = post_to_markdown(post, subreddit="algobetting")

    assert "Poisson​ distribution" in md


def test_subreddit_extracted_from_permalink() -> None:
    assert subreddit_from_permalink(_PERMALINK) == "algobetting"
    assert subreddit_from_permalink("/r/sportsbook/comments/abc123/x/") == "sportsbook"


def test_subreddit_from_unparseable_permalink_is_empty() -> None:
    assert subreddit_from_permalink("") == ""
    assert subreddit_from_permalink("nonsense") == ""


def test_filename_is_slugified_and_carries_the_post_id() -> None:
    name = reddit_post_filename(_post(), subreddit="algobetting")

    assert name.endswith(".md")
    assert "g5hi6j" in name  # the reddit post id keeps filenames unique
    assert "algobetting" in name
    assert " " not in name


def test_filenames_differ_for_same_title_different_post() -> None:
    a = reddit_post_filename(_post(), subreddit="algobetting")
    b = reddit_post_filename(
        _post(permalink="/r/algobetting/comments/zzzzzz/creating_a_collection_of_resources/"),
        subreddit="algobetting",
    )

    assert a != b


# ----------------------------------------------------------------------
# Recurring megathread detection. r/sportsbook posts a fresh pick thread
# every day; 1018 of them are 64% of the archive's bytes and their content
# is bot rules plus "Record: (107-65-11) [+25.71]" pick logging. They are
# excluded by default at conversion time so they cannot crowd out real
# discussion in the top-k.
#
# The predicate is keyword AND date: the date anchor is what keeps
# genuinely event-specific threads ("FIGHT 330 ... Betting and Picks").
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "title",
    [
        "Pick of the Day 10/25/25 (Saturday)",
        "Flex and Cope Daily - 8/13/26 (Thursday)",
        "CFL Betting and Picks - 8/13/26 (Thursday)",
        "Sportsbook and Apps Promos Bonuses Daily Discussion - 8/13/26 (Thursday)",
        "KBO/NPB/CPBL Betting and Picks Daily Discussion 8/9/25",
        "Reddit Daily Picks 8/9",
        "pick of the day 1/2/25",  # case-insensitive
    ],
)
def test_dated_recurring_megathreads_are_detected(title: str) -> None:
    assert is_recurring_megathread(title) is True


@pytest.mark.parametrize(
    "title",
    [
        # Event-specific threads share the keyword but carry no date -- keep them.
        "FIGHT 330: Alder vs. Quill Betting and Picks Discussion",
        "CAGE 54 : Marlow vs Northrop Betting and Picks Discussion",
        "LEAGUE New York: Calder vs. Hartwell Betting and Picks",
        # Ordinary discussion, including one that merely mentions a date.
        "How do you calculate closing line value?",
        "My model's results through 8/13 -- what am I doing wrong?",
        "Creating a collection of resources to introduce beginners",
        "",
    ],
)
def test_real_discussion_is_not_treated_as_a_megathread(title: str) -> None:
    assert is_recurring_megathread(title) is False


# ----------------------------------------------------------------------
# Blank-line separation. The text parser splits a document into elements
# on blank lines (core/parsing/router.py: re.split(r"\n\s*\n", text)), and
# the chunker packs *elements*. Without a blank line between comments the
# entire comment tree is one paragraph -> one element the packer cannot
# subdivide. A 2026-08-14 pilot produced a single 61,520-char (~15k token)
# chunk that way, past BGE-M3's 8192-token limit, so most of it was never
# embedded at all.
# ----------------------------------------------------------------------


def test_comments_are_separated_by_blank_lines_so_they_chunk_independently() -> None:
    post = _post(
        comments=[
            _comment("First comment, comfortably long enough to be retained here."),
            _comment("Second comment, also comfortably long enough to be retained."),
            _comment("Third comment, likewise long enough to survive the filter."),
        ]
    )

    md = post_to_markdown(post, subreddit="algobetting")
    paragraphs = [p for p in re.split(r"\n\s*\n", md) if p.strip()]
    bodies = [p for p in paragraphs if "comment," in p]

    assert len(bodies) == 3, "each comment must be its own paragraph/element"


def test_nested_replies_are_also_separate_paragraphs() -> None:
    post = _post(
        comments=[
            _comment(
                "Parent comment that is definitely long enough to be kept in place.",
                replies=[_comment("Nested reply that is also long enough to be kept in place.")],
            )
        ]
    )

    md = post_to_markdown(post, subreddit="algobetting")
    paragraphs = [p for p in re.split(r"\n\s*\n", md) if p.strip()]

    assert any("Parent comment" in p and "Nested reply" not in p for p in paragraphs)
    assert any("Nested reply" in p and "Parent comment" not in p for p in paragraphs)


def test_multiline_comment_body_stays_in_one_paragraph() -> None:
    """A comment's own wrapped lines must NOT be split apart -- only comments are."""
    post = _post(comments=[_comment("Line one of the comment body.\nLine two of the same body.")])

    md = post_to_markdown(post, subreddit="algobetting")
    paragraphs = [p for p in re.split(r"\n\s*\n", md) if p.strip()]

    assert any("Line one" in p and "Line two" in p for p in paragraphs)

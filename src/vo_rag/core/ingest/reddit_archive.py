"""Convert yars Reddit archive records into Markdown documents (VOR-1404).

The archives are JSON lists of posts shaped ``{title, author, score,
num_comments, created, permalink, url, body, comments[]}``, with
``comments[].replies[]`` nested arbitrarily deep. JSON is not an ingestible
format for vo-rag (``ParserRouter._EXT_MAP`` stops at pdf/docx/epub/txt/md), so
each post is rendered as one Markdown document and ingested through the
existing ``file_drop`` -> ``book_rag`` path.

One post = one document is deliberate: a retrieved chunk then sits inside a
single thread, and the permalink in the header travels with it so an answer can
be traced back to the conversation it came from.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Final

# 36% of r/sportsbook comments are under 40 characters ("tailing", "good luck")
# but they carry only ~4% of the text. Dropping them is close to pure noise
# removal; their replies are promoted rather than lost (see _iter_comments).
DEFAULT_MIN_COMMENT_CHARS: Final = 40

# r/sportsbook posts a fresh pick thread every single day. 1018 of them carry
# 64% of the archive's bytes, and their content is bot rules plus pick logging
# ("Record: (107-65-11) [+25.71]") -- transactional, not discussion. Left in,
# they crowd real answers out of the top-k.
#
# Detection is keyword AND date. The date anchor is load-bearing: without it
# this also swallows genuinely event-specific threads such as
# "FIGHT 330: Alder vs. Quill Betting and Picks Discussion".
_MEGATHREAD_KEYWORD_RE: Final = re.compile(
    r"pick of the day|potd|daily discussion|daily picks|daily thread"
    r"|flex and cope|betting and picks|game thread|what are your picks"
    r"|daily free talk",
    re.IGNORECASE,
)
_MEGATHREAD_DATE_RE: Final = re.compile(r"\d{1,2}\s*[/-]\s*\d{1,2}")

_SUBREDDIT_RE: Final = re.compile(r"^/r/([^/]+)/")
_POST_ID_RE: Final = re.compile(r"^/r/[^/]+/comments/([^/]+)/")
_SLUG_RE: Final = re.compile(r"[^a-z0-9]+")
_MAX_SLUG_CHARS: Final = 60
_INDENT: Final = "  "


def is_recurring_megathread(title: str) -> bool:
    """Whether ``title`` looks like a dated, recurring pick/daily megathread.

    Requires a megathread keyword *and* a date, so one-off event threads
    ("FIGHT 330 ... Betting and Picks") are kept while "CFL Betting and Picks -
    8/13/26 (Thursday)" is not.
    """
    text = title or ""
    return bool(_MEGATHREAD_KEYWORD_RE.search(text) and _MEGATHREAD_DATE_RE.search(text))


def subreddit_from_permalink(permalink: str) -> str:
    """``/r/algobetting/comments/g5hi6j/...`` -> ``algobetting`` ("" if unparseable)."""
    match = _SUBREDDIT_RE.match(permalink or "")
    return match.group(1) if match else ""


def post_id_from_permalink(permalink: str) -> str:
    """``/r/algobetting/comments/g5hi6j/...`` -> ``g5hi6j`` ("" if unparseable)."""
    match = _POST_ID_RE.match(permalink or "")
    return match.group(1) if match else ""


def reddit_post_filename(post: Mapping[str, Any], *, subreddit: str) -> str:
    """Stable, collision-free filename: ``<subreddit>-<post_id>-<title-slug>.md``.

    The reddit post id is what makes it unique -- titles repeat constantly
    (every daily thread shares one).
    """
    parts = [
        _slug(subreddit),
        post_id_from_permalink(_text(post.get("permalink"))),
        _slug(_text(post.get("title"))),
    ]
    stem = "-".join(part for part in parts if part) or "post"
    return f"{stem}.md"


def post_to_markdown(
    post: Mapping[str, Any],
    *,
    subreddit: str,
    min_comment_chars: int = DEFAULT_MIN_COMMENT_CHARS,
) -> str:
    """Render one post plus its comment tree as a Markdown document."""
    lines: list[str] = [f"# {_text(post.get('title')) or '(untitled)'}", ""]
    lines.extend(_header_lines(post, subreddit=subreddit))
    lines.append("")

    body = _text(post.get("body"))
    if body:
        lines.extend([body, ""])

    comments = list(_iter_comments(post.get("comments"), 0, min_comment_chars))
    if comments:
        lines.extend(["## Comments", ""])
        for depth, comment in comments:
            lines.extend(_comment_lines(comment, depth))
            # Blank line between comments is load-bearing, not cosmetic: the text
            # parser splits a document into elements on blank lines
            # (core/parsing/router.py) and the chunker packs elements. Run them
            # together and the whole comment tree becomes one element the packer
            # cannot subdivide -- a 2026-08-14 pilot produced a single 61,520-char
            # (~15k token) chunk, well past BGE-M3's 8192-token limit, so most of
            # it was silently truncated out of its own embedding.
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _header_lines(post: Mapping[str, Any], *, subreddit: str) -> list[str]:
    author = _text(post.get("author"))
    fields = (
        ("subreddit", f"r/{subreddit}" if subreddit else ""),
        ("author", f"u/{author}" if author else ""),
        ("score", _text(post.get("score"))),
        ("num_comments", _text(post.get("num_comments"))),
        ("created", _text(post.get("created"))),
        ("permalink", _text(post.get("permalink"))),
        ("url", _text(post.get("url"))),
    )
    return [f"- {key}: {value}" for key, value in fields if value]


def _comment_lines(comment: Mapping[str, Any], depth: int) -> list[str]:
    indent = _INDENT * depth
    author = _text(comment.get("author")) or "unknown"
    score = _text(comment.get("score"))
    attribution = f"u/{author}" + (f" ({score})" if score else "")

    body_lines = _text(comment.get("body")).splitlines() or [""]
    head, *rest = body_lines
    lines = [f"{indent}- **{attribution}**: {head}"]
    lines.extend(f"{indent}{_INDENT}{line}" if line.strip() else "" for line in rest)
    return lines


def _iter_comments(
    comments: Any,
    depth: int,
    min_chars: int,
) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """Walk the comment tree depth-first, yielding ``(depth, comment)``.

    A comment shorter than ``min_chars`` is dropped, but its replies are
    **promoted to its depth** rather than discarded -- a one-word comment is
    noise, a substantive answer underneath it is not.
    """
    if not isinstance(comments, Sequence) or isinstance(comments, str | bytes):
        return
    for comment in comments:
        if not isinstance(comment, Mapping):
            continue
        replies = comment.get("replies")
        if len(_text(comment.get("body"))) >= min_chars:
            yield depth, comment
            yield from _iter_comments(replies, depth + 1, min_chars)
        else:
            yield from _iter_comments(replies, depth, min_chars)


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", (text or "").lower()).strip("-")[:_MAX_SLUG_CHARS].strip("-")


def _text(value: Any) -> str:
    """Coerce a possibly-missing, possibly-blank archive field to a clean string."""
    if value is None:
        return ""
    return str(value).strip()


__all__ = [
    "DEFAULT_MIN_COMMENT_CHARS",
    "is_recurring_megathread",
    "post_id_from_permalink",
    "post_to_markdown",
    "reddit_post_filename",
    "subreddit_from_permalink",
]

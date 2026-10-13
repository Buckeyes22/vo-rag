from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from vo_rag.core.chunking.chapter_path import derive_chapter_path
from vo_rag.core.chunking.counters import TokenCounterFn
from vo_rag.core.chunking.heading_filter import (
    is_back_matter_heading,
    is_labelled_heading,
    is_numbered_title,
)
from vo_rag.core.chunking.ids import _chunk_id
from vo_rag.core.chunking.models import DropReason, PendingShortText, RetrievalChunk
from vo_rag.core.chunking.section_packer import SectionPacker
from vo_rag.core.parsing.text_normalization import normalize_extracted_text


class ShortTextMerger:
    """Page-based placement of short-text candidates, plus page helpers.

    ``HierarchicalChunker`` no longer routes short text here (VOR-1452: it is
    emitted in place and merged by ``SmallChunkMerger``). ``merge_or_record``
    never drops text: a candidate joins the page-adjacent text/table chunk of
    its own section, or becomes a chunk of its own.
    """

    def __init__(self, *, counter: TokenCounterFn) -> None:
        self.count = counter

    def merge_or_record(
        self,
        chunks: list[RetrievalChunk],
        candidates: list[PendingShortText],
        *,
        document_id: str,
        source_id: str,
        doc_title: str,
        record_drop: Callable[[int | None, DropReason], None],
    ) -> list[RetrievalChunk]:
        for candidate in candidates:
            candidate_pages = set(candidate.pages)
            if not candidate.content.strip():
                for page in sorted(candidate_pages):
                    record_drop(page, "empty_after_normalize")
                continue

            target = self._adjacent_chunk_target(
                chunks, candidate_pages, section_path=candidate.section_path
            )
            if target is None:
                chunks.append(
                    self._short_text_chunk(
                        document_id=document_id,
                        source_id=source_id,
                        doc_title=doc_title,
                        candidate=candidate,
                    )
                )
                continue

            target_index, prepend = target
            self._merge_short_text_into_chunk(chunks[target_index], candidate, prepend=prepend)

        return chunks

    def _short_text_chunk(
        self,
        *,
        document_id: str,
        source_id: str,
        doc_title: str,
        candidate: PendingShortText,
    ) -> RetrievalChunk:
        page_number = candidate.pages[0] if candidate.pages else None
        page_range = self.page_range(candidate.pages)
        embedding_text = f"{candidate.section_path}\n\n{candidate.content}"
        return RetrievalChunk(
            chunk_id=_chunk_id(source_id, candidate.content, "text"),
            chunk_type="text",
            document_id=document_id,
            source_id=source_id,
            page_number=page_number,
            page_range=page_range,
            section_path=candidate.section_path,
            section_title=candidate.section_title,
            chapter_path=derive_chapter_path(candidate.section_path, doc_title),
            content=candidate.content,
            embedding_text=embedding_text,
            token_count=self.count(embedding_text),
            metadata=dict(candidate.metadata),
        )

    def _merge_short_text_into_chunk(
        self,
        chunk: RetrievalChunk,
        candidate: PendingShortText,
        *,
        prepend: bool,
    ) -> None:
        if prepend:
            chunk.content = normalize_extracted_text(f"{candidate.content}\n\n{chunk.content}")
        else:
            chunk.content = normalize_extracted_text(f"{chunk.content}\n\n{candidate.content}")

        pages = sorted(self.chunk_pages(chunk) | set(candidate.pages))
        if pages:
            chunk.page_number = pages[0]
            chunk.page_range = self.page_range(pages)
        chunk.embedding_text = f"{chunk.section_path}\n\n{chunk.content}"
        chunk.chunk_id = _chunk_id(chunk.source_id, chunk.content, chunk.chunk_type)
        chunk.token_count = self.count(chunk.embedding_text)

    def _adjacent_chunk_target(
        self,
        chunks: list[RetrievalChunk],
        pages: set[int],
        *,
        section_path: str,
    ) -> tuple[int, bool] | None:
        """The page-adjacent text/table chunk of ``section_path``: the last one
        starting on or before the candidate's page, else the first after it."""
        same_section = [
            index
            for index, chunk in enumerate(chunks)
            if chunk.section_path == section_path and chunk.chunk_type in _MERGEABLE_CHUNK_TYPES
        ]
        if not same_section:
            return None
        if not pages:
            return same_section[-1], False

        page = min(pages)
        previous: int | None = None
        next_chunk: int | None = None
        for index in same_section:
            chunk_pages = self.chunk_pages(chunks[index])
            if not chunk_pages:
                continue
            if min(chunk_pages) <= page:
                previous = index
            elif next_chunk is None:
                next_chunk = index

        if previous is not None:
            return previous, False
        if next_chunk is not None:
            return next_chunk, True
        return same_section[-1], False

    @staticmethod
    def covered_pages(chunks: list[RetrievalChunk]) -> set[int]:
        pages: set[int] = set()
        for chunk in chunks:
            pages.update(ShortTextMerger.chunk_pages(chunk))
        return pages

    @staticmethod
    def chunk_pages(chunk: RetrievalChunk) -> set[int]:
        if chunk.page_range:
            match = re.fullmatch(r"(\d+)(?:-(\d+))?", chunk.page_range)
            if match:
                start = int(match.group(1))
                end = int(match.group(2) or start)
                return set(range(start, end + 1))
        if chunk.page_number is not None:
            return {chunk.page_number}
        return set()

    @staticmethod
    def page_range(pages: list[int]) -> str | None:
        if not pages:
            return None
        low = min(pages)
        high = max(pages)
        return str(low) if low == high else f"{low}-{high}"


#: Chunk types ``SmallChunkMerger`` merges. Figure composites are out of scope
#: (their content is built separately); they are neither merged nor a barrier.
_MERGEABLE_CHUNK_TYPES = frozenset({"text", "table"})


@dataclass(frozen=True)
class ChunkScope:
    """Where the chunker emitted a chunk, for ``SmallChunkMerger`` (VOR-1452).

    ``epoch`` changes at every accepted depth-1/2 heading and at every change of
    the heading tracker's path (a back-matter heading expiring, the staleness
    guard firing), so two chunks with the same epoch sit in the same depth-2
    section -- even when two real sections share a heading text. ``overlap_prefix``
    is the previous chunk's tail that the chunker prepended to this one.
    ``headings`` is the open heading path as ``(depth, text, serial)``; the
    serial is unique per accepted heading, so a repeated heading text is still
    another section. ``short_text`` marks text the low-signal filter found
    under ``min_text_chunk_words``: always a merge candidate, never dropped.
    ``sibling_group`` marks a chunk built by merging tiny sibling sections.
    """

    epoch: int
    overlap_prefix: str = ""
    headings: tuple[tuple[int, str, int], ...] = ()
    short_text: bool = False
    sibling_group: bool = False


def _squashed(text: str) -> str:
    return " ".join(text.split())


def _path_segments(section_path: str) -> list[str]:
    stripped = section_path.strip().removeprefix("[").removesuffix("]")
    return [segment.strip() for segment in stripped.split(" > ") if segment.strip()]


def _path_tail(first_path: str, second_path: str) -> list[str]:
    """The segments of ``second_path`` below the part it shares with ``first_path``."""
    first, second = _path_segments(first_path), _path_segments(second_path)
    shared = 0
    while shared < min(len(first), len(second)) and first[shared] == second[shared]:
        shared += 1
    return second[shared:]


def _section_number(text: str) -> int | None:
    """The bare number of a numbered title ("3 The Technique of Hurdling" -> 3)."""
    if not is_numbered_title(text):
        return None
    return int(text.split()[0].rstrip("."))


def _numbered_steps(
    items: Sequence[RetrievalChunk],
    scopes: Mapping[int, ChunkScope],
    is_tiny: Callable[[RetrievalChunk], bool],
) -> frozenset[int]:
    """Heading serials of numbered sections that are steps, not chapters.

    Decided by the siblings, not the parent: the PDF parser puts "Chapter 2"
    at depth 1 and both "Session Plan" and "1 Warm up" at depth 2, so the
    parent says nothing. A numbered title ("1 Warm up") is a step when it sits
    in a run of two or more consecutive same-depth siblings under one parent
    whose numbers count on by one (1, 2, 3 ...), and every section of that run
    is a single tiny chunk. A numbered title with a long section in its run,
    or alone among non-numbered siblings ("1 The Sprint Start" beside "The Set
    Position"), is a chapter and never merges with a neighbour.
    """
    siblings: dict[tuple[tuple[int, str, int], ...], list[tuple[int, str]]] = {}
    chunk_count: dict[int, int] = {}
    tiny: dict[int, bool] = {}
    for chunk in items:
        scope = scopes.get(id(chunk))
        if scope is None or not scope.headings or chunk.chunk_type not in _MERGEABLE_CHUNK_TYPES:
            continue
        depth, text, serial = scope.headings[-1]
        group = siblings.setdefault(scope.headings[:-1] + ((depth, "", 0),), [])
        if serial not in chunk_count:
            group.append((serial, text))
        chunk_count[serial] = chunk_count.get(serial, 0) + 1
        tiny[serial] = tiny.get(serial, True) and is_tiny(chunk)
    steps: set[int] = set()
    for group in siblings.values():
        start = 0
        while start < len(group):
            number = _section_number(group[start][1])
            if number is None:
                start += 1
                continue
            end = start + 1
            while end < len(group) and _section_number(group[end][1]) == number + (end - start):
                end += 1
            run = [serial for serial, _ in group[start:end]]
            if len(run) >= 2 and all(chunk_count[s] == 1 and tiny[s] for s in run):
                steps.update(run)
            start = end
    return frozenset(steps)


def _is_chapter_like(headings: tuple[tuple[int, str, int], ...], steps: frozenset[int]) -> bool:
    """Whether the innermost of ``headings`` is a chapter or part of the book.

    A labelled unit ("Chapter 3", "6.12 Title") always is; a numbered title
    ("3 The Technique of Hurdling") is unless its siblings make it a step
    (``_numbered_steps``).
    """
    depth, text, serial = headings[-1]
    if is_labelled_heading(text):
        return True
    return is_numbered_title(text) and serial not in steps


class SmallChunkMerger:
    """Merge chunks below the ``min`` floor into a same-section neighbour (VOR-1452).

    A short fragment -- a one-sentence leaf section ("Try to hold each stretch
    for 8 seconds."), a one-row-group linearized table, a unit-bearing line
    that the low-signal filter keeps -- is never dropped here, only merged:

    - "small" means the chunk's own text (without the overlap prefix) counts
      fewer than ``min_tokens``; only text and table chunks take part;
    - pass 1 merges into the nearest text/table neighbour with the same epoch
      and the same section path, the preceding one first, else the following;
    - pass 2 treats a depth-3+ boundary as soft: a small chunk may merge with a
      neighbour of another leaf section inside the same depth-2 section (same
      epoch) when the neighbour is itself under ``target``; the later leaf's
      section title is kept as a line in front of its text. A path carrying a
      back-matter heading (References, Index, ...) only ever merges with the
      identical path, so no merge crosses a terminal boundary;
    - a merge happens only when the result stays within ``max_tokens`` (so the
      hard ceiling holds) and the front-matter / stale-heading labels agree;
    - pass 3 merges tiny sibling sections: OCR'd books tag day
      blocks ("Monday", "Tuesday", ...) and notes as top-level headings, one
      short paragraph each. Consecutive sections at the same depth under the
      same parent (same heading serials; or both top-level) merge when each
      one is a single chunk under the floor and the result stays under
      ``target``. A real chapter is never under the floor, so no chapter is
      ever swallowed or merged into. The group keeps the first sibling's path
      (parent path + first title) -- the retrieval label points at where the
      text starts, as for every chunk -- each later title leads its text as a
      line, and ``metadata["merged_sections"]`` lists every title in order;
    - pass 0 runs first: a text chunk that is nothing but a
      heading (``SectionPacker.is_heading_only_text``: "**part one:** /
      **jumping out of the fire**", left alone by a figure or page break
      before its body) is not a chunk. It prefixes the next text/table chunk --
      skipping figures, across a section boundary too, since a heading heads
      what follows it (a part title before chapter 1 sits under the previous
      chapter's headings in the parse) -- and that chunk keeps its own labels.
      Never into or out of back matter or another front-matter kind. With no
      such next chunk it is appended to the preceding chunk of its section;
      with neither it stays as it is;
    - a fragment with no eligible neighbour stays its own chunk.

    The merged chunk keeps the host's (normally the neighbour's) chunk_type and
    metadata and, like any chunk, the section path of where its text starts.
    ``metadata["merged_tables"]`` counts linearized tables folded into it (with
    their ``merged_table_element_ids``) and ``metadata["merged_fragments"]``
    counts short text chunks; a table's linearized text is kept verbatim.
    """

    def __init__(
        self,
        *,
        counter: TokenCounterFn,
        min_tokens: int,
        target: int,
        max_tokens: int,
    ) -> None:
        self.count = counter
        self.min = min_tokens
        self.target = target
        self.max = max_tokens

    def merge(
        self,
        chunks: list[RetrievalChunk],
        scopes: Mapping[int, ChunkScope],
    ) -> list[RetrievalChunk]:
        """Return ``chunks`` with small ones folded into neighbours; ``scopes`` is
        keyed by ``id(chunk)`` and chunks without a scope are left alone."""
        if self.min <= 0 or len(chunks) < 2:
            # ``min_=0`` turns merging off: every chunk, short text included,
            # stays as emitted (nothing is dropped either way).
            return chunks
        items = list(chunks)
        live: dict[int, ChunkScope] = {
            id(chunk): scopes[id(chunk)] for chunk in items if id(chunk) in scopes
        }
        self._fold_heading_only(items, live)
        for soft in (False, True):
            index = 0
            while index < len(items):
                if not self._is_small(items[index], live):
                    index += 1
                    continue
                other = self._neighbour(items, index, live, soft=soft)
                if other is None:
                    index += 1
                    continue
                low, high = min(index, other), max(index, other)
                items[low] = self._merged(items[low], items[high], live, small=items[index])
                del items[high]
                # Either way ``index`` now holds the next unexamined chunk (after a
                # backward merge) or the merged chunk (after a forward one).
        self._merge_siblings(items, live)
        self._join_parent_intros(items, live)
        return items

    def _join_parent_intros(
        self, items: list[RetrievalChunk], scopes: dict[int, ChunkScope]
    ) -> None:
        """A small chunk beside a sibling group relabelled to its own path joins it.

        A parent heading's one-line intro ("Before taking a detailed look
        at the event ...") and the tiny sibling sections after it, merged and
        relabelled to that parent by pass 3, now carry the same label; the
        epoch test of passes 1-2 kept them apart. Only an identical path and a
        result within ``max_tokens`` qualify.
        """
        index = 0
        while index < len(items):
            chunk = items[index]
            if not self._is_small(chunk, scopes):
                index += 1
                continue
            merged = False
            for step in (1, -1):
                other = index + step
                while (
                    0 <= other < len(items)
                    and items[other].chunk_type not in _MERGEABLE_CHUNK_TYPES
                ):
                    other += step
                if not 0 <= other < len(items):
                    continue
                neighbour = items[other]
                scope, neighbour_scope = scopes.get(id(chunk)), scopes.get(id(neighbour))
                if (
                    scope is None
                    or neighbour_scope is None
                    or neighbour.section_path != chunk.section_path
                    or not (scope.sibling_group or neighbour_scope.sibling_group)
                    or any(
                        chunk.metadata.get(key) != neighbour.metadata.get(key)
                        for key in ("front_matter_kind", "stale_heading")
                    )
                ):
                    continue
                low, high = min(index, other), max(index, other)
                content = self._merged_content(items[low], items[high], scopes)
                if self.count(f"{items[low].section_path}\n\n{content}") > self.max:
                    continue
                items[low] = self._merged(items[low], items[high], scopes, small=chunk)
                del items[high]
                merged = True
                break
            if not merged:
                index += 1

    def _fold_heading_only(
        self, items: list[RetrievalChunk], scopes: dict[int, ChunkScope]
    ) -> None:
        index = 0
        while index < len(items):
            chunk = items[index]
            if (
                chunk.chunk_type != "text"
                or id(chunk) not in scopes
                or not SectionPacker.is_heading_only_text(self._body(chunk, scopes))
            ):
                index += 1
                continue
            following = index + 1
            while (
                following < len(items) and items[following].chunk_type not in _MERGEABLE_CHUNK_TYPES
            ):
                following += 1
            if following < len(items) and self._can_head(chunk, items[following], scopes):
                items[following] = self._headed(chunk, items[following], scopes)
                del items[index]
                continue
            previous = index - 1
            while previous >= 0 and items[previous].chunk_type not in _MERGEABLE_CHUNK_TYPES:
                previous -= 1
            if previous >= 0 and self._can_merge(
                items[previous], chunk, scopes, soft=False, small=chunk
            ):
                items[previous] = self._merged(items[previous], chunk, scopes, small=chunk)
                del items[index]
                continue
            index += 1

    def _can_head(
        self,
        heading: RetrievalChunk,
        target: RetrievalChunk,
        scopes: Mapping[int, ChunkScope],
    ) -> bool:
        if id(target) not in scopes:
            return False
        if self._in_back_matter(heading.section_path) != self._in_back_matter(target.section_path):
            return False
        if self._in_back_matter(target.section_path) and (
            scopes[id(heading)].epoch != scopes[id(target)].epoch
        ):
            return False
        if heading.metadata.get("front_matter_kind") != target.metadata.get("front_matter_kind"):
            return False
        content = self._headed_content(heading, target, scopes)
        return self.count(f"{target.section_path}\n\n{content}") <= self.max

    def _headed_content(
        self,
        heading: RetrievalChunk,
        target: RetrievalChunk,
        scopes: Mapping[int, ChunkScope],
    ) -> str:
        heading_text = self._body(heading, scopes).strip()
        body = self._body(target, scopes)
        # The target may already open with the heading as its overlap prefix.
        if body.lstrip().startswith(heading_text) or target.content.lstrip().startswith(
            heading_text
        ):
            return target.content
        return normalize_extracted_text(f"{heading_text}\n\n{body}")

    def _headed(
        self,
        heading: RetrievalChunk,
        target: RetrievalChunk,
        scopes: dict[int, ChunkScope],
    ) -> RetrievalChunk:
        """``target`` with ``heading`` in front; ``target`` keeps its labels."""
        content = self._headed_content(heading, target, scopes)
        metadata = dict(target.metadata)
        earlier = target.metadata.get("merged_headings")
        metadata["merged_headings"] = [
            *(earlier if isinstance(earlier, list) else []),
            " ".join(self._body(heading, scopes).split()),
        ]
        pages = sorted(ShortTextMerger.chunk_pages(heading) | ShortTextMerger.chunk_pages(target))
        embedding_text = f"{target.section_path}\n\n{content}"
        headed = target.model_copy(
            update={
                "chunk_id": _chunk_id(target.source_id, content, target.chunk_type),
                "content": content,
                "embedding_text": embedding_text,
                "token_count": self.count(embedding_text),
                "page_number": pages[0] if pages else target.page_number,
                "page_range": ShortTextMerger.page_range(pages) or target.page_range,
                "metadata": metadata,
            },
            deep=True,
        )
        target_scope = scopes[id(target)]
        scopes[id(headed)] = ChunkScope(
            epoch=target_scope.epoch,
            headings=target_scope.headings,
            short_text=target_scope.short_text,
            sibling_group=target_scope.sibling_group,
        )
        return headed

    def _merge_siblings(self, items: list[RetrievalChunk], scopes: dict[int, ChunkScope]) -> None:
        sections: dict[tuple[int, tuple[tuple[int, str, int], ...]], int] = {}
        for chunk in items:
            scope = scopes.get(id(chunk))
            if scope is not None and chunk.chunk_type in _MERGEABLE_CHUNK_TYPES:
                key = (scope.epoch, scope.headings)
                sections[key] = sections.get(key, 0) + 1

        def tiny_section(chunk: RetrievalChunk) -> bool:
            scope = scopes.get(id(chunk))
            if scope is None or not scope.headings:
                return False
            if chunk.chunk_type not in _MERGEABLE_CHUNK_TYPES:
                return False
            if scope.sibling_group:
                return True
            return sections.get((scope.epoch, scope.headings), 0) == 1 and self._is_small(
                chunk, scopes
            )

        steps = _numbered_steps(items, scopes, lambda chunk: self._is_small(chunk, scopes))
        index = 0
        while index < len(items):
            first = items[index]
            if not tiny_section(first):
                index += 1
                continue
            other = index + 1
            while other < len(items) and items[other].chunk_type not in _MERGEABLE_CHUNK_TYPES:
                other += 1
            if other >= len(items):
                break
            second = items[other]
            if not (tiny_section(second) and self._siblings(first, second, scopes, steps)):
                index += 1
                continue
            content = self._merged_content(first, second, scopes)
            if self.count(f"{first.section_path}\n\n{content}") >= self.target:
                index += 1
                continue
            merged = self._merged(first, second, scopes, small=second)
            if scopes[id(first)].headings[-1][0] <= 2:
                merged = self._relabel_to_parent(first, merged, scopes)
            titles = self._section_titles(first) + self._section_titles(second)
            merged.metadata["merged_sections"] = titles
            first_scope = scopes[id(merged)]
            scopes[id(merged)] = ChunkScope(
                epoch=first_scope.epoch,
                overlap_prefix=first_scope.overlap_prefix,
                headings=first_scope.headings,
                sibling_group=True,
            )
            items[index] = merged
            del items[other]

    def _relabel_to_parent(
        self,
        first: RetrievalChunk,
        merged: RetrievalChunk,
        scopes: dict[int, ChunkScope],
    ) -> RetrievalChunk:
        """A merged group of depth-1/2 siblings takes their common parent's label.

        Merges never cross a depth-1/2 boundary under one sibling's name: the
        group is labelled with the parent path -- the document title for
        top-level siblings -- and the first sibling's title leads its text like
        every later one's does. ``merged_sections`` keeps the titles.
        """
        if scopes[id(first)].sibling_group:
            return merged  # already the parent's label
        segments = [
            segment.strip()
            for segment in first.section_path.strip()
            .removeprefix("[")
            .removesuffix("]")
            .split(" > ")
        ]
        doc_title = segments[0]
        title = first.section_title.strip()
        parent = segments[:-1] if len(segments) > 1 and segments[-1] == title else segments
        parent_path = f"[{' > '.join(parent)}]"
        scope = scopes[id(merged)]
        prefix = scope.overlap_prefix
        content = merged.content
        body = content[len(prefix) :].lstrip() if prefix and content.startswith(prefix) else content
        if title and not body.startswith(title):
            body = f"{title}\n{body}"
            content = normalize_extracted_text(
                f"{prefix}\n\n{body}" if prefix and content.startswith(prefix) else body
            )
        embedding_text = f"{parent_path}\n\n{content}"
        relabelled = merged.model_copy(
            update={
                "chunk_id": _chunk_id(merged.source_id, content, merged.chunk_type),
                "section_path": parent_path,
                "section_title": parent[-1],
                "chapter_path": derive_chapter_path(parent_path, doc_title),
                "content": content,
                "embedding_text": embedding_text,
                "token_count": self.count(embedding_text),
            },
            deep=True,
        )
        scopes[id(relabelled)] = ChunkScope(
            epoch=scope.epoch,
            overlap_prefix=prefix if prefix and content.startswith(prefix) else "",
            headings=scope.headings,
            sibling_group=scope.sibling_group,
        )
        return relabelled

    @staticmethod
    def _section_titles(chunk: RetrievalChunk) -> list[str]:
        titles = chunk.metadata.get("merged_sections")
        if isinstance(titles, list):
            return [str(title) for title in titles]
        return [chunk.section_title]

    def _siblings(
        self,
        first: RetrievalChunk,
        second: RetrievalChunk,
        scopes: Mapping[int, ChunkScope],
        steps: frozenset[int] = frozenset(),
    ) -> bool:
        first_headings = scopes[id(first)].headings
        second_headings = scopes[id(second)].headings
        if not first_headings or not second_headings:
            return False
        if first_headings[-1][0] != second_headings[-1][0]:
            return False  # not the same depth
        if first_headings[:-1] != second_headings[:-1]:
            return False  # another parent section
        if first_headings[-1][2] == second_headings[-1][2]:
            return False  # the same section
        if first_headings[-1][0] <= 2 and (
            _is_chapter_like(first_headings, steps) or _is_chapter_like(second_headings, steps)
        ):
            return False  # a chapter/part is never merged with its neighbour
        if self._in_back_matter(first.section_path) or self._in_back_matter(second.section_path):
            return False
        return all(
            first.metadata.get(key) == second.metadata.get(key)
            for key in ("front_matter_kind", "stale_heading")
        )

    def _body(self, chunk: RetrievalChunk, scopes: Mapping[int, ChunkScope]) -> str:
        """``chunk.content`` without the overlap prefix the chunker put in front."""
        scope = scopes.get(id(chunk))
        prefix = scope.overlap_prefix if scope else ""
        if prefix and chunk.content.startswith(prefix):
            return chunk.content[len(prefix) :].lstrip()
        return chunk.content

    def _is_small(self, chunk: RetrievalChunk, scopes: Mapping[int, ChunkScope]) -> bool:
        scope = scopes.get(id(chunk))
        if chunk.chunk_type not in _MERGEABLE_CHUNK_TYPES or scope is None:
            return False
        if scope.short_text:
            return True
        return self.count(self._body(chunk, scopes)) < self.min

    def _neighbour(
        self,
        items: list[RetrievalChunk],
        index: int,
        scopes: Mapping[int, ChunkScope],
        *,
        soft: bool,
    ) -> int | None:
        """The preceding, else the following, text/table chunk it may merge with."""
        for step in (-1, 1):
            other = index + step
            while 0 <= other < len(items) and items[other].chunk_type not in _MERGEABLE_CHUNK_TYPES:
                other += step
            if not 0 <= other < len(items):
                continue
            low, high = min(index, other), max(index, other)
            if self._can_merge(items[low], items[high], scopes, soft=soft, small=items[index]):
                return other
        return None

    def _can_merge(
        self,
        first: RetrievalChunk,
        second: RetrievalChunk,
        scopes: Mapping[int, ChunkScope],
        *,
        soft: bool,
        small: RetrievalChunk,
    ) -> bool:
        first_scope = scopes.get(id(first))
        second_scope = scopes.get(id(second))
        if first_scope is None or second_scope is None:
            return False
        if first_scope.epoch != second_scope.epoch:
            return False
        for key in ("front_matter_kind", "stale_heading"):
            if first.metadata.get(key) != second.metadata.get(key):
                return False
        if first.section_path != second.section_path:
            if not soft:
                return False
            if self._in_back_matter(first.section_path) or self._in_back_matter(
                second.section_path
            ):
                return False
            neighbour = second if small is first else first
            if self.count(self._body(neighbour, scopes)) >= self.target:
                return False
        content = self._merged_content(first, second, scopes)
        return self.count(f"{first.section_path}\n\n{content}") <= self.max

    def _roles(
        self,
        first: RetrievalChunk,
        second: RetrievalChunk,
        scopes: Mapping[int, ChunkScope],
        *,
        small: RetrievalChunk,
    ) -> tuple[RetrievalChunk, RetrievalChunk]:
        """``(host, guest)``: the guest is folded into the host, whose chunk type
        and metadata the merged chunk keeps. Normally the small chunk is the
        guest; a small table beside small text is hosted by the text."""
        host, guest = (second, first) if small is first else (first, second)
        if (
            host.chunk_type == "table"
            and guest.chunk_type == "text"
            and self._is_small(host, scopes)
        ):
            host, guest = guest, host
        return host, guest

    def _merged_content(
        self,
        first: RetrievalChunk,
        second: RetrievalChunk,
        scopes: Mapping[int, ChunkScope],
    ) -> str:
        """``first`` then ``second`` joined. The second chunk's overlap prefix
        (a copy of text just before it) is dropped when the first holds it; when
        the two come from different leaf sections, the second one's section
        title leads its text (the merged chunk keeps the first one's path)."""
        second_text = second.content
        second_body = self._body(second, scopes)
        if second_body != second.content:
            prefix = second.content[: len(second.content) - len(second_body)].strip()
            # Compared with whitespace collapsed: the overlap tail is cut from
            # text whose paragraphs the first chunk joins with blank lines, so
            # "game The whole time" is "game\n\nThe whole time" there (VOR-1453:
            # the copy was kept and the chunk held a paragraph twice).
            if prefix and _squashed(prefix) in _squashed(first.content):
                second_text = second_body
        if first.section_path != second.section_path:
            # Every heading of the second's path below the shared part leads its
            # text, not just the leaf ("January and February" over "Monday"):
            # the merged chunk keeps only the first one's path.
            lead = _path_tail(first.section_path, second.section_path)
            if not lead:
                lead = [second.section_title.strip()] if second.section_title.strip() else []
            if lead and second_text.lstrip().startswith(lead[-1]):
                lead = lead[:-1]
            if lead:
                second_text = "\n".join([*lead, second_text])
        return normalize_extracted_text(f"{first.content}\n\n{second_text}")

    def _merged(
        self,
        first: RetrievalChunk,
        second: RetrievalChunk,
        scopes: dict[int, ChunkScope],
        *,
        small: RetrievalChunk,
    ) -> RetrievalChunk:
        host, guest = self._roles(first, second, scopes, small=small)
        content = self._merged_content(first, second, scopes)
        metadata = dict(host.metadata)
        self._absorb_metadata(metadata, guest)
        pages = sorted(ShortTextMerger.chunk_pages(host) | ShortTextMerger.chunk_pages(guest))
        embedding_text = f"{first.section_path}\n\n{content}"
        merged = host.model_copy(
            update={
                "chunk_id": _chunk_id(host.source_id, content, host.chunk_type),
                # Labelled, like any chunk, by the section its text starts in.
                "section_path": first.section_path,
                "section_title": first.section_title,
                "chapter_path": first.chapter_path,
                "content": content,
                "embedding_text": embedding_text,
                "token_count": self.count(embedding_text),
                "page_number": pages[0] if pages else host.page_number,
                "page_range": ShortTextMerger.page_range(pages) or host.page_range,
                "metadata": metadata,
            },
            deep=True,
        )
        first_scope = scopes[id(first)]
        prefix = first_scope.overlap_prefix
        if prefix and not content.startswith(prefix):
            prefix = ""
        scopes[id(merged)] = ChunkScope(
            epoch=first_scope.epoch,
            overlap_prefix=prefix,
            headings=first_scope.headings,
            sibling_group=first_scope.sibling_group,
        )
        return merged

    @staticmethod
    def _absorb_metadata(metadata: dict[str, object], guest: RetrievalChunk) -> None:
        def as_int(value: object) -> int:
            return value if isinstance(value, int) else 0

        def as_list(value: object) -> list[object]:
            return list(value) if isinstance(value, list) else []

        guest_tables = as_int(guest.metadata.get("merged_tables"))
        guest_table_ids = as_list(guest.metadata.get("merged_table_element_ids"))
        guest_fragments = as_int(guest.metadata.get("merged_fragments"))
        if guest.chunk_type == "table":
            guest_tables += 1
            element_id = guest.metadata.get("element_id")
            if element_id is not None:
                guest_table_ids.insert(0, element_id)
        else:
            guest_fragments += 1
        if guest_tables:
            metadata["merged_tables"] = as_int(metadata.get("merged_tables")) + guest_tables
        if guest_table_ids:
            metadata["merged_table_element_ids"] = (
                as_list(metadata.get("merged_table_element_ids")) + guest_table_ids
            )
        if guest_fragments:
            metadata["merged_fragments"] = (
                as_int(metadata.get("merged_fragments")) + guest_fragments
            )

    @staticmethod
    def _in_back_matter(section_path: str) -> bool:
        stripped = section_path.strip().removeprefix("[").removesuffix("]")
        segments = [segment.strip() for segment in stripped.split(" > ")][1:]
        return any(is_back_matter_heading(segment) for segment in segments if segment)

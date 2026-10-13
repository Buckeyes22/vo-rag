from __future__ import annotations

import re

import structlog

from vo_rag.core.chunking.counters import TokenCounterFn
from vo_rag.core.chunking.ids import _chunk_id
from vo_rag.core.chunking.models import RetrievalChunk
from vo_rag.core.parsing.text_normalization import normalize_extracted_text

log = structlog.get_logger(__name__)


class TokenCeilingBackstop:
    """Boundary-aware splitter plus final invariant check for retrieval chunks."""

    def __init__(
        self,
        *,
        counter: TokenCounterFn,
        max_tokens: int,
        overlap: int = 50,
    ) -> None:
        self.count = counter
        self.max = max_tokens
        self.overlap = min(overlap, max(0, max_tokens // 4))

    def hard_ceiling(self) -> int:
        return int(self.max * 1.15)

    def embedding_token_count(self, section_path: str, content: str) -> int:
        return self.count(f"{section_path}\n\n{content}")

    def _splitting_path(self, section_path: str) -> str:
        """Section path to budget splits against, or ``""`` if it alone exhausts the budget.

        When the path by itself is over ``max`` no piece can fit, and the
        splitter would degrade to one character per piece (VOR-1416: ~297k
        one-char chunks). Splitting cannot fix path overhead, so budget the
        content alone and leave the oversize chunk to the ceiling warning.
        """
        if self.embedding_token_count(section_path, "") < self.max:
            return section_path
        log.warning(
            "book_rag.chunk.section_path_exceeds_budget",
            section_path_chars=len(section_path),
            max_tokens=self.max,
        )
        return ""

    def split_text(self, content: str, section_path: str) -> list[str]:
        if self.embedding_token_count(section_path, content) <= self.hard_ceiling():
            return [content]
        return self._split_bounded_text(content, self._splitting_path(section_path))

    def split_table(self, content: str, section_path: str) -> list[str]:
        if self.embedding_token_count(section_path, content) <= self.hard_ceiling():
            return [content]
        section_path = self._splitting_path(section_path)

        rows = [line.strip() for line in content.splitlines() if line.strip()]
        if len(rows) <= 1:
            return self._split_bounded_text(content, section_path)

        headers: list[str] = []
        while rows and rows[0].startswith(("Comparison across columns:", "Caption:")):
            headers.append(rows.pop(0))
        if not rows:
            return self._split_bounded_text(content, section_path)
        context = "\n".join(headers)
        if context and self.embedding_token_count(section_path, context) <= self.max // 2:
            budget_path = f"{section_path}\n\n{context}"
            return [f"{context}\n{part}" for part in self._split_rows(rows, budget_path)]
        return self._split_rows([*headers, *rows], section_path)

    def enforce(self, chunks: list[RetrievalChunk]) -> list[RetrievalChunk]:
        """Guarantee chunks over max*1.15 are re-split when reducible."""
        ceiling = self.hard_ceiling()
        result: list[RetrievalChunk] = []
        for chunk in chunks:
            if chunk.token_count <= ceiling or chunk.chunk_type == "figure_composite":
                # A figure is one image: its content (caption, page excerpt, VLM
                # description) is bounded by the builder and its embedding_text
                # is a bounded window around it; splitting would duplicate the
                # image and drop the window (VOR-1452).
                result.append(chunk)
                continue
            pieces = self._greedy_pack(chunk.content, self._splitting_path(chunk.section_path))
            if len(pieces) <= 1:
                log.warning(
                    "book_rag.chunk.ceiling_unsplittable",
                    source_id=chunk.source_id,
                    chunk_id=chunk.chunk_id,
                    token_count=chunk.token_count,
                    ceiling=ceiling,
                )
                result.append(chunk)
                continue
            total = len(pieces)
            for part_number, piece in enumerate(pieces, start=1):
                embedding_text = f"{chunk.section_path}\n\n{piece}"
                metadata = dict(chunk.metadata)
                metadata["ceiling_split_part"] = part_number
                metadata["ceiling_split_total"] = total
                result.append(
                    chunk.model_copy(
                        update={
                            "chunk_id": _chunk_id(chunk.source_id, piece, chunk.chunk_type),
                            "content": piece,
                            "embedding_text": embedding_text,
                            "token_count": self.count(embedding_text),
                            "metadata": metadata,
                        }
                    )
                )
        return result

    def tail_for_overlap(self, text: str) -> str:
        if self.overlap <= 0 or not text.strip():
            return ""
        words = text.split()
        if not words:
            return ""
        window = min(len(words), self.overlap * 2)
        candidate_words = words[-window:]
        while candidate_words and self.count(" ".join(candidate_words)) > self.overlap:
            candidate_words = candidate_words[1:]
        return " ".join(candidate_words)

    @staticmethod
    def ends_at_retrieval_boundary(text: str) -> bool:
        stripped = normalize_extracted_text(text).strip()
        if not stripped:
            return True
        stripped = re.sub(r"\s+\d{1,4}\s*$", "", stripped).rstrip()
        if not stripped:
            return True
        if stripped.endswith((".", "!", "?", ")", "]", "}")):
            return True
        last_line = stripped.splitlines()[-1].strip()
        if not last_line:
            return True
        if re.fullmatch(r"(?:[\wα-ωΑ-Ω]+\s*=\s*)?[-+*/=().,\d\sA-Za-zα-ωΑ-Ω^]+", last_line):
            return last_line.endswith((")", ".", "]")) or "=" in last_line
        return False

    def _split_rows(self, rows: list[str], section_path: str) -> list[str]:
        parts: list[str] = []
        overlap_prefix = ""
        index = 0
        while index < len(rows):
            current: list[str] = []
            while index < len(rows):
                candidate_body = "\n".join([*current, rows[index]])
                candidate = self._with_overlap_prefix(overlap_prefix, candidate_body)
                if self.embedding_token_count(section_path, candidate) <= self.max:
                    current.append(rows[index])
                    index += 1
                    continue
                if current:
                    break

                row_parts = self._split_bounded_text(
                    rows[index],
                    section_path,
                    initial_overlap_prefix=overlap_prefix,
                )
                parts.extend(row_parts)
                overlap_prefix = self.tail_for_overlap(row_parts[-1])
                index += 1
                current = []
                break

            if current:
                part = self._with_overlap_prefix(overlap_prefix, "\n".join(current))
                parts.append(part)
                overlap_prefix = self.tail_for_overlap(part)

        return parts

    def _split_bounded_text(
        self,
        content: str,
        section_path: str,
        *,
        initial_overlap_prefix: str = "",
    ) -> list[str]:
        remaining = content.strip()
        parts: list[str] = []
        overlap_prefix = initial_overlap_prefix.strip()
        while remaining:
            body, remaining = self._take_bounded_text_prefix(
                remaining,
                section_path,
                overlap_prefix,
            )
            part = self._with_overlap_prefix(overlap_prefix, body)
            parts.append(part)
            overlap_prefix = self.tail_for_overlap(part)
        return parts

    def _take_bounded_text_prefix(
        self,
        text: str,
        section_path: str,
        overlap_prefix: str,
    ) -> tuple[str, str]:
        text = text.strip()
        if (
            self.embedding_token_count(
                section_path,
                self._with_overlap_prefix(overlap_prefix, text),
            )
            <= self.max
        ):
            return text, ""

        boundary_groups = (
            self._paragraph_boundaries(text),
            self._sentence_boundaries(text),
            self._whitespace_boundaries(text),
        )
        for boundaries in boundary_groups:
            boundary = self._largest_fitting_boundary(
                text,
                boundaries,
                section_path,
                overlap_prefix,
            )
            if boundary is not None:
                body_end, remainder_start = boundary
                return text[:body_end].strip(), text[remainder_start:].strip()

        first_word = re.match(r"\S+", text)
        if first_word is None:
            head, rest = self._char_chop_prefix(text, section_path, overlap_prefix)
            return head, rest
        word = first_word.group(0)
        tail = text[first_word.end() :].strip()
        candidate_word = self._with_overlap_prefix(overlap_prefix, word)
        if self.embedding_token_count(section_path, candidate_word) > self.max:
            head, word_rest = self._char_chop_prefix(word, section_path, overlap_prefix)
            remainder = f"{word_rest} {tail}".strip() if tail else word_rest
            return head, remainder
        return word, tail

    def _largest_fitting_boundary(
        self,
        text: str,
        boundaries: list[tuple[int, int]],
        section_path: str,
        overlap_prefix: str,
    ) -> tuple[int, int] | None:
        for body_end, remainder_start in reversed(boundaries):
            candidate_body = text[:body_end].strip()
            if not candidate_body:
                continue
            candidate = self._with_overlap_prefix(overlap_prefix, candidate_body)
            if self.embedding_token_count(section_path, candidate) <= self.max:
                return body_end, remainder_start
        return None

    @staticmethod
    def _paragraph_boundaries(text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in re.finditer(r"\n\s*\n", text)]

    @staticmethod
    def _sentence_boundaries(text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in re.finditer(r"(?<=[.!?])\s+", text)]

    @staticmethod
    def _whitespace_boundaries(text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in re.finditer(r"\s+", text)]

    @staticmethod
    def _with_overlap_prefix(overlap_prefix: str, body: str) -> str:
        prefix = overlap_prefix.strip()
        body = body.strip()
        if prefix and body:
            return f"{prefix}\n\n{body}"
        return prefix or body

    def _char_chop_prefix(
        self,
        word: str,
        section_path: str,
        overlap_prefix: str,
    ) -> tuple[str, str]:
        lo, hi, best = 1, len(word), 1
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = self._with_overlap_prefix(overlap_prefix, word[:mid])
            if self.embedding_token_count(section_path, candidate) <= self.max:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        return word[:best], word[best:]

    def _char_chop(self, word: str, section_path: str) -> list[str]:
        pieces: list[str] = []
        rest = word
        while rest:
            head, rest = self._char_chop_prefix(rest, section_path, "")
            pieces.append(head)
        return pieces

    def _greedy_pack(self, content: str, section_path: str) -> list[str]:
        tokens = content.split()
        if not tokens:
            return self._char_chop(content, section_path) if content else [content]
        safe: list[str] = []
        for token in tokens:
            if self.embedding_token_count(section_path, token) > self.max:
                safe.extend(self._char_chop(token, section_path))
            else:
                safe.append(token)
        parts: list[str] = []
        current: list[str] = []
        for token in safe:
            trial = " ".join([*current, token])
            if current and self.embedding_token_count(section_path, trial) > self.max:
                parts.append(" ".join(current))
                current = [token]
            else:
                current.append(token)
        if current:
            parts.append(" ".join(current))
        return parts or [content]

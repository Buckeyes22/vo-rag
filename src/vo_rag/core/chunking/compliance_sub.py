from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Callable

from vo_rag.core.chunking.counters import naive_counter
from vo_rag.core.chunking.models import RetrievalChunk
from vo_rag.core.models import VoBaseModel


def _sub_chunk_id(source_id: str, parent_id: str, position: int) -> str:
    key = f"{source_id}|{parent_id}:{position}|sub".encode()
    return str(uuid.UUID(bytes=hashlib.sha256(key).digest()[:16]))


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_RE.split(text.strip()) if s]


class ComplianceSubChunk(VoBaseModel):
    sub_chunk_id: str
    parent_chunk_id: str
    content: str
    token_count: int
    position_in_parent: int
    source_id: str
    page_number: int | None = None


class ComplianceSubChunker:
    def __init__(
        self,
        *,
        max_tokens: int = 200,
        counter: Callable[[str], int] | None = None,
    ) -> None:
        self.max = max_tokens
        self.count = counter or naive_counter

    def split(self, parent: RetrievalChunk) -> list[ComplianceSubChunk]:
        sentences = _split_sentences(parent.content)
        subs: list[ComplianceSubChunk] = []
        buf: list[str] = []
        buf_tok = 0
        pos = 0

        def _emit(text: str, position: int) -> ComplianceSubChunk:
            return ComplianceSubChunk(
                sub_chunk_id=_sub_chunk_id(
                    parent.source_id,
                    parent.chunk_id,
                    position,
                ),
                parent_chunk_id=parent.chunk_id,
                content=text,
                token_count=self.count(text),
                position_in_parent=position,
                source_id=parent.source_id,
                page_number=parent.page_number,
            )

        for sent in sentences:
            st = self.count(sent)
            if st > self.max:
                if buf:
                    subs.append(_emit(" ".join(buf), pos))
                    pos += 1
                    buf, buf_tok = [], 0
                for piece in self._token_slice(sent):
                    subs.append(_emit(piece, pos))
                    pos += 1
                continue
            if buf_tok + st > self.max and buf:
                subs.append(_emit(" ".join(buf), pos))
                pos += 1
                buf, buf_tok = [], 0
            buf.append(sent)
            buf_tok += st
        if buf:
            subs.append(_emit(" ".join(buf), pos))
        return subs

    def _token_slice(self, sentence: str) -> list[str]:
        # Coarse slicing: approximate token budget via char budget (4 chars/token).
        char_budget = self.max * 4
        return [sentence[i : i + char_budget] for i in range(0, len(sentence), char_budget)]

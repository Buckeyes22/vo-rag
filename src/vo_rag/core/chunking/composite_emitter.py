from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Sequence

from pydantic import Field

from vo_rag.core.chunking.chapter_path import derive_chapter_path
from vo_rag.core.chunking.counters import TokenCounterFn
from vo_rag.core.chunking.front_matter import front_matter_kind
from vo_rag.core.chunking.ids import _chunk_id
from vo_rag.core.chunking.models import RetrievalChunk
from vo_rag.core.chunking.section_packer import SectionPacker
from vo_rag.core.models import VoBaseModel
from vo_rag.core.parsing.contract import ElementType, ParsedElement


class CompositeChunk(VoBaseModel):
    """One figure, as the visual sidecar assembled it for the chunker.

    ``embedding_text`` is the retrieval window (caption, page context,
    neighbouring prose, VLM description) and feeds only the emitted chunk's
    ``embedding_text``. The emitted chunk's stored ``content`` is built from
    ``caption``, ``page_context`` (a short excerpt) and ``vlm_description``
    only -- never the neighbour prose, which the text chunks already hold
    (VOR-1452). ``section_title`` is the builder's guess; the chunker labels
    the chunk from its own heading tracker.
    """

    chunk_id: str
    chunk_type: str = "figure_composite"
    section_title: str
    preceding_text: str
    vlm_description: str
    following_text: str
    image_id: str
    image_ref: str
    embedding_text: str
    #: The figure's caption, if any.
    caption: str = ""
    #: The sentences nearest the figure on its page, bounded by the builder.
    page_context: str = ""
    token_count: int
    page_number: int
    source_id: str
    element_index: int
    metadata: dict[str, object] = Field(default_factory=dict)


def figure_content(composite: CompositeChunk, *, section_title: str) -> str:
    """Stored ``content`` of a figure chunk: the figure's own text only.

    "Caption:", "Page context:" and "Visual:" lines (the labels
    ``core.corpus.health`` counts). A figure with none of the three -- no
    caption, no prose on its page, VLM off -- still gets a non-empty content,
    "Figure on page N (section)": the chunk is kept rather than skipped because
    a chunk's ``image_ref`` is the only way the artifact figure gallery and a
    query hit reach the image, and its neighbour window keeps it retrievable
    through ``embedding_text``.
    """
    parts: list[str] = []
    if composite.caption.strip():
        parts.append(f"Caption: {composite.caption.strip()}")
    if composite.page_context.strip():
        parts.append(f"Page context: {composite.page_context.strip()}")
    if composite.vlm_description.strip():
        parts.append(f"Visual: {composite.vlm_description.strip()}")
    if parts:
        return "\n\n".join(parts)
    where = f"Figure on page {composite.page_number}" if composite.page_number else "Figure"
    return f"{where} ({section_title})" if section_title else where


#: Element types that carry the book's running text, for figure placement.
_PLACEMENT_TEXT_TYPES = frozenset(
    {
        ElementType.NARRATIVE_TEXT,
        ElementType.TITLE,
        ElementType.LIST_ITEM,
        ElementType.TABLE,
        ElementType.FIGURE_CAPTION,
        ElementType.EQUATION,
        ElementType.CODE_BLOCK,
    }
)


class FigurePlacement:
    """Where a figure sits in the text's page order.

    A parse stored before VOR-1417 appends every picture after the last text
    element, so a figure's ``element_index`` points past the end of the book
    and its neighbours are the back matter: all 30 figures of one audited book
    were labelled "Titles in This Series" (VOR-1452). A figure whose element
    comes after text of a later page than its own is out of page order; its
    anchor is the end of its page's text -- just before the first text element
    of a later page. An in-order figure's anchor is its own index.
    """

    def __init__(self, elements: Sequence[ParsedElement]) -> None:
        self._count = len(elements)
        # prefix_max[i]: the highest text page among elements[:i] (0 if none).
        prefix_max = [0] * (len(elements) + 1)
        running = 0
        for index, element in enumerate(elements):
            if element.element_type in _PLACEMENT_TEXT_TYPES and element.page_number is not None:
                running = max(running, element.page_number)
            prefix_max[index + 1] = running
        self._prefix_max = prefix_max

    def anchor(self, element_index: int, page_number: int | None) -> int:
        """Insertion point (before element ``anchor``) of a figure."""
        index = min(max(element_index, 0), self._count)
        if page_number is None or self._prefix_max[index] <= page_number:
            return index
        # First i with prefix_max[i] > page: element i - 1 is the first text of
        # a later page, so the figure goes right before it.
        return bisect_right(self._prefix_max, page_number) - 1


def section_front_matter_kind(section_path: str) -> str | None:
    stripped = section_path.strip()
    if stripped.startswith("["):
        stripped = stripped[1:]
    if stripped.endswith("]"):
        stripped = stripped[:-1]
    segments = [segment.strip() for segment in stripped.split(" > ") if segment.strip()]
    for segment in segments[1:]:
        kind = front_matter_kind(segment)
        if kind:
            return kind
    return None


class CompositeEmitter:
    def __init__(
        self,
        *,
        counter: TokenCounterFn,
        max_tokens: int = 600,
        overlap: int = 50,
        table_linearizer: object | Callable[[ParsedElement], str] | None = None,
    ) -> None:
        self.count = counter
        self.packer = SectionPacker(
            counter=counter,
            target=max_tokens,
            max_tokens=max_tokens,
            overlap=overlap,
            table_linearizer=table_linearizer,
        )

    def emit_composite(
        self,
        composite: CompositeChunk,
        *,
        document_id: str,
        source_id: str,
        section_path: str,
        section_title: str,
        doc_title: str,
    ) -> RetrievalChunk:
        title = section_title or composite.section_title
        content = figure_content(composite, section_title=title)
        # The richer window (neighbour prose included) is for retrieval only.
        window = composite.embedding_text.strip() or content
        embedding_text = f"{section_path}\n\n{window}"
        metadata: dict[str, object] = {"element_index": composite.element_index}
        metadata.update(dict(composite.metadata or {}))
        fm_kind = section_front_matter_kind(section_path)
        if fm_kind:
            metadata["front_matter_kind"] = fm_kind
        return RetrievalChunk(
            # The image id keeps two captionless figures of one page distinct.
            chunk_id=_chunk_id(source_id, f"{composite.image_id}\n{content}", "figure_composite"),
            chunk_type="figure_composite",
            document_id=document_id,
            source_id=source_id,
            page_number=composite.page_number,
            page_range=str(composite.page_number),
            section_path=section_path,
            section_title=title,
            chapter_path=derive_chapter_path(section_path, doc_title),
            content=content,
            embedding_text=embedding_text,
            token_count=self.count(embedding_text),
            has_image=True,
            image_id=composite.image_id,
            image_ref=composite.image_ref,
            vlm_description=composite.vlm_description,
            metadata=metadata,
        )

    def emit_table(
        self,
        element: ParsedElement,
        *,
        document_id: str,
        source_id: str,
        section_path: str,
        section_title: str,
        doc_title: str,
    ) -> list[RetrievalChunk]:
        content = self.packer.table_text(element)
        if not content.strip():
            return []
        table_metadata: dict[str, object] = {"element_id": element.element_id}
        fm_kind = section_front_matter_kind(section_path)
        if fm_kind:
            table_metadata["front_matter_kind"] = fm_kind
        table_parts = self.packer.split_table(content, section_path)
        chunks: list[RetrievalChunk] = []
        for part_number, part in enumerate(table_parts, start=1):
            metadata = dict(table_metadata)
            if len(table_parts) > 1:
                metadata["table_split_part"] = part_number
                metadata["table_split_total"] = len(table_parts)
            embedding_text = f"{section_path}\n\n{part}"
            chunks.append(
                RetrievalChunk(
                    chunk_id=_chunk_id(source_id, part, "table"),
                    chunk_type="table",
                    document_id=document_id,
                    source_id=source_id,
                    page_number=element.page_number,
                    page_range=(
                        str(element.page_number) if element.page_number is not None else None
                    ),
                    section_path=section_path,
                    section_title=section_title,
                    chapter_path=derive_chapter_path(section_path, doc_title),
                    content=part,
                    embedding_text=embedding_text,
                    token_count=self.count(embedding_text),
                    metadata=metadata,
                )
            )
        return chunks

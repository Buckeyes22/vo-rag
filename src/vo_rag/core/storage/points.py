"""Point model for Qdrant upserts.

`VoPoint` is the write-side point contract used by every Qdrant writer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class VoPoint(BaseModel):
    """Payload + all four vectors for one Qdrant point.

    Deterministic ID built from (source_id, chunk_index, stage[, version_hash]).
    ``version_hash`` is optional: the book_rag raw/sterile path is unversioned
    (passes None), while other pipelines still pass their own version literal.
    """

    model_config = {"extra": "forbid"}

    source_id: str
    chunk_index: int
    stage: str
    version_hash: str | None = None
    dense: list[float] = Field(min_length=1024, max_length=1024)
    sparse: dict[int, float]
    bm25: dict[int, float]
    colbert: list[list[float]]
    payload: dict[str, object]

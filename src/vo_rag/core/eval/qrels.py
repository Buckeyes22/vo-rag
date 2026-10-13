"""TREC-compatible qrels model + TSV loader for the retrieval eval gate."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import Field

from vo_rag.core.models.common import VoBaseModel


class Judgment(VoBaseModel):
    query_id: str
    doc_id: str
    grade: int = Field(ge=0, le=3)


class Qrels(VoBaseModel):
    pipeline: str
    collection: str
    judgments: list[Judgment] = Field(default_factory=list)

    def by_query(self) -> dict[str, dict[str, int]]:
        """Return {query_id: {doc_id: grade}} for direct use with metric fns."""
        out: dict[str, dict[str, int]] = {}
        for j in self.judgments:
            out.setdefault(j.query_id, {})[j.doc_id] = j.grade
        return out

    def positive_set_by_query(self, *, min_grade: int = 1) -> dict[str, set[str]]:
        """Return {query_id: {doc_id, ...}} where grade >= min_grade."""
        out: dict[str, set[str]] = {}
        for j in self.judgments:
            if j.grade >= min_grade:
                out.setdefault(j.query_id, set()).add(j.doc_id)
        return out

    def positive_count(self, query_id: str, *, min_grade: int = 1) -> int:
        return len(self.positive_set_by_query(min_grade=min_grade).get(query_id, set()))

    def all_query_ids(self) -> list[str]:
        seen: list[str] = []
        present: set[str] = set()
        for j in self.judgments:
            if j.query_id not in present:
                present.add(j.query_id)
                seen.append(j.query_id)
        return seen


def load_qrels_tsv(path: Path | str, *, pipeline: str, collection: str) -> Qrels:
    """Parse TREC qrels format:
        <query_id>\\t<iter>\\t<doc_id>\\t<grade>

    Lines starting with '#' or blank lines are ignored.
    """
    judgments: list[Judgment] = []
    text = Path(path).read_text(encoding="utf-8")
    for line_no, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split("\t")
        if len(fields) < 4:
            fields = stripped.split()
            if len(fields) < 4:
                msg = f"{path}:{line_no}: expected 4 fields, got {len(fields)}"
                raise ValueError(msg)
        try:
            grade = int(fields[3])
        except ValueError as exc:
            msg = f"{path}:{line_no}: grade must be int"
            raise ValueError(msg) from exc
        judgments.append(Judgment(query_id=fields[0], doc_id=fields[2], grade=grade))
    return Qrels(pipeline=pipeline, collection=collection, judgments=judgments)


def save_qrels_tsv(qrels: Qrels, path: Path | str) -> None:
    """Round-trippable TSV serialisation."""
    lines = [f"{j.query_id}\t0\t{j.doc_id}\t{j.grade}" for j in qrels.judgments]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def qrels_hash(qrels: Qrels) -> str:
    """sha256 of the canonical TSV serialisation - for reproducibility."""
    body = "\n".join(
        f"{j.query_id}|{j.doc_id}|{j.grade}"
        for j in sorted(qrels.judgments, key=lambda j: (j.query_id, j.doc_id))
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


__all__ = [
    "Judgment",
    "Qrels",
    "load_qrels_tsv",
    "qrels_hash",
    "save_qrels_tsv",
]

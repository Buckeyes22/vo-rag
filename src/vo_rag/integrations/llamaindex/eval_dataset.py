"""Synthetic eval-set generation over the live corpus with governed LLM injection.

Thin wrappers around LlamaIndex's dataset generators that enforce:

1. **Explicit LLM injection** — ``llm`` is a required kwarg; the functions raise
   before LlamaIndex can fall back to ``Settings.llm``.
2. **Live chunk-ID fidelity** — ``generate_eval_pairs`` produces
   ``EmbeddingQAFinetuneDataset.relevant_docs`` keyed on the *input*
   ``node.id_`` values (content-addressed chunk ids) by construction,
   closing the orphaned-qrels hole.
3. **Provenance gate** — ``assert_qrels_resolve`` raises before an orphaned
   qrels set can be adopted as a baseline (VOR-234).

`generate_rag_dataset` replicates ``RagDatasetGenerator.from_documents``
without the ``spacy`` keyword post-processor, because that dependency is not
in vo-rag's install tree.  The node-parsing behaviour (default
``SentenceSplitter``) is otherwise identical.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vo_rag.core.eval.gate import BaselineV1
from vo_rag.core.eval.non_inferiority import compute_non_inferiority
from vo_rag.core.eval.qrels import Judgment, Qrels, load_qrels_tsv, save_qrels_tsv
from vo_rag.integrations.llamaindex import require_llamaindex

if TYPE_CHECKING:
    from collections.abc import Callable

    from llama_index.core.llama_dataset import LabelledRagDataset
    from llama_index.core.llama_dataset.legacy.embedding import (
        EmbeddingQAFinetuneDataset,
    )
    from llama_index.core.llms import LLM
    from llama_index.core.schema import Document, TextNode

    from vo_rag.core.eval.harness import HarnessReport
    from vo_rag.core.eval.non_inferiority import NonInferiorityResult

DEFAULT_NON_INFERIORITY_MARGIN_PP = 2.0
DEFAULT_NON_INFERIORITY_CONFIDENCE = 0.95


def generate_eval_pairs(
    nodes: list[TextNode],
    *,
    llm: LLM,
    num_questions_per_chunk: int = 2,
) -> EmbeddingQAFinetuneDataset:
    """Wrap ``generate_question_context_pairs`` with governed-LLM enforcement.

    The returned ``EmbeddingQAFinetuneDataset.relevant_docs`` values are the
    *live* ``node.id_`` handles (content-addressed chunk ids) by construction —
    the join target for VOR-233's ``qa_dataset_to_qrels`` lowering.
    """
    if llm is None:
        raise ValueError("llm is required and must not be None")
    require_llamaindex()
    from llama_index.core.evaluation import generate_question_context_pairs

    return generate_question_context_pairs(
        nodes=nodes,
        llm=llm,
        num_questions_per_chunk=num_questions_per_chunk,
    )


def generate_rag_dataset(
    documents: list[Document],
    *,
    llm: LLM,
    num_questions_per_chunk: int = 2,
) -> LabelledRagDataset:
    """Build a ``LabelledRagDataset`` from documents with governed-LLM enforcement.

    ``RagDatasetGenerator.from_documents`` internally runs a
    ``KeywordNodePostprocessor`` that imports ``spacy``, which is not present
    in vo-rag's dependency tree.  We therefore replicate its
    document→node parsing (default ``SentenceSplitter``) and construct the
    generator directly with ``nodes=`` so the ``spacy`` import is avoided.
    The generated ``reference_contexts`` remain valid against the live corpus
    by construction.
    """
    if llm is None:
        raise ValueError("llm is required and must not be None")
    require_llamaindex()
    from llama_index.core.ingestion import run_transformations
    from llama_index.core.llama_dataset.generator import RagDatasetGenerator
    from llama_index.core.node_parser import SentenceSplitter

    # Replicate from_documents node-parsing without the spacy keyword filter.
    nodes = list(run_transformations(documents, [SentenceSplitter()]))
    generator = RagDatasetGenerator(
        nodes=nodes,
        llm=llm,
        num_questions_per_chunk=num_questions_per_chunk,
    )
    return generator.generate_dataset_from_nodes()


def qa_dataset_to_qrels(
    qa: EmbeddingQAFinetuneDataset,
    *,
    pipeline: str,
    collection: str,
) -> Qrels:
    """Transcode a LlamaIndex ``EmbeddingQAFinetuneDataset`` into vo-rag ``Qrels``.

    For each ``(query_id, node_ids)`` entry in ``qa.relevant_docs`` emit one
    ``Judgment(query_id=query_id, doc_id=node_id, grade=1)`` — the canonical
    lowering loop.

    ID fidelity (non-negotiable):
    ``doc_id`` is the *live* content-addressed chunk id carried straight from
    the dataset's ``relevant_docs`` — no re-minting, no point-id substitution.
    Re-keying would reintroduce the orphaned-qrels hole (VOR-234's provenance
    check joins on these IDs).

    Handled elsewhere:
      - chunk-id provenance vs. live collection → VOR-234
      - gate wiring (writes ``qrels.tsv`` and runs ``run_eval_gate``) → VOR-235
    """
    judgments: list[Judgment] = []
    for query_id, node_ids in qa.relevant_docs.items():
        for node_id in node_ids:
            judgments.append(Judgment(query_id=query_id, doc_id=node_id, grade=1))
    return Qrels(pipeline=pipeline, collection=collection, judgments=judgments)


def write_qrels_tsv(qrels: Qrels, path: Path | str) -> None:
    """Persist a ``Qrels`` to a TREC-format TSV (round-trippable with ``load_qrels_tsv``).

    Thin wrapper over ``vo_rag.core.eval.qrels.save_qrels_tsv`` so the
    LlamaIndex-integration package exposes a stable, import-safe surface
    for the gate-wiring unit (VOR-235) and any future caller that
    shouldn't need to know ``save_qrels_tsv`` lives in ``core/eval``.
    """
    save_qrels_tsv(qrels, path)


def refresh_book_rag_qrels(
    qa: EmbeddingQAFinetuneDataset,
    *,
    fixture_root: Path | str,
    resolve_fn: Callable[[set[str]], set[str]],
) -> Path:
    """Lower a LlamaIndex QA dataset into vo-rag qrels, verify provenance, and write the fixture.

    Parameters
    ----------
    qa:
        The LlamaIndex ``EmbeddingQAFinetuneDataset`` to lower (typically
        freshly generated from the live corpus).
    fixture_root:
        Directory under which the per-pipeline fixtures live. The qrels
        are written to ``fixture_root / "book_rag" / "qrels.tsv"``.
    resolve_fn:
        Injected resolver: ``resolve_fn(doc_ids) -> subset_that_exist``.
        Called by ``assert_qrels_resolve``; must return the set of
        ``doc_id`` values that resolve in the live collection.

    Returns
    -------
    Path
        The absolute path to the written ``qrels.tsv``.

    Raises
    ------
    QrelsProvenanceError
        If one or more ``doc_id`` values do not resolve. The fixture is
        **not** written on failure (non-destructive contract).
    """
    qrels = qa_dataset_to_qrels(qa, pipeline="book_rag", collection="eval")
    assert_qrels_resolve(qrels, resolve_fn=resolve_fn)
    out_path = Path(fixture_root) / "book_rag" / "qrels.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_qrels_tsv(qrels, out_path)
    return out_path.resolve()


def assert_book_rag_eval_snapshot_resolves_qrels(
    qrels_path: Path | str,
    snapshot_path: Path | str,
) -> QrelsProvenanceReport:
    """Assert every book_rag qrels ``doc_id`` exists in the corpus snapshot."""
    snapshot_doc_ids: set[str] = set()
    for raw in Path(snapshot_path).read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        snapshot_doc_ids.add(str(record["chunk_id"]))

    qrels = load_qrels_tsv(qrels_path, pipeline="book_rag", collection="eval")
    return assert_qrels_resolve(
        qrels,
        resolve_fn=lambda ids: ids & snapshot_doc_ids,
    )


def freeze_baseline(
    report: HarnessReport, *, path: Path | str, corpus_snapshot: str
) -> dict[str, Any]:
    """Freeze a harness report's aggregate metrics as a regression baseline.

    Writes an explicit schema that binds the baseline to the qrels hash and
    corpus snapshot so stale baselines cannot silently match the wrong
    qrels/corpus.
    """
    baseline = BaselineV1(
        schema_version=1,
        pipeline=report.pipeline,
        qrels_hash=report.qrels_hash,
        corpus_snapshot=corpus_snapshot,
        created_at=datetime.now(UTC).isoformat(),
        aggregate=dict(report.aggregate),
    ).model_dump(mode="json")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return baseline


def assert_non_inferior_to_baseline(
    control_scores: Sequence[float],
    treatment_scores: Sequence[float],
    *,
    metric: str,
    margin_pp: float = DEFAULT_NON_INFERIORITY_MARGIN_PP,
    confidence: float = DEFAULT_NON_INFERIORITY_CONFIDENCE,
) -> NonInferiorityResult:
    """Assert treatment scores clear the pre-registered non-inferiority bar."""
    result = compute_non_inferiority(
        control_scores=list(control_scores),
        treatment_scores=list(treatment_scores),
        metric=metric,
        margin_pp=margin_pp,
        confidence=confidence,
    )
    if result.decision != "non_inferior":
        raise AssertionError(
            f"{metric} non-inferiority failed: decision={result.decision}, "
            f"delta_mean={result.delta_mean:.6f}, "
            f"delta_ci_low={result.delta_ci_low:.6f}, "
            f"margin_pp={margin_pp}, confidence={confidence}"
        )
    return result


@dataclass(frozen=True, slots=True)
class QrelsProvenanceReport:
    """Result of a successful provenance check."""

    resolved: int
    total: int


class QrelsProvenanceError(ValueError):
    """Raised when one or more qrel doc_ids do not resolve in the live collection.

    Carries the unresolved ids so callers can diagnose the orphaning root
    cause (e.g. a corpus re-ingest that changed chunk IDs).
    """

    def __init__(self, unresolved: set[str]) -> None:
        self.unresolved = unresolved
        super().__init__(
            f"{len(unresolved)} qrel doc_id(s) not found in live collection: {sorted(unresolved)!r}"
        )


def assert_qrels_resolve(
    qrels: Qrels,
    *,
    resolve_fn: Callable[[set[str]], set[str]],
) -> QrelsProvenanceReport:
    """Assert that every ``Judgment.doc_id`` in *qrels* resolves live.

    Parameters
    ----------
    qrels:
        The qrels set to validate (typically freshly regenerated).
    resolve_fn:
        Injected resolver: ``resolve_fn(doc_ids) -> subset_that_exist``.
        In production this queries the target collection by ``chunk_id``
        (a KEYWORD payload index — see ``qdrant.py:155``). In unit tests it
        is a fake that returns the intersection with a hard-coded set.

    Returns
    -------
    QrelsProvenanceReport
        ``(resolved, total)`` counts on success.

    Raises
    ------
    QrelsProvenanceError
        If **any** ``doc_id`` is missing from the live set. The message
        lists every unresolved id. A bad generation run can only fail the
        gate, never corrupt the corpus (eval-gate-as-release-gate #6).

        Adjacent: self-retrieval smoke threshold (``score_self_retrieval`` /
        ``gate_decision`` in ``book_rag/retrieval_smoke.py``) is a companion
        soft gate that must also clear before the new qrels is trusted;
        wiring both together is VOR-235's responsibility.
    """
    all_doc_ids: set[str] = {j.doc_id for j in qrels.judgments}
    resolved_ids = resolve_fn(all_doc_ids)
    unresolved = all_doc_ids - resolved_ids
    if unresolved:
        raise QrelsProvenanceError(unresolved)
    return QrelsProvenanceReport(resolved=len(resolved_ids), total=len(all_doc_ids))


__all__ = [
    "assert_non_inferior_to_baseline",
    "assert_book_rag_eval_snapshot_resolves_qrels",
    "assert_qrels_resolve",
    "DEFAULT_NON_INFERIORITY_CONFIDENCE",
    "DEFAULT_NON_INFERIORITY_MARGIN_PP",
    "freeze_baseline",
    "generate_eval_pairs",
    "generate_rag_dataset",
    "qa_dataset_to_qrels",
    "QrelsProvenanceError",
    "QrelsProvenanceReport",
    "refresh_book_rag_qrels",
    "write_qrels_tsv",
]

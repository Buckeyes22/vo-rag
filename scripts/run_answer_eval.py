#!/usr/bin/env python3
"""Run protected answer-quality evaluation over the book_rag eval set."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from vo_rag.core.answer.synthesizer import AnswerProfile, AnswerSynthesizer
from vo_rag.core.config import get_settings
from vo_rag.core.llm.client import LLMClient
from vo_rag.core.models.common import VoBaseModel
from vo_rag.core.retrieval.search import SearchResult
from vo_rag.integrations.llamaindex.answer_eval import (
    AnswerJudgeResult,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    answer_eval_prompt_hashes,
)
from vo_rag.integrations.llamaindex.convert import search_result_to_node_with_score
from vo_rag.integrations.llamaindex.llm import VoRagLLM
from vo_rag.integrations.llamaindex.synthesizer import (
    VO_RAG_RESPONSE_METADATA_KEY,
    VoRagSynthesizer,
)
from vo_rag.pipelines.book_rag.eval import (
    BOOK_RAG_EVAL_COLLECTION,
    BOOK_RAG_EVAL_CORPUS_SNAPSHOT,
    SnapshotBookRagRetriever,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "eval" / "book_rag"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "artifacts" / "eval"
_DEFAULT_CONCURRENCY = 8
_DEFAULT_TOP_K = 10
_MAXIMAL_SCORE = 0.999
_SELF_TEST_MIN_DROP = 0.05
_UNRELATED_SELF_TEST_ANSWER = (
    "This answer discusses tire tread polymers, glass transition temperature, "
    "and rolling resistance rather than the question asked."
)

ProviderName = Literal["anthropic", "llama_cpp", "kimi", "runpod_serverless"]


class FabricationSmellError(RuntimeError):
    """Raised when answer-eval scores look mechanically fabricated."""


class AnswerEvalProvenance(VoBaseModel):
    judge_model: str
    answer_profile: str
    query_set_hash: str
    snapshot_hash: str
    prompt_hashes: dict[str, str]
    created_at: str


class QueryAnswerEvalRecord(VoBaseModel):
    schema_version: Literal[1] = 1
    type: Literal["query"] = "query"
    query_id: str
    query_text: str
    retrieved_chunk_ids: list[str]
    answer: str
    answer_model: str
    faithfulness: AnswerJudgeResult
    relevancy: AnswerJudgeResult


class AggregateAnswerEvalRecord(VoBaseModel):
    schema_version: Literal[1] = 1
    type: Literal["aggregate"] = "aggregate"
    n_queries: int
    aggregate: dict[str, float]
    provenance: AnswerEvalProvenance


@dataclass(frozen=True, slots=True)
class EvalQuery:
    query_id: str
    query_text: str


@dataclass(frozen=True, slots=True)
class EvaluatedQuery:
    record: QueryAnswerEvalRecord
    context: list[SearchResult]


class _SearchHit(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def score(self) -> float: ...

    @property
    def payload(self) -> dict[str, Any]: ...


def load_queries(path: Path | str) -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    query_path = Path(path)
    for line_no, raw in enumerate(query_path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        if not stripped:
            continue
        record: dict[str, Any] = json.loads(stripped)
        query_id = str(record.get("query_id") or "").strip()
        query_text = str(record.get("query_text") or record.get("text") or "").strip()
        if not query_id or not query_text:
            raise ValueError(f"{query_path}:{line_no}: query_id and query_text/text are required")
        queries.append(EvalQuery(query_id=query_id, query_text=query_text))
    if not queries:
        raise ValueError(f"{query_path}: no queries found")
    return queries


async def run_answer_eval(
    *,
    fixture_dir: Path | str = _DEFAULT_FIXTURE_DIR,
    output: Path | str | None = None,
    compare: Path | str | None = None,
    concurrency: int = _DEFAULT_CONCURRENCY,
    top_k: int = _DEFAULT_TOP_K,
    limit: int | None = None,
    self_test: bool = False,
    answer_profile: AnswerProfile = AnswerProfile.QUALITY,
    judge_provider: ProviderName = "runpod_serverless",
    judge_role: str = "text",
    judge_model: str | None = None,
    judge_api_key: str | None = None,
    judge_base_url: str | None = None,
) -> dict[str, Any]:
    """Run answer-quality eval and write JSONL records plus an aggregate."""
    fixture_path = Path(fixture_dir)
    queries_path = fixture_path / "queries.jsonl"
    snapshot_path = fixture_path / BOOK_RAG_EVAL_CORPUS_SNAPSHOT
    queries = load_queries(queries_path)
    if limit is not None:
        queries = queries[:limit]
    if not queries:
        raise ValueError("limit selected zero queries")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    judge_client = LLMClient(
        provider=judge_provider,
        model=judge_model,
        api_key=judge_api_key,
        base_url=judge_base_url,
        role=judge_role,
    )
    judge_llm = VoRagLLM(client=judge_client)
    try:
        evaluated = await _evaluate_queries(
            queries=queries,
            snapshot_path=snapshot_path,
            top_k=top_k,
            concurrency=concurrency,
            judge_llm=judge_llm,
            answer_profile=answer_profile,
        )
        records = [item.record for item in evaluated]
        if self_test:
            self_test_metrics = await _run_self_test(
                evaluated,
                faithfulness=FaithfulnessEvaluator(judge_llm),
                min_drop=_SELF_TEST_MIN_DROP,
            )
        else:
            self_test_metrics = {}

        dumped_records = [record.model_dump(mode="json") for record in records]
        assert_not_uniformly_maximal(dumped_records)

        aggregate = _build_aggregate(
            records,
            provenance=AnswerEvalProvenance(
                judge_model=judge_llm.model_name,
                answer_profile=answer_profile.value,
                query_set_hash=sha256_file(queries_path),
                snapshot_hash=sha256_file(snapshot_path),
                prompt_hashes=answer_eval_prompt_hashes(),
                created_at=datetime.now(UTC).isoformat(),
            ),
            self_test_metrics=self_test_metrics,
        )
        output_path = Path(output) if output is not None else _default_output_path()
        write_answer_eval_jsonl(records, aggregate, output_path)
        result: dict[str, Any] = {
            "output": str(output_path),
            "n_queries": len(records),
            "aggregate": aggregate.aggregate,
            "provenance": aggregate.provenance.model_dump(mode="json"),
        }
        if compare is not None:
            result["comparison"] = compare_answer_eval_runs(output_path, compare)
        return result
    finally:
        await judge_client.aclose()


async def _evaluate_queries(
    *,
    queries: Sequence[EvalQuery],
    snapshot_path: Path,
    top_k: int,
    concurrency: int,
    judge_llm: VoRagLLM,
    answer_profile: AnswerProfile,
) -> list[EvaluatedQuery]:
    retriever = SnapshotBookRagRetriever(snapshot_path)
    answer = AnswerSynthesizer(get_settings().answer)
    synthesizer = VoRagSynthesizer(answer=answer, profile=answer_profile, llm=judge_llm)
    faithfulness = FaithfulnessEvaluator(judge_llm)
    relevancy = RelevancyEvaluator(judge_llm)
    semaphore = asyncio.Semaphore(concurrency)

    async def _one(query: EvalQuery) -> EvaluatedQuery:
        async with semaphore:
            hits = await retriever.hybrid_search(
                query.query_text,
                collection=BOOK_RAG_EVAL_COLLECTION,
                top_k=top_k,
            )
            context = _hits_to_search_results(hits)
            answer_text, answer_model = await _synthesize_answer(
                synthesizer,
                query=query.query_text,
                context=context,
            )
            faithfulness_result, relevancy_result = await asyncio.gather(
                faithfulness.aevaluate(
                    question=query.query_text,
                    answer=answer_text,
                    context=context,
                ),
                relevancy.aevaluate(
                    question=query.query_text,
                    answer=answer_text,
                    context=context,
                ),
            )
            return EvaluatedQuery(
                record=QueryAnswerEvalRecord(
                    query_id=query.query_id,
                    query_text=query.query_text,
                    retrieved_chunk_ids=_retrieved_chunk_ids(context),
                    answer=answer_text,
                    answer_model=answer_model,
                    faithfulness=faithfulness_result,
                    relevancy=relevancy_result,
                ),
                context=context,
            )

    return list(await asyncio.gather(*(_one(query) for query in queries)))


async def _synthesize_answer(
    synthesizer: VoRagSynthesizer,
    *,
    query: str,
    context: Sequence[SearchResult],
) -> tuple[str, str]:
    nodes = [search_result_to_node_with_score(result) for result in context]
    response = await synthesizer.asynthesize(query, nodes)
    return str(response), _answer_model_from_response(response)


async def _run_self_test(
    evaluated: Sequence[EvaluatedQuery],
    *,
    faithfulness: FaithfulnessEvaluator,
    min_drop: float,
) -> dict[str, float]:
    if not evaluated:
        raise FabricationSmellError("self-test requires at least one evaluated query")

    target = max(evaluated, key=lambda item: item.record.faithfulness.score)
    original_score = target.record.faithfulness.score
    swapped_answer = _swapped_answer(evaluated, target)
    swapped = await faithfulness.aevaluate(
        question=target.record.query_text,
        answer=swapped_answer,
        context=target.context,
    )
    drop = original_score - swapped.score
    if drop < min_drop:
        raise FabricationSmellError(
            "anti-fabrication self-test failed: swapped unrelated answer did not "
            f"drop faithfulness measurably; original={original_score:.3f}, "
            f"swapped={swapped.score:.3f}, drop={drop:.3f}, required={min_drop:.3f}"
        )
    return {
        "self_test_original_faithfulness": original_score,
        "self_test_swapped_faithfulness": swapped.score,
        "self_test_faithfulness_drop": drop,
    }


def _swapped_answer(evaluated: Sequence[EvaluatedQuery], target: EvaluatedQuery) -> str:
    target_chunks = set(target.record.retrieved_chunk_ids)
    for item in evaluated:
        if (
            item.record.query_id != target.record.query_id
            and item.record.answer.strip()
            and target_chunks.isdisjoint(item.record.retrieved_chunk_ids)
        ):
            return item.record.answer
    for item in evaluated:
        if item.record.query_id != target.record.query_id and item.record.answer.strip():
            return item.record.answer
    return _UNRELATED_SELF_TEST_ANSWER


def _hits_to_search_results(hits: Sequence[_SearchHit]) -> list[SearchResult]:
    results: list[SearchResult] = []
    for hit in hits:
        payload = dict(hit.payload)
        results.append(
            SearchResult(
                id=str(hit.id),
                score=float(hit.score),
                collection=str(payload.get("collection") or BOOK_RAG_EVAL_COLLECTION),
                payload=payload,
            )
        )
    return results


def _retrieved_chunk_ids(context: Sequence[SearchResult]) -> list[str]:
    out: list[str] = []
    for result in context:
        chunk_id = result.payload.get("chunk_id") or result.id
        out.append(str(chunk_id))
    return out


def _answer_model_from_response(response: object) -> str:
    metadata = getattr(response, "metadata", None)
    if isinstance(metadata, Mapping):
        vo_rag = metadata.get(VO_RAG_RESPONSE_METADATA_KEY)
        if isinstance(vo_rag, Mapping):
            model = vo_rag.get("model")
            if isinstance(model, str):
                return model
    return ""


def _build_aggregate(
    records: Sequence[QueryAnswerEvalRecord],
    *,
    provenance: AnswerEvalProvenance,
    self_test_metrics: Mapping[str, float],
) -> AggregateAnswerEvalRecord:
    faithfulness_scores = [record.faithfulness.score for record in records]
    relevancy_scores = [record.relevancy.score for record in records]
    quarantined = [
        record
        for record in records
        if record.faithfulness.should_quarantine or record.relevancy.should_quarantine
    ]
    aggregate = {
        "faithfulness_mean": _mean(faithfulness_scores),
        "faithfulness_min": min(faithfulness_scores),
        "faithfulness_variance": _variance(faithfulness_scores),
        "relevancy_mean": _mean(relevancy_scores),
        "relevancy_min": min(relevancy_scores),
        "relevancy_variance": _variance(relevancy_scores),
        "quarantine_rate": len(quarantined) / len(records),
        **dict(self_test_metrics),
    }
    return AggregateAnswerEvalRecord(
        n_queries=len(records),
        aggregate=aggregate,
        provenance=provenance,
    )


def write_answer_eval_jsonl(
    records: Sequence[QueryAnswerEvalRecord],
    aggregate: AggregateAnswerEvalRecord,
    path: Path | str,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        for record in records
    ]
    lines.append(json.dumps(aggregate.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def assert_not_uniformly_maximal(records: Sequence[Mapping[str, Any]]) -> None:
    query_records = [record for record in records if record.get("type") == "query"]
    if not query_records:
        return
    if all(
        _score_from_record(record, "faithfulness") >= _MAXIMAL_SCORE
        and _score_from_record(record, "relevancy") >= _MAXIMAL_SCORE
        for record in query_records
    ):
        raise FabricationSmellError(
            "refusing answer-eval run with uniformly perfect faithfulness and "
            "relevancy scores — uniformly perfect judge output is a fabrication smell"
        )


def compare_answer_eval_runs(
    current: Path | str,
    other: Path | str,
) -> dict[str, float | int]:
    current_records = _load_query_record_maps(current)
    other_records = _load_query_record_maps(other)
    matched = sorted(set(current_records) & set(other_records))
    if not matched:
        raise ValueError("no overlapping query_id values to compare")

    result: dict[str, float | int] = {"matched_queries": len(matched)}
    for evaluator in ("faithfulness", "relevancy"):
        deltas = [
            _score_from_record(current_records[query_id], evaluator)
            - _score_from_record(other_records[query_id], evaluator)
            for query_id in matched
        ]
        result[f"{evaluator}_mean_delta"] = _mean(deltas)
    return result


def sha256_file(path: Path | str) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_query_record_maps(path: Path | str) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for record in _read_jsonl(path):
        if record.get("type") != "query":
            continue
        query_id = record.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"{path}: query record missing query_id")
        records[query_id] = record
    return records


def _read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    input_path = Path(path)
    for raw in input_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        record: dict[str, Any] = json.loads(stripped)
        records.append(record)
    return records


def _score_from_record(record: Mapping[str, Any], evaluator: str) -> float:
    section = record.get(evaluator)
    if not isinstance(section, Mapping):
        raise ValueError(f"query record missing {evaluator!r} score section")
    raw_score = section.get("score")
    if not isinstance(raw_score, int | float):
        raise ValueError(f"query record {evaluator!r} score is not numeric")
    return float(raw_score)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _default_output_path() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return _DEFAULT_OUTPUT_DIR / f"book_rag_answer_eval_{stamp}.jsonl"


def _parse_provider(value: str) -> ProviderName:
    if value in {"anthropic", "llama_cpp", "kimi", "runpod_serverless"}:
        return cast(ProviderName, value)
    raise argparse.ArgumentTypeError(f"unsupported judge provider: {value}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, default=_DEFAULT_FIXTURE_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--concurrency", type=int, default=_DEFAULT_CONCURRENCY)
    parser.add_argument("--top-k", type=int, default=_DEFAULT_TOP_K)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--answer-profile",
        choices=[profile.value for profile in AnswerProfile],
        default=AnswerProfile.QUALITY.value,
    )
    parser.add_argument("--judge-provider", type=_parse_provider, default="runpod_serverless")
    parser.add_argument("--judge-role", default="text")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-base-url", default=None)
    return parser.parse_args(argv)


async def amain(args: argparse.Namespace) -> int:
    result = await run_answer_eval(
        fixture_dir=args.fixture_dir,
        output=args.output,
        compare=args.compare,
        concurrency=args.concurrency,
        top_k=args.top_k,
        limit=args.limit,
        self_test=args.self_test,
        answer_profile=AnswerProfile(args.answer_profile),
        judge_provider=args.judge_provider,
        judge_role=args.judge_role,
        judge_model=args.judge_model,
        judge_api_key=args.judge_api_key,
        judge_base_url=args.judge_base_url,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(amain(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())

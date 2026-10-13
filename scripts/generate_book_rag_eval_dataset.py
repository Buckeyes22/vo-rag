#!/usr/bin/env python3
"""Regenerate the book_rag eval dataset from the LIVE corpus.

Samples real ``text`` chunks from the live Qdrant collection, generates
questions FROM each chunk's content via the governed LLM substrate
(``LLMClient`` → RunPod serverless), lowers to qrels via
``refresh_book_rag_qrels`` (provenance-gated: doc_id = live chunk_id,
never re-keyed), writes ``queries.jsonl``, and emits the REAL-chunk source
export consumed by ``scripts/rebuild_book_rag_eval_corpus.py``.

Honesty properties, by construction:

- Questions are generated FROM real chunk text, so the judged chunk is ground
  truth without consulting any retriever (no circularity with the system
  under test).
- The source export contains the judged chunks PLUS a large distractor sample
  (default 50x) drawn from the same live collection, so snapshot retrieval is
  a real needle-in-haystack task and the frozen baseline cannot be perfect.
- Every artifact records provenance: collection, qdrant url, model, endpoint,
  seed, counts, and sha256 hashes.

Requires env: RUNPOD_API_KEY, SERVERLESS_TEXT_ENDPOINT_ID,
SERVERLESS_TEXT_MODEL_NAME (and SERVERLESS_VLM_ENDPOINT_ID for settings
interpolation, unused here).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_FIXTURE_ROOT = _REPO_ROOT / "tests" / "fixtures" / "eval"
_DEFAULT_COLLECTION = "vo_rag_books_raw_v1"

_MIN_CHUNK_CHARS = 400
_MAX_CHUNK_CHARS = 3000


def _scroll_chunk_index(
    client: httpx.Client, base_url: str, collection: str
) -> list[tuple[str, str]]:
    """Return [(point_id, chunk_id)] for every ``text`` chunk in the collection."""
    out: list[tuple[str, str]] = []
    offset: Any = None
    while True:
        body: dict[str, Any] = {
            "limit": 10000,
            "with_payload": ["chunk_id"],
            "with_vector": False,
            "filter": {"must": [{"key": "chunk_type", "match": {"value": "text"}}]},
        }
        if offset is not None:
            body["offset"] = offset
        resp = client.post(f"{base_url}/collections/{collection}/points/scroll", json=body)
        resp.raise_for_status()
        result = resp.json()["result"]
        for point in result["points"]:
            chunk_id = str(point["payload"].get("chunk_id") or "")
            if chunk_id:
                out.append((str(point["id"]), chunk_id))
        offset = result.get("next_page_offset")
        if offset is None:
            return out


def _retrieve_points(
    client: httpx.Client, base_url: str, collection: str, point_ids: list[str]
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for start in range(0, len(point_ids), 100):
        batch = point_ids[start : start + 100]
        resp = client.post(
            f"{base_url}/collections/{collection}/points",
            json={"ids": batch, "with_payload": True, "with_vector": False},
        )
        resp.raise_for_status()
        payloads.extend(p["payload"] for p in resp.json()["result"])
    return payloads


def _export_record(payload: dict[str, Any]) -> dict[str, str] | None:
    chunk_id = str(payload.get("chunk_id") or "").strip()
    source_id = str(payload.get("source_id") or payload.get("canonical_source_id") or "").strip()
    text = str(payload.get("content") or "").strip()
    if not (chunk_id and source_id and text):
        return None
    return {"chunk_id": chunk_id, "source_id": source_id, "text": text}


def _resolve_fn_factory(
    client: httpx.Client, base_url: str, collection: str
) -> Callable[[set[str]], set[str]]:
    def _resolve(doc_ids: set[str]) -> set[str]:
        resolved: set[str] = set()
        ids = sorted(doc_ids)
        for start in range(0, len(ids), 200):
            batch = ids[start : start + 200]
            offset: Any = None
            # Paginate to exhaustion: multiple points can share a chunk_id, so
            # a single page sized to the batch can crowd out distinct matches.
            while True:
                body: dict[str, Any] = {
                    "limit": 500,
                    "with_payload": ["chunk_id"],
                    "with_vector": False,
                    "filter": {"must": [{"key": "chunk_id", "match": {"any": batch}}]},
                }
                if offset is not None:
                    body["offset"] = offset
                resp = client.post(f"{base_url}/collections/{collection}/points/scroll", json=body)
                resp.raise_for_status()
                result = resp.json()["result"]
                for point in result["points"]:
                    resolved.add(str(point["payload"]["chunk_id"]))
                offset = result.get("next_page_offset")
                if offset is None:
                    break
        return resolved

    return _resolve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", required=True, help="e.g. http://localhost:6400")
    parser.add_argument("--collection", default=_DEFAULT_COLLECTION)
    parser.add_argument("--num-chunks", type=int, default=30)
    parser.add_argument("--questions-per-chunk", type=int, default=2)
    parser.add_argument("--distractors", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--fixture-root", type=Path, default=_DEFAULT_FIXTURE_ROOT)
    parser.add_argument(
        "--export-output",
        type=Path,
        default=_DEFAULT_FIXTURE_ROOT / "book_rag" / "source_export.jsonl",
        help="REAL-chunk export consumed by rebuild_book_rag_eval_corpus.py "
        "(not checked in; the snapshot built from it is)",
    )
    args = parser.parse_args()

    from llama_index.core.schema import TextNode

    from vo_rag.core.llm.client import LLMClient
    from vo_rag.integrations.llamaindex.eval_dataset import refresh_book_rag_qrels
    from vo_rag.integrations.llamaindex.llm import VoRagLLM

    rng = random.Random(args.seed)
    base_url = args.qdrant_url.rstrip("/")

    with httpx.Client(timeout=30.0) as qdrant:
        index = _scroll_chunk_index(qdrant, base_url, args.collection)
        print(f"live text chunks: {len(index)}")
        if len(index) < args.num_chunks + args.distractors:
            raise SystemExit("collection smaller than requested sample + distractors")

        # Oversample judged candidates, then filter by content length.
        candidate_ids = rng.sample(index, min(args.num_chunks * 5, len(index)))
        candidates = _retrieve_points(
            qdrant, base_url, args.collection, [pid for pid, _ in candidate_ids]
        )
        judged_records: list[dict[str, str]] = []
        for payload in candidates:
            record = _export_record(payload)
            if record is None:
                continue
            if not (_MIN_CHUNK_CHARS <= len(record["text"]) <= _MAX_CHUNK_CHARS):
                continue
            judged_records.append(record)
            if len(judged_records) >= args.num_chunks:
                break
        if len(judged_records) < args.num_chunks:
            raise SystemExit(f"only {len(judged_records)} usable chunks after length filtering")

        judged_chunk_ids = {r["chunk_id"] for r in judged_records}
        distractor_pool = [(pid, cid) for pid, cid in index if cid not in judged_chunk_ids]
        distractor_ids = rng.sample(distractor_pool, args.distractors)
        distractor_payloads = _retrieve_points(
            qdrant, base_url, args.collection, [pid for pid, _ in distractor_ids]
        )
        distractor_records = [
            r for p in distractor_payloads if (r := _export_record(p)) is not None
        ]

        # Generate questions FROM the judged chunks via the governed substrate.
        # NOTE: this replicates llama-index's generate_question_context_pairs
        # contract (same prompt template, same numbered-line parsing) but
        # drives the ASYNC adapter path in one event loop with bounded
        # concurrency. The sync path (generate_eval_pairs -> llm.complete via
        # run_coro_sync) opens a fresh event loop per call while the shared
        # LLMClient's connection pool stays bound to the first loop — the
        # second call dies with "Event loop is closed". Async is the
        # production pattern; sync complete() is single-call-only.
        import asyncio
        import re as _re
        import uuid

        from llama_index.core.llama_dataset.legacy.embedding import (
            DEFAULT_QA_GENERATE_PROMPT_TMPL,
            EmbeddingQAFinetuneDataset,
        )

        nodes = [TextNode(id_=r["chunk_id"], text=r["text"]) for r in judged_records]
        llm = VoRagLLM(client=LLMClient(provider="runpod_serverless", role="text"))
        print(
            f"generating {args.questions_per_chunk} question(s) x {len(nodes)} chunks "
            f"via {llm.model_name} ..."
        )

        async def _generate() -> EmbeddingQAFinetuneDataset:
            semaphore = asyncio.Semaphore(8)
            node_dict = {node.node_id: node.get_content() for node in nodes}

            async def _one(node_id: str, text: str) -> tuple[str, list[str]]:
                prompt = (
                    DEFAULT_QA_GENERATE_PROMPT_TMPL.format(
                        context_str=text,
                        num_questions_per_chunk=args.questions_per_chunk,
                    )
                    + "\nOutput ONLY the questions, one per line, each ending"
                    " with a question mark. No preamble, no analysis, no"
                    " headings, no numbering explanations."
                )
                async with semaphore:
                    response = await llm.acomplete(prompt)
                # Strict interrogative-span extraction: reasoning-style models
                # wrap output in preamble/labels ("Point 5:*", "Q1:",
                # "**Deconstruct the Context:**") that line-based parsing
                # swallows. Extract clean question sentences only.
                interrogative = _re.compile(
                    r"\b(?:How|What|Why|When|Where|Which|Who|Whose|Whom|Is|Are"
                    r"|Do|Does|Did|Can|Could|Should|Would|Will|If|Under what"
                    r"|In what|To what|Based on|According to|Given)\b[^?\n]{10,400}\?",
                )
                questions: list[str] = []
                for match in interrogative.finditer(str(response)):
                    candidate = match.group(0).strip()
                    if len(candidate) >= 25 and candidate not in questions:
                        questions.append(candidate)
                questions = questions[: args.questions_per_chunk]
                print(f"  {node_id[:12]}: {len(questions)} question(s)")
                return node_id, questions

            results = await asyncio.gather(*(_one(nid, text) for nid, text in node_dict.items()))
            queries: dict[str, str] = {}
            relevant_docs: dict[str, list[str]] = {}
            for node_id, questions in results:
                for question in questions:
                    # Deterministic id: same chunk + question -> same query_id.
                    qid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{node_id}:{question}"))
                    queries[qid] = question
                    relevant_docs[qid] = [node_id]
            return EmbeddingQAFinetuneDataset(
                queries=queries, corpus=node_dict, relevant_docs=relevant_docs
            )

        qa = asyncio.run(_generate())

        # Drop degenerate questions, then adopt via the provenance-gated path.
        kept_queries = {
            qid: text.strip() for qid, text in qa.queries.items() if len(text.strip()) >= 20
        }
        qa.queries = kept_queries
        qa.relevant_docs = {qid: qa.relevant_docs[qid] for qid in kept_queries}
        print(f"questions kept after degeneracy filter: {len(kept_queries)}")
        expected = args.num_chunks * args.questions_per_chunk
        if len(kept_queries) < max(10, expected // 2):
            raise SystemExit(
                f"only {len(kept_queries)}/{expected} usable questions generated — "
                "refusing to write a degenerate eval fixture; inspect the model "
                "output format before retrying"
            )

        qrels_path = refresh_book_rag_qrels(
            qa,
            fixture_root=args.fixture_root,
            resolve_fn=_resolve_fn_factory(qdrant, base_url, args.collection),
        )

    generated_at = datetime.now(UTC).isoformat()
    queries_path = args.fixture_root / "book_rag" / "queries.jsonl"
    queries_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "query_id": qid,
                    "query_text": text,
                    "generated_from_live_corpus": True,
                    "generation_model": llm.model_name,
                    "generated_at": generated_at,
                }
            )
            for qid, text in kept_queries.items()
        )
        + "\n",
        encoding="utf-8",
    )

    export_records = sorted(judged_records + distractor_records, key=lambda r: r["chunk_id"])
    args.export_output.parent.mkdir(parents=True, exist_ok=True)
    args.export_output.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in export_records) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "collection": args.collection,
                "qdrant_url": base_url,
                "seed": args.seed,
                "judged_chunks": len(judged_records),
                "queries": len(kept_queries),
                "distractors": len(distractor_records),
                "qrels_path": str(qrels_path),
                "queries_path": str(queries_path),
                "export_path": str(args.export_output),
                "export_sha256": hashlib.sha256(args.export_output.read_bytes()).hexdigest(),
                "generation_model": llm.model_name,
                "generated_at": generated_at,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

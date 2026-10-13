#!/usr/bin/env python3
"""Acceptance eval against the REMOTE localhost stack (VOR-8).

Drives retrieval entirely over HTTP — EmbeddingClient (query embedding) +
RerankerClient (rerank) against the localhost service — and runs the real
harness over the re-derived benchmark, reproducing the bars end-to-end through
the wire path that production will use.

Env:
  VORAG_EMBED_URL    e.g. http://localhost:8501/embed
  VORAG_RERANK_URL   e.g. http://localhost:8501/rerank
  QDRANT_URL         e.g. http://127.0.0.1:6400
Optional: NDCG_DIR (default /data/vo-rag/eval_out/ndcg), RERANK_DEPTH (100).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from vo_rag.core.embedding.client import EmbeddingClient
from vo_rag.core.eval.harness import run_harness
from vo_rag.core.eval.qrels import load_qrels_tsv
from vo_rag.core.reranking.client import RerankerClient
from vo_rag.core.retrieval.search import HybridRetriever
from vo_rag.core.storage.qdrant import QdrantStore
from vo_rag.pipelines.book_rag.eval import book_rag_variant
from vo_rag.pipelines.book_rag.retrieval_smoke import chunk_id_from_result

COLL = "vo_rag_books_raw_v1"


def _load_query_text(path: Path) -> dict[str, str]:
    import json

    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            out[str(r["query_id"])] = str(r.get("query_text") or r.get("text") or "")
    return out


async def main() -> None:
    ndcg_dir = Path(os.environ.get("NDCG_DIR", "/data/vo-rag/eval_out/ndcg"))
    depth = int(os.environ.get("RERANK_DEPTH", "100"))

    embed_url = os.environ["VORAG_EMBED_URL"]
    rerank_url = os.environ["VORAG_RERANK_URL"]
    embed = EmbeddingClient(embed_url, timeout_s=120)
    reranker = RerankerClient(rerank_url, timeout_s=120)
    qdrant = QdrantStore(os.environ["QDRANT_URL"], None, prefer_grpc=False)
    retriever = HybridRetriever(qdrant, embed)
    print(
        f"remote stack: embed={embed_url} rerank={rerank_url}",
        flush=True,
    )

    qrels = load_qrels_tsv(
        ndcg_dir / "qrels.tsv", pipeline="book_rag", collection="vo_rag_eval_rederived"
    )
    q_text = _load_query_text(ndcg_dir / "queries.jsonl")
    print(f"benchmark: {len(q_text)} queries", flush=True)

    async def search_fn(query: str, top_k: int) -> list[str]:
        cands = await retriever.hybrid_search(
            query, collection=COLL, top_k=depth, prefetch_depth=depth
        )
        reranked = await reranker.rerank(query, list(cands), top_k=top_k)
        return [chunk_id_from_result(r.id, r.payload) for r in reranked]

    report = await run_harness(
        "book_rag",
        qrels,
        book_rag_variant(),
        search_fn=search_fn,
        query_text_fn=lambda qid: q_text.get(qid, qid),
    )
    agg = report.aggregate
    print("\n=== REMOTE STACK nDCG (localhost embed + rerank over HTTP) ===", flush=True)
    for k in ("recall@1", "ndcg@1", "recall@5", "ndcg@5", "recall@10", "ndcg@10", "mrr"):
        if k in agg:
            print(f"  {k:10} = {agg[k]:.4f}", flush=True)
    floors = {"ndcg@10": 0.86, "recall@10": 0.93, "mrr": 0.88}
    print("\n=== vs FLOORS ===", flush=True)
    for k, fl in floors.items():
        got = agg.get(k, 0.0)
        print(f"  {k:10} = {got:.4f} (floor {fl}) {'PASS' if got >= fl else 'BELOW'}", flush=True)
    await embed.aclose()
    await reranker.aclose()


if __name__ == "__main__":
    asyncio.run(main())

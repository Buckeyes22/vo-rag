"""Source-faithful brain points through real writer/executor, without external services."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from vo_rag.api.singletons import SurfaceServices
from vo_rag.core.retrieval.search import SearchResult
from vo_rag.core.storage.qdrant import deterministic_point_id
from vo_rag.ops import brain_sync

API_KEY = "brain-boundary-test-key"  # pragma: allowlist secret
BRAIN = "vo_rag_brain_v1"


def source_hits() -> list[SearchResult]:
    hits = [
        SearchResult(
            id="foreign",
            score=1.0,
            collection="vo_rag_books_raw_v1",
            payload={"content": "Unrequested library material."},
        )
    ]
    for path, repetitions, condition in [
        ("wiki/retention.md", 200, "Only enterprise firmware 3 permits seven-day retention."),
        (
            "sources/long-reference.md",
            6500,
            "Only laboratory edition 4 permits permanent archival.",
        ),
    ]:
        body = "# Device\n## Retention\n" + "Background.\n" * repetitions + condition + "\n"
        parts = brain_sync.markdown_passages(body, title_context="Device")
        points = brain_sync.build_points(
            path=path,
            title="Device",
            frontmatter={
                "source_origin": "upstream",
                "status": "reference",
                "updated": "2026-10-01",
            },
            commit="a" * 40,
            chunks=[(p.heading, p.content) for p in parts],
            embeddings={"dense": [[1.0] * 1024] * len(parts)},
            content_hash=hashlib.sha256(body.encode()).hexdigest(),
            passage_details=parts,
        )
        for point in points:
            hits.append(
                SearchResult(
                    id=deterministic_point_id(
                        point.source_id, point.chunk_index, point.stage, point.version_hash
                    ),
                    score=0.9,
                    collection=BRAIN,
                    payload=cast(dict[str, Any], point.payload),
                )
            )
    return hits


class RecorderRetriever:
    def __init__(self, path: Path | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.path = path

    async def cross_collection_search(self, **kwargs: Any) -> list[SearchResult]:
        self.calls.append(kwargs)
        if self.path:
            self.path.write_text(json.dumps(self.calls, default=str))
        return source_hits()[: kwargs["top_k"]]


def make_services(path: Path | None = None) -> SurfaceServices:
    settings = SimpleNamespace(
        api=SimpleNamespace(api_key=API_KEY, api_keys_json="", cors_origins=["*"]),
        portal=SimpleNamespace(csrf_secret="fixture-csrf"),
        answer=SimpleNamespace(allow_hosted=False, allow_raw_hosted_egress=False),
        reranker=SimpleNamespace(auto_merge_enabled=False, auto_merge_threshold=0.5),
    )
    return SurfaceServices(
        settings=settings, retriever=RecorderRetriever(path), db_pool=None, reranker=None
    )

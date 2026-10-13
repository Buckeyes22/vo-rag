"""Complete source units, never partial matching prose, may be redundant."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, cast

import pytest

from vo_rag.core.retrieval.brain_redundancy import select_ranked_evidence
from vo_rag.core.retrieval.execution import RerankFn, execute_retrieval
from vo_rag.core.retrieval.search import HybridRetriever, SearchResult

ROOT = "# Device\n\nPerformance modes affect latency and throughput."
QUALIFIER = "## Enterprise\n\nApplies only to Enterprise firmware 3."


def _segment(text: str, occurrence: int, point: str) -> dict[str, Any]:
    return {
        "text": text,
        "source_id": "wiki/device.md",
        "collection": "vo_rag_brain_v1",
        "source_revision": "a" * 40,
        "section_occurrence": occurrence,
        "point_id": point,
        "content_hash": hashlib.sha256(text.encode()).hexdigest(),
    }


def _hit(
    identifier: str, local: str, ancestors: list[dict[str, Any]], heading: str
) -> SearchResult:
    title = "Device" + (" / " + heading if heading != "(page top)" else "")
    return SearchResult(
        identifier,
        1.0,
        "vo_rag_brain_v1",
        {
            "source_id": "wiki/device.md",
            "path": "wiki/device.md",
            "commit": "a" * 40,
            "title": "Device",
            "heading": heading,
            "source_section_text": local,
            "content": title + "\n\n" + "\n\n".join([*(a["text"] for a in ancestors), local]),
            "governing_context_version": 1,
            "governing_context_complete": True,
            "governing_context": ancestors,
            "status": "accepted",
            "edition": "Enterprise",
            "source_origin": "curated",
            "chunk_id": identifier,
            "chunk_index": 0,
        },
    )


def test_root_unit_already_in_selected_descendant_frees_slot_without_losing_source() -> None:
    child = _hit(
        "cache",
        "## Cache\n\nRepeated prefixes reuse processing.",
        [_segment(ROOT, 1, "cache")],
        "Cache",
    )
    root = _hit("root", ROOT, [], "(page top)")
    fast = _hit("fast", "## Fast\n\nUp to 2.5x faster.", [_segment(ROOT, 1, "fast")], "Fast")
    before = deepcopy([child, root, fast])
    selected = select_ranked_evidence([child, root, fast], 2)
    assert [hit.id for hit in selected] == ["cache", "fast"]
    assert selected[0].payload == before[0].payload
    assert ROOT in selected[0].payload["content"]
    assert [child, root, fast] == before


@pytest.mark.parametrize(
    "change",
    [
        "edition",
        "revision",
        "authority",
        "unknown_policy",
        "foreign_collection",
        "missing_manifest",
        "bad_hash",
        "absent_text",
        "extra_fact",
        "partial_unit",
        "local_occurrence",
        "bool_occurrence",
        "contradictory_alias",
    ],
)
def test_uncertain_or_different_scope_cannot_suppress_complete_original(change: str) -> None:
    child = _hit("child", "## Speed\n\nFast processing.", [_segment(ROOT, 1, "child")], "Speed")
    root = _hit("root", ROOT, [], "(page top)")
    if change == "edition":
        child.payload["edition"] = "Community"
    elif change == "revision":
        child.payload["commit"] = "b" * 40
        child.payload["governing_context"][0]["source_revision"] = "b" * 40
    elif change == "authority":
        child.payload["status"] = "proposed"
    elif change == "unknown_policy":
        child.payload["future_scope"] = "restricted"
    elif change == "foreign_collection":
        child.collection = "vo_rag_books_raw_v1"
    elif change == "missing_manifest":
        child.payload["governing_context_complete"] = False
    elif change == "bad_hash":
        child.payload["governing_context"][0]["content_hash"] = "bad"
    elif change == "absent_text":
        child.payload["content"] = "No original unit here."
    elif change == "extra_fact":
        root.payload["content"] += "\n\nSeparate retention fact."
    elif change == "partial_unit":
        root.payload["source_section_text"] = ROOT.split("\n\n")[-1]
    elif change == "local_occurrence":
        root.payload["source_section_occurrence"] = 9
    elif change == "bool_occurrence":
        root.payload["source_section_occurrence"] = True
    elif change == "contradictory_alias":
        root.payload["source_section_occurrence"] = 1
        root.payload["section_occurrence"] = 9
    assert [hit.id for hit in select_ranked_evidence([child, root], 2)] == ["child", "root"]


def test_repeated_same_text_cannot_drop_its_distinct_original_governors() -> None:
    child = _hit("child", "## Limits\n\n7 days.", [_segment(ROOT, 1, "child")], "Limits")
    scoped = _hit("scoped", ROOT, [_segment(QUALIFIER, 3, "scoped")], "Limits")
    assert [hit.id for hit in select_ranked_evidence([child, scoped], 2)] == ["child", "scoped"]
    # Exact governor text with a contradictory declared occurrence is still distinct.
    child.payload["governing_context"].insert(0, _segment(QUALIFIER, 8, "child"))
    child.payload["content"] = "Device / Limits\n\n" + QUALIFIER + "\n\n" + ROOT + "\n\n7 days."
    assert len(select_ranked_evidence([child, scoped], 2)) == 2
    child.payload["governing_context"][0]["section_occurrence"] = 3
    assert len(select_ranked_evidence([child, scoped], 2)) == 1


def test_only_already_selected_descendant_can_supply_original_evidence() -> None:
    root = _hit("root", ROOT, [], "(page top)")
    child = _hit("child", "## Fast\n\nFast processing.", [_segment(ROOT, 1, "child")], "Fast")
    assert select_ranked_evidence([root, child], 1) == [root]
    assert select_ranked_evidence([child, root], 0) == []


async def test_owning_retrieval_retains_normal_depth_and_topk_with_exact_payloads() -> None:
    child = _hit("cache", "## Cache\n\nRepeated prefixes.", [_segment(ROOT, 1, "cache")], "Cache")
    root = _hit("root", ROOT, [], "(page top)")
    fast = _hit("fast", "## Fast\n\nUp to 2.5x faster.", [_segment(ROOT, 1, "fast")], "Fast")
    original = deepcopy([child, root, fast])

    class Retriever:
        async def cross_collection_search(self, **kwargs: Any) -> list[SearchResult]:
            assert kwargs["top_k"] == 100
            assert kwargs["collections"] == ["vo_rag_brain_v1"]
            return [child, root, fast]

    def rank(query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        assert len(candidates) == 3
        assert top_k == 40
        return candidates

    results = await execute_retrieval(
        retriever=cast(HybridRetriever, Retriever()),
        query="Latency and throughput?",
        requested_collections=["vo_rag_brain_v1"],
        top_k=2,
        rerank_fn=cast(RerankFn, rank),
    )
    assert [hit.id for hit in results] == ["cache", "fast"]
    assert [hit.payload for hit in results] == [original[0].payload, original[2].payload]


def test_identical_unit_under_additional_condition_is_not_unqualified_root_evidence() -> None:
    root = _hit("root", ROOT, [], "(page top)")
    child = _hit(
        "child",
        "## Speed\n\nFast.",
        [_segment(QUALIFIER, 0, "child"), _segment(ROOT, 1, "child")],
        "Speed",
    )
    assert [hit.id for hit in select_ranked_evidence([child, root], 2)] == ["child", "root"]

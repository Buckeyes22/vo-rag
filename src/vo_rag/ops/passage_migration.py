"""Audit or restartably backfill passage contracts through the owning storage service."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from qdrant_client.models import (
    DeleteVectors,
    DeleteVectorsOperation,
    Modifier,
    PointVectors,
    SetPayload,
    SetPayloadOperation,
    SparseVectorParams,
    UpdateVectors,
    UpdateVectorsOperation,
)

from vo_rag.core.config.settings import get_settings
from vo_rag.core.retrieval.bm25_tokenizer import tokenize, tokens_to_bm25_sparse
from vo_rag.core.storage.qdrant import COLLECTION_NAMES, QdrantStore, bge_sparse_to_qdrant
from vo_rag.ops.brain_sync import BRAIN_COLLECTION, WRITER_VERSION, load_state, save_state

DERIVED_BM25_COLLECTIONS = {
    "vo_rag_books_summary_tree_v1",
    "vo_rag_books_raptor_v1",
    "vo_rag_books_graph_nodes_v1",
}


def _prepare_target_state(state_path: Path, collections: list[str]) -> dict[str, Any]:
    """Bind acknowledged work to an explicit immutable target/contract plan."""
    targets = sorted(set(collections))
    if not targets or any(not isinstance(name, str) or not name.strip() for name in targets):
        raise ValueError("apply requires explicit nonempty collection targets")
    state = load_state(state_path)
    plan = {"version": 1, "writer_contract_version": 1, "collections": targets}
    if BRAIN_COLLECTION in targets:
        plan["brain_writer_version"] = WRITER_VERSION
    previous = state.get("target_plan")
    if previous is not None and previous != plan:
        raise ValueError(
            "migration checkpoint target/contract differs; use its exact plan or a new state file"
        )
    if previous is None:
        legacy = set(state.get("collections", {}))
        if legacy and legacy != set(targets) - {BRAIN_COLLECTION}:
            raise ValueError(
                "legacy migration checkpoint target differs; use its exact plan or a new state file"
            )
    state["target_plan"] = plan
    save_state(state_path, state)
    return state


def repair_payload(point_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Return only missing or newly declared contract fields, preserving lineage."""
    patch: dict[str, Any] = {
        "writer_contract_version": 1,
        "colbert_representation": payload.get("colbert_representation")
        or "legacy_unused_for_scoring",
    }
    if not payload.get("chunk_type"):
        patch["chunk_type"] = "text"
    if not payload.get("chunk_id"):
        patch["chunk_id"] = point_id
    if not payload.get("source_id"):
        patch["source_id"] = payload.get("book_id") or point_id
    if not payload.get("content") and (payload.get("summary_text") or payload.get("label")):
        patch["content"] = payload.get("summary_text") or payload.get("label")
    return patch


async def migrate(
    store: QdrantStore, collections: list[str], state_path: Path, *, apply: bool = False
) -> dict[str, Any]:
    """Backfill batches idempotently; checkpoint only after acknowledged writes.

    No vectors are re-embedded. Derived-node BM25 is rebuilt from source text;
    book channels and raw/sterile parent identities remain unchanged.
    """
    state = _prepare_target_state(state_path, collections) if apply else {}
    progress = state.setdefault("collections", {})
    for collection in collections:
        if collection == BRAIN_COLLECTION:
            continue  # brain_sync writer-version invalidation owns re-chunking
        item = progress.setdefault(collection, {"offset": None, "scanned": 0, "changed": 0})
        if item.get("complete"):
            continue
        offset = item.get("offset")
        while True:
            points, next_offset = await store._client.scroll(
                collection_name=collection,
                offset=offset,
                limit=256,
                with_payload=True,
                with_vectors=False,
            )
            operations: list[
                SetPayloadOperation | UpdateVectorsOperation | DeleteVectorsOperation
            ] = []
            batch_changed = 0
            patch_groups: dict[str, tuple[dict[str, Any], list[Any]]] = {}
            for point in points:
                payload = point.payload or {}
                text = payload.get("summary_text") or payload.get("label")
                empty_lexical_stamp = (
                    collection in DERIVED_BM25_COLLECTIONS
                    and isinstance(text, str)
                    and not tokenize(text)
                    and payload.get("bm25_representation") != "absent_no_tokens"
                )
                if payload.get("writer_contract_version") == 1 and not empty_lexical_stamp:
                    continue
                patch = repair_payload(str(point.id), payload)
                if collection in DERIVED_BM25_COLLECTIONS:
                    text = payload.get("summary_text") or payload.get("label")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f"{collection}/{point.id}: missing derived source text")
                    lexical = tokens_to_bm25_sparse(tokenize(text))
                    if lexical:
                        operations.append(
                            UpdateVectorsOperation(
                                update_vectors=UpdateVectors(
                                    points=[
                                        PointVectors(
                                            id=point.id,
                                            vector={"bm25": bge_sparse_to_qdrant(lexical)},
                                        )
                                    ]
                                )
                            )
                        )
                        patch["bm25_representation"] = "hashed_tokens_v1"
                    else:
                        # Qdrant rejects an update whose only named vector is
                        # empty. Remove the invalid legacy channel; inventing
                        # a sentinel token would introduce false lexical votes.
                        operations.append(
                            DeleteVectorsOperation(
                                delete_vectors=DeleteVectors(points=[point.id], vector=["bm25"])
                            )
                        )
                        patch["bm25_representation"] = "absent_no_tokens"
                key = json.dumps(patch, sort_keys=True, ensure_ascii=False)
                if key not in patch_groups:
                    patch_groups[key] = (patch, [])
                patch_groups[key][1].append(point.id)
                batch_changed += 1
            # Every vector replacement/deletion precedes the contract stamp.
            # Group identical additive patches within this bounded batch only.
            operations.extend(
                SetPayloadOperation(set_payload=SetPayload(points=ids, payload=patch))
                for patch, ids in patch_groups.values()
            )
            pending = item.get("pending", {})
            if apply and pending.get("offset") == offset:
                batch_changed = max(batch_changed, pending.get("changed", 0))
            if apply and operations:
                # Persist intent without advancing the acknowledged cursor or
                # counters. A partial/lost reply may already stamp some rows;
                # replay still counts each originally planned repair once.
                item["pending"] = {"offset": offset, "changed": batch_changed}
                save_state(state_path, state)
                await store._client.batch_update_points(
                    collection_name=collection, update_operations=operations, wait=True
                )
            item.pop("pending", None)
            item["changed"] += batch_changed
            item["scanned"] += len(points)
            item["offset"] = next_offset
            item["complete"] = next_offset is None
            if apply:
                save_state(state_path, state)
            if next_offset is None:
                break
            offset = next_offset
    return state


async def reconcile_sparse_modifiers(
    store: QdrantStore, collections: list[str], *, apply: bool = False
) -> dict[str, Any]:
    """Audit and optionally align existing named sparse scoring with IDF.

    This owning maintenance operation updates collection configuration only;
    no points, indexes, lineage or learned vector values are replaced.
    """
    report: dict[str, Any] = {}
    for collection in collections:
        info = await store._client.get_collection(collection)
        configured = info.config.params.sparse_vectors or {}
        before = {
            name: params.modifier
            for name, params in configured.items()
            if name in {"sparse", "bm25"}
        }
        missing = [name for name, value in before.items() if value != Modifier.IDF]
        if apply and missing:
            acknowledged = await store._client.update_collection(
                collection_name=collection,
                sparse_vectors_config={
                    name: SparseVectorParams(modifier=Modifier.IDF) for name in missing
                },
            )
            if not acknowledged:
                raise RuntimeError(f"{collection}: sparse modifier update was not acknowledged")
        report[collection] = {
            "before": before,
            "requires_idf": missing,
            "applied": bool(apply and missing),
        }
    return report


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    requested = args.collection or []
    all_collections = bool(getattr(args, "all_collections", False))
    if requested and all_collections:
        raise ValueError("--collection and --all are mutually exclusive")
    if any(not isinstance(name, str) or not name.strip() for name in requested):
        raise ValueError("collection targets must be nonempty names")
    if args.apply and not requested and not all_collections:
        raise ValueError("--apply requires --collection or explicit --all")
    settings = get_settings()
    store = QdrantStore(
        settings.qdrant.url, api_key=settings.qdrant.api_key or None, prefer_grpc=False
    )
    try:
        existing = {item.name for item in (await store._client.get_collections()).collections}
        collections = (
            sorted(set(requested)) if requested else sorted(set(COLLECTION_NAMES) & existing)
        )
        unknown = set(collections) - existing
        if unknown:
            raise ValueError(f"collections do not exist: {sorted(unknown)}")
        if args.apply:
            _prepare_target_state(args.state, collections)
        modifiers = await reconcile_sparse_modifiers(store, collections, apply=args.apply)
        result = await migrate(store, collections, args.state, apply=args.apply)
        result["sparse_modifiers"] = modifiers
        result["selection"] = {
            "collections": collections,
            "explicit_all": all_collections,
            "apply": args.apply,
        }
        if BRAIN_COLLECTION in collections:
            result["brain_delegation"] = {
                "owner": "brain_sync",
                "required_writer_version": WRITER_VERSION,
                "performed": False,
            }

        result["inventory"] = {
            "existing_declared": sorted(set(COLLECTION_NAMES) & existing),
            "undeclared": sorted(existing - set(COLLECTION_NAMES)),
            "brain_owner": "brain_sync writer-version migration",
        }
        return result
    finally:
        await store._client.close()


def main() -> None:
    """Run a read-only audit by default, or an explicitly requested backfill."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--state", type=Path, default=Path.home() / ".local/state/vo-rag/passage-migration-v1.json"
    )
    parser.add_argument("--collection", action="append")
    parser.add_argument(
        "--all",
        dest="all_collections",
        action="store_true",
        help="explicitly select every existing declared collection",
    )
    args = parser.parse_args()
    print(json.dumps(asyncio.run(_run(args)), sort_keys=True))


if __name__ == "__main__":
    main()

__all__ = [
    "DERIVED_BM25_COLLECTIONS",
    "main",
    "migrate",
    "reconcile_sparse_modifiers",
    "repair_payload",
]

"""Read-only catalog routes for the portal: envelopes, artifacts, stats, figures."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from vo_rag.api.dependencies import get_services
from vo_rag.api.singletons import SurfaceServices

router = APIRouter()


def _require_pool(services: SurfaceServices) -> Any:
    if services.db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="db_pool_unavailable",
        )
    return services.db_pool


@router.get("/api/v1/envelopes")
async def envelopes_list(
    services: SurfaceServices = Depends(get_services),  # noqa: B008
    limit: int = Query(default=50, ge=1, le=200),
    status_filter: str | None = Query(default=None, alias="status"),
) -> dict[str, Any]:
    from vo_rag.core.db.queries.envelopes import list_envelopes

    pool = _require_pool(services)
    envelopes, next_cursor = await list_envelopes(pool, limit=limit, status=status_filter)
    return {"items": [e.model_dump(mode="json") for e in envelopes], "next_cursor": next_cursor}


@router.get("/api/v1/envelopes/{envelope_id}")
async def envelope_detail(
    envelope_id: str,
    services: SurfaceServices = Depends(get_services),  # noqa: B008
) -> dict[str, Any]:
    from vo_rag.core.db.queries.envelopes import get_envelope

    pool = _require_pool(services)
    envelope = await get_envelope(pool, envelope_id)
    if envelope is None:
        raise HTTPException(status_code=404, detail="envelope_not_found")
    return envelope.model_dump(mode="json")


@router.get("/api/v1/artifacts")
async def artifacts_list(
    services: SurfaceServices = Depends(get_services),  # noqa: B008
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = None,
    pipeline: str | None = None,
    tags: Annotated[list[str] | None, Query()] = None,
    license_class: str | None = None,
) -> dict[str, Any]:
    from vo_rag.core.db.queries.artifacts import list_artifacts

    pool = _require_pool(services)
    artifacts, next_cursor = await list_artifacts(
        pool,
        cursor=cursor,
        limit=limit,
        pipeline=pipeline,
        tags=tags,
        license_class=license_class,
    )
    return {"items": [a.model_dump(mode="json") for a in artifacts], "next_cursor": next_cursor}


@router.get("/api/v1/artifacts/{artifact_id}")
async def artifact_detail(
    artifact_id: str,
    services: SurfaceServices = Depends(get_services),  # noqa: B008
) -> dict[str, Any]:
    from vo_rag.core.db.queries.artifacts import get_artifact

    pool = _require_pool(services)
    artifact = await get_artifact(pool, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="artifact_not_found")
    return artifact.model_dump(mode="json")


@router.get("/api/v1/artifacts/{artifact_id}/images")
async def artifact_images(
    artifact_id: str,
    services: SurfaceServices = Depends(get_services),  # noqa: B008
) -> dict[str, Any]:
    """Figure images for one artifact: image_ref payloads -> /api/v1/figures URLs."""
    from vo_rag.core.db.queries.artifacts import get_artifact

    pool = _require_pool(services)
    artifact = await get_artifact(pool, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="artifact_not_found")
    if services.qdrant is None or not artifact.qdrant_point_ids or not artifact.qdrant_collection:
        return {"items": []}

    records = await services.qdrant._client.retrieve(  # noqa: SLF001
        collection_name=artifact.qdrant_collection,
        ids=list(artifact.qdrant_point_ids),
        with_payload=True,
    )
    items = []
    for record in records:
        payload = getattr(record, "payload", None) or {}
        ref = payload.get("image_ref")
        if not ref:
            continue
        items.append(
            {
                "id": str(payload.get("image_id") or record.id),
                "url": f"/api/v1/figures/{ref}",
                "caption": (payload.get("vlm_description") or "")[:280] or None,
            }
        )
    return {"items": items}


@router.get("/api/v1/collections/stats")
async def collections_stats(
    services: SurfaceServices = Depends(get_services),  # noqa: B008
) -> dict[str, Any]:
    if services.qdrant is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="qdrant_unavailable",
        )
    client = services.qdrant._client  # noqa: SLF001
    listed = await client.get_collections()
    out: dict[str, dict[str, int]] = {}
    for col in listed.collections:
        count = await client.count(collection_name=col.name, exact=True)
        out[col.name] = {"points": int(count.count)}
    return {"collections": out}

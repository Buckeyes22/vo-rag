"""GET /api/v1/figures/{src}/{filename} — stream figure bytes from NVMe.

Query hits carry ``image_ref`` (``{src}/{filename}``) in their payload;
clients turn it into ``/api/v1/figures/{image_ref}``. Keys are
content-addressed by source hash, so responses are immutable-cacheable.
Auth: ApiKeyMiddleware is app-wide and this path is not in EXEMPT_PATHS,
so the existing API-key posture applies unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from vo_rag.api.dependencies import get_services
from vo_rag.api.singletons import SurfaceServices
from vo_rag.core.storage.figures import MIME_BY_EXT, FigureStore

router = APIRouter()
log = logging.getLogger("vo_rag.api.routes.figures")

_IMMUTABLE_CACHE = "public, max-age=31536000, immutable"


@router.get("/api/v1/figures/{src}/{filename}")
async def get_figure(
    src: str,
    filename: str,
    services: SurfaceServices = Depends(get_services),  # noqa: B008
) -> FileResponse:
    store = FigureStore(Path(str(services.settings.figures.root)))
    try:
        # path_for never touches the filesystem; traversal/malformed keys
        # are rejected before any disk access.
        path = store.path_for(f"{src}/{filename}")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_figure_ref"
        ) from None
    if not path.is_file():
        # Key in Qdrant but file absent: client degrades to description-only;
        # recovery is re-ingest (regen) or restore from backup.
        log.warning("figure_missing src=%s filename=%s", src, filename)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="figure_not_found")
    ext = filename.rsplit(".", 1)[-1].lower()
    return FileResponse(
        path,
        media_type=MIME_BY_EXT[ext],
        headers={
            "Cache-Control": _IMMUTABLE_CACHE,
            # Belt-and-braces with the store's svg/MIME refusal: never let
            # a browser content-sniff its way past the declared image type.
            "X-Content-Type-Options": "nosniff",
        },
    )

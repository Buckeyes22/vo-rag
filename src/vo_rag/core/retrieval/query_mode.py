"""QueryMode + collection resolver leaf.

The collection name strings ARE the persistence contract — every Qdrant
read and write targets one of these. Silently renaming any of them means
orphaning every Qdrant point.
"""

from __future__ import annotations

from enum import StrEnum


class QueryMode(StrEnum):
    INTERNAL = "internal"
    PUBLICATION = "publication"
    PARTNER = "partner"


RAW_COLLECTIONS = {"vo_rag_books_raw_v1"}
STERILE_COLLECTIONS = {"vo_rag_books_sterile_v1"}
RAW_TO_STERILE = {"vo_rag_books_raw_v1": "vo_rag_books_sterile_v1"}
MODE_AGNOSTIC_COLLECTIONS = (
    "vo_rag_articles_v1",
    "vo_rag_repos_v1",
    "vo_rag_research_v1",
    "vo_rag_ideas_v1",
    "vo_rag_tools_v1",
    "vo_rag_urls_v1",
    "vo_rag_structured_v1",
    "vo_rag_code_chunks_v1",
    "vo_rag_reddit_v1",
)


def resolve_collections(mode: QueryMode, requested: list[str] | None) -> list[str]:
    """Return the actual collections to search for a query mode."""
    if requested is None:
        default_books = ["vo_rag_books_raw_v1"]
        if mode in {QueryMode.PUBLICATION, QueryMode.PARTNER}:
            default_books = ["vo_rag_books_sterile_v1"]
        requested = [*default_books, *MODE_AGNOSTIC_COLLECTIONS]

    if not requested or any(not isinstance(name, str) or not name.strip() for name in requested):
        raise ValueError("explicit collection selection must contain nonempty names")

    resolved: list[str] = []
    for collection in requested:
        if mode in {QueryMode.PUBLICATION, QueryMode.PARTNER}:
            collection = RAW_TO_STERILE.get(collection, collection)

        if collection not in resolved:
            resolved.append(collection)

    return resolved


__all__ = [
    "QueryMode",
    "RAW_COLLECTIONS",
    "STERILE_COLLECTIONS",
    "RAW_TO_STERILE",
    "MODE_AGNOSTIC_COLLECTIONS",
    "resolve_collections",
]

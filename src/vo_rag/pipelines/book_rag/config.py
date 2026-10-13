"""Pipeline-level configuration for the book_rag leaf.

Pydantic v2 strict model. ``sterilize_enabled`` is a first-class branch field
(default ``False`` — sterilization is opt-in only, "off by default
everywhere": it runs a per-chunk LLM summary pass and writes the commercial
``*_sterile_v1`` collection, so it must never run on a default ingest unless
explicitly requested).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from vo_rag.pipelines.book_rag.domain_vocab import DOMAIN_VOCABULARY


class BookRagConfig(BaseModel):
    model_config = ConfigDict(strict=True)

    sterilize_enabled: bool = False  # opt-in only; see module docstring
    vlm_concurrency: int = 1
    min_vlm_describe_yield: float = 0.90
    min_visual_sterile_yield: float = 0.70
    domain_vocab: list[str] = DOMAIN_VOCABULARY

    # Visual sidecar (VOR-53).  Extraction runs first; description is an
    # optional VLM layer on top.
    #
    # extraction is DEFAULT-ON (the standard setting): the no-VLM composite
    # path is CPU-only and turns the image bytes the parser already uploads to
    # Garage into ``figure_composite`` chunks. Leaving it off meant every
    # ingest uploaded images but referenced none of them (0 image chunks) —
    # the fragmentation that forced ticket-named image collections. The one
    # pipeline now always produces image chunks. ``enable_visual_description``
    # (the VLM caption layer) stays opt-in — it needs a GPU VLM endpoint.
    enable_visual_extraction: bool = True
    enable_visual_description: bool = False

    # Advanced-understanding layers (Layer 1-3). All opt-in: the DAG's
    # advanced_understanding step is a no-op when every flag is False, so the
    # default ingest path is parse -> chunk/embed -> sterilize -> validate ->
    # smoke. Enabling a flag invokes its core builder (summary/raptor/dsi/graph).
    enable_summary_tree: bool = False
    enable_raptor: bool = False
    enable_dsi: bool = False
    enable_graph: bool = False

    # Terminal self-query retrieval-smoke gate. Default-on (prod behaviour
    # unchanged). It is the only LLM-dependent ingest step once descriptions and
    # advanced layers are off, so a defer-LLM ingest (parse/extract-images/
    # chunk/embed only) turns it off — and it is also auto-skipped whenever the
    # workflow has no LLM client (serverless unconfigured).
    retrieval_smoke_enabled: bool = True

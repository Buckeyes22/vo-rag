"""Query metadata schema — load-bearing schema for the 300-query benchmark."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from vo_rag.core.models.common import VoBaseModel


class QueryClass(StrEnum):
    LOCAL_FACTUAL_LOOKUP = "local_factual_lookup"
    FIGURE_DEPENDENT = "figure_dependent"
    INTRA_SECTION_REFERENTIAL_AMBIGUITY = "intra_section_referential_ambiguity"
    CHAPTER_SPANNING_SYNTHESIS = "chapter_spanning_synthesis"
    WHOLE_BOOK_SUMMARY = "whole_book_summary"
    CROSS_BOOK_THEMATIC = "cross_book_thematic"
    ENTITY_RELATIONSHIP = "entity_relationship"


class CitationGranularity(StrEnum):
    CHUNK = "chunk"
    SECTION = "section"
    CHAPTER = "chapter"


class PilotQuery(VoBaseModel):
    query_id: str
    query_text: str
    query_class_primary: QueryClass
    query_class_secondary: QueryClass | None = None
    requires_cross_section_concept_aggregation: bool
    requires_book_routing: bool
    requires_figure_evidence: bool
    requires_entity_relationship_reasoning: bool
    requires_referential_disambiguation: bool
    requires_summary_coverage: bool
    expected_answer: str
    expected_supporting_chunk_ids: list[str] = Field(default_factory=list)
    expected_supporting_section_paths: list[str] = Field(default_factory=list)
    required_citation_granularity: CitationGranularity

    @model_validator(mode="after")
    def query_id_not_empty(self) -> PilotQuery:
        if not self.query_id:
            raise ValueError("query_id must not be empty")
        return self

    @model_validator(mode="after")
    def query_text_not_empty(self) -> PilotQuery:
        if not self.query_text.strip():
            raise ValueError("query_text must not be empty after stripping whitespace")
        return self

    @model_validator(mode="after")
    def class_implies_required_flags(self) -> PilotQuery:
        if (
            self.query_class_primary == QueryClass.ENTITY_RELATIONSHIP
            and not self.requires_entity_relationship_reasoning
        ):
            raise ValueError(
                "query_class_primary ENTITY_RELATIONSHIP requires "
                "requires_entity_relationship_reasoning to be True"
            )
        if (
            self.query_class_primary == QueryClass.FIGURE_DEPENDENT
            and not self.requires_figure_evidence
        ):
            raise ValueError(
                "query_class_primary FIGURE_DEPENDENT requires requires_figure_evidence to be True"
            )
        if (
            self.query_class_primary == QueryClass.INTRA_SECTION_REFERENTIAL_AMBIGUITY
            and not self.requires_referential_disambiguation
        ):
            raise ValueError(
                "query_class_primary INTRA_SECTION_REFERENTIAL_AMBIGUITY requires "
                "requires_referential_disambiguation to be True"
            )
        if (
            self.query_class_primary == QueryClass.WHOLE_BOOK_SUMMARY
            and not self.requires_summary_coverage
        ):
            raise ValueError(
                "query_class_primary WHOLE_BOOK_SUMMARY requires "
                "requires_summary_coverage to be True"
            )
        return self

    @model_validator(mode="after")
    def cross_section_flag_for_inherently_aggregating_classes(self) -> PilotQuery:
        if (
            self.query_class_primary
            in (QueryClass.CHAPTER_SPANNING_SYNTHESIS, QueryClass.CROSS_BOOK_THEMATIC)
            and not self.requires_cross_section_concept_aggregation
        ):
            raise ValueError(
                f"query_class_primary {self.query_class_primary.value} inherently "
                "requires cross-section concept aggregation; "
                "requires_cross_section_concept_aggregation must be True. "
                "Downstream gate decisions depend on this flag being set correctly."
            )
        return self


__all__ = [
    "CitationGranularity",
    "PilotQuery",
    "QueryClass",
]

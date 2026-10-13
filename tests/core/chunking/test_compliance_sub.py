from __future__ import annotations

from vo_rag.core.chunking.compliance_sub import ComplianceSubChunker
from vo_rag.core.chunking.hierarchical import RetrievalChunk


def _parent(content: str) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id="parent-1",
        chunk_type="text",
        document_id="d",
        source_id="s",
        section_path="[Book > Ch1]",
        section_title="Ch1",
        content=content,
        embedding_text=content,
        token_count=len(content) // 4,
        page_number=1,
    )


def test_split_produces_bounded_sub_chunks() -> None:
    text = ("This is sentence one. " * 30) + ("Sentence two here. " * 20)
    subs = ComplianceSubChunker(max_tokens=50).split(_parent(text))
    for sub in subs:
        assert sub.token_count <= 60, f"sub {sub.position_in_parent} too big"


def test_positions_sequential_and_zero_indexed() -> None:
    text = "one. two. three. four. five."
    subs = ComplianceSubChunker(max_tokens=20).split(_parent(text))
    positions = [sub.position_in_parent for sub in subs]
    assert positions == list(range(len(subs)))


def test_subs_reconstruct_parent_whitespace_normalized() -> None:
    text = "alpha. bravo. charlie. delta."
    subs = ComplianceSubChunker(max_tokens=20).split(_parent(text))
    rebuilt = " ".join(sub.content for sub in subs)

    def norm(value: str) -> str:
        return " ".join(value.split())

    assert norm(rebuilt) == norm(text)


def test_deterministic_ids() -> None:
    text = "foo. bar. baz."
    first = ComplianceSubChunker(max_tokens=20).split(_parent(text))
    second = ComplianceSubChunker(max_tokens=20).split(_parent(text))
    assert [sub.sub_chunk_id for sub in first] == [sub.sub_chunk_id for sub in second]


def test_oversize_sentence_is_sliced() -> None:
    giant = "x" * 5000
    subs = ComplianceSubChunker(max_tokens=50).split(_parent(giant))
    assert len(subs) >= 2
    assert all(sub.content for sub in subs)

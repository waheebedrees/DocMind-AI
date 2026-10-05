"""Unit tests for passage assembly and context rendering."""

from __future__ import annotations

from uuid import UUID, uuid4

from app.rag.retrieval.context import build_passages, render_context
from app.rag.retrieval.types import Candidate, Passage

DOC_A = UUID("11111111-2222-3333-4444-555555555555")
DOC_B = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


def make_candidate(
    *,
    document_id: UUID = DOC_A,
    chunk_index: int = 0,
    text: str = "text",
    page_number: int | None = None,
    section: str | None = None,
    token_count: int | None = None,
    rerank_score: float | None = None,
    fused_score: float = 0.0,
) -> Candidate:
    return Candidate(
        chunk_id=uuid4(),
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        page_number=page_number,
        section=section,
        token_count=token_count,
        fused_score=fused_score,
        rerank_score=rerank_score,
    )


# --- build_passages --------------------------------------------------


def test_single_selected_no_neighbors_yields_one_passage():
    sel = [make_candidate(chunk_index=0, text="hello", token_count=10)]
    out = build_passages(sel, [], window=1, max_tokens=1000)
    assert len(out) == 1
    assert out[0].citation_id == 1
    assert out[0].chunk_ids == (sel[0].chunk_id,)
    assert out[0].chunk_range == (0, 0)
    assert out[0].text == "hello"


def test_neighbors_merge_into_contiguous_run():
    s = make_candidate(chunk_index=2, text="middle", rerank_score=0.9)
    n1 = make_candidate(chunk_index=1, text="before")
    n2 = make_candidate(chunk_index=3, text="after")
    out = build_passages([s], [n1, n2], window=1, max_tokens=1000)
    assert len(out) == 1
    assert out[0].chunk_range == (1, 3)
    assert out[0].text == "before\nmiddle\nafter"
    assert set(out[0].chunk_ids) == {s.chunk_id, n1.chunk_id, n2.chunk_id}


def test_gap_splits_into_separate_passages():
    """Two selected chunks with a gap produce two separate passages."""
    a = make_candidate(chunk_index=0, text="a", rerank_score=0.9)
    b = make_candidate(chunk_index=10, text="b", rerank_score=0.5)
    out = build_passages([a, b], [], window=1, max_tokens=1000)
    assert len(out) == 2
    assert out[0].chunk_range == (0, 0)
    assert out[1].chunk_range == (10, 10)


def test_neighbor_outside_window_is_not_included():
    """A neighbor at ±2 with window=1 is not pulled into the passage."""
    s = make_candidate(chunk_index=0, text="a", rerank_score=0.9)
    n = make_candidate(chunk_index=2, text="b")  # outside 0±1
    out = build_passages([s], [n], window=1, max_tokens=1000)
    assert len(out) == 1
    assert out[0].chunk_range == (0, 0)


def test_two_documents_produce_separate_passages():
    s1 = make_candidate(document_id=DOC_A, chunk_index=0, text="a")
    s2 = make_candidate(document_id=DOC_B, chunk_index=0, text="b")
    out = build_passages([s1, s2], [], window=0, max_tokens=1000)
    assert len(out) == 2
    assert {p.document_id for p in out} == {DOC_A, DOC_B}


def test_sorted_by_best_score_in_run():
    low = make_candidate(chunk_index=0, rerank_score=0.1)
    high = make_candidate(chunk_index=10, rerank_score=0.9)
    out = build_passages([low, high], [], window=0, max_tokens=1000)
    assert out[0].score == 0.9
    assert out[1].score == 0.1


def test_token_budget_skips_later_passages():
    a = make_candidate(chunk_index=0, token_count=600, rerank_score=0.9)
    b = make_candidate(chunk_index=10, token_count=600, rerank_score=0.8)
    out = build_passages([a, b], [], window=0, max_tokens=700)
    assert len(out) == 1
    assert out[0].chunk_ids == (a.chunk_id,)


def test_token_budget_smaller_later_passage_still_fits():
    a = make_candidate(chunk_index=0, token_count=900, rerank_score=0.9)
    b = make_candidate(chunk_index=10, token_count=100, rerank_score=0.5)
    out = build_passages([a, b], [], window=0, max_tokens=1000)
    # a fits (900 ≤ 1000), b fits (100 + 900 = 1000)
    assert {p.document_id for p in out} == {DOC_A}
    assert len(out) == 2


def test_first_passage_truncated_to_one_chunk_when_over_budget():
    """If even the top run exceeds max_tokens, keep its best chunk."""
    a = make_candidate(chunk_index=0, token_count=1000, text="a", rerank_score=0.9)
    b = make_candidate(chunk_index=1, token_count=1000, text="b")
    out = build_passages([a], [b], window=1, max_tokens=1500)
    # run is [0, 1], cost 2000 > 1500, and it's the first run so truncate.
    assert len(out) == 1
    assert len(out[0].chunk_ids) == 1


def test_page_range_set_from_chunks():
    a = make_candidate(chunk_index=0, page_number=3)
    b = make_candidate(chunk_index=1, page_number=5)
    out = build_passages([a], [b], window=1, max_tokens=1000)
    assert out[0].page_start == 3
    assert out[0].page_end == 5


def test_pages_none_when_no_chunk_has_page_number():
    a = make_candidate(chunk_index=0, page_number=None)
    out = build_passages([a], [], window=0, max_tokens=1000)
    assert out[0].page_start is None
    assert out[0].page_end is None


def test_section_from_first_non_none_chunk():
    a = make_candidate(chunk_index=0, section=None)
    b = make_candidate(chunk_index=1, section="Intro")
    out = build_passages([a], [b], window=1, max_tokens=1000)
    assert out[0].section == "Intro"


def test_citation_ids_are_one_indexed():
    s1 = make_candidate(chunk_index=0, rerank_score=0.9)
    s2 = make_candidate(chunk_index=10, rerank_score=0.1)
    out = build_passages([s1, s2], [], window=0, max_tokens=1000)
    assert [p.citation_id for p in out] == [1, 2]


def test_selected_overrides_neighbor_with_same_position():
    """If a chunk appears in both lists, the selected version wins (scores)."""
    shared_id = uuid4()
    n = Candidate(
        chunk_id=shared_id,
        document_id=DOC_A,
        chunk_index=0,
        text="n",
        page_number=None,
        section=None,
        token_count=None,
    )
    s = Candidate(
        chunk_id=shared_id,
        document_id=DOC_A,
        chunk_index=0,
        text="s",
        page_number=None,
        section=None,
        token_count=None,
        rerank_score=0.9,
    )
    out = build_passages([s], [n], window=0, max_tokens=1000)
    assert out[0].text == "s"


# --- render_context --------------------------------------------------


def _passage(
    cid: int = 1,
    *,
    text: str = "text",
    page_start: int | None = None,
    page_end: int | None = None,
    section: str | None = None,
) -> Passage:
    return Passage(
        citation_id=cid,
        document_id=DOC_A,
        chunk_ids=(uuid4(),),
        chunk_range=(0, 0),
        page_start=page_start,
        page_end=page_end,
        section=section,
        text=text,
        score=0.5,
        token_count=10,
    )


def test_render_context_empty():
    assert render_context([]) == ""


def test_render_context_single_page():
    out = render_context([_passage(1, text="hi", page_start=3, page_end=3)])
    assert "[1] (p.3)" in out
    assert "hi" in out


def test_render_context_multi_page():
    out = render_context([_passage(1, text="hi", page_start=3, page_end=7)])
    assert "[1] (pp.3-7)" in out


def test_render_context_no_pages_uses_na():
    out = render_context([_passage(1, text="hi")])
    assert "(n/a)" in out


def test_render_context_includes_section():
    out = render_context([_passage(1, text="hi", section="Intro", page_start=1, page_end=1)])
    assert "[1] Intro (p.1)" in out


def test_render_context_joins_with_separator():
    out = render_context([_passage(1, text="a"), _passage(2, text="b")])
    assert "\n\n---\n\n" in out
    assert out.count("---") == 1

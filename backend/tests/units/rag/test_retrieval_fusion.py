"""Unit tests for RRF and MMR fusion."""

from __future__ import annotations

from uuid import UUID, uuid4

import numpy as np
import pytest
from app.rag.retrieval.ranking import mmr_select, reciprocal_rank_fusion
from app.rag.retrieval.types import Candidate


def make_candidate(
    *,
    chunk_id: UUID | None = None,
    vector_rank: int | None = None,
    keyword_rank: int | None = None,
    vector_score: float | None = None,
    keyword_score: float | None = None,
    rerank_score: float | None = None,
    embedding: np.ndarray | None = None,
    text: str = "text",
) -> Candidate:
    return Candidate(
        chunk_id=chunk_id or uuid4(),
        document_id=uuid4(),
        chunk_index=0,
        text=text,
        page_number=None,
        section=None,
        token_count=None,
        embedding=embedding,
        vector_rank=vector_rank,
        keyword_rank=keyword_rank,
        vector_score=vector_score,
        keyword_score=keyword_score,
        rerank_score=rerank_score,
    )


# --- RRF -------------------------------------------------------------


def test_rrf_empty_inputs():
    assert reciprocal_rank_fusion([], []) == []


def test_rrf_only_vector():
    a = make_candidate(vector_rank=1)
    b = make_candidate(vector_rank=2)
    out = reciprocal_rank_fusion([a, b], [])
    assert [c.chunk_id for c in out] == [a.chunk_id, b.chunk_id]
    assert a.fused_score == pytest.approx(1 / 61)
    assert b.fused_score == pytest.approx(1 / 62)


def test_rrf_only_keyword():
    a = make_candidate(keyword_rank=1)
    out = reciprocal_rank_fusion([], [a])
    assert [c.chunk_id for c in out] == [a.chunk_id]
    assert a.fused_score == pytest.approx(1 / 61)


def test_rrf_skips_candidates_without_rank():
    a = make_candidate(vector_rank=None)
    b = make_candidate(vector_rank=1)
    out = reciprocal_rank_fusion([a, b], [])
    assert [c.chunk_id for c in out] == [b.chunk_id]


def test_rrf_disjoint_merges_both():
    v = make_candidate(vector_rank=1)
    k = make_candidate(keyword_rank=1)
    out = reciprocal_rank_fusion([v], [k])
    assert {c.chunk_id for c in out} == {v.chunk_id, k.chunk_id}


def test_rrf_overlap_combines_scores_and_preserves_vector_rank():
    shared_id = uuid4()
    v = make_candidate(chunk_id=shared_id, vector_rank=1, vector_score=0.9)
    k = make_candidate(chunk_id=shared_id, keyword_rank=2, keyword_score=4.2)
    out = reciprocal_rank_fusion([v], [k])

    assert len(out) == 1
    only = out[0]
    assert only.chunk_id == shared_id
    # vector_rank kept from the vector list; keyword_rank merged in
    assert only.vector_rank == 1
    assert only.keyword_rank == 2
    assert only.vector_score == 0.9
    assert only.keyword_score == 4.2
    assert only.fused_score == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_custom_k():
    a = make_candidate(vector_rank=1)
    reciprocal_rank_fusion([a], [], k=10)
    assert a.fused_score == pytest.approx(1 / 11)


def test_rrf_custom_weights():
    a = make_candidate(vector_rank=1)
    reciprocal_rank_fusion([a], [], vector_weight=2.0)
    assert a.fused_score == pytest.approx(2 / 61)


def test_rrf_sorted_descending_by_fused_score():
    high = make_candidate(vector_rank=1)
    low = make_candidate(vector_rank=10)
    mid = make_candidate(vector_rank=5)
    out = reciprocal_rank_fusion([low, mid, high], [])
    assert [c.chunk_id for c in out] == [high.chunk_id, mid.chunk_id, low.chunk_id]


def test_rrf_does_not_include_keyword_rank_for_non_overlapping():
    a = make_candidate(vector_rank=1)
    b = make_candidate(keyword_rank=1)
    out = reciprocal_rank_fusion([a], [b])
    by_id = {c.chunk_id: c for c in out}
    assert by_id[a.chunk_id].keyword_rank is None
    assert by_id[b.chunk_id].vector_rank is None


# --- MMR -------------------------------------------------------------


def _unit(*coords: float) -> np.ndarray:
    v = np.array(coords, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_mmr_returns_first_k_when_pool_small():
    cands = [make_candidate(embedding=_unit(1, 0)) for _ in range(3)]
    out = mmr_select(cands, k=5, lambda_=0.5)
    assert out == cands[:3]


def test_mmr_lambda_one_returns_first_k():
    cands = [make_candidate(embedding=_unit(1, 0)) for _ in range(5)]
    out = mmr_select(cands, k=2, lambda_=1.0)
    assert out == cands[:2]


def test_mmr_no_embeddings_returns_first_k():
    cands = [make_candidate(embedding=None) for _ in range(4)]
    out = mmr_select(cands, k=2, lambda_=0.5)
    assert out == cands[:2]


def test_mmr_orders_by_relevance_when_no_penalty():
    """lambda_=1 → pure relevance ordering (input assumed sorted)."""
    a = make_candidate(embedding=_unit(1, 0))
    b = make_candidate(embedding=_unit(0, 1))
    out = mmr_select([a, b], k=2, lambda_=1.0)
    assert out == [a, b]


def test_mmr_prefers_diverse_at_low_lambda():
    """lambda_=0 → pick most dissimilar to already selected."""
    a = make_candidate(embedding=_unit(1, 0))
    b = make_candidate(embedding=_unit(1, 0))  # same direction as a
    c = make_candidate(embedding=_unit(0, 1))  # orthogonal

    out = mmr_select([a, b, c], k=2, lambda_=0.0)
    # First pick is index 0 (all equal relevance at start); second
    # should prefer c over b, since b is identical to a.
    assert out[0] is a
    assert out[1] is c


def test_mmr_appends_no_embedding_candidates_after_selected():
    a = make_candidate(embedding=_unit(1, 0))
    b = make_candidate(embedding=_unit(0, 1))
    c = make_candidate(embedding=None)
    out = mmr_select([a, b, c], k=3, lambda_=0.5)
    assert c in out
    # no-emb candidates come after all embedded ones
    assert out.index(c) == 2


def test_mmr_truncates_to_k_with_mixed_pool():
    a = make_candidate(embedding=_unit(1, 0))
    b = make_candidate(embedding=None)
    c = make_candidate(embedding=None)
    out = mmr_select([a, b, c], k=2, lambda_=0.5)
    assert len(out) == 2


def test_mmr_uses_rerank_score_when_present():
    """relevance property prefers rerank_score over fused_score."""
    a = make_candidate(embedding=_unit(1, 0), rerank_score=0.9)
    b = make_candidate(embedding=_unit(0, 1), rerank_score=0.1)
    out = mmr_select([a, b], k=2, lambda_=0.5)
    assert out[0] is a


def test_mmr_handles_zero_relevance_span():
    """All relevance equal → no divide-by-zero, still produces a result."""
    a = make_candidate(embedding=_unit(1, 0), rerank_score=0.5)
    b = make_candidate(embedding=_unit(0, 1), rerank_score=0.5)
    out = mmr_select([a, b], k=2, lambda_=0.5)
    assert len(out) == 2


def test_mmr_zero_vectors_do_not_crash():
    """Norm is guarded by 1e-12; a zero vector should not produce NaN."""
    z = np.zeros(3, dtype=np.float32)
    a = make_candidate(embedding=z)
    b = make_candidate(embedding=_unit(1, 0, 0))
    out = mmr_select([a, b], k=2, lambda_=0.5)
    assert len(out) == 2

from collections.abc import Sequence
from uuid import UUID

import numpy as np

from app.rag.retrieval.types import Candidate


def reciprocal_rank_fusion(
    vector: Sequence[Candidate],
    keyword: Sequence[Candidate],
    *,
    k: int = 60,
    vector_weight: float = 1.0,
    keyword_weight: float = 1.0,
) -> list[Candidate]:
    """score(d) = Σ w / (k + rank). Scale-free, so cosine and ts_rank never get compared directly."""
    merged: dict[UUID, Candidate] = {}

    for c in vector:
        if c.vector_rank is None:
            continue
        merged[c.chunk_id] = c
        c.fused_score = vector_weight / (k + c.vector_rank)

    for c in keyword:
        if c.keyword_rank is None:
            continue
        existing = merged.get(c.chunk_id)
        if existing is None:
            merged[c.chunk_id] = c
            c.fused_score = keyword_weight / (k + c.keyword_rank)
        else:
            existing.keyword_rank = c.keyword_rank
            existing.keyword_score = c.keyword_score
            existing.fused_score += keyword_weight / (k + c.keyword_rank)

    return sorted(merged.values(), key=lambda c: c.fused_score, reverse=True)


def mmr_select(candidates: Sequence[Candidate], *, k: int, lambda_: float) -> list[Candidate]:
    """Maximal Marginal Relevance: λ·relevance − (1−λ)·max_sim_to_selected.

    Relevance is min-max normalized (rerank score if present, else fused score)
    so it is comparable to cosine similarity in [0, 1]."""
    if len(candidates) <= k or lambda_ >= 1.0:
        return list(candidates[:k])

    pool = [c for c in candidates if c.embedding is not None]
    no_emb = [c for c in candidates if c.embedding is None]
    if not pool:
        return list(candidates[:k])

    rel = np.array([c.relevance for c in pool], dtype=np.float32)
    span = rel.max() - rel.min()
    rel = (rel - rel.min()) / span if span > 0 else np.ones_like(rel)

    embeddings: list[np.ndarray] = [c.embedding for c in pool if c.embedding is not None]
    emb = np.stack(embeddings)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    sim = emb @ emb.T
    selected: list[int] = []
    remaining = list(range(len(pool)))
    max_sim = np.zeros(len(pool), dtype=np.float32)

    while remaining and len(selected) < k:
        scores = lambda_ * rel[remaining] - (1 - lambda_) * max_sim[remaining]
        best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)
        max_sim = np.maximum(max_sim, sim[best])

    result = [pool[i] for i in selected]
    return (result + no_emb)[:k]

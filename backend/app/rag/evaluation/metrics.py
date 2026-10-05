"""Retrieval and answer quality metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence


def hit_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """1.0 if any gold chunk appears in top-k, else 0.0."""
    if not gold:
        return 0.0
    return float(any(r in gold for r in retrieved[:k]))


def precision_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Fraction of top-k that are relevant."""
    if not gold or k == 0:
        return 0.0
    top = retrieved[:k]
    return sum(1 for r in top if r in gold) / k


def recall_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Fraction of gold chunks found in top-k."""
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    return len(top & gold) / len(gold)


def mrr_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Reciprocal rank of first relevant chunk in top-k."""
    if not gold:
        return 0.0
    for i, r in enumerate(retrieved[:k], start=1):
        if r in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    if not gold:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 2)  # i+2 because i is 0-indexed, log2(1)=0
        for i, r in enumerate(retrieved[:k])
        if r in gold
    )
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def rank_metrics(
    retrieved: Sequence[str],
    gold: set[str],
    ks: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> dict[str, float | None]:
    """Compute all retrieval metrics at multiple k values."""
    if not gold:
        return {f"{m}@{k}": None for k in ks for m in ("hit", "precision", "recall", "mrr", "ndcg")}

    result: dict[str, float | None] = {}
    for k in ks:
        result[f"hit@{k}"] = hit_at_k(retrieved, gold, k)
        result[f"precision@{k}"] = precision_at_k(retrieved, gold, k)
        result[f"recall@{k}"] = recall_at_k(retrieved, gold, k)
        result[f"mrr@{k}"] = mrr_at_k(retrieved, gold, k)
        result[f"ndcg@{k}"] = ndcg_at_k(retrieved, gold, k)
    return result


def token_f1(prediction: str, reference: str) -> float:
    """Word-level F1 between prediction and reference."""
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

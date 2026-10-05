"""Test retrieval metrics calculations."""

from app.rag.evaluation.metrics import (
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    rank_metrics,
    recall_at_k,
    token_f1,
)


class TestHitAtK:
    def test_hit_found(self):
        assert hit_at_k(["a", "b", "c"], {"b"}, 3) == 1.0

    def test_hit_not_found(self):
        assert hit_at_k(["a", "b", "c"], {"d"}, 3) == 0.0

    def test_hit_beyond_k(self):
        assert hit_at_k(["a", "b", "c", "d"], {"d"}, 2) == 0.0

    def test_empty_gold(self):
        assert hit_at_k(["a"], set(), 5) == 0.0


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_half_relevant(self):
        assert precision_at_k(["a", "b", "c", "d"], {"a", "c"}, 4) == 0.5

    def test_none_relevant(self):
        assert precision_at_k(["x", "y"], {"a"}, 2) == 0.0


class TestRecallAtK:
    def test_full_recall(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "x"], {"a", "b"}, 2) == 0.5


class TestMRR:
    def test_first(self):
        assert mrr_at_k(["a", "b"], {"a"}, 5) == 1.0

    def test_second(self):
        assert mrr_at_k(["x", "a"], {"a"}, 5) == 0.5

    def test_not_found(self):
        assert mrr_at_k(["x", "y"], {"a"}, 2) == 0.0


class TestNDCG:
    def test_perfect(self):
        assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_imperfect(self):
        # gold={"a"}, retrieved=["x", "a"] at k=2
        # DCG = 1/log2(3) ≈ 0.6309
        # IDCG = 1/log2(2) = 1.0
        val = ndcg_at_k(["x", "a"], {"a"}, 2)
        assert 0.63 < val < 0.64


class TestTokenF1:
    def test_exact(self):
        assert token_f1("nine core tables", "nine core tables") == 1.0

    def test_partial(self):
        f1 = token_f1("the system has nine tables", "nine core tables")
        assert 0.3 < f1 < 0.7

    def test_no_overlap(self):
        assert token_f1("hello world", "foo bar") == 0.0


class TestRankMetrics:
    def test_returns_all_metrics(self):
        result = rank_metrics(["a", "b", "c"], {"b"}, ks=(1, 3))
        assert "hit@1" in result
        assert "mrr@3" in result
        assert "ndcg@3" in result

    def test_empty_gold_returns_none(self):
        result = rank_metrics(["a"], set(), ks=(1,))
        assert all(v is None for v in result.values())

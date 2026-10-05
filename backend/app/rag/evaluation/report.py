"""Aggregate eval results into a readable report."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from app.rag.evaluation.dataset import EvalResult


def aggregate(results: list[EvalResult]) -> dict:
    """Aggregate per-case metrics into mean/median/min/max per stage."""
    if not results:
        return {}

    # Collect all metric values per stage per metric name
    # stage -> metric_name -> list[float]
    by_stage: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        for stage, stage_metrics in r.metrics.items():
            for name, value in stage_metrics.items():
                if value is not None:
                    by_stage[stage][name].append(value)

    agg: dict[str, dict[str, dict[str, float]]] = {}
    for stage, metric_values in by_stage.items():
        agg[stage] = {}
        for name, values in sorted(metric_values.items()):
            agg[stage][name] = {
                "mean": round(statistics.mean(values), 4),
                "median": round(statistics.median(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "n": len(values),
            }

    return agg


def latency_stats(results: list[EvalResult]) -> dict:
    """Aggregate latency across all cases."""
    if not results:
        return {}

    totals = [r.total_ms for r in results]

    # Per-stage
    stage_times: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for stage, ms in r.timings_ms.items():
            stage_times[stage].append(ms)

    stats = {
        "total_ms": {
            "mean": round(statistics.mean(totals), 1),
            "p50": round(statistics.median(totals), 1),
            "p95": round(sorted(totals)[int(len(totals) * 0.95)] if totals else 0, 1),
            "max": round(max(totals), 1),
        },
    }
    for stage, times in sorted(stage_times.items()):
        stats[stage] = {
            "mean": round(statistics.mean(times), 1),
            "p50": round(statistics.median(times), 1),
            "p95": round(sorted(times)[int(len(times) * 0.95)] if times else 0, 1),
            "max": round(max(times), 1),
        }
    return stats


def print_report(results: list[EvalResult]) -> None:
    """Print a human-readable evaluation report to stdout."""
    n = len(results)
    answerable = [r for r in results if r.case.answerable]
    unanswerable = [r for r in results if not r.case.answerable]
    reranked = [r for r in results if r.reranked]

    print(f"\n{'=' * 70}")
    print("RAG EVALUATION REPORT")
    print(f"{'=' * 70}")
    print(f"Total cases:      {n}")
    print(f"Answerable:       {len(answerable)}")
    print(f"Unanswerable:     {len(unanswerable)}")
    print(f"Reranked:         {len(reranked)}/{n}")
    print()

    # Stage-by-stage metrics (answerable only)
    if answerable:
        agg = aggregate(answerable)
        # Pick the key metrics to display
        key_metrics = ["hit@1", "hit@3", "hit@5", "mrr@5", "recall@5", "ndcg@5"]

        # Determine which stages exist
        stage_order = [
            "vector",
            "keyword",
            "fused",
            "post_rerank",
            "selected",
            "expanded",
        ]
        stages = [s for s in stage_order if s in agg]

        # Print table header
        header = f"{'stage':<15}" + "".join(f"{m:>12}" for m in key_metrics)
        print(header)
        print("-" * len(header))

        for stage in stages:
            row = f"{stage:<15}"
            for m in key_metrics:
                val = agg[stage].get(m, {}).get("mean")
                if val is not None:
                    row += f"{val:>12.4f}"
                else:
                    row += f"{'—':>12}"
            print(row)

        print()

        # Per-stage improvement over vector-only
        if "vector" in agg and "fused" in agg:
            print("Fusion lift over vector-only:")
            for m in ["hit@5", "mrr@5", "recall@5"]:
                v = agg["vector"].get(m, {}).get("mean", 0)
                f = agg["fused"].get(m, {}).get("mean", 0)
                delta = f - v
                print(f"  {m}: {v:.4f} → {f:.4f} ({delta:+.4f})")
            print()

        if "fused" in agg and "post_rerank" in agg:
            print("Rerank lift over fusion:")
            for m in ["hit@5", "mrr@5", "recall@5"]:
                f = agg["fused"].get(m, {}).get("mean", 0)
                r = agg["post_rerank"].get(m, {}).get("mean", 0)
                delta = r - f
                print(f"  {m}: {f:.4f} → {r:.4f} ({delta:+.4f})")
            print()

    # Latency
    print(f"\n{'LATENCY':=^70}")
    lat = latency_stats(results)
    for stage, stats in lat.items():
        print(f"  {stage:<20} mean={stats['mean']:>8.1f}ms  p50={stats['p50']:>8.1f}ms  p95={stats['p95']:>8.1f}ms  max={stats['max']:>8.1f}ms")
    print()

    # Per-case failures (cases where hit@5 == 0)
    if answerable:
        failures = [r for r in answerable if r.metrics.get("selected", {}).get("hit@5") == 0.0]
        if failures:
            print(f"\n{'FAILURES (hit@5=0 in selected)':=^70}")
            for r in failures:
                print(f"  [{r.case.id}] {r.case.query[:80]}")
                print(f"    gold: {sorted(r.case.relevant_chunk_ids)[:3]}...")
                top3 = r.stage_chunks.get("selected", [])[:3]
                print(f"    got:  {top3}")
            print()


def save_report(results: list[EvalResult], path: str | Path) -> None:
    """Save full results as JSON for later analysis."""
    out = []
    for r in results:
        out.append(
            {
                "case_id": r.case.id,
                "query": r.case.query,
                "answerable": r.case.answerable,
                "reranked": r.reranked,
                "total_ms": r.total_ms,
                "timings_ms": r.timings_ms,
                "stage_chunks": r.stage_chunks,
                "metrics": r.metrics,
                "passages_count": len(r.passages_text),
            }
        )

    with open(path, "w") as f:
        json.dump(
            {
                "n_cases": len(results),
                "aggregate": aggregate(results),
                "latency": latency_stats(results),
                "cases": out,
            },
            f,
            indent=2,
        )

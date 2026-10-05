"""Run evaluation from command line.

Usage:
    python -m eval.cli \
        --eval-set tests/eval/fixtures/eval_set.jsonl \
        --output eval_results.json \
        --top-k 8
"""

import argparse
import asyncio
import sys

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.rag.evaluation.dataset import load_eval_set
from app.rag.evaluation.report import print_report, save_report
from app.rag.evaluation.runner import run_eval


async def main(args: argparse.Namespace) -> None:
    cases = load_eval_set(args.eval_set)
    if not cases:
        print("No eval cases found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(cases)} eval cases")

    # Optional reranker
    reranker = None
    if settings.retrieval.rerank_enabled:
        try:
            from app.rag.retrieval.rerank import TEIReranker

            reranker = TEIReranker(url=settings.retrieval.rerank_url)
            print(f"Reranker enabled: {settings.retrieval.rerank_url}")
        except Exception as e:  # noqa: BLE001 - optional reranker; any init failure disables it
            print(f"Reranker init failed: {e}", file=sys.stderr)

    async with AsyncSessionLocal() as session:
        results = await run_eval(
            session,
            cases,
            cfg=settings.retrieval,
            reranker=reranker,
            top_k=args.top_k,
        )

    print_report(results)

    if args.output:
        save_report(results, args.output)
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG evaluation pipeline")
    parser.add_argument("--eval-set", required=True, help="Path to JSONL eval set")
    parser.add_argument("--output", default=None, help="Path to save JSON results")
    parser.add_argument("--top-k", type=int, default=8, help="Top-k for MMR selection")
    args = parser.parse_args()
    asyncio.run(main(args))

"""Evaluation pipeline runner.

Runs eval cases through RetrievalService, captures per-stage chunk IDs,
computes metrics at every stage.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.retrieval_settings import RetrievalSettings
from app.rag.evaluation.dataset import EvalCase, EvalResult
from app.rag.evaluation.metrics import rank_metrics
from app.rag.retrieval.rerank import Reranker
from app.rag.retrieval.retrieval import RetrievalService

log = get_logger(__name__)


class InstrumentedRetrievalService(RetrievalService):
    """RetrievalService subclass that captures per-stage candidates.

    Does NOT change any retrieval logic. Only records what each stage
    produced so the eval runner can compute metrics per stage.

    Why subclass instead of modifying the original?
    - Zero production risk: original service is untouched.
    - Eval-only code stays in eval package.
    - Can be deleted without affecting anything.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        cfg: RetrievalSettings | None = None,
        reranker: Reranker | None = None,
    ):
        super().__init__(session, cfg=cfg, reranker=reranker)
        # Populated during retrieve()
        self._trace: dict[str, list[str]] = {}
        self._trace_scores: dict[str, list[float]] = {}

    async def retrieve_instrumented(
        self,
        *,
        user_id: UUID,
        query: str,
        top_k: int = 8,
        document_ids: list[UUID] | None = None,
    ) -> EvalResult:
        """Run full pipeline, capture chunk IDs at every stage."""
        from app.rag.retrieval.context import build_passages
        from app.rag.retrieval.ranking import (
            mmr_select,
            reciprocal_rank_fusion,
        )

        self._trace = {}
        self._trace_scores = {}

        if document_ids:
            await self._validate_scope(user_id, document_ids)

        cfg = self.cfg
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        query = " ".join(query.split())

        # --- embed + keyword ---
        t0 = time.perf_counter()
        embed_task = asyncio.create_task(self._embed(query))
        keyword = await self.keyword_search(
            user_id=user_id,
            document_ids=document_ids,
            query=query,
            language=cfg.fts_language,
            limit=cfg.keyword_candidates,
        )
        query_vector = await embed_task
        timings["embed+keyword"] = round((time.perf_counter() - t0) * 1000, 1)

        self._record("keyword", keyword)

        # --- vector ---
        t0 = time.perf_counter()
        vector = await self.vector_search(
            user_id=user_id,
            document_ids=document_ids,
            ef_search=cfg.hnsw_ef_search,
            embedding=query_vector,
            iterative_scan=cfg.hnsw_iterative_scan,
            limit=cfg.vector_candidates,
        )
        timings["vector"] = round((time.perf_counter() - t0) * 1000, 1)

        self._record("vector", vector)

        # --- fuse ---
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(
            vector=vector,
            keyword=keyword,
            k=cfg.rrf_k,
            vector_weight=cfg.vector_weight,
            keyword_weight=cfg.keyword_weight,
        )
        timings["fuse"] = round((time.perf_counter() - t0) * 1000, 1)

        self._record("fused", fused)

        if not fused:
            return self._build_empty_result(query, timings, t_total)

        # --- rerank ---
        candidates = fused[: cfg.rerank_candidates]
        self._record("rerank_input", candidates)
        reranked = False

        if cfg.rerank_enabled and self.reranker is not None:
            t0 = time.perf_counter()
            reranked = await self._rerank(query, candidates)
            timings["rerank"] = round((time.perf_counter() - t0) * 1000, 1)

            if reranked:
                candidates.sort(
                    key=lambda c: c.rerank_score if c.rerank_score is not None else float("-inf"),
                    reverse=True,
                )
                if cfg.rerank_min_score is not None:
                    threshold = cfg.rerank_min_score
                    candidates = [c for c in candidates if c.rerank_score is not None and c.rerank_score >= threshold]

        self._record("post_rerank", candidates)

        # --- mmr ---
        t0 = time.perf_counter()
        selected = mmr_select(candidates, k=top_k, lambda_=cfg.mmr_lambda)
        timings["mmr"] = round((time.perf_counter() - t0) * 1000, 1)

        self._record("selected", selected)

        # --- expand ---
        t0 = time.perf_counter()
        neighbors = await self._neighbors(selected, window=cfg.neighbor_window)
        passages = build_passages(
            selected=selected,
            neighbors=neighbors,
            window=cfg.neighbor_window,
            max_tokens=cfg.max_context_tokens,
        )
        timings["expand"] = round((time.perf_counter() - t0) * 1000, 1)

        total_ms = round((time.perf_counter() - t_total) * 1000, 1)

        # expanded = selected chunks + neighbor chunks
        all_expanded = selected + neighbors
        self._record("expanded", all_expanded)

        return EvalResult(
            case=EvalCase(
                id="",
                query=query,
                user_id=user_id,
                document_ids=document_ids or [],
                relevant_chunk_ids=set(),
            ),
            stage_chunks=dict(self._trace),
            stage_scores=dict(self._trace_scores),
            metrics={},
            timings_ms=timings,
            reranked=reranked,
            passages_text=[p.text for p in passages],
            total_ms=total_ms,
        )

    def _record(self, stage: str, candidates: Sequence) -> None:
        """Record chunk IDs and scores for a stage."""
        self._trace[stage] = [str(c.chunk_id) for c in candidates]
        self._trace_scores[stage] = [c.relevance if hasattr(c, "relevance") else 0.0 for c in candidates]

    def _build_empty_result(
        self,
        query: str,
        timings: dict[str, float],
        t_start: float,
    ) -> EvalResult:
        return EvalResult(
            case=EvalCase(
                id="",
                query=query,
                user_id=UUID(int=0),
                document_ids=[],
                relevant_chunk_ids=set(),
            ),
            stage_chunks=dict(self._trace),
            stage_scores=dict(self._trace_scores),
            metrics={},
            timings_ms=timings,
            reranked=False,
            passages_text=[],
            total_ms=round((time.perf_counter() - t_start) * 1000, 1),
        )


def compute_metrics(
    result: EvalResult,
    gold: set[str],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, dict[str, float | None]]:
    """Compute retrieval metrics for every recorded stage."""
    metrics = {}
    for stage, chunk_ids in result.stage_chunks.items():
        metrics[stage] = rank_metrics(chunk_ids, gold, ks)
    return metrics


async def run_eval(
    session: AsyncSession,
    cases: list[EvalCase],
    *,
    cfg: RetrievalSettings | None = None,
    reranker: Reranker | None = None,
    top_k: int = 8,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> list[EvalResult]:
    """Run all eval cases, compute metrics, return results."""
    service = InstrumentedRetrievalService(session, cfg=cfg, reranker=reranker)
    results: list[EvalResult] = []

    for case in cases:
        try:
            result = await service.retrieve_instrumented(
                user_id=case.user_id,
                query=case.query,
                top_k=top_k,
                document_ids=case.document_ids or None,
            )
            result.case = case
            result.metrics = compute_metrics(result, case.relevant_chunk_ids, ks)
            results.append(result)

            log.info(
                "eval_case_done",
                case_id=case.id,
                total_ms=result.total_ms,
                reranked=result.reranked,
                stages=list(result.stage_chunks.keys()),
            )
        except Exception:
            log.exception("eval_case_failed", case_id=case.id)
            continue

    return results

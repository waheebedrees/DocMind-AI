import asyncio
import time
from collections.abc import Sequence
from contextlib import contextmanager
from uuid import UUID

import numpy as np
from pydantic import BaseModel
from sqlalchemy import func, select, text, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import DocumentNotReady, EmbeddingUnavailable, NotFoundError
from app.core.logging import get_logger
from app.db.repositories.documents import DocumentRepository
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.rag.ingestion import embed_in_batches
from app.rag.retrieval.context import build_passages, render_context
from app.rag.retrieval.ranking import mmr_select, reciprocal_rank_fusion
from app.rag.retrieval.types import Candidate, RetrievalResult
from app.services.llm_clients import get_embedder

log = get_logger(__name__)


_COLS = (
    DocumentChunk.id,
    DocumentChunk.document_id,
    DocumentChunk.chunk_index,
    DocumentChunk.text,
    DocumentChunk.page_number,
    DocumentChunk.section,
    DocumentChunk.token_count,
    DocumentChunk.embedding,
)


def _scope(stmt, user_id: UUID, document_ids: Sequence[UUID] | None):
    stmt = stmt.join(Document, Document.id == DocumentChunk.document_id).where(
        Document.user_id == user_id,
        Document.status == DocumentStatus.INDEXED,
    )
    if document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
    return stmt


def _to_candidate(row) -> Candidate:
    emb = row.embedding
    return Candidate(
        chunk_id=row.id,
        document_id=row.document_id,
        chunk_index=row.chunk_index,
        text=row.text,
        page_number=row.page_number,
        section=row.section,
        token_count=row.token_count,
        embedding=None if emb is None else np.asarray(emb, dtype=np.float32),
    )


class RetrievalSettings(BaseModel):
    # candidate generation
    vector_candidates: int = 50
    keyword_candidates: int = 50
    fts_language: str = "english"
    hnsw_ef_search: int = 100
    hnsw_iterative_scan: bool = True  # requires pgvector >= 0.8

    # fusion
    rrf_k: int = 60
    vector_weight: float = 1.0
    keyword_weight: float = 1.0

    # rerank
    rerank_enabled: bool = False

    # e.g. TEI: http://reranker:8080/rerank
    rerank_url: str | None = None
    rerank_candidates: int = 30
    rerank_timeout_s: float = 20.0
    # model-specific; calibrate on your eval set
    rerank_min_score: float | None = None

    # selection
    mmr_lambda: float = 0.7
    neighbor_window: int = 1
    max_context_tokens: int = 6000


class RetrievalService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.cfg = RetrievalSettings()
        self.docs = DocumentRepository(session)
        self.reranker = None

    async def _validate_scope(self, user_id: UUID, document_ids: list[UUID]) -> None:
        for did in set(document_ids):
            doc = await self.docs.get_for_user(user_id, did)
            if doc is None:
                raise NotFoundError("document not found", code="invalid_document_id")
            if doc.status != DocumentStatus.INDEXED:
                raise DocumentNotReady(f"document {did} is {doc.status.value}")

    async def _neighbors(self, selected: list[Candidate], window: int) -> list[Candidate]:
        """Fetch chunks by (document_id, chunk_index). Ownership was already
        enforced for these document_ids by the scoped searches."""

        if window <= 0:
            return []
        have = {(c.document_id, c.chunk_index) for c in selected}
        wanted = {(c.document_id, i) for c in selected for i in range(c.chunk_index - window, c.chunk_index + window + 1) if i >= 0} - have

        positions = sorted(wanted)

        if not positions:
            return []
        stmt = select(*_COLS).where(tuple_(DocumentChunk.document_id, DocumentChunk.chunk_index).in_(list(positions)))
        rows = (await self.session.execute(stmt)).all()

        return [_to_candidate(r) for r in rows]

    async def vector_search(
        self,
        *,
        user_id: UUID,
        embedding: Sequence[float],
        document_ids: Sequence[UUID] | None,
        limit: int,
        ef_search: int,
        iterative_scan: bool,
    ) -> list[Candidate]:

        # set_config(..., true) == SET LOCAL: scoped to this transaction and safe with pooling.
        await self.session.execute(text("SELECT set_config('hnsw.ef_search', :v, true)"), {"v": str(ef_search)})
        if iterative_scan:
            # Without this, HNSW returns ef_search rows *before* the user filter,
            # so tenants with few docs get far fewer than `limit` results.
            await self.session.execute(text("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)"))

        distance = DocumentChunk.embedding.cosine_distance(embedding).label("distance")
        stmt = _scope(select(*_COLS, distance), user_id, document_ids).order_by(distance).limit(limit)

        rows = (await self.session.execute(stmt)).all()

        # relaxed_order may return slightly unordered rows
        rows = sorted(rows, key=lambda r: r.distance)
        out = []
        for rank, r in enumerate(rows, start=1):
            c = _to_candidate(r)
            c.vector_rank, c.vector_score = rank, 1.0 - float(r.distance)
            out.append(c)
        return out

    async def keyword_search(
        self,
        *,
        user_id: UUID,
        query: str,
        document_ids: Sequence[UUID] | None,
        limit: int,
        language: str,
    ) -> list[Candidate]:
        # websearch_to_tsquery never raises on user input (quotes, OR, -exclusion).
        tsq = func.websearch_to_tsquery(language, query)
        rank = func.ts_rank_cd(DocumentChunk.text_search, tsq, 32).label("rank")  # 32 = normalize to 0..1
        stmt = _scope(select(*_COLS, rank), user_id, document_ids).where(DocumentChunk.text_search.op("@@")(tsq)).order_by(rank.desc()).limit(limit)
        out = []
        for i, r in enumerate((await self.session.execute(stmt)).all(), start=1):
            c = _to_candidate(r)
            c.keyword_rank, c.keyword_score = i, float(r.rank)
            out.append(c)
        return out

    async def retrieve(
        self,
        *,
        user_id: UUID,
        query: str,
        top_k: int = 8,
        document_ids: list[UUID] | None = None,
    ) -> RetrievalResult:
        if document_ids:
            await self._validate_scope(user_id, document_ids)

        cfg = self.cfg
        timings: dict[str, float] = {}

        @contextmanager
        def timed(name: str):
            t0 = time.perf_counter()
            try:
                yield
            finally:
                timings[name] = round((time.perf_counter() - t0) * 1000, 1)

        query = " ".join(query.split())
        if not query:
            raise ValueError("query must not be empty")

        # 1. Embedding (network) and keyword search (DB) run concurrently.
        #    Only ONE coroutine touches the AsyncSession, which is required:
        #    a single session must never run concurrent queries.
        with timed("embed"):
            embed_task = asyncio.create_task(self._embed(query))
            try:
                keyword = await self.keyword_search(
                    user_id=user_id,
                    document_ids=document_ids,
                    query=query,
                    language=cfg.fts_language,
                    limit=cfg.keyword_candidates,
                )
                query_vector = await embed_task
            except BaseException:
                embed_task.cancel()
                raise

        # 2. Vector search
        with timed("vector"):
            vector = await self.vector_search(
                user_id=user_id,
                document_ids=document_ids,
                ef_search=cfg.hnsw_ef_search,
                embedding=query_vector,
                iterative_scan=cfg.hnsw_iterative_scan,
                limit=cfg.vector_candidates,
            )

        # 3. Fuse

        fused = reciprocal_rank_fusion(
            vector=vector,
            keyword=keyword,
            k=cfg.rrf_k,
            vector_weight=cfg.vector_weight,
            keyword_weight=cfg.keyword_weight,
        )

        counts = {
            "vector": len(vector),
            "keyword": len(keyword),
            "fused": len(fused),
        }

        if not fused:
            return RetrievalResult(query=query, passages=[], context="", candidate_counts=counts, reranked=False, timings_ms=timings)

        candidates = fused[: cfg.rerank_candidates]
        reranked = False

        if self.cfg.rerank_enabled and self.reranker is not None:
            with timed("rerank"):
                reranked = await self._rerank(query, candidates)

                if reranked:
                    candidates.sort(
                        key=lambda c: c.rerank_score if c.rerank_score is not None else float("-inf"),
                        reverse=True,
                    )
                    if cfg.rerank_min_score is not None:
                        threshold = cfg.rerank_min_score  # local so mypy narrows in the comprehension
                        candidates = [c for c in candidates if c.rerank_score is not None and c.rerank_score >= threshold]
                        counts["after_threshold"] = len(candidates)

        # 5. Diversify
        with timed("mmr"):
            selected = mmr_select(candidates, k=top_k, lambda_=cfg.mmr_lambda)

        # 6. Neighbor expansion and context
        with timed("expand"):
            neighbors = await self._neighbors(selected, window=cfg.neighbor_window)
            passages = build_passages(selected=selected, neighbors=neighbors, window=cfg.neighbor_window, max_tokens=cfg.max_context_tokens)
            context = render_context(passages)

        log.info(
            "retrieval_done",
            user_id=str(user_id),
            top_k=top_k,
            reranked=reranked,
            passages=len(passages),
            timings_ms=timings,
            **counts,
        )
        return RetrievalResult(query=query, passages=passages, context=context, candidate_counts=counts, reranked=reranked, timings_ms=timings)

    async def _embed(self, query: str) -> list[float]:
        spec = settings.embedding_spec
        try:
            async with asyncio.timeout(spec.embed_timeout_s):
                vecs = await embed_in_batches(get_embedder(), [spec.query_prefix + query])
        except (TimeoutError, OSError) as exc:
            raise EmbeddingUnavailable(str(exc)) from exc
        vec = list(vecs[0])
        expected = settings.embedding_spec.dimension
        if len(vec) != expected:
            raise RuntimeError(f"query embedding dim {len(vec)} != index dim {expected}")
        return vec

    async def _rerank(self, query: str, candidates: list[Candidate]) -> bool:
        reranker = self.reranker
        if reranker is None:
            return False
        log.info("calling_rerank")
        try:
            async with asyncio.timeout(self.cfg.rerank_timeout_s):
                scores = await reranker.score(query, [c.text for c in candidates])
        except Exception as exc:  # noqa: BLE001 - reranking is an optional quality layer
            log.warning("rerank_failed_fallback_to_fusion", error=repr(exc))
            return False

        for c, s in zip(candidates, scores, strict=True):
            c.rerank_score = s
        return True

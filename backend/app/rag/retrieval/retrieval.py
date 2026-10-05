"""Hybrid retrieval service.

This module implements the read path for DocMind: given a user's query
and optional document scope, return the most relevant passages with
enough context to render into an LLM prompt.

The pipeline is:

1. **Candidate generation** — vector search (pgvector HNSW) and keyword
   search (Postgres FTS) run *concurrently*. The vector query is
   scoped to the user's INDEXED documents; the keyword query uses the
   same scope.
2. **Fusion** — Reciprocal Rank Fusion merges the two ranked lists.
3. **Rerank** (optional) — a cross-encoder rescores the top N
   candidates. Disabled by default; failures here are logged and
   silently fall back to the fused ranking.
4. **Diversification** — Maximal Marginal Relevance selects ``top_k``
   passages that balance relevance and novelty.
5. **Expansion** — neighboring chunks (by ``chunk_index``) are fetched
   and merged into the passages so sentences aren't cut mid-thought.

Every query is scoped to a ``user_id``. Cross-tenant reads are the
responsibility of the caller passing the right ``user_id``; the service
does not authenticate.
"""

import asyncio
import time
from collections.abc import Sequence
from contextlib import contextmanager
from uuid import UUID

import numpy as np
from sqlalchemy import func, select, text, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import DocumentNotReady, EmbeddingUnavailable, NotFoundError
from app.core.logging import get_logger
from app.core.retrieval_settings import RetrievalSettings
from app.db.repositories.documents import DocumentRepository
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.rag.ingestion import embed_in_batches
from app.rag.retrieval.context import build_passages, render_context
from app.rag.retrieval.query_preprocess import QueryPreprocess
from app.rag.retrieval.ranking import mmr_select, reciprocal_rank_fusion
from app.rag.retrieval.rerank import Reranker
from app.rag.retrieval.types import Candidate, RetrievalResult
from app.services.llm_clients import get_embedder

log = get_logger(__name__)

# Columns fetched for every candidate. Keep in sync with
# ``Candidate``'s fields — ``_to_candidate`` reads each column by name.
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
    """Apply user ownership and index-status filters to a chunk query.

    Joins ``documents`` and constrains the result to chunks whose
    parent document is owned by ``user_id`` and currently INDEXED.
    This is the single point where the retrieval path enforces
    ownership; every search function funnels through it.

    Args:
        stmt: A ``select()`` over ``DocumentChunk``.
        user_id: The authenticated user whose documents may be read.
        document_ids: Optional allow-list. When non-empty, only these
            documents are searched. Ownership is *not* verified here —
            the caller must have already checked that every ID belongs
            to ``user_id`` (see ``_validate_scope``).

    Returns:
        The same statement with the join and where-clauses applied.
    """
    stmt = stmt.join(Document, Document.id == DocumentChunk.document_id).where(
        Document.user_id == user_id,
        Document.status == DocumentStatus.INDEXED,
    )
    if document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
    return stmt


def _to_candidate(row) -> Candidate:
    """Convert a DB row into a ``Candidate``, decoding the embedding.

    The embedding column comes back from asyncpg as a list of floats;
    MMR and rerank expect a numpy array, so it's converted here rather
    than in each consumer.

    Args:
        row: A row containing every column in ``_COLS``.

    Returns:
        A ``Candidate`` with ``embedding`` set to a ``float32`` numpy
        array, or ``None`` if the column was NULL (which happens when
        a chunk was written without an embedding — should not occur for
        indexed documents).
    """
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


class RetrievalService:
    """Hybrid retrieval over a user's indexed documents.

    Construct with an open ``AsyncSession``. The service is not
    thread-safe and not concurrent-safe against itself: a single
    instance must not be used from two coroutines at once, because
    ``AsyncSession`` is not. Use one instance per request.

    The reranker is set externally (``service.reranker = my_reranker``)
    after construction; the service treats a missing reranker as
    "rerank disabled" regardless of ``cfg.rerank_enabled``.

    Args:
        session: The request's database session.
        cfg: Optional override. Defaults to ``settings.retrieval``.
            Pass a custom instance in tests; production should
            always use the default.
        reranker: Optional reranker. If ``cfg.rerank_enabled`` is
            True and this is None, reranking is skipped and a
            warning is logged — see ``_rerank``.

    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        cfg: RetrievalSettings | None = None,
        reranker: "Reranker | None" = None,
    ):
        self.session = session
        self.cfg = cfg or RetrievalSettings()
        self.docs = DocumentRepository(session)
        self.qp = QueryPreprocess(language=self.cfg.fts_language)

        self.reranker = reranker

    async def _validate_scope(self, user_id: UUID, document_ids: list[UUID]) -> None:
        """Verify every requested document belongs to the user and is INDEXED.

        Called once at the top of ``retrieve`` when ``document_ids`` is
        non-empty. The searches themselves re-enforce ownership via
        ``_scope``; this method's job is to fail *early* with a clear
        error rather than silently returning fewer results because a
        document didn't match the filter.

        Args:
            user_id: The authenticated user.
            document_ids: The requested scope. Duplicates are tolerated.

        Raises:
            NotFoundError: A document does not exist or is not owned by
                ``user_id``. Uses code ``invalid_document_id`` so the
                caller can't distinguish "doesn't exist" from "not
                yours" — deliberate, to avoid leaking document IDs.
            DocumentNotReady: A document exists and is owned but is not
                in INDEXED status. The message includes the current
                status.
        """
        for did in set(document_ids):
            doc = await self.docs.get_for_user(user_id, did)
            if doc is None:
                raise NotFoundError("document not found", code="invalid_document_id")
            if doc.status != DocumentStatus.INDEXED:
                raise DocumentNotReady(f"document {did} is {doc.status.value}")

    async def _neighbors(self, selected: list[Candidate], window: int) -> list[Candidate]:
        """Fetch chunks adjacent to the selected ones for context.

        Given a set of selected chunks, find every ``(document_id,
        chunk_index)`` within ``window`` positions and fetch those
        chunks in one query. Chunks already in ``selected`` are excluded.

        Ownership was enforced by the scoped searches that produced
        ``selected``, so the query itself does not re-join
        ``documents``. This is a deliberate optimization — it means
        ``_neighbors`` must not be called with candidates from an
        unscoped source.

        Args:
            selected: The chunks chosen by MMR.
            window: How many positions on either side to include. 0
                returns an empty list without querying.

        Returns:
            The neighboring chunks, unordered.
        """
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
        """Find the ``limit`` chunks closest to the query embedding.

        Sets two pgvector parameters as transaction-local configs
        (``SET LOCAL``), so they don't leak across pooled connections.

        Args:
            user_id: Owner filter. Applied via ``_scope``.
            embedding: Query vector. Length must match the column
                dimension; mismatches fail at the SQL layer.
            document_ids: Optional allow-list. Ownership must already
                be verified by the caller.
            limit: Maximum number of candidates to return.
            ef_search: Value for ``hnsw.ef_search``. Higher = better
                recall, slower. Typically 2-4x ``limit``.
            iterative_scan: If True, set ``hnsw.iterative_scan`` to
                ``relaxed_order`` so the HNSW index keeps scanning past
                the initial ``ef_search`` window until the user filter
                has enough rows. Requires pgvector >= 0.8; older
                versions raise on the ``set_config`` call.

        Returns:
            Candidates sorted by ascending cosine distance, with
            ``vector_rank`` (1-based) and ``vector_score``
            (``1 - distance``) populated on each.
        """
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
        """Full-text search via Postgres ``ts_rank_cd``.

        Builds an OR ``to_tsquery`` from alphanumeric tokens. AND-style
        helpers such as ``websearch_to_tsquery`` drop a chunk if any
        question term is missing, which is too strict for RAG candidate
        generation. OR keeps high recall; ``ts_rank_cd`` orders by how
        many tokens matched.

        Returns:
            Candidates sorted by descending ``ts_rank_cd``. Empty if the
            query tokenizes to nothing or has no lexical match.
        """

        tsq = self.qp.to_tsquery(query)
        if tsq is None:
            return []

        rank = func.ts_rank_cd(DocumentChunk.text_search, tsq, 32).label("rank")
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
        """Run the full retrieval pipeline for a single query.

        Steps, in order:

        1. Validate scope (if ``document_ids`` is given).
        2. Run embedding (network) and keyword search (DB) concurrently.
        3. Vector search.
        4. Fuse with RRF.
        5. Rerank (optional; failures fall back silently).
        6. MMR selection to ``top_k``.
        7. Neighbor expansion and context rendering.

        Timing for each stage is recorded in ``result.timings_ms``.

        Args:
            user_id: The authenticated user. Every query is scoped to
                this user's INDEXED documents.
            query: Raw query string. Whitespace is normalized; empty
                strings raise.
            top_k: Number of passages to return after MMR. Not
                currently bounded — pass a sensible value.
            document_ids: Optional allow-list. When provided, each ID
                must belong to ``user_id`` and be INDEXED, or
                ``_validate_scope`` raises.

        Returns:
            A ``RetrievalResult`` with the selected passages, the
            rendered context string, candidate counts per stage, whether
            reranking was applied, and per-stage timings in ms.

        Raises:
            NotFoundError: ``document_ids`` includes a document that
                doesn't exist or isn't owned by the user.
            DocumentNotReady: ``document_ids`` includes a document that
                exists but isn't INDEXED.
            EmbeddingUnavailable: The embedding call timed out or hit
                an I/O error.
            RuntimeError: The query embedding's dimension doesn't match
                the index dimension. This indicates the model was
                swapped without reindexing.
        """
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

        query = self.qp.normalize(query)

        # 1. Embedding (network) and keyword search (DB) run concurrently.
        #    Only ONE coroutine touches the AsyncSession, which is required:
        #    a single session must never run concurrent queries.
        with timed("embed+keyword"):
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
        """Embed a single query string with the configured prefix.

        Uses ``settings.embedding_spec.query_prefix`` — the query-side
        prefix, not the passage-side one. Getting these backwards
        degrades recall silently, which is why the spec carries both
        as separate fields rather than a single ``prefix``.

        Args:
            query: The normalized query string (whitespace collapsed,
                non-empty).

        Returns:
            A single embedding vector as a plain list of floats.

        Raises:
            EmbeddingUnavailable: The embedding call timed out or hit
                an I/O error.
            RuntimeError: The returned vector's length doesn't match
                ``settings.embedding_spec.dimension``. Indicates the
                model was swapped without reindexing.
        """
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
        """Rescore candidates with a cross-encoder.

        The reranker is an optional quality layer: any failure here —
        timeout, network, a broken response — is logged and swallowed,
        and the caller falls back to the fused ranking. This is
        deliberate. A dead reranker should degrade recall, not break
        retrieval.

        Args:
            query: The normalized query string.
            candidates: The candidates to rerank. Mutated in place:
                each candidate's ``rerank_score`` is set on success.

        Returns:
            True if the candidates were scored and their
            ``rerank_score`` fields set; False if the reranker is
            absent or the call failed.
        """
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

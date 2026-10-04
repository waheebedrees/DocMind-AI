"""Retrieval route.

Single endpoint: ``POST /retrieval`` takes a query and returns the top
passages plus a pre-rendered context string. The response is the full
contract for an LLM prompt — a client can send ``context`` straight to
a chat completion without assembling anything itself, or pick from
``passages`` if it wants to build its own prompt.
"""

from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.deps import CurrentUserId, DbSession
from app.core.exceptions import DocumentNotReady, EmbeddingUnavailable
from app.rag.retrieval import RetrievalService

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


class RetrieveRequest(BaseModel):
    """Input for a retrieval query.

    Attributes:
        query: Raw user query. Whitespace is normalized server-side.
            Length is bounded so a pathological client can't force the
            embedder to process a 10 MB string.
        top_k: Number of passages to return after MMR. Bounded at 30
            because context windows and rerank latency both grow
            linearly with this value, and 30 is already beyond what
            most chat models can usefully consume.
        document_ids: Optional allow-list of documents to search. When
            provided, every ID must belong to the caller and be
            INDEXED, or the request fails with 409. Capped at 100 to
            bound the scope-check query.
    """

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(8, ge=1, le=30)
    document_ids: list[UUID] | None = Field(None, max_length=100)


class PassageOut(BaseModel):
    """One retrieved passage as returned to the client.

    The ``citation_id`` is a stable, per-response identifier the client
    can use to reference a passage in a follow-up turn (e.g. "expand on
    source [2]"). It is not a database ID and is not stable across
    requests.

    Attributes:
        citation_id: 1-based index into ``RetrieveResponse.passages``.
            Matches the citation marker rendered into ``context``.
        document_id: The document this passage came from. The client
            can use this with ``GET /documents/{id}`` for metadata.
        page_start: First page of the passage, or ``None`` if the
            source document had no page structure (e.g. a plain text
            file).
        page_end: Last page, inclusive. Equal to ``page_start`` for a
            single-page passage. ``None`` under the same conditions as
            ``page_start``.
        section: Section heading from the source document, if the
            chunker recorded one. ``None`` for unstructured content.
        text: The passage text. May be longer than a single chunk when
            neighbor expansion merged adjacent chunks.
        score: Relevance score. The scale depends on whether reranking
            ran: without reranking this is the fused RRF score
            (unbounded, useful for ordering only); with reranking it's
            the cross-encoder's score (model-specific, calibrate on
            your eval set before using as a threshold).
    """

    citation_id: int
    document_id: UUID
    page_start: int | None
    page_end: int | None
    section: str | None
    text: str
    score: float


class RetrieveResponse(BaseModel):
    """Output of a retrieval query.

    Attributes:
        query: The normalized query string as it was processed
            (whitespace collapsed). Echoed so the client can log it
            alongside the response without reconstructing it.
        passages: The selected passages, in presentation order
            (relevance and MMR-diversified). Empty list if the query
            matched nothing.
        context: Pre-rendered context string with inline citation
            markers. Safe to pass directly into an LLM prompt; the
            markers reference ``citation_id`` in ``passages``.
        reranked: True if a cross-encoder scored the passages. False if
            reranking is disabled *or* if it was enabled but failed and
            the pipeline fell back to the fused ranking. False does not
            imply the configuration is wrong.
        timings_ms: Per-stage wall-clock durations. Keys vary by run —
            reranking stages only appear when reranking ran.
    """

    query: str
    passages: list[PassageOut]
    context: str
    reranked: bool
    timings_ms: dict[str, float]


@router.post("", response_model=RetrieveResponse)
async def retrieve(
    body: RetrieveRequest,
    user_id: CurrentUserId,
    session: DbSession,
):
    """Retrieve passages relevant to a query.

    Runs the hybrid retrieval pipeline (vector + keyword, fused with
    RRF, optionally reranked, MMR-diversified, neighbor-expanded) and
    returns both the passages and a ready-to-use context string.

    A ``RetrievalService`` is constructed per request so that settings
    are not shared across concurrent calls — the service holds an
    ``AsyncSession`` and is not concurrency-safe.

    Args:
        body: The query, ``top_k``, and optional document scope.
        user_id: The authenticated user. Every search is scoped to
            this user's INDEXED documents; no cross-tenant reads are
            possible.
        session: Database session for the retrieval service.

    Returns:
        A ``RetrieveResponse`` with passages, rendered context, and
        per-stage timings.

    Raises:
        HTTPException: 409 if ``document_ids`` includes a document that
            exists but is not yet INDEXED. Retrying after the pipeline
            finishes will succeed.
        HTTPException: 503 if the embedding backend is unavailable.
            Not retried server-side; the client should retry after a
            delay.
        HTTPException: 404 (via the app-level exception handler) if
            ``document_ids`` includes a document that doesn't exist or
            isn't owned by the caller. This is raised by
            ``_validate_scope`` and is *not* caught here — see *Known
            limitations*.
    """
    service = RetrievalService(session)
    try:
        r = await service.retrieve(
            user_id=user_id,
            query=body.query,
            top_k=body.top_k,
            document_ids=body.document_ids,
        )
    except DocumentNotReady as e:
        raise HTTPException(409, str(e)) from e
    except EmbeddingUnavailable as e:
        raise HTTPException(503, "embedding service unavailable") from e

    return RetrieveResponse(
        query=r.query,
        passages=[PassageOut(**{f: getattr(p, f) for f in PassageOut.model_fields}) for p in r.passages],
        context=r.context,
        reranked=r.reranked,
        timings_ms=r.timings_ms,
    )

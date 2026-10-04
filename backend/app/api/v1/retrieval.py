from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.deps import CurrentUserId, DbSession
from app.core.exceptions import DocumentNotReady, EmbeddingUnavailable
from app.rag.retrieval import RetrievalService

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(8, ge=1, le=30)
    document_ids: list[UUID] | None = Field(None, max_length=100)


class PassageOut(BaseModel):
    citation_id: int
    document_id: UUID
    page_start: int | None
    page_end: int | None
    section: str | None
    text: str
    score: float


class RetrieveResponse(BaseModel):
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
    service = RetrievalService(session)
    try:
        r = await service.retrieve(user_id=user_id, query=body.query, top_k=body.top_k, document_ids=body.document_ids)
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

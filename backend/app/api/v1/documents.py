from uuid import UUID

from fastapi import Request, UploadFile, status
from fastapi.routing import APIRouter

from app.core.auth import get_exception_400, get_exception_404
from app.core.config import settings
from app.core.deps import ArqPooleDep, CurrentUserId, DocumentServiceDep, StorageDep
from app.models.enums import _STAGE_TASK, JobStage
from app.schemas.document import DocumentResponse, DocumentStateResponse, ReprocessRequest
from app.services.storage import iter_upload

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
)
async def upload(
    request: Request, file: UploadFile, user_id: CurrentUserId, storage: StorageDep, document_service: DocumentServiceDep, arg_pool: ArqPooleDep
) -> DocumentResponse:
    if file.filename is None:
        raise get_exception_400("filename required")
    stored = await storage.put_stream(
        user_id=user_id,
        stream=iter_upload(file, chunk_size=settings.upload_chunk_bytes),
        filename=file.filename,
        max_bytes=settings.max_upload_bytes,
    )
    res = await document_service.get_or_create(user_id=user_id, stored=stored)
    request_id = request.headers.get("X-Request-ID")
    if res and res.current_stage:
        task = _STAGE_TASK[res.current_stage]
        if task is not None:
            await arg_pool.enqueue_job(function=task, job_id=str(res.job_id), _job_id=str(res.job_id), request_id=request_id)
    return res


@router.get(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
)
async def get_document_state(
    document_id: UUID,
    user_id: CurrentUserId,
    document_service: DocumentServiceDep,
) -> DocumentStateResponse:

    try:
        doc = await document_service.get_document_state(user_id=user_id, document_id=document_id)
        return doc
    except Exception as exec:
        raise get_exception_404("Document not found") from exec


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
)
async def delete(
    document_id: UUID,
    user_id: CurrentUserId,
    document_service: DocumentServiceDep,
    storage: StorageDep,
) -> bool:
    try:
        deleted = await document_service.delete_user_document(user_id, document_id=document_id, storage=storage)
        return deleted
    except Exception as exec:
        raise get_exception_404("Document not found") from exec


@router.post(
    "/{document_id}/reprocess",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
)
async def reprocess(
    request: Request,
    document_id: UUID,
    body: ReprocessRequest,
    user_id: CurrentUserId,
    document_service: DocumentServiceDep,
    storage: StorageDep,
    arg_pool: ArqPooleDep,
) -> DocumentResponse:
    res = await document_service.reprocess(
        user_id,
        document_id,
        from_stage=JobStage(body.from_stage),
        storage=storage,
    )

    request_id = request.headers.get("X-Request-ID")
    if res and res.current_stage:
        task = _STAGE_TASK[res.current_stage]
        if task is not None:
            await arg_pool.enqueue_job(function=task, job_id=str(res.job_id), _job_id=str(res.job_id), request_id=request_id)
    return res

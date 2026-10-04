"""Document management routes.

Endpoints under ``/documents``:

* ``POST   /upload``            — accept a file, persist it, start the
  ingestion pipeline.
* ``GET    /{document_id}``     — current processing state.
* ``DELETE /{document_id}``     — remove the document and its artifacts.
* ``POST   /{document_id}/reprocess`` — re-run the pipeline from a
  given stage.

Upload and reprocess both terminate by enqueueing the first ARQ task
for the job the service produced. The enqueue is fire-and-forget: the
HTTP response does not wait for the pipeline, and the client is
expected to poll ``GET /{document_id}`` for progress.
"""

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
    request: Request,
    file: UploadFile,
    user_id: CurrentUserId,
    storage: StorageDep,
    document_service: DocumentServiceDep,
    arg_pool: ArqPooleDep,
) -> DocumentResponse:
    """Accept a file upload and start the ingestion pipeline.

    The file is streamed directly to storage; the request body is
    never buffered in memory. ``settings.max_upload_bytes`` is enforced
    mid-stream, so an oversized upload is aborted partway rather than
    accepted and rejected.

    Deduplication is per-user: if the same content hash already exists
    for this user, the existing storage object is reused
    (``stored.deduplicated = True``) and no bytes are written. The
    pipeline still runs if the document is not yet indexed, so
    re-uploading a failed document retries it.

    On success, the first stage's ARQ task is enqueued and the response
    returns immediately. The client should poll
    ``GET /documents/{document_id}`` to observe progress.

    Args:
        request: Used only to read ``X-Request-ID`` for the enqueue,
            so the pipeline stage can log with the same correlation ID
            as the HTTP request.
        file: The uploaded file. ``file.filename`` must be non-None;
            an empty filename is rejected with 400.
        user_id: The authenticated user. Every document is keyed by
            this ID; cross-tenant uploads are impossible.
        storage: Storage backend. Accepts the byte stream and returns
            a ``StoredObject`` with the content hash and the canonical
            storage key.
        document_service: Owns the ``get_or_create`` logic — decides
            whether this content hash is a new document or a
            re-upload, and creates the first ``ProcessingJob`` row.
        arg_pool: ARQ connection pool. Used to enqueue the first
            pipeline task.

    Returns:
        A ``DocumentResponse`` describing the document and its current
        pipeline stage.

    Raises:
        HTTPException: 400 if the upload has no filename. Storage
            errors (``UploadTooLarge``, ``UnsupportedMime``) propagate
            as their own status codes via the app's exception handler.
    """
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
            await arg_pool.enqueue_job(
                function=task,
                job_id=str(res.job_id),
                _job_id=str(res.job_id),
                request_id=request_id,
            )
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
    """Return the current processing state of a document.

    Used by clients to poll after upload or reprocess. The response
    includes the document's status, its current stage (if any), the
    job ID of the running stage, and any accumulated error message.

    Args:
        document_id: The document to look up.
        user_id: The authenticated user. The lookup is scoped to this
            user; a document that exists but belongs to someone else
            is reported as 404, not 403.
        document_service: Owns the lookup and the state projection.

    Returns:
        A ``DocumentStateResponse`` describing the document's pipeline
        state.

    Raises:
        HTTPException: 404 if the document doesn't exist, isn't owned
            by ``user_id``, or any other error occurred inside the
            service. See *Known limitations* below.
    """
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
    """Delete a document and all of its artifacts.

    Removal covers three things: the document row, its processing job
    rows, and every artifact in storage keyed by its content hash
    (parsed document, cleaned document, chunks, embeddings). The delete
    is best-effort on the storage side — if a file is already missing,
    that's not an error.

    Args:
        document_id: The document to delete.
        user_id: The authenticated user. Scoped delete: a document
            belonging to another user is reported as 404.
        document_service: Owns the row and artifact cleanup.
        storage: Storage backend, passed through to the service so it
            can remove artifacts.

    Returns:
        True if a document was deleted. False is not currently
        returned; see *Known limitations*.

    Raises:
        HTTPException: 404 if the document doesn't exist or isn't
            owned by ``user_id``, or if the service raised for any
            other reason. See *Known limitations*.
    """
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
    """Re-run the ingestion pipeline from a caller-chosen stage.

    The service resets the document and creates a new job for
    ``from_stage``. Artifacts for stages *before* ``from_stage`` are
    kept; artifacts for ``from_stage`` and later are not pre-emptively
    deleted — the stages themselves overwrite their outputs on the
    next run, which is why ``_do_extract`` and ``_do_clean`` check for
    existing artifacts before redoing work.

    ``from_stage = "extract"`` is a full reindex. ``from_stage =
    "embed"`` is the common case after changing the embedding model —
    it reuses the existing chunks and only re-embeds.

    On success, the first stage's task is enqueued. The client should
    poll ``GET /documents/{document_id}`` for progress.

    Args:
        request: Used only to read ``X-Request-ID`` for the enqueue.
        document_id: The document to reprocess.
        body: ``{"from_stage": "chunk"}`` or equivalent. The value is
            parsed as a ``JobStage``; an unknown string yields a 422
            from FastAPI's body validation before this handler runs.
        user_id: The authenticated user. The reprocess is scoped.
        document_service: Owns the reset logic and the new job creation.
        storage: Storage backend, passed to the service for artifact
            inspection and cleanup.
        arg_pool: ARQ connection pool. Used to enqueue the first stage
            of the new run.

    Returns:
        A ``DocumentResponse`` for the document, with ``current_stage``
        set to the stage that was just enqueued.

    Raises:
        HTTPException: 422 if ``from_stage`` isn't a valid stage name
            (handled by FastAPI before this function runs).
        HTTPException: 404 if the document doesn't exist or isn't owned
            by ``user_id`` — raised by the service and mapped by the
            app's exception handler.
    """
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
            await arg_pool.enqueue_job(
                function=task,
                job_id=str(res.job_id),
                _job_id=str(res.job_id),
                request_id=request_id,
            )
    return res

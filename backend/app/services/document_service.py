import contextlib
from uuid import UUID

from app.core.exceptions import DocumentNotFound
from app.core.logging import get_logger
from app.db.repositories.chunks import ChunkRepository
from app.db.repositories.documents import DocumentRepository
from app.db.repositories.jobs import JobRepository
from app.models.enums import (
    _STAGE_ORDER,
    _STAGE_ORDER_VALUES,
    DocumentStatus,
    JobStage,
    JobStatus,
)
from app.schemas.document import DocumentResponse, DocumentStateResponse, StageStates
from app.services.storage import StoredObject
from app.services.storage.base import BaseStorage, ObjectNotFound
from app.services.storage.keys import all_artifact_keys, artifact_keys_from

log = get_logger(__name__)


def _current_stage(
    status: DocumentStatus,
    stages: list[StageStates],
) -> JobStage | None:
    # Only meaningful while a document is actively moving.
    if status in (DocumentStatus.FAILED):
        return None
    # First stage that's running, else first that's not done.
    for s in stages:
        if s.status == JobStatus.RUNNING:
            return s.stage
    for s in stages:
        if s.status != JobStatus.DONE:
            return s.stage
    return None


class DocumentService:
    def __init__(self, document_repo: DocumentRepository, job_repo: JobRepository, chunk_repo: ChunkRepository):
        self.document_repo = document_repo
        self.job_repo = job_repo
        self.chunk_repo = chunk_repo

    async def delete_user_document(
        self,
        user_id: UUID,
        document_id: UUID,
        storage: BaseStorage,
    ) -> bool:
        doc = await self.document_repo.get_for_user(
            user_id=user_id,
            document_id=document_id,
        )
        if doc is None:
            return False

        # Capture before the row is gone.
        storage_key = doc.storage_key
        doc_id = doc.id
        # 1 DB delete + commit. Cascades drop chunks and jobs.
        #    If this fails, we return False and storage is untouched.
        await self.document_repo.delete(doc_id)
        await self.document_repo.session.commit()

        log.info("document_deleted", document_id=str(doc_id), user_id=str(user_id))

        # 2 Storage cleanup — best-effort, never blocks the response.
        #    Failures are logged and swept up later by the orphan sweeper.
        if storage_key:
            try:
                await storage.delete(storage_key)
            except ObjectNotFound:
                pass
            except Exception as exc:  # noqa: BLE001
                log.warning("storage_delete_failed", key=storage_key, error=str(exc))

        if doc.content_hash:
            for key in all_artifact_keys(doc.content_hash):
                try:
                    await storage.delete_raw(key)
                except ObjectNotFound:
                    pass
                except Exception as exc:  # noqa: BLE001
                    log.warning("storage_delete_failed", key=key, error=str(exc))

        return True

    async def get_or_create(self, user_id: UUID, stored: StoredObject) -> DocumentResponse:

        existing = await self.document_repo.find_by_content_hash(user_id=user_id, content_hash=stored.content_hash)
        if existing is not None:
            # Surface the existing job so callers can see/poll it. Do NOT
            # create a new one — this document has already been (or is being)
            # processed.
            latest_job = await self.job_repo.latest_by_document(existing.id)
            if latest_job:
                return DocumentResponse(
                    job_id=latest_job.id,
                    document_id=existing.id,
                    mime_type=stored.mime_type,
                    current_stage=latest_job.stage,
                    size_bytes=stored.size_bytes,
                )
            raise
        doc = await self.document_repo.create(
            user_id=user_id,
            storage_key=stored.key,
            content_hash=stored.content_hash,
            mime_type=stored.mime_type,
            filename=stored.filename,
            size_bytes=stored.size_bytes,
            status=DocumentStatus.PENDING,
        )

        job = await self.job_repo.create_enqueue(document_id=doc.id)

        return DocumentResponse(
            document_id=doc.id,
            job_id=job.id,
            current_stage=job.stage,
            mime_type=stored.mime_type,
            size_bytes=stored.size_bytes,
        )

    async def get_document_state(
        self,
        user_id: UUID,
        document_id: UUID,
    ) -> DocumentStateResponse:
        document = await self.document_repo.get_for_user(user_id=user_id, document_id=document_id)
        if document is None:
            raise ValueError("document not found")
        latest = await self.job_repo.latest_by_stage(document.id)
        stages: list[StageStates] = []
        done_count = 0
        for stage in _STAGE_ORDER:
            job = latest.get(stage)
            if job is None:
                stages.append(
                    StageStates(
                        stage=stage,
                        status=JobStatus.QUEUED,
                        started_at=None,
                        finished_at=None,
                        duration_ms=None,
                    )
                )
                continue
            duration_ms = None
            if job.started_at and job.finished_at:
                duration_ms = int((job.finished_at - job.started_at).total_seconds() * 1000)
            if job.status == JobStatus.DONE:
                done_count += 1
            stages.append(
                StageStates(
                    stage=job.stage,
                    status=job.status,
                    started_at=job.started_at,
                    finished_at=job.finished_at,
                    duration_ms=duration_ms,
                )
            )

        return DocumentStateResponse(
            document_id=document.id,
            status=document.status,
            progress=round(done_count / len(_STAGE_ORDER), 2),
            stages=stages,
            current_stage=_current_stage(document.status, stages),
            error_message=document.error_message,
            chunk_count=document.chunk_count,
            page_count=document.page_count,
            indexed_at=document.indexed_at,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )

    async def reprocess(
        self,
        user_id: UUID,
        document_id: UUID,
        *,
        from_stage: JobStage,
        storage: BaseStorage,
    ) -> "DocumentResponse":
        """Force reprocessing from `from_stage` onward.

        Keeps upstream artifacts. Deletes DB chunks (if index will rerun) and
        resets document status, then enqueues the stage. If you bumped a
        version in settings, the cache key changed automatically and the
        stage will recompute; if you didn't, this still forces a recompute
        by deleting the current-key artifacts first.
        """
        doc = await self.document_repo.get_for_user(user_id, document_id)
        if doc is None:
            raise DocumentNotFound("document not found", code="invalid_document_id")

        if doc.content_hash is None:
            raise DocumentNotFound("document not found", code="invalid_document_id")
        stage_value = from_stage.value
        stage_idx = _STAGE_ORDER_VALUES.index(stage_value)

        # Delete cache for this stage and everything downstream (by current
        # fingerprint). No-op if the fingerprint already changed.
        for key in artifact_keys_from(doc.content_hash, stage_value):
            with contextlib.suppress(ObjectNotFound):
                await storage.delete_raw(key)

        # If index will rerun, drop the existing chunks so the doc isn't
        # serving stale results while reprocessing.
        if stage_idx <= _STAGE_ORDER_VALUES.index("index"):
            await self.chunk_repo.delete_for_document(document_id)
            await self.document_repo.set_status(document_id, DocumentStatus.PENDING)
            await self.document_repo.set_index_results(
                document_id,
                chunk_count=0,
                indexed_at=None,
                page_count=doc.page_count,
            )
            await self.document_repo.session.commit()

        job = await self.job_repo.create_enqueue(
            document_id=document_id,
            stage=from_stage,
        )
        await self.document_repo.session.commit()

        job = await self.job_repo.create_enqueue(document_id=doc.id)

        return DocumentResponse(
            document_id=doc.id,
            current_stage=job.stage,
            job_id=job.id,
            mime_type=doc.mime_type,
            size_bytes=doc.size_bytes,
        )

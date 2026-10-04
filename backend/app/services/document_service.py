"""Document service.

The orchestration layer between the document API and the persistence
layer. Owns:

* the upload dedup path (``get_or_create``),
* the delete path (rows + storage artifacts),
* the read model for the state endpoint (``get_document_state``),
* the reprocess path (artifact invalidation + job re-enqueue).

The service never touches the HTTP layer. It raises domain exceptions
(``DocumentNotFound``, ``ValueError``) and lets the router map them to
status codes. It commits its own transactions where the surrounding
operation spans multiple rows and must be atomic.
"""

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
    """Derive the document's active stage from its per-stage job states.

    Two-pass inference:

    1. If any stage is RUNNING, that's the current stage.
    2. Otherwise, the first stage that isn't DONE is the current stage
       — it's either queued or was never started.

    A FAILED document reports ``None``: it's not actively moving, and
    pointing at whichever stage failed would be misleading for a client
    that treats ``current_stage`` as "what's happening now." The client
    should look at ``error_message`` and the per-stage statuses instead.

    Args:
        status: The document's overall status.
        stages: Per-stage job states, in canonical order (see
            ``_STAGE_ORDER``). Order matters — the second loop returns
            the *first* non-DONE stage it finds.

    Returns:
        The current stage, or ``None`` if the document is FAILED or
        every stage is DONE (i.e. the pipeline finished).
    """
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
    """Orchestration for document lifecycle operations.

    Constructed per-request with three repositories bound to the same
    ``AsyncSession``. The service owns transaction boundaries for
    operations that span multiple tables — the repositories it wraps
    only ``flush``.

    Not thread-safe and not concurrent-safe against itself: a single
    instance must not be used from two coroutines at once, because its
    repositories share one ``AsyncSession``.
    """

    def __init__(
        self,
        document_repo: DocumentRepository,
        job_repo: JobRepository,
        chunk_repo: ChunkRepository,
    ):
        self.document_repo = document_repo
        self.job_repo = job_repo
        self.chunk_repo = chunk_repo

    async def delete_user_document(
        self,
        user_id: UUID,
        document_id: UUID,
        storage: BaseStorage,
    ) -> bool:
        """Delete a document, its rows, and its storage artifacts.

        Order is deliberate:

        1. Load the document scoped to ``user_id``. Return ``False`` if
           not found — the caller maps this to 404.
        2. Delete the document row and commit. Cascades drop the
           document's chunks and jobs. If this fails, storage is
           untouched and the operation is a clean no-op.
        3. Delete storage objects *after* the commit, best-effort. A
           failure here leaves orphaned files but not orphaned rows —
           the orphan sweeper cleans them up later. Never blocks the
           response.

        The commit in step 2 is intentional: a partial delete (rows
        gone, storage left) is recoverable; the reverse (storage gone,
        rows left) is not.

        Args:
            user_id: The owner. Lookup is scoped; a document belonging
                to another user is treated as not-found.
            document_id: The document to delete.
            storage: Storage backend. Receives a per-key delete for the
                source file and every pipeline artifact keyed by the
                document's content hash.

        Returns:
            True if a document was deleted, False if it didn't exist or
            isn't owned by ``user_id``.

        Note:
            Storage failures are logged at WARNING and swallowed. The
            return value reflects row deletion, not storage success.
        """
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

    async def get_or_create(
        self,
        user_id: UUID,
        stored: StoredObject,
    ) -> DocumentResponse:
        """Return the document for an upload, creating it if new.

        The dedup check is by ``(user_id, content_hash)``. On a repeat
        upload of the same content:

        * If the document already has a job, that job is returned. The
          caller sees the pipeline's *current* position, not the
          position it was at when the first upload happened.
        * If the document exists but has no jobs at all — a state that
          should not occur on a healthy system — this function raises a
          bare ``RuntimeError`` (see *Known limitations*).

        On a new upload, creates both the ``Document`` row (PENDING)
        and the first ``ProcessingJob`` (EXTRACT, QUEUED). The caller
        is responsible for enqueueing the ARQ task.

        Args:
            user_id: The authenticated user. Every document is keyed by
                this ID; cross-tenant dedup is impossible.
            stored: The ``StoredObject`` returned by
                ``BaseStorage.put_stream``. Supplies the content hash
                used for dedup and the storage key, MIME type, filename,
                and size for the new row.

        Returns:
            A ``DocumentResponse`` describing either the existing or
            the newly created document.

        Raises:
            RuntimeError: If the document exists but has no job. Should
                be unreachable; see *Known limitations*.
        """
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
        """Build the read model for the document-state endpoint.

        Projects the document row plus its per-stage job history into a
        flat response with:

        * overall status and progress (fraction of stages DONE),
        * one ``StageStates`` per canonical stage, in order,
        * the current stage (see ``_current_stage``),
        * the document's error message, if any.

        A stage that has never run appears with status QUEUED and null
        timestamps. A stage that ran and failed appears with status
        FAILED and its own error context (via the document's error
        message — per-stage errors are not surfaced here).

        Args:
            user_id: The authenticated user. Lookup is scoped; another
                user's document is treated as not-found.
            document_id: The document to describe.

        Returns:
            A ``DocumentStateResponse`` suitable for polling after
            upload or reprocess.

        Raises:
            ValueError: If the document doesn't exist or isn't owned by
                ``user_id``. The router maps this to 404 (see *Known
                limitations*).
        """
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
        """Force reprocessing from ``from_stage`` onward.

        Two goals, in tension:

        * **Invalidate caches** for ``from_stage`` and every stage after
          it, so the pipeline recomputes instead of reusing stale
          artifacts. If the cache key includes a version or model
          fingerprint, a bump changes the key and the delete is a no-op;
          if it doesn't, the delete is what forces recomputation.
        * **Preserve upstream artifacts.** Reproducing from ``chunk``
          reuses the existing extract and clean artifacts. The
          downstream stages overwrite their outputs on the next run.

        If the pipeline will reach INDEX (which it always does — INDEX
        is the final stage), existing chunks are dropped up front and
        the document is reset to PENDING with ``indexed_at`` cleared.
        This prevents serving stale results while the pipeline reruns.

        Commits are explicit: artifact deletion is a storage side
        effect, but the DB changes and the new job row are committed
        in one transaction before returning.

        Args:
            user_id: The authenticated user. Lookup is scoped.
            document_id: The document to reprocess.
            from_stage: The stage to restart at. Stages before it are
                skipped.
            storage: Storage backend, used to delete stale artifacts.

        Returns:
            A ``DocumentResponse`` describing the document and the
            newly created job.

        Raises:
            DocumentNotFound: The document doesn't exist, isn't owned by
                ``user_id``, or has no ``content_hash`` (a document that
                was never successfully uploaded). Uses code
                ``invalid_document_id``.
            ValueError: If ``from_stage`` isn't a valid stage — raised
                by ``_STAGE_ORDER_VALUES.index``.
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

"""Processing-job repository.

Owns persistence for ``ProcessingJob`` rows: the state machine
(queued → running → done/failed), per-document history, and the reads
the sweeper uses to detect stuck jobs. The pipeline's ``run_stage``
calls into this via ``Ingester``; the API's enqueue path calls
``create_enqueue`` directly.
"""

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import and_, exists, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.db.repositories.base import BaseRepository
from app.models.enums import JobStage, JobStatus
from app.models.processing_job import ProcessingJob


class JobRepository(BaseRepository[ProcessingJob]):
    """Persistence for ``ProcessingJob`` rows.

    No method commits; the caller owns the transaction. Write methods
    raise ``NotFoundError`` when the job is missing, except where
    documented otherwise.
    """

    def __init__(self, session: AsyncSession):
        super().__init__(ProcessingJob, session)

    async def create_enqueue(
        self,
        document_id: UUID,
        stage: JobStage = JobStage.EXTRACT,
        status: JobStatus = JobStatus.QUEUED,
    ) -> ProcessingJob:
        """Create a new job row and flush it so its ID is available.

        The job is created in QUEUED by default. The caller is
        responsible for enqueueing the corresponding ARQ task *after*
        committing — if the worker dies between commit and enqueue,
        this row exists with no ARQ task pointing at it, and
        reconciliation falls to the sweeper.

        Args:
            document_id: The document this job processes.
            stage: Which pipeline stage the job runs.
            status: Initial status. Defaults to QUEUED; pass RUNNING
                only when adopting a job that was already in flight.

        Returns:
            The new job, with ``id`` populated after the flush.
        """
        job = ProcessingJob(document_id=document_id, stage=stage, status=status, details={})
        self.session.add(job)
        await self.session.flush()
        return job

    async def mark_running(self, job_id: UUID) -> ProcessingJob:
        """Transition a job to RUNNING and record the start time.

        Idempotent: a job already in RUNNING is refreshed and returned
        without error, so an ARQ retry of the same attempt doesn't
        fail. ``started_at`` is *overwritten* on every call, which is
        fine for retries but means the column always reflects the most
        recent attempt, not the first.

        Args:
            job_id: The job to start.

        Returns:
            The job, now RUNNING.

        Raises:
            NotFoundError: The job doesn't exist, or its status can't
                transition to RUNNING (i.e. it's already DONE or
                FAILED).
        """
        job = await self.get_by_id(job_id)
        if job is None:
            raise NotFoundError(
                "Job Not found",
                code="invalid_job_id ",
            )
        if job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
            raise NotFoundError(
                f"Job {job_id} cannot transition to running from {job.status}",
                code="invalid_job_transition",
            )

        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(UTC)
        await self.session.flush()
        await self.session.refresh(job)
        return job

    async def mark_done(
        self,
        job_id: UUID,
        details: dict | None = None,
    ) -> ProcessingJob:
        """Transition a job to DONE and merge stage details.

        The job's existing ``details`` are preserved; keys from the
        argument overwrite matching keys. On a retry after a partial
        write, this is what keeps prior attempt history intact.

        Args:
            job_id: The job to complete.
            details: Stage output to merge into ``details``. ``None``
                or ``{}`` leaves ``details`` unchanged.

        Returns:
            The job, now DONE with ``finished_at`` set.

        Raises:
            NotFoundError: The job doesn't exist.
        """
        job = await self.get_by_id(job_id)
        if job is None:
            raise NotFoundError(
                "Job Not found",
                code="invalid_job_id ",
            )

        job.status = JobStatus.DONE
        job.finished_at = datetime.now(UTC)

        if details:
            job.details = {**(job.details or {}), **details}
        await self.session.flush()
        await self.session.refresh(job)
        return job

    async def mark_failed(
        self,
        job_id: UUID,
        error: str,
        details: dict | None = None,
    ) -> ProcessingJob:
        """Transition a job to FAILED and record the error.

        Writes ``details["error"] = error`` and merges ``details``
        *after* the error, so an ``error`` key in the argument dict
        overrides the error string. That's rarely what you want; pass
        additional context under a different key.

        Args:
            job_id: The job to fail.
            error: Human-readable failure description.
            details: Optional additional fields to merge into
                ``details``.

        Returns:
            The job, now FAILED with ``finished_at`` set.

        Raises:
            NotFoundError: The job doesn't exist.
        """
        job = await self.get_by_id(job_id)
        if job is None:
            raise NotFoundError(
                "Job Not found",
                code="invalid_job_id ",
            )
        job.status = JobStatus.FAILED
        job.finished_at = datetime.now(UTC)
        job.details = {**(job.details or {}), "error": error, **(details or {})}
        await self.session.flush()
        await self.session.refresh(job)
        return job

    async def list_for_document(
        self,
        document_id: UUID,
    ) -> list[ProcessingJob]:
        """List every job for a document, oldest first.

        Includes jobs in every status — DONE, FAILED, RUNNING, QUEUED.
        Useful for diagnostics; not used on the hot path.

        Args:
            document_id: The document whose jobs to list.

        Returns:
            All jobs for the document, oldest first.
        """
        rows = await self.session.scalars(select(ProcessingJob).where(ProcessingJob.document_id == document_id).order_by(ProcessingJob.created_at))
        return list(rows.all())

    async def latest_by_stage(
        self,
        document_id: UUID,
    ) -> dict[JobStage, ProcessingJob]:
        """Map each stage to its most recent job for a document.

        "Most recent" is by ``created_at``. A stage that has been
        re-run will have multiple rows; only the newest is returned.
        A stage that has never run is absent from the dict.

        Args:
            document_id: The document whose jobs to inspect.

        Returns:
            A dict keyed by ``JobStage``. Missing stages are absent —
            callers must handle ``KeyError`` or use ``.get``.
        """
        rows = await self.session.scalars(
            select(ProcessingJob).where(ProcessingJob.document_id == document_id).order_by(ProcessingJob.created_at.desc())
        )
        latest: dict[JobStage, ProcessingJob] = {}
        for job in rows.all():
            latest.setdefault(job.stage, job)
        return latest

    async def has_active_job(self, document_id: UUID) -> bool:
        """True if the document has any QUEUED or RUNNING job.

        Used by the API to decide whether an upload should enqueue a
        new pipeline run or attach to the one already in flight.

        Args:
            document_id: The document to check.

        Returns:
            True if at least one job for the document is QUEUED or
            RUNNING; False otherwise.
        """
        stmt = select(
            exists().where(
                ProcessingJob.document_id == document_id,
                ProcessingJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
            )
        )
        return bool(await self.session.scalar(stmt))

    async def record_attempt(
        self,
        job_id: UUID,
        *,
        error: str,
        attempt: int,
    ) -> None:
        """Append a transient failure to the job's attempt history.

        Called by ``run_stage`` before raising ``Retry``. Does not
        change status — the job stays RUNNING so ARQ's retry sees it in
        the state it left it.

        The history is a list under ``details["attempt"]``. Each entry
        is ``{"attempt": int, "error": str}``.

        Args:
            job_id: The job being retried.
            error: Failure description.
            attempt: 1-based attempt number that just failed.

        Raises:
            NotFoundError: The job doesn't exist.
            AttributeError: The job's ``details`` is ``None``. See
                *Known limitations*.
        """
        job = await self.get_by_id(job_id)
        if job is None:
            raise NotFoundError(
                "Job Not found",
                code="invalid_job_id ",
            )
        history = list(job.details.get("attempt", []))
        history.append({"attempt": attempt, "error": error})
        job.details = {**job.details, "attempt": history}
        await self.session.flush()

    async def list_queued(
        self,
        *,
        cutoff: datetime,
        status: JobStatus,
    ) -> list[ProcessingJob]:
        """List jobs in a given status older than ``cutoff``.

        Despite the name, the ``status`` parameter means this is a
        generic "stale by status" query. The sweeper uses it with
        ``status=QUEUED`` to find jobs that were created but never
        picked up.

        Args:
            cutoff: Jobs created before this timestamp are returned.
            status: The status to filter on.

        Returns:
            Matching jobs, oldest first.
        """
        result = await self.session.execute(
            select(ProcessingJob)
            .where(
                ProcessingJob.status == status,
                ProcessingJob.created_at < cutoff,
            )
            .order_by(ProcessingJob.created_at)
        )
        return list(result.scalars().all())

    async def last_completed(
        self,
        document_id: UUID,
    ) -> ProcessingJob | None:
        """Most recent DONE job for a document.

        Ordered by ``finished_at``, so this reflects *when the job
        completed*, not when it was created. If two jobs finished in
        the same second (unlikely but possible), the ordering is
        unspecified.

        Args:
            document_id: The document to inspect.

        Returns:
            The most recently finished DONE job, or ``None`` if the
            document has no DONE jobs.
        """
        stmt = (
            select(ProcessingJob)
            .where(
                ProcessingJob.status == JobStatus.DONE,
                ProcessingJob.document_id == document_id,
            )
            .order_by(ProcessingJob.finished_at.desc())
            .limit(1)
        )
        return await self.session.scalar(stmt)

    async def list_stuck(
        self,
        *,
        queued_before: datetime,
        running_before: datetime,
    ) -> list[ProcessingJob]:
        """List jobs whose status has been held past the expected time.

        Two cases, combined with OR:

        * QUEUED jobs created before ``queued_before`` — the enqueue
          step was lost (worker crash, Redis blip).
        * RUNNING jobs whose ``started_at`` is before
          ``running_before`` — the worker died mid-stage.

        Args:
            queued_before: Cutoff for QUEUED jobs, by ``created_at``.
            running_before: Cutoff for RUNNING jobs, by ``started_at``.

        Returns:
            Stuck jobs of either kind. Ordering is unspecified.
        """
        stmt = select(ProcessingJob).where(
            or_(
                and_(
                    ProcessingJob.status == JobStatus.QUEUED,
                    ProcessingJob.created_at < queued_before,
                ),
                and_(
                    ProcessingJob.status == JobStatus.RUNNING,
                    ProcessingJob.started_at < running_before,
                ),
            )
        )
        rows = await self.session.scalars(stmt)
        return list(rows.all())

    async def latest_by_document(
        self,
        document_id: UUID,
    ) -> ProcessingJob | None:
        """Most recent job for a document, regardless of stage or status.

        Used by the dedup path in ``DocumentService.get_or_create``:
        when an upload hits an existing ``Document`` row, the caller
        wants the job that is (or was) actually processing that
        document, not ``None``.

        Args:
            document_id: The document to inspect.

        Returns:
            The newest job, or ``None`` if the document has never had
            one.
        """
        stmt = select(ProcessingJob).where(ProcessingJob.document_id == document_id).order_by(ProcessingJob.created_at.desc()).limit(1)
        return await self.session.scalar(stmt)

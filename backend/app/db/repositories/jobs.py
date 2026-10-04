from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import and_, exists, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.db.repositories.base import BaseRepository
from app.models.enums import JobStage, JobStatus
from app.models.processing_job import ProcessingJob


class JobRepository(BaseRepository[ProcessingJob]):
    def __init__(self, session: AsyncSession):
        super().__init__(ProcessingJob, session)

    async def create_enqueue(
        self,
        document_id: UUID,
        stage: JobStage = JobStage.EXTRACT,
        status: JobStatus = JobStatus.QUEUED,
    ) -> ProcessingJob:

        job = ProcessingJob(document_id=document_id, stage=stage, status=status, details={})
        self.session.add(job)
        await self.session.flush()

        return job

    async def mark_running(
        self,
        job_id: UUID,
    ) -> ProcessingJob:

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

    async def mark_done(self, job_id: UUID, details: dict | None = None) -> ProcessingJob:
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

    async def mark_failed(self, job_id: UUID, error: str, details: dict | None = None) -> ProcessingJob:
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
        rows = await self.session.scalars(select(ProcessingJob).where(ProcessingJob.document_id == document_id).order_by(ProcessingJob.created_at))
        return list(rows.all())

    async def latest_by_stage(self, document_id: UUID) -> dict[JobStage, ProcessingJob]:
        rows = await self.session.scalars(
            select(ProcessingJob).where(ProcessingJob.document_id == document_id).order_by(ProcessingJob.created_at.desc())
        )
        latest: dict[JobStage, ProcessingJob] = {}
        for job in rows.all():
            latest.setdefault(job.stage, job)
        return latest

    async def has_active_job(self, document_id: UUID) -> bool:
        """True if the document has any QUEUED or RUNNING job."""
        stmt = select(
            exists().where(
                ProcessingJob.document_id == document_id,
                ProcessingJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
            )
        )
        return bool(await self.session.scalar(stmt))

    async def record_attempt(self, job_id: UUID, *, error: str, attempt: int) -> None:
        """
        Append  a Transient error with out changing status
        used when ARQ retries Terminal failures go through make_failed
        Args:
            job_id (UUID): job UUid
            error (str): error
            attempt (int): number of attempt
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
        """Jobs in the given status older than cutoff."""
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
        """Most recent DONE job for a document, or None."""
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
        """Jobs whose current status has been held past the expected time.

        QUEUED jobs older than ``queued_before`` were created but never picked
        up — the enqueue step was lost (worker crash, Redis blip).

        RUNNING jobs whose ``started_at`` is older than ``running_before``
        were picked up but never finished — the worker died mid-stage.
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

    async def latest_by_document(self, document_id: UUID) -> ProcessingJob | None:
        """Most recent job for a document, regardless of stage or status.

        Used by the dedup path in DocumentService.get_or_create: when an
        upload hits an existing Document row, we want to hand the caller the
        job that is (or was) actually processing that document, not None.
        """
        stmt = select(ProcessingJob).where(ProcessingJob.document_id == document_id).order_by(ProcessingJob.created_at.desc()).limit(1)
        return await self.session.scalar(stmt)

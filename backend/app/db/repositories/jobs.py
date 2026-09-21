from sqlalchemy import exists, select
from uuid import UUID, uuid4
from typing import Optional
from sqlalchemy import select, func, insert, delete, Result
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.base import BaseRepository
from app.models.processing_job import ProcessingJob
from app.models.enums import JobStage, JobStatus
from app.core.exceptions import NotFoundError


class JobRepository(BaseRepository[ProcessingJob]):
    def __init__(self, session: AsyncSession):
        super().__init__(ProcessingJob, session)

    async def create_enqueue(
        self,
        document_id: UUID,
        stage: JobStage,
        status: JobStatus = JobStatus.QUEUED,
    ) -> ProcessingJob:

        job = ProcessingJob(document_id=document_id, stage=stage, status=status, details={})
        self.session.add(job)
        await self.session.flush()

        return job

    async def make_running(
        self,
        job_id: UUID,
    ) -> ProcessingJob:

        job = await self.get_by_id(job_id)
        if job is None:
            raise NotFoundError(
                f"Job Not found",
                code="invalid_job_id ",
            )
        if job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
            raise NotFoundError(
                f"Job {job_id} cannot transition to running from {job.status}",
                code="invalid_job_transition",
            )

        job.status = JobStatus.RUNNING
        job.started_at = func.now()
        await self.session.flush()
        await self.session.refresh(job)
        return job

    async def make_done(self, job_id: UUID, details: dict | None = None) -> ProcessingJob:
        job = await self.get_by_id(job_id)
        if job is None:
            raise NotFoundError(
                f"Job Not found",
                code="invalid_job_id ",
            )
        
        job.status = JobStatus.DONE
        job.finished_at = func.now()
        
        if details:
            job.details = {**(job.details or {}), **details}
        await self.session.flush()
        await self.session.refresh(job)
        return job

    async def make_failed(self, job_id: UUID, error: str, details: dict | None = None) -> ProcessingJob:
        job = await self.get_by_id(job_id)
        if job is None:
            raise NotFoundError(
                f"Job Not found",
                code="invalid_job_id ",
            )
        job.status = JobStatus.FAILED
        job.finished_at = func.now()
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


        stmt = select(
            exists().where(
                ProcessingJob.document_id == document_id,
                ProcessingJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
            )
        )
        return bool(await self.session.scalar(stmt))

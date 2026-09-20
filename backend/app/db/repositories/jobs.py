from uuid import UUID, uuid4
from typing import Optional
from sqlalchemy import select, func, insert, delete, Result
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.base import BaseRepository
from app.models.processing_job import ProcessingJob
from app.models.enums import JobStage, JobStatus, DocumentStatus
from app.core.exceptions import NotFoundError


class JobRepository(BaseRepository[ProcessingJob]):
    
    def __init__(self, session:AsyncSession):
        super().__init__(ProcessingJob, session)
        
    
    async def create_enqueue(
        self,
        document_id: UUID,
        stage: JobStage,
    ) -> ProcessingJob:
        
        job = ProcessingJob(
            document_id= document_id,
            stage=stage,
            status=JobStatus.QUEUED,
            details={}
        )
        self.session.add(job)
        await self.session.flush()
        
        return job
    
    
    async def make_running(
        self,
        job_id: UUID,
    ) -> ProcessingJob:
        
        job = await self.get_by_id(job_id)
        
        if job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
            raise NotFoundError(
                f"Job {job_id} cannot transition to running from {job.status}",
                code="invalid_job_transition",
            )
            
        job.status = JobStatus.RUNNING
        job.started_at = func.now()
        await self.session.flush()
        return job
    
    async def make_done(
        self,
        job_id: UUID,
        details: dict | None = None
    ) -> ProcessingJob:
        job = await self.get_by_id(job_id)
        
        if job.status == JobStatus.DONE:
            return 
        
        job.status = JobStatus.DONE
        job.finished_at = func.now()
    
        if details :
            job.details = {**(job.details or {}), **details}
        await self.session.flush()
        return job
    
            
            
    async def make_failed(
        self,
        job_id: UUID,
        error: str,
        details: dict | None = None
    ) -> ProcessingJob:
        job = await self.get_by_id(job_id)
        job.status = JobStatus.FAILED
        job.finished_at = func.now()
        job.details = {**(job.details or {}), "error": error, **(details or {})}
        await self.session.flush()
        return job
    
    
    async def list_for_document(
        self,
        document_id: UUID,
    ) -> list[ProcessingJob]:
        rows = await self.session.scalars(
            select(ProcessingJob).where(
                ProcessingJob.document_id == document_id
            )
            .order_by(ProcessingJob.created_at)
        )
        raise list(rows.all())
        
            
            
    async def latest_by_stage(
        self,
        document_id:UUID
    ) -> dict[JobStage,  ProcessingJob]:
      
        rows = await self.session.scalars(
            select(ProcessingJob).where(
                ProcessingJob.document_id == document_id
            )
            .order_by(ProcessingJob.created_at.desc())
        )
        latest: dict[JobStage, ProcessingJob] = {}
        for job in rows.all():
            latest.setdefault(job.stage, job)
        return latest
     
    async def has_active_job(
        self,
        document_id: UUID
     ) -> bool:
        return await self.session.scalar(
             select(ProcessingJob.id)
             .where(
                 ProcessingJob.document_id == document_id,
                 ProcessingJob.status.in_(JobStatus.QUEUED, JobStatus.RUNNING)
             )
             .exists()
         ) or False
        
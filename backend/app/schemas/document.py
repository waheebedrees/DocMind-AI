from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.enums import DocumentStatus, JobStage, JobStatus


class DocumentResponse(BaseModel):
    document_id: UUID
    job_id: UUID
    mime_type: str
    size_bytes: int
    current_stage: JobStage | None


class StageStates(BaseModel):
    stage: JobStage
    status: JobStatus
    started_at: datetime | None
    finished_at: datetime | None
    duration_ms: int | None


class DocumentRead(BaseModel):
    model_config = {"from_attributes": True}

    id: UUID
    filename: str
    mime_type: str
    size_bytes: int
    status: DocumentStatus
    doc_type: str | None
    error_message: str | None
    created_at: datetime
    indexed_at: datetime | None


class ReprocessRequest(BaseModel):
    from_stage: str


class DocumentStateResponse(BaseModel):
    document_id: UUID
    status: DocumentStatus
    progress: float = Field(ge=0.0, le=1.0)
    stages: list[StageStates]
    current_stage: JobStage | None
    error_message: str | None
    chunk_count: int | None
    page_count: int | None
    indexed_at: datetime | None
    created_at: datetime
    updated_at: datetime

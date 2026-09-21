from app.models.processing_job import ProcessingJob
from app.models.enums import JobStage, JobStatus
from unittest.mock import AsyncMock, MagicMock

import pytest
from app.db.repositories.user_repo import UserRepository
import uuid
from datetime import datetime, timedelta, timezone


from app.models.document import Document
from app.models.enums import DocumentStatus
from app.models.user import User

from app.db.repositories.documents import DocumentRepository


@pytest.fixture
def mock_user_repository():
    return MagicMock(spec=UserRepository)


@pytest.fixture
def pdf_bytes() -> bytes:
    # Minimal valid PDF — libmagic detects it as application/pdf.
    return b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n" + b"x" * 2048


@pytest.fixture
def zip_bytes() -> bytes:
    # ZIP magic bytes — libmagic detects it, but it is not in ALLOWED_MIMES.
    return b"PK\x03\x04" + b"\x00" * 512


def make_user(
    email: str | None = None,
    password_hash: str = "x",
) -> User:
    uid = uuid.uuid4().hex[:8]
    return User(email=email or f"user-{uid}@example.com", password_hash=password_hash)


def make_document(
    user_id: uuid.UUID,
    filename: str = "doc.pdf",
    mime_type: str = "application/pdf",
    size_bytes: int = 1024,
    storage_key: str = "s3://bucket/key",
    content_hash: str = "deadbeef",
    status: DocumentStatus = DocumentStatus.PENDING,
    created_at: datetime | None = None,
) -> Document:

    doc = Document(
        user_id=user_id,
        filename=filename,
        size_bytes=size_bytes,
        storage_key=storage_key,
        content_hash=content_hash,
        status=status,
        mime_type=mime_type,
    )
    if created_at is not None:
        doc.created_at = created_at
    return doc


def make_job(
    document_id,
    *,
    stage: JobStage = JobStage.EXTRACT,
    status: JobStatus = JobStatus.QUEUED,
    details: dict | None = None,
) -> ProcessingJob:
    return ProcessingJob(
        document_id=document_id,
        stage=stage,
        status=status,
        details=details if details is not None else {},
    )

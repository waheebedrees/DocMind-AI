import itertools
from app.core.config import settings
import uuid
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from app.db.repositories.user_repo import UserRepository
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.enums import DocumentStatus, JobStage, JobStatus
from app.models.processing_job import ProcessingJob
from app.models.user import User


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


def make_chunk(
    document_id: uuid.UUID,
    *,
    chunk_index: int = 0,
    text: str = "chunk text",
    token_count: int = 3,
    embedding: list[float] | None = None,
    metadata: dict | None = None,
) -> DocumentChunk:
    if embedding is None:
        embedding = [0.0] * settings.embedding_dim
    return DocumentChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        token_count=token_count,
        embedding=embedding,
        metadata_=metadata if metadata is not None else {},
    )

@pytest.fixture
async def other_user(session):
    u = make_user(email="other@example.com")
    session.add(u)
    await session.flush()
    return u


@pytest.fixture
async def user(session):
    u = make_user()
    session.add(u)
    await session.flush()
    return u


@pytest.fixture
async def document(session, user):
    doc = make_document(user_id=user.id)
    session.add(doc)
    await session.flush()
    return doc


@pytest.fixture
async def other_document(session, user):
    doc = make_document(user_id=user.id, filename="other.pdf",
                        content_hash="cafebabe")
    session.add(doc)
    await session.flush()
    return doc


@pytest.fixture
async def chunk(session, document):
    c = make_chunk(document_id=document.id)
    session.add(c)
    await session.flush()
    return c


@pytest.fixture
def chunk_factory(session, document):
    """Factory: each call creates a distinct DocumentChunk on `document`.

    Assigns a fresh chunk_index per call so tests needing N citations
    don't trip the unique (document_id, chunk_index) constraint.
    """
    counter = itertools.count()

    async def _make():
        c = make_chunk(document_id=document.id, chunk_index=next(counter))
        session.add(c)
        await session.flush()
        return c

    return _make

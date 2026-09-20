import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.enums import DocumentStatus
from app.models.user import User

from app.db.repositories.documents import DocumentRepository
pytestmark = pytest.mark.unit


def make_user(
    email: str | None = None,
    password_hash: str = "x",
) -> User:
    uid = uuid.uuid4().hex[:8]
    return User(
        email=email or f"user-{uid}@example.com",
        password_hash=password_hash
    )

def make_document(
    user_id:uuid.UUID,
    filename: str = 'doc.pdf',
    mime_type: str = 'application/pdf',
    size_bytes: int = 1024,
    storage_key: str = "s3://bucket/key",
    content_hash: str = 'deadbeef',
    status: DocumentStatus = DocumentStatus.PENDING,
    created_at: datetime | None = None 
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




async def _persist(db:AsyncSession, *objs):
    for o in objs:
        db.add(o)
    await db.flush()
    
    for o in objs:
        await  db.refresh(o)
    return objs if len(objs) > 1 else objs[0]




class TestCreate:
    async def test_create_persists_all_fields(self, db: AsyncSession):
        user = await _persist(db, make_user())
        repo = DocumentRepository(db)
        
        doc = await repo.create(
            user_id=user.id,
            filename='report.pdf',
            mime_type='application/pdf',
            size_bytes=2048,
            storage_key="s3://docs/report.pdf",
            content_hash="abc123",
        )

        assert doc.id is not None
        assert doc.user_id == user.id
        assert doc.filename == "report.pdf"
        assert doc.mime_type == "application/pdf"
        assert doc.size_bytes == 2048
        assert doc.storage_key == "s3://docs/report.pdf"
        assert doc.content_hash == "abc123"
        assert doc.status == DocumentStatus.PENDING

    
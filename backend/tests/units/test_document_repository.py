import uuid
from datetime import UTC, datetime, timedelta

import pytest
from app.db.repositories.documents import DocumentRepository
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.models.user import User
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.exceptions import NotFoundError

pytestmark = pytest.mark.unit


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


async def _persist(db: AsyncSession, *objs):
    for o in objs:
        db.add(o)
    await db.flush()

    for o in objs:
        await db.refresh(o)
    return objs if len(objs) > 1 else objs[0]


class TestCreate:
    async def test_create_persists_all_fields(self, db: AsyncSession):
        user = await _persist(db, make_user())
        repo = DocumentRepository(db)

        doc = await repo.create(
            user_id=user.id,
            filename="report.pdf",
            mime_type="application/pdf",
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

    async def test_create_is_visible_after_rollback_save_flush(self, db: AsyncSession):

        user = await _persist(db, make_user())

        repo = DocumentRepository(db)
        doc = await repo.create(
            user_id=user.id,
            filename="a.pdf",
            mime_type="application/pdf",
            size_bytes=1,
            storage_key="k",
            content_hash="h",
        )

        found = await repo.get_for_user(doc.id, user.id)
        assert found is not None
        assert found.id == doc.id


class TestGetForUser:
    async def test_returns_document_when_owner_matches(self, db: AsyncSession):
        user = await _persist(db, make_user())
        doc = await _persist(db, make_document(user_id=user.id, filename="mine.pdf"))

        repo = DocumentRepository(db)

        result = await repo.get_for_user(document_id=doc.id, user_id=user.id)

        assert result is not None
        assert result.id == doc.id
        assert result.filename == "mine.pdf"

    async def test_return_none_when_wrong_user(self, db: AsyncSession):

        owner = await _persist(db, make_user())
        other = await _persist(db, make_user())
        doc = await _persist(db, make_document(user_id=owner.id))
        repo = DocumentRepository(db)

        result = await repo.get_for_user(doc.id, other.id)

        assert result is None

    async def test_return_none_when_document_missing(self, db: AsyncSession):
        user = await _persist(db, make_user())
        repo = DocumentRepository(db)
        result = await repo.get_for_user(uuid.uuid4(), user.id)
        assert result is None


class TestFindByContentHash:
    async def test_finds_document_by_content_hash(self, db: AsyncSession):
        user = await _persist(db, make_user())
        doc = await _persist(db, make_document(user_id=user.id, content_hash="xyz123456789"))

        repo = DocumentRepository(db)

        result = await repo.find_by_content_hash(user_id=user.id, content_hash="xyz123456789")
        assert result is not None

        assert result.id == doc.id

    async def test_returns_none_when_hash_not_seen(self, db: AsyncSession):
        user = await _persist(db, make_user())
        repo = DocumentRepository(db)
        result = await repo.find_by_content_hash(user_id=user.id, content_hash="not seen")

        assert result is None

    async def test_does_not_leak_across_users(self, db: AsyncSession):

        u1 = await _persist(db, make_user())
        u2 = await _persist(db, make_user())
        d1 = await _persist(db, make_document(user_id=u1.id, content_hash="same-hash"))
        d2 = await _persist(db, make_document(user_id=u2.id, content_hash="same-hash"))

        repo = DocumentRepository(db)
        assert (await repo.find_by_content_hash(u1.id, "same-hash")).id == d1.id
        assert (await repo.find_by_content_hash(u2.id, "same-hash")).id == d2.id


class TestListForUser:
    async def test_returns_all_documents_newest_first(self, db: AsyncSession):
        user = await _persist(db, make_user())
        base = datetime(2026, 1, 1, tzinfo=UTC)
        older = await _persist(
            db,
            make_document(
                user_id=user.id,
                filename="old.pdf",
                created_at=base,
            ),
        )
        newest = await _persist(db, make_document(user_id=user.id, filename="new.pdf", created_at=base + timedelta(hours=1)))

        repo = DocumentRepository(db)
        docs, total = await repo.list_for_user(user.id)
        assert total == 2
        assert [d.id for d in docs] == [newest.id, older.id]

    async def test_status_filter_is_applied(self, db: AsyncSession):
        user = await _persist(db, make_user())

        pending = await _persist(
            db,
            make_document(
                user.id,
                status=DocumentStatus.PENDING,
            ),
        )
        await _persist(
            db,
            make_document(
                user.id,
                status=DocumentStatus.INDEXED,
                content_hash="x",
            ),
        )
        repo = DocumentRepository(db)

        docs, total = await repo.list_for_user(user.id, status=DocumentStatus.PENDING)
        assert total == 1
        assert [d.id for d in docs] == [pending.id]

    async def test_pagination_limit_and_offset(self, db: AsyncSession):
        user = await _persist(db, make_user())
        base = datetime(2026, 1, 1, tzinfo=UTC)
        for i in range(5):
            await _persist(db, make_document(user.id, filename=f"f{i}.pdf", content_hash=f"h{i}", created_at=base + timedelta(minutes=i)))

        repo = DocumentRepository(db)
        page1, total = await repo.list_for_user(user.id, limit=2, offset=0)
        page2, _ = await repo.list_for_user(user.id, limit=2, offset=2)
        page3, _ = await repo.list_for_user(user.id, limit=2, offset=4)

        assert total == 5
        assert [d.filename for d in page1] == ["f4.pdf", "f3.pdf"]
        assert [d.filename for d in page2] == ["f2.pdf", "f1.pdf"]
        assert [d.filename for d in page3] == ["f0.pdf"]

    async def test_empty_for_user_with_no_documents(self, db: AsyncSession):
        user = await _persist(db, make_user())

        repo = DocumentRepository(db)
        docs, total = await repo.list_for_user(user.id)

        assert docs == []
        assert total == 0

    async def test_does_not_return_other_users_documents(self, db: AsyncSession):
        u1 = await _persist(db, make_user())
        u2 = await _persist(db, make_user())

        await _persist(db, make_document(u1.id, content_hash="a"))
        await _persist(db, make_document(u2.id, content_hash="b"))

        repo = DocumentRepository(db)
        docs, total = await repo.list_for_user(u1.id)

        assert docs[0].user_id == u1.id
        assert total == 1


class TestSetstatus:
    async def test_transitions(self, db: AsyncSession):
        user = await _persist(db, make_user())

        doc = await _persist(db, make_document(user.id))

        repo = DocumentRepository(db)

        await repo.set_status(doc.id, DocumentStatus.PROCESSING)

        assert doc.status == DocumentStatus.PROCESSING
        assert doc.error_message is None

    async def test_stores_error_message_on_failure(self, db: AsyncSession):
        user = await _persist(db, make_user())

        doc = await _persist(db, make_document(user.id))
        repo = DocumentRepository(db)
        await repo.set_status(doc.id, DocumentStatus.FAILED, error_messages="boom")

        assert doc.status == DocumentStatus.FAILED
        assert doc.error_message == "boom"

    async def test_indexed_sets_indexed_at(self, db: AsyncSession):
        user = await _persist(db, make_user())
        doc = await _persist(db, make_document(user.id))
        repo = DocumentRepository(db)

        await repo.set_status(doc.id, DocumentStatus.INDEXED)

        assert doc.status == DocumentStatus.INDEXED
        assert doc.indexed_at is not None
        assert isinstance(doc.indexed_at, datetime)

    async def test_missing_document_raises(self, db: AsyncSession):
        repo = DocumentRepository(db)

        with pytest.raises(NotFoundError):
            await repo.set_status(uuid.uuid4(), DocumentStatus.PROCESSING)


class TestSetParsedKey:
    async def test_updates_parsed_key(self, db: AsyncSession):
        user = await _persist(db, make_user())
        doc = await _persist(db, make_document(user.id))
        repo = DocumentRepository(db)

        await repo.set_parsed_key(doc.id, "s3://parsed/doc.json")

        assert doc.parsed_key == "s3://parsed/doc.json"


class TestAggregates:
    async def test_count_for_user(self, db: AsyncSession):
        user = await _persist(db, make_user())
        for i in range(3):
            await _persist(db, make_document(user.id, content_hash=f"h{i}"))

        repo = DocumentRepository(db)
        assert await repo.count_for_user(user.id) == 3

    async def test_count_for_user_is_zero_when_empty(self, db: AsyncSession):
        user = await _persist(db, make_user())
        repo = DocumentRepository(db)
        assert await repo.count_for_user(user.id) == 0

    async def test_total_bytes_sums_size(self, db: AsyncSession):
        user = await _persist(db, make_user())
        for i, size in enumerate([100, 200, 300]):
            await _persist(db, make_document(user.id, size_bytes=size, content_hash=f"h{i}"))

        repo = DocumentRepository(db)
        assert await repo.total_bytes_for_user(user.id) == 600

    async def test_total_bytes_is_zero_when_empty(self, db: AsyncSession):
        user = await _persist(db, make_user())
        repo = DocumentRepository(db)

        assert await repo.total_bytes_for_user(user.id) == 0

    async def test_aggregates_do_not_cross_user(self, db: AsyncSession):
        u1 = await _persist(db, make_user())
        u2 = await _persist(db, make_user())
        await _persist(db, make_document(u1.id, size_bytes=500, content_hash="a"))
        await _persist(db, make_document(u2.id, size_bytes=1000, content_hash="b"))

        repo = DocumentRepository(db)
        assert await repo.count_for_user(u1.id) == 1
        assert await repo.total_bytes_for_user(u1.id) == 500


class TestFindPendingWithJob:
    async def test_returns_old_pending_with_no_job(self, db: AsyncSession):
        user = await _persist(db, make_user())
        stale = datetime.now(UTC) - timedelta(seconds=30)
        doc = await _persist(db, make_document(user.id, status=DocumentStatus.PENDING, created_at=stale))
        repo = DocumentRepository(db)

        assert [d.id for d in await repo.find_pending_without_job(user.id, older_than_seconds=10)] == [doc.id]

    async def test_skip_recent_pending(self, db: AsyncSession):
        user = await _persist(db, make_user())
        await _persist(
            db,
            make_document(
                user.id,
                status=DocumentStatus.PENDING,
                created_at=datetime.now(UTC),
            ),
        )
        repo = DocumentRepository(db)
        result = await repo.find_pending_without_job(user.id, older_than_seconds=3600)
        assert result == []

    async def test_skip_non_pending(self, db: AsyncSession):
        user = await _persist(db, make_user())
        stale = datetime.now(UTC) - timedelta(seconds=600)
        await _persist(
            db,
            make_document(
                user.id,
                status=DocumentStatus.INDEXED,
                created_at=stale,
            ),
        )
        repo = DocumentRepository(db)
        result = await repo.find_pending_without_job(user.id, older_than_seconds=3600)
        assert result == []
        result = await repo.find_pending_without_job(user.id, older_than_seconds=10)
        assert result == []

    async def test_skip_pending_with_active_job(self, db: AsyncSession):
        from app.models.enums import JobStage, JobStatus
        from app.models.processing_job import ProcessingJob

        user = await _persist(db, make_user())
        stale = datetime.now(UTC) - timedelta(seconds=600)

        doc = await _persist(
            db,
            make_document(
                user.id,
                status=DocumentStatus.PENDING,
                created_at=stale,
            ),
        )
        job = ProcessingJob(document_id=doc.id, status=JobStatus.QUEUED, stage=JobStage.CHUNK)
        await _persist(db, job)

        repo = DocumentRepository(db)
        result = await repo.find_pending_without_job(user.id, older_than_seconds=3600)
        assert result == []
        result = await repo.find_pending_without_job(user.id, older_than_seconds=10)
        assert result == []

    async def test_respects_limit(self, db: AsyncSession):
        user = await _persist(db, make_user())
        stale = datetime.now(UTC) - timedelta(seconds=600)
        for i in range(5):
            await _persist(db, make_document(user.id, status=DocumentStatus.PENDING, created_at=stale, content_hash=f"h{i}"))
        repo = DocumentRepository(db)
        res = await repo.find_pending_without_job(user.id, older_than_seconds=10, limit=2)
        assert len(res) == 2

from unittest.mock import AsyncMock, MagicMock
import uuid
from datetime import datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.db.repositories.jobs import JobRepository
from app.models.enums import JobStage, JobStatus
from tests.units.conftest import make_document, make_job, make_user


@pytest.fixture
def repo(db):
    return JobRepository(db)


pytestmark = pytest.mark.unit


async def _persist(db: AsyncSession, *objs):
    for o in objs:
        db.add(o)
    await db.flush()
    for o in objs:
        await db.refresh(o)
    return objs if len(objs) > 1 else objs[0]


async def _a_document(db: AsyncSession):
    user = await _persist(db, make_user())
    return await _persist(db, make_document(user.id))


class TestCreateEnqueue:
    async def test_create_enqueued_job_with_defaults(self, db: AsyncSession, repo: JobRepository):
        doc = await _a_document(db)
        job = await repo.create_enqueue(document_id=doc.id, stage=JobStage.EXTRACT)

        assert job.id is not None
        assert job.document_id == doc.id
        assert job.stage == JobStage.EXTRACT
        assert job.details == {}
        assert job.started_at is None
        assert job.finished_at is None
        
    async def test_two_jobs_have_distinct_ids(self, db: AsyncSession, repo: JobRepository):
        doc = await _a_document(db)
        
        j1 = await repo.create_enqueue(doc.id, JobStage.EXTRACT)
        j2 = await repo.create_enqueue(doc.id, JobStage.INDEX)
        
        assert j1.id != j2.id 
        
        
class TestMakeRunning:

    async def test_queued_transitions_to_running(self, db: AsyncSession, repo: JobRepository):
        doc = await _a_document(db)
        j = await _persist(db, make_job(doc.id))
        result = await repo.make_running(j.id)
        
        assert result.status == JobStatus.RUNNING
        assert result.started_at is not None
        assert isinstance(result.started_at, datetime)
        
    async def test_running_is_idempotent(self, db: AsyncSession, repo: JobRepository):
        doc = await _a_document(db)
        j  = await _persist(db, make_job(doc.id, status=JobStatus.RUNNING))

        result = await repo.make_running(j.id)
        assert result.status == JobStatus.RUNNING
        
    async def test_reject_done_job(self, db: AsyncSession, repo: JobRepository):
        doc = await _a_document(db)
        j  = await _persist(db, make_job(doc.id, status=JobStatus.DONE))

        with pytest.raises(NotFoundError):
            await repo.make_running(j.id)
        
    async def test_reject_failed_job(self, db: AsyncSession, repo: JobRepository):
        
        doc = await _a_document(db)
        j = await _persist(db, make_job(doc.id, status=JobStatus.FAILED))
        with pytest.raises(NotFoundError):
            await repo.make_running(j.id)
        
    async def test_mission_job_raises(self, db: AsyncSession, repo: JobRepository):
        with pytest.raises(NotFoundError):
            await repo.make_running(uuid.uuid4())
        

class TestMakeDone:
    
    async def test_running_transitions_to_done(self, db: AsyncSession, repo: JobRepository):

        doc = await _a_document(db)
        j = await _persist(db, make_job(doc.id, status=JobStatus.RUNNING))
        
        res = await repo.make_done(j.id)
        assert res.status == JobStatus.DONE
        assert res.finished_at is not None
        assert isinstance(res.finished_at, datetime)
        
    async def test_merges_details(self, db: AsyncSession, repo: JobRepository):
        doc = await _a_document(db)
        j = await _persist(db, make_job(
            doc.id, details={"pages": 3, 'lang':'en'}
        ))
        
        result = await repo.make_done(j.id, details={'token':512})
        assert result.details == {"pages": 3, 'lang': 'en', 'token':512}
        
    async def test_done_without_details_keep_existing(self, db: AsyncSession, repo: JobRepository):
        doc = await _a_document(db)
        j = await _persist(db, make_job(
            doc.id, details={"pages": 3, 'lang': 'en'}
        ))

        result = await repo.make_done(j.id)
        assert result.details == {"pages": 3, 'lang': 'en'}

    async def test_already_done_is_idempotent_and_returns_job(self, db: AsyncSession, repo: JobRepository):
        
        doc = await _a_document(db)
        j = await _persist(db, make_job(
            doc.id, status=JobStatus.DONE,
        ))
        res = await repo.make_done(j.id)
        assert res is not None
        assert res.status == JobStatus.DONE
        
    async def test_mission_job_raises(self, db: AsyncSession, repo: JobRepository):
        with pytest.raises(NotFoundError):
            await repo.make_done(uuid.uuid4())


class TestMakeFailed:

    async def test_set_status_and_error_details(self, db: AsyncSession, repo: JobRepository):
        
        doc = await _a_document(db)
        j = await _persist(db, make_job(doc.id))
        
        res = await repo.make_failed(j.id, error="parse timeout")
        assert res.status == JobStatus.FAILED
        assert res.finished_at is not None
        assert isinstance(res.finished_at, datetime)
        assert res.details['error'] == 'parse timeout'

    async def test_mission_job_raises(self, db: AsyncSession, repo: JobRepository):
        with pytest.raises(NotFoundError):
            await repo.make_failed(uuid.uuid4(), error='x')

    async def test_merges_extra_details(self, db: AsyncSession, repo: JobRepository):
        doc = await _a_document(db)
        j = await _persist(db, make_job(
            doc.id,
            details={"pages": 5}
        ))
        
        res = await repo.make_failed(
            j.id, 
            error='boom',
            details={"retryable": True}
        )
        assert res.details == {"pages": 5, 'error': 'boom', "retryable": True}
        
    async def test_details_can_override_error(self, db: AsyncSession, repo: JobRepository):
        doc = await _a_document(db)
        j = await _persist(db, make_job( doc.id))
        
        res = await repo.make_failed(
            j.id, 
            error='first',
            details={"error": "second"}
        )
        assert res.details['error'] == 'second'
        

class TestListForDocument:
    async def test_returns_all_jobs_in_creation_order(self, db: AsyncSession, repo: JobRepository):
        """Catches the `raise` instead of `return` bug."""
        doc = await _a_document(db)
        j1 = await _persist(db, make_job(doc.id, stage=JobStage.EXTRACT))
        j2 = await _persist(db, make_job(doc.id, stage=JobStage.INDEX))

        jobs = await repo.list_for_document(doc.id)

        assert [j.id for j in jobs] == [j1.id, j2.id]

    async def test_empty_when_no_jobs(self, db: AsyncSession, repo: JobRepository):
        doc = await _a_document(db)

        jobs = await repo.list_for_document(doc.id)

        assert jobs == []

    async def test_does_not_leak_across_documents(self, db: AsyncSession, repo: JobRepository):
        user = await _persist(db, make_user())
        d1 = await _persist(db, make_document(user.id, content_hash="a"))
        d2 = await _persist(db, make_document(user.id, content_hash="b"))
        j1 = await _persist(db, make_job(d1.id, stage=JobStage.EXTRACT))
        await _persist(db, make_job(d2.id, stage=JobStage.EXTRACT))
        jobs = await repo.list_for_document(d1.id)

        assert [j.id for j in jobs] == [j1.id]


class TestLatestByStage:
    async def test_returns_newest_job_per_stage(self, db: AsyncSession):
        doc = await _a_document(db)
        # two PARSE jobs, oldest first
        old_parse = await _persist(db, make_job(doc.id, stage=JobStage.EXTRACT))
        new_parse = await _persist(db, make_job(doc.id, stage=JobStage.EXTRACT))
        index_job = await _persist(db, make_job(doc.id, stage=JobStage.INDEX))
        repo = JobRepository(db)

        latest = await repo.latest_by_stage(doc.id)

        assert latest[JobStage.EXTRACT].id == new_parse.id
        assert latest[JobStage.INDEX].id == index_job.id

    async def test_empty_dict_when_no_jobs(self, db: AsyncSession):
        doc = await _a_document(db)
        repo = JobRepository(db)

        assert await repo.latest_by_stage(doc.id) == {}

    async def test_does_not_leak_across_documents(self, db: AsyncSession):
        user = await _persist(db, make_user())
        d1 = await _persist(db, make_document(user.id, content_hash="a"))
        d2 = await _persist(db, make_document(user.id, content_hash="b"))
        await _persist(db, make_job(d1.id, stage=JobStage.EXTRACT))
        other = await _persist(db, make_job(d2.id, stage=JobStage.EXTRACT))
        repo = JobRepository(db)

        latest = await repo.latest_by_stage(d1.id)

        assert latest[JobStage.EXTRACT].id != other.id


class TestHasActiveJob:
    async def test_true_for_queued(self, db: AsyncSession):
        """Catches the in_() signature + Exists/scalar bugs."""
        doc = await _a_document(db)
        await _persist(db, make_job(doc.id, status=JobStatus.QUEUED))
        repo = JobRepository(db)

        assert await repo.has_active_job(doc.id) is True

    async def test_true_for_running(self, db: AsyncSession):
        doc = await _a_document(db)
        await _persist(db, make_job(doc.id, status=JobStatus.RUNNING))
        repo = JobRepository(db)

        assert await repo.has_active_job(doc.id) is True

    async def test_false_for_done(self, db: AsyncSession):
        doc = await _a_document(db)
        await _persist(db, make_job(doc.id, status=JobStatus.DONE))
        repo = JobRepository(db)

        assert await repo.has_active_job(doc.id) is False

    async def test_false_for_failed(self, db: AsyncSession):
        doc = await _a_document(db)
        await _persist(db, make_job(doc.id, status=JobStatus.FAILED))
        repo = JobRepository(db)

        assert await repo.has_active_job(doc.id) is False

    async def test_false_when_no_jobs(self, db: AsyncSession):
        doc = await _a_document(db)
        repo = JobRepository(db)

        assert await repo.has_active_job(doc.id) is False

    async def test_false_when_only_other_document_has_active_job(self, db: AsyncSession):
        user = await _persist(db, make_user())
        d1 = await _persist(db, make_document(user.id, content_hash="a"))
        d2 = await _persist(db, make_document(user.id, content_hash="b"))
        await _persist(db, make_job(d2.id, status=JobStatus.RUNNING))
        repo = JobRepository(db)

        assert await repo.has_active_job(d1.id) is False

"""Unit tests for the ingestion maintenance sweeper."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from app.models.enums import JobStage, JobStatus
from app.services.storage.base import InvalidKey, ObjectNotFound
from app.workers import sweeper as S

USER_ID = UUID("11111111-2222-3333-4444-555555555555")
DOC_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
JOB_ID = UUID("99999999-8888-7777-6666-555555555555")

# Valid shapes for both key paths the sweeper routes on.
PIPELINE_KEY = "_pipeline/cache/abcdef0123456789/" + "a" * 64 + "/chunks.json.gz"
USER_KEY = f"{USER_ID}/ab/" + "a" * 64 + ".pdf"

# --- fixtures ------------------------------------------------------------


@pytest.fixture
def session_cm():
    """A callable that mimics `AsyncSessionLocal()` as an async CM.

    `session_cm.session` is the mock session yielded on `__aenter__`,
    so tests can inspect flush/commit calls.
    """
    session = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    factory = MagicMock(return_value=cm)
    factory.session = session
    return factory


def _job(
    *,
    stage: JobStage = JobStage.EXTRACT,
    status: JobStatus = JobStatus.QUEUED,
    created_at: datetime | None = None,
    started_at: datetime | None = None,
) -> MagicMock:
    j = MagicMock()
    j.id = JOB_ID
    j.stage = stage
    j.status = status
    j.created_at = created_at or (datetime.now(UTC) - timedelta(minutes=30))
    j.started_at = started_at
    return j


class _AsyncKeyIter:
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration from None


def _storage(keys):
    s = MagicMock()
    s.list_keys = MagicMock(return_value=_AsyncKeyIter(keys))
    s.delete = AsyncMock()
    s.delete_raw = AsyncMock()
    return s


# --- _aware --------------------------------------------------------------


def test_aware_naive_becomes_utc():
    naive = datetime(2024, 1, 1, 12, 0, 0)
    assert S._aware(naive) == datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)


def test_aware_utc_passthrough():
    dt = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    assert S._aware(dt) == dt


def test_aware_other_tz_preserves_instant():
    tz = timezone(timedelta(hours=3))
    dt = datetime(2024, 1, 1, 15, 0, tzinfo=tz)  # 15:00+03 = 12:00 UTC
    assert S._aware(dt) == datetime(2024, 1, 1, 12, 0, tzinfo=UTC)


# --- _is_live ------------------------------------------------------------


def test_is_live_exact_match():
    assert S._is_live("any/key", {"any/key"}, set())


def test_is_live_exact_miss():
    assert not S._is_live("any/key", set(), set())


def test_is_live_owner_prefix():
    key = f"{USER_ID}/_pipeline/{DOC_ID}/cache/x.json.gz"
    assert S._is_live(key, set(), {(USER_ID, DOC_ID)})


def test_is_live_owner_prefix_wrong_doc():
    key = f"{USER_ID}/_pipeline/{DOC_ID}/cache/x.json.gz"
    assert not S._is_live(key, set(), {(USER_ID, uuid4())})


def test_is_live_malformed_uuid_returns_false():
    assert not S._is_live("bad/_pipeline/also-bad/x", set(), set())


def test_is_live_short_key_returns_false():
    assert not S._is_live("a/b", set(), set())


# --- _within_grace -------------------------------------------------------


def test_within_grace_unknown_mtime_kept():
    assert S._within_grace(None, datetime.now(UTC))


def test_within_grace_recent_kept():
    cutoff = datetime.now(UTC) - timedelta(hours=2)
    assert S._within_grace(datetime.now(UTC) - timedelta(minutes=5), cutoff)


def test_within_grace_old_eligible():
    cutoff = datetime.now(UTC) - timedelta(hours=2)
    assert not S._within_grace(datetime.now(UTC) - timedelta(hours=5), cutoff)


def test_within_grace_naive_mtime_treated_as_utc():
    cutoff = datetime.now(UTC) - timedelta(hours=2)
    naive_old = (datetime.now(UTC) - timedelta(hours=5)).replace(tzinfo=None)
    assert not S._within_grace(naive_old, cutoff)


async def test_load_live_keys_full_row():
    session = AsyncMock()
    result = MagicMock()
    result.all.return_value = [
        ("users/1/file.pdf", "parsed/k.json", "abc123", USER_ID, DOC_ID),
    ]
    session.execute.return_value = result

    live, prefixes = await S._load_live_keys(session)

    assert "users/1/file.pdf" in live
    assert "parsed/k.json" in live
    assert (USER_ID, DOC_ID) in prefixes


async def test_load_live_keys_handles_null_columns():
    session = AsyncMock()
    result = MagicMock()
    result.all.return_value = [(None, None, None, USER_ID, DOC_ID)]
    session.execute.return_value = result

    live, prefixes = await S._load_live_keys(session)

    assert live == set()
    assert (USER_ID, DOC_ID) in prefixes


async def test_sweep_orphans_skips_live_key(session_cm):
    old = datetime.now(UTC) - timedelta(hours=5)
    storage = _storage([("live/key", old)])

    with patch.object(S, "AsyncSessionLocal", session_cm), patch.object(S, "_load_live_keys", new=AsyncMock(return_value=({"live/key"}, set()))):
        result = await S.sweep_orphans({"storage": storage})

    assert "deleted=0" in result
    storage.delete_raw.assert_not_awaited()


async def test_sweep_orphans_skips_within_grace(session_cm):
    recent = datetime.now(UTC) - timedelta(minutes=5)
    storage = _storage([("orphan/key", recent)])

    with patch.object(S, "AsyncSessionLocal", session_cm), patch.object(S, "_load_live_keys", new=AsyncMock(return_value=(set(), set()))):
        result = await S.sweep_orphans({"storage": storage})

    assert "skipped_grace=1" in result
    storage.delete_raw.assert_not_awaited()


async def test_sweep_orphans_deletes_orphan(session_cm):
    old = datetime.now(UTC) - timedelta(hours=5)
    storage = _storage([(PIPELINE_KEY, old)])

    with patch.object(S, "AsyncSessionLocal", session_cm), patch.object(S, "_load_live_keys", new=AsyncMock(return_value=(set(), set()))):
        result = await S.sweep_orphans({"storage": storage})

    assert "scanned=1" in result
    assert "deleted=1" in result
    storage.delete_raw.assert_awaited_once_with(PIPELINE_KEY)
    storage.delete.assert_not_awaited()


async def test_sweep_orphans_caps_deletes(session_cm):
    old = datetime.now(UTC) - timedelta(hours=5)
    keys = [(PIPELINE_KEY, old)] * (S.MAX_DELETES_PER_RUN + 10)
    storage = _storage(keys)

    with patch.object(S, "AsyncSessionLocal", session_cm), patch.object(S, "_load_live_keys", new=AsyncMock(return_value=(set(), set()))):
        result = await S.sweep_orphans({"storage": storage})

    assert f"deleted={S.MAX_DELETES_PER_RUN}" in result
    assert storage.delete_raw.await_count == S.MAX_DELETES_PER_RUN


async def test_sweep_orphans_tolerates_delete_failure(session_cm):
    old = datetime.now(UTC) - timedelta(hours=5)
    storage = _storage([(PIPELINE_KEY, old), (PIPELINE_KEY, old)])
    storage.delete_raw.side_effect = [RuntimeError("boom"), None]

    with patch.object(S, "AsyncSessionLocal", session_cm), patch.object(S, "_load_live_keys", new=AsyncMock(return_value=(set(), set()))):
        result = await S.sweep_orphans({"storage": storage})

    assert "deleted=1" in result


async def test_sweep_orphans_ignores_object_not_found(session_cm):
    old = datetime.now(UTC) - timedelta(hours=5)
    storage = _storage([(PIPELINE_KEY, old)])
    storage.delete_raw.side_effect = ObjectNotFound("gone")

    with patch.object(S, "AsyncSessionLocal", session_cm), patch.object(S, "_load_live_keys", new=AsyncMock(return_value=(set(), set()))):
        result = await S.sweep_orphans({"storage": storage})

    assert "deleted=0" in result


async def test_sweep_orphans_empty_storage(session_cm):
    storage = _storage([])

    with patch.object(S, "AsyncSessionLocal", session_cm), patch.object(S, "_load_live_keys", new=AsyncMock(return_value=(set(), set()))):
        result = await S.sweep_orphans({"storage": storage})

    assert result == "scanned=0 deleted=0 skipped_grace=0"


async def test_sweep_orphans_handles_user_key_below(session_cm):
    """A user-uploaded orphan (not a pipeline key) is swept via delete().

    Regression guard: an earlier version called delete_raw for every
    key, which raised InvalidKey on user keys and leaked them.
    """

    old = datetime.now(UTC) - timedelta(hours=5)
    user_key = f"{USER_ID}/ab/{'a' * 64}.pdf"
    storage = _storage([(user_key, old)])
    storage.delete_raw.side_effect = __import__("app.services.storage.base", fromlist=["InvalidKey"]).InvalidKey("not a pipeline key")

    with patch.object(S, "AsyncSessionLocal", session_cm), patch.object(S, "_load_live_keys", new=AsyncMock(return_value=(set(), set()))):
        result = await S.sweep_orphans({"storage": storage})

    # After the fix, this should be 1 (deleted via storage.delete).
    assert "deleted=1" in result


# --- sweep_pipeline ------------------------------------------------------


async def test_sweep_pipeline_requeues_stuck_queued(session_cm):
    job = _job(status=JobStatus.QUEUED)
    redis = AsyncMock()
    redis.enqueue_job = AsyncMock(return_value="enqueued")

    with (
        patch.object(S, "AsyncSessionLocal", session_cm),
        patch.object(S.JobRepository, "list_stuck", new=AsyncMock(return_value=[job])),
        patch.dict(S._STAGE_TASK, {JobStage.EXTRACT: "extract_task"}),
    ):
        result = await S.sweep_pipeline({"redis": redis})

    assert "requeued=1" in result
    redis.enqueue_job.assert_awaited_once()


async def test_sweep_pipeline_skips_when_already_enqueued(session_cm):
    job = _job(status=JobStatus.QUEUED)
    redis = AsyncMock()
    redis.enqueue_job = AsyncMock(return_value=None)

    with (
        patch.object(S, "AsyncSessionLocal", session_cm),
        patch.object(S.JobRepository, "list_stuck", new=AsyncMock(return_value=[job])),
        patch.dict(S._STAGE_TASK, {JobStage.EXTRACT: "extract_task"}),
    ):
        result = await S.sweep_pipeline({"redis": redis})

    assert "requeued=0" in result
    assert "skipped=1" in result


async def test_sweep_pipeline_fails_running_job(session_cm):
    job = _job(
        status=JobStatus.RUNNING,
        started_at=datetime.now(UTC) - timedelta(hours=4),
    )
    redis = AsyncMock()
    ingester = MagicMock()
    ingester.fail_stage = AsyncMock()

    with (
        patch.object(S, "AsyncSessionLocal", session_cm),
        patch.object(S.JobRepository, "list_stuck", new=AsyncMock(return_value=[job])),
        patch("app.workers.tasks.Ingester", return_value=ingester),
    ):
        result = await S.sweep_pipeline({"redis": redis})

    assert "failed=1" in result
    ingester.fail_stage.assert_awaited_once()
    session_cm.session.commit.assert_awaited()


async def test_sweep_pipeline_fails_very_old_queued(session_cm):
    """A QUEUED job older than GIVE_UP_AFTER is failed, not requeued."""
    job = _job(
        status=JobStatus.QUEUED,
        created_at=datetime.now(UTC) - S.GIVE_UP_AFTER - timedelta(hours=1),
    )
    redis = AsyncMock()
    ingester = MagicMock()
    ingester.fail_stage = AsyncMock()

    with (
        patch.object(S, "AsyncSessionLocal", session_cm),
        patch.object(S.JobRepository, "list_stuck", new=AsyncMock(return_value=[job])),
        patch.dict(S._STAGE_TASK, {JobStage.EXTRACT: "extract_task"}),
        patch("app.workers.tasks.Ingester", return_value=ingester),
    ):
        result = await S.sweep_pipeline({"redis": redis})

    assert "failed=1" in result
    assert "requeued=0" in result
    redis.enqueue_job.assert_not_awaited()


async def test_sweep_pipeline_skips_unknown_stage(session_cm):
    job = _job(status=JobStatus.QUEUED)
    redis = AsyncMock()

    with (
        patch.object(S, "AsyncSessionLocal", session_cm),
        patch.object(S.JobRepository, "list_stuck", new=AsyncMock(return_value=[job])),
        patch.dict(S._STAGE_TASK, {}, clear=True),
    ):
        result = await S.sweep_pipeline({"redis": redis})

    assert "skipped=1" in result
    redis.enqueue_job.assert_not_awaited()


async def test_sweep_pipeline_handles_empty_stuck_list(session_cm):
    redis = AsyncMock()

    with (
        patch.object(S, "AsyncSessionLocal", session_cm),
        patch.object(S.JobRepository, "list_stuck", new=AsyncMock(return_value=[])),
    ):
        result = await S.sweep_pipeline({"redis": redis})

    assert result == "requeued=0 failed=0 skipped=0"
    redis.enqueue_job.assert_not_awaited()


async def test_sweep_orphans_logs_bad_key_and_continues(session_cm):
    """A malformed key shouldn't crash the sweep or count as a deletion."""
    old = datetime.now(UTC) - timedelta(hours=5)
    storage = _storage([("not-a-key", old), (PIPELINE_KEY, old)])
    # The sweeper only calls `delete`/`delete_raw` with keys that at least
    # attempted validation. Force `delete` to raise InvalidKey on the
    # non-pipeline key to exercise the branch.

    async def delete(k):
        raise InvalidKey(f"not a valid key: {k!r}")

    storage.delete.side_effect = delete

    with patch.object(S, "AsyncSessionLocal", session_cm), patch.object(S, "_load_live_keys", new=AsyncMock(return_value=(set(), set()))):
        result = await S.sweep_orphans({"storage": storage})

    assert "deleted=1" in result  # only the pipeline-key one succeeded

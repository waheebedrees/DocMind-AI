"""Unit tests for EvaluationRunRepository and EvaluationResultRepository."""

from __future__ import annotations

import pytest
from app.db.repositories.evaluation_repo import (
    EvaluationResultRepository,
    EvaluationRunRepository,
)
from app.models.enums import EvaluationStatus

pytestmark = pytest.mark.asyncio


@pytest.fixture
def runs(session):
    return EvaluationRunRepository(session)


@pytest.fixture
def results(session):
    return EvaluationResultRepository(session)


class TestCreate:
    async def test_persists_all_fields(self, runs, session, user):
        run = await runs.create(
            user_id=user.id,
            name="baseline",
            dataset_size=50,
            pipeline_version="v1",
        )
        await session.refresh(run)
        assert run.id is not None
        assert run.user_id == user.id
        assert run.name == "baseline"
        assert run.dataset_size == 50
        assert run.pipeline_version == "v1"
        assert run.status == EvaluationStatus.PENDING
        assert run.metrics == {}
        assert run.finished_at is None


class TestGetForUser:
    async def test_returns_run_when_owner_matches(self, runs, user):
        run = await runs.create(user_id=user.id, name="x", dataset_size=1, pipeline_version="v1")
        assert (await runs.get_for_user(run.id, user.id)).id == run.id

    async def test_returns_none_when_wrong_user(self, runs, user, other_user):
        run = await runs.create(user_id=user.id, name="x", dataset_size=1, pipeline_version="v1")
        assert await runs.get_for_user(run.id, other_user.id) is None

    async def test_returns_none_when_missing(self, runs, user):
        from uuid import uuid4

        assert await runs.get_for_user(uuid4(), user.id) is None


class TestListForUser:
    async def test_status_filter_is_applied(self, runs, user):
        a = await runs.create(user_id=user.id, name="a", dataset_size=1, pipeline_version="v1")
        await runs.create(user_id=user.id, name="b", dataset_size=1, pipeline_version="v1")
        await runs.set_status(a.id, EvaluationStatus.COMPLETED)
        rows = await runs.list_for_user(user.id, status=EvaluationStatus.COMPLETED)
        assert [r.id for r in rows] == [a.id]

    async def test_pagination(self, runs, user):
        for i in range(5):
            await runs.create(user_id=user.id, name=str(i), dataset_size=1, pipeline_version="v1")
        assert len(await runs.list_for_user(user.id, limit=2)) == 2

    async def test_does_not_leak_across_users(self, runs, user, other_user):
        await runs.create(user_id=user.id, name="mine", dataset_size=1, pipeline_version="v1")
        await runs.create(user_id=other_user.id, name="theirs", dataset_size=1, pipeline_version="v1")
        rows = await runs.list_for_user(user.id)
        assert [r.name for r in rows] == ["mine"]

    async def test_empty_for_user_with_no_runs(self, runs, user):
        assert await runs.list_for_user(user.id) == []


class TestSetStatus:
    async def test_transitions(self, runs, user):
        run = await runs.create(user_id=user.id, name="x", dataset_size=1, pipeline_version="v1")
        await runs.set_status(run.id, EvaluationStatus.RUNNING)
        assert run.status == EvaluationStatus.RUNNING
        await runs.set_status(run.id, EvaluationStatus.COMPLETED)
        assert run.status == EvaluationStatus.COMPLETED

    async def test_sets_finished_at_on_terminal(self, runs, user):
        run = await runs.create(user_id=user.id, name="x", dataset_size=1, pipeline_version="v1")
        await runs.set_status(run.id, EvaluationStatus.COMPLETED)
        assert run.finished_at is not None

    async def test_finished_at_is_not_overwritten_on_reentry(self, runs, user):
        run = await runs.create(user_id=user.id, name="x", dataset_size=1, pipeline_version="v1")
        await runs.set_status(run.id, EvaluationStatus.COMPLETED)
        first_finished = run.finished_at
        await runs.set_status(run.id, EvaluationStatus.COMPLETED)
        assert run.finished_at == first_finished

    async def test_metrics_can_be_replaced(self, runs, user):
        run = await runs.create(user_id=user.id, name="x", dataset_size=1, pipeline_version="v1")
        await runs.set_status(run.id, EvaluationStatus.COMPLETED, metrics={"f1": 0.9})
        assert run.metrics == {"f1": 0.9}

    async def test_missing_run_returns_none(self, runs):
        from uuid import uuid4

        assert await runs.set_status(uuid4(), EvaluationStatus.COMPLETED) is None


class TestCountAndDelete:
    async def test_count_for_user(self, runs, user, other_user):
        await runs.create(user_id=user.id, name="a", dataset_size=1, pipeline_version="v1")
        await runs.create(user_id=user.id, name="b", dataset_size=1, pipeline_version="v1")
        await runs.create(user_id=other_user.id, name="c", dataset_size=1, pipeline_version="v1")
        assert await runs.count_for_user(user.id) == 2

    async def test_delete_for_user_is_scoped(self, runs, user, other_user):
        mine = await runs.create(user_id=user.id, name="a", dataset_size=1, pipeline_version="v1")
        theirs = await runs.create(user_id=other_user.id, name="b", dataset_size=1, pipeline_version="v1")
        assert await runs.delete_for_user(mine.id, user.id) == 1
        assert await runs.get_for_user(theirs.id, other_user.id) is not None

    async def test_delete_is_idempotent(self, runs, user):
        run = await runs.create(user_id=user.id, name="a", dataset_size=1, pipeline_version="v1")
        assert await runs.delete_for_user(run.id, user.id) == 1
        assert await runs.delete_for_user(run.id, user.id) == 0


class TestResultBulkInsert:
    async def test_inserts_all_rows_and_returns_count(self, results, runs, user):
        run = await runs.create(user_id=user.id, name="x", dataset_size=2, pipeline_version="v1")
        n = await results.bulk_insert(
            [
                {
                    "run_id": run.id,
                    "question": "q1",
                    "expected_answer": "a1",
                    "actual_answer": "a1",
                    "metrics": {"em": 1.0},
                },
                {
                    "run_id": run.id,
                    "question": "q2",
                    "expected_answer": "a2",
                    "actual_answer": "a2",
                    "metrics": {"em": 1.0},
                },
            ]
        )
        assert n == 2

    async def test_empty_rows_returns_zero(self, results):
        assert await results.bulk_insert([]) == 0


class TestResultListAndCount:
    async def test_list_for_run_in_insertion_order(self, results, runs, user):
        run = await runs.create(user_id=user.id, name="x", dataset_size=2, pipeline_version="v1")
        await results.bulk_insert(
            [
                {
                    "run_id": run.id,
                    "question": q,
                    "expected_answer": "e",
                    "actual_answer": "a",
                    "metrics": {},
                }
                for q in ("q1", "q2", "q3")
            ]
        )
        rows = await results.list_for_run(run.id)
        assert [r.question for r in rows] == ["q1", "q2", "q3"]

    async def test_count_only_target_run(self, results, runs, user):
        a = await runs.create(user_id=user.id, name="a", dataset_size=1, pipeline_version="v1")
        b = await runs.create(user_id=user.id, name="b", dataset_size=1, pipeline_version="v1")
        await results.bulk_insert([{"run_id": a.id, "question": "q", "expected_answer": "e", "actual_answer": "a", "metrics": {}}])
        await results.bulk_insert([{"run_id": b.id, "question": "q", "expected_answer": "e", "actual_answer": "a", "metrics": {}}])
        assert await results.count_for_run(a.id) == 1


class TestResultCascade:
    async def test_deleting_run_cascades_to_results(self, results, runs, session, user):
        run = await runs.create(user_id=user.id, name="x", dataset_size=1, pipeline_version="v1")
        await results.bulk_insert([{"run_id": run.id, "question": "q", "expected_answer": "e", "actual_answer": "a", "metrics": {}}])
        await runs.delete_for_user(run.id, user.id)
        assert await results.count_for_run(run.id) == 0

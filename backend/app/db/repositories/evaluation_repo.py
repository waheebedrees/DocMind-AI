from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import EvaluationStatus
from app.models.evaluation import EvaluationResult, EvaluationRun


class EvaluationRunRepository:
    """Data access for :class:`EvaluationRun`.

    Runs are long-lived records; results are written in bulk once a run
    finishes. This repo never commits — callers own the transaction so
    a run and its results land atomically.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Args:
            session (AsyncSession): database session.
        """
        self.session = session

    async def create(
        self,
        *,
        user_id: UUID,
        name: str,
        dataset_size: int,
        pipeline_version: str,
    ) -> EvaluationRun:
        """Create a run in PENDING state.

        Args:
            user_id (UUID): owner.
            name (str): human-readable label.
            dataset_size (int): number of questions in the dataset.
            pipeline_version (str): identifier of the pipeline under test.

        Returns:
            EvaluationRun: the newly created, flushed instance.
        """
        run = EvaluationRun(
            user_id=user_id,
            name=name,
            dataset_size=dataset_size,
            pipeline_version=pipeline_version,
        )
        self.session.add(run)
        await self.session.flush()
        return run

    async def get_for_user(self, run_id: UUID, user_id: UUID) -> EvaluationRun | None:
        """Fetch a run only if it belongs to ``user_id``.

        Args:
            run_id (UUID): primary key.
            user_id (UUID): expected owner.

        Returns:
            EvaluationRun | None: the run, or None if missing / wrong owner.
        """
        stmt = select(EvaluationRun).where(
            EvaluationRun.id == run_id,
            EvaluationRun.user_id == user_id,
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list_for_user(
        self,
        user_id: UUID,
        *,
        status: EvaluationStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[EvaluationRun]:
        """List a user's runs, newest first.

        Args:
            user_id (UUID): owner.
            status (EvaluationStatus | None): optional status filter.
            limit (int): maximum rows. Defaults to 50.
            offset (int): rows to skip. Defaults to 0.

        Returns:
            list[EvaluationRun]: matching runs (possibly empty).
        """
        stmt = select(EvaluationRun).where(EvaluationRun.user_id == user_id)
        if status is not None:
            stmt = stmt.where(EvaluationRun.status == status)
        stmt = stmt.order_by(EvaluationRun.created_at.desc()
                             ).limit(limit).offset(offset)
        return list((await self.session.execute(stmt)).scalars().all())

    async def set_status(
        self,
        run_id: UUID,
        status: EvaluationStatus,
        *,
        metrics: dict | None = None,
    ) -> EvaluationRun | None:
        """Transition a run's status, optionally replacing its metrics.

        Stamps ``finished_at`` the first time the run reaches a terminal
        state (COMPLETED or FAILED) and leaves it untouched on re-entry so
        retries don't overwrite the original finish time.

        Args:
            run_id (UUID): primary key.
            status (EvaluationStatus): new status.
            metrics (dict | None): if provided, replaces ``run.metrics``.

        Returns:
            EvaluationRun | None: the updated run, or None if missing.
        """
        run = await self.session.get(EvaluationRun, run_id)
        if run is None:
            return None
        run.status = status
        if metrics is not None:
            run.metrics = metrics
        if status in {EvaluationStatus.COMPLETED, EvaluationStatus.FAILED} and run.finished_at is None:
            run.finished_at = datetime.now(UTC)
        await self.session.flush()
        return run

    async def count_for_user(self, user_id: UUID) -> int:
        """Count a user's runs.

        Args:
            user_id (UUID): owner.

        Returns:
            int: total number of runs.
        """
        stmt = (
            select(func.count())
            .select_from(EvaluationRun)
            .where(EvaluationRun.user_id == user_id)
        )
        return int((await self.session.scalar(stmt)) or 0)

    async def delete_for_user(self, run_id: UUID, user_id: UUID) -> int:
        """Delete a run owned by ``user_id``.

        Cascades to ``evaluation_results`` via FK ondelete rules.

        Args:
            run_id (UUID): primary key.
            user_id (UUID): expected owner.

        Returns:
            int: number of rows deleted (0 or 1).
        """
        stmt = delete(EvaluationRun).where(
            EvaluationRun.id == run_id,
            EvaluationRun.user_id == user_id,
        )
        result = await self.session.execute(stmt)
        return result.rowcount or 0


class EvaluationResultRepository:
    """Data access for :class:`EvaluationResult`.

    Results are append-only: they are written once when a run completes
    and read back for reporting. There is intentionally no ``update``.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Args:
            session (AsyncSession): database session.
        """
        self.session = session

    async def bulk_insert(self, rows: list[dict]) -> int:
        """Insert results for a run in one flush.

        Intended to be called in the same transaction as
        :meth:`EvaluationRunRepository.set_status` so the run is only
        marked COMPLETED once its results are durable.

        Args:
            rows (list[dict]): mappings accepted by
                ``EvaluationResult(**row)``. Each must include ``run_id``,
                ``question``, ``expected_answer``, ``actual_answer``, and
                ``metrics``.

        Returns:
            int: number of rows inserted (0 if ``rows`` is empty).
        """
        if not rows:
            return 0
        self.session.add_all([EvaluationResult(**r) for r in rows])
        await self.session.flush()
        return len(rows)

    async def list_for_run(self, run_id: UUID) -> list[EvaluationResult]:
        """Return a run's results, oldest first.

        Args:
            run_id (UUID): parent run.

        Returns:
            list[EvaluationResult]: results in insertion order.
        """
        stmt = (
            select(EvaluationResult)
            .where(EvaluationResult.run_id == run_id)
            .order_by(EvaluationResult.created_at, EvaluationResult.id)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def count_for_run(self, run_id: UUID) -> int:
        """Count results for a run.

        Args:
            run_id (UUID): parent run.

        Returns:
            int: number of results.
        """
        stmt = (
            select(func.count())
            .select_from(EvaluationResult)
            .where(EvaluationResult.run_id == run_id)
        )
        return int((await self.session.scalar(stmt)) or 0)

    async def delete_for_run(self, run_id: UUID) -> int:
        """Delete a run's results.

        Redundant with the run cascade; exposed for re-running a failed
        evaluation without dropping the run row.

        Args:
            run_id (UUID): parent run.

        Returns:
            int: number of results deleted.
        """
        stmt = delete(EvaluationResult).where(
            EvaluationResult.run_id == run_id)
        result = await self.session.execute(stmt)
        return result.rowcount or 0

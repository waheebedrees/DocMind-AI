from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import select

from app.core.logging import get_logger
from app.db.repositories.jobs import JobRepository
from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.enums import JobStatus
from app.services.storage import BaseStorage
from app.services.storage.base import ObjectNotFound
from app.services.storage.keys import (
    chunks_key,
    clean_key,
    embedded_chunks_key,
    parsed_key,
)
from app.workers.tasks import STAGE_TASK  # public alias of _STAGE_TASK

log = get_logger(__name__)

GRACE_PERIOD = timedelta(hours=2)
MAX_DELETES_PER_RUN = 1000

# Cover the commit→enqueue race, not a backed-up queue.
STUCK_QUEUE_AFTER = timedelta(minutes=20)

# Must exceed the slowest stage budget: embed timeout 1800s × MAX_TRIES,
# plus backoff. After this we consider the worker dead.
STUCK_RUNNING_AFTER = timedelta(hours=3)
GIVE_UP_AFTER = timedelta(hours=12)


def _aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


async def _load_live_keys(session) -> set[str]:
    """Every storage object a current document row still needs."""
    result = await session.execute(
        select(
            Document.storage_key,
            Document.parsed_key,
            Document.content_hash,
            Document.user_id,
            Document.id,
        )
    )
    live: set[str] = set()
    prefixes: set[tuple[UUID, UUID]] = set()

    for storage_key, parsed, content_hash, user_id, doc_id in result.all():
        prefixes.add((user_id, doc_id))
        if storage_key:
            live.add(storage_key)
        if parsed:
            live.add(parsed)
        if content_hash:
            live.add(parsed_key(content_hash))
            live.add(clean_key(content_hash))
            live.add(chunks_key(content_hash))
            live.add(embedded_chunks_key(content_hash))

    return live, prefixes


def _is_live(
    key: str,
    live_keys: set[str],
    live_prefixes: set[tuple[UUID, UUID]],
) -> bool:
    if key in live_keys:
        return True

    parts = key.split("/")
    if len(parts) >= 4 and parts[1] == "_pipeline":
        try:
            pair = (UUID(parts[0]), UUID(parts[2]))
        except ValueError:
            return False
        return pair in live_prefixes

    return False


def _within_grace(last_modified: datetime | None, cutoff: datetime) -> bool:
    # Unknown mtime → assume in-flight. Never delete what you cannot date.
    if last_modified is None:
        return True
    return _aware(last_modified) > cutoff


async def sweep_orphans(ctx: dict) -> str:
    """Delete storage keys not referenced by any document row."""
    storage: BaseStorage = ctx["storage"]
    cutoff = datetime.now(UTC) - GRACE_PERIOD

    async with AsyncSessionLocal() as session:
        live_keys, live_prefixes = await _load_live_keys(session)

    scanned = deleted = skipped_grace = 0

    async for key, last_modified in storage.list_keys():
        scanned += 1

        if _is_live(key, live_keys, live_prefixes):
            continue

        if _within_grace(last_modified, cutoff):
            skipped_grace += 1
            continue

        try:
            await storage.delete_raw(key)
            deleted += 1
            log.info("orphan_deleted", key=key)
        except ObjectNotFound:
            pass
        except Exception as exc:  # noqa: BLE001
            log.warning("orphan_delete_failed", key=key, error=str(exc))
            continue

        if deleted >= MAX_DELETES_PER_RUN:
            log.info("sweep_batch_cap_reached", deleted=deleted)
            break

    log.info(
        "sweep_done",
        scanned=scanned,
        deleted=deleted,
        skipped_grace=skipped_grace,
    )
    return f"scanned={scanned} deleted={deleted} skipped_grace={skipped_grace}"


async def sweep_pipeline(ctx: dict) -> str:
    """Re-enqueue QUEUED jobs that never got an ARQ task.

    RUNNING jobs older than STUCK_RUNNING_AFTER are failed, not
    re-enqueued. Re-using ``_job_id`` cannot revive an ARQ result that
    already finished, and a new job id would race a worker that is
    still in ``complete_stage``.
    """
    redis = ctx["redis"]
    now = datetime.now(UTC)

    async with AsyncSessionLocal() as session:
        job_repo = JobRepository(session)
        stuck = await job_repo.list_stuck(
            queued_before=now - STUCK_QUEUE_AFTER,
            running_before=now - STUCK_RUNNING_AFTER,
        )

    requeued = failed = skipped = 0

    for job in stuck:
        task = STAGE_TASK.get(job.stage)
        if task is None:
            log.warning(
                "sweep_unknown_stage",
                job_id=str(job.id),
                stage=str(job.stage),
            )
            skipped += 1
            continue

        age_anchor = job.started_at or job.created_at
        too_old = age_anchor is not None and _aware(age_anchor) < now - GIVE_UP_AFTER

        if job.status == JobStatus.RUNNING or too_old:
            async with AsyncSessionLocal() as session:
                from app.workers.tasks import Ingester

                await Ingester(session).fail_stage(
                    job.id,
                    error="sweep: worker presumed dead" if not too_old else "sweep: retries/age exhausted",
                )
                await session.commit()
            failed += 1
            log.warning(
                "sweep_failed_job",
                job_id=str(job.id),
                stage=job.stage.value,
                status=job.status.value,
            )
            continue

        if job.status != JobStatus.QUEUED:
            skipped += 1
            continue

        enqueued = await redis.enqueue_job(
            task,
            str(job.id),
            _job_id=str(job.id),
            request_id=str(uuid4()),
        )
        if enqueued is not None:
            requeued += 1
            log.info(
                "sweep_requeued_job",
                job_id=str(job.id),
                stage=job.stage.value,
            )
        else:
            # Still sitting in ARQ — queue is slow, not lost.
            skipped += 1

    log.info(
        "sweep_pipeline_done",
        requeued=requeued,
        failed=failed,
        skipped=skipped,
    )
    return f"requeued={requeued} failed={failed} skipped={skipped}"

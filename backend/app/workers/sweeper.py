"""Scheduled maintenance for the ingestion pipeline.

Two independent cron jobs, both idempotent and safe to run concurrently
with live traffic:

* ``sweep_orphans`` — deletes storage objects that no document row
  references. Catches leaks from crashes between the DB delete and the
  storage delete, and from any future code path that forgets to clean
  up after itself.
* ``sweep_pipeline`` — re-enqueues QUEUED jobs whose ARQ task was lost,
  and fails RUNNING jobs whose worker is presumed dead. Catches the
  commit→enqueue race and mid-stage worker crashes.

Both jobs are bounded: they cap work per pass and skip anything younger
than a grace period, so a burst of activity doesn't turn into a thundering
herd of deletes or re-enqueues.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.repositories.jobs import JobRepository
from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.enums import JobStatus
from app.services.storage import BaseStorage
from app.services.storage.base import _PIPELINE_KEY_RE, InvalidKey, ObjectNotFound
from app.services.storage.keys import (
    chunks_key,
    clean_key,
    embedded_chunks_key,
    parsed_key,
)
from app.workers.tasks import _STAGE_TASK

log = get_logger(__name__)

# How long a storage object can exist without a referencing document
# row before it's eligible for deletion. Must exceed the longest
# plausible in-flight window: an upload that streams slowly, a pipeline
# stage that's mid-write, a delete whose DB commit hasn't propagated.
GRACE_PERIOD = timedelta(hours=2)

# Cap on objects deleted per sweep pass. Bounds the cost of a single
# cron tick if a large batch of orphans accumulated (e.g. after an
# incident). Sweeps are cheap to re-run; a stuck sweep is not.
MAX_DELETES_PER_RUN = 1000

# A QUEUED job older than this with no ARQ task is presumed lost. Sized
# to cover the commit→enqueue race in ``run_stage`` — long enough that
# a slow Redis write isn't mistaken for a lost task, short enough that
# a genuinely lost job is recovered promptly.
STUCK_QUEUE_AFTER = timedelta(minutes=20)

# A RUNNING job older than this is presumed dead. Must exceed the
# slowest stage budget (EMBED: 1800s × MAX_TRIES retries, plus
# exponential backoff). If this is set too low, a stage that's merely
# slow gets marked FAILED while still running, producing a race between
# the sweeper and the worker.
STUCK_RUNNING_AFTER = timedelta(hours=3)

# Absolute age after which a job is failed regardless of status. Guards
# against a job that keeps getting requeued by the sweeper but never
# completes — an infinite requeue loop is worse than a failed job
# because it hides a real problem indefinitely.
GIVE_UP_AFTER = timedelta(hours=12)


def _aware(dt: datetime) -> datetime:
    """Return ``dt`` as a UTC-aware datetime.

    Normalizes naive datetimes (which come from some DB drivers) to
    UTC. Aware datetimes are converted, not just relabeled — passing a
    datetime in another timezone yields the correct UTC instant, not a
    mislabeled local time.

    Args:
        dt: A datetime, naive or aware.

    Returns:
        The same instant as a UTC-aware datetime.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


async def _load_live_keys(session: AsyncSession) -> tuple[set[str], set[tuple[UUID, UUID]]]:
    """Load every storage reference that a live document row still needs.

    Produces two sets:

    * **Exact keys** — the source-file key, the parsed-document key, and
      the four pipeline artifact keys derived from the content hash.
      Any storage object whose key is in this set is live.
    * **Owner prefixes** — ``(user_id, document_id)`` pairs. Any object
      under ``<user_id>/_pipeline/<document_id>/`` belongs to a live
      document, even if the exact artifact key isn't in the first set
      (e.g. a new artifact type added since this function was written).

    The second set is a defensive fallback: if a future stage writes an
    artifact under a key that this function doesn't know about, the
    owner-prefix check keeps it from being swept.

    Args:
        session: An open ``AsyncSession``. The query is read-only; the
            session is not committed or closed here.

    Returns:
        A ``(live_keys, live_prefixes)`` tuple.
    """
    result = (
        await session.execute(
            select(
                Document.storage_key,
                Document.parsed_key,
                Document.content_hash,
                Document.user_id,
                Document.id,
            )
        )
    ).all()

    live: set[str] = set()
    prefixes: set[tuple[UUID, UUID]] = set()

    for storage_key, parsed, content_hash, user_id, doc_id in result:
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
    """True if a storage key is still referenced by a live document.

    Two checks, either sufficient:

    1. Exact match against ``live_keys`` — the fast path.
    2. The key lives under ``<user_id>/_pipeline/<document_id>/`` and
       that pair is in ``live_prefixes``.

    The prefix check is what keeps artifacts written by a future stage
    from being swept. If the key is malformed (bad UUID segments), the
    check returns ``False`` rather than raising — the sweeper must not
    crash on one bad key.

    Args:
        key: A storage key returned by ``storage.list_keys()``.
        live_keys: Exact keys in use, from ``_load_live_keys``.
        live_prefixes: Owner pairs in use, from ``_load_live_keys``.

    Returns:
        True if the key should be kept. False means the caller should
        still apply the grace-period check before deleting.
    """
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
    """True if an object is too recent to consider deleting.

    An object with no known modification time is treated as in-flight —
    we never delete what we cannot date. Storage backends that don't
    report mtimes therefore never have their objects swept; those
    deployments need a different strategy.

    Args:
        last_modified: The object's mtime from ``storage.list_keys``,
            or ``None`` if the backend doesn't report one.
        cutoff: Objects modified after this are within the grace
            period and should be kept.

    Returns:
        True if the object should be skipped this pass.
    """
    # Unknown mtime → assume in-flight. Never delete what you cannot date.
    if last_modified is None:
        return True
    return _aware(last_modified) > cutoff


async def sweep_orphans(ctx: dict) -> str:
    """Delete storage objects that no document row references.

    Runs the two-phase check for every key: exact match, then owner
    prefix. Survivors of both checks are subject to the grace period,
    then deleted.

    Deletion is capped at ``MAX_DELETES_PER_RUN`` per pass. A large
    backlog of orphans is drained across multiple cron ticks rather
    than in one long-running operation.

    Args:
        ctx: ARQ context. Must contain ``storage`` (a ``BaseStorage``).

    Returns:
        A human-readable summary string for the cron log:
        ``"scanned=N deleted=N skipped_grace=N"``.
    """
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
            if _PIPELINE_KEY_RE.match(key):
                await storage.delete_raw(key)
            else:
                await storage.delete(key)
            deleted += 1
            log.info("orphan_deleted", key=key)
        except ObjectNotFound:
            pass
        except InvalidKey:
            log.warning("orphan_bad_key", key=key)
        except Exception as exc:  # noqa: BLE001 - best-effort orphan cleanup
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
    """Re-enqueue lost QUEUED jobs and fail dead RUNNING jobs.

    Three outcomes per stuck job:

    * **QUEUED, recent enough** — re-enqueued with the same ARQ
      ``_job_id``. Idempotent: if ARQ already has the task, the enqueue
      returns ``None`` and the job is counted as skipped.
    * **RUNNING, or past ``GIVE_UP_AFTER``** — marked FAILED. The job's
      worker is presumed dead; a new ARQ task would race a worker that
      is still in ``complete_stage``.
    * **Anything else** — skipped. Currently only DONE and FAILED land
      here, and both are terminal.

    RUNNING jobs are deliberately *not* re-enqueued. Two reasons:

    1. Re-using ``_job_id`` cannot revive an ARQ result that already
       finished — ARQ treats the job ID as terminal once the function
       returns.
    2. Generating a new job ID would race the original worker if it's
       still alive. Two workers running the same stage would double
       the work and possibly double-write artifacts.

    Failing is the safe choice: the document is marked FAILED and a
    human or the reprocess endpoint can restart it deliberately.

    Args:
        ctx: ARQ context. Must contain ``redis`` (an arq connection).

    Returns:
        A human-readable summary string for the cron log:
        ``"requeued=N failed=N skipped=N"``.
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
        task = _STAGE_TASK.get(job.stage)
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

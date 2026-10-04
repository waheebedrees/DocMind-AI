import asyncio
from typing import TypeVar

from app.core.decorator import with_async_backoff
from app.core.exceptions import (
    DocMindError,
    PermanentEmbeddingError,
    PermanentError,
    TransientEmbeddingError,
    TransientError,
)
from app.core.logging import get_logger
from app.services.llm_clients.base import EmbeddingClient

log = get_logger(__name__)


T = TypeVar("T")


async def embed_in_batches(
    client: EmbeddingClient,
    texts: list[str],
    *,
    concurrency: int = 4,
) -> list[list[float]]:
    """Embed texts in batches with bounded concurrency.

    Preserves input order. Cancels siblings on the first failure via
    TaskGroup, so a permanent error on batch 3 doesn't let batches 4..N
    keep hammering a dead backend.
    """
    if not texts:
        return []

    batch_size = client.max_batch_size
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    semaphore = asyncio.Semaphore(concurrency)

    async def one_batch(batch: list[str], idx: int) -> list[list[float]]:
        async with semaphore:
            return await _embed_with_retry(client, batch, batch_index=idx)

    log.info(
        "embedding_started",
        total_texts=len(texts),
        batches=len(batches),
        batch_size=batch_size,
        concurrency=concurrency,
    )

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(one_batch(b, i)) for i, b in enumerate(batches)]

    flat = [vec for task in tasks for vec in task.result()]
    if len(flat) != len(texts):
        # The client contract said "one vector per text" — a mismatch here
        # means the implementation is broken, not the input.
        raise PermanentEmbeddingError(f"embedding count mismatch: {len(flat)} vectors for {len(texts)} texts")
    return flat


@with_async_backoff(max_retries=4, initial_delay=1.0, backoff_factor=2.0, max_delay=30.0)
async def _embed_with_retry(
    client: EmbeddingClient,
    batch: list[str],
    *,
    batch_index: int,
) -> list[list[float]]:
    """One batch, retried on transient failures.

    The backoff decorator is responsible for the actual retry; this
    function's job is to make sure every exception leaving it is
    correctly classified.
    """
    try:
        return await client.embed_passages(batch)
    except PermanentError as exc:
        # Includes PermanentEmbeddingError and EmbeddingUnavailable.
        # Re-raise unchanged so the decorator does not retry.
        log.error("embedding_permanent", batch_index=batch_index, error=repr(exc))
        raise
    except TransientError as exc:
        # Includes TransientEmbeddingError. Let the decorator retry.
        log.warning("embedding_transient", batch_index=batch_index, error=repr(exc))
        raise
    except DocMindError:
        # Some other domain error — don't mask it, don't retry.
        raise
    except Exception as exc:
        # Unknown exception from the client. Assume transient: the cost
        # of a bounded retry is lower than the cost of dropping a batch.
        log.warning(
            "embedding_unclassified",
            batch_index=batch_index,
            error=repr(exc),
        )
        raise TransientEmbeddingError(f"unclassified embedding failure: {exc}") from exc

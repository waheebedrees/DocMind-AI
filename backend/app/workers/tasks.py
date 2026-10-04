"""Ingestion pipeline.

This module owns the ingestion half of DocMind: the ARQ task wrappers
that drive a document through extract => clean => chunk => embed =? index,
plus the ``Ingester`` service that enforces the legal state transitions
at each step.
"""

import asyncio
import gzip
import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any
from uuid import UUID

from arq import Retry, func
from docling_core.types.doc import DoclingDocument
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    DocumentNotFound,
    InvalidTransition,
    JobAlreadyDone,
    JobAlreadyFailed,
    JobNotFound,
    PermanentError,
)
from app.core.logging import get_logger
from app.db.repositories.chunks import PreparedChunk
from app.db.session import AsyncSessionLocal
from app.models.enums import JobStage
from app.rag.ingestion import Ingester
from app.services.storage import BaseStorage, ObjectNotFound
from app.services.storage.keys import (
    chunks_key,
    clean_key,
    embedded_chunks_key,
    parsed_key,
)

log = get_logger(__file__)

StageFn = Callable[[dict, AsyncSession, UUID], Awaitable[dict[str, Any]]]


# The canonical stage order. ``_STAGE_INDEX`` maps a stage to its
# position so ``run_stage`` can find the next stage in O(1). A stage's
# task name is the ARQ function name, so it must match the decorated
# function names below.

_STAGE_SEQUENCE: tuple[tuple[JobStage, str], ...] = (
    (JobStage.EXTRACT, "process_extract"),
    (JobStage.CLEAN, "process_clean"),
    (JobStage.CHUNK, "process_chunk"),
    (JobStage.EMBED, "process_embed"),
    (JobStage.INDEX, "process_index"),
)
_STAGE_INDEX = {stage: i for i, (stage, _) in enumerate(_STAGE_SEQUENCE)}
_STAGE_TASK = dict(_STAGE_SEQUENCE)

# Per-stage timeouts, in seconds. Embedding is the outlier because the
# first call in a process pays the model-load cost (~8 s for
# bge-small on CPU; more for larger models).
_STAGE_TIMEOUT = {
    JobStage.EXTRACT: 900,
    JobStage.CLEAN: 300,
    JobStage.CHUNK: 600,
    JobStage.EMBED: 1800,
    JobStage.INDEX: 600,
}

# Errors that ``run_stage`` treats as permanent. Anything not in this
# tuple is retried up to ``MAX_TRIES``. ``PermanentError`` covers every
# subclass raised by the embedding stack; the stdlib exceptions are
# here because they typically indicate a code bug rather than a
# transient failure.
_PERMANENT_ERRORS = (
    PermanentError,  # covers PermanentError and its subclasses
    ObjectNotFound,
    DocumentNotFound,
    JobNotFound,
    TypeError,
    ValueError,
    KeyError,
    AttributeError,
)


def _backoff(job_try: int) -> int:
    """Compute the next retry delay in seconds.

    Sequence: 10s, 20s, 40s, 80s, ..., capped at 5 minutes. ``job_try``
    is 1-based (ARQ's convention), so the first failure sleeps 10 s.

    Args:
        job_try: The attempt number that just failed.

    Returns:
        Delay in seconds before the next attempt.
    """
    return min(5 * 2**job_try, 300)


async def run_stage(
    ctx: dict,
    *,
    job_id: UUID,
    stage: JobStage,
    fn: StageFn,
    request_id: str,
) -> None:
    """Execute one ingestion stage and enqueue the next.

    The lifecycle:

    1. Mark the job running and the document PROCESSING (idempotent).
    2. Invoke ``fn(ctx, session, document_id)`` and commit.
    3. In one transaction, mark the stage done and create the next
       stage's job row.
    4. After commit, enqueue the next stage's ARQ task.

    If the worker dies between commit and enqueue, the next stage's job
    row exists but no ARQ task points at it. Reconciliation is the
    responsibility of the scheduled recovery job, not this function.

    Errors from ``fn`` are classified:

    * In ``_PERMANENT_ERRORS`` — the job and document are marked FAILED,
      and the function returns. ARQ will not retry.
    * Anything else — the attempt is recorded and ``Retry`` is raised.
      ARQ reschedules. After ``MAX_TRIES`` attempts, the job is marked
      FAILED with a "retries exhausted" error.

    Args:
        ctx: ARQ context. Must contain ``job_try`` (int), ``redis`` (an
            arq connection), and ``storage`` (a BaseStorage).
        job_id: The job row this invocation is executing. If the row has
            been deleted, the function returns silently.
        stage: The stage being run. Used for logging and to decide which
            stage comes next.
        fn: The stage implementation. Receives an open ``AsyncSession``
            and the ``document_id``; must not open its own session.
        request_id: Correlation ID from the original HTTP request.
            Propagated to logs and to the next stage's enqueue.

    Raises:
        Retry: If ``fn`` raises a non-permanent error and retries
            remain. ARQ catches this and reschedules.
    """
    job_try = ctx.get("job_try", 1)
    slog = log.bind(
        job_id=str(job_id),
        stage=stage.value,
        job_try=job_try,
        request_id=request_id,
    )

    async with AsyncSessionLocal() as session:
        ingester = Ingester(session)
        try:
            job = await ingester.start_stage(job_id, stage=stage)
        except JobNotFound:
            slog.warning("job_missing")
            return
        except (JobAlreadyDone, JobAlreadyFailed) as exc:
            slog.info("job_already_finished", error=str(exc))
            return
        except DocumentNotFound as exc:
            await ingester.fail_stage(job_id, error=f"{type(exc).__name__}: {exc}")
            await session.commit()
            slog.warning("document_missing", error=str(exc))
            return
        except InvalidTransition as exc:
            await ingester.fail_stage(job_id, error=f"{type(exc).__name__}: {exc}")
            await session.commit()
            slog.error("invalid_transition", error=str(exc))
            return

        document_id = job.document_id
        await session.commit()

    try:
        async with AsyncSessionLocal() as session:
            result = await fn(ctx, session, document_id)
            await session.commit()
    except _PERMANENT_ERRORS as exc:
        async with AsyncSessionLocal() as session:
            await Ingester(session).fail_stage(job_id, error=f"{type(exc).__name__}: {exc}")
            await session.commit()
        slog.error("stage_permanent_failure", document_id=str(document_id), error=str(exc))
        return
    except Exception as exc:
        if job_try >= settings.job_settings.max_tries:
            async with AsyncSessionLocal() as session:
                await Ingester(session).fail_stage(job_id, error=f"retries exhausted: {exc!r}")
                await session.commit()
            slog.error("stage_permanent_failure", document_id=str(document_id), error=str(exc))
            return

        async with AsyncSessionLocal() as session:
            await Ingester(session).record_attempt(job_id, error=f"{type(exc).__name__}: {exc}", attempt=job_try)
            await session.commit()
        slog.warning("stage_transient_failure", error=str(exc), attempt=job_try)
        raise Retry(defer=_backoff(job_try)) from exc

    idx = _STAGE_INDEX[stage]
    is_last = idx == len(_STAGE_SEQUENCE) - 1
    next_stage = None if is_last else _STAGE_SEQUENCE[idx + 1][0]
    next_job_id: UUID | None = None

    try:
        async with AsyncSessionLocal() as session:
            next_job = await Ingester(session).complete_stage(
                job_id,
                details=result,
                next_stage=next_stage,
            )
            if next_job is not None:
                next_job_id = next_job.id
            await session.commit()
    except JobAlreadyDone:
        slog.info("stage_already_completed")
        return
    except (DocumentNotFound, InvalidTransition) as exc:
        async with AsyncSessionLocal() as session:
            await Ingester(session).fail_stage(job_id, error=f"{type(exc).__name__}: {exc}")
            await session.commit()
        slog.error("stage_completion_failed", document_id=str(document_id), error=str(exc))
        return

    slog.info("stage_completed", document_id=str(document_id), details=result)

    if is_last:
        slog.info("pipeline_completed", document_id=str(document_id))
        return

    if next_job_id is None:
        slog.error("missing_next_job_id", next=next_stage.value if next_stage else None)
        return

    next_task = _STAGE_SEQUENCE[idx + 1][1]
    try:
        enqueued = await ctx["redis"].enqueue_job(
            next_task,
            str(next_job_id),
            _job_id=str(next_job_id),
            request_id=request_id,
        )
    except (ConnectionError, TimeoutError, RuntimeError):
        slog.exception("stage_enqueue_failed", next=next_task, next_job_id=str(next_job_id))
        return

    if enqueued is None:
        slog.warning("stage_enqueue_duplicate", next_job_id=str(next_job_id))
    else:
        slog.info("stage_enqueued", next=next_task, next_job_id=str(next_job_id))


# Parsed documents and chunk rows are persisted as gzipped JSON in the
# storage backend, keyed by content hash. The helpers below are the
# only code that knows the on-disk format; everything else calls them.


async def save_parsed_document(
    storage: BaseStorage,
    doc: DoclingDocument,
    key: str,
) -> str:
    """Persist a ``DoclingDocument`` to storage as gzipped JSON.

    Args:
        storage: The storage backend to write to.
        doc: The document to serialize.
        key: The storage key to write. Callers derive it from the
            document's content hash so identical content reuses the
            same artifact.

    Returns:
        The key that was written. Returned for convenience so callers
        can chain into a log line or a return dict.
    """
    await storage.put_bytes(key, gzip.compress(doc.model_dump_json().encode()))
    return key


async def load_parsed_document(storage: BaseStorage, key: str) -> DoclingDocument:
    """Load a ``DoclingDocument`` from a gzipped JSON artifact.

    Args:
        storage: The storage backend to read from.
        key: The key produced by ``save_parsed_document``.

    Returns:
        The decoded document.

    Raises:
        ObjectNotFound: The key does not exist. Permanent.
    """
    payload = await storage.get_bytes(key)
    return DoclingDocument.model_validate_json(gzip.decompress(payload))


async def _save_rows(
    storage: BaseStorage,
    key: str,
    rows: list[PreparedChunk],
):
    """Persist a list of ``PreparedChunk`` as gzipped JSON.

    Internal helper shared by the chunk and embed stages. Not part of
    the module's public surface — callers should use the stage-specific
    loaders.

    Args:
        storage: The storage backend to write to.
        key: The storage key to write.
        rows: The rows to serialize. Dataclass fields are written
            verbatim; tuple fields are converted to JSON arrays by
            ``asdict``.

    Returns:
        The key that was written.
    """
    payload = json.dumps([asdict(r) for r in rows], separators=(",", ":"))
    await storage.put_bytes(key, gzip.compress(payload.encode()))


async def _load_rows(storage: BaseStorage, key: str) -> list[PreparedChunk]:
    """Load ``PreparedChunk`` rows from a gzipped JSON artifact.

    Restores tuple fields (``doc_item_labels``) from the JSON array form
    back to tuples, since ``asdict`` stringifies them on write.

    Args:
        storage: The storage backend to read from.
        key: The key produced by ``_save_rows``.

    Returns:
        The decoded rows, in the order they were written.
    """
    payload = await storage.get_bytes(key)
    items = json.loads(gzip.decompress(payload))
    return [PreparedChunk(**{**i, "doc_item_labels": tuple(i.get("doc_item_labels", ()))}) for i in items]


async def load_chunks(storage: BaseStorage, key: str) -> list[PreparedChunk]:
    """Load the chunk artifact written by the chunk stage."""
    return await _load_rows(storage, key)


async def load_embedded_chunks(
    storage: BaseStorage,
    key: str,
) -> list[PreparedChunk]:
    """Load the embedded-chunk artifact written by the embed stage."""
    return await _load_rows(storage, key)


def _docling_stats(doc: DoclingDocument) -> dict[str, int]:
    """Derive page/text/table counts from a parsed document.

    Replaces the old sidecar ``.meta.json`` artifact: the stats are
    always consistent with the document because they're recomputed
    from it, not persisted separately.
    """
    return {
        "pages": len(doc.pages),
        "text_items": len(doc.texts),
        "tables": len(doc.tables),
    }


async def do_extract(ctx: dict, session: AsyncSession, document_id: UUID):
    """Parse the source file into a ``DoclingDocument``.

    Cached by content hash: if the parsed artifact already exists (same
    content was uploaded before), it's loaded and returned without
    re-parsing. The cache-hit path recomputes stats from the artifact
    rather than reading a sidecar, so stats are always consistent.

    Side effects: writes the parsed artifact, and sets
    ``document.parsed_key`` so downstream stages can find it.

    Returns:
        ``{"parsed_key": str, "cached": bool, "pages": int,
        "text_items": int, "tables": int}``. ``pages`` flows into the
        document's ``page_count`` via ``complete_stage``.
    """
    storage: BaseStorage = ctx["storage"]
    ingester = Ingester(session)
    document = await ingester.get_document(document_id, "extract")
    extract_key = parsed_key(document.content_hash)

    if await storage.exists(extract_key):
        log.info("extract_cache_hit", document_id=str(document_id), content_hash=document.content_hash)
        await ingester.set_parsed_key(document_id=document_id, parsed_key=extract_key)
        dl_doc = await load_parsed_document(storage, extract_key)
        return {"parsed_key": extract_key, "cached": True, **_docling_stats(dl_doc)}

    async with storage.materialize(document.storage_key) as path:
        log.info("extract_docling_start", document_id=str(document_id))
        dl_doc = await asyncio.to_thread(ingester.extract_document, path)
        log.info("extract_docling_done", document_id=str(document_id), pages=len(dl_doc.pages))

    await save_parsed_document(storage, dl_doc, extract_key)
    meta = _docling_stats(dl_doc)
    await ingester.set_parsed_key(document_id=document_id, parsed_key=extract_key)
    log.info("extract_done", document_id=str(document_id), parsed_key=extract_key, **meta)
    return {"parsed_key": extract_key, "cached": False, **meta}


async def do_clean(ctx: dict, session: AsyncSession, document_id: UUID):
    """Apply cleaning rules to the parsed document.

    Reads the artifact referenced by ``document.parsed_key``, runs the
    cleaner, and writes the result to a new key. Cached by content
    hash: if the clean artifact exists, it's reused.

    Returns:
        ``{"parsed_key": str, "cached": bool}`` plus whatever counters
        the cleaner returned.
    """
    storage: BaseStorage = ctx["storage"]
    ingester = Ingester(session)
    document = await ingester.get_document(document_id, "clean")
    key = clean_key(document.content_hash)

    if await storage.exists(key):
        log.info("clean_cache_hit", document_id=str(document_id))
        return {"parsed_key": key, "cached": True}

    if document.parsed_key is None:
        raise PermanentError("parsed_key missing; extract stage did not run")

    dl_doc = await load_parsed_document(storage, document.parsed_key)

    stats = await asyncio.to_thread(ingester.clean_document, dl_doc)

    await save_parsed_document(storage, dl_doc, key)

    log.info("cleaning_done", document_id=str(document_id))
    return {"parsed_key": key, "cached": False, **stats}


async def do_chunk(ctx: dict, session: AsyncSession, document_id: UUID):
    """Split the cleaned document into chunks and persist them.

    Reads the clean artifact, runs the chunker, and writes the chunk
    list to a content-hash-keyed artifact that the embed stage reads.

    Returns:
        ``{"chunk_count": int, "chunks_key": str}``.
    """
    storage: BaseStorage = ctx["storage"]
    ingester = Ingester(session)
    document = await ingester.get_document(document_id, "chunk")

    dl_doc = await load_parsed_document(storage, clean_key(document.content_hash))

    chunk_key = chunks_key(document.content_hash)

    chunk_list = await asyncio.to_thread(ingester.chunk_document, dl_doc)

    # Persist chunks as JSON so embed reads from a stable artifact.
    await _save_rows(storage, chunk_key, chunk_list)

    log.info("chunking_done", chunk_count=len(chunk_list), document_id=str(document_id))
    return {"chunk_count": len(chunk_list), "chunks_key": chunk_key}


async def do_embed(ctx: dict, session: AsyncSession, document_id: UUID):
    """Embed every chunk for a document and persist the result.

    Reads the chunk artifact, calls ``Ingester.embed`` (which owns the
    timeout and classification), and writes the embedded rows to a
    content-hash-keyed artifact that the index stage reads.

    Raises:
        PermanentError: The chunk artifact is empty. A document that
            produced zero chunks cannot be embedded.
        TransientEmbeddingError: The embedder timed out. Retryable.
        PermanentEmbeddingError: The embedder returned a malformed
            vector. Not retryable.

    Returns:
        ``{"chunk_count": int, "dimension": int,
        "embedded_chunks_key": str}``.
    """
    storage: BaseStorage = ctx["storage"]
    ingester = Ingester(session)
    document = await ingester.get_document(document_id, "embed")
    embed_key = embedded_chunks_key(document.content_hash)

    chunk_list = await load_chunks(storage, chunks_key(document.content_hash))
    if not chunk_list:
        raise PermanentError("no chunks to embed")
    rows = await ingester.embed(chunk_list)

    dimension = 0
    if rows[0].embedding:
        dimension = len(rows[0].embedding)

    log.info("embedding_done", document_id=str(document_id), chunk_count=len(rows))
    await _save_rows(storage, embed_key, rows)

    return {
        "chunk_count": len(rows),
        "dimension": dimension,
        "embedded_chunks_key": embed_key,
    }


async def do_index(ctx: dict, session: AsyncSession, document_id: UUID):
    """Write embedded chunks into the database.

    Validates that every row has an embedding of the expected dimension
    before touching the database — a mismatch means the model was
    swapped without reindexing, which is permanent.

    Replaces all existing chunks for the document. Idempotent.

    Raises:
        PermanentError: The artifact is empty, a row has no embedding,
            or a row's dimension doesn't match
            ``settings.embedding_spec.dimension``.

    Returns:
        ``{"inserted": int, "replaced": int}``. ``inserted`` flows into
        the document's ``metadata_["chunk_count"]`` via
        ``complete_stage``.
    """
    ingester = Ingester(session)
    document = await ingester.get_document(document_id, "index")
    storage: BaseStorage = ctx["storage"]
    embed_key = embedded_chunks_key(document.content_hash)

    rows = await load_embedded_chunks(storage, embed_key)
    if not rows:
        raise PermanentError("no chunks to index")

    expected = settings.embedding_spec.dimension
    for row in rows:
        if row.embedding is None:
            raise PermanentError(f"chunk {row.chunk_index} has no embedding — embed stage did not run")
        if len(row.embedding) != expected:
            raise PermanentError(f"chunk {row.chunk_index} has {len(row.embedding)} dims, expected {expected}")

    deleted, inserted = await ingester.bulk_insert(document_id=document_id, rows=rows)

    log.info("chunks_indexed", document_id=str(document_id), deleted=deleted, inserted=inserted)
    return {"inserted": inserted, "replaced": deleted}


def _make_task(stage: JobStage, fn: StageFn):
    """Wrap a stage implementation as an ARQ task.

    ARQ passes the job's arguments positionally and expects a callable
    with a stable ``__name__`` (it uses the name as the task key in
    Redis). This factory binds the stage and its implementation into a
    single-argument async function named after the stage, then returns
    it.

    Args:
        stage: Which stage the task implements. Used to look up the
            task name and to pass into ``run_stage``.
        fn: The stage implementation.

    Returns:
        An async function with signature ``(ctx, job_id, request_id)``
        suitable for ARQ.
    """

    async def task(ctx: dict, job_id: str, request_id: str) -> None:
        await run_stage(ctx, job_id=UUID(job_id), stage=stage, fn=fn, request_id=request_id)

    task.__name__ = task.__qualname__ = _STAGE_TASK[stage]
    return task


process_extract = _make_task(JobStage.EXTRACT, do_extract)
process_clean = _make_task(JobStage.CLEAN, do_clean)
process_chunk = _make_task(JobStage.CHUNK, do_chunk)
process_embed = _make_task(JobStage.EMBED, do_embed)
process_index = _make_task(JobStage.INDEX, do_index)


# ARQ worker registration. The timeout and max_tries are looked up per
# stage so that embed, which pays the model-load cost on first call,
# gets a longer budget than the compute-light stages.
WORKER_FUNCTIONS = [
    func(task, name=task.__name__, timeout=_STAGE_TIMEOUT[stage], max_tries=settings.job_settings.max_tries)
    for stage, task in (
        (JobStage.EXTRACT, process_extract),
        (JobStage.CLEAN, process_clean),
        (JobStage.CHUNK, process_chunk),
        (JobStage.EMBED, process_embed),
        (JobStage.INDEX, process_index),
    )
]

"""ARQ task definitions for the ingestion pipeline.

Each task is a thin wrapper around run_stage. The stage logic lives
in app/services/ingestion/. This module is only responsible for the
task signature ARQ expects and the fn binding.
"""

import asyncio
import gzip
import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from arq import Retry, func
from docling_core.types.doc import DoclingDocument
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.db.repositories.chunks import ChunkRepository, PreparedChunk
from app.db.repositories.documents import DocumentRepository
from app.db.repositories.jobs import JobRepository
from app.db.session import AsyncSessionLocal
from app.models.enums import DocumentStatus, JobStage
from app.rag.chunk.chunking import chunk_with_splitter
from app.rag.cleaning import clean_document
from app.rag.extraction import extract_document
from app.rag.ingestion import embed_in_batches, get_embedder
from app.services.storage import BaseStorage
from app.services.storage.keys import all_artifact_keys, chunks_key, clean_key, embedded_chunks_key, parsed_key

log = get_logger(__name__)

MAX_TRIES = 5
StageFn = Callable[[dict, AsyncSession, UUID], Awaitable[dict[str, Any]]]


class PermanentError(Exception):
    """Retrying will not help (bad input, missing data, config mismatch)."""


class TransientError(Exception):
    """Optional marker. Any exception not in _PERMANENT_ERRORS is retried anyway."""


_PERMANENT_ERRORS = (PermanentError, TypeError, ValueError, KeyError, AttributeError)


_STAGE_SEQUENCE: tuple[tuple[JobStage, str], ...] = (
    (JobStage.EXTRACT, "process_extract"),
    (JobStage.CLEAN, "process_clean"),
    (JobStage.CHUNK, "process_chunk"),
    (JobStage.EMBED, "process_embed"),
    (JobStage.INDEX, "process_index"),
)
_STAGE_INDEX = {stage: i for i, (stage, _) in enumerate(_STAGE_SEQUENCE)}
_STAGE_TASK = dict(_STAGE_SEQUENCE)

_STAGE_TIMEOUT = {
    JobStage.EXTRACT: 900,
    JobStage.CLEAN: 300,
    JobStage.CHUNK: 600,
    JobStage.EMBED: 1800,
    JobStage.INDEX: 600,
}


def _backoff(job_try: int) -> int:
    # 10s, 20s, 40s, 80s ... capped at 5 min
    return min(5 * 2**job_try, 300)


async def run_stage(ctx: dict, *, job_id: UUID, stage: JobStage, fn: StageFn, request_id: str) -> None:
    """
    Execute one pipeline stage with full state management.

    The ARQ worker calls this via process_extract / process_clean / etc.
    """
    job_try = ctx.get("job_try", 1)
    slog = log.bind(
        job_id=str(job_id),
        stage=stage.value,
        job_try=job_try,
        request_id=request_id,
    )

    # 1 mark running (idempotent)
    async with AsyncSessionLocal() as session:
        jobs, docs = JobRepository(session), DocumentRepository(session)
        job = await jobs.get_by_id(job_id)
        if job is None:
            slog.warning("job_missing")
            return
        document_id = job.document_id
        await jobs.mark_running(job.id)

        if await docs.get_by_id(document_id) is None:
            await jobs.mark_failed(job_id, error="document missing", details={})
            await session.commit()
            slog.warning("document_missing", document_id=str(document_id))
            return

        await docs.set_status(document_id, DocumentStatus.PROCESSING)
        await session.commit()

    slog.info("stage_started", document_id=str(document_id))

    # 2 run the stage
    try:
        async with AsyncSessionLocal() as session:
            result = await fn(ctx, session, document_id)
            await session.commit()
    except _PERMANENT_ERRORS as exc:
        await _record_permanent(job_id, document_id, exc, slog)
        return
    except Exception as exc:
        if job_try >= MAX_TRIES:
            # arq will not call us again after this, so finalize state now.
            await _record_permanent(job_id, document_id, PermanentError(f"retries exhausted: {exc!r}"), slog)
            return
        await _record_transient(job_id, exc, job_try, slog)
        # this is what makes arq retry
        raise Retry(defer=_backoff(job_try)) from exc

    # 3 mark done + prepare next step in ONE transaction
    idx = _STAGE_INDEX[stage]
    is_last = idx == len(_STAGE_SEQUENCE) - 1
    next_job_id: UUID | None = None

    async with AsyncSessionLocal() as session:
        await JobRepository(session).mark_done(job_id, details=result)
        docs = DocumentRepository(session)
        if is_last:
            await docs.set_status(document_id, DocumentStatus.INDEXED)
            await docs.set_index_results(
                document_id,
                chunk_count=result.get("inserted", 0),
                indexed_at=datetime.now(UTC),
                page_count=result.get("page_count"),
            )
        else:
            next_stage = _STAGE_SEQUENCE[idx + 1][0]
            next_job = await JobRepository(session).create_enqueue(
                document_id=document_id,
                stage=next_stage,
            )
            next_job_id = next_job.id
        await session.commit()

    slog.info("stage_completed", document_id=str(document_id), details=result)

    if is_last:
        slog.info("pipeline_completed", document_id=str(document_id))
        return

    # 4 enqueue next stage (the explicit _job_id makes this idempotent)
    next_task = _STAGE_SEQUENCE[idx + 1][1]
    enqueued = await ctx["redis"].enqueue_job(
        next_task,
        str(next_job_id),
        _job_id=str(next_job_id),
        request_id=request_id,
    )
    if enqueued is None:
        slog.warning(
            "stage_enqueue_duplicate",
            next_job_id=str(next_job_id),
        )
    else:
        slog.info("stage_enqueued", next=next_task, next_job_id=str(next_job_id))


async def _record_transient(job_id: UUID, exc: Exception, attempt: int, slog) -> None:
    async with AsyncSessionLocal() as session:
        await JobRepository(session).recode_attempt(job_id, error=f"{type(exc).__name__}: {exc}", attempt=attempt)
        await session.commit()
    slog.warning("stage_transient_failure", error=str(exc), attempt=attempt)


async def _record_permanent(job_id: UUID, document_id: UUID, exc: Exception, slog) -> None:
    msg = f"{type(exc).__name__}: {exc}"
    async with AsyncSessionLocal() as session:
        await JobRepository(session).mark_failed(job_id, error=msg, details={})
        await DocumentRepository(session).set_status(document_id=document_id, status=DocumentStatus.FAILED, error_messages=msg)
        await session.commit()
    slog.error("stage_permanent_failure", document_id=str(document_id), error=msg)


def save_parsed_document(storage: BaseStorage, doc: DoclingDocument, key: str) -> str:

    storage.write_sync(key, gzip.compress(doc.model_dump_json().encode()))
    return key


def load_parsed_document(storage: BaseStorage, key: str) -> DoclingDocument:
    return DoclingDocument.model_validate_json(gzip.decompress(storage.read_sync(key)))


def _save_rows(storage: BaseStorage, key: str, rows: list[PreparedChunk]) -> str:
    payload = json.dumps([asdict(r) for r in rows], separators=(",", ":"))
    storage.write_sync(key, gzip.compress(payload.encode()))
    return key


def _load_rows(storage: BaseStorage, key: str) -> list[PreparedChunk]:
    items = json.loads(gzip.decompress(storage.read_sync(key)))
    return [PreparedChunk(**{**i, "doc_item_labels": tuple(i.get("doc_item_labels", ()))}) for i in items]


def save_chunks(storage: BaseStorage, user_id: UUID, document_id: UUID, rows: list[PreparedChunk]) -> str:
    return _save_rows(storage, chunks_key(user_id=user_id, document_id=document_id), rows)


def load_chunks(storage: BaseStorage, key: str) -> list[PreparedChunk]:
    return _load_rows(storage, key)


def save_embedded_chunks(storage: BaseStorage, user_id: UUID, document_id: UUID, rows: list[PreparedChunk]) -> str:
    return _save_rows(storage, embedded_chunks_key(user_id=user_id, document_id=document_id), rows)


def load_embedded_chunks(storage: BaseStorage, key: str) -> list[PreparedChunk]:
    return _load_rows(storage, key)


async def _get_document(session: AsyncSession, document_id: UUID, stage: str):
    doc = await DocumentRepository(session).get_by_id(document_id)
    if doc is None:
        raise PermanentError(f"document {document_id} not found in {stage} stage")
    return doc


async def _do_extract(ctx: dict, session: AsyncSession, document_id: UUID):

    storage: BaseStorage = ctx["storage"]
    document = await _get_document(session, document_id, "extract")

    # Docling runs synchronously and is CPU-bound — must go off-loop.
    async with storage.materialize(document.storage_key) as path:
        dl_doc = await asyncio.to_thread(extract_document, path)

    key = parsed_key(document.user_id, document_id)

    await asyncio.to_thread(save_parsed_document, storage, dl_doc, key)
    await DocumentRepository(session).set_parsed_key(document_id=document_id, parsed_key=key)

    log.info("extract_done", document_id=str(document_id), parsed_key=key)
    return {
        "pages": len(dl_doc.pages),
        "text_items": len(dl_doc.texts),
        "tables": len(dl_doc.tables),
        "parsed_key": key,
    }


async def _do_clean(ctx: dict, session: AsyncSession, document_id: UUID):
    storage: BaseStorage = ctx["storage"]
    document = await _get_document(session, document_id, "clean")
    if document.parsed_key is None:
        raise PermanentError("parsed_key missing; extract stage did not run")

    dl_doc = await asyncio.to_thread(load_parsed_document, storage, document.parsed_key)
    stats = await asyncio.to_thread(clean_document, dl_doc)
    key = await asyncio.to_thread(save_parsed_document, storage, dl_doc, clean_key(document.user_id, document_id))

    log.info("cleaning_done", document_id=str(document_id))
    return {"parsed_key": key, **stats}


async def _do_chunk(ctx: dict, session: AsyncSession, document_id: UUID):
    storage: BaseStorage = ctx["storage"]
    document = await _get_document(session, document_id, "chunk")

    dl_doc = await asyncio.to_thread(load_parsed_document, storage, clean_key(user_id=document.user_id, document_id=document_id))
    chunk_list = await asyncio.to_thread(lambda: list(chunk_with_splitter(document=dl_doc)))

    # Persist chunks as JSON so embed reads from a stable artifact.
    key = await asyncio.to_thread(save_chunks, storage, document.user_id, document_id, chunk_list)

    log.info("chunking_done", chunk_count=len(chunk_list), document_id=str(document_id))
    return {"chunk_count": len(chunk_list), "chunks_key": key}


async def _do_embed(ctx: dict, session: AsyncSession, document_id: UUID):
    storage: BaseStorage = ctx["storage"]
    document = await _get_document(session, document_id, "embed")
    chunk_list = await asyncio.to_thread(load_chunks, storage, chunks_key(user_id=document.user_id, document_id=document_id))
    if not chunk_list:
        raise PermanentError("no chunks to embed")

    embedder = get_embedder()
    # httpx timeouts / 5xx propagate as-is
    # run_stage retries them with backoff
    vectors = await embed_in_batches(embedder, [c.text for c in chunk_list])

    rows = [
        PreparedChunk(
            chunk_index=c.chunk_index,
            text=c.text,
            page_number=c.page_number,
            section=c.section,
            token_count=c.token_count,
            doc_item_labels=c.doc_item_labels,
            embedding=v,
        )
        for c, v in zip(chunk_list, vectors, strict=True)
    ]
    key = await asyncio.to_thread(save_embedded_chunks, storage, document.user_id, document_id, rows)

    log.info("embedding_done", document_id=str(document_id), chunk_count=len(rows))
    return {"chunk_count": len(rows), "dimension": embedder.dimension, "embedded_chunks_key": key}


async def _do_index(ctx: dict, session: AsyncSession, document_id: UUID):
    document = await _get_document(session, document_id, "index")
    storage: BaseStorage = ctx["storage"]

    rows = await asyncio.to_thread(load_embedded_chunks, storage, embedded_chunks_key(document.user_id, document_id))
    if not rows:
        raise PermanentError("no chunks to index")

    expected = settings.embedding_spec.dimension

    for row in rows:
        if row.embedding is None:
            raise PermanentError(f"chunk {row.chunk_index} has no embedding — embed stage did not run")
        if len(row.embedding) != expected:
            raise PermanentError(f"chunk {row.chunk_index} has {len(row.embedding)} dims, expected {expected}")

    chunk_repo = ChunkRepository(session)
    deleted = await chunk_repo.delete_for_document(document_id)
    inserted = await chunk_repo.bulk_insert(document_id=document_id, rows=rows)

    # Pipeline artifacts are no longer needed after indexing
    for f in all_artifact_keys(document.user_id, document_id):
        await storage.delete_raw(f)

    log.info("chunks_indexed", document_id=str(document_id), deleted=deleted, inserted=inserted)
    return {"inserted": inserted, "replaced": deleted}


def _make_task(stage: JobStage, fn: StageFn):
    async def task(ctx: dict, job_id: str, request_id: str) -> None:
        await run_stage(ctx, job_id=UUID(job_id), stage=stage, fn=fn, request_id=request_id)

    task.__name__ = task.__qualname__ = _STAGE_TASK[stage]
    return task


process_extract = _make_task(JobStage.EXTRACT, _do_extract)
process_clean = _make_task(JobStage.CLEAN, _do_clean)
process_chunk = _make_task(JobStage.CHUNK, _do_chunk)
process_embed = _make_task(JobStage.EMBED, _do_embed)
process_index = _make_task(JobStage.INDEX, _do_index)


WORKER_FUNCTIONS = [
    func(task, name=task.__name__, timeout=_STAGE_TIMEOUT[stage], max_tries=MAX_TRIES)
    for stage, task in (
        (JobStage.EXTRACT, process_extract),
        (JobStage.CLEAN, process_clean),
        (JobStage.CHUNK, process_chunk),
        (JobStage.EMBED, process_embed),
        (JobStage.INDEX, process_index),
    )
]

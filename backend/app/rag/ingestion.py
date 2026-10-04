"""Ingestion pipeline.

This module owns the ingestion half of DocMind: the ARQ task wrappers
that drive a document through extract => clean => chunk => embed =? index,
plus the ``Ingester`` service that enforces the legal state transitions
at each step.
"""

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from docling_core.types.doc import DoclingDocument
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import DocumentNotFound, InvalidTransition, JobAlreadyDone, JobAlreadyFailed, JobNotFound, TransientEmbeddingError
from app.core.logging import get_logger
from app.db.repositories.chunks import ChunkRepository, PreparedChunk
from app.db.repositories.documents import DocumentRepository
from app.db.repositories.jobs import JobRepository
from app.models.enums import DocumentStatus, JobStage, JobStatus
from app.models.processing_job import ProcessingJob
from app.rag.chunk.chunking import chunk_with_splitter
from app.rag.cleaning import clean_document
from app.rag.embed import embed_in_batches
from app.rag.extraction import extract_document
from app.services.llm_clients import get_embedder

log = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Ingester:
    """Enforce the document/job state machine for the ingestion pipeline.
    The class wraps an ``AsyncSession`` and never commits. Callers own
    transaction boundaries; every method here ends in ``flush()`` so the
    caller can inspect results before deciding to commit.
    """

    def __init__(self, session: AsyncSession):
        self._session = session
        self._jobs = JobRepository(session)
        self._documents = DocumentRepository(session)
        self._chunks = ChunkRepository(session)

    @staticmethod
    def extract_document(source: str | Path) -> DoclingDocument:
        """Parse a document into a ``DoclingDocument``"""
        return extract_document(source)

    @staticmethod
    def clean_document(dl_doc: DoclingDocument) -> dict[str, int]:
        """Apply cleaning rules to a parsed document, in place."""
        return clean_document(dl_doc)

    @staticmethod
    def chunk_document(dl_doc: DoclingDocument) -> list[PreparedChunk]:
        """Split a cleaned document into embeddable chunks."""
        return list(chunk_with_splitter(document=dl_doc))

    async def get_job(self, job_id: UUID) -> ProcessingJob | None:
        """Fetch a job by ID, or ``None`` if it doesn't exist."""
        return await self._jobs.get_by_id(job_id)

    async def set_parsed_key(self, document_id: UUID, parsed_key: str) -> None:
        """Record the storage key of the parsed artifact on the document."""
        await self._documents.set_parsed_key(
            document_id=document_id,
            parsed_key=parsed_key,
        )

    async def get_document(self, document_id: UUID, stage: str = "internal"):
        """Fetch a document, raising ``DocumentNotFound`` if missing"""
        doc = await self._documents.get_by_id(document_id)
        if doc is None:
            raise DocumentNotFound(f"document {document_id} not found in {stage} stage")
        return doc

    async def record_attempt(
        self,
        job_id: UUID,
        *,
        error: str,
        attempt: int,
    ) -> None:
        """Record a transient failure without changing job status."""
        job = await self._jobs.get_by_id(job_id)
        if job is None or job.status != JobStatus.RUNNING:
            return
        await self._jobs.record_attempt(job_id, error=error, attempt=attempt)

    async def start_stage(
        self,
        job_id: UUID,
        *,
        stage: JobStage | None = None,
    ) -> ProcessingJob:
        """Transition job queued → running and its document → processing.

        Idempotent for a job already RUNNING or FAILED, so an ARQ retry
        can call it again without error. Commits the "document missing"
        case itself, since there is no caller to commit for.

        Args:
            job_id: The job to start.

        Returns:
            The job row, now in RUNNING state. The caller owns the
            commit.

        Raises:
            JobNotFound: The job row is gone. ``run_stage`` treats this
                as a benign exit.
            DocumentNotFound: The job exists but its document doesn't.
                The job is marked FAILED and the transaction committed
                before raising.
            InvalidTransition: The job or document is in a state that
                cannot transition to running/processing.
        """

        job = await self._jobs.get_by_id(job_id)
        if job is None:
            raise JobNotFound(str(job_id))

        if stage is not None and job.stage != stage:
            raise InvalidTransition(f"job {job_id} is for stage {job.stage}, not {stage}")

        if job.status == JobStatus.DONE:
            raise JobAlreadyDone(f"job {job_id} already done")
        if job.status == JobStatus.FAILED:
            raise JobAlreadyFailed(f"job {job_id} already failed")
        if job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
            raise InvalidTransition(f"job {job_id} cannot start from status {job.status}")

        doc = await self._documents.get_by_id(job.document_id)
        if doc is None:
            raise DocumentNotFound(str(job.document_id))

        if doc.status not in (DocumentStatus.PENDING, DocumentStatus.PROCESSING):
            raise InvalidTransition(f"document {doc.id} cannot start from status {doc.status}")

        if job.status == JobStatus.QUEUED:
            job.status = JobStatus.RUNNING
            job.started_at = _utcnow()
        elif job.started_at is None:
            job.started_at = _utcnow()

        if doc.status == DocumentStatus.PENDING:
            await self._documents.set_status(doc.id, DocumentStatus.PROCESSING)

        await self._session.flush()
        return job

    async def fail_stage(self, job_id: UUID, *, error: str) -> None:
        """Transition job → FAILED and its document → FAILED.

        Best-effort: if either row is missing, its update is skipped.
        Used for permanent errors and retry exhaustion.

        Args:
            job_id: The job that failed.
            error: Human-readable description. Written to the job's
                ``details["error"]`` and the document's
                ``error_message`` column.
        """
        job = await self._jobs.get_by_id(job_id)
        if job is None or job.status in (JobStatus.DONE, JobStatus.FAILED):
            await self._session.flush()
            return

        job.status = JobStatus.FAILED
        job.finished_at = _utcnow()
        job.details = {**(job.details or {}), "error": error}

        doc = await self._documents.get_by_id(job.document_id)
        if doc is not None and doc.status in (
            DocumentStatus.PENDING,
            DocumentStatus.PROCESSING,
        ):
            doc.status = DocumentStatus.FAILED
            doc.error_message = error

        await self._session.flush()

    async def complete_stage(
        self,
        job_id: UUID,
        *,
        details: dict,
        next_stage: JobStage | None,
    ) -> ProcessingJob | None:
        """Transition job RUNNING => DONE and advance the pipeline."""

        job = await self._jobs.get_by_id(job_id)
        if job is None:
            raise JobNotFound(str(job_id))
        if job.status == JobStatus.DONE:
            raise JobAlreadyDone(f"job {job_id} already done")
        if job.status != JobStatus.RUNNING:
            raise InvalidTransition(f"job {job_id} cannot complete from status {job.status}")

        doc = await self._documents.get_by_id(job.document_id)
        if doc is None:
            raise DocumentNotFound(str(job.document_id))
        if doc.status != DocumentStatus.PROCESSING:
            raise InvalidTransition(f"document {doc.id} cannot complete from status {doc.status}")

        job.status = JobStatus.DONE
        job.finished_at = _utcnow()
        job.details = {**(job.details or {}), **details}

        if job.stage == JobStage.EXTRACT and "pages" in details:
            doc.page_count = details["pages"]

        if next_stage is None:
            doc.status = DocumentStatus.INDEXED
            doc.indexed_at = _utcnow()
            doc.metadata_ = {
                **(doc.metadata_ or {}),
                "chunk_count": details.get("inserted", 0),
            }
            next_job = None
        else:
            next_job = await self._jobs.create_enqueue(
                document_id=doc.id,
                stage=next_stage,
            )

        await self._session.flush()
        return next_job

    async def embed(self, chunk_list: list[PreparedChunk]) -> list[PreparedChunk]:
        """Embed a list of chunks and attach vectors to each."""

        embedder = get_embedder()
        timeout_s = settings.embedding_spec.embed_timeout_s
        try:
            async with asyncio.timeout(timeout_s):
                vectors = await embed_in_batches(embedder, [c.text for c in chunk_list])
        except (TimeoutError, OSError) as exc:
            raise TransientEmbeddingError(f"embedding timeout: {exc}") from exc

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
        return rows

    async def bulk_insert(
        self,
        document_id: UUID,
        rows: Sequence[PreparedChunk],
    ) -> tuple[int, int]:
        """Replace all chunks for a document with a new set."""
        deleted = await self._chunks.delete_for_document(document_id)
        inserted = await self._chunks.bulk_insert(document_id, rows=rows)
        return deleted, inserted

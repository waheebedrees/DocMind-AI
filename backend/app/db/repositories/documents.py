"""Document repository.

Owns persistence for the ``Document`` model: creation-time dedup,
status transitions, per-user queries, and the reconciliation reads the
sweeper uses to recover stuck documents. Every method here is
per-user scoped unless the name says otherwise.
"""

from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.db.repositories.base import BaseRepository
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.models.processing_job import ProcessingJob


class DocumentRepository(BaseRepository[Document]):
    """Persistence for ``Document`` rows.

    Wraps an ``AsyncSession``. No method commits; the caller owns the
    transaction. Read methods return ``None`` for "not found"; write
    methods raise ``NotFoundError`` when the target row is missing,
    unless the docstring says otherwise.
    """

    def __init__(self, session: AsyncSession):
        super().__init__(Document, session)

    async def get_for_user(
        self,
        user_id: UUID,
        document_id: UUID,
    ) -> Document | None:
        """Fetch a document scoped to its owner.

        This is the only safe way to look up a document on behalf of a
        user. A document that exists but belongs to another user is
        indistinguishable from one that doesn't exist — both return
        ``None``. Callers turn that into a 404, not a 403, so that
        document IDs don't leak.

        Args:
            user_id: The owner to scope the lookup to.
            document_id: The document to load.

        Returns:
            The document, or ``None`` if it doesn't exist or isn't
            owned by ``user_id``.
        """
        stmt = select(self.model).where(Document.id == document_id, Document.user_id == user_id)
        doc = await self.session.execute(stmt)
        return doc.scalar()

    async def find_by_content_hash(
        self,
        user_id: UUID,
        content_hash: str,
    ) -> Document | None:
        """Deduplication lookup for re-uploading the same file.

        Dedup is *per user*, not global. Two users uploading the same
        bytes each get their own ``Document`` row; the *storage* object
        may be shared, but the logical documents are not. Do not build
        cross-user accounting on this method.

        Args:
            user_id: The owner to scope the lookup to.
            content_hash: SHA-256 hex digest of the file's bytes.

        Returns:
            The existing document with this content hash, or ``None``
            if the user has not uploaded this content before.
        """
        return await self.session.scalar(
            select(Document).where(
                Document.user_id == user_id,
                Document.content_hash == content_hash,
            )
        )

    async def list_for_user(
        self,
        user_id: UUID,
        status: DocumentStatus | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[Document], int]:
        """List a user's documents, newest first, with a total count.

        The count is of the *unpaginated* result set — it's the total
        number of documents matching the filter, not the number
        returned. Pagination controls that use it to render "showing X
        of Y" work correctly; controls that use it as ``has_more`` do
        not.

        Args:
            user_id: The owner to list for.
            status: Optional filter. ``None`` returns documents in
                every status.
            limit: Page size. Defaults to 20.
            offset: Documents to skip. Defaults to 0.

        Returns:
            A ``(documents, total)`` tuple. ``documents`` has at most
            ``limit`` entries; ``total`` is the count of all documents
            matching the filter, before pagination.
        """
        stmt = select(Document).where(Document.user_id == user_id)
        if status is not None:
            stmt = stmt.where(Document.status == status)

        total = await self.total(stmt)

        rows = await self.session.scalars(
            self.page(
                stmt.order_by(Document.created_at.desc()),
                limit=limit,
                offset=offset,
            )
        )
        return list(rows.all()), total

    async def set_status(
        self,
        document_id: UUID,
        status: DocumentStatus,
        error_messages: str | None = None,
    ) -> None:
        """Update a document's status and, optionally, its error field.

        The ``error_message`` column is *overwritten* on every call,
        including with ``None``. Calling ``set_status(id, PROCESSING)``
        after a failure clears the previous error. If you want to
        preserve prior errors, pass the existing value back in.

        When the new status is ``INDEXED``, ``indexed_at`` is set to
        the current UTC time. No other status touches ``indexed_at``.

        Args:
            document_id: The document to update.
            status: The new status.
            error_messages: Text to write to ``error_message``. Pass
                ``None`` to clear it.

        Raises:
            NotFoundError: The document doesn't exist. Uses code
                ``invalid_document_id``.
        """
        doc = await self.get_by_id(document_id)
        if doc is None:
            raise NotFoundError(
                "document Not found",
                code="invalid_document_id ",
            )
        doc.status = status
        doc.error_message = error_messages
        if status == DocumentStatus.INDEXED:
            doc.indexed_at = datetime.now(UTC)

        await self.session.flush()

    async def set_parsed_key(self, document_id: UUID, parsed_key: str) -> None:
        """Record the storage key of the parsed-document artifact.

        Called by the extract stage so downstream stages (clean, chunk)
        can locate the artifact without re-deriving the key from the
        content hash.

        Args:
            document_id: The document to update.
            parsed_key: The storage key returned by
                ``save_parsed_document``.

        Raises:
            NotFoundError: The document doesn't exist.
        """
        doc = await self.get_by_id(document_id)
        if doc is None:
            raise NotFoundError(
                "document Not found",
                code="invalid_document_id ",
            )
        doc.parsed_key = parsed_key
        await self.session.flush()

    async def count_for_user(self, user_id: UUID) -> int:
        """Count all of a user's documents, regardless of status.

        Args:
            user_id: The owner.

        Returns:
            The number of documents. Zero if the user has none.
        """
        result = await self.session.scalar(select(func.count(Document.id)).where(Document.user_id == user_id))
        return int(result or 0)

    async def total_bytes_for_user(self, user_id: UUID) -> int:
        """Sum the ``size_bytes`` of all of a user's documents.

        Used for storage quota enforcement. Counts logical documents,
        not unique storage objects — if the user uploaded the same
        content twice under different filenames, both ``size_bytes``
        values are summed even though storage holds one object.

        Args:
            user_id: The owner.

        Returns:
            Total bytes, or 0 if the user has no documents.
        """
        result = await self.session.scalar(select(func.coalesce(func.sum(Document.size_bytes), 0)).where(Document.user_id == user_id))
        return int(result or 0)

    async def find_pending_without_job(
        self,
        user_id: UUID,
        older_than_seconds: int = 12,
        limit: int = 20,
    ) -> list[Document]:
        """Find PENDING documents that have no active job.

        Used by the reconciliation sweeper to detect uploads whose
        enqueue step was lost (worker crash between commit and
        ``enqueue_job``). A document is a candidate if it has been
        PENDING for longer than ``older_than_seconds`` and no job for
        it is currently QUEUED or RUNNING.

        Args:
            user_id: Scope the sweep to one user. Typically called in
                a loop over users by the cron job.
            older_than_seconds: Grace period. Documents younger than
                this are ignored — the enqueue may simply not have
                happened yet.
            limit: Maximum number of documents to return. Bounds the
                work the caller does per pass.

        Returns:
            Candidates for re-enqueue, oldest first is *not* guaranteed.
        """
        cutoff = datetime.now(UTC) - timedelta(seconds=older_than_seconds)
        active = (
            select(ProcessingJob.id)
            .where(
                ProcessingJob.document_id == Document.id,
                ProcessingJob.status.in_(["queued", "running"]),
            )
            .exists()
        )
        rows = await self.session.scalars(
            select(Document)
            .where(
                Document.user_id == user_id,
                Document.status == DocumentStatus.PENDING,
                Document.created_at < cutoff,
                ~active,
            )
            .limit(limit)
        )
        return list(rows.all())

    async def set_index_results(
        self,
        document_id: UUID,
        *,
        chunk_count: int,
        indexed_at: datetime | None,
        page_count: int | None = None,
    ) -> None:
        """Write final index-time metadata onto a document.

        Sets ``metadata_["chunk_count"]`` and ``indexed_at``. Optionally
        overwrites ``page_count`` — used by stages that know the page
        count but aren't the extract stage.

        **Does not flush or commit.** The caller must do so. Silently
        returns if the document is missing — see *Known limitations*.

        Args:
            document_id: The document to update.
            chunk_count: Number of chunks that were indexed.
            indexed_at: Timestamp to record, or ``None`` to clear it.
            page_count: If provided, overwrites the document's page
                count.
        """
        doc = await self.get_by_id(document_id)
        if doc is None:
            return
        meta = dict(doc.metadata_ or {})
        meta["chunk_count"] = chunk_count
        doc.metadata_ = meta
        doc.indexed_at = indexed_at
        if page_count is not None:
            doc.page_count = page_count

    async def list_processing_older_than(
        self,
        cutoff: datetime,
    ) -> list[Document]:
        """List PROCESSING documents not updated since ``cutoff``.

        The sweeper uses this to find documents that got stuck in
        PROCESSING — the worker died before ``complete_stage`` could
        move them forward, and no further job will touch them.

        Args:
            cutoff: Documents whose ``updated_at`` is older than this
                are returned.

        Returns:
            Stuck documents, oldest first. The ordering lets the
            sweeper prioritize the longest-stuck ones if it needs to
            cap work per pass.
        """
        stmt = (
            select(Document)
            .where(
                Document.status == DocumentStatus.PROCESSING,
                Document.updated_at < cutoff,
            )
            .order_by(Document.updated_at)
        )
        rows = await self.session.scalars(stmt)
        return list(rows.all())

    async def reconcile_indexed(
        self,
        document_id: UUID,
        last_job: ProcessingJob,
    ) -> None:
        """Mark a document INDEXED based on its final completed job.

        Used by the sweeper when a document is stuck in PROCESSING but
        the index stage already finished — the state transition that
        would have flipped it to INDEXED never committed. Trusts
        ``last_job`` to be the *final* DONE job (i.e. stage == INDEX);
        passing an earlier stage's job produces wrong metadata.

        Sets ``indexed_at`` to the job's ``finished_at``, or now if
        that's unset. Writes ``chunk_count`` from
        ``last_job.details["inserted"]``, defaulting to 0.

        Silently returns if the document is missing.

        Args:
            document_id: The document to reconcile.
            last_job: The final DONE job for the document. Its
                ``details`` are read for the chunk count.
        """
        doc = await self.get_by_id(document_id)
        if doc is None:
            return
        doc.status = DocumentStatus.INDEXED
        doc.indexed_at = last_job.finished_at or datetime.now(UTC)
        doc.metadata_ = {
            **(doc.metadata_ or {}),
            "chunk_count": (last_job.details or {}).get("inserted", 0),
        }
        await self.session.flush()

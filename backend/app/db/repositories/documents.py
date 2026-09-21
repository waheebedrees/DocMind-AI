from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.base import BaseRepository
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.models.processing_job import ProcessingJob
from app.core.exceptions import NotFoundError


class DocumentRepository(BaseRepository[Document]):
    def __init__(self, session: AsyncSession):
        super().__init__(Document, session)

    async def get_for_user(self, document_id: UUID, user_id: UUID) -> Document | None:

        stmt = select(self.model).where(Document.id == document_id, Document.user_id == user_id)
        doc = await self.session.execute(stmt)

        return doc.scalar()

    async def find_by_content_hash(self, user_id: UUID, content_hash: str) -> Document | None:
        """Deduplication lookup for re-uploading the same file."""
        return await self.session.scalar(select(Document).where(Document.user_id == user_id, Document.content_hash == content_hash))

    async def list_for_user(
        self, user_id: UUID, status: DocumentStatus | None = None, limit: int = 20, offset: int = 0
    ) -> tuple[list[Document], int]:
        """_summary_
        list all document recodes for this user

        Args:
            user_id (UUID): owner user id
            status (DocumentStatus | None, optional): document status. Defaults to None.
            limit (int, optional): number of document. Defaults to 20.
            offset (int, optional): number of skip documents. Defaults to 0.

        Returns:
            list[Document]: list of user's documents
        """
        stmt = select(Document).where(Document.user_id == user_id)
        if status is not None:
            stmt = stmt.where(Document.status == status)

        total = await self.total(stmt)

        rows = await self.session.scalars(self.page(stmt.order_by(Document.created_at.desc()), limit=limit, offset=offset))
        return list(rows.all()), total

    async def set_status(self, document_id: UUID, status: DocumentStatus, error_messages: str | None = None) -> None:

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
        doc = await self.get_by_id(document_id)
        if doc is None:
            raise NotFoundError(
                "document Not found",
                code="invalid_document_id ",
            )        
        doc.parsed_key = parsed_key
        await self.session.flush()

    async def count_for_user(
        self,
        user_id: UUID,
    ) -> int:

        result = await self.session.scalar(select(func.count(Document.id)).where(Document.user_id == user_id))
        return int(result or 0)

    async def total_bytes_for_user(self, user_id: UUID) -> int:
        result = await self.session.scalar(select(func.coalesce(func.sum(Document.size_bytes), 0)).where(Document.user_id == user_id))
        return int(result or 0)

    async def find_pending_without_job(self, user_id: UUID, older_than_seconds: int = 12, limit: int = 20) -> list[Document]:

        cutoff = datetime.now(UTC) - timedelta(seconds=older_than_seconds)
        active = select(ProcessingJob.id).where(ProcessingJob.document_id == Document.id, ProcessingJob.status.in_(["queued", "running"])).exists()
        rows = await self.session.scalars(
            select(Document)
            .where(Document.user_id == user_id, Document.status == DocumentStatus.PENDING, Document.created_at < cutoff, ~active)
            .limit(limit)
        )
        return list(rows.all())

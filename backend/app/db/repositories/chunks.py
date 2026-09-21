from uuid import UUID, uuid4
from typing import Optional
from sqlalchemy import select, func, insert, delete, Result
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.base import BaseRepository
from app.models.chunk import DocumentChunk
from app.models.enums import DocumentStatus

from collections.abc import Sequence

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkRow:
    """Flat input shape for bulk_insert — decoupled from ingestion.Chunk."""

    chunk_index: int
    text: str
    page_number: int | None
    section: str | None
    token_count: int
    doc_item_labels: tuple[str, ...]
    embedding: list[float]


class ChunkRepository(BaseRepository[DocumentChunk]):
    def __init__(self, session: AsyncSession):
        super().__init__(DocumentChunk, session)

    async def bulk_insert(self, document_id: UUID, rows: Sequence[ChunkRow]):
        if not rows:
            return 0

        payload = [
            {
                "id": uuid4(),
                "document_id": document_id,
                "chunk_index": r.chunk_index,
                "text": r.text,
                "page_number": r.page_number,
                "section": r.section,
                "token_count": r.token_count,
                "metadata_": {"doc_item_labels": list(r.doc_item_labels)},

                "embedding": r.embedding,
            }
            for r in rows
        ]

        await self.session.execute(insert(DocumentChunk), payload)
        await self.session.flush()
        return len(payload)

    async def delete_for_document(self, document_id: UUID) -> int:
        result = await self.session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        return result.rowcount or 0

    async def list_for_document(self, document_id: UUID) -> list[DocumentChunk]:
        rows = await self.session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document_id).order_by(DocumentChunk.chunk_index))
        return list(rows.all())

    async def count_for_document(self, document_id: UUID) -> int:
        result = await self.session.scalar(select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == document_id))
        return int(result or 0)

"""Chunk repository.

Owns persistence for ``DocumentChunk`` rows: bulk insert, per-document
replacement, and reads for retrieval and diagnostics. Chunk writes are
always bulk — the index stage replaces a document's entire chunk set in
one transaction rather than updating rows individually.
"""

from collections.abc import Sequence
from uuid import UUID, uuid4

from sqlalchemy import delete, func, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.base import BaseRepository
from app.models.chunk import DocumentChunk
from app.rag.chunk import PreparedChunk


class ChunkRepository(BaseRepository[DocumentChunk]):
    """Persistence for ``DocumentChunk`` rows.

    No method commits; the caller owns the transaction. There are no
    "not found" raises — a document with no chunks is a valid state,
    not an error.
    """

    def __init__(self, session: AsyncSession):
        super().__init__(DocumentChunk, session)

    async def bulk_insert(
        self,
        document_id: UUID,
        rows: Sequence[PreparedChunk],
    ) -> int:
        """Insert a batch of chunks for a document.

        Each row's UUID is generated here — ``PreparedChunk`` carries no
        primary key. ``doc_item_labels`` is stored inside the
        ``metadata_`` JSON column as a list (JSON has no tuple type).

        Does not dedup or replace. Callers who want replace semantics
        should call ``delete_for_document`` first, or use the index
        stage's ``Ingester.bulk_insert`` which wraps both.

        Args:
            document_id: The document the chunks belong to.
            rows: The chunks to insert. An empty sequence is a no-op.

        Returns:
            The number of rows inserted (0 for an empty input).
        """
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
        """Delete every chunk for a document.

        Returns the deleted count by materializing the IDs first — see
        *Known limitations* for why this is more expensive than it
        needs to be.

        Args:
            document_id: The document whose chunks to delete.

        Returns:
            The number of chunks deleted. 0 if the document had none.
        """
        result = await self.session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id).returning(DocumentChunk.id))
        return len(result.scalars().all())

    async def list_for_document(
        self,
        document_id: UUID,
    ) -> list[DocumentChunk]:
        """List every chunk for a document, in reading order.

        Ordered by ``chunk_index``. This is the order the chunker
        produced, which reflects the source document's layout.

        Args:
            document_id: The document whose chunks to list.

        Returns:
            Chunks ordered by ``chunk_index``. Empty list if the
            document has no chunks.
        """
        rows = await self.session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document_id).order_by(DocumentChunk.chunk_index))
        return list(rows.all())

    async def count_for_document(self, document_id: UUID) -> int:
        """Count the chunks for a document.

        Used for diagnostics and to verify that a reprocess produced
        the expected number of chunks.

        Args:
            document_id: The document to count.

        Returns:
            The number of chunks, or 0 if the document has none.
        """
        result = await self.session.scalar(select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == document_id))
        return int(result or 0)

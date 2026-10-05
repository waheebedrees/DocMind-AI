from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.base import BaseRepository
from app.models.extraction import Extraction


class ExtractionRepository(BaseRepository[Extraction]):
    """Data access for :class:`Extraction`.

    Extractions are keyed by ``(document_id, schema_name)`` in practice —
    one extraction per schema per document — but the model doesn't enforce
    that, so this repo exposes both a per-document listing and a
    lookup-by-schema helper rather than pretending the pair is unique.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Args:
            session (AsyncSession): database session.
        """
        super().__init__(Extraction, session)

    async def list_for_document(self, document_id: UUID) -> list[Extraction]:
        """Return a document's extractions, newest first.

        Args:
            document_id (UUID): source document.

        Returns:
            list[Extraction]: matching extractions (possibly empty).
        """
        stmt = select(Extraction).where(Extraction.document_id == document_id).order_by(Extraction.created_at.desc())
        return list((await self.session.execute(stmt)).scalars().all())

    async def get_for_document_by_schema(self, document_id: UUID, schema_name: str) -> Extraction | None:
        """Return the newest extraction of a given schema for a document.

        "Newest" rather than "the" because the model permits multiple rows
        per ``(document_id, schema_name)``. Callers that need the unique
        invariant should add it as a DB constraint first.

        Args:
            document_id (UUID): source document.
            schema_name (str): schema identifier.

        Returns:
            Extraction | None: the most recent match, or None.
        """
        stmt = (
            select(Extraction)
            .where(
                Extraction.document_id == document_id,
                Extraction.schema_name == schema_name,
            )
            .order_by(Extraction.created_at.desc())
            .limit(1)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list_needing_review(
        self,
        *,
        document_id: UUID | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Extraction]:
        """List extractions flagged for human review, oldest first.

        Oldest-first so a review queue drains FIFO.

        Args:
            document_id (UUID | None): restrict to one document.
            limit (int): maximum rows. Defaults to 100.
            offset (int): rows to skip. Defaults to 0.

        Returns:
            list[Extraction]: flagged extractions.
        """
        stmt = select(Extraction).where(Extraction.needs_review.is_(True))
        if document_id is not None:
            stmt = stmt.where(Extraction.document_id == document_id)
        stmt = stmt.order_by(Extraction.created_at, Extraction.id).limit(limit).offset(offset)
        return list((await self.session.execute(stmt)).scalars().all())

    async def set_needs_review(self, extraction_id: UUID, needs_review: bool) -> Extraction | None:
        """Flip the review flag on an extraction.

        Args:
            extraction_id (UUID): primary key.
            needs_review (bool): new flag value.

        Returns:
            Extraction | None: the updated row, or None if missing.
        """
        row = await self.session.get(Extraction, extraction_id)
        if row is None:
            return None
        row.needs_review = needs_review
        await self.session.flush()
        return row

    async def count_for_document(self, document_id: UUID) -> int:
        """Count a document's extractions.

        Args:
            document_id (UUID): source document.

        Returns:
            int: number of extractions.
        """
        stmt = select(func.count()).select_from(Extraction).where(Extraction.document_id == document_id)
        return int((await self.session.scalar(stmt)) or 0)

    async def delete_for_document(self, document_id: UUID) -> int:
        """Delete every extraction for a document.

        Redundant with the document cascade; exposed for reprocessing
        flows that clear and regenerate.

        Args:
            document_id (UUID): source document.

        Returns:
            int: number of rows deleted.
        """
        stmt = delete(Extraction).where(Extraction.document_id == document_id)
        return await self.execute_rowcount(stmt)

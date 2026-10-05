from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.base import BaseRepository
from app.models.citation import Citation


class CitationRepository(BaseRepository[Citation]):
    """Data access for :class:`Citation`.

    Citations are written in bulk right after a message is generated, so
    the primary entry point is :meth:`bulk_insert`. Reads are ordered by
    ``rank`` — the stable, caller-assigned position — not by score.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Args:
            session (AsyncSession): database session.
        """
        super().__init__(Citation, session)

    async def bulk_insert(self, rows: Sequence[dict]) -> int:
        """Insert many citations in a single flush.

        The caller owns the transaction; this method does not commit. It
        is intended to be called in the same transaction as the message
        the citations belong to, so a failure rolls back both.

        Args:
            rows (Sequence[dict]): mappings accepted by ``Citation(**row)``.
                Each row must include ``message_id``, ``chunk_id``,
                ``score``, and ``rank``.

        Returns:
            int: number of rows inserted (0 if ``rows`` is empty).

        Raises:
            IntegrityError: on duplicate ``(message_id, chunk_id)`` or a
                check-constraint violation (e.g. rank < 1, score out of
                [0, 1]).
        """
        if not rows:
            return 0
        self.session.add_all([Citation(**r) for r in rows])
        await self.session.flush()
        return len(rows)

    async def list_for_message(self, message_id: UUID) -> list[Citation]:
        """Return a message's citations in rank order.

        Args:
            message_id (UUID): parent message.

        Returns:
            list[Citation]: citations ordered by ``rank`` ascending
                (possibly empty).
        """
        stmt = select(Citation).where(Citation.message_id == message_id).order_by(Citation.rank)
        return list((await self.session.execute(stmt)).scalars().all())

    async def count_for_message(self, message_id: UUID) -> int:
        """Count citations attached to a message.

        Args:
            message_id (UUID): parent message.

        Returns:
            int: number of citations.
        """
        stmt = select(func.count()).select_from(Citation).where(Citation.message_id == message_id)
        return int((await self.session.scalar(stmt)) or 0)

    async def delete_for_message(self, message_id: UUID) -> int:
        """Delete a message's citations.

        Redundant with the message cascade in practice; exposed for
        regeneration flows that clear and reattach citations without
        recreating the message.

        Args:
            message_id (UUID): parent message.

        Returns:
            int: number of citations deleted.
        """
        stmt = delete(Citation).where(Citation.message_id == message_id)
        return await self.execute_rowcount(stmt)

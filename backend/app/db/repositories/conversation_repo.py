from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.base import BaseRepository
from app.models.conversation import Conversation


class ConversationRepository(BaseRepository[Conversation]):
    """Data access for :class:`Conversation`.

    The session is injected and never committed here. The caller owns the
    transaction boundary; this repo only ``flush()``es so that generated
    primary keys and server defaults are visible inside the same session.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Args:
            session (AsyncSession): database session.
        """
        super().__init__(Conversation, session)

    async def get_for_user(self, conversation_id: UUID, user_id: UUID) -> Conversation | None:
        """Fetch a conversation only if it belongs to ``user_id``.

        Returning None rather than raising keeps authorization failures
        indistinguishable from a missing row, which is what the API layer
        wants (both become 404).

        Args:
            conversation_id (UUID): primary key.
            user_id (UUID): expected owner.

        Returns:
            Conversation | None: the conversation, or None if not found / wrong owner.
        """
        stmt = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list_for_user(
        self,
        user_id: UUID,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Conversation]:
        """List a user's conversations, most recently updated first.

        Args:
            user_id (UUID): owner.
            limit (int): maximum rows to return. Defaults to 50.
            offset (int): rows to skip. Defaults to 0.

        Returns:
            list[Conversation]: matching conversations (possibly empty).
        """
        stmt = select(Conversation).where(Conversation.user_id == user_id).order_by(Conversation.updated_at.desc()).limit(limit).offset(offset)
        return list((await self.session.execute(stmt)).scalars().all())

    async def count_for_user(self, user_id: UUID) -> int:
        """Count a user's conversations.

        Args:
            user_id (UUID): owner.

        Returns:
            int: total number of conversations for this user.
        """
        stmt = select(func.count()).select_from(Conversation).where(Conversation.user_id == user_id)
        return int((await self.session.scalar(stmt)) or 0)

    async def set_title(self, conversation_id: UUID, title: str | None) -> None:
        """Update a conversation's title.

        Args:
            conversation_id (UUID): primary key.
            title (str | None): new title, or None to clear it.

        Raises:
            LookupError: if no conversation has this id.
        """
        conv = await self.session.get(Conversation, conversation_id)
        if conv is None:
            raise LookupError(f"conversation {conversation_id} not found")
        conv.title = title
        await self.session.flush()

    async def delete_for_user(self, conversation_id: UUID, user_id: UUID) -> int:
        """Delete a conversation owned by ``user_id``.

        Cascades to messages and citations via FK ondelete rules. Does not
        raise if the row is missing; callers can treat 0 as "nothing to do".

        Args:
            conversation_id (UUID): primary key.
            user_id (UUID): expected owner.

        Returns:
            int: number of rows deleted (0 or 1).
        """
        stmt = delete(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
        return await self.execute_rowcount(stmt)

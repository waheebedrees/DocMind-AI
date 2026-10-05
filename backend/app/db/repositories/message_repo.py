from __future__ import annotations

from decimal import Decimal
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import MessageRole
from app.models.message import Message


class MessageRepository:
    """Data access for :class:`Message`.

    Messages are always reached through their conversation, so this repo
    does not repeat user-scoping; the conversation lookup is the
    authorization boundary. Like the other repos, it flushes but never
    commits.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Args:
            session (AsyncSession): database session.
        """
        self.session = session

    async def create(
        self,
        *,
        conversation_id: UUID,
        role: MessageRole,
        content: str,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        cost_usd: Decimal | None = None,
        latency_ms: int | None = None,
    ) -> Message:
        """Persist a new message in a conversation.

        Args:
            conversation_id (UUID): parent conversation.
            role (MessageRole): speaker role.
            content (str): message body.
            tokens_in (int | None): prompt tokens, if known.
            tokens_out (int | None): completion tokens, if known.
            cost_usd (Decimal | None): billable cost, if known.
            latency_ms (int | None): end-to-end latency, if known.

        Returns:
            Message: the newly created, flushed instance.
        """
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        self.session.add(msg)
        await self.session.flush()
        return msg

    async def get(self, message_id: UUID) -> Message | None:
        """Fetch a message by primary key.

        Args:
            message_id (UUID): primary key.

        Returns:
            Message | None: the message, or None if not found.
        """
        return await self.session.get(Message, message_id)

    async def list_for_conversation(self, conversation_id: UUID) -> list[Message]:
        """Return every message in a conversation, oldest first.

        Ties on ``created_at`` are broken by ``id`` so the order is stable
        across calls (important for prompt reconstruction and for tests).

        Args:
            conversation_id (UUID): parent conversation.

        Returns:
            list[Message]: messages in chronological order (possibly empty).
        """
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at, Message.id)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def latest_for_conversation(
        self, conversation_id: UUID, *, limit: int = 20
    ) -> list[Message]:
        """Return the newest ``limit`` messages, oldest-first.

        Queries in descending order so the DB can use the
        ``(conversation_id, created_at)`` index with a LIMIT, then reverses
        in memory. The returned order matches ``list_for_conversation`` so
        callers can append directly to a prompt.

        Args:
            conversation_id (UUID): parent conversation.
            limit (int): maximum messages to return. Defaults to 20.

        Returns:
            list[Message]: up to ``limit`` messages, oldest-first.
        """
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
        )
        rows = list((await self.session.execute(stmt)).scalars().all())
        rows.reverse()
        return rows

    async def count_for_conversation(self, conversation_id: UUID) -> int:
        """Count messages in a conversation.

        Args:
            conversation_id (UUID): parent conversation.

        Returns:
            int: number of messages.
        """
        stmt = (
            select(func.count())
            .select_from(Message)
            .where(Message.conversation_id == conversation_id)
        )
        return int((await self.session.scalar(stmt)) or 0)

    async def delete_for_conversation(self, conversation_id: UUID) -> int:
        """Delete every message in a conversation.

        Usually unnecessary — deleting the conversation cascades — but
        exposed for callers that want to clear history without dropping
        the conversation row.

        Args:
            conversation_id (UUID): parent conversation.

        Returns:
            int: number of messages deleted.
        """
        stmt = delete(Message).where(
            Message.conversation_id == conversation_id)
        result = await self.session.execute(stmt)
        return result.rowcount or 0
